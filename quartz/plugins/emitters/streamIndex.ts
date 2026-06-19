import type { Root } from 'hast'
import { Fragment, h } from 'preact'
import { render } from 'preact-render-to-string'
import type { FullPageLayout } from '../../cfg'
import type { QuartzComponentProps } from '../../types/component'
import type { StaticResources } from '../../util/resources'
import type { StreamEntry } from '../transformers/stream'
import type { QuartzPluginData } from '../vfile'
import type { ContentLayout } from './contentIndex'
import { sharedPageComponents, defaultContentPageLayout } from '../../../quartz.layout'
import HeaderConstructor from '../../components/Header'
import StreamPageComponent from '../../components/pages/StreamPage'
import { pageResources, renderPage } from '../../components/renderPage'
import { renderStreamEntry, renderProtectedEntryBody } from '../../components/stream/Entry'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { joinSegments, pathToRoot, type FullSlug } from '../../util/path'
import { EncryptedPayload, encryptContent, resolveProtectedPassword } from '../../util/protected'
import {
  buildStreamDayPathFromIso,
  buildStreamMonthPath,
  buildStreamOnPath,
  buildStreamYearPath,
  formatStreamDate,
  groupStreamEntries,
  groupStreamEntriesByYear,
  isDraftEntry,
  isProtectedEntry,
  isRestrictedEntry,
} from '../../util/stream'
import { generateStreamAtomFeed } from '../../util/stream-feed'
import {
  buildStreamRouteTree,
  cloneStreamEntries,
  rebaseStreamEntries,
} from '../../util/stream-route-tree'
import { write } from './helpers'

const formatIsoAsYMD = (iso?: string | null): string | null => {
  if (!iso) return null
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return null
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const day = String(date.getUTCDate()).padStart(2, '0')
  return `${year}/${month}/${day}`
}

const streamSlugFromPath = (path: string): FullSlug => path.replace(/^\//, '') as FullSlug

const renderStreamRoute = async (
  ctx: BuildCtx,
  sourceTree: Root,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
  layout: FullPageLayout,
) => {
  const slug = fileData.slug!
  const streamData = fileData.streamData
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg: ctx.cfg.configuration,
    children: [],
    tree: streamData ? buildStreamRouteTree(streamData.entries, sourceTree) : sourceTree,
    allFiles,
  }
  const html = renderPage(ctx, slug, componentData, layout, externalResources, false)
  return write({ ctx, slug, ext: '.html', content: html })
}

async function* processStreamIndex(
  ctx: BuildCtx,
  fileData: QuartzPluginData,
  tree: Root,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
) {
  yield write({
    ctx,
    slug: joinSegments('stream', 'index') as FullSlug,
    ext: '.xml',
    content: generateStreamAtomFeed(ctx, fileData),
  })

  const filteredHeader = sharedPageComponents.header.filter(component => {
    const name = component.displayName || component.name || ''
    return name !== 'Breadcrumbs' && name !== 'StackedNotes'
  })
  const filteredBefore = defaultContentPageLayout.beforeBody.filter(
    c => c.displayName !== 'Byline' || c.name !== 'Byline',
  )

  const layout: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    header: filteredHeader,
    beforeBody: filteredBefore,
    afterBody: [],
    pageBody: StreamPageComponent(),
  }

  const visibleEntries = fileData!.streamData!.entries.filter(entry => !isDraftEntry(entry))
  const groups = groupStreamEntries(visibleEntries)
  if (groups.length === 0) return

  const lines = groups
    .map(group => {
      const isoSource =
        group.isoDate ??
        group.entries.find(entry => entry.date)?.date ??
        (group.timestamp ? new Date(group.timestamp).toISOString() : null)

      const path = buildStreamDayPathFromIso(isoSource) ?? null
      const publicEntries = group.entries.filter(entry => !isRestrictedEntry(entry))
      if (publicEntries.length === 0) return null

      const entries = publicEntries.map(entry => {
        const vnode = renderStreamEntry(entry, fileData!.filePath!, {
          groupId: group.id,
          timestampValue: group.timestamp,
          showDate: true,
          resolvedIsoDate: entry.date ?? group.isoDate,
          mode: 'listing',
        })

        return {
          id: entry.id,
          html: render(h(Fragment, null, vnode)),
          metadata: entry.metadata,
          isoDate: entry.date ?? group.isoDate ?? null,
          displayDate:
            formatIsoAsYMD(entry.date ?? group.isoDate ?? isoSource) ??
            formatStreamDate(entry.date ?? group.isoDate) ??
            null,
        }
      })

      return JSON.stringify({
        groupId: group.id,
        timestamp: group.timestamp ?? null,
        isoDate: group.isoDate ?? null,
        groupSize: publicEntries.length,
        path,
        entries,
      })
    })
    .filter((line): line is string => line !== null)

  const payload = lines.join('\n')

  yield write({ ctx, slug: 'streams' as FullSlug, ext: '.jsonl', content: payload })

  const yearGroups = groupStreamEntriesByYear(visibleEntries)
  const legendPageLayout: ContentLayout = 'default'
  const legendFrontmatter = {
    ...fileData!.frontmatter,
    streamCanonical: buildStreamOnPath(),
    pageLayout: legendPageLayout,
  }

  const routeData = [
    {
      slug: streamSlugFromPath(buildStreamOnPath()),
      entries: visibleEntries,
      title: 'stream / on',
    },
  ]
  for (const yearGroup of yearGroups) {
    routeData.push({
      slug: streamSlugFromPath(buildStreamYearPath(yearGroup.yearText)),
      entries: yearGroup.entries,
      title: `stream / ${yearGroup.yearText}`,
    })

    for (const monthGroup of yearGroup.months) {
      routeData.push({
        slug: streamSlugFromPath(buildStreamMonthPath(monthGroup.yearText, monthGroup.monthText)),
        entries: monthGroup.entries,
        title: `stream / ${monthGroup.yearText} / ${monthGroup.monthText}`,
      })
    }
  }

  const routeFiles = await mapConcurrent(routeData, defaultIoConcurrency, route =>
    renderStreamRoute(
      ctx,
      tree,
      {
        ...fileData,
        slug: route.slug,
        streamData: { entries: cloneStreamEntries(route.entries) },
        frontmatter: { ...legendFrontmatter, title: route.title },
      },
      allFiles,
      resources,
      layout,
    ),
  )
  yield* routeFiles

  const hasAnyProtected = visibleEntries.some(isProtectedEntry)
  let streamPassword: string | undefined
  if (hasAnyProtected) {
    try {
      streamPassword = resolveProtectedPassword(fileData)
    } catch (error) {
      if (!ctx.argv.watch || ctx.argv.force) {
        throw error
      }
    }
  }

  const sourceSlug = fileData.slug! as FullSlug

  const protectedPayloadsForEntries = (
    entries: StreamEntry[],
  ): Record<string, EncryptedPayload> => {
    const payloads: Record<string, EncryptedPayload> = {}
    if (!streamPassword) return payloads

    for (const entry of entries) {
      if (isProtectedEntry(entry)) {
        const bodyHtml = renderProtectedEntryBody(entry, fileData!.filePath!)
        payloads[entry.id] = encryptContent(bodyHtml, streamPassword)
      }
    }

    return payloads
  }

  const groupFiles = await mapConcurrent(groups, defaultIoConcurrency, async group => {
    const isoSource =
      group.isoDate ??
      group.entries.find(entry => entry.date)?.date ??
      (group.timestamp ? new Date(group.timestamp).toISOString() : null)

    const onPath = buildStreamDayPathFromIso(isoSource)
    if (!onPath) return undefined

    const slug = onPath.replace(/^\//, '') as FullSlug
    const titleDate = formatIsoAsYMD(isoSource) ?? formatIsoAsYMD(group.isoDate)
    const title = titleDate ?? fileData!.frontmatter?.title ?? 'stream'

    const rebasedEntries = rebaseStreamEntries(group.entries, slug, sourceSlug)

    const fileDataForGroup: QuartzPluginData = {
      ...fileData,
      slug,
      streamData: { entries: rebasedEntries },
      frontmatter: {
        ...fileData!.frontmatter,
        title,
        streamCanonical: '/stream',
        pageLayout: 'default',
      },
    }

    const externalResources = pageResources(pathToRoot(slug), resources, ctx)
    const componentData: QuartzComponentProps = {
      ctx,
      fileData: fileDataForGroup,
      externalResources,
      cfg: ctx.cfg.configuration,
      children: [],
      tree: buildStreamRouteTree(rebasedEntries, tree),
      allFiles,
    }

    const renderGroupPage = () => {
      componentData.tree = buildStreamRouteTree(fileDataForGroup.streamData?.entries ?? [], tree)
      return renderPage(ctx, slug, componentData, layout, externalResources, false)
    }

    let html = renderGroupPage()
    if (streamPassword && rebasedEntries.some(isProtectedEntry)) {
      const streamData = fileDataForGroup.streamData
      const protectedPayloads = protectedPayloadsForEntries(streamData?.entries ?? [])
      if (streamData && Object.keys(protectedPayloads).length > 0) {
        fileDataForGroup.streamData = { ...streamData, protectedPayloads }
        html = renderGroupPage()
      }
    }

    return write({ ctx, slug, ext: '.html', content: html })
  })

  for (const file of groupFiles) {
    if (file) yield file
  }
}

export const StreamIndex: QuartzEmitterPlugin = () => {
  const Header = HeaderConstructor()
  return {
    name: 'StreamIndex',
    getQuartzComponents() {
      const filteredHeader = sharedPageComponents.header.filter(component => {
        const name = component.displayName || component.name || ''
        return name !== 'Breadcrumbs' && name !== 'StackedNotes'
      })
      const filteredBefore = defaultContentPageLayout.beforeBody.filter(
        c => c.displayName !== 'Byline' || c.name !== 'Byline',
      )
      const layout: FullPageLayout = {
        ...sharedPageComponents,
        ...defaultContentPageLayout,
        header: filteredHeader,
        beforeBody: filteredBefore,
        afterBody: [],
        pageBody: StreamPageComponent(),
      }
      return [
        layout.head,
        Header,
        ...layout.header,
        ...layout.beforeBody,
        layout.pageBody,
        ...layout.afterBody,
        ...layout.sidebar,
        layout.footer,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = contentDataFor(content)

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        if (data.slug !== 'stream' || !data.streamData) continue

        yield* processStreamIndex(ctx, data, tree, allFiles, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)
      const changedSlugs = new Set<string>()

      for (const changeEvent of changeEvents) {
        if (changeEvent.file) {
          if (changeEvent.type === 'add' || changeEvent.type === 'change') {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
          continue
        }

        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          const changedPath = changeEvent.path
          for (const [_, vf] of content) {
            const deps = (vf.data.codeDependencies as string[] | undefined) ?? []
            if (deps.includes(changedPath)) {
              changedSlugs.add(vf.data.slug!)
            }
          }
        }
      }

      if (!changedSlugs.has('stream')) return

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        const slug = data.slug!
        if (slug !== 'stream' || !data.streamData) continue
        yield* processStreamIndex(ctx, data, tree, allFiles, resources)
      }
    },
  }
}
