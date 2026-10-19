import type { Root } from 'hast'
import { toString } from 'hast-util-to-string'
import readingTime from 'reading-time'
import type { FullPageLayout } from '../../cfg'
import type { QuartzComponentProps } from '../../types/component'
import type { StaticResources } from '../../util/resources'
import type { StreamEntry } from '../transformers/stream'
import type { QuartzPluginData } from '../vfile'
import type { ContentLayout } from './contentIndex'
import { sharedPageComponents, defaultContentPageLayout } from '../../../quartz.layout'
import HeaderConstructor from '../../components/Header'
import StreamPageComponent from '../../components/pages/StreamPage'
import { pageResources, renderPage, renderPageWithMetadata } from '../../components/renderPage'
import { renderProtectedEntryBody } from '../../components/stream/Entry'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { isDevelopmentBuild } from '../../util/build-mode'
import { componentSourceNames } from '../../util/component-source'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { descriptionToPlainText } from '../../util/description'
import { joinSegments, pathToRoot, type FullSlug } from '../../util/path'
import { EncryptedPayload, encryptContent, resolveProtectedPassword } from '../../util/protected'
import {
  buildStreamDayPathFromIso,
  buildStreamMonthPath,
  buildStreamOnPath,
  buildStreamYearPath,
  groupStreamEntries,
  groupStreamEntriesByYear,
  isDraftEntry,
  isProtectedEntry,
  isRestrictedEntry,
} from '../../util/stream'
import { generateStreamAtomFeed } from '../../util/stream-feed'
import { buildStreamManifestGroup } from '../../util/stream-manifest'
import {
  buildStreamPageTree,
  cloneStreamEntries,
  rebaseStreamEntries,
} from '../../util/stream-route-tree'
import { removeWritten, write } from './helpers'
import { createOgImageGenerator } from './ogImage'
import { renderedStreamEntries } from './streamRenderedText'

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

// Noon UTC of the entry's stream day, so the formatted date names the day in the route's path.
const streamDay = (iso: string | null | undefined): Date | undefined => {
  const parts = buildStreamDayPathFromIso(iso)?.split('/').slice(-3).map(Number)
  return parts ? new Date(Date.UTC(parts[0], parts[1] - 1, parts[2], 12)) : undefined
}

// An archive route is a page of its own: its date, words, tags and social summary come from the
// entries it shows, so its head, press kit and social image describe the route and not stream.md.
// Private and protected entries add nothing.
const archiveFileData = (
  fileData: QuartzPluginData,
  slug: FullSlug,
  entries: StreamEntry[],
  frontmatter: Record<string, unknown>,
  date?: Date,
): QuartzPluginData => {
  const shown = entries.filter(entry => !isRestrictedEntry(entry))
  const text = shown
    .map(entry =>
      [entry.title, entry.description, toString({ type: 'root', children: entry.content })]
        .filter(Boolean)
        .join('\n'),
    )
    .join('\n\n')
  const summary = shown
    .flatMap(entry => {
      const title = entry.title?.trim()
      const description = descriptionToPlainText(entry.description, slug)
      if (title) return [description ? `${title}: ${description}` : title]
      return description ? [description] : []
    })
    // A year or the whole archive would otherwise list every entry; the newest three stand for it.
    .slice(0, 3)
    .join(' · ')
  const day =
    date ??
    streamDay(
      shown
        .map(entry => entry.date)
        .filter(Boolean)
        .sort()
        .at(-1),
    )
  const tags = [
    ...new Set(
      shown.flatMap(entry =>
        Array.isArray(entry.metadata.tags) ? entry.metadata.tags.map(String) : [],
      ),
    ),
  ]
  return {
    ...fileData,
    slug,
    text,
    readingTime: readingTime(text),
    dates: day ? { created: day, modified: day, published: day } : undefined,
    frontmatter: {
      ...fileData.frontmatter!,
      ...frontmatter,
      tags,
      ...(summary ? { socialDescription: summary } : {}),
    },
  }
}

const manifestForRenderedGroup = (
  group: ReturnType<typeof groupStreamEntries>[number],
  html: string,
) => {
  const entries = renderedStreamEntries(html)
  return buildStreamManifestGroup(group, entry => {
    const rendered = entries.get(entry.id)
    if (rendered === undefined) throw new Error(`missing rendered stream content for ${entry.id}`)
    return rendered
  })
}

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
    tree: streamData ? buildStreamPageTree(streamData.entries, sourceTree, slug) : sourceTree,
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
  emittedRoutes: Set<FullSlug>,
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
    component => !componentSourceNames(component).includes('Byline'),
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

  const socialPages: QuartzPluginData[] = []
  const routeFiles = await mapConcurrent(routeData, defaultIoConcurrency, async route => {
    const routeFile = {
      ...archiveFileData(fileData, route.slug, route.entries, {
        ...legendFrontmatter,
        title: route.title,
      }),
      streamData: { entries: cloneStreamEntries(route.entries) },
    }
    socialPages.push(routeFile)
    return renderStreamRoute(ctx, tree, routeFile, allFiles, resources, layout)
  })
  yield* routeFiles

  const hasAnyProtected = visibleEntries.some(isProtectedEntry)
  let streamPassword: string | undefined
  if (hasAnyProtected) {
    try {
      streamPassword = resolveProtectedPassword(fileData)
    } catch (error) {
      if (!isDevelopmentBuild(ctx.argv)) {
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
      ...archiveFileData(
        fileData,
        slug,
        rebasedEntries,
        { title, streamCanonical: '/stream', pageLayout: 'default' },
        streamDay(isoSource),
      ),
      streamData: { entries: rebasedEntries },
    }
    socialPages.push(fileDataForGroup)

    const externalResources = pageResources(pathToRoot(slug), resources, ctx)
    const componentData: QuartzComponentProps = {
      ctx,
      fileData: fileDataForGroup,
      externalResources,
      cfg: ctx.cfg.configuration,
      children: [],
      tree: buildStreamPageTree(rebasedEntries, tree, slug),
      allFiles,
    }

    const renderGroupPage = () => {
      componentData.tree = buildStreamPageTree(
        fileDataForGroup.streamData?.entries ?? [],
        tree,
        slug,
      )
      return renderPageWithMetadata(ctx, slug, componentData, layout, externalResources, false)
    }

    let rendered = renderGroupPage()
    if (streamPassword && rebasedEntries.some(isProtectedEntry)) {
      const streamData = fileDataForGroup.streamData
      const protectedPayloads = protectedPayloadsForEntries(
        rendered.fileData.streamData?.entries ?? [],
      )
      if (streamData && Object.keys(protectedPayloads).length > 0) {
        fileDataForGroup.streamData = { ...streamData, protectedPayloads }
        rendered = renderGroupPage()
      }
    }
    const { html } = rendered

    const result = {
      slug,
      file: await write({ ctx, slug, ext: '.html', content: html }),
      manifest: manifestForRenderedGroup(group, html),
    }
    return result
  })

  const lines = groupFiles
    .map(result => result?.manifest)
    .filter(group => group !== undefined && group !== null)
    .map(group => JSON.stringify(group))
  yield write({ ctx, slug: 'streams' as FullSlug, ext: '.jsonl', content: lines.join('\n') })

  for (const result of groupFiles) {
    if (result) yield result.file
  }

  // CustomOgImages sees only stream.md, while every archive route's head names its own
  // `<slug>-og-image.webp`. The watcher skips social images, as CustomOgImages does.
  if (!isDevelopmentBuild(ctx.argv)) {
    const generateOgImage = await createOgImageGenerator(ctx)
    yield* await mapConcurrent(socialPages, defaultIoConcurrency, page => generateOgImage(page))
  }

  const nextRoutes = new Set([
    ...routeData.map(route => route.slug),
    ...groupFiles.flatMap(result => (result ? [result.slug] : [])),
  ])
  for (const slug of emittedRoutes) {
    if (!nextRoutes.has(slug)) await removeWritten(ctx, slug, '.html')
  }
  emittedRoutes.clear()
  for (const slug of nextRoutes) emittedRoutes.add(slug)
}

export const StreamIndex: QuartzEmitterPlugin = () => {
  const Header = HeaderConstructor()
  const emittedRoutes = new Set<FullSlug>()
  return {
    name: 'StreamIndex',
    getQuartzComponents() {
      const filteredHeader = sharedPageComponents.header.filter(component => {
        const name = component.displayName || component.name || ''
        return name !== 'Breadcrumbs' && name !== 'StackedNotes'
      })
      const filteredBefore = defaultContentPageLayout.beforeBody.filter(
        component => !componentSourceNames(component).includes('Byline'),
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

        yield* processStreamIndex(ctx, data, tree, allFiles, resources, emittedRoutes)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)
      if (
        !changeEvents.some(event => event.type !== 'delete' && event.file?.data.slug === 'stream')
      )
        return

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        const slug = data.slug!
        if (slug !== 'stream' || !data.streamData) continue
        yield* processStreamIndex(ctx, data, tree, allFiles, resources, emittedRoutes)
      }
    },
  }
}
