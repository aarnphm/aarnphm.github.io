import type { Root, Element, ElementContent } from 'hast'
import type { VNode } from 'preact'
import { toHtml } from 'hast-util-to-html'
import { toString } from 'hast-util-to-string'
import { render } from 'preact-render-to-string'
import type { FullPageLayout } from '../../cfg'
import type { QuartzComponentProps } from '../../types/component'
import type { StaticResources } from '../../util/resources'
import type { StreamEntry } from '../transformers/stream'
import type { QuartzPluginData } from '../vfile'
import { version } from '../../../package.json'
import { sharedPageComponents, defaultContentPageLayout } from '../../../quartz.layout'
import StreamPageComponent from '../../components/pages/StreamPage'
import { pageResources, renderPage } from '../../components/renderPage'
import { renderStreamEntry, formatStreamDate, buildOnPath } from '../../components/stream/Entry'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import { escapeHTML } from '../../util/escape'
import { joinSegments, pathToRoot, FullSlug, normalizeHastElement } from '../../util/path'
import { groupStreamEntries } from '../../util/stream'
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

const isElement = (node: ElementContent): node is Element => node.type === 'element'
const sanitizeXml = (input: string): string => {
  let sanitized = ''
  for (let i = 0; i < input.length; i++) {
    const code = input.charCodeAt(i)
    const isInvalid =
      (code >= 0 && code <= 8) ||
      code === 11 ||
      code === 12 ||
      (code >= 14 && code <= 31) ||
      code === 127

    if (isInvalid) continue
    sanitized += input[i]
  }
  return sanitized
}

const sanitizeNullable = (input?: string | null): string | undefined => {
  if (input == null) return undefined
  const sanitized = sanitizeXml(input)
  return sanitized.length > 0 ? sanitized : undefined
}

const parseDateValue = (value: unknown): Date | undefined => {
  if (typeof value === 'number') {
    const date = new Date(value)
    return Number.isNaN(date.getTime()) ? undefined : date
  }

  if (typeof value === 'string') {
    const date = new Date(value)
    return Number.isNaN(date.getTime()) ? undefined : date
  }

  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? undefined : value
  }

  return undefined
}

const extractStreamTags = (metadata: Record<string, unknown>): string[] => {
  const tagsValue = metadata.tags
  if (Array.isArray(tagsValue)) {
    return tagsValue.map(tag => String(tag).trim()).filter(tag => tag.length > 0)
  }
  if (typeof tagsValue === 'string') {
    const tag = tagsValue.trim()
    return tag.length > 0 ? [tag] : []
  }
  return []
}

const entrySummary = (entry: StreamEntry): string | undefined => {
  if (entry.description) {
    const description = String(entry.description).trim()
    if (description.length > 0) {
      return sanitizeNullable(description)
    }
  }

  const plain = toString({ type: 'root', children: entry.content }).trim()
  if (plain.length === 0) return undefined
  return sanitizeNullable(plain.slice(0, 280))
}

const entryContentHtml = (entry: StreamEntry): string => {
  const content = sanitizeXml(toHtml({ type: 'root', children: entry.content }))
  const descriptionHtml = sanitizeNullable(entry.descriptionHtml)
  if (descriptionHtml && content.length > 0) return `${descriptionHtml}\n${content}`
  return descriptionHtml ?? content
}

const resolveEntryDate = (entry: StreamEntry, fallback: Date): Date =>
  parseDateValue(entry.date) ?? parseDateValue(entry.timestamp) ?? fallback

const absolutePath = (baseUrl: string, path: string): string => {
  const normalized = path.replace(/^\/+/, '')
  return `https://${joinSegments(baseUrl, encodeURI(normalized))}`
}

const generateStreamAtomFeed = (ctx: BuildCtx, fileData: QuartzPluginData): string => {
  const cfg = ctx.cfg.configuration
  const base = cfg.baseUrl ?? 'example.com'
  const streamData = fileData.streamData
  const entries = streamData?.entries ?? []
  const fallbackDate =
    parseDateValue(fileData.frontmatter?.modified) ??
    parseDateValue(fileData.frontmatter?.date) ??
    new Date()
  const streamPath = '/stream'
  const streamLink = absolutePath(base, streamPath)
  const introHtml =
    typeof fileData.frontmatter?.rss === 'string'
      ? sanitizeNullable(fileData.frontmatter.rss)
      : undefined
  const feedTitle = escapeHTML(String(fileData.frontmatter?.title ?? 'stream'))
  const subtitleSource =
    sanitizeNullable(
      typeof fileData.frontmatter?.description === 'string'
        ? fileData.frontmatter.description
        : undefined,
    ) ?? `recent stream entries on ${cfg.pageTitle}`
  const feedSubtitle = escapeHTML(subtitleSource)

  let latestUpdated = fallbackDate
  const items = entries
    .map((entry, idx) => {
      const published = resolveEntryDate(entry, fallbackDate)
      if (published.getTime() > latestUpdated.getTime()) {
        latestUpdated = published
      }

      const isoPublished = published.toISOString()
      const itemPath = buildOnPath(entry.date) ?? streamPath
      const itemLink = absolutePath(base, itemPath)
      const itemId = `${itemLink}#${entry.id}`
      const titleSource = sanitizeNullable(entry.title?.trim()) ?? `stream entry ${idx + 1}`
      const summary = entrySummary(entry)
      const tags = extractStreamTags(entry.metadata)
      const content = entryContentHtml(entry)
      const escapedContent = escapeHTML(content)
      const publishedTime = formatStreamDate(isoPublished) ?? formatIsoAsYMD(isoPublished) ?? ''

      return `<entry>
    <title>${escapeHTML(titleSource)}</title>
    <link href="${itemLink}" />
    <id>${itemId}</id>
    ${summary ? `<summary>${escapeHTML(summary)}</summary>` : ''}
    <published>${isoPublished}</published>
    <updated>${isoPublished}</updated>
    <publishedTime>${escapeHTML(publishedTime)}</publishedTime>
    ${tags.map(tag => `<category term="${escapeHTML(tag)}" label="${escapeHTML(tag)}" />`).join('\n')}
    <author>
      <name>Aaron Pham</name>
      <email>contact@aarnphm.xyz</email>
    </author>
    <content type="html">${escapedContent}</content>
  </entry>`
    })
    .join('')

  return `<?xml version="1.0" encoding="UTF-8" ?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:quartz="https://quartz.jzhao.xyz/ns">
  <title>${feedTitle}</title>
  <subtitle>${feedSubtitle}</subtitle>
  <link href="${streamLink}" />
  <link rel="alternate" type="text/html" href="${streamLink}" />
  <category term="stream" />
  <id>${streamLink}</id>
  <updated>${latestUpdated.toISOString()}</updated>
  <contributor>
    <name>Aaron Pham</name>
    <email>contact@aarnphm.xyz</email>
  </contributor>
  <logo>https://${base}/icon.png</logo>
  <icon>https://${base}/icon.png</icon>
  <generator>Quartz v${version} -- quartz.jzhao.xyz</generator>
  <rights type="html">${escapeHTML(`&amp;copy; ${new Date().getFullYear()} Aaron Pham`)}</rights>
  ${introHtml ? `<quartz:intro>${introHtml}</quartz:intro>` : ''}
  ${items}
</feed>`
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

  const groups = groupStreamEntries(fileData!.streamData!.entries)
  if (groups.length === 0) return

  const lines = groups.map(group => {
    const isoSource =
      group.isoDate ??
      group.entries.find(entry => entry.date)?.date ??
      (group.timestamp ? new Date(group.timestamp).toISOString() : null)

    const path = buildOnPath(isoSource!) ?? null
    const entries = group.entries.map(entry => {
      const vnode = renderStreamEntry(entry, fileData!.filePath!, {
        groupId: group.id,
        timestampValue: group.timestamp,
        showDate: true,
        resolvedIsoDate: entry.date ?? group.isoDate,
      })

      return {
        id: entry.id,
        html: render(vnode as VNode<any>),
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
      groupSize: group.entries.length,
      path,
      entries,
    })
  })

  const payload = lines.join('\n')

  yield write({ ctx, slug: 'streams' as FullSlug, ext: '.jsonl', content: payload })

  for (const group of groups) {
    const isoSource =
      group.isoDate ??
      group.entries.find(entry => entry.date)?.date ??
      (group.timestamp ? new Date(group.timestamp).toISOString() : null)

    const onPath = buildOnPath(isoSource!)
    if (!onPath) continue

    const slug = onPath.replace(/^\//, '') as FullSlug
    const titleDate = formatIsoAsYMD(isoSource) ?? formatIsoAsYMD(group.isoDate)
    const title = titleDate ?? fileData!.frontmatter?.title ?? 'stream'
    const sourceSlug = fileData.slug! as FullSlug
    const rebasedEntries = group.entries.map(entry => ({
      ...entry,
      content: entry.content.map(node =>
        isElement(node) ? (normalizeHastElement(node, slug, sourceSlug) as ElementContent) : node,
      ),
    }))

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
      tree,
      allFiles,
    }

    const html = renderPage(ctx, slug, componentData, layout, externalResources, false)

    yield write({ ctx, slug, ext: '.html', content: html })
  }
}

export const StreamIndex: QuartzEmitterPlugin = () => {
  return {
    name: 'StreamIndex',
    async *emit(ctx, content, resources) {
      const allFiles = content.map(([, file]) => file.data as QuartzPluginData)

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        if (data.slug !== 'stream' || !data.streamData) continue

        yield* processStreamIndex(ctx, data, tree, allFiles, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map(([, file]) => file.data as QuartzPluginData)
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
