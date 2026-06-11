import { toHtml } from 'hast-util-to-html'
import { toString } from 'hast-util-to-string'
import type { StreamEntry } from '../plugins/transformers/stream'
import type { QuartzPluginData } from '../plugins/vfile'
import type { BuildCtx } from './ctx'
import { version } from '../../package.json'
import { descriptionToPlainText } from './description'
import { escapeHTML } from './escape'
import { joinSegments } from './path'
import {
  buildStreamEntryPathFromIso,
  formatStreamDate,
  isDraftEntry,
  isPrivateEntry,
  isProtectedEntry,
  isRestrictedEntry,
} from './stream'

const formatIsoAsYMD = (iso?: string | null): string | null => {
  if (!iso) return null
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return null
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const day = String(date.getUTCDate()).padStart(2, '0')
  return `${year}/${month}/${day}`
}

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
    const description = descriptionToPlainText(String(entry.description), 'stream')
    if (description.length > 0) {
      return sanitizeNullable(description)
    }
  }

  if (isPrivateEntry(entry)) return 'private'
  if (isProtectedEntry(entry)) return 'protected'

  const plain = toString({ type: 'root', children: entry.content }).trim()
  if (plain.length === 0) return undefined
  return sanitizeNullable(plain.slice(0, 280))
}

const entryContentHtml = (entry: StreamEntry): string => {
  return sanitizeXml(toHtml({ type: 'root', children: entry.content }))
}

const resolveEntryDate = (entry: StreamEntry, fallback: Date): Date =>
  parseDateValue(entry.date) ?? parseDateValue(entry.timestamp) ?? fallback

const absolutePath = (baseUrl: string, path: string): string => {
  const normalized = path.replace(/^\/+/, '')
  return `https://${joinSegments(baseUrl, encodeURI(normalized))}`
}

export const generateStreamAtomFeed = (ctx: BuildCtx, fileData: QuartzPluginData): string => {
  const cfg = ctx.cfg.configuration
  const base = cfg.baseUrl ?? 'example.com'
  const streamData = fileData.streamData
  const entries = (streamData?.entries ?? []).filter(entry => !isDraftEntry(entry))
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
      const itemPath = buildStreamEntryPathFromIso(entry.date, entry.id) ?? streamPath
      const itemLink = absolutePath(base, itemPath)
      const itemId = `${itemLink}#${entry.id}`
      const titleSource = sanitizeNullable(entry.title?.trim()) ?? `stream entry ${idx + 1}`
      const restricted = isRestrictedEntry(entry)
      const summary = entrySummary(entry)
      const tags = extractStreamTags(entry.metadata)
      const content = restricted ? '' : entryContentHtml(entry)
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
