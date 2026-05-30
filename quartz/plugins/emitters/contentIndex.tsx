import { Root } from 'hast'
import { toHtml } from 'hast-util-to-html'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'path'
import { ReadTimeResults } from 'reading-time'
import type { ChangeEvent } from '../../types/plugin'
import type { BuildCtx } from '../../util/ctx'
import type { StaticResources } from '../../util/resources'
import { version } from '../../../package.json'
import { GlobalConfiguration } from '../../cfg'
import { formatDate, getDate } from '../../components/Date'
import { i18n } from '../../i18n'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { escapeHTML } from '../../util/escape'
import {
  pageListingChanged,
  pageNavigationChanged,
  pageSearchChanged,
  pageSitemapChanged,
} from '../../util/listing-signature'
import {
  FilePath,
  FullSlug,
  SimpleSlug,
  getAllSegmentPrefixes,
  joinSegments,
  simplifySlug,
  sluggify,
  slugifyFilePath,
} from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { pageTitlePatchEvents } from '../../util/title-patch'
import { ArenaData } from '../transformers/arena'
import { ProcessedContent, QuartzPluginData } from '../vfile'
import { write, writeKnownChanged } from './helpers'

export type ContentIndexMap = Map<FullSlug, ContentDetails>
export type ContentLayout =
  | 'default'
  | 'letter'
  | 'technical'
  | 'technical-tractatus'
  | 'reflection'
  | 'letter-poem'
  | 'L->ET|A'
  | 'L->EAT'
  | 'A|L'
  | 'L'
  | 'triathlon'
export type ContentDetails = {
  slug: string
  title: string
  filePath: FilePath
  links: SimpleSlug[]
  aliases: string[]
  tags: string[]
  layout: ContentLayout
  content?: string
  fileName: FilePath
  richContent?: string
  fileData?: QuartzPluginData
  date?: Date
  readingTime?: Partial<ReadTimeResults>
  description?: string
  protected?: boolean
}

interface Options {
  enableSiteMap: boolean
  enableAtom: boolean
  atomLimit?: number
  includeEmptyFiles: boolean
  enableSecurity: boolean
}

const defaultOptions: Options = {
  enableSiteMap: true,
  enableAtom: true,
  atomLimit: 10,
  includeEmptyFiles: true,
  enableSecurity: true,
}

const SEARCH_INDEX_CHUNK_COUNT = 64
const SEARCH_INDEX_KIND = 'quartz-search-index-v1'

interface AtomFeedOptions {
  limit?: number
  title?: string
  subtitle?: string
  linkPath?: string
  category?: string
  introHtml?: string
}

type AtomFeedCache = { entries: Map<FullSlug, string> }
type AtomFeedResult = { content: string; cache: AtomFeedCache }

// eslint-disable-next-line no-control-regex
const INVALID_XML_CHARS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g

function sanitizeXml(input: string): string {
  return input.replace(INVALID_XML_CHARS, '')
}

function sanitizeNullable(input?: string | null): string | undefined {
  if (input == null) {
    return undefined
  }
  const sanitized = sanitizeXml(input)
  return sanitized
}

function generateSiteMap(cfg: GlobalConfiguration, idx: ContentIndexMap): string {
  const base = cfg.baseUrl ?? ''
  const createURLEntry = (slug: SimpleSlug, content: ContentDetails): string => {
    let modifiedDate = content.date
    if (!modifiedDate && content.fileData!.frontmatter?.modified) {
      modifiedDate = new Date(content.fileData!.frontmatter.modified)
    }
    return `<url>
    <loc>https://${joinSegments(base, encodeURI(slug))}</loc>
    <lastmod>${modifiedDate?.toISOString()}</lastmod>
  </url>`
  }

  const urls = Array.from(idx)
    .sort(([slugA], [slugB]) => slugA.localeCompare(slugB))
    .map(([slug, content]) => createURLEntry(simplifySlug(slug), content))
    .join('')
  return `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">${urls}</urlset>`
}

function shouldIncludeInFeed(slug: FullSlug, content: ContentDetails): boolean {
  if (
    slug.includes('.bases') ||
    content.fileName.includes('.bases') ||
    slug.includes('.canvas') ||
    content.fileName.includes('.canvas')
  ) {
    return false
  }

  const frontmatter = content.fileData?.frontmatter
  if (!frontmatter) {
    return true
  }

  if (frontmatter.noindex === true) {
    return false
  }

  return true
}

function isContentPageFile(fp: FilePath): boolean {
  return fp.endsWith('.md') || fp.endsWith('.base') || fp.endsWith('.canvas')
}

function isGraphFilePage(fp: FilePath): boolean {
  return fp.endsWith('.ipynb') || fp.endsWith('.base')
}

function fileTitle(fp: FilePath): string {
  const ext = path.extname(fp)
  const base = path.basename(fp, ext)
  return base.length > 0 ? base : path.basename(fp)
}

function fileTags(fp: FilePath): string[] {
  const ext = path.extname(fp).replace(/^\./, '').toLowerCase()
  return ext.length > 0 ? [ext, 'file'] : ['file']
}

function fileSearchContent(fp: FilePath): string {
  const contentPath = joinSegments('content', fp)
  return sanitizeXml([fileTitle(fp), fp, contentPath].join(' '))
}

type SerializedIndexCache = { entries: Map<FullSlug, string> }

function serializeContentDetails(content: ContentDetails, includeContent: boolean): string {
  const serializable = { ...content }
  if (!includeContent) {
    delete serializable.content
  }
  delete serializable.fileData
  delete serializable.richContent
  return JSON.stringify(serializable)
}

function serializeIndexJson(
  idx: ContentIndexMap,
  includeContent: boolean,
  previous?: SerializedIndexCache,
  changedSlugs?: ReadonlySet<FullSlug>,
): { content: string; cache: SerializedIndexCache } {
  if (!previous || !changedSlugs) {
    const entries = new Map<FullSlug, string>()
    for (const [slug, content] of idx) {
      entries.set(slug, serializeContentDetails(content, includeContent))
    }
    return { content: serializeEntries(entries), cache: { entries } }
  }

  const entries = new Map(previous.entries)
  for (const slug of entries.keys()) {
    if (!idx.has(slug)) {
      entries.delete(slug)
    }
  }
  for (const slug of changedSlugs) {
    const details = idx.get(slug)
    if (details) {
      entries.set(slug, serializeContentDetails(details, includeContent))
    }
  }
  return { content: serializeEntries(entries), cache: { entries } }
}

function serializeEntries(entries: ReadonlyMap<FullSlug, string>): string {
  const parts: string[] = []
  const sortedEntries = Array.from(entries).sort(([slugA], [slugB]) => slugA.localeCompare(slugB))
  for (const [slug, content] of sortedEntries) {
    parts.push(serializedEntry(slug, content))
  }
  return `{${parts.join(',')}}`
}

function serializedEntry(slug: FullSlug, content: string): string {
  return `${JSON.stringify(slug)}:${content}`
}

function searchChunkId(slug: FullSlug): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < slug.length; i += 1) {
    hash ^= slug.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return (hash % SEARCH_INDEX_CHUNK_COUNT).toString(16).padStart(2, '0')
}

function searchChunkIds(): string[] {
  return Array.from({ length: SEARCH_INDEX_CHUNK_COUNT }, (_, index) =>
    index.toString(16).padStart(2, '0'),
  )
}

function searchChunkSlug(id: string): FullSlug {
  return joinSegments('static', 'search-index', id) as FullSlug
}

function searchChunkManifest(): string {
  return JSON.stringify({
    kind: SEARCH_INDEX_KIND,
    chunks: searchChunkIds().map(id => `search-index/${id}.json`),
  })
}

function searchChunkEntries(
  entries: ReadonlyMap<FullSlug, string>,
  id: string,
): Map<FullSlug, string> {
  const chunk = new Map<FullSlug, string>()
  for (const [slug, content] of entries) {
    if (searchChunkId(slug) === id) {
      chunk.set(slug, content)
    }
  }
  return chunk
}

async function writeSearchChunk(
  ctx: BuildCtx,
  id: string,
  entries: ReadonlyMap<FullSlug, string>,
  knownChanged = false,
): Promise<FilePath> {
  const content = serializeEntries(searchChunkEntries(entries, id))
  const slug = searchChunkSlug(id)
  return knownChanged
    ? writeKnownChanged({ ctx, content, slug, ext: '.json' })
    : write({ ctx, content, slug, ext: '.json' })
}

async function writeSearchManifest(ctx: BuildCtx): Promise<FilePath> {
  return write({
    ctx,
    content: searchChunkManifest(),
    slug: joinSegments('static', 'searchIndex') as FullSlug,
    ext: '.json',
  })
}

function addFilePageToIndex(ctx: BuildCtx, idx: ContentIndexMap, file: FilePath) {
  const slug = slugifyFilePath(file, path.extname(file) === '.ipynb')
  if (idx.has(slug)) {
    return
  }

  const contentPath = joinSegments('content', file)
  idx.set(slug, {
    slug,
    title: fileTitle(file),
    links: [],
    filePath: joinSegments(ctx.argv.directory, file) as FilePath,
    fileName: file,
    tags: fileTags(file),
    aliases: [file, contentPath, slug],
    content: fileSearchContent(file),
    layout: 'default',
    description: contentPath,
  })
}

function deriveFolderDisplayTitle(
  folderSlug: string,
  folderIndex: ContentIndexMap,
): string | undefined {
  const slugSegments = folderSlug
    .split('/')
    .filter(segment => segment.length > 0)
    .map(segment => segment.toLowerCase())

  if (slugSegments.length === 0) {
    return undefined
  }

  const canonicalSlug = slugSegments.join('/')

  for (const [, content] of folderIndex) {
    const normalizedFileName = content.fileName.replace(/\\/g, '/')
    const folderPath = normalizedFileName.includes('/')
      ? normalizedFileName.slice(0, normalizedFileName.lastIndexOf('/'))
      : ''
    if (folderPath.length === 0) {
      continue
    }

    const relativePath = folderPath.startsWith('content/')
      ? folderPath.slice('content/'.length)
      : folderPath

    const relativeSegments = relativePath.split('/').filter(segment => segment.length > 0)

    if (relativeSegments.length < slugSegments.length) {
      continue
    }

    const candidateSegments = relativeSegments.slice(0, slugSegments.length)
    const canonicalCandidate = candidateSegments
      .map(segment => sluggify(segment).toLowerCase())
      .join('/')

    if (canonicalCandidate === canonicalSlug) {
      return candidateSegments.join(' / ')
    }
  }

  return undefined
}

function atomFeedEntry(
  cfg: GlobalConfiguration,
  base: string,
  slug: SimpleSlug,
  content: ContentDetails,
): string {
  const publishedDate = content.date ?? new Date()
  const frontmatterModified = content.fileData?.frontmatter?.modified
  let updatedDate = frontmatterModified ? new Date(frontmatterModified) : content.date
  if (!updatedDate || Number.isNaN(updatedDate.getTime())) {
    updatedDate = publishedDate
  }

  const summary = escapeHTML(sanitizeNullable(content.description) ?? '')
  const richContent = content.richContent ?? ''

  return `<entry>
    <title>${escapeHTML(content.title)}</title>
    <link href="https://${joinSegments(base, encodeURI(slug))}" />
    <link rel="alternate" type="text/markdown" href="https://${joinSegments(base, encodeURI(slug))}.md" />
    <summary>${summary}</summary>
    <published>${publishedDate.toISOString()}</published>
    <updated>${updatedDate.toISOString()}</updated>
    <publishedTime>${formatDate(publishedDate, cfg.locale)}</publishedTime>
    <updatedTime>${formatDate(updatedDate, cfg.locale)}</updatedTime>
    ${content.tags.map(el => `<category term="${escapeHTML(el)}" label="${escapeHTML(el)}" />`).join('\n')}
    <author>
      <name>Aaron Pham</name>
      <email>contact@aarnphm.xyz</email>
    </author>
    <content type="html">${richContent}</content>
  </entry>`
}

function generateAtomFeed(
  cfg: GlobalConfiguration,
  idx: ContentIndexMap,
  options: AtomFeedOptions = {},
  previous?: AtomFeedCache,
  changedSlugs?: ReadonlySet<FullSlug>,
): AtomFeedResult {
  const base = cfg.baseUrl ?? 'example.com'
  const limit = options.limit

  const createURLEntry = (slug: FullSlug, content: ContentDetails): string => {
    const cached = previous?.entries.get(slug)
    if (cached && !changedSlugs?.has(slug)) {
      return cached
    }
    return atomFeedEntry(cfg, base, simplifySlug(slug), content)
  }

  const limitedEntries = limitedFeedEntries(idx, limit)
  const entryCache = new Map<FullSlug, string>()
  const items = limitedEntries.map(([slug, content]) => {
    const entry = createURLEntry(slug, content)
    entryCache.set(slug, entry)
    return entry
  })

  const latestUpdated = limitedEntries.reduce<Date | undefined>((latest, [_, content]) => {
    const frontmatterModified = content.fileData?.frontmatter?.modified
    let candidate = frontmatterModified ? new Date(frontmatterModified) : content.date
    if (candidate && Number.isNaN(candidate.getTime())) candidate = undefined
    if (!candidate) candidate = content.date ?? undefined
    if (!candidate) return latest
    if (!latest || candidate.getTime() > latest.getTime()) return candidate
    return latest
  }, undefined)

  const absoluteLink = options.linkPath
    ? `https://${joinSegments(base, encodeURI(options.linkPath))}`
    : `https://${base}`
  const feedTitle = escapeHTML(options.title ?? cfg.pageTitle)
  const baseSubtitle = limit
    ? i18n(cfg.locale).pages.rss.lastFewNotes({ count: limit })
    : i18n(cfg.locale).pages.rss.recentNotes
  const feedSubtitle = escapeHTML(options.subtitle ?? `${baseSubtitle} on ${cfg.pageTitle}`)
  const feedCategory = escapeHTML(options.category ?? 'evergreen')
  const feedId = options.linkPath ? absoluteLink : `https://${base}`
  const introHtml = sanitizeNullable(options.introHtml)

  const feed = `<?xml version="1.0" encoding="UTF-8" ?>
<?xml-stylesheet href="/static/feed.xsl" type="text/xsl" ?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:quartz="https://quartz.jzhao.xyz/ns">
  <title>${feedTitle}</title>
  <subtitle>${feedSubtitle}</subtitle>
  <link href="${absoluteLink}" />
  <link rel="alternate" type="text/html" href="${absoluteLink}" />
  <category term="${feedCategory}" />
  <id>${feedId}</id>
  <updated>${(latestUpdated ?? new Date()).toISOString()}</updated>
  <contributor>
    <name>Aaron Pham</name>
    <email>contact@aarnphm.xyz</email>
  </contributor>
  <logo>https://${base}/icon.png</logo>
  <icon>https://${base}/icon.png</icon>
  <generator>Quartz v${version} -- quartz.jzhao.xyz</generator>
  <rights type="html">${escapeHTML(`&amp;copy; ${new Date().getFullYear()} Aaron Pham`)}</rights>
  ${introHtml ? `<quartz:intro>${introHtml}</quartz:intro>` : ''}
  ${items.join('')}
</feed>`
  return { content: feed, cache: { entries: entryCache } }
}

function limitedFeedEntries(
  idx: ContentIndexMap,
  limit?: number,
): Array<[FullSlug, ContentDetails]> {
  const filteredEntries = Array.from(idx)
    .sort(([_, f1], [__, f2]) => {
      if (f1.date && f2.date) {
        return f2.date.getTime() - f1.date.getTime()
      } else if (f1.date && !f2.date) {
        return -1
      } else if (!f1.date && f2.date) {
        return 1
      }

      return f1.title.localeCompare(f2.title)
    })
    .filter(([slug, content]) => shouldIncludeInFeed(slug, content))

  return filteredEntries.slice(0, limit ?? filteredEntries.length)
}

function feedContainsAnySlug(
  idx: ContentIndexMap,
  slugs: ReadonlySet<FullSlug>,
  limit?: number,
): boolean {
  if (slugs.size === 0) return false
  return limitedFeedEntries(idx, limit).some(([slug]) => slugs.has(slug))
}

function feedCacheContainsAnySlug(
  cache: AtomFeedCache | undefined,
  slugs: ReadonlySet<FullSlug>,
): boolean {
  if (!cache || slugs.size === 0) return false
  for (const slug of slugs) {
    if (cache.entries.has(slug)) return true
  }
  return false
}

type LinkIndexUpdate = { previous: ContentIndexMap; changeEvents: readonly ChangeEvent[] }
type ReusableContentDetails = Pick<ContentDetails, 'content' | 'richContent' | 'readingTime'>

function normalizedContentSlug(fileData: QuartzPluginData): FullSlug | undefined {
  const slug = fileData.slug
  if (typeof slug !== 'string') return undefined
  return (slug === 'are.na' ? 'arena' : slug) as FullSlug
}

function removeContentFromIndex(linkIndex: ContentIndexMap, fileData: QuartzPluginData): void {
  const slug = normalizedContentSlug(fileData)
  if (!slug) return
  linkIndex.delete(slug)

  if (slug === 'arena') {
    for (const [entrySlug, details] of linkIndex) {
      if (entrySlug.startsWith('arena/') && details.filePath === fileData.filePath) {
        linkIndex.delete(entrySlug)
      }
    }
  }
}

function changedContentSlugs(changeEvents: readonly ChangeEvent[]): Set<FullSlug> {
  const slugs = new Set<FullSlug>()
  for (const changeEvent of changeEvents) {
    if (changeEvent.file) {
      const slug = normalizedContentSlug(changeEvent.file.data)
      if (slug) {
        slugs.add(slug)
      }
      continue
    }

    if (isContentPageFile(changeEvent.path)) {
      slugs.add(slugifyFilePath(changeEvent.path))
    }
  }
  return slugs
}

function filterVisibleLinks(
  links: readonly SimpleSlug[],
  contentBySlug: ReadonlyMap<string, QuartzPluginData>,
): SimpleSlug[] {
  return links.filter(link => {
    const targetFile = contentBySlug.get(link)
    if (targetFile?.frontmatter?.noindex === true || targetFile?.frontmatter?.protected === true) {
      return false
    }

    return true
  })
}

function refreshVisibleLinks(
  linkIndex: ContentIndexMap,
  contentBySlug: ReadonlyMap<string, QuartzPluginData>,
): void {
  for (const details of linkIndex.values()) {
    details.links = filterVisibleLinks(details.fileData?.links ?? details.links, contentBySlug)
  }
}

function contentDataBySlug(content: ProcessedContent[]): Map<string, QuartzPluginData> {
  const contentBySlug = new Map<string, QuartzPluginData>()
  for (const [, file] of content) {
    const slug = file.data.slug
    if (typeof slug === 'string') {
      contentBySlug.set(slug, file.data)
    }
  }
  return contentBySlug
}

function processedContentBySlug(content: ProcessedContent[]): Map<FullSlug, ProcessedContent> {
  const contentBySlug = new Map<FullSlug, ProcessedContent>()
  for (const entry of content) {
    const slug = normalizedContentSlug(entry[1].data)
    if (slug) {
      contentBySlug.set(slug, entry)
    }
  }
  return contentBySlug
}

function contentFileName(ctx: BuildCtx, fileData: QuartzPluginData): FilePath {
  const fullPath = fileData.filePath!
  const relativePath = fullPath.substring(ctx.argv.directory.length + 1)
  if (relativePath.endsWith('.bases')) return relativePath as FilePath
  return relativePath.replace('.md', '') as FilePath
}

function stringDataField(
  data: QuartzPluginData,
  key: 'htmlReuseBody' | 'htmlReuseFrontmatter',
): string | undefined {
  const value = data[key]
  return typeof value === 'string' ? value : undefined
}

function canReuseIndexedContent(current: QuartzPluginData, previous: QuartzPluginData): boolean {
  if (!current.filePath?.endsWith('.md') || !previous.filePath?.endsWith('.md')) return false
  if ((current.frontmatter?.protected ?? false) !== (previous.frontmatter?.protected ?? false)) {
    return false
  }

  const currentBody = stringDataField(current, 'htmlReuseBody')
  const previousBody = stringDataField(previous, 'htmlReuseBody')
  const currentFrontmatter = stringDataField(current, 'htmlReuseFrontmatter')
  const previousFrontmatter = stringDataField(previous, 'htmlReuseFrontmatter')
  return (
    currentBody !== undefined &&
    previousBody !== undefined &&
    currentFrontmatter !== undefined &&
    previousFrontmatter !== undefined &&
    currentBody === previousBody &&
    currentFrontmatter === previousFrontmatter &&
    (current.text ?? '') === (previous.text ?? '')
  )
}

function reusableContentDetails(update?: LinkIndexUpdate): Map<FullSlug, ReusableContentDetails> {
  const reusable = new Map<FullSlug, ReusableContentDetails>()
  if (!update) return reusable

  for (const changeEvent of update.changeEvents) {
    if (changeEvent.type !== 'change' || !changeEvent.file || !changeEvent.previousFile) continue
    if (!canReuseIndexedContent(changeEvent.file.data, changeEvent.previousFile.data)) continue

    const slug = normalizedContentSlug(changeEvent.file.data)
    if (!slug) continue
    const previousDetails = update.previous.get(slug)
    if (!previousDetails) continue
    reusable.set(slug, {
      content: previousDetails.content,
      richContent: previousDetails.richContent,
      readingTime: previousDetails.readingTime,
    })
  }

  return reusable
}

function addProcessedContentToIndex(
  ctx: BuildCtx,
  linkIndex: ContentIndexMap,
  tree: ProcessedContent[0],
  file: ProcessedContent[1],
  opts: Options,
  contentBySlug: ReadonlyMap<string, QuartzPluginData>,
  includeRichContent: boolean,
  reusableDetails?: ReusableContentDetails,
): void {
  let slug = file.data.slug!
  const date = getDate(ctx.cfg.configuration, file.data) ?? new Date()

  if (slug === 'are.na') {
    slug = 'arena' as FullSlug
  }

  if (file.data.canvas) {
    const jcast = file.data.canvas
    const searchableContent = file.data.text ?? ''
    const renderedSlug = slug.replace('.canvas', '') as FullSlug

    linkIndex.set(renderedSlug, {
      slug: renderedSlug,
      title: file.data.frontmatter?.title ?? path.basename(file.data.filePath!, '.canvas'),
      links: filterVisibleLinks(file.data.links ?? [], contentBySlug),
      filePath: file.data.filePath!,
      fileName: file.data.filePath!,
      tags: ['canvas', ...(file.data.frontmatter?.tags ?? [])],
      aliases: file.data.frontmatter?.aliases ?? [],
      content: sanitizeXml(searchableContent),
      richContent: '',
      date: date,
      readingTime: {
        minutes: Math.max(1, Math.ceil(searchableContent.split(/\s+/).length / 200)),
        words: searchableContent.split(/\s+/).filter(w => w.length > 0).length,
      },
      layout: file.data.frontmatter?.pageLayout ?? 'default',
      description:
        file.data.frontmatter?.description ?? `Canvas with ${jcast.data.nodeMap.size} nodes`,
      fileData: file.data,
    })
    return
  }

  if (!opts.includeEmptyFiles && (!file.data.text || file.data.text === '')) {
    return
  }

  const links = filterVisibleLinks(file.data.links ?? [], contentBySlug)
  const isProtected = file.data.frontmatter?.protected === true
  const richContent =
    reusableDetails?.richContent ??
    (includeRichContent && !isProtected
      ? sanitizeXml(escapeHTML(toHtml(tree as Root, { allowDangerousHtml: true })))
      : '')
  const content = reusableDetails?.content ?? (isProtected ? '' : sanitizeXml(file.data.text ?? ''))
  const readingTime = reusableDetails?.readingTime ?? {
    minutes: Math.ceil(file.data.readingTime ? file.data.readingTime.minutes! : 0),
    words: Math.ceil(file.data.readingTime ? file.data.readingTime.words! : 0),
  }

  linkIndex.set(slug, {
    slug,
    title: file.data.frontmatter ? file.data.frontmatter.title! : '',
    links,
    filePath: file.data.filePath!,
    fileName: contentFileName(ctx, file.data),
    tags: file.data.frontmatter?.tags ?? [],
    aliases: file.data.frontmatter?.aliases ?? [],
    content,
    richContent: isProtected ? '' : richContent,
    date: date,
    readingTime,
    fileData: file.data,
    layout: file.data.frontmatter!.pageLayout,
    description: file.data.description,
    protected: isProtected,
  })

  if (slug !== 'arena') {
    return
  }

  const arenaData = file.data.arenaData as ArenaData | undefined
  if (!arenaData) {
    return
  }

  for (const channel of arenaData.channels) {
    const channelSlug = joinSegments('arena', channel.slug) as FullSlug
    linkIndex.set(channelSlug, {
      slug: channelSlug,
      title: channel.name,
      links: ['arena' as SimpleSlug],
      filePath: file.data.filePath!,
      fileName: file.data.filePath!.replace('.md', '') as FilePath,
      tags: file.data.frontmatter?.tags ?? [],
      aliases: [],
      content: channel.blocks.map(b => b.title || b.content).join(' '),
      richContent: '',
      date: date,
      readingTime: { minutes: 1, words: channel.blocks.length * 10 },
      layout: 'default',
      description: `${channel.blocks.length} blocks in ${channel.name}`,
    })
  }
}

function buildLinkIndex(
  ctx: BuildCtx,
  content: ProcessedContent[],
  opts: Options,
  update?: LinkIndexUpdate,
): ContentIndexMap {
  const linkIndex: ContentIndexMap = update ? new Map(update.previous) : new Map()
  const contentBySlug = contentDataBySlug(content)
  const changedSlugs = update ? changedContentSlugs(update.changeEvents) : undefined
  const reusableDetails = reusableContentDetails(update)
  if (update) {
    for (const changeEvent of update.changeEvents) {
      if (changeEvent.file) {
        removeContentFromIndex(linkIndex, changeEvent.file.data)
      } else if (isGraphFilePage(changeEvent.path)) {
        linkIndex.delete(
          slugifyFilePath(changeEvent.path, path.extname(changeEvent.path) === '.ipynb'),
        )
      }
    }
  }

  for (const [tree, file] of content) {
    const slug = normalizedContentSlug(file.data)
    if (!slug) continue
    if (changedSlugs && !changedSlugs.has(slug)) continue
    removeContentFromIndex(linkIndex, file.data)
    addProcessedContentToIndex(
      ctx,
      linkIndex,
      tree,
      file,
      opts,
      contentBySlug,
      true,
      reusableDetails.get(slug),
    )
  }

  refreshVisibleLinks(linkIndex, contentBySlug)
  return linkIndex
}

function buildFolderFeedMap(linkIndex: ContentIndexMap): Map<string, ContentIndexMap> {
  const folderFeedMap = new Map<string, ContentIndexMap>()
  for (const [slug, details] of linkIndex) {
    const prefixes = getAllSegmentPrefixes(slug)
    if (prefixes.length <= 1) {
      continue
    }

    prefixes.pop()
    for (const prefix of prefixes) {
      const normalizedPrefix = prefix.replace(/^\/+|\/+$/g, '')
      if (normalizedPrefix.length === 0) {
        continue
      }
      if (!folderFeedMap.has(normalizedPrefix)) {
        folderFeedMap.set(normalizedPrefix, new Map())
      }
      folderFeedMap.get(normalizedPrefix)!.set(slug, details)
    }
  }

  return folderFeedMap
}

function hasFeedEntries(folderIndex: ContentIndexMap): boolean {
  return Array.from(folderIndex).some(([slug, content]) => shouldIncludeInFeed(slug, content))
}

function addGraphFilePages(ctx: BuildCtx, linkIndex: ContentIndexMap) {
  for (const file of ctx.allFiles) {
    if (isGraphFilePage(file)) {
      addFilePageToIndex(ctx, linkIndex, file)
    }
  }
}

function buildSearchIndex(ctx: BuildCtx, linkIndex: ContentIndexMap): ContentIndexMap {
  const searchIndex: ContentIndexMap = new Map(linkIndex)
  for (const file of ctx.allFiles) {
    if (isContentPageFile(file)) {
      continue
    }

    addFilePageToIndex(ctx, searchIndex, file)
  }

  return searchIndex
}

async function* writeSecurityFiles(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
): AsyncGenerator<FilePath> {
  const baseDomain = cfg.baseUrl ?? 'aarnphm.xyz'
  const securityPolicyEntry =
    linkIndex.get('security-policy' as FullSlug) ??
    Array.from(linkIndex.values()).find(details => {
      const normalizedFile = details.fileName.replace(/\\/g, '/')
      return (
        normalizedFile === ('content/security policy' as FilePath) ||
        normalizedFile.endsWith('/security policy') ||
        details.slug === 'security-policy'
      )
    })

  const fallbackSlug = securityPolicyEntry
    ? simplifySlug(securityPolicyEntry.slug as FullSlug)
    : ('security-policy' as SimpleSlug)
  const policyPermalink = securityPolicyEntry?.fileData?.frontmatter?.permalinks?.[0]
  const policyHref = policyPermalink
    ? `https://${joinSegments(baseDomain, policyPermalink.replace(/^\/+/, ''))}`
    : `https://${joinSegments(baseDomain, fallbackSlug)}`

  const modifiedSource =
    securityPolicyEntry?.fileData?.frontmatter?.modified ?? securityPolicyEntry?.date?.toISOString()
  const lastModifiedDate = modifiedSource ? new Date(modifiedSource) : new Date()
  const safeLastModified = Number.isNaN(lastModifiedDate.getTime()) ? new Date() : lastModifiedDate
  const expiresDate = new Date(safeLastModified.getTime() + 1000 * 60 * 60 * 24 * 180)

  const securityTxt = `Contact: mailto:security@aarnphm.xyz
Encryption: https://${joinSegments(baseDomain, 'pgp-key.txt')}
Policy: ${policyHref}
Canonical: https://${joinSegments(baseDomain, '.well-known', 'security.txt')}
Preferred-Languages: en
Last-Modified: ${safeLastModified.toISOString()}
Expires: ${expiresDate.toISOString()}
`

  yield write({
    ctx,
    content: securityTxt,
    slug: joinSegments('.well-known', 'security') as FullSlug,
    ext: '.txt',
  })
  yield write({ ctx, content: securityTxt, slug: 'security' as FullSlug, ext: '.txt' })
}

async function writeSiteMap(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
): Promise<FilePath> {
  return write({
    ctx,
    content: generateSiteMap(cfg, linkIndex),
    slug: 'sitemap' as FullSlug,
    ext: '.xml',
  })
}

async function writeRootAtomFeed(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
  opts: Options,
  previous?: AtomFeedCache,
  changedSlugs?: ReadonlySet<FullSlug>,
  knownChanged = false,
): Promise<{ file: FilePath; cache: AtomFeedCache }> {
  const feed = generateAtomFeed(cfg, linkIndex, { limit: opts.atomLimit }, previous, changedSlugs)
  const file = knownChanged
    ? await writeKnownChanged({
        ctx,
        content: feed.content,
        slug: 'index' as FullSlug,
        ext: '.xml',
      })
    : await write({ ctx, content: feed.content, slug: 'index' as FullSlug, ext: '.xml' })
  return { file, cache: feed.cache }
}

async function writePatchedRootAtomFeed(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
  previous: AtomFeedCache,
  changedSlugs: ReadonlySet<FullSlug>,
): Promise<{ file: FilePath; cache: AtomFeedCache } | undefined> {
  const base = cfg.baseUrl ?? 'example.com'
  const entries = new Map(previous.entries)
  const slug = 'index' as FullSlug
  const pathToFeed = joinSegments(ctx.argv.output, `${slug}.xml`) as FilePath
  let content = await fs.readFile(pathToFeed, 'utf8')
  let patchedAny = false

  for (const changedSlug of changedSlugs) {
    const previousEntry = entries.get(changedSlug)
    if (!previousEntry) continue
    const details = linkIndex.get(changedSlug)
    if (!details) return undefined
    const currentEntry = atomFeedEntry(cfg, base, simplifySlug(changedSlug), details)
    const next = content.replace(previousEntry, currentEntry)
    if (next === content) return undefined
    content = next
    entries.set(changedSlug, currentEntry)
    patchedAny = true
  }

  if (!patchedAny) return undefined
  return { file: await writeKnownChanged({ ctx, content, slug, ext: '.xml' }), cache: { entries } }
}

async function writeFolderAtomFeed(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
  folderSlug: string,
  folderIndex: ContentIndexMap,
  opts: Options,
): Promise<FilePath | undefined> {
  if (!hasFeedEntries(folderIndex)) {
    return undefined
  }

  const folderKey = folderSlug as FullSlug
  const folderDetails =
    linkIndex.get(folderKey) ??
    Array.from(linkIndex.entries()).find(
      ([existingSlug]) => simplifySlug(existingSlug) === folderSlug,
    )?.[1]
  const fallbackName = folderSlug.split('/').pop() ?? folderSlug
  const folderPathTitle =
    deriveFolderDisplayTitle(folderSlug, folderIndex) ??
    folderSlug
      .split('/')
      .filter(segment => segment.length > 0)
      .map(segment => decodeURIComponent(segment).replace(/-/g, ' '))
      .join(' / ')
  const title =
    (folderPathTitle.length > 0 ? `/${folderPathTitle}` : undefined) ??
    folderDetails?.title ??
    fallbackName.replace(/-/g, ' ')
  const rawIntro = folderDetails?.fileData?.frontmatter?.rss
  const folderIntroHtml = typeof rawIntro === 'string' ? rawIntro : undefined
  const subtitle = `${i18n(cfg.locale).pages.rss.recentNotes} in ${title} on ${cfg.pageTitle}`

  const feed = generateAtomFeed(cfg, folderIndex, {
    limit: opts.atomLimit,
    title,
    subtitle,
    linkPath: folderSlug,
    category: folderSlug,
    introHtml: folderIntroHtml,
  })
  return write({
    ctx,
    content: feed.content,
    slug: joinSegments(folderSlug, 'index') as FullSlug,
    ext: '.xml',
  })
}

async function removeFolderAtomFeed(ctx: BuildCtx, folderSlug: string) {
  const file = joinSegments(ctx.argv.output, folderSlug, 'index.xml') as FilePath
  await fs.rm(file, { force: true })
}

async function writeContentIndexJson(
  ctx: BuildCtx,
  linkIndex: ContentIndexMap,
  previous?: SerializedIndexCache,
  changedSlugs?: ReadonlySet<FullSlug>,
  knownChanged = false,
): Promise<{ file: FilePath; cache: SerializedIndexCache }> {
  const serialized = serializeIndexJson(linkIndex, false, previous, changedSlugs)
  const slug = joinSegments('static', 'contentIndex') as FullSlug
  const file = knownChanged
    ? await writeKnownChanged({ ctx, content: serialized.content, slug, ext: '.json' })
    : await write({ ctx, content: serialized.content, slug, ext: '.json' })
  return { file, cache: serialized.cache }
}

function patchSerializedEntry(
  content: string,
  slug: FullSlug,
  previousEntry: string,
  currentEntry: string,
): string | undefined {
  const previous = serializedEntry(slug, previousEntry)
  const current = serializedEntry(slug, currentEntry)
  const next = content.replace(previous, current)
  return next === content ? undefined : next
}

async function writePatchedSerializedIndexJson(
  ctx: BuildCtx,
  outputSlug: FullSlug,
  previous: SerializedIndexCache,
  updates: ReadonlyMap<FullSlug, string>,
): Promise<{ file: FilePath; cache: SerializedIndexCache } | undefined> {
  const entries = new Map(previous.entries)
  const pathToIndex = joinSegments(ctx.argv.output, `${outputSlug}.json`) as FilePath
  let content = await fs.readFile(pathToIndex, 'utf8')

  for (const [slug, currentEntry] of updates) {
    const previousEntry = entries.get(slug)
    if (!previousEntry) return undefined
    const patched = patchSerializedEntry(content, slug, previousEntry, currentEntry)
    if (!patched) return undefined
    content = patched
    entries.set(slug, currentEntry)
  }

  return {
    file: await writeKnownChanged({ ctx, content, slug: outputSlug, ext: '.json' }),
    cache: { entries },
  }
}

async function writeSearchIndexJson(
  ctx: BuildCtx,
  searchIndex: ContentIndexMap,
  previous?: SerializedIndexCache,
  changedSlugs?: ReadonlySet<FullSlug>,
): Promise<{ files: FilePath[]; cache: SerializedIndexCache }> {
  const serialized = serializeIndexJson(searchIndex, true, previous, changedSlugs)
  const chunkIds = changedSlugs
    ? Array.from(new Set(Array.from(changedSlugs, searchChunkId))).sort()
    : searchChunkIds()
  const [manifestFile, chunkFiles] = await Promise.all([
    previous ? Promise.resolve(undefined) : writeSearchManifest(ctx),
    mapConcurrent(chunkIds, defaultIoConcurrency, id =>
      writeSearchChunk(ctx, id, serialized.cache.entries),
    ),
  ])
  const files = manifestFile ? [manifestFile, ...chunkFiles] : chunkFiles
  return { files, cache: serialized.cache }
}

async function writePatchedSearchIndexJson(
  ctx: BuildCtx,
  previous: SerializedIndexCache,
  updates: ReadonlyMap<FullSlug, string>,
  deletes: ReadonlySet<FullSlug>,
  knownChanged = false,
): Promise<{ files: FilePath[]; cache: SerializedIndexCache }> {
  const entries = new Map(previous.entries)
  for (const slug of deletes) {
    entries.delete(slug)
  }
  for (const [slug, content] of updates) {
    entries.set(slug, content)
  }
  const changedChunks = new Set<string>()
  for (const slug of deletes) {
    changedChunks.add(searchChunkId(slug))
  }
  for (const slug of updates.keys()) {
    changedChunks.add(searchChunkId(slug))
  }
  if (changedChunks.size === 0) {
    return { files: [], cache: { entries } }
  }
  const files = await mapConcurrent(Array.from(changedChunks), defaultIoConcurrency, id =>
    writeSearchChunk(ctx, id, entries, knownChanged),
  )
  return { files, cache: { entries } }
}

async function writePatchedSearchIndexChunks(
  ctx: BuildCtx,
  previous: SerializedIndexCache,
  updates: ReadonlyMap<FullSlug, string>,
): Promise<{ files: FilePath[]; cache: SerializedIndexCache } | undefined> {
  const entries = new Map(previous.entries)
  const updatesByChunk = new Map<string, Map<FullSlug, string>>()

  for (const [slug, currentEntry] of updates) {
    if (!entries.has(slug)) return undefined
    const id = searchChunkId(slug)
    const chunkUpdates = updatesByChunk.get(id)
    if (chunkUpdates) {
      chunkUpdates.set(slug, currentEntry)
    } else {
      updatesByChunk.set(id, new Map([[slug, currentEntry]]))
    }
  }

  const files: FilePath[] = []
  for (const [id, chunkUpdates] of updatesByChunk) {
    const slug = searchChunkSlug(id)
    const pathToChunk = joinSegments(ctx.argv.output, `${slug}.json`) as FilePath
    let content = await fs.readFile(pathToChunk, 'utf8')

    for (const [entrySlug, currentEntry] of chunkUpdates) {
      const previousEntry = entries.get(entrySlug)
      if (!previousEntry) return undefined
      const patched = patchSerializedEntry(content, entrySlug, previousEntry, currentEntry)
      if (!patched) return undefined
      content = patched
      entries.set(entrySlug, currentEntry)
    }

    files.push(await writeKnownChanged({ ctx, content, slug, ext: '.json' }))
  }

  return { files, cache: { entries } }
}

function isSourcePath(path: FilePath): boolean {
  return (
    path.startsWith('quartz/') ||
    path.startsWith('worker/') ||
    path.startsWith('.github/') ||
    path === ('package.json' as FilePath) ||
    path === ('pnpm-lock.yaml' as FilePath) ||
    path === ('tsconfig.json' as FilePath)
  )
}

function changeSlug(changeEvent: ChangeEvent): FullSlug | undefined {
  const slug = changeEvent.file?.data.slug
  if (typeof slug === 'string') {
    return slug as FullSlug
  }
  if (isContentPageFile(changeEvent.path) || isGraphFilePage(changeEvent.path)) {
    return slugifyFilePath(changeEvent.path, path.extname(changeEvent.path) === '.ipynb')
  }
  return undefined
}

function affectedFolderFeedSlugs(changeEvents: readonly ChangeEvent[]): Set<string> {
  const slugs = new Set<string>()
  for (const changeEvent of changeEvents) {
    const slug = changeSlug(changeEvent)
    if (!slug) continue
    const prefixes = getAllSegmentPrefixes(slug)
    prefixes.pop()
    for (const prefix of prefixes) {
      const normalizedPrefix = prefix.replace(/^\/+|\/+$/g, '')
      if (normalizedPrefix.length > 0) {
        slugs.add(normalizedPrefix)
      }
    }
  }

  return slugs
}

function pageFolderFeedSlugs(fileData: QuartzPluginData, folderFeedSlugs: ReadonlySet<string>) {
  const slug = fileData.slug
  if (typeof slug !== 'string') return []
  const prefixes = getAllSegmentPrefixes(slug)
  prefixes.pop()
  return prefixes
    .map(prefix => prefix.replace(/^\/+|\/+$/g, ''))
    .filter(prefix => prefix.length > 0 && folderFeedSlugs.has(prefix))
}

function filePageSlug(file: FilePath): FullSlug {
  return slugifyFilePath(file, path.extname(file) === '.ipynb')
}

function isSecurityPolicyChange(changeEvent: ChangeEvent): boolean {
  if (changeEvent.file?.data.slug === 'security-policy') {
    return true
  }
  const normalized = changeEvent.path.replace(/\\/g, '/')
  return normalized === 'security policy.md' || normalized.endsWith('/security policy.md')
}

function buildSearchIndexPatch(
  ctx: BuildCtx,
  content: ProcessedContent[],
  opts: Options,
  changeEvents: readonly ChangeEvent[],
): { updates: Map<FullSlug, string>; deletes: Set<FullSlug> } | undefined {
  const contentBySlug = contentDataBySlug(content)
  const processedBySlug = processedContentBySlug(content)
  const updates = new Map<FullSlug, string>()
  const deletes = new Set<FullSlug>()

  for (const changeEvent of changeEvents) {
    if (isSourcePath(changeEvent.path)) {
      continue
    }

    if (changeEvent.file) {
      const slug = normalizedContentSlug(changeEvent.file.data)
      if (!slug || slug === 'arena') {
        return undefined
      }
      if (changeEvent.type === 'delete') {
        deletes.add(slug)
        continue
      }

      const processed = processedBySlug.get(slug)
      if (!processed) {
        deletes.add(slug)
        continue
      }

      const entryIndex: ContentIndexMap = new Map()
      addProcessedContentToIndex(
        ctx,
        entryIndex,
        processed[0],
        processed[1],
        opts,
        contentBySlug,
        false,
      )
      if (entryIndex.size !== 1) {
        return undefined
      }
      for (const [entrySlug, details] of entryIndex) {
        updates.set(entrySlug, serializeContentDetails(details, true))
      }
      continue
    }

    if (isGraphFilePage(changeEvent.path)) {
      return undefined
    }
    if (isContentPageFile(changeEvent.path)) {
      continue
    }

    const slug = filePageSlug(changeEvent.path)
    if (changeEvent.type === 'delete') {
      deletes.add(slug)
      continue
    }

    const fileIndex: ContentIndexMap = new Map()
    addFilePageToIndex(ctx, fileIndex, changeEvent.path)
    const details = fileIndex.get(slug)
    if (details) {
      updates.set(slug, serializeContentDetails(details, true))
    }
  }

  return { updates, deletes }
}

function titlePatchContentIndexUpdates(
  previous: ContentIndexMap,
  changeEvents: readonly ChangeEvent[],
):
  | {
      linkIndex: ContentIndexMap
      changedSlugs: Set<FullSlug>
      contentUpdates: Map<FullSlug, string>
      searchUpdates: Map<FullSlug, string>
    }
  | undefined {
  const titlePatches = pageTitlePatchEvents(changeEvents)
  if (!titlePatches) return undefined
  const linkIndex = new Map(previous)
  const changedSlugs = new Set<FullSlug>()
  const contentUpdates = new Map<FullSlug, string>()
  const searchUpdates = new Map<FullSlug, string>()

  for (let i = 0; i < changeEvents.length; i += 1) {
    const changeEvent = changeEvents[i]
    const fileData = changeEvent.file?.data
    if (!fileData) return undefined
    const slug = normalizedContentSlug(fileData)
    if (!slug || slug === 'arena') return undefined
    const previousDetails = linkIndex.get(slug)
    if (!previousDetails) return undefined
    const nextDetails = { ...previousDetails, title: titlePatches[i].currentTitle, fileData }
    linkIndex.set(slug, nextDetails)
    changedSlugs.add(slug)
    contentUpdates.set(slug, serializeContentDetails(nextDetails, false))
    searchUpdates.set(slug, serializeContentDetails(nextDetails, true))
  }

  return { linkIndex, changedSlugs, contentUpdates, searchUpdates }
}

type ContentIndexPartialPlan = {
  content: boolean
  graph: boolean
  search: boolean
  security: boolean
  listing: boolean
  sitemap: boolean
  changedSlugs: Set<FullSlug>
  affectedFolders: Set<string>
}

function planPartialContentIndex(changeEvents: readonly ChangeEvent[]): ContentIndexPartialPlan {
  let content = false
  let graph = false
  let search = false
  let security = false
  let listing = false
  let sitemap = false
  const changedSlugs = new Set<FullSlug>()

  for (const changeEvent of changeEvents) {
    if (isSourcePath(changeEvent.path)) {
      continue
    }

    const file = changeEvent.file
    if (file) {
      search ||= changeEvent.type !== 'change'
      search ||= pageSearchChanged(file.data, changeEvent.previousFile?.data)
      const slug = normalizedContentSlug(file.data)
      if (slug) {
        changedSlugs.add(slug)
      }
      content ||= changeEvent.type !== 'change'
      content ||= pageNavigationChanged(file.data, changeEvent.previousFile?.data)
      listing ||= changeEvent.type !== 'change'
      listing ||= pageListingChanged(file.data, changeEvent.previousFile?.data)
      sitemap ||= changeEvent.type !== 'change'
      sitemap ||= pageSitemapChanged(file.data, changeEvent.previousFile?.data)
      security ||= isSecurityPolicyChange(changeEvent)
      continue
    }

    if (isGraphFilePage(changeEvent.path)) {
      graph ||= changeEvent.type !== 'change'
      search ||= changeEvent.type !== 'change'
      changedSlugs.add(filePageSlug(changeEvent.path))
      continue
    }

    if (!isContentPageFile(changeEvent.path) && changeEvent.type !== 'change') {
      search = true
      changedSlugs.add(filePageSlug(changeEvent.path))
    }
  }

  return {
    content,
    graph,
    search,
    security,
    listing,
    sitemap,
    changedSlugs,
    affectedFolders: affectedFolderFeedSlugs(changeEvents),
  }
}

export const ContentIndex: QuartzEmitterPlugin<Partial<Options>> = userOpts => {
  const opts: Options = { ...defaultOptions, ...userOpts }
  const folderFeedSlugs = new Set<string>()
  let cachedLinkIndex: ContentIndexMap | undefined
  let cachedContentIndexJson: SerializedIndexCache | undefined
  let cachedRootAtomFeed: AtomFeedCache | undefined
  let cachedSearchIndexJson: SerializedIndexCache | undefined

  return {
    name: 'ContentIndex',
    async *emit(ctx, content, _resources) {
      const cfg = ctx.cfg.configuration
      const writesAtom = opts.enableAtom && !ctx.argv.watch
      const linkIndex = buildLinkIndex(ctx, content, opts)
      cachedLinkIndex = new Map(linkIndex)

      yield write({
        ctx,
        content: `# As a condition of accessing this website, you agree to abide by the following
# content signals:

# (a)  If a content-signal = yes, you may collect content for the corresponding
#      use.
# (b)  If a content-signal = no, you may not collect content for the
#      corresponding use.
# (c)  If the website operator does not include a content signal for a
#      corresponding use, the website operator neither grants nor restricts
#      permission via content signal with respect to the corresponding use.

# The content signals and their meanings are:

# search:   building a search index and providing search results (e.g., returning
#           hyperlinks and short excerpts from your website's contents). Search does not
#           include providing AI-generated search summaries.
# ai-input: inputting content into one or more AI models (e.g., retrieval
#           augmented generation, grounding, or other real-time taking of content for
#           generative AI search answers).
# ai-train: training or fine-tuning AI models.

# ANY RESTRICTIONS EXPRESSED VIA CONTENT SIGNALS ARE EXPRESS RESERVATIONS OF
# RIGHTS UNDER ARTICLE 4 OF THE EUROPEAN UNION DIRECTIVE 2019/790 ON COPYRIGHT
# AND RELATED RIGHTS IN THE DIGITAL SINGLE MARKET.

User-Agent: *
Content-signal: search=yes,ai-train=yes,ai-input=yes
Allow: /

User-agent: Amazonbot
Disallow: /

User-agent: Applebot-Extended
Disallow: /

User-agent: Bytespider
Disallow: /

User-agent: meta-externalagent
Disallow: /

User-agent: PerplexityBot
Disallow: /

User-agent: Perplexity-User
Disallow: /

Sitemap: https://${joinSegments(cfg.baseUrl ?? 'https://example.com', 'sitemap.xml')}
`,
        slug: 'robots' as FullSlug,
        ext: '.txt',
      })

      if (opts.enableSecurity) {
        yield* writeSecurityFiles(ctx, cfg, linkIndex)
      }

      if (opts.enableSiteMap) {
        yield writeSiteMap(ctx, cfg, linkIndex)
      }

      if (writesAtom) {
        const rootFeedWrite = await writeRootAtomFeed(ctx, cfg, linkIndex, opts)
        cachedRootAtomFeed = rootFeedWrite.cache
        yield rootFeedWrite.file
        const folderFeedMap = buildFolderFeedMap(linkIndex)
        folderFeedSlugs.clear()
        const sortedFolderFeeds = Array.from(folderFeedMap.entries()).sort(([a], [b]) =>
          a.localeCompare(b),
        )

        const folderFeedWrites = await mapConcurrent(
          sortedFolderFeeds,
          defaultIoConcurrency,
          async ([folderSlug, folderIndex]) => {
            const file = await writeFolderAtomFeed(
              ctx,
              cfg,
              linkIndex,
              folderSlug,
              folderIndex,
              opts,
            )
            return file ? { folderSlug, file } : undefined
          },
        )

        for (const write of folderFeedWrites) {
          if (!write) continue
          folderFeedSlugs.add(write.folderSlug.replace(/^\/+/, ''))
          yield write.file
        }
      } else {
        folderFeedSlugs.clear()
      }

      addGraphFilePages(ctx, linkIndex)
      const contentIndexWrite = await writeContentIndexJson(ctx, linkIndex)
      cachedContentIndexJson = contentIndexWrite.cache
      yield contentIndexWrite.file
      const searchIndexWrite = await writeSearchIndexJson(ctx, buildSearchIndex(ctx, linkIndex))
      cachedSearchIndexJson = searchIndexWrite.cache
      yield* searchIndexWrite.files

      if (
        process.env.NODE_ENV === 'development' ||
        process.env.NODE_ENV === 'test' ||
        ctx.argv.watch
      ) {
        const slug = joinSegments('.well-known', 'appspecific', 'com.chrome.devtools') as FullSlug
        const root = path.resolve(path.dirname(ctx.argv.directory)) as FilePath
        const dir = path.dirname(joinSegments(ctx.argv.output, slug)) as FilePath
        await fs.mkdir(dir, { recursive: true })
        yield write({
          ctx,
          content: JSON.stringify({
            workspace: {
              root,
              uuid: crypto
                .createHash('sha256')
                .update(root)
                .digest('hex')
                .replace(/(.{8})(.{4})(.{4})(.{4})(.{12})/, '$1-$2-$3-$4-$5')
                .substring(0, 36),
            },
          }),
          slug,
          ext: '.json',
        })
      }
    },
    async *partialEmit(ctx, content, _resources, changeEvents) {
      const writesAtom = opts.enableAtom && !ctx.argv.watch
      const plan = planPartialContentIndex(changeEvents)
      if (!plan.content && !plan.graph && !plan.search && !plan.security) {
        return
      }

      if (
        plan.content &&
        plan.search &&
        !plan.graph &&
        !plan.security &&
        !plan.sitemap &&
        plan.affectedFolders.size === 0 &&
        cachedLinkIndex &&
        cachedContentIndexJson &&
        cachedSearchIndexJson
      ) {
        const titleUpdate = titlePatchContentIndexUpdates(cachedLinkIndex, changeEvents)
        if (titleUpdate) {
          const perf = new PerfTimer()
          const cfg = ctx.cfg.configuration
          const files: FilePath[] = []
          cachedLinkIndex = new Map(titleUpdate.linkIndex)
          if (writesAtom) {
            const writesRootFeed =
              feedContainsAnySlug(
                titleUpdate.linkIndex,
                titleUpdate.changedSlugs,
                opts.atomLimit,
              ) || feedCacheContainsAnySlug(cachedRootAtomFeed, titleUpdate.changedSlugs)
            if (writesRootFeed) {
              let rootFeedWrite = cachedRootAtomFeed
                ? await writePatchedRootAtomFeed(
                    ctx,
                    cfg,
                    titleUpdate.linkIndex,
                    cachedRootAtomFeed,
                    titleUpdate.changedSlugs,
                  )
                : undefined
              if (!rootFeedWrite) {
                rootFeedWrite = await writeRootAtomFeed(
                  ctx,
                  cfg,
                  titleUpdate.linkIndex,
                  opts,
                  cachedRootAtomFeed,
                  titleUpdate.changedSlugs,
                  true,
                )
              }
              cachedRootAtomFeed = rootFeedWrite.cache
              files.push(rootFeedWrite.file)
            }
          }
          let contentIndexWrite = await writePatchedSerializedIndexJson(
            ctx,
            joinSegments('static', 'contentIndex') as FullSlug,
            cachedContentIndexJson,
            titleUpdate.contentUpdates,
          )
          if (!contentIndexWrite) {
            contentIndexWrite = await writeContentIndexJson(
              ctx,
              titleUpdate.linkIndex,
              cachedContentIndexJson,
              titleUpdate.changedSlugs,
              true,
            )
          }
          cachedContentIndexJson = contentIndexWrite.cache
          files.push(contentIndexWrite.file)
          let searchIndexWrite = await writePatchedSearchIndexChunks(
            ctx,
            cachedSearchIndexJson,
            titleUpdate.searchUpdates,
          )
          if (!searchIndexWrite) {
            searchIndexWrite = await writePatchedSearchIndexJson(
              ctx,
              cachedSearchIndexJson,
              titleUpdate.searchUpdates,
              new Set(),
              true,
            )
          }
          cachedSearchIndexJson = searchIndexWrite.cache
          files.push(...searchIndexWrite.files)
          logBuildSpan(
            ctx.argv,
            'contentIndex:titlePatch',
            `${files.length} files`,
            perf.elapsedMs(),
          )
          yield* files
          return
        }
      }

      if (!plan.content && !plan.graph && plan.search && cachedSearchIndexJson) {
        const patch = buildSearchIndexPatch(ctx, content, opts, changeEvents)
        if (patch) {
          const searchIndexWrite = await writePatchedSearchIndexJson(
            ctx,
            cachedSearchIndexJson,
            patch.updates,
            patch.deletes,
          )
          cachedSearchIndexJson = searchIndexWrite.cache
          cachedLinkIndex = undefined
          yield* searchIndexWrite.files
          return
        }
      }

      const cfg = ctx.cfg.configuration
      const linkIndex = cachedLinkIndex
        ? buildLinkIndex(ctx, content, opts, { previous: cachedLinkIndex, changeEvents })
        : buildLinkIndex(ctx, content, opts)
      cachedLinkIndex = new Map(linkIndex)

      if (opts.enableSecurity && plan.security) {
        yield* writeSecurityFiles(ctx, cfg, linkIndex)
      }

      if (plan.content) {
        if (opts.enableSiteMap && plan.sitemap) {
          yield writeSiteMap(ctx, cfg, linkIndex)
        }

        const writesRootFeed =
          writesAtom &&
          (feedContainsAnySlug(linkIndex, plan.changedSlugs, opts.atomLimit) ||
            feedCacheContainsAnySlug(cachedRootAtomFeed, plan.changedSlugs))

        if (writesAtom && writesRootFeed) {
          const rootFeedWrite = await writeRootAtomFeed(
            ctx,
            cfg,
            linkIndex,
            opts,
            cachedRootAtomFeed,
            plan.changedSlugs,
          )
          cachedRootAtomFeed = rootFeedWrite.cache
          yield rootFeedWrite.file
          if (plan.affectedFolders.size > 0) {
            const folderFeedMap = buildFolderFeedMap(linkIndex)
            folderFeedSlugs.clear()

            for (const [folderSlug, folderIndex] of folderFeedMap) {
              if (hasFeedEntries(folderIndex)) {
                folderFeedSlugs.add(folderSlug.replace(/^\/+/, ''))
              }
            }

            const affectedFeeds = Array.from(plan.affectedFolders).sort((a, b) =>
              a.localeCompare(b),
            )
            const folderWrites = await mapConcurrent(
              affectedFeeds,
              defaultIoConcurrency,
              async folderSlug => {
                const folderIndex = folderFeedMap.get(folderSlug)
                if (!folderIndex) {
                  await removeFolderAtomFeed(ctx, folderSlug)
                  return undefined
                }

                const file = await writeFolderAtomFeed(
                  ctx,
                  cfg,
                  linkIndex,
                  folderSlug,
                  folderIndex,
                  opts,
                )
                if (file) return file
                await removeFolderAtomFeed(ctx, folderSlug)
                return undefined
              },
            )
            for (const file of folderWrites) {
              if (file) yield file
            }
          }
        }

        if (writesAtom && !writesRootFeed) {
          const folderFeedMap = buildFolderFeedMap(linkIndex)
          const affectedFeeds = Array.from(plan.affectedFolders).sort((a, b) => a.localeCompare(b))
          const folderWrites = await mapConcurrent(
            affectedFeeds,
            defaultIoConcurrency,
            async folderSlug => {
              const folderIndex = folderFeedMap.get(folderSlug)
              if (
                !folderIndex ||
                !feedContainsAnySlug(folderIndex, plan.changedSlugs, opts.atomLimit)
              ) {
                return undefined
              }

              const file = await writeFolderAtomFeed(
                ctx,
                cfg,
                linkIndex,
                folderSlug,
                folderIndex,
                opts,
              )
              if (file) return file
              await removeFolderAtomFeed(ctx, folderSlug)
              return undefined
            },
          )
          for (const file of folderWrites) {
            if (file) yield file
          }
        }
      }

      const writesContentIndex = plan.content || plan.graph
      let searchSource = linkIndex
      if (writesContentIndex) {
        searchSource = new Map(linkIndex)
        addGraphFilePages(ctx, searchSource)
        const contentIndexWrite = await writeContentIndexJson(
          ctx,
          searchSource,
          cachedContentIndexJson,
          plan.changedSlugs,
        )
        cachedContentIndexJson = contentIndexWrite.cache
        yield contentIndexWrite.file
      }

      if (plan.search || plan.graph) {
        let patchedSearchIndex = false
        if (!plan.graph && cachedSearchIndexJson) {
          const patch = buildSearchIndexPatch(ctx, content, opts, changeEvents)
          if (patch) {
            const searchIndexWrite = await writePatchedSearchIndexJson(
              ctx,
              cachedSearchIndexJson,
              patch.updates,
              patch.deletes,
            )
            cachedSearchIndexJson = searchIndexWrite.cache
            yield* searchIndexWrite.files
            patchedSearchIndex = true
          }
        }

        if (!patchedSearchIndex) {
          const incrementalSearchSlugs =
            cachedSearchIndexJson && !plan.content && !plan.graph ? plan.changedSlugs : undefined
          const searchIndexWrite = await writeSearchIndexJson(
            ctx,
            buildSearchIndex(ctx, searchSource),
            cachedSearchIndexJson,
            incrementalSearchSlugs,
          )
          cachedSearchIndexJson = searchIndexWrite.cache
          yield* searchIndexWrite.files
        }
      }
    },
    externalResources: ({ argv, cfg }) => {
      const additionalHead: StaticResources['additionalHead'] = [
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css"
          media="print"
          // @ts-ignore
          onload="this.media='all'"
        />,
      ]

      if (opts.enableAtom && !argv.watch) {
        additionalHead.push(
          <link
            rel="alternate"
            type="application/atom+xml"
            title="atom feed"
            href={`https://${cfg.configuration.baseUrl}/index.xml`}
          />,
        )
        additionalHead.push(fileData => (
          <>
            {pageFolderFeedSlugs(fileData, folderFeedSlugs).map(folderSlug => (
              <link
                key={`atom-${folderSlug}`}
                rel="alternate"
                type="application/atom+xml"
                title={`atom feed for ${folderSlug}`}
                href={`https://${cfg.configuration.baseUrl}/${folderSlug}/index.xml`}
              />
            ))}
          </>
        ))
      }

      return { additionalHead }
    },
  }
}
