import { Root } from 'hast'
import { toHtml } from 'hast-util-to-html'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'path'
import { ReadTimeResults } from 'reading-time'
import type { ChangeEvent } from '../../types/plugin'
import type { BuildCtx } from '../../util/ctx'
import { version } from '../../../package.json'
import { GlobalConfiguration } from '../../cfg'
import { formatDate, getDate } from '../../components/Date'
import { i18n } from '../../i18n'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { escapeHTML } from '../../util/escape'
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
import { ArenaData } from '../transformers/arena'
import { ProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'

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
export type ContentDetails = {
  slug: string
  title: string
  filePath: FilePath
  links: SimpleSlug[]
  aliases: string[]
  tags: string[]
  layout: ContentLayout
  content: string
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

interface AtomFeedOptions {
  limit?: number
  title?: string
  subtitle?: string
  linkPath?: string
  category?: string
  introHtml?: string
}

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

function serializeContentIndex(idx: ContentIndexMap) {
  return Object.fromEntries(
    Array.from(idx).map(([slug, content]) => {
      const serializable = { ...content }
      delete serializable.fileData
      delete serializable.richContent
      return [slug, serializable]
    }),
  )
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

function generateAtomFeed(
  cfg: GlobalConfiguration,
  idx: ContentIndexMap,
  options: AtomFeedOptions = {},
): string {
  const base = cfg.baseUrl ?? 'example.com'
  const limit = options.limit

  const createURLEntry = (slug: SimpleSlug, content: ContentDetails): string => {
    const publishedDate = content.date ?? new Date()
    const frontmatterModified = content.fileData?.frontmatter?.modified
    let updatedDate = frontmatterModified ? new Date(frontmatterModified) : content.date
    if (!updatedDate || Number.isNaN(updatedDate.getTime())) {
      updatedDate = publishedDate
    }

    const summary = escapeHTML(sanitizeNullable(content.description) ?? '')
    const richContent = sanitizeXml(content.richContent ?? '')

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

  const limitedEntries = filteredEntries.slice(0, limit ?? filteredEntries.length)
  const items = limitedEntries
    .map(([slug, content]) => createURLEntry(simplifySlug(slug), content))
    .join('')

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

  return `<?xml version="1.0" encoding="UTF-8" ?>
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
  ${items}
</feed>`
}

function buildLinkIndex(
  ctx: BuildCtx,
  content: ProcessedContent[],
  opts: Options,
): ContentIndexMap {
  const linkIndex: ContentIndexMap = new Map()
  for (const [tree, file] of content) {
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
        links: file.data.links ?? [],
        filePath: file.data.filePath!,
        fileName: file.data.filePath!,
        tags: ['canvas', ...(file.data.frontmatter?.tags ?? [])],
        aliases: file.data.frontmatter?.aliases ?? [],
        content: sanitizeXml(searchableContent),
        richContent: '',
        date: date,
        readingTime: {
          minutes: Math.max(1, Math.ceil(searchableContent.split(/\s+/).length / 200)),
          words: searchableContent.split(/\ks+/).filter(w => w.length > 0).length,
        },
        layout: file.data.frontmatter?.pageLayout ?? 'default',
        description:
          file.data.frontmatter?.description ?? `Canvas with ${jcast.data.nodeMap.size} nodes`,
        fileData: file.data,
      })
      continue
    }

    if (opts.includeEmptyFiles || (file.data.text && file.data.text !== '')) {
      const links = (file.data.links ?? []).filter(link => {
        const targetFile = content.find(([, target]) => String(target.data.slug) === link)?.[1]
        if (
          targetFile?.data.frontmatter?.noindex === true ||
          targetFile?.data.frontmatter?.protected === true
        )
          return false

        return true
      })

      file.data.links = links

      const rawHtml = toHtml(tree as Root, { allowDangerousHtml: true })
      const sanitizedRichContent = sanitizeXml(escapeHTML(rawHtml))
      const sanitizedContent = sanitizeXml(file.data.text ?? '')
      const isProtected = file.data.frontmatter?.protected === true

      const getFileName = () => {
        const fullPath = file.data.filePath!
        const relativePath = fullPath.substring(ctx.argv.directory.length + 1)
        if (relativePath.endsWith('.bases')) return relativePath as FilePath
        return relativePath.replace('.md', '') as FilePath
      }

      linkIndex.set(slug, {
        slug,
        title: file.data.frontmatter ? file.data.frontmatter.title! : '',
        links,
        filePath: file.data.filePath!,
        fileName: getFileName(),
        tags: file.data.frontmatter?.tags ?? [],
        aliases: file.data.frontmatter?.aliases ?? [],
        content: isProtected ? '' : sanitizedContent,
        richContent: isProtected ? '' : sanitizedRichContent,
        date: date,
        readingTime: {
          minutes: Math.ceil(file.data.readingTime ? file.data.readingTime.minutes! : 0),
          words: Math.ceil(file.data.readingTime ? file.data.readingTime.words! : 0),
        },
        fileData: file.data,
        layout: file.data.frontmatter!.pageLayout,
        description: file.data.description,
        protected: isProtected,
      })

      if (slug === 'arena') {
        const arenaData = file.data.arenaData as ArenaData | undefined
        if (arenaData) {
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
      }
    }
  }

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
): Promise<FilePath> {
  return write({
    ctx,
    content: generateAtomFeed(cfg, linkIndex, { limit: opts.atomLimit }),
    slug: 'index' as FullSlug,
    ext: '.xml',
  })
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

  return write({
    ctx,
    content: generateAtomFeed(cfg, folderIndex, {
      limit: opts.atomLimit,
      title,
      subtitle,
      linkPath: folderSlug,
      category: folderSlug,
      introHtml: folderIntroHtml,
    }),
    slug: joinSegments(folderSlug, 'index') as FullSlug,
    ext: '.xml',
  })
}

async function removeFolderAtomFeed(ctx: BuildCtx, folderSlug: string) {
  const file = joinSegments(ctx.argv.output, folderSlug, 'index.xml') as FilePath
  await fs.rm(file, { force: true })
}

async function writeContentIndexJson(ctx: BuildCtx, linkIndex: ContentIndexMap): Promise<FilePath> {
  return write({
    ctx,
    content: JSON.stringify(serializeContentIndex(linkIndex)),
    slug: joinSegments('static', 'contentIndex') as FullSlug,
    ext: '.json',
  })
}

async function writeSearchIndexJson(
  ctx: BuildCtx,
  searchIndex: ContentIndexMap,
): Promise<FilePath> {
  return write({
    ctx,
    content: JSON.stringify(serializeContentIndex(searchIndex)),
    slug: joinSegments('static', 'searchIndex') as FullSlug,
    ext: '.json',
  })
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

function isSecurityPolicyChange(changeEvent: ChangeEvent): boolean {
  if (changeEvent.file?.data.slug === 'security-policy') {
    return true
  }
  const normalized = changeEvent.path.replace(/\\/g, '/')
  return normalized === 'security policy.md' || normalized.endsWith('/security policy.md')
}

type ContentIndexPartialPlan = {
  content: boolean
  graph: boolean
  search: boolean
  security: boolean
  affectedFolders: Set<string>
}

function planPartialContentIndex(changeEvents: readonly ChangeEvent[]): ContentIndexPartialPlan {
  let content = false
  let graph = false
  let search = false
  let security = false

  for (const changeEvent of changeEvents) {
    if (isSourcePath(changeEvent.path)) {
      continue
    }

    const hasParsedFile = changeEvent.file !== undefined
    if (hasParsedFile) {
      content = true
      search = true
      security ||= isSecurityPolicyChange(changeEvent)
      continue
    }

    if (isGraphFilePage(changeEvent.path)) {
      graph ||= changeEvent.type !== 'change'
      search ||= changeEvent.type !== 'change'
      continue
    }

    if (!isContentPageFile(changeEvent.path) && changeEvent.type !== 'change') {
      search = true
    }
  }

  return {
    content,
    graph,
    search,
    security,
    affectedFolders: affectedFolderFeedSlugs(changeEvents),
  }
}

export const ContentIndex: QuartzEmitterPlugin<Partial<Options>> = userOpts => {
  const opts: Options = { ...defaultOptions, ...userOpts }
  const folderFeedSlugs = new Set<string>()

  return {
    name: 'ContentIndex',
    async *emit(ctx, content, _resources) {
      const cfg = ctx.cfg.configuration
      const linkIndex = buildLinkIndex(ctx, content, opts)

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

      if (opts.enableAtom) {
        yield writeRootAtomFeed(ctx, cfg, linkIndex, opts)
        const folderFeedMap = buildFolderFeedMap(linkIndex)
        folderFeedSlugs.clear()
        const sortedFolderFeeds = Array.from(folderFeedMap.entries()).sort(([a], [b]) =>
          a.localeCompare(b),
        )

        for (const [folderSlug, folderIndex] of sortedFolderFeeds) {
          const file = await writeFolderAtomFeed(ctx, cfg, linkIndex, folderSlug, folderIndex, opts)
          if (!file) {
            continue
          }
          folderFeedSlugs.add(folderSlug.replace(/^\/+/, ''))
          yield file
        }
      }

      addGraphFilePages(ctx, linkIndex)
      yield writeContentIndexJson(ctx, linkIndex)
      yield writeSearchIndexJson(ctx, buildSearchIndex(ctx, linkIndex))

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
      const plan = planPartialContentIndex(changeEvents)
      if (!plan.content && !plan.graph && !plan.search && !plan.security) {
        return
      }

      const cfg = ctx.cfg.configuration
      const linkIndex = buildLinkIndex(ctx, content, opts)

      if (opts.enableSecurity && plan.security) {
        yield* writeSecurityFiles(ctx, cfg, linkIndex)
      }

      if (plan.content) {
        if (opts.enableSiteMap) {
          yield writeSiteMap(ctx, cfg, linkIndex)
        }

        if (opts.enableAtom) {
          yield writeRootAtomFeed(ctx, cfg, linkIndex, opts)
          const folderFeedMap = buildFolderFeedMap(linkIndex)
          folderFeedSlugs.clear()

          for (const [folderSlug, folderIndex] of folderFeedMap) {
            if (hasFeedEntries(folderIndex)) {
              folderFeedSlugs.add(folderSlug.replace(/^\/+/, ''))
            }
          }

          const affectedFeeds = Array.from(plan.affectedFolders).sort((a, b) => a.localeCompare(b))
          for (const folderSlug of affectedFeeds) {
            const folderIndex = folderFeedMap.get(folderSlug)
            if (!folderIndex) {
              await removeFolderAtomFeed(ctx, folderSlug)
              continue
            }

            const file = await writeFolderAtomFeed(
              ctx,
              cfg,
              linkIndex,
              folderSlug,
              folderIndex,
              opts,
            )
            if (file) {
              yield file
            } else {
              await removeFolderAtomFeed(ctx, folderSlug)
            }
          }
        }
      }

      const writesContentIndex = plan.content || plan.graph
      if (writesContentIndex) {
        addGraphFilePages(ctx, linkIndex)
        yield writeContentIndexJson(ctx, linkIndex)
      }

      if (plan.content || plan.graph || plan.search) {
        yield writeSearchIndexJson(ctx, buildSearchIndex(ctx, linkIndex))
      }
    },
    externalResources: ({ cfg }) => {
      const additionalHead = [
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css"
          media="print"
          // @ts-ignore
          onload="this.media='all'"
        />,
      ]

      if (opts.enableAtom) {
        additionalHead.push(
          <link
            rel="alternate"
            type="application/atom+xml"
            title="atom feed"
            href={`https://${cfg.configuration.baseUrl}/index.xml`}
          />,
        )
        for (const folderSlug of folderFeedSlugs) {
          additionalHead.push(
            <link
              key={`atom-${folderSlug}`}
              rel="alternate"
              type="application/atom+xml"
              title={`atom feed for ${folderSlug}`}
              href={`https://${cfg.configuration.baseUrl}/${folderSlug}/index.xml`}
            />,
          )
        }
      }

      return { additionalHead }
    },
  }
}
