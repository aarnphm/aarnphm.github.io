import { toHtml } from 'hast-util-to-html'
import fs from 'node:fs/promises'
import type { BuildCtx } from '../../../util/ctx'
import { version } from '../../../../package.json'
import { GlobalConfiguration } from '../../../cfg'
import { formatDate } from '../../../components/Date'
import { i18n } from '../../../i18n'
import { resolveAsset, resolveExtractedStaticResource } from '../../../util/asset-manifest'
import { escapeHTML } from '../../../util/escape'
import {
  FilePath,
  FullSlug,
  getAllSegmentPrefixes,
  joinSegments,
  SimpleSlug,
  simplifySlug,
  sluggify,
} from '../../../util/path'
import { componentCssBundleKey } from '../../../util/resource-bundles'
import { QuartzPluginData } from '../../vfile'
import {
  atomFeedDocument,
  atomFeedStylesheet,
  atomFeedStylesheetHref,
  type AtomFeedPresentation,
} from '../atom-feed'
import { xsltPolyfillPath } from '../component-resources/asset-paths'
import { write, writeKnownChanged } from '../helpers'
import {
  richContentTreeByDetails,
  sanitizeXml,
  type ContentDetails,
  type ContentIndexMap,
  type Options,
} from './link-index'

interface AtomFeedOptions {
  limit?: number
  title?: string
  subtitle?: string
  linkPath?: string
  category?: string
  introHtml?: string
}

export type AtomFeedCache = { entries: Map<FullSlug, string> }
type AtomFeedResult = { content: string; cache: AtomFeedCache }

function sanitizeNullable(input?: string | null): string | undefined {
  if (input == null) {
    return undefined
  }
  const sanitized = sanitizeXml(input)
  return sanitized
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

function resolveRichContent(content: ContentDetails): string {
  if (content.richContent !== undefined) {
    return content.richContent
  }
  const tree = richContentTreeByDetails.get(content)
  if (!tree) {
    content.richContent = ''
    return ''
  }
  const rendered = sanitizeXml(escapeHTML(toHtml(tree, { allowDangerousHtml: true })))
  content.richContent = rendered
  richContentTreeByDetails.delete(content)
  return rendered
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
  const richContent = resolveRichContent(content)

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
  presentation: AtomFeedPresentation,
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

  const latestUpdated = limitedEntries.reduce<Date | undefined>((latest, [, content]) => {
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

  const feedContent = `  <title>${feedTitle}</title>
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
  ${items.join('')}`
  const feed = atomFeedDocument(feedContent, presentation)
  return { content: feed, cache: { entries: entryCache } }
}

function limitedFeedEntries(
  idx: ContentIndexMap,
  limit?: number,
): Array<[FullSlug, ContentDetails]> {
  const filteredEntries = Array.from(idx)
    .sort(([slug1, f1], [slug2, f2]) => {
      if (f1.date && f2.date) {
        const dateComparison = f2.date.getTime() - f1.date.getTime()
        if (dateComparison) return dateComparison
      } else if (f1.date && !f2.date) {
        return -1
      } else if (!f1.date && f2.date) {
        return 1
      }

      return f1.title.localeCompare(f2.title) || (slug1 < slug2 ? -1 : slug1 > slug2 ? 1 : 0)
    })
    .filter(([slug, content]) => shouldIncludeInFeed(slug, content))

  return filteredEntries.slice(0, limit ?? filteredEntries.length)
}

export function feedContainsAnySlug(
  idx: ContentIndexMap,
  slugs: ReadonlySet<FullSlug>,
  limit?: number,
): boolean {
  if (slugs.size === 0) return false
  return limitedFeedEntries(idx, limit).some(([slug]) => slugs.has(slug))
}

export function feedCacheContainsAnySlug(
  cache: AtomFeedCache | undefined,
  slugs: ReadonlySet<FullSlug>,
): boolean {
  if (!cache || slugs.size === 0) return false
  for (const slug of slugs) {
    if (cache.entries.has(slug)) return true
  }
  return false
}

export function buildFolderFeedMap(linkIndex: ContentIndexMap): Map<string, ContentIndexMap> {
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

export function hasFeedEntries(folderIndex: ContentIndexMap): boolean {
  return Array.from(folderIndex).some(([slug, content]) => shouldIncludeInFeed(slug, content))
}

function atomFeedPresentation(ctx: BuildCtx): AtomFeedPresentation {
  return {
    stylesheetHref: atomFeedStylesheetHref,
    polyfillSrc: `/${resolveAsset(ctx, xsltPolyfillPath)}`,
  }
}

export async function writeAtomFeedStylesheet(ctx: BuildCtx): Promise<FilePath> {
  return write({
    ctx,
    content: atomFeedStylesheet({
      indexStylesheetHref: `/${resolveAsset(ctx, 'index.css')}`,
      componentStylesheetHref: `/${resolveExtractedStaticResource(ctx, componentCssBundleKey)}`,
    }),
    slug: joinSegments('static', 'feed'),
    ext: '.xsl',
  })
}

export async function writeRootAtomFeed(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
  opts: Options,
  previous?: AtomFeedCache,
  changedSlugs?: ReadonlySet<FullSlug>,
  knownChanged = false,
): Promise<{ file: FilePath; cache: AtomFeedCache }> {
  const feed = generateAtomFeed(
    cfg,
    linkIndex,
    atomFeedPresentation(ctx),
    { limit: opts.atomLimit },
    previous,
    changedSlugs,
  )
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

export async function writePatchedRootAtomFeed(
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

const simplifiedLinkIndexCache = new WeakMap<ContentIndexMap, Map<SimpleSlug, ContentDetails>>()

function lookupSimplifiedLinkIndex(
  linkIndex: ContentIndexMap,
  folderSlug: string,
): ContentDetails | undefined {
  let simplified = simplifiedLinkIndexCache.get(linkIndex)
  if (!simplified) {
    simplified = new Map()
    for (const [existingSlug, details] of linkIndex.entries()) {
      const key = simplifySlug(existingSlug)
      if (!simplified.has(key)) {
        simplified.set(key, details)
      }
    }
    simplifiedLinkIndexCache.set(linkIndex, simplified)
  }
  return simplified.get(folderSlug as SimpleSlug)
}

export async function writeFolderAtomFeed(
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
  const folderDetails = linkIndex.get(folderKey) ?? lookupSimplifiedLinkIndex(linkIndex, folderSlug)
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

  const feed = generateAtomFeed(cfg, folderIndex, atomFeedPresentation(ctx), {
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

export async function removeFolderAtomFeed(ctx: BuildCtx, folderSlug: string) {
  const file = joinSegments(ctx.argv.output, folderSlug, 'index.xml') as FilePath
  await fs.rm(file, { force: true })
}

export function pageFolderFeedSlugs(
  fileData: QuartzPluginData,
  folderFeedSlugs: ReadonlySet<string>,
) {
  const slug = fileData.slug
  if (typeof slug !== 'string') return []
  const prefixes = getAllSegmentPrefixes(slug)
  prefixes.pop()
  return prefixes
    .map(prefix => prefix.replace(/^\/+|\/+$/g, ''))
    .filter(prefix => prefix.length > 0 && folderFeedSlugs.has(prefix))
}
