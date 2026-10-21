import { Root } from 'hast'
import path from 'path'
import { ReadTimeResults } from 'reading-time'
import type { ChangeEvent } from '../../../types/plugin'
import type { BuildCtx } from '../../../util/ctx'
import { getDate } from '../../../components/Date'
import { descriptionToPlainText } from '../../../util/description'
import { FilePath, FullSlug, joinSegments, SimpleSlug, slugifyFilePath } from '../../../util/path'
import { isSearchExcluded } from '../../../util/search-exclude'
import { ArenaData } from '../../transformers/arena'
import { ProcessedContent, QuartzPluginData } from '../../vfile'

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
  | 'course'
  | 'courses'
  | 'fr'
  | 'university'
  | 'flashcards'
  | 'resource'
  | 'study-log'
  | 'statements'
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

export interface Options {
  enableSiteMap: boolean
  enableAtom: boolean
  atomLimit?: number
  includeEmptyFiles: boolean
  enableSecurity: boolean
}

// eslint-disable-next-line no-control-regex
const INVALID_XML_CHARS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g

export function sanitizeXml(input: string): string {
  return input.replace(INVALID_XML_CHARS, '')
}

export function isContentPageFile(fp: FilePath): boolean {
  return fp.endsWith('.md') || fp.endsWith('.base') || fp.endsWith('.canvas')
}

export function isGraphFilePage(fp: FilePath): boolean {
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

export function addFilePageToIndex(ctx: BuildCtx, idx: ContentIndexMap, file: FilePath) {
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

export const richContentTreeByDetails = new WeakMap<ContentDetails, Root>()

type LinkIndexUpdate = { previous: ContentIndexMap; changeEvents: readonly ChangeEvent[] }
type ReusableContentDetails = Pick<ContentDetails, 'content' | 'richContent' | 'readingTime'> & {
  richContentTree?: Root
}

export function normalizedContentSlug(fileData: QuartzPluginData): FullSlug | undefined {
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

export function contentDataBySlug(content: ProcessedContent[]): Map<string, QuartzPluginData> {
  const contentBySlug = new Map<string, QuartzPluginData>()
  for (const [, file] of content) {
    const slug = file.data.slug
    if (typeof slug === 'string') {
      contentBySlug.set(slug, file.data)
    }
  }
  return contentBySlug
}

export function processedContentBySlug(
  content: ProcessedContent[],
): Map<FullSlug, ProcessedContent> {
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
      richContentTree: richContentTreeByDetails.get(previousDetails),
      readingTime: previousDetails.readingTime,
    })
  }

  return reusable
}

export function addProcessedContentToIndex(
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

  if (file.data.flashcards && !file.data.flashcardRoute?.default) return

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
      description: descriptionToPlainText(
        file.data.frontmatter?.description ?? `Canvas with ${jcast.data.nodeMap.size} nodes`,
        renderedSlug,
      ),
      fileData: file.data,
    })
    return
  }

  if (!opts.includeEmptyFiles && (!file.data.text || file.data.text === '')) {
    return
  }

  const links = filterVisibleLinks(file.data.links ?? [], contentBySlug)
  const isProtected = file.data.frontmatter?.protected === true
  const content = reusableDetails?.content ?? (isProtected ? '' : sanitizeXml(file.data.text ?? ''))
  const readingTime = reusableDetails?.readingTime ?? {
    minutes: Math.ceil(file.data.readingTime ? file.data.readingTime.minutes! : 0),
    words: Math.ceil(file.data.readingTime ? file.data.readingTime.words! : 0),
  }

  const details: ContentDetails = {
    slug,
    title: file.data.frontmatter ? file.data.frontmatter.title! : '',
    links,
    filePath: file.data.filePath!,
    fileName: contentFileName(ctx, file.data),
    tags: file.data.frontmatter?.tags ?? [],
    aliases: file.data.frontmatter?.aliases ?? [],
    content,
    richContent: isProtected || !includeRichContent ? '' : reusableDetails?.richContent,
    date: date,
    readingTime,
    fileData: file.data,
    layout: file.data.frontmatter!.pageLayout,
    description: descriptionToPlainText(file.data.rawDescription ?? file.data.description, slug),
    protected: isProtected,
  }
  if (!isProtected && includeRichContent && details.richContent === undefined) {
    richContentTreeByDetails.set(details, reusableDetails?.richContentTree ?? (tree as Root))
  }
  linkIndex.set(slug, details)

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

export function buildLinkIndex(
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

  const includeRichContent = Boolean(opts.enableAtom && !ctx.argv.watch)
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
      includeRichContent,
      reusableDetails.get(slug),
    )
  }

  refreshVisibleLinks(linkIndex, contentBySlug)
  return linkIndex
}

export function addGraphFilePages(ctx: BuildCtx, linkIndex: ContentIndexMap) {
  for (const file of ctx.allFiles) {
    if (isGraphFilePage(file)) {
      addFilePageToIndex(ctx, linkIndex, file)
    }
  }
}

export function buildSearchIndex(ctx: BuildCtx, linkIndex: ContentIndexMap): ContentIndexMap {
  const searchIndex: ContentIndexMap = new Map()
  for (const [slug, details] of linkIndex) {
    if (isSearchExcluded(slug) || isSearchExcluded(details.fileName)) continue
    searchIndex.set(slug, details)
  }
  for (const file of ctx.allFiles) {
    if (isContentPageFile(file) || isSearchExcluded(file)) {
      continue
    }

    addFilePageToIndex(ctx, searchIndex, file)
  }

  return searchIndex
}
