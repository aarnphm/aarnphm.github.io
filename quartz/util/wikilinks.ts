import type { Root } from 'mdast'
import type { Node } from 'unist'
import isAbsoluteUrl from 'is-absolute-url'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { visit } from 'unist-util-visit'
import {
  Wikilink,
  WikilinkData,
  isWikilink,
  wikilink,
  wikilinkFromMarkdown,
} from '../extensions/micromark-extension-ofm-wikilinks'
import {
  FilePath,
  FullSlug,
  getFileExtension,
  joinSegments,
  slugAnchor,
  slugifyFilePath,
  sluggify,
  stripSlashes,
} from './path'

export type { WikilinkData, Wikilink }

export interface WikilinkRange {
  wikilink: WikilinkData
  start: number
  end: number
}

export function parseWikilink(raw: string): WikilinkData | null {
  const trimmed = raw.trim()
  if (!trimmed) {
    return null
  }

  const ranges = extractWikilinksWithPositions(trimmed)
  if (ranges.length !== 1) {
    return null
  }

  const [range] = ranges
  if (range.start !== 0 || range.end !== trimmed.length) {
    return null
  }

  return range.wikilink
}

export function extractWikilinksWithPositions(text: string): WikilinkRange[] {
  if (!text || !text.includes('[[')) {
    return []
  }

  const tree: Root = fromMarkdown(text, {
    extensions: [wikilink()],
    mdastExtensions: [wikilinkFromMarkdown({ hasSlug: () => false })],
  })

  const ranges: WikilinkRange[] = []

  visit(tree, (node: Node) => {
    if (!isWikilink(node)) return
    const data = node.data?.wikilink
    if (!data) return
    const start = node.position?.start?.offset
    const end = node.position?.end?.offset
    if (typeof start !== 'number' || typeof end !== 'number') return
    ranges.push({ wikilink: data, start, end })
  })

  ranges.sort((a, b) => a.start - b.start)
  return ranges
}

export function extractWikilinks(text: string): WikilinkData[] {
  return extractWikilinksWithPositions(text).map(range => range.wikilink)
}

export interface ResolvedWikilinkTarget {
  slug: FullSlug
  anchor?: string
}

function ensureFilePath(target: string): FilePath {
  if (target.length === 0) {
    return 'index.md' as FilePath
  }

  if (target.endsWith('/')) {
    return `${target}index.md` as FilePath
  }

  const ext = getFileExtension(target as FilePath)
  if (ext) {
    return target as FilePath
  }

  return `${target}.md` as FilePath
}

export function resolveWikilinkTarget(
  link: WikilinkData,
  currentSlug: FullSlug,
): ResolvedWikilinkTarget | null {
  if (link.target && isAbsoluteUrl(link.target, { httpOnly: false })) {
    return null
  }

  const normalizedTarget = link.target?.replace(/\\/g, '/') ?? ''

  // Remove leading slash if present - wikilinks are absolute from content root
  const target = normalizedTarget.startsWith('/') ? normalizedTarget.slice(1) : normalizedTarget

  // If empty target, link to current page
  if (!target) {
    return { slug: currentSlug, anchor: link.anchor }
  }

  const filePath = ensureFilePath(target)
  const slug = slugifyFilePath(stripSlashes(filePath) as FilePath)
  const ext = getFileExtension(filePath)

  if (ext === '.base' && link.anchor && !link.anchor.startsWith('#^')) {
    const anchorText = link.anchorText?.trim()
    const viewSegment = sluggify(
      anchorText && anchorText.length > 0 ? anchorText : link.anchor.replace(/^#/, ''),
    )
    if (viewSegment.length > 0) {
      const viewSlug = joinSegments(slug, viewSegment) as FullSlug
      return { slug: viewSlug }
    }
  }

  return { slug, anchor: link.anchor }
}

export function stripWikilinkFormatting(text: string): string {
  if (!text) {
    return text
  }

  const ranges = extractWikilinksWithPositions(text)
  if (ranges.length === 0) {
    return text
  }

  let result = ''
  let lastIndex = 0

  for (const range of ranges) {
    if (range.start > lastIndex) {
      result += text.slice(lastIndex, range.start)
    }
    result += range.wikilink.alias ?? range.wikilink.target
    lastIndex = range.end
  }

  if (lastIndex < text.length) {
    result += text.slice(lastIndex)
  }

  return result
}

/**
 * Resolve wikilink anchor text to a heading slug.
 *
 * Matches Obsidian's behavior:
 * - LaTeX normalization: "architectural skeleton of $ mu$" → "architectural-skeleton-of-mu"
 * - Nested paths: "Parent#Child#Grandchild" uses only the last segment ("Grandchild")
 * - No validation of document structure or parent/child relationships
 *
 * The browser handles anchor navigation - we just normalize the text to match
 * the slugs that github-slugger produces for headings.
 *
 * @param anchorText - The anchor portion of a wikilink (e.g., "section" or "Parent#Child")
 * @returns The normalized slug (e.g., "section" or "child")
 *
 * @example
 * resolveAnchor("Section") → "section"
 * resolveAnchor("NVIDIA#cuda") → "cuda"
 * resolveAnchor("architectural skeleton of $ mu$") → "architectural-skeleton-of-mu"
 * resolveAnchor("Parent#Child#Grandchild") → "grandchild"
 */
export function resolveAnchor(anchorText: string): string {
  // if anchor contains #, take the last segment (Obsidian behavior)
  let text = anchorText.trim()
  if (text.includes('#')) {
    const segments = text.split('#')
    text = segments[segments.length - 1].trim()
  }
  // slugify using github-slugger (same as heading slugs)
  return slugAnchor(text)
}

/**
 * escape wikilinks in table context.
 * obsidian requires escaping pipes and hashes inside table cells.
 *
 * @param wikilink - raw wikilink string
 * @returns escaped wikilink suitable for tables
 */
export function escapeWikilinkForTable(wikilink: string): string {
  let escaped = wikilink
  // escape hash for headers
  escaped = escaped.replace('#', '\\#')
  // escape pipe characters if not already escaped
  // regex: match pipe that's not preceded by backslash (or preceded by even number of backslashes)
  escaped = escaped.replace(/((^|[^\\])(\\\\)*)\|/g, '$1\\|')
  return escaped
}
