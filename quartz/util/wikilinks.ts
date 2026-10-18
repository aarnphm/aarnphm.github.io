import isAbsoluteUrl from 'is-absolute-url'
import type { Wikilink, WikilinkData } from '../extensions/micromark-extension-ofm-wikilinks'
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

const PARSE_WIKILINK_CACHE_MAX = 4096
const parseWikilinkCache = new Map<string, WikilinkData | null>()

const unescapeMarkdownString = (str: string): string =>
  str.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g, '$1')

function parseMetadata(raw: string): Record<string, unknown> | undefined {
  try {
    return JSON.parse(`{${raw}}`) as Record<string, unknown>
  } catch {
    try {
      const normalized = raw
        .replace(/(\w+):/g, '"$1":')
        .replace(/'/g, '"')
        .replace(/,\s*([}\]])/g, '$1')
      return JSON.parse(`{${normalized}}`) as Record<string, unknown>
    } catch {
      return undefined
    }
  }
}

function applyAnchorSegment(data: WikilinkData, rawChunk: string): void {
  let anchorText = unescapeMarkdownString(rawChunk)
  if (anchorText.endsWith('\\')) {
    anchorText = anchorText.slice(0, -1)
  }
  const isBlockRef = anchorText.startsWith('^')
  let cleanAnchor = isBlockRef ? anchorText.slice(1) : anchorText
  if (isBlockRef || data.anchorIsBlockRef) {
    data.anchorIsBlockRef = true
  }
  const segments = data.anchorSegments ?? []
  segments.push(cleanAnchor)
  data.anchorSegments = segments

  if (cleanAnchor.includes('/') && cleanAnchor.includes(' ')) {
    const lastSlash = cleanAnchor.lastIndexOf('/')
    const afterSlash = cleanAnchor.substring(lastSlash + 1)
    const spaceIndex = afterSlash.indexOf(' ')
    if (spaceIndex >= 0) {
      const headingText = afterSlash.substring(spaceIndex + 1).trim()
      if (headingText) {
        cleanAnchor = headingText
      }
    }
  }

  cleanAnchor = cleanAnchor
    .replace(/\$([^$]+)\$/g, (_, content: string) => content.trim())
    .replace(/\s+/g, ' ')
    .trim()

  data.anchorText = cleanAnchor
  const slugifiedAnchor = slugAnchor(cleanAnchor)
  const blockPrefix = data.anchorIsBlockRef ? '^' : ''
  data.anchor = `#${blockPrefix}${slugifiedAnchor}`
}

function isEscapedAt(text: string, index: number): boolean {
  let backslashes = 0
  for (let i = index - 1; i >= 0 && text.charCodeAt(i) === 92; i--) {
    backslashes++
  }
  return (backslashes & 1) === 1
}

function tryParseWikilinkAt(text: string, start: number): WikilinkRange | null {
  const len = text.length
  let pos = start
  let embed = false
  if (text[pos] === '!') {
    if (text[pos + 1] !== '[' || text[pos + 2] !== '[') return null
    embed = true
    pos += 3
  } else if (text[pos] === '[' && text[pos + 1] === '[') {
    pos += 2
  } else {
    return null
  }

  const data: WikilinkData = { raw: '', target: '', embed }

  if (pos >= len) return null
  let ch = text[pos]
  if (ch === '\n' || ch === '\r') return null

  let phase: 'anchorMarker' | 'anchorStart' | 'metadataStart' | 'aliasMarker' | 'close' = 'close'
  if (ch === '#') {
    phase = 'anchorMarker'
  } else if (ch === '|') {
    phase = 'aliasMarker'
  } else if (ch === ']') {
    phase = 'close'
  } else {
    const targetStartPos = pos
    let prevBackslash = false
    let transitioned = false
    while (pos < len) {
      ch = text[pos]
      if (ch === '\n' || ch === '\r') return null
      if (ch === '\\' && !prevBackslash) {
        prevBackslash = true
        pos++
        continue
      }
      if (ch === '#' && !prevBackslash) {
        let t = unescapeMarkdownString(text.slice(targetStartPos, pos))
        if (t.endsWith('\\')) t = t.slice(0, -1)
        data.target = t
        phase = 'anchorMarker'
        transitioned = true
        break
      }
      if (ch === '|') {
        let t = unescapeMarkdownString(text.slice(targetStartPos, pos))
        if (t.endsWith('\\')) t = t.slice(0, -1)
        data.target = t
        phase = 'aliasMarker'
        transitioned = true
        break
      }
      if (ch === ']' && !prevBackslash) {
        let t = unescapeMarkdownString(text.slice(targetStartPos, pos))
        if (t.endsWith('\\')) t = t.slice(0, -1)
        data.target = t
        phase = 'close'
        transitioned = true
        break
      }
      prevBackslash = false
      pos++
    }
    if (!transitioned) return null
  }

  if (phase === 'anchorMarker') {
    pos++
    if (pos >= len) return null
    if (text[pos] === '{') {
      pos++
      phase = 'metadataStart'
    } else {
      phase = 'anchorStart'
    }
  }

  while (phase === 'anchorStart') {
    if (pos >= len) return null
    ch = text[pos]
    if (ch === '\n' || ch === '\r') return null
    if (ch === '|') {
      phase = 'aliasMarker'
      break
    }
    if (ch === ']') {
      phase = 'close'
      break
    }
    let segStart = pos
    let prevBackslash = false
    let transitioned = false
    while (pos < len) {
      ch = text[pos]
      if (ch === '\n' || ch === '\r') return null
      if (ch === '\\' && !prevBackslash) {
        prevBackslash = true
        pos++
        continue
      }
      if (ch === '#' && !prevBackslash) {
        applyAnchorSegment(data, text.slice(segStart, pos))
        pos++
        while (pos < len && text[pos] === '#') pos++
        if (pos >= len || text[pos] === '\n' || text[pos] === '\r') return null
        if (text[pos] === '{') {
          pos++
          phase = 'metadataStart'
          transitioned = true
          break
        }
        segStart = pos
        prevBackslash = false
        continue
      }
      if (ch === '|') {
        applyAnchorSegment(data, text.slice(segStart, pos))
        phase = 'aliasMarker'
        transitioned = true
        break
      }
      if (ch === ']' && !prevBackslash) {
        applyAnchorSegment(data, text.slice(segStart, pos))
        phase = 'close'
        transitioned = true
        break
      }
      prevBackslash = false
      pos++
    }
    if (!transitioned) return null
  }

  if (phase === 'metadataStart') {
    if (pos >= len) return null
    ch = text[pos]
    if (ch === '\n' || ch === '\r') return null
    if (ch === '}') {
      pos++
      if (pos >= len) return null
      if (text[pos] === '|') phase = 'aliasMarker'
      else if (text[pos] === ']') phase = 'close'
      else return null
    } else if (ch === '|') {
      phase = 'aliasMarker'
    } else if (ch === ']') {
      phase = 'close'
    } else {
      const metaStart = pos
      let depth = 1
      let prevBackslash = false
      let closed = false
      while (pos < len) {
        ch = text[pos]
        if (ch === '\n' || ch === '\r') return null
        if (ch === '\\' && !prevBackslash) {
          prevBackslash = true
          pos++
          continue
        }
        if (ch === '{' && !prevBackslash) {
          depth++
          prevBackslash = false
          pos++
          continue
        }
        if (ch === '}' && !prevBackslash) {
          depth--
          if (depth === 0) {
            const metaContent = unescapeMarkdownString(text.slice(metaStart, pos))
            data.metadata = `{${metaContent}}`
            data.metadataParsed = parseMetadata(metaContent)
            pos++
            if (pos >= len) return null
            if (text[pos] === '|') phase = 'aliasMarker'
            else if (text[pos] === ']') phase = 'close'
            else return null
            closed = true
            break
          }
          prevBackslash = false
          pos++
          continue
        }
        prevBackslash = false
        pos++
      }
      if (!closed) return null
    }
  }

  if (phase === 'aliasMarker') {
    pos++
    if (pos >= len) return null
    ch = text[pos]
    if (ch === '\n' || ch === '\r') return null
    if (ch === ']') {
      phase = 'close'
    } else {
      const aliasStartPos = pos
      let prevBackslash = false
      let closed = false
      while (pos < len) {
        ch = text[pos]
        if (ch === '\n' || ch === '\r') return null
        if (ch === '\\' && !prevBackslash) {
          prevBackslash = true
          pos++
          continue
        }
        if (ch === ']' && !prevBackslash) {
          data.alias = unescapeMarkdownString(text.slice(aliasStartPos, pos))
          phase = 'close'
          closed = true
          break
        }
        prevBackslash = false
        pos++
      }
      if (!closed) return null
    }
  }

  if (phase !== 'close' || text[pos] !== ']' || text[pos + 1] !== ']') {
    return null
  }

  const end = pos + 2
  data.raw = text.slice(start, end)

  const targetPath = data.target.trim()
  const ext = getFileExtension(targetPath as FilePath) ?? ''
  const displayAnchor = data.anchor?.trim() ?? ''
  if (ext === '.base' && displayAnchor && !displayAnchor.startsWith('#^')) {
    const anchorText = data.anchorText?.trim()
    const viewSegment = sluggify(
      anchorText && anchorText.length > 0 ? anchorText : displayAnchor.replace(/^#/, ''),
    )
    if (viewSegment.length > 0 && !data.alias && anchorText) {
      data.alias = anchorText
    }
  }

  return { wikilink: data, start, end }
}

export function parseWikilink(raw: string): WikilinkData | null {
  const trimmed = raw.trim()
  if (
    !trimmed ||
    (!trimmed.startsWith('[[') && !trimmed.startsWith('![[')) ||
    !trimmed.endsWith(']]')
  ) {
    return null
  }

  const cached = parseWikilinkCache.get(trimmed)
  if (cached !== undefined) {
    return cached
  }

  const parsed = tryParseWikilinkAt(trimmed, 0)
  const result =
    parsed && parsed.start === 0 && parsed.end === trimmed.length ? parsed.wikilink : null

  if (parseWikilinkCache.size >= PARSE_WIKILINK_CACHE_MAX) {
    const oldestKey = parseWikilinkCache.keys().next().value
    if (oldestKey !== undefined) {
      parseWikilinkCache.delete(oldestKey)
    }
  }
  parseWikilinkCache.set(trimmed, result)
  return result
}

const EXTRACT_WIKILINKS_CACHE_MAX = 4096
const extractWikilinksCache = new Map<string, WikilinkRange[]>()

export function extractWikilinksWithPositions(text: string): WikilinkRange[] {
  if (!text || !text.includes('[[')) {
    return []
  }

  const canCache = text.length <= 256
  if (canCache) {
    const cached = extractWikilinksCache.get(text)
    if (cached !== undefined) {
      return cached
    }
  }

  const ranges: WikilinkRange[] = []
  let i = 0
  const len = text.length
  while (i < len) {
    const nextBracket = text.indexOf('[[', i)
    if (nextBracket === -1) break
    if (isEscapedAt(text, nextBracket)) {
      i = nextBracket + 1
      continue
    }
    const hasBang =
      nextBracket > 0 && text[nextBracket - 1] === '!' && !isEscapedAt(text, nextBracket - 1)
    const start = hasBang ? nextBracket - 1 : nextBracket
    const parsed = tryParseWikilinkAt(text, start)
    if (parsed) {
      ranges.push(parsed)
      i = parsed.end
    } else {
      i = nextBracket + 1
    }
  }

  if (canCache) {
    if (extractWikilinksCache.size >= EXTRACT_WIKILINKS_CACHE_MAX) {
      const oldestKey = extractWikilinksCache.keys().next().value
      if (oldestKey !== undefined) {
        extractWikilinksCache.delete(oldestKey)
      }
    }
    extractWikilinksCache.set(text, ranges)
  }

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
  // escape pipe characters if not already escaped
  // regex: match pipe that's not preceded by backslash (or preceded by even number of backslashes)
  escaped = escaped.replace(/((^|[^\\])(\\\\)*)\|/g, '$1\\|')
  return escaped
}
