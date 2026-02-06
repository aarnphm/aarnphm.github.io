import type { Element, ElementContent, Root } from 'hast'
import type { ArenaBlock } from '../plugins/transformers/arena'
import type { QuartzComponentProps } from '../types/component'
import type { FilePath } from './path'
import { transcludeFinal } from '../components/renderPage'
import { htmlToJsx } from './jsx'
import {
  FullSlug,
  joinSegments,
  pathToRoot,
  isAbsoluteURL,
  resolveRelative,
  simplifySlug,
} from './path'

const cloneContent = <T extends ElementContent>(node: T): T =>
  typeof structuredClone === 'function'
    ? structuredClone(node)
    : (JSON.parse(JSON.stringify(node)) as T)

function rebaseUrl(value: string, currentSlug: FullSlug): string {
  if (!value) return value
  // don't touch anchors, absolute URLs, or site-absolute paths
  if (value.startsWith('#') || isAbsoluteURL(value) || value.startsWith('/')) return value
  // value could be ./foo, ../foo, thoughts/foo, etc — make it relative to currentSlug
  const root = pathToRoot(currentSlug)
  return joinSegments(root, value)
}

export function adjustArenaNode(
  raw: ElementContent,
  currentSlug: FullSlug,
  opts?: { rewriteFirstAnchorTo?: FullSlug },
): ElementContent {
  const node = cloneContent(raw)
  let anchorRewritten = false

  const visit = (el: ElementContent) => {
    if (el.type !== 'element') return
    const e = el as Element

    // Rewrite resources
    if (e.properties) {
      const href = e.properties.href as string | undefined
      const src = e.properties.src as string | undefined
      if (href && !isAbsoluteURL(href)) {
        // For top-level heading anchor override target to the channel page
        if (!anchorRewritten && opts?.rewriteFirstAnchorTo && e.tagName === 'a') {
          const original = (e.properties['data-slug'] as string | undefined) ?? href
          e.properties['data-origin-slug'] = original
          const target = opts.rewriteFirstAnchorTo
          if (simplifySlug(currentSlug) === 'arena' && target.startsWith('arena/')) {
            const parts = target.split('/')
            const leaf = parts[parts.length - 1]
            e.properties.href = joinSegments('.', leaf)
            e.properties['data-slug'] = target
          } else {
            e.properties.href = resolveRelative(currentSlug, target)
            e.properties['data-slug'] = target
          }
          anchorRewritten = true
        } else if (!href.startsWith('#')) {
          e.properties.href = rebaseUrl(href, currentSlug)
        }
      }
      if (src && !isAbsoluteURL(src)) {
        e.properties.src = rebaseUrl(src, currentSlug)
      }
    }

    if (e.children) e.children.forEach(visit)
  }

  visit(node)
  return node
}

export function toArenaJsx(
  filePath: string,
  node: ElementContent,
  currentSlug: FullSlug,
  componentData: QuartzComponentProps,
) {
  const adjusted = adjustArenaNode(node, currentSlug)
  const root: Root = { type: 'root', children: [adjusted] }
  const visited = new Set<FullSlug>([currentSlug])
  const processed = transcludeFinal(root, componentData, { visited }, { dynalist: false })
  return htmlToJsx(filePath as FilePath, processed)
}

export function toArenaRoot(
  node: ElementContent,
  currentSlug: FullSlug,
  componentData: QuartzComponentProps,
): Root {
  const adjusted = adjustArenaNode(node, currentSlug)
  const root: Root = { type: 'root', children: [adjusted] }
  const visited = new Set<FullSlug>([currentSlug])
  return transcludeFinal(root, componentData, { visited }, { dynalist: false })
}

export function toArenaHeadingJsx(
  filePath: string,
  node: ElementContent,
  currentSlug: FullSlug,
  channelSlug: FullSlug,
  componentData: QuartzComponentProps,
) {
  const adjusted = adjustArenaNode(node, currentSlug, { rewriteFirstAnchorTo: channelSlug })
  const root: Root = { type: 'root', children: [adjusted] }
  const visited = new Set<FullSlug>([currentSlug])
  const processed = transcludeFinal(root, componentData, { visited }, { dynalist: false })
  return htmlToJsx(filePath as FilePath, processed)
}

function stripAnchorsInPlace(el: ElementContent) {
  if (el.type !== 'element') return
  const e = el as Element
  if (e.tagName === 'a') {
    e.tagName = 'span'
    if (e.properties) {
      delete e.properties.href
      delete e.properties.target
      delete e.properties.dataSlug
      delete e.properties.dataNoPopover
      delete e.properties['data-slug']
      delete e.properties['data-no-popover']
    }
  }
  if (e.children) e.children.forEach(stripAnchorsInPlace)
}

export function toArenaHeadingInlineJsx(
  filePath: string,
  node: ElementContent,
  currentSlug: FullSlug,
  channelSlug: FullSlug,
  componentData: QuartzComponentProps,
) {
  const adjusted = adjustArenaNode(node, currentSlug, { rewriteFirstAnchorTo: channelSlug })
  stripAnchorsInPlace(adjusted)
  const root: Root = { type: 'root', children: [adjusted] }
  const visited = new Set<FullSlug>([currentSlug])
  const processed = transcludeFinal(root, componentData, { visited }, { dynalist: false })
  return htmlToJsx(filePath as FilePath, processed)
}

export function fromHtmlStringToArenaJsx(
  filePath: string,
  root: Root,
  currentSlug: FullSlug,
  componentData: QuartzComponentProps,
) {
  // Adjust every child root element
  root.children = root.children.map(ch =>
    ch.type === 'element' ? (adjustArenaNode(ch, currentSlug) as ElementContent) : ch,
  )
  const visited = new Set<FullSlug>([currentSlug])
  const processed = transcludeFinal(root, componentData, { visited }, { dynalist: false })
  return htmlToJsx(filePath as FilePath, processed)
}

function parseUsDate(mdy: string): number | undefined {
  // Accept M/D/YYYY or MM/DD/YYYY
  const m = mdy.trim().match(/^([0-9]{1,2})\/([0-9]{1,2})\/([0-9]{4})$/)
  if (!m) return undefined
  const month = Number(m[1])
  const day = Number(m[2])
  const year = Number(m[3])
  if (month < 1 || month > 12 || day < 1 || day > 31) return undefined
  return Date.UTC(year, month - 1, day)
}

function parseIsoDate(s: string): number | undefined {
  // Accept YYYY-MM-DD or YYYY/MM/DD
  const iso = s.trim().replace(/\//g, '-')
  const m = iso.match(/^([0-9]{4})-([0-9]{2})-([0-9]{2})$/)
  if (!m) return undefined
  const year = Number(m[1])
  const month = Number(m[2])
  const day = Number(m[3])
  if (month < 1 || month > 12 || day < 1 || day > 31) return undefined
  return Date.UTC(year, month - 1, day)
}

export function arenaBlockTimestamp(block: ArenaBlock): number {
  const meta = block.metadata || {}
  const candidate = (meta['accessed'] || meta['accessed_date'] || meta['date'] || '').toString()
  if (!candidate) return -Infinity
  return (
    parseUsDate(candidate) ??
    parseIsoDate(candidate) ??
    // As a last resort, let Date parse; if invalid, NaN -> use -Infinity
    ((): number => {
      const t = Date.parse(candidate)
      return Number.isNaN(t) ? -Infinity : t
    })()
  )
}
