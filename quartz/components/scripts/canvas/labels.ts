import type { NodeData } from './data'

export const EDGE_LABEL_LINE_HEIGHT = 1.2

export type SvgTextMetrics = { prefixWidths: number[]; ellipsisWidth: number }

// Prefix widths are measured once, so refitting at another width needs no text layout.
export function measureSvgText(element: SVGTextElement, text: string): SvgTextMetrics {
  element.textContent = '…'
  const ellipsisWidth = element.getComputedTextLength()
  element.textContent = text
  const prefixWidths = Array.from({ length: text.length + 1 }, (_, i) =>
    i === 0 ? 0 : element.getSubStringLength(0, i),
  )
  return { prefixWidths, ellipsisWidth }
}

export function fitSvgText(text: string, metrics: SvgTextMetrics, maxWidth: number): string {
  const { prefixWidths, ellipsisWidth } = metrics
  if (prefixWidths[text.length] <= maxWidth) return text
  let end = text.length - 1
  while (end > 0 && prefixWidths[end] + ellipsisWidth > maxWidth) end--
  // keep surrogate pairs whole
  if (end > 0 && /[\uDC00-\uDFFF]/.test(text[end])) end--
  return `${text.slice(0, end).trimEnd()}…`
}

// Two lines: break at the last space that fits, then fit the rest from the same prefix widths.
export function wrapSvgText(text: string, metrics: SvgTextMetrics, maxWidth: number): string[] {
  const { prefixWidths, ellipsisWidth } = metrics
  if (prefixWidths[text.length] <= maxWidth) return [text]
  let cut = text.lastIndexOf(' ', text.length)
  while (cut > 0 && prefixWidths[cut] > maxWidth) cut = text.lastIndexOf(' ', cut - 1)
  if (cut <= 0) return [fitSvgText(text, metrics, maxWidth)]
  const base = prefixWidths[cut + 1]
  let end = text.length
  while (end > cut + 1 && prefixWidths[end] - base > maxWidth) end--
  const rest = text.slice(cut + 1, end)
  const clipped = end < text.length
  if (clipped) {
    let e = end
    while (e > cut + 1 && prefixWidths[e] - base + ellipsisWidth > maxWidth) e--
    return [text.slice(0, cut), `${text.slice(cut + 1, e).trimEnd()}…`]
  }
  return [text.slice(0, cut), rest]
}

// Cards show the note name, matching the emitter's displayName, never the vault path.
export function cardName(node: NodeData): string {
  // a text node has no title; its first line stands in at fit zoom
  if (node.type === 'text') return firstLine(node.text)
  return node.displayName || node.file?.split('/').pop()?.replace(/\.md$/, '') || ''
}

function firstLine(text: string | undefined): string {
  const line = (text ?? '').split('\n').find(l => l.trim()) ?? ''
  return line
    .replace(/^#+\s*/, '')
    .replace(/\[\[[^\]|]*\|([^\]]*)\]\]/g, '$1')
    .replace(/\[\[([^\]]*)\]\]/g, '$1')
    .trim()
}
