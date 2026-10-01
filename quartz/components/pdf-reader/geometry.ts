import type { PdfMarkTarget, PdfQuad, PdfQuote, PdfRect } from '../../util/pdf-marks'
import { compactQuoteText } from '../arena-feed/quote-highlights'

/** The slice of a PDF.js PageViewport the reader needs; `scale: 1` keeps page rotation and CropBox. */
export interface ViewportLike {
  width: number
  height: number
  convertToPdfPoint(x: number, y: number): number[]
  convertToViewportPoint(x: number, y: number): number[]
}

/** A box in fractions of the page, so painted marks survive zoom and resize without repainting. */
export interface PageBox {
  left: number
  top: number
  width: number
  height: number
}

const QUOTE_CONTEXT = 32
const MAX_QUADS = 64

function rectQuad([x0, y0, x1, y1]: PdfRect): PdfQuad {
  // PDF user space grows upward, so the top edge is y1.
  return [x0, y1, x1, y1, x0, y0, x1, y0]
}

function quadBox(quad: PdfQuad, viewport: ViewportLike): PageBox {
  const xs: number[] = []
  const ys: number[] = []
  for (let index = 0; index < 8; index += 2) {
    const [x, y] = viewport.convertToViewportPoint(quad[index], quad[index + 1])
    xs.push(x)
    ys.push(y)
  }
  const left = Math.min(...xs)
  const top = Math.min(...ys)
  return {
    left: left / viewport.width,
    top: top / viewport.height,
    width: (Math.max(...xs) - left) / viewport.width,
    height: (Math.max(...ys) - top) / viewport.height,
  }
}

export function targetBoxes(target: PdfMarkTarget, viewport: ViewportLike): PageBox[] {
  if (target.type === 'page') return []
  const quads = target.type === 'region' ? [rectQuad(target.rect)] : target.quads
  return quads.map(quad => quadBox(quad, viewport))
}

/** Converts a page-fraction box into PDF user space corners in /QuadPoints order. */
export function boxQuad(box: PageBox, viewport: ViewportLike): PdfQuad {
  const left = box.left * viewport.width
  const right = (box.left + box.width) * viewport.width
  const top = box.top * viewport.height
  const bottom = (box.top + box.height) * viewport.height
  return [
    ...viewport.convertToPdfPoint(left, top),
    ...viewport.convertToPdfPoint(right, top),
    ...viewport.convertToPdfPoint(left, bottom),
    ...viewport.convertToPdfPoint(right, bottom),
  ].map(value => Math.round(value * 100) / 100) as PdfQuad
}

export function boxRect(box: PageBox, viewport: ViewportLike): PdfRect {
  const quad = boxQuad(box, viewport)
  const xs = [quad[0], quad[2], quad[4], quad[6]]
  const ys = [quad[1], quad[3], quad[5], quad[7]]
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
}

function textNodesIn(range: Range): Text[] {
  const root = range.commonAncestorContainer
  if (root instanceof Text) return [root]
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const nodes: Text[] = []
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node instanceof Text && node.data.trim() && range.intersectsNode(node)) nodes.push(node)
  }
  return nodes
}

/**
 * Client rects for the selected glyphs only. `Range.getClientRects()` also returns the border box of
 * every fully selected span, which double-counts lines.
 */
function selectedTextRects(range: Range): DOMRect[] {
  const rects: DOMRect[] = []
  for (const node of textNodesIn(range)) {
    const part = document.createRange()
    part.selectNodeContents(node)
    if (node === range.startContainer) part.setStart(node, range.startOffset)
    if (node === range.endContainer) part.setEnd(node, range.endOffset)
    for (const rect of part.getClientRects()) {
      if (rect.width > 0.5 && rect.height > 0.5) rects.push(rect)
    }
  }
  return rects
}

/** Groups glyph rects into line boxes: same line when they overlap vertically by half the shorter. */
function mergeLines(rects: PageBox[]): PageBox[] {
  const sorted = [...rects].sort((a, b) => a.top - b.top || a.left - b.left)
  const lines: PageBox[][] = []
  for (const rect of sorted) {
    const line = lines.find(candidate => {
      const head = candidate[0]
      const overlap =
        Math.min(head.top + head.height, rect.top + rect.height) - Math.max(head.top, rect.top)
      return overlap >= Math.min(head.height, rect.height) * 0.5
    })
    if (line) line.push(rect)
    else lines.push([rect])
  }

  const boxes: PageBox[] = []
  for (const line of lines) {
    line.sort((a, b) => a.left - b.left)
    let current = { ...line[0] }
    for (const rect of line.slice(1)) {
      const gap = rect.left - (current.left + current.width)
      if (gap <= Math.max(current.height, rect.height) * 0.6) {
        const top = Math.min(current.top, rect.top)
        const bottom = Math.max(current.top + current.height, rect.top + rect.height)
        const right = Math.max(current.left + current.width, rect.left + rect.width)
        current = { left: current.left, top, width: right - current.left, height: bottom - top }
      } else {
        boxes.push(current)
        current = { ...rect }
      }
    }
    boxes.push(current)
  }
  return boxes
}

function compactTail(range: Range, fromEnd: boolean): string {
  const text = compactQuoteText(range.toString())
  return fromEnd ? text.slice(-QUOTE_CONTEXT) : text.slice(0, QUOTE_CONTEXT)
}

export interface PageSelection {
  page: number
  boxes: PageBox[]
  quote: PdfQuote
  /** True when the selection continued past this page and was clipped to it. */
  clipped: boolean
}

/**
 * Resolves a DOM selection inside the reader into one page's line boxes and its text quote. Marks
 * belong to a single page, so a selection that crosses pages keeps its first page.
 */
export function selectionOnPage(range: Range): PageSelection | null {
  const startElement =
    range.startContainer instanceof Element
      ? range.startContainer
      : range.startContainer.parentElement
  const pageElement = startElement?.closest<HTMLElement>('[data-page-number]')
  const textLayer = pageElement?.querySelector<HTMLElement>('.pdf-reader-text')
  if (!pageElement || !textLayer) return null
  const page = Number(pageElement.dataset.pageNumber)

  const clippedRange = range.cloneRange()
  const clipped = !textLayer.contains(range.endContainer)
  if (clipped) clippedRange.setEnd(textLayer, textLayer.childNodes.length)

  const exact = clippedRange.toString().replace(/\s+/g, ' ').trim()
  if (!exact) return null
  if (exact.length > 2048) return null

  const pageRect = pageElement.getBoundingClientRect()
  if (pageRect.width <= 0 || pageRect.height <= 0) return null
  const fractions = selectedTextRects(clippedRange)
    .filter(rect => rect.bottom > pageRect.top && rect.top < pageRect.bottom)
    .map(rect => ({
      left: (rect.left - pageRect.left) / pageRect.width,
      top: (rect.top - pageRect.top) / pageRect.height,
      width: rect.width / pageRect.width,
      height: rect.height / pageRect.height,
    }))
  const boxes = mergeLines(fractions)
  if (boxes.length === 0 || boxes.length > MAX_QUADS) return null

  const before = document.createRange()
  before.setStart(textLayer, 0)
  before.setEnd(clippedRange.startContainer, clippedRange.startOffset)
  const after = document.createRange()
  after.setStart(clippedRange.endContainer, clippedRange.endOffset)
  after.setEnd(textLayer, textLayer.childNodes.length)

  return {
    page,
    boxes,
    quote: { exact, prefix: compactTail(before, true), suffix: compactTail(after, false) },
    clipped,
  }
}

/** Point-in-box hit test in page fractions; returns ids topmost-last so the caller can pick the end. */
export function hitTest(
  boxesById: Map<string, PageBox[]>,
  x: number,
  y: number,
  slop = 0.004,
): string[] {
  const hits: string[] = []
  for (const [id, boxes] of boxesById) {
    if (
      boxes.some(
        box =>
          x >= box.left - slop &&
          x <= box.left + box.width + slop &&
          y >= box.top - slop &&
          y <= box.top + box.height + slop,
      )
    ) {
      hits.push(id)
    }
  }
  return hits
}
