import type { CardRun, Fonts } from './spec'
import { captureLayer, type CardLayer } from '../press-kit-layers'
import { BREAK, type Run, type RunStyle } from '../press-kit-text'
import { drawMath, type MathImage } from './math'

type Voice = 'header' | 'body' | 'serif'

// Text sits on an alphabetic baseline this far below the top of its em box.
export const ASCENT = 0.8
const MONO_SCALE = 0.86

export interface Block {
  voice: Voice
  leading: number
  /** weight of roman runs */
  weight?: number
  italic?: boolean
}

interface Piece {
  text: string
  font: string
  width: number
  space: boolean
  math?: MathImage
  mathSize?: number
}

interface Line {
  pieces: Piece[]
  width: number
  paragraphEnd: boolean
  center?: boolean
}

export interface Fitted {
  size: number
  lines: Line[]
  height: number
  block: Block
}

const widths = new Map<string, number>()

export function measure(ctx: CanvasRenderingContext2D, font: string, text: string): number {
  const key = `${font}\u0000${text}`
  let width = widths.get(key)
  if (width === undefined) {
    if (widths.size > 20_000) widths.clear()
    ctx.font = font
    width = ctx.measureText(text).width
    widths.set(key, width)
  }
  return width
}

function family(fonts: Fonts, voice: Voice): string {
  return voice === 'header' ? fonts.header : voice === 'serif' ? fonts.serif : fonts.body
}

function fontFor(fonts: Fonts, block: Block, style: RunStyle, size: number): string {
  if (style === 'mono') {
    return `400 ${Math.round(size * MONO_SCALE)}px ${fonts.mono}`
  }
  const base = block.weight ?? (block.voice === 'header' ? fonts.headerWeight : 400)
  const weight = style === 'bold' || style === 'bold-italic' ? Math.max(600, base) : base
  const italic = style === 'italic' || style === 'bold-italic' || !!block.italic
  return `${italic ? 'italic ' : ''}${weight} ${Math.round(size)}px ${family(fonts, block.voice)}`
}

const segmenter =
  typeof Intl !== 'undefined' && 'Segmenter' in Intl
    ? new Intl.Segmenter(undefined, { granularity: 'word' })
    : undefined
const graphemes =
  typeof Intl !== 'undefined' && 'Segmenter' in Intl
    ? new Intl.Segmenter(undefined, { granularity: 'grapheme' })
    : undefined

const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u

/** Break opportunities: whitespace, and between CJK characters. Punctuation stays on its word. */
function words(text: string): string[] {
  if (!segmenter) return text.split(/(\s+)/).filter(Boolean)
  const out: string[] = []
  for (const { segment } of segmenter.segment(text)) {
    const last = out.at(-1)
    const joins =
      last !== undefined &&
      !/\s$/.test(last) &&
      !/^\s/.test(segment) &&
      !(CJK.test(last) && CJK.test(segment))
    if (joins) out[out.length - 1] = last + segment
    else out.push(segment)
  }
  return out
}

export function chars(text: string): string[] {
  return graphemes ? Array.from(graphemes.segment(text), s => s.segment) : Array.from(text)
}

function layout(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  runs: CardRun[],
  block: Block,
  size: number,
  maxWidth: number,
): Line[] {
  const lines: Line[] = []
  let pieces: Piece[] = []
  let width = 0

  const flush = (paragraphEnd: boolean, center = false) => {
    while (pieces.at(-1)?.space) width -= pieces.pop()!.width
    if (pieces.length > 0 || paragraphEnd) lines.push({ pieces, width, paragraphEnd, center })
    pieces = []
    width = 0
  }

  for (const run of runs) {
    if (run.text === BREAK) {
      flush(true)
      continue
    }
    const font = fontFor(fonts, block, run.style, size)
    if (run.math) {
      if (run.display && pieces.length > 0) flush(false)
      const mathSize = Math.min(size, maxWidth / run.math.width)
      const w = run.math.width * mathSize
      if (width + w > maxWidth && pieces.length > 0) flush(false)
      pieces.push({ text: run.text, font, width: w, space: false, math: run.math, mathSize })
      width += w
      if (run.display) flush(false, true)
      continue
    }
    for (const word of words(run.text)) {
      const space = /^\s+$/.test(word)
      if (space) {
        if (pieces.length === 0) continue
        const w = measure(ctx, font, ' ')
        pieces.push({ text: ' ', font, width: w, space: true })
        width += w
        continue
      }
      const w = measure(ctx, font, word)
      if (width + w <= maxWidth) {
        pieces.push({ text: word, font, width: w, space: false })
        width += w
        continue
      }
      if (w <= maxWidth) {
        flush(false)
        pieces.push({ text: word, font, width: w, space: false })
        width = w
        continue
      }
      // A word wider than the line (for example, a URL) breaks between graphemes.
      let chunk = ''
      for (const ch of chars(word)) {
        const next = measure(ctx, font, chunk + ch)
        if (width + next > maxWidth && (chunk || pieces.length > 0)) {
          if (chunk) {
            const cw = measure(ctx, font, chunk)
            pieces.push({ text: chunk, font, width: cw, space: false })
            width += cw
          }
          flush(false)
          chunk = ch
        } else {
          chunk += ch
        }
      }
      if (chunk) {
        const cw = measure(ctx, font, chunk)
        pieces.push({ text: chunk, font, width: cw, space: false })
        width += cw
      }
    }
  }
  if (pieces.length > 0) flush(true)
  const last = lines.at(-1)
  if (last) last.paragraphEnd = false
  return lines
}

function lineMetrics(line: Line, size: number, block: Block) {
  let ascent = size * ASCENT
  let descent = size * (1 - ASCENT)
  for (const piece of line.pieces) {
    if (!piece.math || piece.mathSize === undefined) continue
    ascent = Math.max(ascent, piece.math.ascent * piece.mathSize)
    descent = Math.max(descent, piece.math.descent * piece.mathSize)
  }
  return { ascent, descent, height: Math.max(size * block.leading, ascent + descent) }
}

function heightOf(lines: Line[], size: number, block: Block): number {
  return lines.reduce((h, line) => {
    const { height } = lineMetrics(line, size, block)
    return h + height + (line.paragraphEnd ? size * block.leading * 0.45 : 0)
  }, 0)
}

function ellipsize(ctx: CanvasRenderingContext2D, line: Line, maxWidth: number): Line {
  const pieces = line.pieces.map(piece => ({ ...piece }))
  const font = pieces.at(-1)?.font
  if (!font) return line
  const dots = measure(ctx, font, '…')
  let width = pieces.reduce((w, piece) => w + piece.width, 0)
  while (pieces.length > 0 && (width + dots > maxWidth || pieces.at(-1)!.space)) {
    const last = pieces.at(-1)!
    if (last.math || last.space || last.text.length <= 1) {
      width -= last.width
      pieces.pop()
      continue
    }
    const trimmed = chars(last.text).slice(0, -1).join('')
    width -= last.width
    last.text = trimmed
    last.width = measure(ctx, last.font, trimmed)
    width += last.width
  }
  const end = pieces.at(-1)
  if (end && !end.math) {
    end.text = end.text.replace(/[\s.,;:!?·—–-]+$/, '')
    end.width = measure(ctx, end.font, end.text)
  }
  width = pieces.reduce((w, piece) => w + piece.width, 0)
  pieces.push({ text: '…', font: end?.font ?? font, width: dots, space: false })
  return { pieces, width: width + dots, paragraphEnd: false }
}

export function fit(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  runs: CardRun[],
  block: Block,
  maxWidth: number,
  maxHeight: number,
  [min, max]: [number, number],
  maxLines = Infinity,
): Fitted {
  if (runs.length === 0 || maxHeight <= 0) return { size: min, lines: [], height: 0, block }
  const step = Math.max(1, Math.round((max - min) / 28))
  for (let size = max; size >= min; size -= step) {
    const lines = layout(ctx, fonts, runs, block, size, maxWidth)
    const height = heightOf(lines, size, block)
    if (lines.length <= maxLines && height <= maxHeight) return { size, lines, height, block }
  }
  const lines = layout(ctx, fonts, runs, block, min, maxWidth)
  let kept = 0
  let height = 0
  const lh = min * block.leading
  for (const line of lines) {
    const next = height + lineMetrics(line, min, block).height
    if (kept >= maxLines || next > maxHeight) break
    height = next + (line.paragraphEnd ? lh * 0.45 : 0)
    kept++
  }
  if (kept === 0) return { size: min, lines: [], height: 0, block }
  const out = lines.slice(0, kept)
  out[kept - 1] = ellipsize(ctx, { ...out[kept - 1], paragraphEnd: false }, maxWidth)
  return { size: min, lines: out, height: heightOf(out, min, block), block }
}

export function draw(
  ctx: CanvasRenderingContext2D,
  fitted: Fitted,
  x: number,
  y: number,
  color: string,
  highlight?: { bg: string; ink: string },
  align: 'left' | 'right' = 'left',
  maxWidth = 0,
  layers?: CardLayer[],
  id = 'text',
): number {
  if (layers && fitted.lines.length) {
    const inset = highlight ? fitted.size * 0.12 : 0
    return captureLayer(
      ctx,
      layers,
      id,
      id,
      {
        x: x - inset,
        y,
        width: Math.max(maxWidth, ...fitted.lines.map(line => line.width)) + inset * 2,
        height: fitted.height,
      },
      context => draw(context, fitted, x, y, color, highlight, align, maxWidth),
    )
  }
  const { size, lines, block } = fitted
  const lh = size * block.leading
  let top = y
  ctx.textBaseline = 'alphabetic'
  for (const line of lines) {
    const metrics = lineMetrics(line, size, block)
    const baseline = top + (metrics.height - metrics.ascent - metrics.descent) / 2 + metrics.ascent
    let cx = align === 'right' ? x + maxWidth - line.width : x
    if (line.center && maxWidth > 0) cx = x + (maxWidth - line.width) / 2
    if (highlight && line.width > 0) {
      const padX = size * 0.12
      ctx.fillStyle = highlight.bg
      ctx.fillRect(cx - padX, top, line.width + padX * 2, metrics.height)
    }
    ctx.fillStyle = highlight ? highlight.ink : color
    for (const piece of line.pieces) {
      if (piece.math && piece.mathSize !== undefined) {
        drawMath(ctx, piece.math, cx, baseline, piece.mathSize, highlight ? highlight.ink : color)
      } else {
        ctx.font = piece.font
        ctx.fillText(piece.text, cx, baseline)
      }
      cx += piece.width
    }
    top += metrics.height + (line.paragraphEnd ? lh * 0.45 : 0)
  }
  return top - y
}

function plainRun(text: string, style: RunStyle = 'roman'): Run[] {
  return text ? [{ text, style }] : []
}

export function oneLine(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  text: string,
  block: Block,
  size: number,
  maxWidth: number,
  style: RunStyle = 'roman',
): Fitted {
  return fit(
    ctx,
    fonts,
    plainRun(text, style),
    block,
    maxWidth,
    size * block.leading,
    [size, size],
    1,
  )
}

/** Every face the card can draw with, so callers can wait on `document.fonts` before drawing. */
export function cardFaces(fonts: Fonts, size = 48): string[] {
  const out = new Set<string>()
  const blocks: Block[] = [
    { voice: 'header', leading: 1 },
    { voice: 'body', leading: 1, italic: true },
    { voice: 'body', leading: 1, weight: 500 },
    { voice: 'body', leading: 1, weight: 600 },
    { voice: 'body', leading: 1 },
    { voice: 'serif', leading: 1 },
  ]
  for (const block of blocks) {
    for (const style of ['roman', 'italic', 'bold', 'bold-italic', 'mono'] satisfies RunStyle[]) {
      out.add(fontFor(fonts, block, style, size))
    }
  }
  return [...out]
}
