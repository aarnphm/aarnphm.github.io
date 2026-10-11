import { drawMath, mathKey, prepareMath, type MathImage } from './press-kit-math'
import { BREAK, type Run, type RunStyle } from './press-kit-text'

type CardRun = Run & { math?: MathImage }

export type FormatKey = 'story' | 'portrait' | 'square' | 'landscape'
export type ThemeKey = 'papier' | 'encre' | 'rose' | 'sauge'
export type CardKind = 'note' | 'quote' | 'entry'
type Anchor = 'start' | 'center' | 'end'

interface FormatSpec {
  label: string
  width: number
  height: number
  /** top, right, bottom, left. Story keeps clear of the platform's header and reply bar. */
  pad: [number, number, number, number]
  meta: number
  title: [number, number]
  titleLines: number
  prose: [number, number]
  quote: [number, number]
  gap: number
}

export const FORMATS: Record<FormatKey, FormatSpec> = {
  story: {
    label: 'story',
    width: 1080,
    height: 1920,
    pad: [220, 96, 220, 96],
    meta: 28,
    title: [64, 136],
    titleLines: 7,
    prose: [34, 50],
    quote: [44, 96],
    gap: 48,
  },
  portrait: {
    label: 'portrait',
    width: 1080,
    height: 1350,
    pad: [96, 88, 88, 88],
    meta: 26,
    title: [56, 116],
    titleLines: 6,
    prose: [30, 44],
    quote: [40, 84],
    gap: 40,
  },
  square: {
    label: 'carré',
    width: 1080,
    height: 1080,
    pad: [80, 80, 72, 80],
    meta: 24,
    title: [48, 100],
    titleLines: 5,
    prose: [28, 40],
    quote: [36, 72],
    gap: 34,
  },
  landscape: {
    label: 'paysage',
    width: 1200,
    height: 630,
    pad: [52, 64, 48, 64],
    meta: 20,
    title: [40, 76],
    titleLines: 3,
    prose: [22, 30],
    quote: [28, 52],
    gap: 22,
  },
}

export interface SiteColors {
  light: string
  lightgray: string
  gray: string
  darkgray: string
  dark: string
  secondary: string
}

export interface Palette {
  label: string
  bg: string
  ink: string
  muted: string
  rule: string
  /** quotation mark */
  mark: string
  /** highlighter behind text, and the ink drawn on it */
  hl: string
  hlInk: string
}

const ROSE = '#fdb2a2'
const SAGE = '#cdd597'

function rgb(color: string): [number, number, number] {
  const hex = color.match(/^#([0-9a-f]{6})$/i)
  if (hex) {
    const n = Number.parseInt(hex[1], 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
  }
  const parts = color.match(/\d+(\.\d+)?/g)?.map(Number) ?? [0, 0, 0]
  return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0]
}

const alpha = (color: string, a: number) => `rgba(${rgb(color).join(', ')}, ${a})`

export function palettes(light: SiteColors, dark: SiteColors): Record<ThemeKey, Palette> {
  const onColor = (label: string, bg: string): Palette => ({
    label,
    bg,
    ink: light.dark,
    muted: alpha(light.dark, 0.66),
    rule: alpha(light.dark, 0.28),
    mark: light.dark,
    hl: light.light,
    hlInk: light.dark,
  })
  return {
    papier: {
      label: 'papier',
      bg: light.light,
      ink: light.dark,
      muted: light.darkgray,
      rule: alpha(light.dark, 0.2),
      mark: ROSE,
      hl: light.secondary,
      hlInk: light.dark,
    },
    encre: {
      label: 'encre',
      bg: dark.light,
      ink: dark.dark,
      muted: dark.darkgray,
      rule: alpha(dark.dark, 0.22),
      mark: SAGE,
      hl: ROSE,
      hlInk: dark.light,
    },
    rose: onColor('rose', ROSE),
    sauge: onColor('sauge', SAGE),
  }
}

export interface Fonts {
  header: string
  headerWeight: number
  body: string
  serif: string
  mono: string
}

type Voice = 'header' | 'body' | 'serif'

export interface CardModel {
  kind: CardKind
  crumbs: string[]
  title: CardRun[]
  description: CardRun[]
  body: CardRun[]
  date?: string
  reading?: string
  tags: string[]
  section?: string
  author: string
  avatar?: CanvasImageSource
  /** the site's base social card, or its night twin on encre (`/static/og-image{,-dark}.webp`) */
  art?: HTMLImageElement
  /** the point crops keep in view, in the 1200×630 card's pixels */
  artFocus?: [number, number]
  show: {
    art: boolean
    description: boolean
    excerpt: boolean
    meta: boolean
    highlight: boolean
    section: boolean
  }
}

// Text sits on an alphabetic baseline this far below the top of its em box.
const ASCENT = 0.8
const MONO_SCALE = 0.86

interface Block {
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

interface Fitted {
  size: number
  lines: Line[]
  height: number
  block: Block
}

const widths = new Map<string, number>()

function measure(ctx: CanvasRenderingContext2D, font: string, text: string): number {
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

function chars(text: string): string[] {
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

function fit(
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

function draw(
  ctx: CanvasRenderingContext2D,
  fitted: Fitted,
  x: number,
  y: number,
  color: string,
  highlight?: { bg: string; ink: string },
  align: 'left' | 'right' = 'left',
  maxWidth = 0,
): number {
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

function oneLine(
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

const monoFont = (fonts: Fonts, px: number) => `400 ${Math.round(px)}px ${fonts.mono}`

function clip(ctx: CanvasRenderingContext2D, font: string, text: string, maxWidth: number): string {
  if (measure(ctx, font, text) <= maxWidth) return text
  const cs = chars(text)
  while (cs.length > 0 && measure(ctx, font, cs.join('') + '…') > maxWidth) cs.pop()
  return cs.join('').trimEnd() + '…'
}

/** One line of metadata inside the box `[x, x + width]`; returns the drawn width. */
function label(
  ctx: CanvasRenderingContext2D,
  font: string,
  text: string,
  x: number,
  baseline: number,
  width: number,
  color: string,
  align: 'left' | 'right' = 'left',
): number {
  const shown = clip(ctx, font, text, width)
  const w = measure(ctx, font, shown)
  ctx.font = font
  ctx.fillStyle = color
  ctx.textBaseline = 'alphabetic'
  ctx.fillText(shown, align === 'right' ? x + width - w : x, baseline)
  return w
}

/** `site / a / b / c`, eliding middle segments until it fits; the last segment is highlighted. */
function drawCrumbs(
  ctx: CanvasRenderingContext2D,
  font: string,
  size: number,
  crumbs: string[],
  x: number,
  baseline: number,
  maxWidth: number,
  palette: Palette,
) {
  if (crumbs.length === 0) return
  const sep = ' / '
  let parts = crumbs
  for (let keep = crumbs.length - 2; keep >= 1; keep--) {
    if (measure(ctx, font, parts.join(sep)) <= maxWidth) break
    parts = [crumbs[0], '…', ...crumbs.slice(-keep)]
  }
  let cx = x
  parts.forEach((part, index) => {
    const last = index === parts.length - 1
    if (index > 0) cx += label(ctx, font, sep, cx, baseline, Infinity, palette.muted)
    if (!last || parts.length === 1) {
      cx += label(ctx, font, part, cx, baseline, x + maxWidth - cx, palette.muted)
      return
    }
    const padX = size * 0.28
    const shown = clip(ctx, font, part, x + maxWidth - cx - padX)
    const w = measure(ctx, font, shown)
    ctx.fillStyle = palette.hl
    ctx.fillRect(cx - padX, baseline - size * 0.92, w + padX * 2, size * 1.24)
    label(ctx, font, shown, cx, baseline, Infinity, palette.hlInk)
  })
}

/** One row of boxed `#tag` pills, as on the page. */
function drawTags(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  tags: string[],
  x: number,
  bottom: number,
  size: number,
  maxWidth: number,
  palette: Palette,
): void {
  if (tags.length === 0) return
  const font = monoFont(fonts, size)
  const padX = size * 0.45
  const padY = size * 0.32
  const gap = size * 0.5
  const h = size + padY * 2
  let cx = x
  const placed: { text: string; x: number; w: number }[] = []
  for (const tag of tags) {
    const text = `#${tag}`
    const w = measure(ctx, font, text) + padX * 2
    if (cx + w > x + maxWidth) break
    placed.push({ text, x: cx, w })
    cx += w + gap
  }
  if (placed.length === 0) return
  const top = bottom - h
  ctx.lineWidth = Math.max(1.5, size / 14)
  ctx.strokeStyle = palette.muted
  ctx.font = font
  ctx.fillStyle = palette.muted
  ctx.textBaseline = 'alphabetic'
  for (const pill of placed) {
    const inset = ctx.lineWidth / 2
    ctx.strokeRect(pill.x + inset, top + inset, pill.w - inset * 2, h - inset * 2)
    ctx.fillText(pill.text, pill.x + padX, top + padY + size * ASCENT)
  }
}

function drawRule(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, color: string) {
  ctx.fillStyle = color
  ctx.fillRect(x, Math.round(y), w, 2)
}

function place(anchor: Anchor, top: number, bottom: number, height: number): number {
  if (anchor === 'start') return top
  if (anchor === 'end') return Math.max(top, bottom - height)
  return Math.max(top, top + (bottom - top - height) / 2)
}

interface Frame {
  x: number
  w: number
  top: number
  bottom: number
}

/** Header row and footer shared by every template; returns the free middle. */
function chrome(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  header: 'crumbs' | 'author',
): Frame {
  const [pt, pr, pb, pl] = spec.pad
  const x = pl
  const w = spec.width - pl - pr
  const meta = spec.meta
  const mono = monoFont(fonts, meta)
  let top = pt

  if (header === 'author') {
    const avatar = Math.round(meta * 2.6)
    if (model.avatar) {
      ctx.save()
      ctx.beginPath()
      ctx.arc(x + avatar / 2, top + avatar / 2, avatar / 2, 0, Math.PI * 2)
      ctx.clip()
      ctx.drawImage(model.avatar, x, top, avatar, avatar)
      ctx.restore()
    }
    const textX = model.avatar ? x + avatar + meta * 0.8 : x
    const nameFont = `500 ${Math.round(meta * 1.25)}px ${fonts.body}`
    label(ctx, nameFont, model.author, textX, top + avatar / 2 - meta * 0.18, w, palette.ink)
    if (model.date) {
      label(ctx, mono, model.date, textX, top + avatar / 2 + meta * 1.08, w, palette.muted)
    }
    top += avatar
  } else {
    const baseline = top + meta * ASCENT
    const dateWidth = model.date && model.show.meta ? measure(ctx, mono, model.date) : 0
    const crumbWidth = dateWidth ? w - dateWidth - meta * 2 : w
    drawCrumbs(ctx, mono, meta, model.crumbs, x, baseline, crumbWidth, palette)
    if (dateWidth) label(ctx, mono, model.date!, x, baseline, w, palette.muted, 'right')
    top += meta * 1.3
  }

  const footerBottom = spec.height - pb
  const tagSize = meta * 0.92
  // Align reading time with the text inside the tag pills.
  const baseline = footerBottom - tagSize * (1 - ASCENT + 0.32)
  const showReading = !!model.reading && model.show.meta && model.kind !== 'quote'
  const readingWidth = showReading ? measure(ctx, mono, model.reading!) + meta * 2 : 0
  if (model.show.meta) {
    drawTags(ctx, fonts, model.tags, x, footerBottom, tagSize, w - readingWidth, palette)
  }
  if (showReading) label(ctx, mono, model.reading!, x, baseline, w, palette.muted, 'right')

  const bottom = baseline - meta - meta * 0.75
  drawRule(ctx, x, bottom, w, palette.rule)
  return { x, w, top: top + spec.gap, bottom: bottom - spec.gap * 0.5 }
}

// The mascot on the ridge, in the base card's own pixels; crops keep it in view.
const ART_FOCUS: [number, number] = [640, 360]

/** Cover-crops `art` into the box around `focus`. */
function drawCover(
  ctx: CanvasRenderingContext2D,
  art: HTMLImageElement,
  focus: [number, number],
  x: number,
  y: number,
  w: number,
  h: number,
) {
  const aw = art.naturalWidth
  const ah = art.naturalHeight
  const scale = Math.max(w / aw, h / ah)
  const sw = w / scale
  const sh = h / scale
  const sx = Math.min(Math.max(focus[0] * (aw / 1200) - sw / 2, 0), aw - sw)
  const sy = Math.min(Math.max(focus[1] * (ah / 630) - sh / 2, 0), ah - sh)
  ctx.drawImage(art, sx, sy, sw, sh, x, y, w, h)
}

/**
 * Lays the base card into the free middle and returns what is left for text. Paysage takes a
 * right-hand column; the taller formats take a band above the text. Papier and encre share their
 * art's ground (the day card's paper, the night card's sky), so the band runs edge to edge; other
 * grounds get it as a ruled print.
 */
function drawArt(
  ctx: CanvasRenderingContext2D,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  frame: Frame,
): Frame {
  const art = model.art
  if (!model.show.art || !art?.naturalWidth) return frame
  const avail = frame.bottom - frame.top
  const bleed = palette.label === 'papier' || palette.label === 'encre'
  // Without a frame the crop would cut the ridges in straight lines, so cut edges fade to paper.
  const fade = (x: number, y: number, w: number, h: number, edge: 'left' | 'bottom') => {
    const gradient =
      edge === 'left'
        ? ctx.createLinearGradient(x, 0, x + w, 0)
        : ctx.createLinearGradient(0, y + h, 0, y)
    gradient.addColorStop(0, palette.bg)
    gradient.addColorStop(1, alpha(palette.bg, 0))
    ctx.fillStyle = gradient
    ctx.fillRect(x, y, w, h)
  }
  const print = (x: number, y: number, w: number, h: number, cut: ('left' | 'bottom')[]) => {
    drawCover(ctx, art, model.artFocus ?? ART_FOCUS, x, y, w, h)
    if (bleed) {
      const band = Math.round(h * 0.28)
      if (cut.includes('bottom')) fade(x, y + h - band, w, band, 'bottom')
      if (cut.includes('left')) fade(x, y, band, h, 'left')
      return
    }
    ctx.strokeStyle = palette.rule
    ctx.lineWidth = 2
    ctx.strokeRect(x + 1, y + 1, w - 2, h - 2)
  }
  if (spec === FORMATS.landscape) {
    const w = Math.round(frame.w * 0.4)
    print(frame.x + frame.w - w, frame.top, w, avail, ['left', 'bottom'])
    return { ...frame, w: frame.w - w - spec.gap * 1.5 }
  }
  const h = Math.round(Math.min(frame.w / 2.2, avail * 0.4))
  if (bleed) print(0, frame.top, spec.width, h, ['bottom'])
  else print(frame.x, frame.top, frame.w, h, [])
  return { ...frame, top: frame.top + h + spec.gap }
}

function drawNote(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
) {
  const frame = drawArt(
    ctx,
    model,
    spec,
    palette,
    chrome(ctx, fonts, model, spec, palette, 'crumbs'),
  )
  const avail = frame.bottom - frame.top
  const gap = spec.gap * 0.7
  const hasDesc = model.show.description && model.description.length > 0
  const hasBody = model.show.excerpt && model.body.length > 0

  const titleBlock: Block = { voice: 'header', leading: 1.04 }
  const title = fit(
    ctx,
    fonts,
    model.title,
    titleBlock,
    frame.w,
    avail * (hasDesc || hasBody ? 0.58 : 1),
    spec.title,
    spec.titleLines,
  )
  let rest = avail - title.height - gap

  const descBlock: Block = { voice: 'body', leading: 1.3, italic: true, weight: 400 }
  const desc = hasDesc
    ? fit(
        ctx,
        fonts,
        model.description,
        descBlock,
        frame.w * 0.94,
        rest * (hasBody ? 0.45 : 1),
        spec.prose,
        5,
      )
    : undefined
  if (desc && desc.lines.length > 0) rest -= desc.height + gap

  const bodyBlock: Block = { voice: 'body', leading: 1.42, weight: 400 }
  const proseMin = Math.round(spec.prose[0] * 0.86)
  const proseMax = Math.round(spec.prose[1] * 0.8)
  let body = hasBody
    ? fit(ctx, fonts, model.body, bodyBlock, frame.w * 0.94, rest, [
        proseMin,
        Math.max(proseMin, proseMax),
      ])
    : undefined
  if (body && body.lines.length < 2) body = undefined

  const parts = [title, desc, body].filter(
    (part): part is Fitted => !!part && part.lines.length > 0,
  )
  const total = parts.reduce((h, part) => h + part.height, 0) + gap * (parts.length - 1)
  // Paysage has no room for a low-slung title, so the block centres between header and footer.
  const anchor: Anchor = spec === FORMATS.landscape ? 'center' : 'end'
  let y = place(anchor, frame.top, frame.bottom, total)

  y += draw(ctx, title, frame.x, y, palette.ink) + gap
  if (desc && desc.lines.length > 0) y += draw(ctx, desc, frame.x, y, palette.ink) + gap
  if (body) draw(ctx, body, frame.x, y, palette.muted)
}

function drawQuote(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
) {
  const frame = drawArt(
    ctx,
    model,
    spec,
    palette,
    chrome(ctx, fonts, model, spec, palette, 'crumbs'),
  )
  const avail = frame.bottom - frame.top
  const gap = spec.gap * 0.8

  const attrSize = spec.meta * 1.35
  const attrBlock: Block = { voice: 'header', leading: 1.15 }
  const attribution = fit(
    ctx,
    fonts,
    [{ text: '— ', style: 'roman' }, ...model.title],
    attrBlock,
    frame.w,
    attrSize * 1.15 * 2,
    [attrSize * 0.8, attrSize],
    2,
  )
  const sectionBlock: Block = { voice: 'body', leading: 1.3, italic: true }
  const section =
    model.show.section && model.section
      ? oneLine(ctx, fonts, model.section, sectionBlock, spec.meta * 1.05, frame.w)
      : undefined
  const attrH = attribution.height + (section ? section.height + spec.meta * 0.3 : 0)

  const markSize = spec.quote[1] * 1.5
  const markH = markSize * 0.62
  const quoteBlock: Block = { voice: 'serif', leading: 1.28, weight: 400 }
  const quote = fit(
    ctx,
    fonts,
    model.body,
    quoteBlock,
    frame.w - (model.show.highlight ? spec.quote[0] * 0.3 : 0),
    avail - attrH - markH - gap * 2,
    spec.quote,
  )

  const total = markH + quote.height + gap + attrH + gap * 0.4
  let y = place('center', frame.top, frame.bottom, total)

  ctx.font = `400 ${Math.round(markSize)}px ${fonts.serif}`
  ctx.fillStyle = palette.mark
  ctx.textBaseline = 'alphabetic'
  ctx.fillText('“', frame.x - markSize * 0.04, y + markSize * 0.74)
  y += markH + gap * 0.4

  const inset = model.show.highlight ? quote.size * 0.12 : 0
  y +=
    draw(
      ctx,
      quote,
      frame.x + inset,
      y,
      palette.ink,
      model.show.highlight ? { bg: palette.hl, ink: palette.hlInk } : undefined,
      'left',
      frame.w - inset * 2,
    ) + gap
  y += draw(ctx, attribution, frame.x, y, palette.ink)
  if (section) draw(ctx, section, frame.x, y + spec.meta * 0.3, palette.muted)
}

function drawEntry(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
) {
  const frame = drawArt(
    ctx,
    model,
    spec,
    palette,
    chrome(ctx, fonts, model, spec, palette, 'author'),
  )
  const avail = frame.bottom - frame.top
  const gap = spec.gap * 0.5

  const titleBlock: Block = { voice: 'body', leading: 1.18, weight: 600 }
  const title = fit(
    ctx,
    fonts,
    model.title,
    titleBlock,
    frame.w,
    avail * 0.3,
    [spec.prose[0] * 1.1, spec.prose[1] * 1.3],
    3,
  )
  const bodyBlock: Block = { voice: 'body', leading: 1.42, weight: 400 }
  const body = fit(
    ctx,
    fonts,
    model.body,
    bodyBlock,
    frame.w,
    avail - title.height - (title.lines.length ? gap : 0),
    [spec.prose[0] * 0.9, spec.prose[1] * 1.1],
  )
  const total = title.height + (title.lines.length ? gap : 0) + body.height
  const anchor: Anchor = spec === FORMATS.story ? 'center' : 'start'
  let y = place(anchor, frame.top, frame.bottom, total)
  if (title.lines.length) y += draw(ctx, title, frame.x, y, palette.ink) + gap
  draw(ctx, body, frame.x, y, palette.ink)
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

export async function prepareCard(model: CardModel): Promise<CardModel> {
  const images = await prepareMath([...model.title, ...model.description, ...model.body])
  const prepare = (runs: Run[]): CardRun[] =>
    runs.map(run => (run.style === 'math' ? { ...run, math: images.get(mathKey(run)) } : run))
  return {
    ...model,
    title: prepare(model.title),
    description: prepare(model.description),
    body: prepare(model.body),
  }
}

export function renderCard(
  canvas: HTMLCanvasElement,
  model: CardModel,
  format: FormatKey,
  palette: Palette,
  fonts: Fonts,
) {
  const spec = FORMATS[format]
  if (canvas.width !== spec.width) canvas.width = spec.width
  if (canvas.height !== spec.height) canvas.height = spec.height
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.fillStyle = palette.bg
  ctx.fillRect(0, 0, spec.width, spec.height)
  if (model.kind === 'quote') drawQuote(ctx, fonts, model, spec, palette)
  else if (model.kind === 'entry') drawEntry(ctx, fonts, model, spec, palette)
  else drawNote(ctx, fonts, model, spec, palette)
}
