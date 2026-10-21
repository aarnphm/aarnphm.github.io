import type { CardModel, Fonts, FormatSpec, Palette } from './spec'
import { captureLayer, type CardLayer } from '../press-kit-layers'
import { ASCENT, chars, measure } from './typography'

export const monoFont = (fonts: Fonts, px: number) => `400 ${Math.round(px)}px ${fonts.mono}`

function clip(ctx: CanvasRenderingContext2D, font: string, text: string, maxWidth: number): string {
  if (measure(ctx, font, text) <= maxWidth) return text
  const cs = chars(text)
  while (cs.length > 0 && measure(ctx, font, cs.join('') + '…') > maxWidth) cs.pop()
  return cs.join('').trimEnd() + '…'
}

/** One line of metadata inside the box `[x, x + width]`; returns the drawn width. */
export function label(
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

export function drawRule(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  color: string,
) {
  ctx.fillStyle = color
  ctx.fillRect(x, Math.round(y), w, 2)
}

export interface Frame {
  x: number
  w: number
  top: number
  bottom: number
}

/** Header row and footer shared by every template; returns the free middle. */
export function chrome(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  layers?: CardLayer[],
): Frame {
  const [pt, pr, pb, pl] = spec.pad
  const x = pl
  const w = spec.width - pl - pr
  const meta = spec.meta
  const mono = monoFont(fonts, meta)
  let top = pt

  const headerBaseline = top + meta * ASCENT
  const training = model.kind === 'day' || model.kind === 'activity'
  const showDate = !training && (model.kind === 'entry' ? model.show.date : model.show.meta)
  const dateWidth = model.date && showDate ? measure(ctx, mono, model.date) : 0
  const crumbWidth = dateWidth ? w - dateWidth - meta * 2 : w
  if (!training && model.kind !== 'entry')
    captureLayer(
      ctx,
      layers,
      'breadcrumbs',
      'chemin',
      { x, y: pt - meta * 0.2, width: crumbWidth, height: meta * 1.5 },
      context =>
        drawCrumbs(context, mono, meta, model.crumbs, x, headerBaseline, crumbWidth, palette),
    )
  if (dateWidth)
    captureLayer(
      ctx,
      layers,
      'date',
      'date',
      { x: x + w - dateWidth, y: pt, width: dateWidth, height: meta * 1.3 },
      context => label(context, mono, model.date!, x, headerBaseline, w, palette.muted, 'right'),
    )
  if (dateWidth || (!training && model.kind !== 'entry')) top += meta * 1.3

  const footerBottom = spec.height - pb
  const tagSize = meta * 0.92
  // Align reading time with the text inside the tag pills.
  const baseline = footerBottom - tagSize * (1 - ASCENT + 0.32)
  const showReading =
    !!model.reading &&
    (model.kind === 'entry' ? model.show.metrics !== false : model.show.meta) &&
    model.kind !== 'quote'
  const readingWidth = showReading ? measure(ctx, mono, model.reading!) + meta * 2 : 0
  if (model.show.meta) {
    if (model.tags.length)
      captureLayer(
        ctx,
        layers,
        'tags',
        'étiquettes',
        { x, y: footerBottom - tagSize * 1.7, width: w - readingWidth, height: tagSize * 1.7 },
        context =>
          drawTags(context, fonts, model.tags, x, footerBottom, tagSize, w - readingWidth, palette),
      )
  }
  if (showReading)
    captureLayer(
      ctx,
      layers,
      'reading',
      'temps de lecture',
      { x: x + w - readingWidth, y: baseline - meta, width: readingWidth, height: meta * 1.3 },
      context => label(context, mono, model.reading!, x, baseline, w, palette.muted, 'right'),
    )

  const bottom = baseline - meta - meta * 0.75
  captureLayer(ctx, layers, 'rule', 'séparateur', { x, y: bottom, width: w, height: 2 }, context =>
    drawRule(context, x, bottom, w, palette.rule),
  )
  return { x, w, top: top + spec.gap, bottom: bottom - spec.gap * 0.5 }
}
