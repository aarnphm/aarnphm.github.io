import { captureLayer, type CardLayer } from '../press-kit-layers'
import { chrome, type Frame } from './chrome'
import { alpha, FORMATS, type CardModel, type Fonts, type FormatSpec, type Palette } from './spec'
import { draw, fit, oneLine, type Block, type Fitted } from './typography'

type Anchor = 'start' | 'center' | 'end'

function place(anchor: Anchor, top: number, bottom: number, height: number): number {
  if (anchor === 'start') return top
  if (anchor === 'end') return Math.max(top, bottom - height)
  return Math.max(top, top + (bottom - top - height) / 2)
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
  layers?: CardLayer[],
): Frame {
  const art = model.art
  if (!model.show.art || !art?.naturalWidth) return frame
  const avail = frame.bottom - frame.top
  const bleed = palette.label === 'papier' || palette.label === 'encre'
  // Without a frame the crop would cut the ridges in straight lines, so cut edges fade to paper.
  const fade = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    edge: 'left' | 'bottom',
  ) => {
    const gradient =
      edge === 'left'
        ? ctx.createLinearGradient(x, 0, x + w, 0)
        : ctx.createLinearGradient(0, y + h, 0, y)
    gradient.addColorStop(0, palette.bg)
    gradient.addColorStop(1, alpha(palette.bg, 0))
    ctx.fillStyle = gradient
    ctx.fillRect(x, y, w, h)
  }
  const print = (x: number, y: number, w: number, h: number, cut: ('left' | 'bottom')[]) =>
    captureLayer(ctx, layers, 'art', 'illustration', { x, y, width: w, height: h }, ctx => {
      drawCover(ctx, art, model.artFocus ?? ART_FOCUS, x, y, w, h)
      if (bleed) {
        const band = Math.round(h * 0.28)
        if (cut.includes('bottom')) fade(ctx, x, y + h - band, w, band, 'bottom')
        if (cut.includes('left')) fade(ctx, x, y, band, h, 'left')
        return
      }
      ctx.strokeStyle = palette.rule
      ctx.lineWidth = 2
      ctx.strokeRect(x + 1, y + 1, w - 2, h - 2)
    })
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

export function drawNote(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  layers?: CardLayer[],
) {
  const frame = drawArt(
    ctx,
    model,
    spec,
    palette,
    chrome(ctx, fonts, model, spec, palette, layers),
    layers,
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

  y += draw(ctx, title, frame.x, y, palette.ink, undefined, 'left', 0, layers, 'titre') + gap
  if (desc && desc.lines.length > 0)
    y += draw(ctx, desc, frame.x, y, palette.ink, undefined, 'left', 0, layers, 'description') + gap
  if (body) draw(ctx, body, frame.x, y, palette.muted, undefined, 'left', 0, layers, 'texte')
}

export function drawQuote(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  layers?: CardLayer[],
) {
  const frame = drawArt(
    ctx,
    model,
    spec,
    palette,
    chrome(ctx, fonts, model, spec, palette, layers),
    layers,
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

  captureLayer(
    ctx,
    layers,
    'quote-mark',
    'guillemet',
    { x: frame.x - markSize * 0.04, y, width: markSize, height: markSize },
    context => {
      context.font = `400 ${Math.round(markSize)}px ${fonts.serif}`
      context.fillStyle = palette.mark
      context.textBaseline = 'alphabetic'
      context.fillText('“', frame.x - markSize * 0.04, y + markSize * 0.74)
    },
  )
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
      layers,
      'citation',
    ) + gap
  y += draw(ctx, attribution, frame.x, y, palette.ink, undefined, 'left', 0, layers, 'attribution')
  if (section)
    draw(
      ctx,
      section,
      frame.x,
      y + spec.meta * 0.3,
      palette.muted,
      undefined,
      'left',
      0,
      layers,
      'section',
    )
}

export function drawEntry(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  layers?: CardLayer[],
) {
  const frame = drawArt(
    ctx,
    model,
    spec,
    palette,
    chrome(ctx, fonts, model, spec, palette, layers),
    layers,
  )
  const avail = frame.bottom - frame.top
  const gap = spec.gap * 0.7

  // An entry sets its title and text as a note does; the body stays in ink, since it is the post.
  const titleBlock: Block = { voice: 'body', weight: 500, leading: 1.04 }
  const hasText = model.body.length > 0 || model.description.length > 0
  const title = fit(
    ctx,
    fonts,
    model.show.title ? model.title : [],
    titleBlock,
    frame.w,
    avail * (hasText ? 0.4 : 1),
    spec.title,
    Math.min(spec.titleLines, 3),
  )
  let rest = avail - title.height - (title.lines.length ? gap : 0)
  // An entry has no description toggle, so its description always shows.
  const descBlock: Block = { voice: 'body', leading: 1.3, italic: true, weight: 400 }
  const desc =
    model.description.length > 0
      ? fit(
          ctx,
          fonts,
          model.description,
          descBlock,
          frame.w * 0.94,
          rest * (model.body.length > 0 ? 0.35 : 1),
          spec.prose,
          3,
        )
      : undefined
  if (desc && desc.lines.length > 0) rest -= desc.height + gap
  const bodyBlock: Block = { voice: 'body', leading: 1.42, weight: 400 }
  const body = fit(ctx, fonts, model.body, bodyBlock, frame.w * 0.94, rest, [
    spec.prose[0] * 0.9,
    spec.prose[1] * 1.1,
  ])
  const parts = [title, desc, body].filter(
    (part): part is Fitted => !!part && part.lines.length > 0,
  )
  const total = parts.reduce((h, part) => h + part.height, 0) + gap * (parts.length - 1)
  const anchor: Anchor = spec === FORMATS.landscape ? 'center' : 'end'
  let y = place(anchor, frame.top, frame.bottom, total)
  for (const part of parts)
    y +=
      draw(
        ctx,
        part,
        frame.x,
        y,
        palette.ink,
        undefined,
        'left',
        0,
        layers,
        part === title ? 'titre' : part === desc ? 'description' : 'texte',
      ) + gap
}
