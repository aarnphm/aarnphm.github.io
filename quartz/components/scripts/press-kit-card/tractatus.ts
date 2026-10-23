import type { Frame } from './chrome'
import type { CardModel, Fonts, FormatSpec, Palette } from './spec'
import { captureLayer, type CardLayer } from '../press-kit-layers'
import { draw, fit, measure, type Block } from './typography'

export function drawTractatus(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  frame: Frame,
  layers?: CardLayer[],
) {
  const passages = model.propositions ?? []
  const block: Block = { voice: 'body', leading: 1.42, weight: 400 }
  const gap = spec.gap * 0.45
  const attribution = fit(
    ctx,
    fonts,
    model.title,
    { voice: 'header', leading: 1.15 },
    frame.w,
    spec.meta * 3,
    [spec.meta, spec.meta * 1.35],
    2,
  )
  const section =
    model.show.section && model.section
      ? fit(
          ctx,
          fonts,
          [{ text: model.section, style: 'italic' }],
          block,
          frame.w,
          spec.meta * 2,
          [spec.meta, spec.meta],
          1,
        )
      : undefined
  const bottom = frame.bottom - attribution.height - (section ? section.height + gap : 0) - spec.gap
  const available = bottom - frame.top
  const minSize = Math.max(18, Math.round(spec.prose[0] * 0.65))
  // Keep every visible proposition's number attached to its text; dense selections disclose omissions.
  const limit = Math.max(1, Math.floor((available - minSize * 2) / (minSize * 2 + gap)))
  const visible = passages.slice(0, limit)
  const omitted = passages.length - visible.length
  const minDepth = Math.min(...visible.map(passage => passage.depth), 0)
  const layout = (size: number, constrain: boolean) => {
    const mono = `400 ${size}px ${fonts.mono}`
    const numberWidth = Math.max(...visible.map(p => measure(ctx, mono, p.number))) + size * 0.9
    return visible.map(passage => {
      const indent = Math.min(3, Math.max(0, passage.depth - minDepth)) * size * 0.45
      const fitted = fit(
        ctx,
        fonts,
        passage.runs,
        block,
        frame.w - numberWidth - indent,
        constrain
          ? Math.max(
              size * block.leading,
              (available - gap * visible.length - (omitted ? size * 2 : 0)) / visible.length,
            )
          : Infinity,
        [size, size],
      )
      return { passage, fitted, indent, numberWidth }
    })
  }
  let size = spec.prose[1]
  let rows = layout(size, false)
  const height = () =>
    rows.reduce((sum, row) => sum + row.fitted.height + gap, 0) + (omitted ? size * 2 : 0)
  while (size > minSize && height() > available) rows = layout(--size, false)
  if (height() > available) rows = layout(size, true)
  let y = frame.top + Math.max(0, (available - height()) / 2)
  for (const [index, row] of rows.entries()) {
    const { passage, fitted, indent, numberWidth } = row
    const number = fit(
      ctx,
      fonts,
      [{ text: passage.number, style: 'mono' }],
      block,
      numberWidth,
      size * 2,
      [size, size],
      1,
    )
    captureLayer(
      ctx,
      layers,
      `proposition-${index}-${passage.number}`,
      passage.number,
      { x: frame.x, y, width: frame.w, height: Math.max(number.height, fitted.height) },
      context => {
        draw(context, number, frame.x + indent, y, palette.muted)
        draw(
          context,
          fitted,
          frame.x + numberWidth + indent,
          y,
          palette.ink,
          model.show.highlight ? { bg: palette.hl, ink: palette.hlInk } : undefined,
        )
      },
    )
    y += Math.max(number.height, fitted.height) + gap
  }
  if (omitted) {
    const label = model.locale === 'fr' ? 'propositions supplémentaires' : 'more propositions'
    const rest = fit(
      ctx,
      fonts,
      [{ text: `… +${omitted} ${label}`, style: 'mono' }],
      block,
      frame.w,
      size * 2,
      [size, size],
      1,
    )
    draw(ctx, rest, frame.x, y, palette.muted, undefined, 'left', 0, layers, 'propositions-omises')
  }
  let attributionY = bottom + spec.gap
  attributionY += draw(
    ctx,
    attribution,
    frame.x,
    attributionY,
    palette.ink,
    undefined,
    'left',
    0,
    layers,
    'attribution',
  )
  if (section)
    draw(
      ctx,
      section,
      frame.x,
      attributionY + gap,
      palette.muted,
      undefined,
      'left',
      0,
      layers,
      'section',
    )
}
