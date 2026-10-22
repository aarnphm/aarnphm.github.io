import type { Run } from './press-kit-text'
import { drawCalendar } from './press-kit-card/calendar'
import { chrome } from './press-kit-card/chrome'
import { drawEntry, drawNote, drawQuote } from './press-kit-card/editorial'
import { mathKey, prepareMath } from './press-kit-card/math'
import {
  cardResolution,
  FORMATS,
  type CardModel,
  type CardRun,
  type Fonts,
  type FormatKey,
  type Palette,
} from './press-kit-card/spec'
import { svgImage } from './press-kit-card/svg'
import { trainingCardSvg } from './press-kit-card/training-card'
import { svgLayers, type CardLayer, type CardScene, type LayerRenderer } from './press-kit-layers'
export { FORMATS, cardResolution, metricShown, palettes } from './press-kit-card/spec'
export type {
  CardModel,
  Fonts,
  FormatKey,
  Palette,
  SiteColors,
  ThemeKey,
} from './press-kit-card/spec'
export { cardFaces } from './press-kit-card/typography'
export type { Frame } from './press-kit-card/chrome'

export async function prepareCard(
  model: CardModel,
  palette: Palette,
  format: FormatKey,
  fonts: Fonts,
  editable = false,
): Promise<CardModel> {
  if (model.kind === 'activity' || model.kind === 'day') {
    const context = document.createElement('canvas').getContext('2d')
    if (!context) throw new Error('Canvas is unavailable')
    const frame = chrome(context, fonts, model, FORMATS[format], palette)
    const renderers = editable ? new Map<string, LayerRenderer>() : undefined
    const svg = trainingCardSvg(model, format, palette, fonts, frame, renderers)
    return editable
      ? { ...model, trainingSvg: svg, trainingLayers: renderers }
      : { ...model, training: await svgImage(svg) }
  }
  const spec = FORMATS[format]
  const pixelsPerEm = Math.ceil(
    Math.max(...spec.title, ...spec.prose, ...spec.quote) * cardResolution(format).scale,
  )
  const images = await prepareMath(
    [...model.title, ...model.description, ...model.body],
    pixelsPerEm,
  )
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
  layers?: CardLayer[],
) {
  const spec = FORMATS[format]
  const resolution = cardResolution(format)
  if (canvas.width !== resolution.width) canvas.width = resolution.width
  if (canvas.height !== resolution.height) canvas.height = resolution.height
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(resolution.scale, 0, 0, resolution.scale, 0, 0)
  ctx.clearRect(0, 0, spec.width, spec.height)
  const training = model.kind === 'activity' || model.kind === 'day'
  if (!training || !model.show.transparent) {
    ctx.fillStyle = palette.bg
    ctx.fillRect(0, 0, spec.width, spec.height)
  }
  if (model.kind === 'quote') drawQuote(ctx, fonts, model, spec, palette, layers)
  else if (training) {
    if (model.training) ctx.drawImage(model.training, 0, 0, spec.width, spec.height)
    if (!model.overlay || model.overlay === 'card') chrome(ctx, fonts, model, spec, palette, layers)
  } else if (model.kind === 'calendar') drawCalendar(ctx, fonts, model, spec, palette, layers)
  else if (model.kind === 'entry') drawEntry(ctx, fonts, model, spec, palette, layers)
  else drawNote(ctx, fonts, model, spec, palette, layers)
}

export async function prepareCardScene(
  model: CardModel,
  format: FormatKey,
  palette: Palette,
  fonts: Fonts,
  signal?: AbortSignal,
): Promise<CardScene> {
  const { width, height } = FORMATS[format]
  const { scale } = cardResolution(format)
  const layers = model.trainingSvg
    ? await svgLayers(
        model.trainingSvg,
        width,
        height,
        scale,
        svgImage,
        signal,
        model.trainingLayers,
      )
    : []
  const canvas = document.createElement('canvas')
  renderCard(canvas, model, format, palette, fonts, layers)
  canvas.width = canvas.height = 1
  return {
    width,
    height,
    scale,
    background:
      (model.kind === 'activity' || model.kind === 'day') && model.show.transparent
        ? null
        : palette.bg,
    layers,
  }
}
