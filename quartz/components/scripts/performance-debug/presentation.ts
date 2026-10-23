import {
  BUCKET_CAPACITY,
  frameChartRatio,
  monotoneSlopes,
  smoothWindow,
  type ChartSeries,
} from './sampling'

export interface ChartGeometry {
  accent: string
  font: string
  gray: string
  height: number
  sage: string
  salmon: string
  surface: string
  text: string
  width: number
}

export interface LongAnimationFrameScript {
  duration: number
  invoker: string
  invokerType: string
  sourceFunctionName: string
  sourceURL: string
}

export interface Metric {
  element: HTMLElement
  name: HTMLElement
  value: HTMLElement
}

export const formatDuration = (duration: number): string => {
  if (duration >= 1000) return `${(duration / 1000).toFixed(1)} s`
  return duration >= 10 ? `${duration.toFixed(1)} ms` : `${duration.toFixed(2)} ms`
}

export const formatHeap = (bytes: number): string => `${(bytes / 1048576).toFixed(1)} MB`

export const createElement = <Tag extends keyof HTMLElementTagNameMap>(
  tag: Tag,
  className: string,
): HTMLElementTagNameMap[Tag] => {
  const element = document.createElement(tag)
  element.className = className
  return element
}

export const createMetric = (label: string): Metric => {
  const element = createElement('div', 'site-perf-metric')
  const name = createElement('span', 'site-perf-metric-label')
  const value = createElement('span', 'site-perf-metric-value')
  name.textContent = label
  value.textContent = '—'
  element.append(name, value)
  return { element, name, value }
}

const traceCurve = (
  context: CanvasRenderingContext2D,
  xs: readonly number[],
  ys: readonly number[],
  connect: boolean,
): void => {
  const slopes = monotoneSlopes(ys)
  if (connect) context.lineTo(xs[0], ys[0])
  else context.moveTo(xs[0], ys[0])
  for (let index = 0; index < ys.length - 1; index += 1) {
    const step = xs[index + 1] - xs[index]
    context.bezierCurveTo(
      xs[index] + step / 3,
      ys[index] + slopes[index] / 3,
      xs[index + 1] - step / 3,
      ys[index + 1] - slopes[index + 1] / 3,
      xs[index + 1],
      ys[index + 1],
    )
  }
}

export const drawChart = (
  canvas: HTMLCanvasElement,
  series: ChartSeries,
  geometry: ChartGeometry,
  budget: number,
): void => {
  const context = canvas.getContext('2d')
  const { accent, font, gray, height, sage, salmon, surface, text, width } = geometry
  if (!context || width <= 0 || height <= 0) return
  const pixelRatio = Math.min(window.devicePixelRatio, 2)
  const targetWidth = Math.round(width * pixelRatio)
  const targetHeight = Math.round(height * pixelRatio)
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth
    canvas.height = targetHeight
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  context.clearRect(0, 0, width, height)
  const toY = (duration: number): number => height - frameChartRatio(duration) * height
  if (series.mean.length >= 2) {
    const step = width / (BUCKET_CAPACITY - 1)
    const offset = width - (series.mean.length - 1) * step
    const xs = series.mean.map((_, index) => offset + index * step)
    // Main-thread work fills from the floor; where it meets the interval line, frames drop.
    const workY = smoothWindow(series.work).map(toY)
    context.globalAlpha = 0.72
    context.fillStyle = sage
    context.beginPath()
    context.moveTo(xs[0], height)
    traceCurve(context, xs, workY, true)
    context.lineTo(xs[xs.length - 1], height)
    context.closePath()
    context.fill()
    const upperY = smoothWindow(series.max).map(toY)
    const lowerY = smoothWindow(series.min).map(toY)
    const meanY = smoothWindow(series.mean).map(toY)
    context.globalAlpha = 0.18
    context.fillStyle = accent
    context.beginPath()
    traceCurve(context, xs, upperY, false)
    traceCurve(context, [...xs].reverse(), [...lowerY].reverse(), true)
    context.closePath()
    context.fill()
    context.globalAlpha = 1
    context.strokeStyle = accent
    context.lineWidth = 1.25
    context.lineJoin = 'round'
    context.lineCap = 'round'
    context.beginPath()
    traceCurve(context, xs, meanY, false)
    context.stroke()
    context.fillStyle = accent
    context.beginPath()
    context.arc(xs[xs.length - 1], meanY[meanY.length - 1], 1.6, 0, Math.PI * 2)
    context.fill()
    const rug = Math.max(3, height * 0.16)
    context.fillStyle = salmon
    for (let index = 0; index < series.jank.length; index += 1) {
      if (!series.jank[index]) continue
      context.fillRect(xs[index] - 0.75, height - rug, 1.5, rug)
    }
  }
  context.font = font
  context.textAlign = 'left'
  context.textBaseline = 'alphabetic'
  for (const multiple of [1, 2, 4]) {
    const duration = budget * multiple
    const y = Math.round(toY(duration)) + 0.5
    context.globalAlpha = 0.45
    context.strokeStyle = gray
    context.lineWidth = 1
    context.setLineDash([2, 3])
    context.beginPath()
    context.moveTo(0, y)
    context.lineTo(width, y)
    context.stroke()
    context.setLineDash([])
    const label = String(Math.round(1000 / duration))
    const baseline = Math.max(8, y - 3)
    context.globalAlpha = 1
    context.fillStyle = surface
    context.fillRect(2, baseline - 7, context.measureText(label).width + 3, 8)
    context.globalAlpha = 0.72
    context.fillStyle = text
    context.fillText(label, 3, baseline)
    context.globalAlpha = 1
  }
}

export const scriptLabel = (script: LongAnimationFrameScript): string => {
  // A script block names its own URL as the invoker; the file already says that.
  const invoker = URL.canParse(script.invoker) ? script.invokerType : script.invoker
  const name = script.sourceFunctionName || invoker || 'anonymous'
  const file = script.sourceURL.split('?')[0].split('/').pop()
  return file ? `${name} · ${file}` : name
}
