import {
  SITE_PERFORMANCE_SAMPLE_EVENT,
  type SitePerformanceSample,
  type SitePerformanceSource,
} from './performance-sample'

const FRAME_BUDGET_MS = 1000 / 60
const REFRESH_PERIODS = [1000 / 120, 1000 / 90, 1000 / 60, 1000 / 30]
const FRAME_CAPACITY = 480
const SAMPLE_CAPACITY = 120
const BUCKET_MS = 100
const BUCKET_CAPACITY = 64
const CHART_INTERVAL_MS = 100
const METRIC_INTERVAL_MS = 500
const CHART_FLOOR_MS = 4
const CHART_CEIL_MS = 250
const IDLE_GAP_MS = 1000
const SMOOTHING_PASSES = 2
const EVENT_THRESHOLD_MS = 16
// Refresh intervals jitter around the budget; a frame counts as dropped only past one and a half.
const DROPPED_FACTOR = 1.5
const DEBUG_VALUE = 'performance'

export interface PerformanceDebugReading {
  label: string
  value: string
}

export interface PerformanceDebugProbe {
  read(): readonly PerformanceDebugReading[]
  stop(): void
}

export interface PerformanceDebugProfile {
  label: string
  /** `?<query>=performance` opens the panel on load. */
  query: string
  sources: readonly SitePerformanceSource[]
  /** Text beside the frame rate; the cursor mode shows when this returns nothing. */
  state?: () => string | null | undefined
  /** Surface readings, created when the panel opens and stopped when it closes. */
  probe?: () => PerformanceDebugProbe
}

interface RingBuffer {
  count: number
  cursor: number
  values: Float32Array
}

interface BucketSeries {
  count: number
  cursor: number
  jank: Uint8Array
  max: Float32Array
  mean: Float32Array
  min: Float32Array
  work: Float32Array
}

interface OpenBucket {
  count: number
  jank: boolean
  max: number
  min: number
  start: number
  sum: number
  workCount: number
  workSum: number
}

interface FrameWork {
  own: number
  script: number
  start: number
  time: number
}

export interface FrameSummary {
  fps: number
  p95: number
  slowRatio: number
}

export interface ChartSeries {
  jank: readonly boolean[]
  max: readonly number[]
  mean: readonly number[]
  min: readonly number[]
  work: readonly number[]
}

interface ChartGeometry {
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

interface LongAnimationFrameScript {
  duration: number
  invoker: string
  invokerType: string
  sourceFunctionName: string
  sourceURL: string
}

interface LongAnimationFrameEntry extends PerformanceEntry {
  blockingDuration: number
  scripts: readonly LongAnimationFrameScript[]
}

interface LayoutShiftEntry extends PerformanceEntry {
  hadRecentInput: boolean
  value: number
}

interface HeapMemory {
  usedJSHeapSize: number
}

interface EventObserverInit extends PerformanceObserverInit {
  durationThreshold: number
}

interface Metric {
  element: HTMLElement
  name: HTMLElement
  value: HTMLElement
}

const SOURCE_LABELS: Record<SitePerformanceSource, string> = {
  cursor: 'cursor p95',
  popover: 'popover p95',
  timeline: 'timeline p95',
}

const createRing = (capacity: number): RingBuffer => ({
  count: 0,
  cursor: 0,
  values: new Float32Array(capacity),
})

const pushRing = (ring: RingBuffer, value: number): void => {
  ring.values[ring.cursor] = value
  ring.cursor = (ring.cursor + 1) % ring.values.length
  ring.count = Math.min(ring.count + 1, ring.values.length)
}

const clearRing = (ring: RingBuffer): void => {
  ring.count = 0
  ring.cursor = 0
  ring.values.fill(0)
}

const ringValues = (ring: RingBuffer): number[] => {
  const values: number[] = []
  const start = ring.count === ring.values.length ? ring.cursor : 0
  for (let index = 0; index < ring.count; index += 1)
    values.push(ring.values[(start + index) % ring.values.length])
  return values
}

const createSeries = (capacity: number): BucketSeries => ({
  count: 0,
  cursor: 0,
  jank: new Uint8Array(capacity),
  max: new Float32Array(capacity),
  mean: new Float32Array(capacity),
  min: new Float32Array(capacity),
  work: new Float32Array(capacity),
})

const pushSeries = (series: BucketSeries, bucket: OpenBucket): void => {
  const slot = series.cursor
  series.mean[slot] = bucket.sum / bucket.count
  series.min[slot] = bucket.min
  series.max[slot] = bucket.max
  series.work[slot] = bucket.workCount > 0 ? bucket.workSum / bucket.workCount : 0
  series.jank[slot] = bucket.jank ? 1 : 0
  series.cursor = (series.cursor + 1) % series.mean.length
  series.count = Math.min(series.count + 1, series.mean.length)
}

const clearSeries = (series: BucketSeries): void => {
  series.count = 0
  series.cursor = 0
  series.jank.fill(0)
  series.max.fill(0)
  series.mean.fill(0)
  series.min.fill(0)
  series.work.fill(0)
}

const seriesWindow = (series: BucketSeries): ChartSeries => {
  const capacity = series.mean.length
  const start = series.count === capacity ? series.cursor : 0
  const jank: boolean[] = []
  const max: number[] = []
  const mean: number[] = []
  const min: number[] = []
  const work: number[] = []
  for (let index = 0; index < series.count; index += 1) {
    const slot = (start + index) % capacity
    jank.push(series.jank[slot] === 1)
    max.push(series.max[slot])
    mean.push(series.mean[slot])
    min.push(series.min[slot])
    work.push(series.work[slot])
  }
  return { jank, max, mean, min, work }
}

export const summarizeFrameDurations = (
  values: readonly number[],
  frameBudget = FRAME_BUDGET_MS,
): FrameSummary => {
  if (values.length === 0) return { fps: 0, p95: 0, slowRatio: 0 }
  const sorted = [...values].sort((left, right) => left - right)
  const average = values.reduce((sum, value) => sum + value, 0) / values.length
  const p95Index = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)
  const slow = values.reduce((count, value) => count + Number(value > frameBudget), 0)
  return {
    fps: average > 0 ? 1000 / average : 0,
    p95: sorted[p95Index],
    slowRatio: slow / values.length,
  }
}

export const estimateFrameBudget = (values: readonly number[]): number => {
  if (values.length < 12) return FRAME_BUDGET_MS
  const sorted = [...values].sort((left, right) => left - right)
  const fastest = sorted[Math.floor(sorted.length * 0.1)]
  let budget = FRAME_BUDGET_MS
  let closest = Number.POSITIVE_INFINITY
  for (const period of REFRESH_PERIODS) {
    const distance = Math.abs(period - fastest)
    if (distance >= closest) continue
    closest = distance
    budget = period
  }
  return budget
}

export const smoothWindow = (values: readonly number[], passes = SMOOTHING_PASSES): number[] => {
  let current = [...values]
  if (current.length < 3) return current
  for (let pass = 0; pass < passes; pass += 1) {
    const source = current
    current = source.map((value, index) => {
      const before = source[Math.max(0, index - 1)]
      const after = source[Math.min(source.length - 1, index + 1)]
      return (before + 2 * value + after) / 4
    })
  }
  return current
}

export const monotoneSlopes = (values: readonly number[]): number[] => {
  const length = values.length
  const slopes = Array.from<number>({ length }).fill(0)
  if (length < 2) return slopes
  const deltas = Array.from({ length: length - 1 }, (_, index) => values[index + 1] - values[index])
  slopes[0] = deltas[0]
  slopes[length - 1] = deltas[length - 2]
  for (let index = 1; index < length - 1; index += 1)
    slopes[index] =
      deltas[index - 1] * deltas[index] <= 0 ? 0 : (deltas[index - 1] + deltas[index]) / 2
  for (let index = 0; index < length - 1; index += 1) {
    if (deltas[index] === 0) {
      slopes[index] = 0
      slopes[index + 1] = 0
      continue
    }
    const alpha = slopes[index] / deltas[index]
    const beta = slopes[index + 1] / deltas[index]
    const magnitude = Math.hypot(alpha, beta)
    if (magnitude <= 3) continue
    const scale = 3 / magnitude
    slopes[index] = scale * alpha * deltas[index]
    slopes[index + 1] = scale * beta * deltas[index]
  }
  return slopes
}

export const frameChartRatio = (duration: number): number => {
  const floor = Math.log(CHART_FLOOR_MS)
  const span = Math.log(CHART_CEIL_MS) - floor
  const clamped = Math.min(CHART_CEIL_MS, Math.max(CHART_FLOOR_MS, duration))
  return (Math.log(clamped) - floor) / span
}

const formatDuration = (duration: number): string => {
  if (duration >= 1000) return `${(duration / 1000).toFixed(1)} s`
  return duration >= 10 ? `${duration.toFixed(1)} ms` : `${duration.toFixed(2)} ms`
}

const formatHeap = (bytes: number): string => `${(bytes / 1048576).toFixed(1)} MB`

const percentile95 = (ring: RingBuffer): number => {
  const values = ringValues(ring)
  if (values.length === 0) return 0
  values.sort((left, right) => left - right)
  return values[Math.min(values.length - 1, Math.ceil(values.length * 0.95) - 1)]
}

const createElement = <Tag extends keyof HTMLElementTagNameMap>(
  tag: Tag,
  className: string,
): HTMLElementTagNameMap[Tag] => {
  const element = document.createElement(tag)
  element.className = className
  return element
}

const createMetric = (label: string): Metric => {
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

const drawChart = (
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

const scriptLabel = (script: LongAnimationFrameScript): string => {
  // A script block names its own URL as the invoker; the file already says that.
  const invoker = URL.canParse(script.invoker) ? script.invokerType : script.invoker
  const name = script.sourceFunctionName || invoker || 'anonymous'
  const file = script.sourceURL.split('?')[0].split('/').pop()
  return file ? `${name} · ${file}` : name
}

export const setupPerformanceDebug = (profile: PerformanceDebugProfile): (() => void) => {
  const panel = createElement('aside', 'site-perf-debug')
  panel.hidden = true
  panel.setAttribute('aria-label', profile.label)
  const header = createElement('div', 'site-perf-head')
  const title = createElement('span', 'site-perf-title')
  title.textContent = 'performance'
  const shortcut = createElement('kbd', 'site-perf-shortcut')
  shortcut.textContent = '⌥⇧D'
  const close = createElement('button', 'site-perf-close')
  close.type = 'button'
  close.setAttribute('aria-label', 'close performance debug')
  close.dataset.siteCursorClose = ''
  close.innerHTML =
    '<svg data-site-cursor-icon viewBox="0 0 12 12" aria-hidden="true"><path d="M2.25 2.25 9.75 9.75M9.75 2.25 2.25 9.75"/></svg>'
  header.append(title, shortcut, close)

  const primary = createElement('div', 'site-perf-primary')
  const fps = createElement('strong', 'site-perf-fps')
  fps.textContent = '— fps'
  const state = createElement('span', 'site-perf-state')
  state.textContent = 'idle'
  primary.append(fps, state)

  const canvas = createElement('canvas', 'site-perf-chart')
  canvas.setAttribute(
    'aria-label',
    'frame interval and main-thread work per frame over the last six seconds',
  )
  canvas.setAttribute('role', 'img')
  const legend = createElement('div', 'site-perf-legend')
  legend.setAttribute('aria-hidden', 'true')
  for (const [key, text] of [
    ['interval', 'interval'],
    ['work', 'work'],
    ['jank', 'long frame'],
  ]) {
    const item = createElement('span', 'site-perf-legend-item')
    item.dataset.series = key
    item.textContent = text
    legend.append(item)
  }

  const metrics = createElement('div', 'site-perf-metrics')
  const frameMetric = createMetric('frame p95')
  const droppedMetric = createMetric('dropped')
  const workMetric = createMetric('work p95')
  const scriptMetric = createMetric('raf p95')
  const renderMetric = createMetric('render p95')
  const longFrameMetric = createMetric('long frames')
  const worstMetric = createMetric('worst frame')
  const blockedMetric = createMetric('blocked')
  const shiftMetric = createMetric('shift')
  const scrollMetric = createMetric('scroll')
  const sourceMetrics = profile.sources.map(
    source => [source, createMetric(SOURCE_LABELS[source])] as const,
  )
  const heapMetric = createMetric('heap')
  const nodeMetric = createMetric('dom nodes')
  metrics.append(
    frameMetric.element,
    droppedMetric.element,
    workMetric.element,
    scriptMetric.element,
    renderMetric.element,
    longFrameMetric.element,
    worstMetric.element,
    blockedMetric.element,
    shiftMetric.element,
    scrollMetric.element,
    ...sourceMetrics.map(([, metric]) => metric.element),
    heapMetric.element,
    nodeMetric.element,
  )

  const readings = createElement('div', 'site-perf-metrics site-perf-readings')
  readings.hidden = true
  const readingMetrics: Metric[] = []

  const attribution = createElement('div', 'site-perf-attribution')
  const scriptRow = createMetric('worst script')
  scriptRow.element.className = 'site-perf-attribution-row'
  const eventRow = createMetric('worst event')
  eventRow.element.className = 'site-perf-attribution-row'
  attribution.append(scriptRow.element, eventRow.element)

  panel.append(header, primary, canvas, legend, metrics, readings, attribution)
  document.body.appendChild(panel)

  const frames = createRing(FRAME_CAPACITY)
  const work = createRing(FRAME_CAPACITY)
  const script = createRing(FRAME_CAPACITY)
  const render = createRing(FRAME_CAPACITY)
  const series = createSeries(BUCKET_CAPACITY)
  const bucket: OpenBucket = {
    count: 0,
    jank: false,
    max: 0,
    min: 0,
    start: 0,
    sum: 0,
    workCount: 0,
    workSum: 0,
  }
  const samples: Record<SitePerformanceSource, RingBuffer> = {
    cursor: createRing(SAMPLE_CAPACITY),
    popover: createRing(SAMPLE_CAPACITY),
    timeline: createRing(SAMPLE_CAPACITY),
  }
  const frameWork: FrameWork = { own: 0, script: 0, start: 0, time: -1 }
  let probe: PerformanceDebugProbe | null = null
  let rendered: MessageChannel | null = null
  let nativeRequestFrame: Window['requestAnimationFrame'] | null = null
  let timedRequestFrame: Window['requestAnimationFrame'] | null = null
  let animationFrame = 0
  let active = false
  let lastFrame = 0
  let lastChart = 0
  let lastMetrics = 0
  let scrollEvents = 0
  let previousScrollEvents = 0
  let previousScrollPaint = 0
  let longFrames = 0
  let worstFrame = 0
  let blockedTotal = 0
  let worstScript = ''
  let worstScriptDuration = 0
  let worstInteraction = 0
  let worstInteractionName = ''
  let layoutShift = 0
  let frameBudget = FRAME_BUDGET_MS
  let frameObserver: PerformanceObserver | null = null
  let eventObserver: PerformanceObserver | null = null
  let shiftObserver: PerformanceObserver | null = null
  let chartGeometry: ChartGeometry = {
    accent: '#fc4c02',
    font: '8px monospace',
    gray: '#8b8b8b',
    height: 0,
    sage: '#cdd597',
    salmon: '#fdb2a2',
    surface: '#fffcf0',
    text: '#100f0f',
    width: 0,
  }

  const updateChartGeometry = (): void => {
    const style = getComputedStyle(canvas)
    chartGeometry = {
      accent: style.getPropertyValue('--perf-accent').trim() || '#fc4c02',
      font: `8px ${style.fontFamily}`,
      gray: style.getPropertyValue('--gray').trim() || '#8b8b8b',
      height: canvas.clientHeight,
      sage: style.getPropertyValue('--fig-sage').trim() || '#cdd597',
      salmon: style.getPropertyValue('--fig-salmon').trim() || '#fdb2a2',
      surface: style.backgroundColor,
      text: style.color,
      width: canvas.clientWidth,
    }
  }
  const onThemeChange = (): void => {
    updateChartGeometry()
    if (active) drawChart(canvas, seriesWindow(series), chartGeometry, frameBudget)
  }
  // A page may ease its palette after a theme change (the home trail fades body tokens over
  // 1.2 s), so the colours read on themechange can be mid-fade. Read them again when it ends.
  const onTransitionEnd = (event: TransitionEvent): void => {
    if (event.target instanceof Element && event.target.contains(panel)) onThemeChange()
  }
  const chartResize = new ResizeObserver(updateChartGeometry)
  chartResize.observe(canvas)

  const resetBucket = (start: number): void => {
    bucket.count = 0
    bucket.jank = false
    bucket.max = 0
    bucket.min = 0
    bucket.start = start
    bucket.sum = 0
    bucket.workCount = 0
    bucket.workSum = 0
  }

  const advanceBuckets = (timestamp: number, duration: number): void => {
    if (timestamp - bucket.start > BUCKET_MS * BUCKET_CAPACITY)
      bucket.start = timestamp - BUCKET_MS * BUCKET_CAPACITY
    while (timestamp - bucket.start >= BUCKET_MS) {
      if (bucket.count === 0) {
        bucket.count = 1
        bucket.max = duration
        bucket.min = duration
        bucket.sum = duration
      }
      pushSeries(series, bucket)
      resetBucket(bucket.start + BUCKET_MS)
    }
  }

  const record = (timestamp: number, duration: number): void => {
    pushRing(frames, duration)
    if (bucket.count === 0) {
      bucket.max = duration
      bucket.min = duration
    } else {
      bucket.max = Math.max(bucket.max, duration)
      bucket.min = Math.min(bucket.min, duration)
    }
    bucket.count += 1
    bucket.sum += duration
    if (duration > frameBudget * 3) bucket.jank = true
    advanceBuckets(timestamp, duration)
  }

  // Every callback of one frame receives the same timestamp, so it names the frame.
  const noteFrameWork = (time: number, began: number, ended: number, own: boolean): void => {
    if (frameWork.time !== time) {
      frameWork.time = time
      frameWork.start = began
      frameWork.script = 0
      frameWork.own = 0
    }
    frameWork.start = Math.min(frameWork.start, began)
    if (own) frameWork.own += ended - began
    else frameWork.script += ended - began
  }

  // A message posted during the frame's callbacks runs after its style, layout, and paint.
  const onRendered = (event: MessageEvent<number>): void => {
    if (!active || frameWork.time !== event.data) return
    const total = Math.max(0, performance.now() - frameWork.start - frameWork.own)
    pushRing(work, total)
    pushRing(script, frameWork.script)
    pushRing(render, Math.max(0, total - frameWork.script))
    bucket.workSum += total
    bucket.workCount += 1
  }

  const timeFrameCallbacks = (): void => {
    const native = window.requestAnimationFrame
    nativeRequestFrame = native
    timedRequestFrame = callback =>
      native.call(window, time => {
        const began = performance.now()
        try {
          callback(time)
        } finally {
          noteFrameWork(time, began, performance.now(), false)
        }
      })
    window.requestAnimationFrame = timedRequestFrame
  }

  const restoreFrameCallbacks = (): void => {
    if (nativeRequestFrame && window.requestAnimationFrame === timedRequestFrame)
      window.requestAnimationFrame = nativeRequestFrame
    nativeRequestFrame = null
    timedRequestFrame = null
  }

  const updateReadings = (): void => {
    const values = probe?.read() ?? []
    readings.hidden = values.length === 0
    values.forEach((reading, index) => {
      let metric = readingMetrics[index]
      if (!metric) {
        metric = createMetric(reading.label)
        readingMetrics.push(metric)
        readings.append(metric.element)
      }
      metric.name.textContent = reading.label
      metric.value.textContent = reading.value
    })
    for (const metric of readingMetrics.splice(values.length)) metric.element.remove()
  }

  const updateState = (): void => {
    const cursor = document.querySelector<HTMLElement>('.site-cursor')
    state.textContent = profile.state?.() || cursor?.dataset.mode || 'idle'
  }

  const updateMetrics = (timestamp: number): void => {
    const frameValues = ringValues(frames)
    frameBudget = estimateFrameBudget(frameValues)
    const summary = summarizeFrameDurations(frameValues, frameBudget * DROPPED_FACTOR)
    fps.textContent = `${Math.round(summary.fps)} fps`
    frameMetric.value.textContent = formatDuration(summary.p95)
    droppedMetric.value.textContent = `${Math.round(summary.slowRatio * 100)}%`
    workMetric.value.textContent = formatDuration(percentile95(work))
    scriptMetric.value.textContent = formatDuration(percentile95(script))
    renderMetric.value.textContent = formatDuration(percentile95(render))
    longFrameMetric.value.textContent = String(longFrames)
    worstMetric.value.textContent = formatDuration(worstFrame)
    blockedMetric.value.textContent = formatDuration(blockedTotal)
    shiftMetric.value.textContent = layoutShift.toFixed(3)
    for (const [source, metric] of sourceMetrics)
      metric.value.textContent = formatDuration(percentile95(samples[source]))
    const elapsed = Math.max(1, timestamp - previousScrollPaint)
    const scrollRate = ((scrollEvents - previousScrollEvents) * 1000) / elapsed
    scrollMetric.value.textContent = `${scrollEvents} · ${scrollRate.toFixed(0)} /s`
    previousScrollEvents = scrollEvents
    previousScrollPaint = timestamp
    const heap = (performance as Performance & { memory?: HeapMemory }).memory
    heapMetric.value.textContent = heap ? formatHeap(heap.usedJSHeapSize) : '—'
    nodeMetric.value.textContent = String(document.getElementsByTagName('*').length)
    scriptRow.value.textContent = worstScript
      ? `${worstScript} · ${formatDuration(worstScriptDuration)}`
      : '—'
    eventRow.value.textContent = worstInteractionName
      ? `${worstInteractionName} · ${formatDuration(worstInteraction)}`
      : '—'
    updateReadings()
    updateState()
  }

  const tick = (timestamp: number): void => {
    if (!active) return
    const began = performance.now()
    if (lastFrame > 0) {
      const duration = timestamp - lastFrame
      if (duration <= IDLE_GAP_MS) record(timestamp, duration)
      else resetBucket(timestamp)
    }
    lastFrame = timestamp
    if (timestamp - lastChart >= CHART_INTERVAL_MS) {
      lastChart = timestamp
      drawChart(canvas, seriesWindow(series), chartGeometry, frameBudget)
    }
    if (timestamp - lastMetrics >= METRIC_INTERVAL_MS) {
      lastMetrics = timestamp
      updateMetrics(timestamp)
    }
    rendered?.port2.postMessage(timestamp)
    animationFrame = (nativeRequestFrame ?? window.requestAnimationFrame).call(window, tick)
    noteFrameWork(timestamp, began, performance.now(), true)
  }

  const start = (): void => {
    if (active) return
    active = true
    panel.hidden = false
    document.documentElement.dataset.sitePerformanceDebug = 'true'
    clearRing(frames)
    clearRing(work)
    clearRing(script)
    clearRing(render)
    clearRing(samples.cursor)
    clearRing(samples.timeline)
    clearRing(samples.popover)
    clearSeries(series)
    resetBucket(performance.now())
    frameWork.time = -1
    scrollEvents = 0
    previousScrollEvents = 0
    longFrames = 0
    worstFrame = 0
    blockedTotal = 0
    worstScript = ''
    worstScriptDuration = 0
    worstInteraction = 0
    worstInteractionName = ''
    layoutShift = 0
    frameBudget = FRAME_BUDGET_MS
    updateChartGeometry()
    lastFrame = 0
    lastChart = 0
    lastMetrics = 0
    previousScrollPaint = performance.now()
    probe = profile.probe?.() ?? null
    rendered = new MessageChannel()
    rendered.port1.onmessage = onRendered
    timeFrameCallbacks()
    animationFrame = nativeRequestFrame!.call(window, tick)
    if (PerformanceObserver.supportedEntryTypes.includes('long-animation-frame')) {
      frameObserver = new PerformanceObserver(entries => {
        for (const entry of entries.getEntries() as LongAnimationFrameEntry[]) {
          longFrames += 1
          worstFrame = Math.max(worstFrame, entry.duration)
          blockedTotal += entry.blockingDuration
          bucket.jank = true
          for (const script of entry.scripts ?? []) {
            if (script.duration <= worstScriptDuration) continue
            worstScriptDuration = script.duration
            worstScript = scriptLabel(script)
          }
        }
      })
      frameObserver.observe({ type: 'long-animation-frame' })
    }
    if (PerformanceObserver.supportedEntryTypes.includes('event')) {
      eventObserver = new PerformanceObserver(entries => {
        for (const entry of entries.getEntries()) {
          if (entry.duration <= worstInteraction) continue
          worstInteraction = entry.duration
          worstInteractionName = entry.name
        }
      })
      const eventInit: EventObserverInit = { durationThreshold: EVENT_THRESHOLD_MS, type: 'event' }
      eventObserver.observe(eventInit)
    }
    if (PerformanceObserver.supportedEntryTypes.includes('layout-shift')) {
      shiftObserver = new PerformanceObserver(entries => {
        for (const entry of entries.getEntries() as LayoutShiftEntry[])
          if (!entry.hadRecentInput) layoutShift += entry.value
      })
      shiftObserver.observe({ type: 'layout-shift' })
    }
  }

  const stop = (): void => {
    if (!active) return
    active = false
    panel.hidden = true
    window.cancelAnimationFrame(animationFrame)
    animationFrame = 0
    restoreFrameCallbacks()
    rendered?.port1.close()
    rendered?.port2.close()
    rendered = null
    probe?.stop()
    probe = null
    frameObserver?.disconnect()
    frameObserver = null
    eventObserver?.disconnect()
    eventObserver = null
    shiftObserver?.disconnect()
    shiftObserver = null
    delete document.documentElement.dataset.sitePerformanceDebug
  }

  const toggle = (): void => {
    if (active) stop()
    else start()
  }

  const onKey = (event: KeyboardEvent): void => {
    if (!event.altKey || !event.shiftKey || event.code !== 'KeyD') return
    event.preventDefault()
    toggle()
  }
  const onSample = (event: CustomEvent<SitePerformanceSample>): void => {
    pushRing(samples[event.detail.source], event.detail.duration)
  }
  const onScroll = (): void => {
    scrollEvents += 1
  }
  const onVisibility = (): void => {
    if (document.visibilityState !== 'visible') return
    lastFrame = 0
    resetBucket(performance.now())
  }

  close.addEventListener('click', stop)
  document.addEventListener('keydown', onKey)
  document.addEventListener('themechange', onThemeChange)
  document.addEventListener('transitionend', onTransitionEnd)
  document.addEventListener('visibilitychange', onVisibility)
  window.addEventListener(SITE_PERFORMANCE_SAMPLE_EVENT, onSample)
  window.addEventListener('scroll', onScroll, { capture: true, passive: true })
  if (new URLSearchParams(window.location.search).get(profile.query) === DEBUG_VALUE) start()

  return () => {
    stop()
    close.removeEventListener('click', stop)
    document.removeEventListener('keydown', onKey)
    document.removeEventListener('themechange', onThemeChange)
    document.removeEventListener('transitionend', onTransitionEnd)
    document.removeEventListener('visibilitychange', onVisibility)
    window.removeEventListener(SITE_PERFORMANCE_SAMPLE_EVENT, onSample)
    window.removeEventListener('scroll', onScroll, true)
    chartResize.disconnect()
    panel.remove()
  }
}
