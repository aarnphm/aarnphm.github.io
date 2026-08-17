import {
  SITE_PERFORMANCE_SAMPLE_EVENT,
  type SitePerformanceSample,
  type SitePerformanceSource,
} from '../../scripts/performance-sample'

const FRAME_BUDGET_MS = 1000 / 60
const FRAME_CAPACITY = 240
const SAMPLE_CAPACITY = 120
const PAINT_INTERVAL_MS = 250
const DEBUG_QUERY = 'tri-debug'
const DEBUG_VALUE = 'performance'

interface RingBuffer {
  count: number
  cursor: number
  values: Float32Array
}

export interface FrameSummary {
  fps: number
  p95: number
  slowRatio: number
}

interface ChartGeometry {
  accent: string
  font: string
  gray: string
  height: number
  lightgray: string
  width: number
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

const formatDuration = (duration: number): string =>
  duration >= 10 ? `${duration.toFixed(1)} ms` : `${duration.toFixed(2)} ms`

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

const createMetric = (label: string): { element: HTMLElement; value: HTMLElement } => {
  const element = createElement('div', 'tri-perf-metric')
  const name = createElement('span', 'tri-perf-metric-label')
  const value = createElement('span', 'tri-perf-metric-value')
  name.textContent = label
  value.textContent = '—'
  element.append(name, value)
  return { element, value }
}

const drawChart = (
  canvas: HTMLCanvasElement,
  frameDurations: readonly number[],
  geometry: ChartGeometry,
): void => {
  const context = canvas.getContext('2d')
  const { accent, font, gray, height, lightgray, width } = geometry
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
  const maxFps = 120
  const budgetY = height - (60 / maxFps) * height
  context.strokeStyle = lightgray
  context.lineWidth = 1
  context.setLineDash([2, 3])
  context.beginPath()
  context.moveTo(0, budgetY + 0.5)
  context.lineTo(width, budgetY + 0.5)
  context.stroke()
  context.setLineDash([])
  if (frameDurations.length < 2) return
  context.strokeStyle = accent
  context.lineWidth = 1.25
  context.lineJoin = 'round'
  context.beginPath()
  for (let index = 0; index < frameDurations.length; index += 1) {
    const fps = Math.min(maxFps, 1000 / Math.max(frameDurations[index], 0.01))
    const x = (index / (frameDurations.length - 1)) * width
    const y = height - (fps / maxFps) * height
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  }
  context.stroke()
  context.fillStyle = gray
  context.font = font
  context.textAlign = 'left'
  context.fillText('60', 3, Math.max(8, budgetY - 3))
}

const debugEnabledByQuery = (): boolean =>
  new URLSearchParams(window.location.search).get(DEBUG_QUERY) === DEBUG_VALUE

export const setupPerformanceDebug = (root: HTMLElement): (() => void) | null => {
  const timeline = root.querySelector<HTMLElement>('.tri-scroll')
  if (!timeline) return null

  const panel = createElement('aside', 'tri-perf-debug')
  panel.hidden = true
  panel.setAttribute('aria-label', 'triathlon performance debug')
  const header = createElement('div', 'tri-perf-head')
  const title = createElement('span', 'tri-perf-title')
  title.textContent = 'performance'
  const shortcut = createElement('kbd', 'tri-perf-shortcut')
  shortcut.textContent = '⌥⇧D'
  const close = createElement('button', 'tri-perf-close')
  close.type = 'button'
  close.setAttribute('aria-label', 'close performance debug')
  close.dataset.siteCursorClose = ''
  close.innerHTML =
    '<svg data-site-cursor-icon viewBox="0 0 12 12" aria-hidden="true"><path d="M2.25 2.25 9.75 9.75M9.75 2.25 2.25 9.75"/></svg>'
  header.append(title, shortcut, close)

  const primary = createElement('div', 'tri-perf-primary')
  const fps = createElement('strong', 'tri-perf-fps')
  fps.textContent = '— fps'
  const state = createElement('span', 'tri-perf-state')
  state.textContent = 'idle'
  primary.append(fps, state)

  const canvas = createElement('canvas', 'tri-perf-chart')
  canvas.setAttribute('aria-label', 'recent frames per second')
  canvas.setAttribute('role', 'img')

  const metrics = createElement('div', 'tri-perf-metrics')
  const frameMetric = createMetric('frame p95')
  const slowMetric = createMetric('over 16.7 ms')
  const longFrameMetric = createMetric('long frames')
  const worstMetric = createMetric('worst frame')
  const cursorMetric = createMetric('cursor p95')
  const timelineMetric = createMetric('timeline p95')
  const popoverMetric = createMetric('popover p95')
  const scrollMetric = createMetric('scroll events')
  metrics.append(
    frameMetric.element,
    slowMetric.element,
    longFrameMetric.element,
    worstMetric.element,
    cursorMetric.element,
    timelineMetric.element,
    popoverMetric.element,
    scrollMetric.element,
  )
  panel.append(header, primary, canvas, metrics)
  document.body.appendChild(panel)

  const frames = createRing(FRAME_CAPACITY)
  const samples: Record<SitePerformanceSource, RingBuffer> = {
    cursor: createRing(SAMPLE_CAPACITY),
    popover: createRing(SAMPLE_CAPACITY),
    timeline: createRing(SAMPLE_CAPACITY),
  }
  let animationFrame = 0
  let active = false
  let lastFrame = 0
  let lastPaint = 0
  let scrollEvents = 0
  let previousScrollEvents = 0
  let previousScrollPaint = 0
  let longFrames = 0
  let worstFrame = 0
  let observer: PerformanceObserver | null = null
  let chartGeometry: ChartGeometry = {
    accent: '#fc4c02',
    font: '8px monospace',
    gray: '#8b8b8b',
    height: 0,
    lightgray: '#d8d8d8',
    width: 0,
  }

  const updateChartGeometry = (): void => {
    const style = getComputedStyle(canvas)
    chartGeometry = {
      accent: style.getPropertyValue('--tri-accent').trim() || '#fc4c02',
      font: `8px ${style.fontFamily}`,
      gray: style.getPropertyValue('--gray').trim() || '#8b8b8b',
      height: canvas.clientHeight,
      lightgray: style.getPropertyValue('--lightgray').trim() || '#d8d8d8',
      width: canvas.clientWidth,
    }
  }
  const chartResize = new ResizeObserver(updateChartGeometry)
  chartResize.observe(canvas)

  const updateState = (): void => {
    const cursor = document.querySelector<HTMLElement>('.site-cursor')
    const openPanel = root.className.match(/tri-(analytics|calc|map|training)-open/)?.[1]
    state.textContent = openPanel ?? cursor?.dataset.mode ?? 'idle'
  }

  const updatePanel = (timestamp: number): void => {
    const frameValues = ringValues(frames)
    const summary = summarizeFrameDurations(frameValues)
    fps.textContent = `${Math.round(summary.fps)} fps`
    frameMetric.value.textContent = formatDuration(summary.p95)
    slowMetric.value.textContent = `${Math.round(summary.slowRatio * 100)}%`
    longFrameMetric.value.textContent = String(longFrames)
    worstMetric.value.textContent = formatDuration(worstFrame)
    cursorMetric.value.textContent = formatDuration(percentile95(samples.cursor))
    timelineMetric.value.textContent = formatDuration(percentile95(samples.timeline))
    popoverMetric.value.textContent = formatDuration(percentile95(samples.popover))
    const elapsed = Math.max(1, timestamp - previousScrollPaint)
    const scrollRate = ((scrollEvents - previousScrollEvents) * 1000) / elapsed
    scrollMetric.value.textContent = `${scrollEvents} · ${scrollRate.toFixed(0)} /s`
    previousScrollEvents = scrollEvents
    previousScrollPaint = timestamp
    updateState()
    drawChart(canvas, frameValues, chartGeometry)
  }

  const tick = (timestamp: number): void => {
    if (!active) return
    if (lastFrame > 0) {
      const duration = timestamp - lastFrame
      if (duration < 250) pushRing(frames, duration)
    }
    lastFrame = timestamp
    if (timestamp - lastPaint >= PAINT_INTERVAL_MS) {
      lastPaint = timestamp
      updatePanel(timestamp)
    }
    animationFrame = window.requestAnimationFrame(tick)
  }

  const start = (): void => {
    if (active) return
    active = true
    panel.hidden = false
    document.documentElement.dataset.sitePerformanceDebug = 'true'
    clearRing(frames)
    clearRing(samples.cursor)
    clearRing(samples.timeline)
    clearRing(samples.popover)
    scrollEvents = 0
    previousScrollEvents = 0
    longFrames = 0
    worstFrame = 0
    updateChartGeometry()
    lastFrame = 0
    lastPaint = 0
    previousScrollPaint = performance.now()
    animationFrame = window.requestAnimationFrame(tick)
    if (PerformanceObserver.supportedEntryTypes.includes('long-animation-frame')) {
      observer = new PerformanceObserver(entries => {
        for (const entry of entries.getEntries()) {
          longFrames += 1
          worstFrame = Math.max(worstFrame, entry.duration)
        }
      })
      observer.observe({ type: 'long-animation-frame' })
    }
  }

  const stop = (): void => {
    if (!active) return
    active = false
    panel.hidden = true
    window.cancelAnimationFrame(animationFrame)
    animationFrame = 0
    observer?.disconnect()
    observer = null
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

  close.addEventListener('click', stop)
  document.addEventListener('keydown', onKey)
  window.addEventListener(SITE_PERFORMANCE_SAMPLE_EVENT, onSample)
  window.addEventListener('scroll', onScroll, { capture: true, passive: true })
  if (debugEnabledByQuery()) start()

  return () => {
    stop()
    close.removeEventListener('click', stop)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener(SITE_PERFORMANCE_SAMPLE_EVENT, onSample)
    window.removeEventListener('scroll', onScroll, true)
    chartResize.disconnect()
    panel.remove()
  }
}
