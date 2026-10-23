import { setText } from '../../../util/set-text'
import {
  SITE_PERFORMANCE_SAMPLE_EVENT,
  type SitePerformanceSample,
  type SitePerformanceSource,
} from '../performance-sample'
import {
  createElement,
  createMetric,
  drawChart,
  formatDuration,
  formatHeap,
  scriptLabel,
  type ChartGeometry,
  type LongAnimationFrameScript,
  type Metric,
} from './presentation'
import {
  matchesPerformanceDebugQuery,
  type PerformanceDebugProbe,
  type PerformanceDebugProfile,
} from './profile'
import {
  FRAME_BUDGET_MS,
  BUCKET_CAPACITY,
  clearRing,
  clearSeries,
  createRing,
  createSeries,
  estimateFrameBudget,
  percentile95,
  pushRing,
  pushSeries,
  ringValues,
  seriesWindow,
  summarizeFrameDurations,
  type OpenBucket,
  type RingBuffer,
} from './sampling'

const FRAME_CAPACITY = 480

const SAMPLE_CAPACITY = 120

const BUCKET_MS = 100

const CHART_INTERVAL_MS = 100

const METRIC_INTERVAL_MS = 500

const IDLE_GAP_MS = 1000

const EVENT_THRESHOLD_MS = 16

// Refresh intervals jitter around the budget; a frame counts as dropped only past one and a half.
const DROPPED_FACTOR = 1.5

interface FrameWork {
  own: number
  script: number
  start: number
  time: number
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

const SOURCE_LABELS: Record<SitePerformanceSource, string> = {
  cursor: 'cursor p95',
  popover: 'popover p95',
  timeline: 'timeline p95',
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

  for (const existing of document.querySelectorAll('.site-perf-debug')) existing.remove()
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
  let chartGeometryDirty = false

  const updateChartGeometry = (): void => {
    if (!active) return
    chartGeometryDirty = false
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
    chartGeometryDirty = true
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
      setText(metric.name, reading.label)
      setText(metric.value, reading.value)
    })
    for (const metric of readingMetrics.splice(values.length)) metric.element.remove()
  }

  const updateState = (): void => {
    const cursor = document.querySelector<HTMLElement>('.site-cursor')
    setText(state, profile.state?.() || cursor?.dataset.mode || 'idle')
  }

  const updateMetrics = (timestamp: number): void => {
    const frameValues = ringValues(frames)
    frameBudget = estimateFrameBudget(frameValues)
    const summary = summarizeFrameDurations(frameValues, frameBudget * DROPPED_FACTOR)
    setText(fps, `${Math.round(summary.fps)} fps`)
    setText(frameMetric.value, formatDuration(summary.p95))
    setText(droppedMetric.value, `${Math.round(summary.slowRatio * 100)}%`)
    setText(workMetric.value, formatDuration(percentile95(work)))
    setText(scriptMetric.value, formatDuration(percentile95(script)))
    setText(renderMetric.value, formatDuration(percentile95(render)))
    setText(longFrameMetric.value, String(longFrames))
    setText(worstMetric.value, formatDuration(worstFrame))
    setText(blockedMetric.value, formatDuration(blockedTotal))
    setText(shiftMetric.value, layoutShift.toFixed(3))
    for (const [source, metric] of sourceMetrics)
      setText(metric.value, formatDuration(percentile95(samples[source])))
    const elapsed = Math.max(1, timestamp - previousScrollPaint)
    const scrollRate = ((scrollEvents - previousScrollEvents) * 1000) / elapsed
    setText(scrollMetric.value, `${scrollEvents} · ${scrollRate.toFixed(0)} /s`)
    previousScrollEvents = scrollEvents
    previousScrollPaint = timestamp
    const heap = (performance as Performance & { memory?: HeapMemory }).memory
    setText(heapMetric.value, heap ? formatHeap(heap.usedJSHeapSize) : '—')
    setText(nodeMetric.value, String(document.getElementsByTagName('*').length))
    setText(
      scriptRow.value,
      worstScript ? `${worstScript} · ${formatDuration(worstScriptDuration)}` : '—',
    )
    setText(
      eventRow.value,
      worstInteractionName ? `${worstInteractionName} · ${formatDuration(worstInteraction)}` : '—',
    )
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
      if (chartGeometryDirty) updateChartGeometry()
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
    if (!event.altKey || !event.shiftKey || event.ctrlKey || event.metaKey || event.code !== 'KeyD')
      return
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
  if (matchesPerformanceDebugQuery(window.location.search, profile.query)) start()

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
