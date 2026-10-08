import type {
  EnvironmentChartSample,
  GardenEnvironmentSample,
} from '../../../util/activity-environment'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import {
  ENVIRONMENT_CHART_VIEWS,
  ALL_ENVIRONMENT_CHART_VIEWS,
  ENVIRONMENT_PROVIDER_CHART_VIEWS,
  environmentChartReadout,
  relativeWindReadout,
  type EnvironmentChartView,
} from '../../../util/triathlon-card'
import { isRecord } from '../../../util/type-guards'

export type EnvironmentView = EnvironmentChartView

const isEnvironmentView = (value: string | undefined): value is EnvironmentView =>
  ALL_ENVIRONMENT_CHART_VIEWS.some(view => view === value)

export const environmentViewFromKey = (
  selected: EnvironmentView,
  key: string,
  views: readonly EnvironmentView[] = ENVIRONMENT_CHART_VIEWS,
): EnvironmentView | null => {
  const current = views.indexOf(selected)
  if (current < 0 || views.length === 0) return null
  const index =
    key === 'Home'
      ? 0
      : key === 'End'
        ? views.length - 1
        : key === 'ArrowLeft'
          ? (current - 1 + views.length) % views.length
          : key === 'ArrowRight'
            ? (current + 1) % views.length
            : -1
  return views[index] ?? null
}

export const environmentCursorIndex = (value: string | undefined, sampleCount: number): number => {
  const finalIndex = Math.max(0, sampleCount - 1)
  if (value == null || value === '') return finalIndex
  const index = Number(value)
  return Number.isInteger(index) ? Math.min(finalIndex, Math.max(0, index)) : finalIndex
}

const finiteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const nullableFiniteNumber = (value: unknown): value is number | null =>
  value === null || finiteNumber(value)

const isEnvironmentSample = (value: unknown): value is GardenEnvironmentSample => {
  if (!isRecord(value)) return false
  if (!('elapsedS' in value) || !finiteNumber(value.elapsedS) || value.elapsedS < 0) return false
  if (!('distanceKm' in value) || !finiteNumber(value.distanceKm) || value.distanceKm < 0)
    return false
  if (
    'windSpeedKph' in value &&
    !(
      value.windSpeedKph === null ||
      (finiteNumber(value.windSpeedKph) && value.windSpeedKph >= 0 && value.windSpeedKph <= 1_000)
    )
  )
    return false
  return [
    'uvIndex',
    'cumulativeSed',
    'cumulativeMovingTelemetrySed',
    'ambientTemperatureC',
    'cloudCoverPct',
    'headwindKph',
    'crosswindKph',
    'apparentAirSpeedKph',
    'yawDeg',
  ].every(key => key in value && nullableFiniteNumber(value[key]))
}

export const parseEnvironmentSamples = (
  serialized: string | undefined,
): GardenEnvironmentSample[] => {
  try {
    const parsed: unknown = JSON.parse(serialized ?? 'null')
    if (!Array.isArray(parsed) || parsed.length < 2 || parsed.length > 512) return []
    const samples = parsed.filter(isEnvironmentSample)
    if (samples.length !== parsed.length) return []
    for (let index = 1; index < samples.length; index += 1)
      if (samples[index].elapsedS < samples[index - 1].elapsedS) return []
    return samples
  } catch {
    return []
  }
}

const currentSampleKeys = new Set([
  'elapsedS',
  'surfaceCurrentSpeedMps',
  'surfaceCurrentDirectionDeg',
])

const isEnvironmentCurrentSample = (value: unknown): value is EnvironmentChartSample =>
  isRecord(value) &&
  Object.keys(value).every(key => currentSampleKeys.has(key)) &&
  finiteNumber(value.elapsedS) &&
  value.elapsedS >= 0 &&
  (value.surfaceCurrentSpeedMps === null ||
    (finiteNumber(value.surfaceCurrentSpeedMps) &&
      value.surfaceCurrentSpeedMps >= 0 &&
      value.surfaceCurrentSpeedMps <= 10)) &&
  (value.surfaceCurrentDirectionDeg === null ||
    (finiteNumber(value.surfaceCurrentDirectionDeg) &&
      value.surfaceCurrentDirectionDeg >= 0 &&
      value.surfaceCurrentDirectionDeg < 360))

export const parseEnvironmentCurrentSamples = (
  serialized: string | undefined,
): EnvironmentChartSample[] => {
  try {
    const parsed: unknown = JSON.parse(serialized ?? 'null')
    if (!Array.isArray(parsed) || parsed.length < 2 || parsed.length > 512) return []
    const samples = parsed.filter(isEnvironmentCurrentSample)
    if (samples.length !== parsed.length) return []
    for (let index = 1; index < samples.length; index += 1)
      if (samples[index].elapsedS <= samples[index - 1].elapsedS) return []
    return samples
  } catch {
    return []
  }
}

const providerSampleKeys = new Set([
  'elapsedS',
  'distanceKm',
  'providerHeadwindKph',
  'providerCrosswindKph',
  'weatherCostW',
  'movingAirPenaltyKm',
  'headwindKph',
  'crosswindKph',
])

const isEnvironmentProviderSample = (value: unknown): value is EnvironmentChartSample =>
  isRecord(value) &&
  Object.keys(value).every(key => providerSampleKeys.has(key)) &&
  finiteNumber(value.elapsedS) &&
  value.elapsedS >= 0 &&
  [
    'providerHeadwindKph',
    'providerCrosswindKph',
    'weatherCostW',
    'movingAirPenaltyKm',
    'headwindKph',
    'crosswindKph',
  ].every(key => key in value && nullableFiniteNumber(value[key]))

export const parseEnvironmentProviderSamples = (
  serialized: string | undefined,
): EnvironmentChartSample[] => {
  try {
    const parsed: unknown = JSON.parse(serialized ?? 'null')
    if (!Array.isArray(parsed) || parsed.length < 2 || parsed.length > 512) return []
    const samples = parsed.filter(isEnvironmentProviderSample)
    if (samples.length !== parsed.length) return []
    for (let index = 1; index < samples.length; index += 1)
      if (samples[index].elapsedS <= samples[index - 1].elapsedS) return []
    return samples
  } catch {
    return []
  }
}

const relativeWindShares = (analysis: HTMLElement): number[] | null => {
  try {
    const parsed: unknown = JSON.parse(analysis.dataset.environmentRelativeWind ?? 'null')
    return Array.isArray(parsed) &&
      parsed.length === 8 &&
      parsed.every(share => finiteNumber(share) && share >= 0 && share <= 100)
      ? parsed
      : null
  } catch {
    return null
  }
}

const readSamples = (analysis: HTMLElement): EnvironmentChartSample[] => {
  const view = analysis.dataset.environmentView
  if (view === 'current-speed' || view === 'current-direction')
    return parseEnvironmentCurrentSamples(analysis.dataset.environmentCurrentSeries)
  if (view === 'relative-wind') return []
  if (
    (view === 'wind' && analysis.dataset.environmentWindSource === 'mywindsock') ||
    ENVIRONMENT_PROVIDER_CHART_VIEWS.some(providerView => providerView === view)
  )
    return parseEnvironmentProviderSamples(analysis.dataset.environmentProviderSeries)
  return parseEnvironmentSamples(analysis.dataset.environmentSeries)
}

const selectedViews = (analysis: HTMLElement): EnvironmentView[] =>
  Array.from(analysis.querySelectorAll<HTMLElement>('[data-environment-tab]')).flatMap(tab => {
    const view = tab.dataset.environmentTab
    return isEnvironmentView(view) ? [view] : []
  })

const setView = (analysis: HTMLElement, selected: EnvironmentView, focus: boolean): void => {
  const tabs = Array.from(analysis.querySelectorAll<HTMLButtonElement>('[data-environment-tab]'))
  const panels = Array.from(analysis.querySelectorAll<HTMLElement>('[data-environment-panel]'))
  const views = selectedViews(analysis)
  if (
    tabs.length === 0 ||
    tabs.length !== panels.length ||
    views.length !== tabs.length ||
    !views.includes(selected)
  )
    return
  const tablist = analysis.querySelector<HTMLElement>('.tri-environment-tablist')
  if (focus && tablist) tablist.dataset.motion = 'instant'
  analysis.dataset.environmentView = selected
  for (const tab of tabs) {
    const active = tab.dataset.environmentTab === selected
    tab.setAttribute('aria-selected', String(active))
    tab.tabIndex = active ? 0 : -1
  }
  if (focus) {
    tablist?.getBoundingClientRect()
    if (tablist) delete tablist.dataset.motion
    const activeTab = tabs.find(tab => tab.dataset.environmentTab === selected)
    activeTab?.focus({ preventScroll: true })
    activeTab?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }
  for (const panel of panels) {
    const active = panel.dataset.environmentPanel === selected
    panel.hidden = !active
    panel.inert = !active
    panel.setAttribute('aria-hidden', String(!active))
  }
  const mode = analysis.querySelector<HTMLElement>('.tri-environment-mode')
  if (mode) mode.hidden = selected !== 'cumulative'
}

const coefficient = (analysis: HTMLElement): number | null => {
  const value = Number(analysis.dataset.environmentScoreCoefficient)
  return Number.isFinite(value) && value > 0 ? value : null
}

const scoreClock = (analysis: HTMLElement): 'elapsed' | 'moving-telemetry' | null => {
  const value = analysis.dataset.environmentScoreClock
  return value === 'elapsed' || value === 'moving-telemetry' ? value : null
}

const cumulativeMode = (analysis: HTMLElement): 'score' | 'sed' =>
  analysis.dataset.environmentCumulativeMode === 'score' && coefficient(analysis) != null
    ? 'score'
    : 'sed'

const updateCursor = (
  analysis: HTMLElement,
  presentation: TriathlonPresentation,
  samples: readonly EnvironmentChartSample[],
  index: number,
  unavailableElapsedS?: number,
): void => {
  const output = analysis.querySelector<HTMLOutputElement>('[data-environment-readout]')
  if (analysis.dataset.environmentView === 'relative-wind') {
    const shares = relativeWindShares(analysis)
    if (output) output.value = shares ? relativeWindReadout(presentation, shares) : '—'
    return
  }
  const elapsed = Number(analysis.dataset.environmentElapsed)
  if (!Number.isFinite(elapsed) || elapsed <= 0) return
  const sample = samples[index] ?? { elapsedS: unavailableElapsedS ?? elapsed }
  const x = 2 + Math.min(1, Math.max(0, sample.elapsedS / elapsed)) * 96
  const coefficientSed = coefficient(analysis)
  const doseClock = scoreClock(analysis)
  const readout = environmentChartReadout(
    presentation,
    sample,
    coefficientSed != null && doseClock != null ? { coefficientSed, doseClock } : null,
    isEnvironmentView(analysis.dataset.environmentView)
      ? analysis.dataset.environmentView
      : undefined,
  )
  for (const chart of analysis.querySelectorAll<SVGElement>('[data-environment-chart]')) {
    const cursor = chart.querySelector<SVGLineElement>('.tri-environment-cursor')
    cursor?.setAttribute('x1', x.toFixed(3))
    cursor?.setAttribute('x2', x.toFixed(3))
    chart.setAttribute('aria-valuenow', `${Math.round(sample.elapsedS)}`)
    chart.setAttribute('aria-valuetext', readout)
  }
  if (output) output.value = readout
  analysis.dataset.environmentCursorElapsed = `${sample.elapsedS}`
  if (samples.length > 0) analysis.dataset.environmentSampleIndex = `${index}`
  else delete analysis.dataset.environmentSampleIndex
}

export const environmentSampleIndexAtElapsed = (
  samples: readonly EnvironmentChartSample[],
  elapsedS: number,
): number => {
  let closest = 0
  let delta = Number.POSITIVE_INFINITY
  for (const [index, sample] of samples.entries()) {
    const next = Math.abs(sample.elapsedS - elapsedS)
    if (next >= delta) continue
    closest = index
    delta = next
  }
  return closest
}

const eventElapsed = (analysis: HTMLElement, chart: SVGElement, clientX: number): number | null => {
  const elapsed = Number(analysis.dataset.environmentElapsed)
  const bounds = chart.getBoundingClientRect()
  if (!Number.isFinite(elapsed) || elapsed <= 0 || bounds.width <= 0) return null
  const fraction = (clientX - bounds.left) / bounds.width
  return Math.min(1, Math.max(0, (fraction - 0.02) / 0.96)) * elapsed
}

const setSelection = (analysis: HTMLElement, startS: number, endS: number): void => {
  const elapsed = Number(analysis.dataset.environmentElapsed)
  if (!Number.isFinite(elapsed) || elapsed <= 0) return
  const left = 2 + (Math.min(startS, endS) / elapsed) * 96
  const right = 2 + (Math.max(startS, endS) / elapsed) * 96
  for (const selection of analysis.querySelectorAll<SVGRectElement>('.tri-environment-selection')) {
    selection.setAttribute('x', left.toFixed(3))
    selection.setAttribute('width', Math.max(0.15, right - left).toFixed(3))
  }
  analysis.dataset.environmentSelectionStart = `${Math.min(startS, endS)}`
  analysis.dataset.environmentSelectionEnd = `${Math.max(startS, endS)}`
}

const setMode = (analysis: HTMLElement, mode: 'score' | 'sed'): void => {
  if (mode === 'score' && coefficient(analysis) == null) return
  analysis.dataset.environmentCumulativeMode = mode
  for (const button of analysis.querySelectorAll<HTMLButtonElement>('[data-environment-mode]'))
    button.setAttribute('aria-pressed', String(button.dataset.environmentMode === mode))
  for (const series of analysis.querySelectorAll<SVGElement>(
    '[data-environment-cumulative-series]',
  )) {
    if (series.dataset.environmentCumulativeSeries === mode) series.removeAttribute('hidden')
    else series.setAttribute('hidden', '')
  }
  for (const axis of analysis.querySelectorAll<HTMLElement>('[data-environment-cumulative-axis]')) {
    if (axis.dataset.environmentCumulativeAxis === mode) axis.removeAttribute('hidden')
    else axis.setAttribute('hidden', '')
  }
}

export const setupEnvironmentTabs = (
  root: HTMLElement,
  presentation: () => TriathlonPresentation,
): (() => void) => {
  let selection: { analysis: HTMLElement; startS: number; pointerId: number } | null = null

  const analysisFrom = (target: Element): HTMLElement | null =>
    target.closest<HTMLElement>('[data-environment-tabs]')

  const selectView = (analysis: HTMLElement, view: EnvironmentView, focus: boolean): void => {
    const previousElapsed = Number(analysis.dataset.environmentCursorElapsed)
    setView(analysis, view, focus)
    const samples = readSamples(analysis)
    const elapsedS = Number.isFinite(previousElapsed)
      ? previousElapsed
      : Number(analysis.dataset.environmentElapsed)
    updateCursor(
      analysis,
      presentation(),
      samples,
      environmentSampleIndexAtElapsed(samples, elapsedS),
      elapsedS,
    )
  }

  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const analysis = analysisFrom(event.target)
    if (!analysis || !root.contains(analysis)) return
    const tab = event.target.closest<HTMLButtonElement>('[data-environment-tab]')
    const view = tab?.dataset.environmentTab
    if (tab && isEnvironmentView(view)) {
      selectView(analysis, view, false)
      return
    }
    const mode =
      event.target.closest<HTMLButtonElement>('[data-environment-mode]')?.dataset.environmentMode
    if (mode === 'score' || mode === 'sed') setMode(analysis, mode)
  }

  const onKeyDown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    if (!(event.target instanceof Element)) return
    const analysis = analysisFrom(event.target)
    if (!analysis || !root.contains(analysis)) return
    if (event.target instanceof HTMLButtonElement) {
      const selected = event.target.dataset.environmentTab
      if (!isEnvironmentView(selected)) return
      const next = environmentViewFromKey(selected, event.key, selectedViews(analysis))
      if (!next) return
      event.preventDefault()
      event.stopPropagation()
      selectView(analysis, next, true)
      return
    }
    const chart = event.target.closest<SVGElement>('[data-environment-chart]')
    if (!chart) return
    const samples = readSamples(analysis)
    if (samples.length === 0) return
    const current = environmentCursorIndex(analysis.dataset.environmentSampleIndex, samples.length)
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? samples.length - 1
          : event.key === 'ArrowLeft' || event.key === 'ArrowDown'
            ? Math.max(0, current - 1)
            : event.key === 'ArrowRight' || event.key === 'ArrowUp'
              ? Math.min(samples.length - 1, current + 1)
              : -1
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    updateCursor(analysis, presentation(), samples, next)
  }

  const onLabelTransitionEnd = (event: TransitionEvent): void => {
    if (
      (event.propertyName !== 'inline-size' && event.propertyName !== 'width') ||
      !(event.target instanceof HTMLElement) ||
      !event.target.classList.contains('tri-environment-tab-full')
    )
      return
    const tab = event.target.closest<HTMLButtonElement>('[data-environment-tab]')
    if (tab?.getAttribute('aria-selected') === 'true')
      tab.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }

  const onPointerMove = (event: PointerEvent): void => {
    if (!(event.target instanceof Element)) return
    const chart = event.target.closest<SVGElement>('[data-environment-chart]')
    const analysis = chart ? analysisFrom(chart) : null
    if (!chart || !analysis || !root.contains(analysis)) return
    const elapsed = eventElapsed(analysis, chart, event.clientX)
    const samples = readSamples(analysis)
    if (elapsed == null || samples.length === 0) return
    updateCursor(
      analysis,
      presentation(),
      samples,
      environmentSampleIndexAtElapsed(samples, elapsed),
    )
    if (selection?.analysis === analysis && selection.pointerId === event.pointerId)
      setSelection(analysis, selection.startS, elapsed)
  }

  const onPointerDown = (event: PointerEvent): void => {
    if (!(event.target instanceof Element)) return
    const chart = event.target.closest<SVGElement>('[data-environment-chart]')
    const analysis = chart ? analysisFrom(chart) : null
    if (!chart || !analysis || !root.contains(analysis)) return
    const elapsed = eventElapsed(analysis, chart, event.clientX)
    if (elapsed == null) return
    selection = { analysis, startS: elapsed, pointerId: event.pointerId }
    chart.setPointerCapture(event.pointerId)
    setSelection(analysis, elapsed, elapsed)
  }

  const endSelection = (event: PointerEvent): void => {
    if (selection?.pointerId !== event.pointerId) return
    selection = null
  }

  for (const analysis of root.querySelectorAll<HTMLElement>('[data-environment-tabs]')) {
    const view = analysis.dataset.environmentView
    const views = selectedViews(analysis)
    const initial = isEnvironmentView(view) && views.includes(view) ? view : views[0]
    if (initial) setView(analysis, initial, false)
    setMode(analysis, cumulativeMode(analysis))
    const samples = readSamples(analysis)
    updateCursor(analysis, presentation(), samples, samples.length - 1)
  }
  root.addEventListener('click', onClick)
  root.addEventListener('keydown', onKeyDown)
  root.addEventListener('transitionend', onLabelTransitionEnd)
  root.addEventListener('pointermove', onPointerMove)
  root.addEventListener('pointerdown', onPointerDown)
  root.addEventListener('pointerup', endSelection)
  root.addEventListener('pointercancel', endSelection)
  return () => {
    root.removeEventListener('click', onClick)
    root.removeEventListener('keydown', onKeyDown)
    root.removeEventListener('transitionend', onLabelTransitionEnd)
    root.removeEventListener('pointermove', onPointerMove)
    root.removeEventListener('pointerdown', onPointerDown)
    root.removeEventListener('pointerup', endSelection)
    root.removeEventListener('pointercancel', endSelection)
    selection = null
  }
}
