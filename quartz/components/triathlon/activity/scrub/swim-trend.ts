import { swimChartMetric, type SwimChartMetric } from '../../../../util/swim-metrics'
import {
  clock,
  swimTrendHoverAt,
  type SwimTrendChartPoint,
  type SwimTrendMode,
} from '../../../../util/triathlon-card'
import {
  swimActivityDisplayValue,
  swimActivityPointText,
  swimActivityValueText,
} from '../../../../util/triathlon-i18n'
import { setText } from '../../runtime/dom'
import { activate, clearActiveBar, scrubText, type ScrubHoverState } from './hover-readouts'

const swimCache = new WeakMap<
  SVGSVGElement,
  { lengths: SwimTrendChartPoint[]; '100m': SwimTrendChartPoint[] }
>()

const isSwimTrendPoint = (value: unknown): value is SwimTrendChartPoint =>
  typeof value === 'object' &&
  value !== null &&
  'elapsedS' in value &&
  typeof value.elapsedS === 'number' &&
  Number.isFinite(value.elapsedS) &&
  'cumulativeDistanceM' in value &&
  typeof value.cumulativeDistanceM === 'number' &&
  Number.isFinite(value.cumulativeDistanceM) &&
  'value' in value &&
  typeof value.value === 'number' &&
  Number.isFinite(value.value) &&
  'xPct' in value &&
  typeof value.xPct === 'number' &&
  Number.isFinite(value.xPct) &&
  'yPct' in value &&
  typeof value.yPct === 'number' &&
  Number.isFinite(value.yPct) &&
  (!('windowStartDistanceM' in value) ||
    (typeof value.windowStartDistanceM === 'number' && Number.isFinite(value.windowStartDistanceM)))

const swimMode = (svg: SVGSVGElement): SwimTrendMode =>
  svg.dataset.swimMode === '100m' ? '100m' : 'lengths'

const decodeSwimData = (value: string | undefined): SwimTrendChartPoint[] => {
  const parsed: unknown = JSON.parse(value ?? '[]')
  return Array.isArray(parsed) ? parsed.filter(isSwimTrendPoint) : []
}

export const swimData = (
  svg: SVGSVGElement,
  mode: SwimTrendMode = swimMode(svg),
): SwimTrendChartPoint[] => {
  const cached = swimCache.get(svg)
  if (cached) return cached[mode]
  const value = {
    lengths: decodeSwimData(svg.dataset.swimSeriesLengths),
    '100m': decodeSwimData(svg.dataset.swimSeriesHundred),
  }
  swimCache.set(svg, value)
  return value[mode]
}

export const selectedSwimIndex = (svg: SVGSVGElement): number => {
  const value = Number(svg.dataset.swimIndex ?? 0)
  return Number.isInteger(value) ? value : 0
}

const swimKind = (svg: SVGSVGElement): SwimChartMetric => swimChartMetric(svg.dataset.swimKind)

const swimKindLabel = (kind: SwimChartMetric): string =>
  kind === 'pace'
    ? 'pace'
    : kind === 'cadence'
      ? 'cadence'
      : kind === 'rate'
        ? 'stroke rate'
        : 'SWOLF'

export const swimAriaLabel = (state: ScrubHoverState, svg: SVGSVGElement): string =>
  `${scrubText(state, 'swim')} ${scrubText(state, swimKindLabel(swimKind(svg)))} · ${scrubText(state, swimMode(svg) === '100m' ? '100 m' : 'lengths')}`

const swimDisplayValue = (state: ScrubHoverState, kind: SwimChartMetric, value: number): string =>
  swimActivityDisplayValue(state.presentation().locale, kind, value, clock(value))

const swimTextPoint = (point: SwimTrendChartPoint) => ({
  elapsed: clock(point.elapsedS),
  cumulativeDistanceM: point.cumulativeDistanceM,
  ...(point.windowStartDistanceM == null
    ? {}
    : { windowStartDistanceM: point.windowStartDistanceM }),
})

const swimValueText = (
  state: ScrubHoverState,
  kind: SwimChartMetric,
  point: SwimTrendChartPoint,
): string =>
  swimActivityValueText(
    state.presentation().locale,
    kind,
    swimTextPoint(point),
    point.value,
    clock(point.value),
  )

export const showSwim = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  fraction: number,
  activateChart = true,
): void => {
  const wrap = svg.closest<HTMLElement>('.tri-zone')
  const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
  const point = wrap?.querySelector<HTMLElement>('.tri-swim-trend-hover')
  const readoutPosition = wrap?.querySelector<HTMLElement>('.tri-swim-trend-readout-position')
  const readoutValue = wrap?.querySelector<HTMLElement>('.tri-swim-trend-readout-value')
  if (!wrap || !point || !readoutPosition || !readoutValue) return
  const kind = swimKind(svg)
  const hover = swimTrendHoverAt(swimData(svg), fraction)
  if (!hover) return
  cursor?.setAttribute('x1', hover.xPct.toFixed(2))
  cursor?.setAttribute('x2', hover.xPct.toFixed(2))
  point.style.left = `${hover.xPct.toFixed(2)}%`
  point.style.top = `${hover.yPct.toFixed(2)}%`
  point.hidden = !activateChart
  setText(readoutPosition, swimActivityPointText(state.presentation().locale, swimTextPoint(hover)))
  setText(readoutValue, swimDisplayValue(state, kind, hover.value))
  svg.dataset.swimIndex = String(hover.index)
  svg.setAttribute('aria-label', swimAriaLabel(state, svg))
  svg.setAttribute('aria-valuenow', String(Math.round(hover.cumulativeDistanceM)))
  svg.setAttribute('aria-valuetext', swimValueText(state, kind, hover))
  if (activateChart) {
    clearActiveBar(state)
    activate(state, wrap)
  }
}

export const showSwimIndex = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  requestedIndex: number,
  activateChart = true,
): void => {
  const points = swimData(svg)
  if (points.length === 0) return
  const index = Math.min(points.length - 1, Math.max(0, requestedIndex))
  showSwim(state, svg, points[index].xPct / 100, activateChart)
}

export const onSwimRestore = (state: ScrubHoverState, event: Event): void => {
  if (
    !(event.target instanceof SVGSVGElement) ||
    !event.target.classList.contains('tri-swim-trend-svg')
  )
    return
  const svg = event.target
  const distanceM = Number(svg.dataset.swimRestoreDistance)
  const totalDistanceM = Number(svg.getAttribute('aria-valuemax'))
  const activateChart = svg.dataset.swimRestoreActive === 'true'
  delete svg.dataset.swimRestoreDistance
  delete svg.dataset.swimRestoreActive
  if (!Number.isFinite(distanceM) || !Number.isFinite(totalDistanceM) || totalDistanceM <= 0) return
  showSwim(state, svg, distanceM / totalDistanceM, activateChart)
}

const setSwimLayer = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  mode: SwimTrendMode,
  animate: boolean,
): void => {
  const previousMode = swimMode(svg)
  if (previousMode === mode) return
  const previous = svg.querySelector<SVGGElement>(
    `.tri-swim-series[data-swim-mode="${previousMode}"]`,
  )
  const next = svg.querySelector<SVGGElement>(`.tri-swim-series[data-swim-mode="${mode}"]`)
  if (!previous || !next) return
  const previousOpacity = getComputedStyle(previous).opacity
  const nextOpacity = getComputedStyle(next).opacity
  state.swimAnimations.get(previous)?.cancel()
  state.swimAnimations.get(next)?.cancel()
  state.swimAnimations.delete(previous)
  state.swimAnimations.delete(next)
  previous.classList.remove('tri-swim-series--active')
  previous.setAttribute('aria-hidden', 'true')
  next.classList.add('tri-swim-series--active')
  next.setAttribute('aria-hidden', 'false')
  svg.dataset.swimMode = mode
  if (!animate || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
  const timing: KeyframeAnimationOptions = {
    duration: 180,
    easing: 'cubic-bezier(0.77, 0, 0.175, 1)',
  }
  state.swimAnimations.set(
    previous,
    previous.animate([{ opacity: previousOpacity }, { opacity: 0 }], timing),
  )
  state.swimAnimations.set(next, next.animate([{ opacity: nextOpacity }, { opacity: 1 }], timing))
}

export const setSwimMode = (
  state: ScrubHoverState,
  section: HTMLElement,
  mode: SwimTrendMode,
  animate: boolean,
): void => {
  const toggle = section.querySelector<HTMLElement>('.tri-swim-mode-toggle')
  if (!toggle || toggle.dataset.swimMode === mode) return
  toggle.dataset.swimMode = mode
  for (const option of toggle.querySelectorAll<HTMLButtonElement>('.tri-swim-mode'))
    option.setAttribute('aria-pressed', String(option.dataset.swimMode === mode))
  for (const svg of section.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
    const previous = swimData(svg)
    const selected = Math.min(previous.length - 1, Math.max(0, selectedSwimIndex(svg)))
    const fraction = previous[selected]?.xPct != null ? previous[selected].xPct / 100 : 1
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const wasActive = wrap?.classList.contains('tri-chart--hover') ?? false
    setSwimLayer(state, svg, mode, animate)
    delete svg.dataset.swimIndex
    showSwim(state, svg, fraction, wasActive)
  }
}
