import type { GarminRunWalkSegment } from '../../../../plugins/stores/garmin'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import { clock, powerHistogramReadout, runWalkSegmentAt } from '../../../../util/triathlon-card'
import { triText } from '../../../../util/triathlon-i18n'
import { setText } from '../../runtime/dom'

export type ScrubHoverState = {
  readonly scope: HTMLElement
  readonly presentation: () => TriathlonPresentation
  activeWrap: HTMLElement | null
  activeBar: Element | null
  focusedSvg: SVGSVGElement | null
  readonly swimAnimations: Map<SVGGElement, Animation>
}

export const scrubText = (state: ScrubHoverState, key: string): string =>
  triText(state.presentation().locale, key)

const histogramCache = new WeakMap<SVGSVGElement, number[]>()

export const histogramData = (svg: SVGSVGElement): number[] => {
  const cached = histogramCache.get(svg)
  if (cached) return cached
  const parsed: unknown = JSON.parse(svg.dataset.hist ?? '[]')
  const histogram: number[] = []
  if (!Array.isArray(parsed)) return histogram
  for (const seconds of parsed) {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return []
    histogram.push(seconds)
  }
  histogramCache.set(svg, histogram)
  return histogram
}

const runWalkCache = new WeakMap<SVGSVGElement, GarminRunWalkSegment[]>()

export const runWalkSegments = (svg: SVGSVGElement): GarminRunWalkSegment[] => {
  const cached = runWalkCache.get(svg)
  if (cached) return cached
  const segments: GarminRunWalkSegment[] = []
  for (const rect of svg.querySelectorAll<SVGElement>('.tri-run-walk-segment')) {
    const state = rect.dataset.runWalkState
    const startElapsedS = Number(rect.dataset.startElapsedS)
    const endElapsedS = Number(rect.dataset.endElapsedS)
    if (
      (state === 'run' || state === 'walk' || state === 'idle') &&
      Number.isFinite(startElapsedS) &&
      Number.isFinite(endElapsedS) &&
      endElapsedS > startElapsedS
    )
      segments.push({ state, startElapsedS, endElapsedS })
  }
  runWalkCache.set(svg, segments)
  return segments
}

const deactivate = (wrap: HTMLElement): void => {
  wrap.classList.remove('tri-chart--hover')
  const point = wrap.querySelector<HTMLElement>('.tri-swim-trend-hover')
  if (point) point.hidden = true
}

export const activate = (state: ScrubHoverState, wrap: HTMLElement): void => {
  if (state.activeWrap && state.activeWrap !== wrap) deactivate(state.activeWrap)
  state.activeWrap = wrap
  wrap.classList.add('tri-chart--hover')
}

export const clearActiveBar = (state: ScrubHoverState): void => {
  state.activeBar?.classList.remove('tri-hist-bar--on')
  state.activeBar = null
}

export const clear = (state: ScrubHoverState): void => {
  if (state.activeWrap) deactivate(state.activeWrap)
  clearActiveBar(state)
  state.activeWrap = null
}

export const showHistogram = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  requestedBin: number,
): void => {
  const histogram = histogramData(svg)
  const wrap = svg.closest<HTMLElement>('.tri-zone')
  if (!wrap || histogram.length === 0) return
  const bin = Math.max(0, Math.min(histogram.length - 1, requestedBin))
  const value = powerHistogramReadout(histogram, bin, svg.dataset.histUnit === 'idx' ? 'idx' : 'W')
  const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
  cursor?.setAttribute('x1', String(bin + 0.5))
  cursor?.setAttribute('x2', String(bin + 0.5))
  state.activeBar?.classList.remove('tri-hist-bar--on')
  state.activeBar = svg.querySelector(`.tri-hist-bar[data-bin="${bin}"]`)
  state.activeBar?.classList.add('tri-hist-bar--on')
  const readout = wrap.querySelector<HTMLElement>('.tri-chart-readout')
  if (readout) setText(readout, value)
  svg.setAttribute('aria-valuenow', String(bin))
  svg.setAttribute('aria-valuetext', value)
  activate(state, wrap)
}

export const showRunWalk = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  requestedElapsedS: number,
): void => {
  const wrap = svg.closest<HTMLElement>('.tri-run-walk-chart')
  const readout = wrap?.querySelector<HTMLElement>('.tri-chart-readout')
  const durationS = Number(svg.dataset.domainEndElapsedS)
  if (!wrap || !readout || !Number.isFinite(durationS) || durationS <= 0) return
  const elapsedS = Math.max(0, Math.min(durationS, requestedElapsedS))
  const segment = runWalkSegmentAt(runWalkSegments(svg), elapsedS)
  const value = `${clock(elapsedS)} · ${scrubText(state, segment?.state ?? 'unavailable')}`
  setText(readout, value)
  svg.setAttribute('aria-valuenow', String(elapsedS))
  svg.setAttribute('aria-valuetext', value)
  const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
  const x = ((elapsedS / durationS) * 100).toFixed(4)
  cursor?.setAttribute('x1', x)
  cursor?.setAttribute('x2', x)
  activate(state, wrap)
}
