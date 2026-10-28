import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { powerCurveSecondsAt } from '../../../util/power-curve'
import { swimChartMetric } from '../../../util/swim-metrics'
import { clock, powerCurveFraction, type SwimTrendMode } from '../../../util/triathlon-card'
import { swimActivityDistanceText, swimActivityHeaderValue } from '../../../util/triathlon-i18n'
import { setupCyclingWattsCharts } from './cycling-watts'
import {
  clear,
  histogramData,
  runWalkSegments,
  scrubText,
  showHistogram,
  showRunWalk,
  type ScrubHoverState,
} from './scrub/hover-readouts'
import {
  closeShiftMenusOutside,
  onModeKey,
  onShiftMenuKey,
  setCyclingChartModeFromTarget,
  setRunMetricModeFromTarget,
  shiftSettingsFromTarget,
} from './scrub/modes'
import {
  curveData,
  selectedCurveIndex,
  showCurve,
  showCurveIndex,
  syncCurveUnits,
  type CurveRange,
} from './scrub/power-curve'
import {
  onSwimRestore,
  selectedSwimIndex,
  setSwimMode,
  showSwim,
  showSwimIndex,
  swimAriaLabel,
  swimData,
} from './scrub/swim-trend'
import { setupSwimPowerCharts } from './swim-power'
import { setupTorqueCharts } from './torque'
export {
  cyclingChartMode,
  restoreShiftSettings,
  runMetricMode,
  setCyclingChartMode,
  setRunMetricMode,
} from './scrub/modes'
export type { CyclingChartMode, RunMetricMode } from './scrub/modes'

const showFocused = (state: ScrubHoverState): void => {
  if (!state.focusedSvg) {
    clear(state)
    return
  }
  if (state.focusedSvg.classList.contains('tri-curve-svg'))
    showCurveIndex(state, state.focusedSvg, selectedCurveIndex(state.focusedSvg))
  else if (state.focusedSvg.classList.contains('tri-run-walk'))
    showRunWalk(state, state.focusedSvg, Number(state.focusedSvg.getAttribute('aria-valuenow')))
  else if (state.focusedSvg.classList.contains('tri-hist-svg'))
    showHistogram(state, state.focusedSvg, Number(state.focusedSvg.getAttribute('aria-valuenow')))
  else showSwimIndex(state, state.focusedSvg, selectedSwimIndex(state.focusedSvg))
}

const onPointer = (state: ScrubHoverState, event: PointerEvent): void => {
  if (!(event.target instanceof Element)) return
  const svg = event.target.closest<SVGSVGElement>(
    '.tri-curve-svg, .tri-hist-svg, .tri-swim-trend-svg, .tri-run-walk',
  )
  if (!svg) {
    if (state.activeWrap) showFocused(state)
    return
  }
  const r = svg.getBoundingClientRect()
  const frac = r.width > 0 ? Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)) : 0
  if (svg.classList.contains('tri-run-walk')) {
    showRunWalk(state, svg, frac * Number(svg.dataset.domainEndElapsedS))
    return
  }
  if (svg.classList.contains('tri-curve-svg')) {
    const commit = event.type === 'pointerdown'
    showCurve(state, svg, frac, true, commit)
    if (commit) {
      state.focusedSvg = svg
      svg.focus({ preventScroll: true })
    }
    return
  }
  if (svg.classList.contains('tri-swim-trend-svg')) {
    showSwim(state, svg, frac)
    return
  }
  showHistogram(state, svg, Math.floor(frac * histogramData(svg).length))
  if (event.type === 'pointerdown') {
    state.focusedSvg = svg
    svg.focus({ preventScroll: true })
  }
}

const onFocus = (state: ScrubHoverState, event: FocusEvent): void => {
  if (!(event.target instanceof Element)) return
  const svg = event.target.closest<SVGSVGElement>(
    '.tri-curve-svg, .tri-hist-svg, .tri-swim-trend-svg, .tri-run-walk',
  )
  if (!svg) return
  state.focusedSvg = svg
  if (svg.classList.contains('tri-curve-svg')) showCurveIndex(state, svg, selectedCurveIndex(svg))
  else if (svg.classList.contains('tri-run-walk'))
    showRunWalk(state, svg, Number(svg.getAttribute('aria-valuenow')))
  else if (svg.classList.contains('tri-hist-svg'))
    showHistogram(state, svg, Number(svg.getAttribute('aria-valuenow')))
  else showSwimIndex(state, svg, selectedSwimIndex(svg))
}

const onBlur = (state: ScrubHoverState, event: FocusEvent): void => {
  if (!(event.target instanceof Element)) return
  const svg = event.target.closest<SVGSVGElement>(
    '.tri-curve-svg, .tri-hist-svg, .tri-swim-trend-svg, .tri-run-walk',
  )
  if (!svg) return
  if (state.focusedSvg === svg) state.focusedSvg = null
  clear(state)
}

const onKey = (state: ScrubHoverState, event: KeyboardEvent): void => {
  if (!(event.target instanceof Element)) return
  const svg = event.target.closest<SVGSVGElement>(
    '.tri-curve-svg, .tri-hist-svg, .tri-swim-trend-svg, .tri-run-walk',
  )
  if (!svg) return
  if (svg.classList.contains('tri-run-walk')) {
    const segments = runWalkSegments(svg)
    const elapsedS = Number(svg.getAttribute('aria-valuenow'))
    const next =
      event.key === 'ArrowRight' || event.key === 'ArrowUp'
        ? segments.find(segment => segment.startElapsedS > elapsedS)?.startElapsedS
        : event.key === 'ArrowLeft' || event.key === 'ArrowDown'
          ? segments.findLast(segment => segment.startElapsedS < elapsedS)?.startElapsedS
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? Number(svg.dataset.domainEndElapsedS)
              : null
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      state.focusedSvg = null
      svg.blur()
      clear(state)
    } else if (next != null || event.key.startsWith('Arrow')) {
      event.preventDefault()
      showRunWalk(state, svg, next ?? elapsedS)
    }
    return
  }
  const isCurve = svg.classList.contains('tri-curve-svg')
  const isHistogram = svg.classList.contains('tri-hist-svg')
  const length = isCurve
    ? curveData(svg).curve.length
    : isHistogram
      ? histogramData(svg).length
      : swimData(svg).length
  if (length < (isHistogram ? 1 : 2)) return
  const current = isCurve
    ? selectedCurveIndex(svg)
    : isHistogram
      ? Number(svg.getAttribute('aria-valuenow'))
      : selectedSwimIndex(svg)
  let next: number | null = null
  if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = current - 1
  else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = current + 1
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = length - 1
  else if (event.key === 'Escape') {
    event.preventDefault()
    event.stopPropagation()
    state.focusedSvg = null
    svg.blur()
    clear(state)
    return
  }
  if (next == null) return
  event.preventDefault()
  state.focusedSvg = svg
  if (isCurve) showCurveIndex(state, svg, next, true, true)
  else if (isHistogram) showHistogram(state, svg, next)
  else showSwimIndex(state, svg, next)
}

const onChartClick = (state: ScrubHoverState, event: MouseEvent): void => {
  if (shiftSettingsFromTarget(event.target)) return
  if (setRunMetricModeFromTarget(event.target)) return
  if (setCyclingChartModeFromTarget(event.target)) return
  if (!(event.target instanceof Element)) return
  const swimButton = event.target.closest<HTMLButtonElement>('.tri-swim-mode')
  const swimSection = swimButton?.closest<HTMLElement>('.tri-swim-trends')
  if (swimButton && swimSection) {
    const mode: SwimTrendMode = swimButton.dataset.swimMode === '100m' ? '100m' : 'lengths'
    setSwimMode(state, swimSection, mode, event.detail > 0)
    return
  }
  const unitButton = event.target.closest<HTMLButtonElement>('.tri-curve-unit')
  if (unitButton) {
    const wrap = unitButton.closest<HTMLElement>('.tri-curve-chart')
    const svg = wrap?.querySelector<SVGSVGElement>('.tri-curve-svg')
    if (unitButton.disabled || !wrap || !svg) return
    const unit = unitButton.dataset.curveUnit
    if (unit !== 'watts' && unit !== 'watts-per-kg') return
    wrap.dataset.curveUnit = unit
    syncCurveUnits(state, svg)
    delete svg.dataset.curveIndex
    showCurveIndex(state, svg, selectedCurveIndex(svg), wrap.classList.contains('tri-chart--hover'))
    return
  }
  const curveTick = event.target.closest<HTMLButtonElement>('.tri-curve-tick')
  const curveAxis = event.target.closest<HTMLElement>('.tri-curve-chart .tri-cax-xax')
  const curveWrap = (curveTick ?? curveAxis)?.closest<HTMLElement>('.tri-curve-chart')
  const curveSvg = curveWrap?.querySelector<SVGSVGElement>('.tri-curve-svg')
  if (curveWrap && curveSvg) {
    if (curveTick) {
      const seconds = Number(curveTick.dataset.curveSeconds)
      const data = curveData(curveSvg).curve
      if (seconds > 0 && data.length >= 2)
        showCurve(
          state,
          curveSvg,
          powerCurveFraction(
            seconds,
            powerCurveSecondsAt(data, 0),
            powerCurveSecondsAt(data, data.length - 1),
          ),
          true,
          true,
        )
    } else if (curveAxis) {
      const rect = curveAxis.getBoundingClientRect()
      if (rect.width > 0)
        showCurve(state, curveSvg, (event.clientX - rect.left) / rect.width, true, true)
    }
    state.focusedSvg = curveSvg
    curveSvg.focus({ preventScroll: true })
    return
  }
  const button = event.target.closest<HTMLButtonElement>('.tri-curve-range')
  const wrap = button?.closest<HTMLElement>('.tri-zone')
  const svg = wrap?.querySelector<SVGSVGElement>('.tri-curve-svg')
  if (!button || button.disabled || !wrap || !svg) return
  const range: CurveRange = button.dataset.curveRange === 'year' ? 'year' : 'six-weeks'
  const data = curveData(svg)
  const reference = range === 'year' ? data.year : data.sixWeeks
  if (reference.length === 0) return
  const index = selectedCurveIndex(svg)
  const wasActive = wrap.classList.contains('tri-chart--hover')
  svg.dataset.curveRange = range
  for (const option of wrap.querySelectorAll<HTMLButtonElement>('.tri-curve-range'))
    option.setAttribute('aria-pressed', String(option.dataset.curveRange === range))
  for (const path of svg.querySelectorAll<SVGElement>('.tri-curve-ref[data-curve-range]'))
    path.toggleAttribute('hidden', path.dataset.curveRange !== range)
  delete svg.dataset.curveIndex
  showCurveIndex(state, svg, index, wasActive)
}

const onLocale = (state: ScrubHoverState): void => {
  for (const svg of state.scope.querySelectorAll<SVGSVGElement>('.tri-run-walk')) {
    svg.setAttribute('aria-label', scrubText(state, 'run/walk'))
    if (svg === state.focusedSvg || svg.closest('.tri-chart--hover'))
      showRunWalk(state, svg, Number(svg.getAttribute('aria-valuenow')))
  }
  for (const average of state.scope.querySelectorAll<HTMLElement>('.tri-swim-trend-value')) {
    const kind = swimChartMetric(average.dataset.swimAverageKind)
    const value = Number(average.dataset.swimAverageValue)
    if (Number.isFinite(value))
      average.textContent = swimActivityHeaderValue(
        state.presentation().locale,
        kind,
        value,
        clock(value),
      )
  }
  for (const svg of state.scope.querySelectorAll<SVGSVGElement>('.tri-curve-svg')) {
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const rideModelLabel = wrap?.querySelector<HTMLElement>(
      '.tri-curve-readout-row--model-ride .tri-curve-readout-label--model',
    )
    if (rideModelLabel) rideModelLabel.textContent = scrubText(state, 'this ride eCP model')
    syncCurveUnits(state, svg)
    delete svg.dataset.curveIndex
    showCurveIndex(state, svg, selectedCurveIndex(svg), false)
    svg.setAttribute('aria-label', scrubText(state, 'power curve'))
  }
  for (const svg of state.scope.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
    svg.setAttribute('aria-label', swimAriaLabel(state, svg))
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const totalDistanceM = Number(svg.getAttribute('aria-valuemax'))
    if (wrap && Number.isFinite(totalDistanceM)) {
      const distances = [0, totalDistanceM / 2, totalDistanceM]
      wrap.querySelectorAll<HTMLElement>('.tri-cax-xt').forEach((tick, index) => {
        const distanceM = distances[index]
        if (distanceM != null)
          tick.textContent = swimActivityDistanceText(state.presentation().locale, distanceM)
      })
    }
    showSwimIndex(
      state,
      svg,
      selectedSwimIndex(svg),
      wrap?.classList.contains('tri-chart--hover') ?? false,
    )
  }
}

export const setupChartScrub = (
  scope: HTMLElement,
  presentation: () => TriathlonPresentation,
): (() => void) => {
  const cleanupCyclingWatts = setupCyclingWattsCharts(scope, presentation)
  const cleanupTorque = setupTorqueCharts(scope)
  const cleanupSwimPower = setupSwimPowerCharts(scope)
  const state: ScrubHoverState = {
    scope,
    presentation,
    activeWrap: null,
    activeBar: null,
    focusedSvg: null,
    swimAnimations: new Map<SVGGElement, Animation>(),
  }
  const handlePointer = (event: PointerEvent): void => onPointer(state, event)
  const handlePointerLeave = (): void => showFocused(state)
  const handleFocus = (event: FocusEvent): void => onFocus(state, event)
  const handleBlur = (event: FocusEvent): void => onBlur(state, event)
  const handleKey = (event: KeyboardEvent): void => onKey(state, event)
  const handleClick = (event: MouseEvent): void => onChartClick(state, event)
  const handleShiftMenuOutside = (event: PointerEvent): void =>
    closeShiftMenusOutside(scope, event.target)
  const handleSwimRestore = (event: Event): void => onSwimRestore(state, event)
  const handleLocale = (): void => onLocale(state)
  scope.addEventListener('pointermove', handlePointer)
  scope.addEventListener('pointerdown', handlePointer)
  scope.addEventListener('pointerleave', handlePointerLeave)
  scope.addEventListener('pointercancel', handlePointerLeave)
  scope.addEventListener('focusin', handleFocus)
  scope.addEventListener('focusout', handleBlur)
  scope.addEventListener('keydown', handleKey)
  scope.addEventListener('keydown', onModeKey)
  scope.addEventListener('click', handleClick)
  scope.addEventListener('keydown', onShiftMenuKey)
  document.addEventListener('pointerdown', handleShiftMenuOutside)
  scope.addEventListener('tri:swim-restore', handleSwimRestore)
  window.addEventListener('tri:locale', handleLocale)
  handleLocale()
  return () => {
    cleanupCyclingWatts()
    cleanupTorque()
    cleanupSwimPower()
    clear(state)
    for (const animation of state.swimAnimations.values()) animation.cancel()
    state.swimAnimations.clear()
    scope.removeEventListener('pointermove', handlePointer)
    scope.removeEventListener('pointerdown', handlePointer)
    scope.removeEventListener('pointerleave', handlePointerLeave)
    scope.removeEventListener('pointercancel', handlePointerLeave)
    scope.removeEventListener('focusin', handleFocus)
    scope.removeEventListener('focusout', handleBlur)
    scope.removeEventListener('keydown', handleKey)
    scope.removeEventListener('keydown', onModeKey)
    scope.removeEventListener('click', handleClick)
    scope.removeEventListener('keydown', onShiftMenuKey)
    document.removeEventListener('pointerdown', handleShiftMenuOutside)
    scope.removeEventListener('tri:swim-restore', handleSwimRestore)
    window.removeEventListener('tri:locale', handleLocale)
  }
}
