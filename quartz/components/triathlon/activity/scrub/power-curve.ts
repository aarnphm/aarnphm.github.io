import type { PowerCurvePoint } from '../../../../plugins/stores/strava'
import {
  decodePowerCurves,
  powerCurveSecondsAt,
  type PowerCurveData,
} from '../../../../util/power-curve'
import {
  buildPowerCurveGrid,
  nearestPowerCurvePoint,
  powerCurveAxisTicks,
  powerCurveFraction,
  powerCurveHoverAt,
  powerCurveValueText,
  zoneClock,
} from '../../../../util/triathlon-card'
import { powerCurveReferenceLabel } from '../../../../util/triathlon-i18n'
import { createDomFactory, setText } from '../../runtime/dom'
import { syncPowerCurveActivityLink } from '../power-links'
import { activate, clearActiveBar, scrubText, type ScrubHoverState } from './hover-readouts'

export type CurveRange = 'six-weeks' | 'year'
type CurveModelValue = { label: string; watts: number }

const curveCache = new WeakMap<
  SVGSVGElement,
  { curve: PowerCurveData; sixWeeks: PowerCurveData; year: PowerCurveData }
>()

export const curveData = (
  svg: SVGSVGElement,
): { curve: PowerCurveData; sixWeeks: PowerCurveData; year: PowerCurveData } => {
  const cached = curveCache.get(svg)
  if (cached) return cached
  const [curve, sixWeeks, year] = decodePowerCurves(svg.dataset.powerCurves)
  const value = { curve: curve ?? [], sixWeeks: sixWeeks ?? [], year: year ?? [] }
  curveCache.set(svg, value)
  return value
}

const curveRange = (svg: SVGSVGElement): CurveRange =>
  svg.dataset.curveRange === 'year' ? 'year' : 'six-weeks'

const curveReference = (
  svg: SVGSVGElement,
  data: { sixWeeks: PowerCurveData; year: PowerCurveData },
): PowerCurveData => (curveRange(svg) === 'year' ? data.year : data.sixWeeks)

const curveReferenceYear = (svg: SVGSVGElement): number | null => {
  if (curveRange(svg) !== 'year') return null
  const year = Number(svg.dataset.curveYear)
  return Number.isInteger(year) && year > 0 ? year : null
}

export const selectedCurveIndex = (svg: SVGSVGElement): number => {
  const value = Number(svg.dataset.curveSelectedIndex ?? 0)
  return Number.isInteger(value) ? value : 0
}

const curveWeightKg = (svg: SVGSVGElement): number | null => {
  if (svg.closest<HTMLElement>('.tri-curve-chart')?.dataset.curveUnit !== 'watts-per-kg')
    return null
  const kg = Number(svg.dataset.curveWeightKg)
  return Number.isFinite(kg) && kg > 0 ? kg : null
}

const curvePowerText = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  watts: number,
  compact = false,
): string => powerCurveValueText(watts, curveWeightKg(svg), state.presentation().locale, compact)

export const syncCurveUnits = (state: ScrubHoverState, svg: SVGSVGElement): void => {
  const wrap = svg.closest<HTMLElement>('.tri-curve-chart')
  if (!wrap) return
  const weightKg = curveWeightKg(svg)
  const unit = weightKg == null ? 'watts' : 'watts-per-kg'
  wrap.dataset.curveUnit = unit
  for (const button of wrap.querySelectorAll<HTMLButtonElement>('.tri-curve-unit'))
    button.setAttribute('aria-pressed', String(button.dataset.curveUnit === unit))
  const note = wrap.querySelector<HTMLElement>('.tri-curve-weight-note')
  if (note) note.hidden = Number(svg.dataset.curveWeightKg) > 0 && weightKg == null
  for (const value of wrap.querySelectorAll<HTMLElement>('.tri-curve-power-value'))
    value.textContent = curvePowerText(
      state,
      svg,
      Number(value.dataset.curveWatts),
      value.dataset.curveCompact === 'true',
    )

  const axis = wrap.querySelector<HTMLElement>('.tri-cax-yax')
  const maxWatts = Number(svg.dataset.curveDomainMax)
  const height = svg.viewBox.baseVal.height
  if (!axis || !Number.isFinite(maxWatts) || maxWatts <= 0 || height <= 0) return
  // All series share the selected Garmin weight, so their watt-domain geometry stays valid.
  const ticks = powerCurveAxisTicks(maxWatts, weightKg, state.presentation().locale)
  axis.replaceChildren(
    ...ticks.map(tick => {
      const label = document.createElement('span')
      label.className = 'tri-cax-yt'
      label.textContent = tick.label
      label.style.top = `${(((height - (tick.watts / maxWatts) * (height - 1)) / height) * 100).toFixed(2)}%`
      return label
    }),
  )
  svg.querySelector('.tri-power-curve-grid--power')?.replaceWith(
    buildPowerCurveGrid(
      createDomFactory(state.presentation()),
      'power',
      ticks.map(tick => height - (tick.watts / maxWatts) * (height - 1)),
      height,
    ),
  )
}

const syncCurveModelReadouts = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  wrap: HTMLElement,
  seconds: number,
): CurveModelValue[] => {
  const values: CurveModelValue[] = []
  for (const row of wrap.querySelectorAll<HTMLElement>('.tri-curve-readout-row--model')) {
    const criticalPower = Number(row.dataset.curveCriticalPower)
    const wPrime = Number(row.dataset.curveWPrime)
    const minSeconds = Number(row.dataset.curveModelMinSeconds)
    const maxSeconds = Number(row.dataset.curveModelMaxSeconds)
    const visible =
      Number.isFinite(criticalPower) &&
      Number.isFinite(wPrime) &&
      Number.isFinite(minSeconds) &&
      Number.isFinite(maxSeconds) &&
      seconds >= minSeconds &&
      seconds <= maxSeconds
    row.hidden = !visible
    if (!visible) continue
    const watts = Math.round(criticalPower + wPrime / seconds)
    const value = row.querySelector<HTMLElement>('.tri-curve-readout-value--model')
    const label = row.querySelector<HTMLElement>('.tri-curve-readout-label--model')
    if (value) setText(value, curvePowerText(state, svg, watts))
    values.push({ label: label?.textContent ?? 'eCP model', watts })
  }
  return values
}

const curveValueText = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  point: PowerCurvePoint,
  referenceWatts: number | null,
  modelValues: readonly CurveModelValue[],
): string =>
  `${zoneClock(point.s)}, ${scrubText(state, svg.dataset.curveSport === 'run' ? 'this run' : 'this ride')} ${curvePowerText(state, svg, point.w)}${referenceWatts == null ? '' : `, ${powerCurveReferenceLabel(state.presentation().locale, curveReferenceYear(svg))} ${curvePowerText(state, svg, referenceWatts)}`}${modelValues.map(model => `, ${model.label} ${curvePowerText(state, svg, model.watts)}`).join('')}`

const placeCurvePoint = (
  point: HTMLElement | null,
  x: number,
  watts: number | null,
  maxWatts: number,
  height: number,
): void => {
  if (!point) return
  point.hidden = watts == null
  if (watts == null) return
  const y = height - (Math.min(maxWatts, Math.max(0, watts)) / maxWatts) * (height - 1)
  point.style.left = `${x.toFixed(2)}%`
  point.style.top = `${((y / height) * 100).toFixed(2)}%`
}

export const showCurve = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  fraction: number,
  activateChart = true,
  commit = false,
): void => {
  const wrap = svg.closest<HTMLElement>('.tri-zone')
  const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
  const readout = wrap?.querySelector<HTMLElement>('.tri-curve-readout')
  if (!wrap || !readout) return
  const data = curveData(svg)
  const curve = data.curve
  const reference = curveReference(svg, data)
  const hover = powerCurveHoverAt(curve, reference, fraction)
  if (!hover) return
  if (commit) {
    svg.dataset.curveSelectedIndex = String(hover.index)
    for (const tick of wrap.querySelectorAll<HTMLButtonElement>('.tri-curve-tick'))
      tick.setAttribute(
        'aria-pressed',
        String(Number(tick.dataset.curveSeconds) === hover.durationS),
      )
  }
  if (svg.dataset.curveIndex !== String(hover.index)) {
    cursor?.setAttribute('x1', hover.xPct.toFixed(2))
    cursor?.setAttribute('x2', hover.xPct.toFixed(2))
    const duration = readout.querySelector<HTMLElement>('.tri-curve-readout-duration')
    const ride = readout.querySelector<HTMLElement>('.tri-curve-readout-value--ride')
    const referenceRow = readout.querySelector<HTMLElement>('.tri-curve-readout-row--ref')
    const referenceValue = readout.querySelector<HTMLElement>('.tri-curve-readout-value--ref')
    const referenceLabel = readout.querySelector<HTMLElement>('.tri-curve-readout-label--ref')
    const maxWatts = Number(svg.dataset.curveDomainMax)
    const height = svg.viewBox.baseVal.height
    if (Number.isFinite(maxWatts) && maxWatts > 0 && height > 0) {
      placeCurvePoint(
        wrap.querySelector<HTMLElement>('.tri-curve-point--ride'),
        hover.xPct,
        hover.watts,
        maxWatts,
        height,
      )
      placeCurvePoint(
        wrap.querySelector<HTMLElement>('.tri-curve-point--ref'),
        hover.xPct,
        hover.referenceWatts,
        maxWatts,
        height,
      )
    }
    if (duration) setText(duration, zoneClock(hover.durationS))
    if (ride) setText(ride, curvePowerText(state, svg, hover.watts))
    if (referenceRow) referenceRow.hidden = hover.referenceWatts == null
    syncPowerCurveActivityLink(
      referenceRow instanceof HTMLAnchorElement ? referenceRow : null,
      nearestPowerCurvePoint(reference, hover.durationS),
      wrap.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset.activityId,
    )
    if (referenceValue && hover.referenceWatts != null)
      setText(referenceValue, curvePowerText(state, svg, hover.referenceWatts))
    if (referenceLabel)
      setText(
        referenceLabel,
        powerCurveReferenceLabel(state.presentation().locale, curveReferenceYear(svg)),
      )
    const modelValues = syncCurveModelReadouts(state, svg, wrap, hover.durationS)
    svg.dataset.curveIndex = String(hover.index)
    svg.setAttribute('aria-valuenow', String(hover.durationS))
    svg.setAttribute(
      'aria-valuetext',
      curveValueText(
        state,
        svg,
        { s: hover.durationS, w: hover.watts },
        hover.referenceWatts,
        modelValues,
      ),
    )
  }
  if (activateChart) {
    clearActiveBar(state)
    activate(state, wrap)
  }
}

export const showCurveIndex = (
  state: ScrubHoverState,
  svg: SVGSVGElement,
  requestedIndex: number,
  activateChart = true,
  commit = false,
): void => {
  const { curve } = curveData(svg)
  if (curve.length < 2) return
  const index = Math.min(curve.length - 1, Math.max(0, requestedIndex))
  showCurve(
    state,
    svg,
    powerCurveFraction(
      powerCurveSecondsAt(curve, index),
      powerCurveSecondsAt(curve, 0),
      powerCurveSecondsAt(curve, curve.length - 1),
    ),
    activateChart,
    commit,
  )
}
