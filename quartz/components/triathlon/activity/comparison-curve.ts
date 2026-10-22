import type { PowerCurveData } from '../../../util/power-curve'
import type {
  ActivityComparisonScrubContext,
  ActivityComparisonScrubState,
} from './comparison-graph'
import { decodePowerCurves } from '../../../util/power-curve'
import { dlabel } from '../../../util/triathlon-card'
import { nearestPowerCurveValue } from '../../../util/triathlon-card'
import { normalizePowerCurvePoints } from '../../../util/triathlon-card'
import { powerCurveFraction } from '../../../util/triathlon-card'
import { powerCurveReferenceLabel } from '../../../util/triathlon-i18n'
import { bindActivityComparisonGraph } from './comparison-graph'
import { positionActivityComparisonCursor } from './comparison-graph'

/** Scrub, duration ticks and reference range for the comparison power curve. */
export const wireComparisonPowerCurve = (scrub: ActivityComparisonScrubContext): (() => void)[] => {
  const { presentation, activities, charts, text } = scrub
  const { setReadout, showChartCursors, activate, render, deactivate } = scrub
  const cleanups: (() => void)[] = []
  const curveChart = charts.find(chart => chart.dataset.compareChart === 'power-curve')
  const curveGraph = curveChart?.querySelector<SVGElement>('.tri-compare-graph')
  const powerCurves = activities.map(activity =>
    normalizePowerCurvePoints(activity.powerCurve ?? []),
  )
  const curves = powerCurves.filter(curve => curve.length >= 2)
  if (curveChart && curveGraph && curves.length > 0) {
    type ComparisonCurveRange = 'six-weeks' | 'year'
    const curveState: ActivityComparisonScrubState = { fraction: 0, selectedFraction: 0 }
    const minDurationS = Math.min(...curves.map(curve => curve[0].s))
    const maxDurationS = Math.max(...curves.map(curve => curve[curve.length - 1].s))
    const [sixWeeks, year] = decodePowerCurves(curveGraph.dataset.powerCurves)
    const curveReferences: Record<ComparisonCurveRange, PowerCurveData> = {
      'six-weeks': sixWeeks ?? [],
      year: year ?? [],
    }
    let curveRange: ComparisonCurveRange =
      curveGraph.dataset.curveRange === 'year' ? 'year' : 'six-weeks'
    const rawCurveYear = Number(curveGraph.dataset.curveYear)
    const curveYear =
      curveGraph.dataset.curveYear && Number.isInteger(rawCurveYear) ? rawCurveYear : null
    const durationAt = (fraction: number): number =>
      Math.exp(
        Math.log(minDurationS) +
          Math.min(1, Math.max(0, fraction)) * (Math.log(maxDurationS) - Math.log(minDurationS)),
      )
    const showCurve = (fraction: number) => {
      curveState.fraction = Math.min(1, Math.max(0, fraction))
      const durationS = durationAt(curveState.fraction)
      positionActivityComparisonCursor(curveGraph, curveState.fraction)
      const values = activities.map((activity, index) => {
        const value = nearestPowerCurveValue(powerCurves[index], durationS)
        return { activity, value: value == null ? '–' : `${value.toLocaleString()} W` }
      })
      showChartCursors([curveChart])
      setReadout(
        'power-curve',
        values.map(({ activity, value }) => ({ activity, value, missing: value === '–' })),
      )
      const referenceWatts = nearestPowerCurveValue(curveReferences[curveRange], durationS)
      const selectedSeconds = Math.max(1, Math.round(durationAt(curveState.selectedFraction ?? 0)))
      for (const tick of curveChart.querySelectorAll<HTMLButtonElement>('.tri-curve-tick'))
        tick.setAttribute(
          'aria-pressed',
          String(Number(tick.dataset.curveSeconds) === selectedSeconds),
        )
      curveGraph.setAttribute('aria-valuenow', Math.round(durationS).toString())
      curveGraph.setAttribute(
        'aria-valuetext',
        `${dlabel(Math.max(1, Math.round(durationS)))}; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '–' ? text('no data') : value}`,
          )
          .join('; ')}${
          referenceWatts == null
            ? ''
            : `; ${powerCurveReferenceLabel(presentation.locale, curveRange === 'year' ? curveYear : null)}: ${referenceWatts.toLocaleString()} W`
        }`,
      )
    }
    const selectCurve = (fraction: number) => {
      const selected = Math.min(1, Math.max(0, fraction))
      curveState.fraction = selected
      curveState.selectedFraction = selected
      showCurve(selected)
      curveGraph.focus({ preventScroll: true })
    }
    const selectCurveRange = (range: ComparisonCurveRange) => {
      if (curveReferences[range].length < 2) return
      curveRange = range
      curveGraph.dataset.curveRange = range
      for (const option of curveChart.querySelectorAll<HTMLButtonElement>('.tri-curve-range'))
        option.setAttribute('aria-pressed', String(option.dataset.curveRange === range))
      for (const path of curveGraph.querySelectorAll<SVGElement>(
        '.tri-compare-curve-ref[data-curve-range]',
      ))
        path.toggleAttribute('hidden', path.dataset.curveRange !== range)
      for (const element of curveChart.querySelectorAll<HTMLElement | SVGElement>(
        '[data-critical-power-range]',
      ))
        element.toggleAttribute('hidden', element.dataset.criticalPowerRange !== range)
      const label = curveChart.querySelector<HTMLElement>('.tri-compare-curve-reference-label')
      if (label) {
        label.removeAttribute('data-i18n')
        label.textContent = powerCurveReferenceLabel(
          presentation.locale,
          range === 'year' ? curveYear : null,
        )
      }
      showCurve(curveState.selectedFraction ?? curveState.fraction)
    }
    const onCurveClick = (event: MouseEvent) => {
      if (!(event.target instanceof Element)) return
      const rangeButton = event.target.closest<HTMLButtonElement>('.tri-curve-range')
      if (rangeButton && !rangeButton.disabled) {
        selectCurveRange(rangeButton.dataset.curveRange === 'year' ? 'year' : 'six-weeks')
        curveGraph.focus({ preventScroll: true })
        return
      }
      const tick = event.target.closest<HTMLButtonElement>('.tri-curve-tick')
      if (tick) {
        const seconds = Number(tick.dataset.curveSeconds)
        if (seconds > 0) selectCurve(powerCurveFraction(seconds, minDurationS, maxDurationS))
        return
      }
      const axis = event.target.closest<HTMLElement>('.tri-cax-xax')
      if (!axis || !curveChart.contains(axis)) return
      const bounds = axis.getBoundingClientRect()
      if (bounds.width > 0) selectCurve((event.clientX - bounds.left) / bounds.width)
    }
    curveChart.addEventListener('click', onCurveClick)
    cleanups.push(
      bindActivityComparisonGraph(curveGraph, curveState, showCurve, activate, render, deactivate),
      () => curveChart.removeEventListener('click', onCurveClick),
    )
  }
  return cleanups
}
