import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { cyclingTorqueDensity } from '../cycling-torque'
import { triText } from '../triathlon-i18n'
import {
  activityCyclingIntensityPoints,
  activityCyclingTorquePoints,
  activityHeartRateTracePoints,
  activityTraceUsesElapsedAxis,
  activityWalkPowerPoints,
} from './activity-metric-data'
import { axisFrame, type ActivityGraphDomain } from './chart-geometry'
import { zoneClock } from './metric-text'
import { buildTraceSeries } from './trace-chart'

const HEART_RATE_TRACE_MIN_BPM = 80

export const buildWalkPowerTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const estimate = d.walkPower
  if (d.sport !== 'walk' || d.deviceWatts || !estimate) return null
  const points = activityWalkPowerPoints(d)
  const source = f.el('span', 'tri-elev-range', triText(f.presentation.locale, 'calculated'), {
    'data-gloss': '',
    'data-gloss-def': `Minetti minimum net metabolic demand above rest, from ${estimate.inputSource === 'strava' ? 'Strava' : 'Garmin'} speed and slope (${estimate.windowSeconds} s smoothing) and Garmin weight ${estimate.weight.kg} kg (${estimate.weight.date}). Experimental estimate, without individual gait, surface, or pack-load calibration. These watts describe walking energy demand and cannot be compared with cycling power-meter watts. Average includes recorded stops and excludes missing intervals.`,
    tabindex: '0',
  })
  return buildTraceSeries(
    f,
    d,
    points,
    point => point.watts,
    'walking power',
    () => `${Math.round(estimate.averageWatts)} W avg`,
    value => `${Math.round(value)} W`,
    undefined,
    d.route.length >= 2 ? selection : undefined,
    graphDomain,
    undefined,
    undefined,
    { wrapAttrs: { 'data-walking-power-source': 'garden-estimate' }, capExtra: [source] },
  )
}

export const buildIntensityFactorChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
): N | null => {
  const trace = d.cyclingIntensityTrace
  if (d.sport !== 'bike' || !d.wahoo || !trace) return null
  const points = activityCyclingIntensityPoints(d)
  const values = points.flatMap(point =>
    point.intensityFactor == null ? [] : [point.intensityFactor],
  )
  if (values.length < 2) return null
  const nativeIf = d.wahoo.metrics.intensityFactor
  const summary =
    nativeIf != null
      ? `${nativeIf.toFixed(3)} · Wahoo`
      : `${values.at(-1)?.toFixed(3)} · ${triText(f.presentation.locale, 'calculated')}`
  const source = f.el('span', 'tri-elev-range', triText(f.presentation.locale, 'cumulative'), {
    'data-gloss': '',
    'data-gloss-def': `${triText(f.presentation.locale, '30 s rolling power, cumulative NP, zeros included')} · NP / FTP · FTP ${trace.ftpWatts.toFixed(1)} W (${trace.ftpSource === 'wahoo-summary' ? 'Wahoo NP / IF' : triText(f.presentation.locale, 'athlete setting')})`,
    tabindex: '0',
  })
  return buildTraceSeries(
    f,
    d,
    points,
    point => point.intensityFactor,
    'intensity factor',
    () => summary,
    value => value.toFixed(2),
    { min: 0, max: Math.max(1.1, ...values) * 1.05 },
    d.route.length >= 2 ? selection : undefined,
    undefined,
    undefined,
    { value: 1, label: '1.00' },
    { wrapAttrs: { 'data-cycling-intensity-source': 'calculated-wahoo' }, capExtra: [source] },
  )
}

export const buildCrankTorqueChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const trace = d.cyclingTorque
  if (d.sport !== 'bike' || !trace || trace.cells.length === 0) return null
  const text = (key: string) => triText(f.presentation.locale, key)
  const panel = f.el('section', 'tri-zone tri-torque-panel', undefined, {
    'data-torque-mode': 'distance',
  })
  const controls = (selected: 'distance' | 'cadence'): N => {
    const group = f.el('div', 'tri-chart-controls tri-torque-controls', undefined, {
      role: 'group',
      'aria-label': text('crank torque view'),
    })
    for (const mode of ['distance', 'cadence'])
      f.add(
        group,
        f.el(
          'button',
          'tri-torque-mode tri-curve-range',
          text(mode === 'distance' && activityTraceUsesElapsedAxis(d) ? 'time' : mode),
          { type: 'button', 'data-torque-mode': mode, 'aria-pressed': String(mode === selected) },
        ),
      )
    return group
  }
  const points = activityCyclingTorquePoints(d)
  const values = points.flatMap(point => (point.torqueNm == null ? [] : [point.torqueNm]))
  const source = f.el('span', 'tri-elev-range', `${text('calculated')} · Wahoo`, {
    'data-gloss': '',
    tabindex: '0',
    'data-gloss-def': text('crank torque definition'),
  })
  const distance = buildTraceSeries(
    f,
    d,
    points,
    point => point.torqueNm,
    'crank torque',
    () =>
      `${trace.summary.averageTorqueNm.toFixed(1)} N·m · ${trace.summary.averageCadenceRpm.toFixed(0)} rpm`,
    value => `${value.toFixed(0)} N·m`,
    { min: 0, max: Math.max(10, ...values) * 1.05 },
    d.route.length >= 2 ? selection : undefined,
    graphDomain,
    undefined,
    undefined,
    { capExtra: [source, controls('distance')], wrapAttrs: { 'data-torque-pane': 'distance' } },
  )
  const cadence = f.el('div', 'tri-elev-wrap tri-torque-cadence', undefined, {
    'data-torque-pane': 'cadence',
    hidden: '',
  })
  const density = cyclingTorqueDensity(trace.cells)
  const xMax = Math.max(100, ...density.cells.map(cell => cell.cadenceRpm + 5))
  const yMax = density.limitNm + 5
  const maxSeconds = Math.max(...density.cells.map(cell => cell.seconds))
  const graph = f.svg('svg', {
    class: 'tri-torque-density',
    viewBox: '0 0 100 30',
    preserveAspectRatio: 'none',
    tabindex: '0',
    role: 'slider',
    'aria-label': text('crank torque by cadence'),
    'aria-valuemin': 0,
    'aria-valuemax': density.cells.length - 1,
    'aria-valuenow': 0,
    'aria-valuetext': text('time in cadence and torque bins'),
  })
  for (const cell of density.cells) {
    const torqueLabel =
      cell.torqueNm === density.limitNm
        ? `≥${density.limitNm}`
        : `${cell.torqueNm}–${cell.torqueNm + 5}`
    const label = `${cell.cadenceRpm}–${cell.cadenceRpm + 5} rpm · ${torqueLabel} N·m · ${zoneClock(cell.seconds)}`
    const rect = f.svg('rect', {
      class: 'tri-torque-cell',
      x: (cell.cadenceRpm / xMax) * 100,
      y: 30 - ((cell.torqueNm + 5) / yMax) * 30,
      width: (5 / xMax) * 100,
      height: (5 / yMax) * 30,
      opacity: 0.15 + 0.85 * Math.sqrt(cell.seconds / maxSeconds),
      'data-torque-cell': label,
    })
    f.add(graph, rect)
  }
  const caption = f.el('div', 'tri-elev-cap')
  f.add(
    caption,
    f.el('span', 'tri-elev-d', text('crank torque by cadence')),
    f.el('span', 'tri-elev-range', `${text('darker means more time')} · ${text('whole ride')}`),
    controls('cadence'),
  )
  f.add(
    cadence,
    caption,
    axisFrame(
      f,
      graph,
      [0, density.limitNm / 2, density.limitNm].map(value => ({
        label: `${value === density.limitNm ? '≥' : ''}${Math.round(value)} N·m`,
        vbY: 30 - (value / yMax) * 30,
      })),
      30,
      [0, xMax / 2, xMax].map(value => ({
        label: `${Math.round(value)} rpm`,
        pct: (value / xMax) * 100,
      })),
      true,
    ),
    f.el('div', 'tri-torque-readout', text('time in cadence and torque bins')),
  )
  f.add(panel, distance, cadence)
  return panel
}

export const buildHeartRateTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const points = activityHeartRateTracePoints(d)
  const heartRates = points.flatMap(point => (point.heartRate == null ? [] : [point.heartRate]))
  const minimum = Math.min(HEART_RATE_TRACE_MIN_BPM, Math.floor(Math.min(...heartRates) / 10) * 10)
  const maximum = Math.max(minimum + 20, ...heartRates)
  return buildTraceSeries(
    f,
    d,
    points,
    point => point.heartRate,
    'hr',
    peak => `${peak} bpm peak`,
    value => `${Math.round(value)}bpm`,
    { min: minimum, max: maximum, intervals: 4 },
    selection,
    graphDomain,
  )
}
