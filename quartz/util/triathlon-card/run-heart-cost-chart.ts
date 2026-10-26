import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { RunHeartCostPoint } from '../run-mechanics'
import type { TriNodeFactory } from './construction-contracts'
import { preferredActivityWind } from '../activity-wind'
import { elapsedAxis, formatElapsedPauses } from '../elapsed-axis'
import { triText } from '../triathlon-i18n'
import { activityThermalTracePoints, runHeartCostValue } from './activity-metric-data'
import { axisFrame, elapsedActivityXTicks, niceTicks } from './chart-geometry'
import {
  analysisChartSelectionBounds,
  buildAnalysisSelectionRect,
  type AnalysisInterval,
} from './chart-selection'
import { addElapsedBreaks, addWindLegend, addWindStrip, elapsedSeriesPath } from './elapsed-chart'
import { elevationValue, formatThermalTemperature, isImperial, zoneClock } from './metric-text'

const quantile = (sorted: readonly number[], q: number): number =>
  sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]

export const buildRunHeartCostChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: AnalysisInterval | null,
): N | null => {
  const trace = d.runHeartCost
  if (d.sport !== 'run' || !trace || trace.points.length < 2) return null
  const points = trace.points
  // The run average swings over its first beats; it joins the domain once five minutes are in.
  const steady = points.flatMap(point =>
    [
      runHeartCostValue(point, '300'),
      point.elapsedS >= 300 ? point.cumulativeMetersPerBeat : null,
    ].filter((value): value is number => value != null && Number.isFinite(value)),
  )
  const short = points
    .map(point => runHeartCostValue(point, '30'))
    .filter((value): value is number => value != null && Number.isFinite(value))
    .sort((a, b) => a - b)
  if (short.length < 2) return null
  const text = (key: string): string => triText(f.presentation.locale, key)
  const totalElapsedS = Math.max(d.elapsedTimeS, points.at(-1)?.elapsedS ?? 0)
  if (!(totalElapsedS > 0)) return null
  const height = 30
  const bottom = 28
  // 30 s values spike at stops and surges; their 5–95% band sets the axis with the steady series.
  const low = Math.min(quantile(short, 0.05), ...steady)
  const high = Math.max(quantile(short, 0.95), ...steady)
  const pad = Math.max(0.02, (high - low) * 0.08)
  const costTicks = niceTicks(Math.max(0, low - pad), high + pad, 3)
  const costFloor = Math.min(low - pad, costTicks[0])
  const costCeiling = Math.max(high + pad, costTicks.at(-1) ?? high)
  const axis = elapsedAxis(0, totalElapsedS, trace.pauses)
  const px = (elapsedS: number): number => axis.fraction(elapsedS) * 100
  const costY = (value: number): number =>
    bottom -
    ((Math.min(costCeiling, Math.max(costFloor, value)) - costFloor) / (costCeiling - costFloor)) *
      (bottom - 1)
  const core = activityThermalTracePoints(d).filter(
    (point): point is typeof point & { coreTemperatureC: number } =>
      point.coreTemperatureC != null && Number.isFinite(point.coreTemperatureC),
  )
  const backdrop: 'core-temperature' | 'elevation' =
    core.length >= 2 ? 'core-temperature' : 'elevation'
  const backdropPoints: { elapsedS: number; value: number }[] =
    backdrop === 'core-temperature'
      ? core.map(point => ({ elapsedS: point.elapsedS, value: point.coreTemperatureC }))
      : points.flatMap(point =>
          point.elevationM != null && Number.isFinite(point.elevationM)
            ? [
                {
                  elapsedS: point.elapsedS,
                  value: elevationValue(f.presentation, point.elevationM),
                },
              ]
            : [],
        )
  const backdropValues = backdropPoints.map(point => point.value)
  const backdropMin = backdropValues.length ? Math.min(...backdropValues) : 0
  const backdropSpan = backdropValues.length
    ? Math.max(backdrop === 'core-temperature' ? 0.5 : 1, Math.max(...backdropValues) - backdropMin)
    : 1
  const backdropY = (value: number): number =>
    bottom - ((value - backdropMin) / backdropSpan) * (bottom - 1)
  const windSource = preferredActivityWind(d)
  const wrap = f.el('div', 'tri-elev-wrap tri-cycling-power tri-run-heart-cost', undefined, {
    'data-tri-trace': 'run-heart-cost',
    'data-run-cost-window': '300',
    'data-run-cost-backdrop': backdrop,
    'data-run-cost-lag-s': String(trace.heartRateLagS),
    'data-cycling-power-wind-provider': windSource.provider,
  })
  const cap = f.el('div', 'tri-elev-cap')
  f.add(
    cap,
    f.el('span', 'tri-elev-d', text('heart rate cost over terrain'), {
      'data-gloss': '',
      'data-gloss-def': text('heart rate cost definition'),
      tabindex: '0',
    }),
  )
  const controls = f.el('div', 'tri-chart-controls tri-run-cost-controls', undefined, {
    role: 'group',
    'aria-label': text('heart rate cost averaging window'),
  })
  for (const window of ['30', '300'])
    f.add(
      controls,
      f.el('button', 'tri-run-cost-window tri-curve-range', window === '30' ? '30 s' : '5 min', {
        type: 'button',
        'data-run-cost-window': window,
        'aria-pressed': String(window === '300'),
        'aria-label': text(window === '30' ? '30 s average' : '5 min average'),
      }),
    )
  const legend = f.el('div', 'tri-cycling-power-legend')
  f.add(
    legend,
    f.el('span', 'tri-cycling-power-key tri-cycling-power-key--trailing', text('5 min average'), {
      'data-run-cost-label': '',
    }),
    f.el('span', 'tri-cycling-power-key tri-cycling-power-key--cumulative', text('run average')),
    f.el(
      'span',
      'tri-cycling-power-key tri-run-cost-key--backdrop',
      text(backdrop === 'core-temperature' ? 'core temperature' : 'elevation'),
    ),
  )
  const graph = f.svg('svg', {
    class: 'tri-elev tri-cycling-power-plot tri-run-cost-plot',
    viewBox: `0 0 100 ${height}`,
    preserveAspectRatio: 'none',
    tabindex: '0',
    role: 'slider',
    'aria-label': text('heart rate cost over terrain'),
    'aria-valuemin': 0,
    'aria-valuemax': totalElapsedS,
    'aria-valuenow': 0,
    'aria-valuetext': zoneClock(0),
    'data-domain-start-elapsed-s': 0,
    'data-domain-end-elapsed-s': totalElapsedS,
    ...(axis.pauses.length ? { 'data-domain-pauses': formatElapsedPauses(axis.pauses) } : {}),
  })
  for (const value of costTicks)
    f.add(
      graph,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, x2: 100, y1: costY(value), y2: costY(value) }),
    )
  if (backdropPoints.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-cycling-power-elevation',
        d: elapsedSeriesPath(
          backdropPoints,
          point => point.value,
          px,
          backdropY,
          bottom,
          true,
          axis.crossesPause,
        ),
      }),
      f.svg('path', {
        class: 'tri-cycling-power-elevation-line',
        d: elapsedSeriesPath(
          backdropPoints,
          point => point.value,
          px,
          backdropY,
          bottom,
          false,
          axis.crossesPause,
        ),
      }),
    )
  addElapsedBreaks(f, graph, axis, height)
  const windSegments = addWindStrip(f, graph, windSource, px, totalElapsedS)
  const bounds = selection
    ? analysisChartSelectionBounds(selection, {
        startElapsedS: 0,
        endElapsedS: totalElapsedS,
        pauses: axis.pauses,
      })
    : undefined
  f.add(graph, buildAnalysisSelectionRect(f, height, bounds))
  for (const series of [
    { name: '30', pick: (point: RunHeartCostPoint) => runHeartCostValue(point, '30') },
    { name: '300', pick: (point: RunHeartCostPoint) => runHeartCostValue(point, '300') },
    { name: 'cumulative', pick: (point: RunHeartCostPoint) => point.cumulativeMetersPerBeat },
  ])
    f.add(
      graph,
      f.svg('path', {
        class: `tri-cycling-power-line${series.name === 'cumulative' ? ' tri-cycling-power-line--cumulative' : ''}`,
        d: elapsedSeriesPath(points, series.pick, px, costY, bottom, false, axis.crossesPause),
        'data-run-cost-series': series.name,
        ...(series.name === '30' ? { hidden: '' } : {}),
      }),
    )
  f.add(graph, f.svg('line', { class: 'tri-elev-cursor', x1: 0, x2: 0, y1: 0, y2: height }))
  addWindLegend(f, legend, windSource, windSegments)
  f.add(cap, legend, controls)
  const backdropLabel = (value: number): string =>
    backdrop === 'core-temperature'
      ? formatThermalTemperature(f.presentation, value)
      : `${Math.round(value)} ${isImperial(f.presentation) ? 'ft' : 'm'}`
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      graph,
      costTicks.map(value => ({ label: `${value.toFixed(2)} m/b`, vbY: costY(value) })),
      height,
      elapsedActivityXTicks(totalElapsedS, axis.elapsedAt),
      true,
      { top: 0, bottom: height },
      [],
      backdropPoints.length >= 2
        ? [backdropMin, backdropMin + backdropSpan / 2, backdropMin + backdropSpan].map(value => ({
            label: backdropLabel(value),
            vbY: backdropY(value),
          }))
        : [],
    ),
  )
  return wrap
}
