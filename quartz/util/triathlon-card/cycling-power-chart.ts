import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { preferredActivityWind } from '../activity-wind'
import { triText } from '../triathlon-i18n'
import { axisFrame, elapsedActivityXTicks, niceTicks } from './chart-geometry'
import {
  analysisChartSelectionBounds,
  buildAnalysisSelectionRect,
  type AnalysisInterval,
} from './chart-selection'
import { addWindLegend, addWindStrip, elapsedSeriesPath } from './elapsed-chart'
import { elevationValue, isImperial, zoneClock } from './metric-text'

export const buildCyclingPowerChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: AnalysisInterval | null,
): N | null => {
  const trace = d.cyclingPowerTrace
  if (d.sport !== 'bike' || !trace || trace.points.length < 2) return null
  const points = trace.points
  const powers = points.flatMap(point =>
    [point.power30sWatts, point.power5mWatts, point.cumulativePowerWatts].filter(
      (value): value is number => value != null && Number.isFinite(value),
    ),
  )
  if (powers.length < 2) return null
  const text = (key: string): string => triText(f.presentation.locale, key)
  const totalElapsedS = Math.max(d.elapsedTimeS, points.at(-1)?.elapsedS ?? 0)
  if (!(totalElapsedS > 0)) return null
  const height = 30
  const bottom = 28
  const maxPower = Math.max(10, ...powers) * 1.05
  const powerTicks = niceTicks(0, maxPower, 3)
  const powerCeiling = Math.max(maxPower, powerTicks.at(-1) ?? maxPower)
  const px = (elapsedS: number): number => (elapsedS / totalElapsedS) * 100
  const powerY = (watts: number): number => bottom - (watts / powerCeiling) * (bottom - 1)
  const elevations = points.flatMap(point =>
    point.elevationM != null && Number.isFinite(point.elevationM)
      ? [elevationValue(f.presentation, point.elevationM)]
      : [],
  )
  const minElevation = elevations.length ? Math.min(...elevations) : 0
  const elevationSpan = elevations.length ? Math.max(1, Math.max(...elevations) - minElevation) : 1
  const elevationY = (value: number): number =>
    bottom - ((value - minElevation) / elevationSpan) * (bottom - 1)
  const path = (
    pick: (point: (typeof points)[number]) => number | null,
    py: (value: number) => number,
    area = false,
  ): string => elapsedSeriesPath(points, pick, px, py, bottom, area)
  const windSource = preferredActivityWind(d)
  const wrap = f.el('div', 'tri-elev-wrap tri-cycling-power', undefined, {
    'data-tri-trace': 'cycling-power',
    'data-cycling-power-window': '30',
    'data-cycling-power-source': trace.source,
    'data-cycling-power-terrain-source': trace.terrainSource,
    'data-cycling-power-wind-provider': windSource.provider,
  })
  const cap = f.el('div', 'tri-elev-cap')
  f.add(
    cap,
    f.el('span', 'tri-elev-d', text('power over terrain'), {
      'data-gloss': '',
      'data-gloss-def': text('recorded power averages'),
      tabindex: '0',
    }),
  )
  const controls = f.el('div', 'tri-chart-controls tri-cycling-power-controls', undefined, {
    role: 'group',
    'aria-label': text('power averaging window'),
  })
  for (const window of ['30', '300'])
    f.add(
      controls,
      f.el(
        'button',
        'tri-cycling-power-window tri-curve-range',
        window === '30' ? '30 s' : '5 min',
        {
          type: 'button',
          'data-cycling-power-window': window,
          'aria-pressed': String(window === '30'),
          'aria-label': text(window === '30' ? '30 s average' : '5 min average'),
        },
      ),
    )
  const legend = f.el('div', 'tri-cycling-power-legend')
  f.add(
    legend,
    f.el('span', 'tri-cycling-power-key tri-cycling-power-key--trailing', text('30 s average'), {
      'data-cycling-power-label': '',
    }),
    f.el('span', 'tri-cycling-power-key tri-cycling-power-key--cumulative', text('ride average')),
  )
  const graph = f.svg('svg', {
    class: 'tri-elev tri-cycling-power-plot',
    viewBox: `0 0 100 ${height}`,
    preserveAspectRatio: 'none',
    tabindex: '0',
    role: 'slider',
    'aria-label': text('power over terrain'),
    'aria-valuemin': 0,
    'aria-valuemax': totalElapsedS,
    'aria-valuenow': 0,
    'aria-valuetext': zoneClock(0),
    'data-domain-start-elapsed-s': 0,
    'data-domain-end-elapsed-s': totalElapsedS,
  })
  for (const value of powerTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-elev-grid',
        x1: 0,
        x2: 100,
        y1: powerY(value),
        y2: powerY(value),
      }),
    )
  if (elevations.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-cycling-power-elevation',
        d: path(
          point =>
            point.elevationM == null ? null : elevationValue(f.presentation, point.elevationM),
          elevationY,
          true,
        ),
      }),
      f.svg('path', {
        class: 'tri-cycling-power-elevation-line',
        d: path(
          point =>
            point.elevationM == null ? null : elevationValue(f.presentation, point.elevationM),
          elevationY,
        ),
      }),
    )
  const windSegments = addWindStrip(f, graph, windSource, px, totalElapsedS)
  const bounds = selection
    ? analysisChartSelectionBounds(selection, { startElapsedS: 0, endElapsedS: totalElapsedS })
    : undefined
  f.add(graph, buildAnalysisSelectionRect(f, height, bounds))
  for (const series of [
    { name: '30', pick: (point: (typeof points)[number]) => point.power30sWatts },
    { name: '300', pick: (point: (typeof points)[number]) => point.power5mWatts },
    { name: 'cumulative', pick: (point: (typeof points)[number]) => point.cumulativePowerWatts },
  ])
    f.add(
      graph,
      f.svg('path', {
        class: `tri-cycling-power-line${series.name === 'cumulative' ? ' tri-cycling-power-line--cumulative' : ''}`,
        d: path(series.pick, powerY),
        'data-cycling-power-series': series.name,
        ...(series.name === '300' ? { hidden: '' } : {}),
      }),
    )
  f.add(graph, f.svg('line', { class: 'tri-elev-cursor', x1: 0, x2: 0, y1: 0, y2: height }))
  addWindLegend(f, legend, windSource, windSegments)
  f.add(cap, legend, controls)
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      graph,
      powerTicks.map(value => ({ label: `${Math.round(value)} W`, vbY: powerY(value) })),
      height,
      elapsedActivityXTicks(totalElapsedS),
      true,
      { top: 0, bottom: height },
      [],
      elevations.length >= 2
        ? [minElevation, minElevation + elevationSpan / 2, minElevation + elevationSpan].map(
            value => ({
              label: `${Math.round(value)} ${isImperial(f.presentation) ? 'ft' : 'm'}`,
              vbY: elevationY(value),
            }),
          )
        : [],
    ),
  )
  return wrap
}
