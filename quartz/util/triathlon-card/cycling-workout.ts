import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { analysisRangeAttrs, analysisRangeMetrics } from './analysis-presentation'
import { axisNumber, niceStep, niceTicks } from './chart-geometry'
import { speedKph } from './metric-text'
import {
  buildWorkoutElevation,
  cyclingWorkoutLaps,
  cyclingWorkoutPowerSummary,
  workoutElevationPaths,
  workoutLapTimeline,
} from './workout-lap-data'

export const buildCyclingWorkoutAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  if (d.sport !== 'bike' || d.route.length < 2) return null
  const laps = cyclingWorkoutLaps(d)
  const summary = cyclingWorkoutPowerSummary(laps)
  if (!summary) return null

  const timeline = workoutLapTimeline(laps)
  const elevationPaths = workoutElevationPaths(d, timeline)

  const highestPowerWatts = summary.highest
  const lowestPowerWatts = summary.lowest
  const averagePowerWatts = summary.average
  const step = niceStep(Math.max(1, highestPowerWatts), 4)
  const powerMax = summary.maximum
  const ticks = niceTicks(0, powerMax, 4).filter(value => value > 0)

  const wrap = f.el('section', 'tri-workout tri-cycling-workout', undefined, {
    'aria-label': 'Cycling workout analysis',
  })
  const head = f.el('div', 'tri-workout-head tri-cycling-workout-head')
  const stats = f.el('div', 'tri-workout-stats tri-cycling-workout-stats')
  f.add(
    stats,
    f.el('span', undefined, `highest ${Math.round(highestPowerWatts)} W`),
    f.el('span', undefined, `avg ${Math.round(averagePowerWatts)} W`),
    f.el('span', undefined, `lowest ${Math.round(lowestPowerWatts)} W`),
  )
  f.add(head, stats)

  const chart = f.el('div', 'tri-workout-chart tri-cycling-workout-chart')
  const yAxis = f.el('div', 'tri-workout-y-axis tri-cycling-workout-y-axis', undefined, {
    'aria-hidden': 'true',
  })
  const viewport = f.el('div', 'tri-workout-viewport tri-cycling-workout-viewport')
  const plot = f.el('div', 'tri-workout-plot tri-cycling-workout-plot', undefined, {
    'data-site-cursor-line': '',
    style: `--tri-cycling-workout-laps:${laps.length}`,
  })
  const grid = f.el('div', 'tri-workout-grid tri-cycling-workout-grid', undefined, {
    'aria-hidden': 'true',
  })
  for (const tick of ticks) {
    const top = 100 - (tick / powerMax) * 100
    f.add(
      yAxis,
      f.el('span', 'tri-workout-y-tick tri-cycling-workout-y-tick', axisNumber(tick, step), {
        style: `top:${top.toFixed(3)}%`,
      }),
    )
    f.add(
      grid,
      f.el('span', 'tri-workout-gridline tri-cycling-workout-gridline', undefined, {
        style: `top:${top.toFixed(3)}%`,
      }),
    )
  }
  f.add(yAxis, f.el('span', 'tri-cycling-workout-y-unit', 'W'))

  const averageLine = f.el('span', 'tri-cycling-workout-average-line', undefined, {
    'aria-hidden': 'true',
    style: `top:${(100 - (averagePowerWatts / powerMax) * 100).toFixed(3)}%`,
  })
  const annotations = f.el('div', 'tri-cycling-workout-annotations', undefined, {
    'aria-hidden': 'true',
  })
  f.add(annotations, averageLine)
  const bars = f.el('div', 'tri-cycling-workout-bars')
  for (const [index, lap] of laps.entries()) {
    const metrics = analysisRangeMetrics(f.presentation, d, lap.range)
    const attrs = analysisRangeAttrs(lap.range)
    const span = timeline.spans[index]
    const start = (span.startS / timeline.durationS) * 100
    const end = (span.endS / timeline.durationS) * 100
    const power = lap.powerWatts ?? 0
    attrs['aria-pressed'] = 'false'
    attrs['aria-label'] = `${lap.range.label}, ${metrics.join(', ')}`
    attrs.style = `--tri-cycling-workout-start:${start.toFixed(3)}%;--tri-cycling-workout-width:${Math.max(0, end - start).toFixed(3)}%;--tri-cycling-workout-height:${((power / powerMax) * 100).toFixed(3)}%`
    const button = f.el(
      'button',
      `tri-cycling-workout-lap tri-workout-lap${lap.powerWatts == null ? ' tri-cycling-workout-lap--unavailable' : ''}`,
      undefined,
      attrs,
    )
    const column = f.el('span', 'tri-cycling-workout-column', undefined, { 'aria-hidden': 'true' })
    const lapSpeedKph = lap.range.averageSpeedKph
    const hoverMetrics = [
      lapSpeedKph != null && Number.isFinite(lapSpeedKph) && lapSpeedKph >= 0
        ? speedKph(f.presentation, lapSpeedKph)
        : null,
      lap.powerWatts == null ? '—' : `${Math.round(lap.powerWatts)} W`,
    ].filter(value => value != null)
    f.add(
      column,
      f.el('span', 'tri-cycling-workout-bar tri-workout-lap-bar', undefined, {
        'aria-hidden': 'true',
      }),
      f.el(
        'span',
        'tri-cycling-workout-metrics tri-workout-lap-tooltip',
        hoverMetrics.join(' · '),
        { 'aria-hidden': 'true' },
      ),
    )
    f.add(
      button,
      column,
      f.el('span', 'tri-cycling-workout-label', `${lap.index}`, { 'aria-hidden': 'true' }),
    )
    f.add(bars, button)
  }

  if (elevationPaths) f.add(plot, buildWorkoutElevation(f, elevationPaths))
  f.add(plot, grid, bars, annotations)
  f.add(viewport, plot)
  f.add(chart, yAxis, viewport)
  f.add(wrap, head, chart)
  return wrap
}
