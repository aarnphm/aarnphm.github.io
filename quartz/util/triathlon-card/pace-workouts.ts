import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { analysisRangeAttrs, analysisRangeMetrics } from './analysis-presentation'
import { clock, isImperial, KM_TO_MI, runPaceSeconds } from './metric-text'
import {
  buildWorkoutElevation,
  paceDelta,
  runLapSplits,
  runWorkoutLaps,
  runWorkoutPaceAxis,
  swimWorkoutLaps,
  workoutElevationPaths,
  workoutLapTimeline,
} from './workout-lap-data'

export const buildSwimWorkoutAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  if (d.sport !== 'swim') return null
  const laps = swimWorkoutLaps(d)
  if (laps.length === 0) return null
  const timeline = workoutLapTimeline(laps)
  const paceAxis = runWorkoutPaceAxis(laps.map(lap => lap.paceS))
  const paceSpan = paceAxis.max - paceAxis.min
  const speeds = laps.map(lap => lap.speedKph)
  const minSpeedKph = Math.min(...speeds)
  const maxSpeedKph = Math.max(...speeds)
  const speedSpan = maxSpeedKph - minSpeedKph
  const durationS = laps.reduce(
    (total, lap) => total + (lap.range.movingTimeS ?? lap.range.durationS),
    0,
  )
  const distanceKm = laps.reduce((total, lap) => total + lap.range.distanceKm, 0)
  const averagePaceS = 360 / ((distanceKm / durationS) * 3600)
  const fastestPaceS = Math.min(...laps.map(lap => lap.paceS))
  const slowestPaceS = Math.max(...laps.map(lap => lap.paceS))
  const elevationPaths = workoutElevationPaths(d, timeline)

  const wrap = f.el('section', 'tri-workout tri-swim-workout', undefined, {
    'aria-label': 'Swim workout analysis',
    'data-swim-workout-elevation': String(elevationPaths != null),
  })
  const head = f.el('div', 'tri-workout-head tri-swim-workout-head')
  const stats = f.el('div', 'tri-workout-stats tri-swim-workout-stats')
  f.add(
    stats,
    f.el('span', undefined, `fastest ${clock(fastestPaceS)} /100m`),
    f.el('span', undefined, `avg ${clock(averagePaceS)} /100m`),
    f.el('span', undefined, `slowest ${clock(slowestPaceS)} /100m`),
  )
  f.add(head, stats)

  const chart = f.el('div', 'tri-workout-chart tri-swim-workout-chart')
  const yAxis = f.el('div', 'tri-workout-y-axis tri-swim-workout-y-axis', undefined, {
    'aria-hidden': 'true',
  })
  const viewport = f.el('div', 'tri-workout-viewport tri-swim-workout-viewport')
  const plot = f.el('div', 'tri-workout-plot tri-swim-workout-plot', undefined, {
    'data-site-cursor-line': '',
    style: `--tri-swim-workout-laps:${laps.length}`,
  })
  if (elevationPaths) f.add(plot, buildWorkoutElevation(f, elevationPaths))
  const grid = f.el('div', 'tri-workout-grid tri-swim-workout-grid', undefined, {
    'aria-hidden': 'true',
  })
  for (const tick of paceAxis.ticks) {
    const top = ((tick - paceAxis.min) / paceSpan) * 100
    f.add(
      yAxis,
      f.el('span', 'tri-workout-y-tick tri-swim-workout-y-tick', clock(tick), {
        style: `top:${top.toFixed(3)}%`,
      }),
    )
    f.add(
      grid,
      f.el('span', 'tri-workout-gridline tri-swim-workout-gridline', undefined, {
        style: `top:${top.toFixed(3)}%`,
      }),
    )
  }
  const bars = f.el('div', 'tri-swim-workout-bars')
  for (const [index, lap] of laps.entries()) {
    const metrics = analysisRangeMetrics(f.presentation, d, lap.range)
    const attrs = analysisRangeAttrs(lap.range)
    const span = timeline.spans[index]
    const start = (span.startS / timeline.durationS) * 100
    const end = (span.endS / timeline.durationS) * 100
    const height = Math.max(3, ((paceAxis.max - lap.paceS) / paceSpan) * 100)
    const intensity = speedSpan > 0 ? 0.42 + ((lap.speedKph - minSpeedKph) / speedSpan) * 0.5 : 0.72
    attrs['aria-pressed'] = 'false'
    attrs['aria-label'] = `${lap.range.label}, ${metrics.join(', ')}`
    attrs.style = `--tri-swim-workout-start:${start.toFixed(3)}%;--tri-swim-workout-width:${Math.max(0, end - start).toFixed(3)}%;--tri-swim-workout-height:${height.toFixed(3)}%;--tri-swim-workout-opacity:${intensity.toFixed(3)}`
    const button = f.el('button', 'tri-swim-workout-lap tri-workout-lap', undefined, attrs)
    const column = f.el('span', 'tri-swim-workout-column', undefined, { 'aria-hidden': 'true' })
    f.add(
      column,
      f.el('span', 'tri-swim-workout-bar tri-workout-lap-bar', undefined, {
        'aria-hidden': 'true',
      }),
      f.el('span', 'tri-swim-workout-pace tri-workout-lap-tooltip', `${clock(lap.paceS)} /100m`, {
        'aria-hidden': 'true',
      }),
    )
    f.add(
      button,
      column,
      f.el('span', 'tri-swim-workout-label', `${lap.index}`, { 'aria-hidden': 'true' }),
    )
    f.add(bars, button)
  }
  f.add(plot, grid, bars)
  f.add(viewport, plot)
  f.add(chart, yAxis, viewport)
  f.add(wrap, head, chart)
  return wrap
}

export const buildRunWorkoutAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  if (d.sport !== 'run') return null
  const laps = runWorkoutLaps(f.presentation, d)
  if (laps.length === 0) return null
  const timeline = workoutLapTimeline(laps)
  const elevationPaths = workoutElevationPaths(d, timeline)
  const imperial = isImperial(f.presentation)
  const paceUnit = imperial ? '/mi' : '/km'
  const paceAxis = runWorkoutPaceAxis(laps.map(lap => lap.paceS))
  const paceSpan = paceAxis.max - paceAxis.min
  const speeds = laps.map(lap => lap.speedKph)
  const minSpeedKph = Math.min(...speeds)
  const maxSpeedKph = Math.max(...speeds)
  const speedSpan = maxSpeedKph - minSpeedKph
  const movingTimeS = laps.reduce(
    (total, lap) => total + (lap.range.distanceKm / lap.speedKph) * 3600,
    0,
  )
  const distanceKm = laps.reduce((total, lap) => total + lap.range.distanceKm, 0)
  const averageSpeedKph = (distanceKm / movingTimeS) * 3600
  const fastestPaceS = Math.min(...laps.map(lap => lap.paceS))
  const slowestPaceS = Math.max(...laps.map(lap => lap.paceS))

  const wrap = f.el('section', 'tri-workout tri-run-workout', undefined, {
    'aria-label': 'Run workout analysis',
    'data-run-workout-elevation': String(elevationPaths != null),
  })
  const head = f.el('div', 'tri-workout-head tri-run-workout-head')
  const stats = f.el('div', 'tri-workout-stats tri-run-workout-stats')
  f.add(
    stats,
    f.el('span', undefined, `fastest ${clock(fastestPaceS)} ${paceUnit}`),
    f.el(
      'span',
      undefined,
      `avg ${clock(runPaceSeconds(f.presentation, averageSpeedKph))} ${paceUnit}`,
    ),
    f.el('span', undefined, `slowest ${clock(slowestPaceS)} ${paceUnit}`),
  )
  f.add(head, stats)

  const chart = f.el('div', 'tri-workout-chart tri-run-workout-chart')
  const yAxis = f.el('div', 'tri-workout-y-axis tri-run-workout-y-axis', undefined, {
    'aria-hidden': 'true',
  })
  const viewport = f.el('div', 'tri-workout-viewport tri-run-workout-viewport')
  const plot = f.el('div', 'tri-workout-plot tri-run-workout-plot', undefined, {
    'data-site-cursor-line': '',
    style: `--tri-run-workout-laps:${laps.length}`,
  })
  if (elevationPaths) f.add(plot, buildWorkoutElevation(f, elevationPaths))
  const grid = f.el('div', 'tri-workout-grid tri-run-workout-grid', undefined, {
    'aria-hidden': 'true',
  })
  const bars = f.el('div', 'tri-run-workout-bars')
  for (const tick of paceAxis.ticks) {
    const top = ((tick - paceAxis.min) / paceSpan) * 100
    f.add(
      yAxis,
      f.el('span', 'tri-workout-y-tick tri-run-workout-y-tick', clock(tick), {
        style: `top:${top.toFixed(3)}%`,
      }),
    )
    f.add(
      grid,
      f.el('span', 'tri-workout-gridline tri-run-workout-gridline', undefined, {
        style: `top:${top.toFixed(3)}%`,
      }),
    )
  }
  for (const [index, lap] of laps.entries()) {
    const metrics = analysisRangeMetrics(f.presentation, d, lap.range)
    const attrs = analysisRangeAttrs(lap.range)
    const span = timeline.spans[index]
    const start = (span.startS / timeline.durationS) * 100
    const end = (span.endS / timeline.durationS) * 100
    const height = Math.max(3, ((paceAxis.max - lap.paceS) / paceSpan) * 100)
    const intensity = speedSpan > 0 ? 0.42 + ((lap.speedKph - minSpeedKph) / speedSpan) * 0.5 : 0.72
    attrs['aria-pressed'] = 'false'
    attrs['aria-label'] = `${lap.range.label}, ${metrics.join(', ')}`
    attrs.style = `--tri-run-workout-start:${start.toFixed(3)}%;--tri-run-workout-width:${Math.max(0, end - start).toFixed(3)}%;--tri-run-workout-height:${height.toFixed(3)}%;--tri-run-workout-opacity:${intensity.toFixed(3)}`
    const button = f.el('button', 'tri-run-workout-lap tri-workout-lap', undefined, attrs)
    const column = f.el('span', 'tri-run-workout-column', undefined, { 'aria-hidden': 'true' })
    f.add(
      column,
      f.el('span', 'tri-run-workout-bar tri-workout-lap-bar', undefined, { 'aria-hidden': 'true' }),
      f.el(
        'span',
        'tri-run-workout-pace tri-workout-lap-tooltip',
        `${clock(lap.paceS)} ${paceUnit}`,
        { 'aria-hidden': 'true' },
      ),
    )
    f.add(
      button,
      column,
      f.el('span', 'tri-run-workout-label', `${lap.index}`, { 'aria-hidden': 'true' }),
    )
    f.add(bars, button)
  }
  f.add(plot, grid, bars)
  f.add(viewport, plot)
  f.add(chart, yAxis, viewport)
  f.add(wrap, head, chart)
  return wrap
}

export const buildRunLapSplits = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  if (d.sport !== 'run') return null
  const imperial = isImperial(f.presentation)
  const splits = runLapSplits(f.presentation, d)
  const wrap = f.el('section', 'tri-run-splits', undefined, { 'aria-label': 'Run lap splits' })
  const head = f.el('div', 'tri-run-splits-head')
  if (splits.length === 0) {
    const columns = f.el('div', 'tri-run-splits-columns', undefined, { 'aria-hidden': 'true' })
    const list = f.el('div', 'tri-run-splits-list')
    f.add(list, f.el('span', 'tri-run-splits-empty', 'no lap found'))
    f.add(wrap, head, columns, list)
    return wrap
  }
  const maxSpeedKph = Math.max(...splits.map(split => split.speedKph))
  const totalDistanceKm = splits.reduce((total, split) => total + split.range.distanceKm, 0)
  const totalDurationS = splits.reduce(
    (total, split) => total + (split.range.movingTimeS ?? split.range.durationS),
    0,
  )
  const averageSpeedKph = (totalDistanceKm / totalDurationS) * 3600
  const averagePct = Math.max(0, Math.min(100, (averageSpeedKph / maxSpeedKph) * 100))
  const paceUnit = imperial ? '/mi' : '/km'
  f.add(
    head,
    f.el(
      'span',
      'tri-run-splits-average',
      `avg ${clock(runPaceSeconds(f.presentation, averageSpeedKph))} ${paceUnit}`,
    ),
  )
  const columns = f.el('div', 'tri-run-splits-columns', undefined, { 'aria-hidden': 'true' })
  f.add(
    columns,
    f.el('span', undefined, 'split'),
    f.el('span', undefined, imperial ? 'mi' : 'km'),
    f.el('span', undefined, 'pace'),
    f.el('span', undefined, '+/−'),
  )
  const list = f.el('div', 'tri-run-splits-list')
  for (const split of splits) {
    const metrics = analysisRangeMetrics(f.presentation, d, split.range)
    const attrs = analysisRangeAttrs(split.range)
    const delta = paceDelta(split.deltaS)
    attrs['aria-pressed'] = 'false'
    attrs['aria-label'] =
      `${split.range.label}, ${metrics.join(', ')}, ${delta === '—' ? 'first lap' : `${delta} versus previous lap`}`
    attrs.style = `--tri-run-split-width:${Math.max(24, (split.speedKph / maxSpeedKph) * 100).toFixed(3)}%;--tri-run-split-average:${averagePct.toFixed(3)}%`
    const button = f.el('button', 'tri-run-split', undefined, attrs)
    const track = f.el('span', 'tri-run-split-track')
    f.add(
      track,
      f.el('span', 'tri-run-split-fill', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-run-split-average-marker', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-run-split-pace', `${clock(split.paceS)} ${paceUnit}`),
    )
    const deltaClass =
      split.deltaS == null || Math.round(split.deltaS) === 0
        ? 'tri-run-split-delta'
        : `tri-run-split-delta tri-run-split-delta--${split.deltaS > 0 ? 'faster' : 'slower'}`
    f.add(
      button,
      f.el('span', 'tri-run-split-lap', `${split.index}`),
      f.el(
        'span',
        'tri-run-split-distance',
        imperial
          ? (split.range.distanceKm * KM_TO_MI).toFixed(2)
          : split.range.distanceKm.toFixed(2),
      ),
      track,
      f.el('span', deltaClass, delta),
    )
    f.add(list, button)
  }
  f.add(wrap, head, columns, list)
  return wrap
}
