import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { runPaceZoneRange } from '../run-pace-zones'
import { validAnalysisRanges } from './analysis-data'
import { niceStep } from './chart-geometry'
import { clock, isImperial, KM_TO_MI, runPaceSeconds } from './metric-text'

type CyclingWorkoutLap = { range: ActivityAnalysisRange; index: number; powerWatts: number | null }

export const cyclingWorkoutLaps = (d: StravaActivityDetail): CyclingWorkoutLap[] =>
  validAnalysisRanges(d)
    .filter(range => range.kind === 'lap')
    .sort((left, right) => left.startElapsedS - right.startElapsedS)
    .map((range, index) => ({
      range,
      index: index + 1,
      powerWatts:
        range.averageWatts != null && Number.isFinite(range.averageWatts) && range.averageWatts >= 0
          ? range.averageWatts
          : null,
    }))

export const cyclingWorkoutPowerSummary = (laps: readonly CyclingWorkoutLap[]) => {
  const poweredLaps = laps.filter(
    (lap): lap is CyclingWorkoutLap & { powerWatts: number } => lap.powerWatts != null,
  )
  if (!poweredLaps.length) return null
  const highest = Math.max(...poweredLaps.map(lap => lap.powerWatts))
  const lowest = Math.min(...poweredLaps.map(lap => lap.powerWatts))
  const weighted = poweredLaps.reduce(
    (summary, lap) => {
      const durationS =
        lap.range.movingTimeS != null &&
        Number.isFinite(lap.range.movingTimeS) &&
        lap.range.movingTimeS > 0
          ? lap.range.movingTimeS
          : lap.range.durationS
      return {
        wattsSeconds: summary.wattsSeconds + lap.powerWatts * durationS,
        durationS: summary.durationS + durationS,
      }
    },
    { wattsSeconds: 0, durationS: 0 },
  )
  const step = niceStep(Math.max(1, highest), 4)
  return {
    highest,
    lowest,
    average: weighted.wattsSeconds / weighted.durationS,
    maximum: Math.max(step, Math.ceil(highest / step) * step),
  }
}

type WorkoutElevationPaths = { area: string; line: string }

export const workoutLapTimeline = (laps: readonly { range: ActivityAnalysisRange }[]) => {
  let durationS = 0
  const spans = laps.map(({ range }) => {
    const startS = durationS
    const activeS = range.movingTimeS
    durationS +=
      activeS != null && Number.isFinite(activeS) && activeS > 0 ? activeS : range.durationS
    return { range, startS, endS: durationS }
  })
  return {
    durationS,
    spans,
    project: (point: StravaActivityDetail['route'][number]): number | null => {
      const span = spans.find(
        ({ range }) => point.elapsedS >= range.startElapsedS && point.elapsedS <= range.endElapsedS,
      )
      if (!span || !Number.isFinite(point.d)) return null
      const { range, startS, endS } = span
      // Lap summaries supply active duration, but no per-record timer. Distance
      // places terrain within that lap without allocating width to stopped time.
      const fraction = Math.max(
        0,
        Math.min(
          1,
          (point.d - range.startDistanceKm) / (range.endDistanceKm - range.startDistanceKm),
        ),
      )
      return ((startS + fraction * (endS - startS)) / durationS) * 100
    },
  }
}

export const workoutElevationPaths = (
  d: StravaActivityDetail,
  timeline: ReturnType<typeof workoutLapTimeline>,
): WorkoutElevationPaths | null => {
  if (d.sport === 'swim' && d.swimLocation === 'pool') return null
  const requireGps = d.sport === 'run' || d.sport === 'swim'
  if (requireGps && !d.mapRoute.some(segment => segment.length >= 2)) return null
  const route = d.route.flatMap(point => {
    if (
      !Number.isFinite(point.elapsedS) ||
      !Number.isFinite(point.alt) ||
      (requireGps && (!Number.isFinite(point.lat) || !Number.isFinite(point.lng)))
    )
      return []
    const x = timeline.project(point)
    return x == null ? [] : [{ x, alt: point.alt }]
  })
  if (route.length < 2 || timeline.durationS <= 0) return null
  const minAltitude = Math.min(...route.map(point => point.alt))
  const maxAltitude = Math.max(...route.map(point => point.alt))
  const altitudeSpan = Math.max(1, maxAltitude - minAltitude)
  const points = route.map(point => {
    const y = 100 - ((point.alt - minAltitude) / altitudeSpan) * 100
    return `${point.x.toFixed(3)} ${y.toFixed(3)}`
  })
  return {
    area: `M ${route[0].x.toFixed(3)} 100 L ${points.join(' L ')} L ${route[route.length - 1].x.toFixed(3)} 100 Z`,
    line: `M ${points.join(' L ')}`,
  }
}

export const buildWorkoutElevation = <N>(f: TriNodeFactory<N>, paths: WorkoutElevationPaths): N => {
  const elevation = f.svg('svg', {
    class: 'tri-workout-elevation',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'none',
    'aria-hidden': 'true',
  })
  f.add(
    elevation,
    f.svg('path', { class: 'tri-workout-elevation-area', d: paths.area }),
    f.svg('path', { class: 'tri-workout-elevation-line', d: paths.line }),
  )
  return elevation
}

type RunLapSplit = {
  range: ActivityAnalysisRange
  index: number
  speedKph: number
  paceS: number
  deltaS: number | null
}

const projectedRunSplits = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): ActivityAnalysisRange[] => {
  const imperial = isImperial(presentation)
  const source = imperial ? (d.runSplitsStandard ?? []) : (d.runSplitsMetric ?? [])
  const ranges: ActivityAnalysisRange[] = []
  let startDistanceKm = 0
  let startElapsedS = 0
  for (const split of source) {
    if (
      !Number.isFinite(split.distanceKm) ||
      split.distanceKm <= 0 ||
      !Number.isFinite(split.elapsedTimeS) ||
      split.elapsedTimeS <= 0 ||
      !Number.isFinite(split.movingTimeS) ||
      split.movingTimeS <= 0 ||
      !Number.isFinite(split.averageSpeedKph) ||
      split.averageSpeedKph <= 0
    )
      continue
    const endDistanceKm = startDistanceKm + split.distanceKm
    const endElapsedS = startElapsedS + split.elapsedTimeS
    ranges.push({
      kind: 'lap',
      id: `split:${imperial ? 'standard' : 'metric'}:${split.split}`,
      label: `Split ${split.split}`,
      startElapsedS,
      endElapsedS,
      startDistanceKm,
      endDistanceKm,
      durationS: split.movingTimeS,
      distanceKm: split.distanceKm,
      elevationGainM: null,
      averageSpeedKph: split.averageSpeedKph,
      averageHeartRate: null,
      averageWatts: null,
      averageCadence: null,
    })
    startDistanceKm = endDistanceKm
    startElapsedS = endElapsedS
  }
  return ranges
}

export const runLapSplits = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): RunLapSplit[] => {
  const nativeSplits = projectedRunSplits(presentation, d)
  const ranges =
    nativeSplits.length > 0
      ? nativeSplits
      : validAnalysisRanges(d).filter(candidate => candidate.kind === 'lap')
  const splits: RunLapSplit[] = []
  let previousPaceS: number | null = null
  for (const [index, range] of ranges.entries()) {
    const speedKph =
      range.averageSpeedKph != null && range.averageSpeedKph > 0
        ? range.averageSpeedKph
        : (range.distanceKm / (range.movingTimeS ?? range.durationS)) * 3600
    if (!Number.isFinite(speedKph) || speedKph <= 0) continue
    const paceS = runPaceSeconds(presentation, speedKph)
    splits.push({
      range,
      index: index + 1,
      speedKph,
      paceS,
      deltaS: previousPaceS == null ? null : previousPaceS - paceS,
    })
    previousPaceS = paceS
  }
  return splits
}

export const paceDelta = (seconds: number | null): string => {
  if (seconds == null) return '—'
  const rounded = Math.round(seconds)
  if (rounded === 0) return '0:00'
  return `${rounded > 0 ? '+' : '−'}${clock(Math.abs(rounded))}`
}

export const runWorkoutLaps = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): RunLapSplit[] => {
  const laps: RunLapSplit[] = []
  for (const [index, range] of validAnalysisRanges(d)
    .filter(candidate => candidate.kind === 'lap')
    .sort((left, right) => left.startElapsedS - right.startElapsedS)
    .entries()) {
    const speedKph =
      range.averageSpeedKph != null && range.averageSpeedKph > 0
        ? range.averageSpeedKph
        : (range.distanceKm / (range.movingTimeS ?? range.durationS)) * 3600
    if (!Number.isFinite(speedKph) || speedKph <= 0) continue
    laps.push({
      range,
      index: index + 1,
      speedKph,
      paceS: runPaceSeconds(presentation, speedKph),
      deltaS: null,
    })
  }
  return laps
}

export const runWorkoutPaceAxis = (
  paces: number[],
): { min: number; max: number; ticks: number[] } => {
  const fastest = Math.min(...paces)
  const slowest = Math.max(...paces)
  const targetStep = Math.max(1, (slowest - fastest) / 5)
  const step = [30, 60, 120, 180, 300, 600, 900].find(candidate => candidate >= targetStep) ?? 900
  const min = Math.max(30, Math.floor(fastest / step) * step)
  let max = Math.ceil(slowest / step) * step
  if (max <= min) max = min + step
  const ticks: number[] = []
  for (let value = min; value <= max; value += step) ticks.push(value)
  return { min, max, ticks }
}

type SwimWorkoutLap = {
  range: ActivityAnalysisRange
  index: number
  speedKph: number
  paceS: number
}

export const swimWorkoutLaps = (d: StravaActivityDetail): SwimWorkoutLap[] => {
  const laps: SwimWorkoutLap[] = []
  for (const range of validAnalysisRanges(d)
    .filter(candidate => candidate.kind === 'lap')
    .sort((left, right) => left.startElapsedS - right.startElapsedS)) {
    const durationS = range.movingTimeS ?? range.durationS
    const speedKph =
      range.averageSpeedKph != null && range.averageSpeedKph > 0
        ? range.averageSpeedKph
        : (range.distanceKm / durationS) * 3600
    if (!Number.isFinite(speedKph) || speedKph <= 0) continue
    laps.push({ range, index: laps.length + 1, speedKph, paceS: 360 / speedKph })
  }
  return laps
}

export const runPaceZoneRangeText = (
  presentation: TriathlonPresentation,
  boundsSPerKm: readonly number[],
  index: number,
): string => {
  const range = runPaceZoneRange(boundsSPerKm, index)
  if (!range) return '—'
  const imperial = isImperial(presentation)
  const scale = imperial ? KM_TO_MI : 1
  const unit = imperial ? '/mi' : '/km'
  if (range.fastestSPerKm == null)
    return range.slowestSPerKm == null ? '—' : `<${clock(range.slowestSPerKm / scale)}${unit}`
  if (range.slowestSPerKm == null) return `>${clock(range.fastestSPerKm / scale)}${unit}`
  return `${clock(range.fastestSPerKm / scale)}–${clock(range.slowestSPerKm / scale)}${unit}`
}
