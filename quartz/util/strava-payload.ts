import { readFileSync, statSync } from 'node:fs'
import type { AppleCache, AppleRunningDynamicsSample, AppleSwim } from '../plugins/stores/apple'
import type { GarminCache } from '../plugins/stores/garmin'
import type { OuraCache } from '../plugins/stores/oura'
import type { ManualFuelingEntry } from '../plugins/stores/tracking'
import type { WeatherCache } from '../plugins/stores/weather'
import { ATHLETE } from '../plugins/stores/analytics'
import {
  applyManualFueling,
  buildPayload,
  type SwimActivityInterval,
  type StravaActivityDetail,
  type StravaPayload,
  type StravaRawCache,
} from '../plugins/stores/strava'
import { matchAppleRun } from './apple-run-match'
import { matchAppleSwims } from './apple-swim-match'
import { joinSegments, QUARTZ } from './path'
import { swimPaceSeconds, swimStrokeRate } from './swim-metrics'

export const stravaCachePath = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
export const ouraCachePath = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
export const garminCachePath = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
export const appleCachePath = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')
export const weatherCachePath = joinSegments(QUARTZ, '.quartz-cache', 'weather.json')

const readJson = <T>(path: string): T | null => {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as T
  } catch {
    return null
  }
}

const stamp = (path: string): number => {
  try {
    return statSync(path).mtimeMs
  } catch {
    return 0
  }
}

export function swimActivityIntervals(swim: AppleSwim): {
  durationS: number | null
  intervals: SwimActivityInterval[]
} {
  const raw = (swim.intervals ?? [])
    .slice()
    .sort((a, b) => a.start.localeCompare(b.start) || a.end.localeCompare(b.end))
  const firstStart = raw[0]?.start
  const startMs = Date.parse(swim.start ?? firstStart ?? '')
  if (!Number.isFinite(startMs)) return { durationS: null, intervals: [] }
  const intervalEndMs = raw.reduce((latest, interval) => {
    const end = Date.parse(interval.end)
    return Number.isFinite(end) ? Math.max(latest, end) : latest
  }, startMs)
  const workoutEndMs = Date.parse(swim.end ?? '')
  const endMs = Number.isFinite(workoutEndMs)
    ? Math.max(workoutEndMs, intervalEndMs)
    : intervalEndMs
  const durationS = endMs > startMs ? Math.round((endMs - startMs) / 1000) : null
  let distanceM = 0
  const intervals: SwimActivityInterval[] = []
  for (const interval of raw) {
    const intervalStartMs = Date.parse(interval.start)
    const intervalEndMs = Date.parse(interval.end)
    if (
      !Number.isFinite(intervalStartMs) ||
      !Number.isFinite(intervalEndMs) ||
      intervalEndMs <= intervalStartMs ||
      intervalEndMs <= startMs ||
      !Number.isFinite(interval.distanceM) ||
      interval.distanceM <= 0
    )
      continue
    const timestampDurationS = (intervalEndMs - intervalStartMs) / 1000
    const exportedDurationS =
      interval.durationS != null && Number.isFinite(interval.durationS) && interval.durationS > 0
        ? interval.durationS
        : null
    const activeTimeS = exportedDurationS ?? timestampDurationS
    const startElapsedS =
      Math.round(
        Math.max(
          0,
          interval.startElapsedS != null &&
            Number.isFinite(interval.startElapsedS) &&
            interval.startElapsedS >= 0
            ? interval.startElapsedS
            : (intervalStartMs - startMs) / 1000,
        ) * 10,
      ) / 10
    const exportedEndElapsedS =
      interval.endElapsedS != null &&
      Number.isFinite(interval.endElapsedS) &&
      interval.endElapsedS > startElapsedS
        ? interval.endElapsedS
        : null
    const endElapsedS =
      Math.round(
        Math.max(
          0,
          exportedEndElapsedS ??
            (exportedDurationS != null
              ? startElapsedS + exportedDurationS
              : (intervalEndMs - startMs) / 1000),
        ) * 10,
      ) / 10
    distanceM += interval.distanceM
    intervals.push({
      startElapsedS,
      endElapsedS,
      distanceM: Math.round(interval.distanceM * 10) / 10,
      durationS: Math.round(activeTimeS * 10) / 10,
      cumulativeDistanceM: Math.round(distanceM * 10) / 10,
      paceSPer100m: swimPaceSeconds(interval.distanceM, activeTimeS),
      strokeCount:
        interval.stroke !== 'kickboard' && interval.strokeCount != null
          ? Math.round(interval.strokeCount * 10) / 10
          : null,
      strokeTimeS:
        interval.stroke !== 'kickboard' && interval.strokeTimeS != null
          ? Math.round(interval.strokeTimeS * 10) / 10
          : null,
      strokeRateSpm:
        interval.stroke === 'kickboard'
          ? null
          : swimStrokeRate(interval.strokeCount ?? 0, interval.strokeTimeS ?? 0),
      stroke: interval.stroke,
    })
  }
  return { durationS, intervals }
}

export function enrichSwimMetrics(payload: StravaPayload, apple: AppleCache | null): void {
  const details = Object.values(payload.details).filter(
    (detail): detail is StravaActivityDetail => detail.sport === 'swim',
  )
  const matches = matchAppleSwims(
    Object.values(apple?.swims ?? {}),
    details.map(detail => ({
      id: detail.id,
      date: detail.date,
      start: detail.start,
      distanceM: detail.distanceKm * 1_000,
    })),
  )

  payload.swimTrend = []
  for (const detail of details) {
    const swim = matches.get(detail.id)
    if (swim && Object.keys(swim.strokes).length > 0) detail.strokes = swim.strokes
    const applePace = swim ? swimPaceSeconds(swim.totalM, swim.activeTimeS ?? 0) : null
    detail.swimPaceSPer100m =
      applePace ?? swimPaceSeconds(detail.distanceKm * 1_000, detail.movingTimeS)
    detail.strokeCount = swim?.strokeCount ?? null
    detail.strokeRateSpm = swim
      ? swimStrokeRate(swim.strokeCount ?? 0, swim.strokeTimeS ?? 0)
      : null
    const activity = swim ? swimActivityIntervals(swim) : null
    detail.swimDurationS = activity?.durationS ?? null
    detail.swimIntervals = activity?.intervals ?? []
    if (detail.swimPaceSPer100m == null && detail.strokeRateSpm == null) continue
    payload.swimTrend.push({
      id: detail.id,
      date: detail.date,
      start: detail.start,
      paceSPer100m: detail.swimPaceSPer100m,
      strokeRateSpm: detail.strokeRateSpm,
    })
  }
  payload.swimTrend.sort((a, b) => a.start.localeCompare(b.start) || a.id - b.id)
}

const RUN_DYNAMICS_SAMPLE_MS = 10_000

function timedRunningDynamics(
  samples: AppleRunningDynamicsSample[] | undefined,
): { timeMs: number; value: number }[] {
  return (samples ?? [])
    .map(sample => ({ timeMs: Date.parse(sample.time), value: sample.value }))
    .filter(sample => Number.isFinite(sample.timeMs) && Number.isFinite(sample.value))
    .sort((left, right) => left.timeMs - right.timeMs)
}

function runningDynamicsAt(
  samples: { timeMs: number; value: number }[],
  timeMs: number,
): number | null {
  let low = 0
  let high = samples.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (samples[middle].timeMs < timeMs) low = middle + 1
    else high = middle
  }
  const previous = samples[low - 1]
  const next = samples[low]
  const nearest =
    previous && next
      ? timeMs - previous.timeMs <= next.timeMs - timeMs
        ? previous
        : next
      : (previous ?? next)
  return nearest && Math.abs(nearest.timeMs - timeMs) <= RUN_DYNAMICS_SAMPLE_MS
    ? nearest.value
    : null
}

export function enrichRunDynamics(payload: StravaPayload, apple: AppleCache | null): void {
  const workouts = Object.values(apple?.workouts ?? {})
  if (workouts.length === 0) return
  for (const detail of Object.values(payload.details)) {
    if (!detail || detail.sport !== 'run' || detail.route.length < 2) continue
    const workout = matchAppleRun(
      { start: detail.start, distanceM: detail.distanceKm * 1_000 },
      workouts,
    )
    if (!workout) continue
    const detailStartMs = Date.parse(detail.start)
    const strideLengthM = timedRunningDynamics(workout.strideLengthM)
    const groundContactTimeMs = timedRunningDynamics(workout.groundContactTimeMs)
    const verticalOscillationCm = timedRunningDynamics(workout.verticalOscillationCm)
    for (const point of detail.route) {
      const pointTimeMs = detailStartMs + point.elapsedS * 1_000
      point.strideLengthM = runningDynamicsAt(strideLengthM, pointTimeMs)
      point.groundContactTimeMs = runningDynamicsAt(groundContactTimeMs, pointTimeMs)
      point.verticalOscillationCm = runningDynamicsAt(verticalOscillationCm, pointTimeMs)
    }
  }
}

let memo: { key: string; payload: StravaPayload } | null = null

export function loadStravaPayloadSync(
  since?: string,
  manualFueling: readonly ManualFuelingEntry[] = [],
): StravaPayload {
  const manualKey = manualFueling
    .map(entry => `${entry.activityId}:${entry.caloriesConsumed}`)
    .join(',')
  const key = `${since ?? ''}:${manualKey}:${stamp(stravaCachePath)}:${stamp(ouraCachePath)}:${stamp(garminCachePath)}:${stamp(weatherCachePath)}:${stamp(appleCachePath)}`
  if (memo?.key !== key) {
    const apple = readJson<AppleCache>(appleCachePath)
    const payload = buildPayload(
      readJson<StravaRawCache>(stravaCachePath),
      readJson<OuraCache>(ouraCachePath),
      readJson<GarminCache>(garminCachePath),
      since,
      readJson<WeatherCache>(weatherCachePath),
      ATHLETE.ftp,
    )
    applyManualFueling(payload, manualFueling)
    enrichSwimMetrics(payload, apple)
    enrichRunDynamics(payload, apple)
    memo = { key, payload }
  }
  return memo.payload
}
