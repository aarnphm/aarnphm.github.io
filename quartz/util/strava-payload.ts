import { readFileSync, statSync } from 'node:fs'
import type { AppleCache, AppleRunningDynamicsSample, AppleSwim } from '../plugins/stores/apple'
import type { GarminCache } from '../plugins/stores/garmin'
import type { OuraCache } from '../plugins/stores/oura'
import type { ManualFuelingEntry, ManualStrengthEntry } from '../plugins/stores/tracking'
import type { WeatherCache } from '../plugins/stores/weather'
import { ATHLETE } from '../plugins/stores/analytics'
import {
  coreBodyTemperatureSamplesForWindow,
  isUsableCoreTemperatureSample,
  parseCoreBodyTemperatureCache,
  type CoreBodyTemperatureActivitySample,
  type CoreBodyTemperatureCache,
} from '../plugins/stores/core-body-temperature'
import {
  applyManualFueling,
  applyManualStrength,
  buildPayload,
  type SwimActivityInterval,
  type StravaActivityDetail,
  type StravaPayload,
  type StravaRawCache,
} from '../plugins/stores/strava'
import { matchAppleRun } from './apple-run-match'
import { matchAppleSwims, matchAppleSwimTelemetry } from './apple-swim-match'
import { joinSegments, QUARTZ } from './path'
import { swimPaceSeconds, swimStrokeRate } from './swim-metrics'

export const stravaCachePath = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
export const ouraCachePath = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
export const garminCachePath = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
export const appleCachePath = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')
export const coreBodyTemperatureCachePath = joinSegments(
  QUARTZ,
  '.quartz-cache',
  'core-body-temperature.json',
)
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

const completeSwimIntervalPace = (
  swim: AppleSwim,
  intervals: SwimActivityInterval[],
): number | null => {
  if (!Number.isFinite(swim.totalM) || swim.totalM <= 0) return null
  let distanceM = 0
  let activeTimeS = 0
  for (const interval of intervals) {
    if (interval.paceSPer100m == null) continue
    distanceM += interval.distanceM
    activeTimeS += interval.durationS
  }
  const toleranceM = Math.max(1, swim.totalM * 0.001)
  if (Math.abs(distanceM - swim.totalM) > toleranceM) return null
  return swimPaceSeconds(distanceM, activeTimeS)
}

export function enrichSwimMetrics(payload: StravaPayload, apple: AppleCache | null): void {
  const details = Object.values(payload.details).filter(
    (detail): detail is StravaActivityDetail => detail.sport === 'swim',
  )
  const swims = Object.values(apple?.swims ?? {})
  const candidates = details.map(detail => ({
    id: detail.id,
    date: detail.date,
    start: detail.start,
    distanceM: detail.distanceKm * 1_000,
  }))
  const matches = matchAppleSwims(swims, candidates)
  const telemetryMatches = matchAppleSwimTelemetry(swims, candidates)

  payload.swimTrend = []
  for (const detail of details) {
    const swim = matches.get(detail.id)
    const telemetry = telemetryMatches.get(detail.id) ?? swim
    const strokeDistribution = [swim, telemetry].find(
      candidate => candidate != null && Object.keys(candidate.strokes).length > 0,
    )
    if (strokeDistribution) detail.strokes = strokeDistribution.strokes
    const activity = swim ? swimActivityIntervals(swim) : null
    const applePace = swim
      ? (completeSwimIntervalPace(swim, activity?.intervals ?? []) ??
        swimPaceSeconds(swim.totalM, swim.activeTimeS ?? 0))
      : null
    detail.swimPaceSPer100m =
      applePace ?? swimPaceSeconds(detail.distanceKm * 1_000, detail.movingTimeS)
    const matchedStrokeRate = swim
      ? swimStrokeRate(swim.strokeCount ?? 0, swim.strokeTimeS ?? 0)
      : null
    const telemetryStrokeRate = telemetry
      ? swimStrokeRate(telemetry.strokeCount ?? 0, telemetry.strokeTimeS ?? 0)
      : null
    detail.strokeCount =
      matchedStrokeRate != null
        ? (swim?.strokeCount ?? null)
        : telemetryStrokeRate != null
          ? (telemetry?.strokeCount ?? null)
          : (swim?.strokeCount ?? telemetry?.strokeCount ?? null)
    detail.strokeRateSpm = matchedStrokeRate ?? telemetryStrokeRate
    detail.swimDurationS = activity?.durationS ?? null
    detail.swimIntervals = activity?.intervals ?? []
    detail.swimLocation = telemetry?.location ?? null
    detail.waterTemperatureC = telemetry?.waterTemperatureC ?? null
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

const CORE_SAMPLE_MAX_DISTANCE_S = 90

type CoreMetric = 'coreTemperatureC' | 'skinTemperatureC' | 'heatStrainIndex'

function coreMetricAt(
  samples: CoreBodyTemperatureActivitySample[],
  elapsedS: number,
  metric: CoreMetric,
): number | null {
  const values = samples
    .map(sample => ({ elapsedS: sample.elapsedS, value: sample[metric] }))
    .filter(
      (sample): sample is { elapsedS: number; value: number } =>
        sample.value != null && Number.isFinite(sample.value),
    )
  if (values.length === 0) return null
  let low = 0
  let high = values.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (values[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const previous = values[low - 1]
  const next = values[low]
  if (!previous && !next) return null
  const nearest =
    previous && next
      ? elapsedS - previous.elapsedS <= next.elapsedS - elapsedS
        ? previous
        : next
      : (previous ?? next)
  if (Math.abs(nearest.elapsedS - elapsedS) > CORE_SAMPLE_MAX_DISTANCE_S) return null
  if (!previous || !next || next.elapsedS === previous.elapsedS) return nearest.value
  if (
    elapsedS < previous.elapsedS ||
    elapsedS > next.elapsedS ||
    next.elapsedS - previous.elapsedS > CORE_SAMPLE_MAX_DISTANCE_S * 2
  )
    return nearest.value
  const fraction = (elapsedS - previous.elapsedS) / (next.elapsedS - previous.elapsedS)
  return previous.value + (next.value - previous.value) * fraction
}

export function enrichCoreBodyTemperature(
  payload: StravaPayload,
  core: CoreBodyTemperatureCache | null,
): void {
  if (!core) return
  for (const detail of Object.values(payload.details)) {
    if (!detail || (detail.sport !== 'bike' && detail.sport !== 'run') || detail.route.length < 2)
      continue
    const durationS = detail.route.at(-1)?.elapsedS ?? detail.movingTimeS
    const samples = coreBodyTemperatureSamplesForWindow(detail.start, durationS, core).filter(
      sample => isUsableCoreTemperatureSample(sample),
    )
    if (samples.length === 0) continue
    for (const point of detail.route) {
      const coreTemperatureC = coreMetricAt(samples, point.elapsedS, 'coreTemperatureC')
      const skinTemperatureC = coreMetricAt(samples, point.elapsedS, 'skinTemperatureC')
      const heatStrainIndex = coreMetricAt(samples, point.elapsedS, 'heatStrainIndex')
      if (coreTemperatureC != null)
        point.coreTemperatureC = Math.round(coreTemperatureC * 100) / 100
      if (skinTemperatureC != null)
        point.skinTemperatureC = Math.round(skinTemperatureC * 100) / 100
      if (heatStrainIndex != null) point.heatStrainIndex = Math.round(heatStrainIndex * 10) / 10
      if (coreTemperatureC != null || skinTemperatureC != null || heatStrainIndex != null)
        point.coreTemperatureSource = 'core-app'
    }
  }
}

let memo: { key: string; payload: StravaPayload } | null = null

export function loadStravaPayloadSync(
  since?: string,
  manualFueling: readonly ManualFuelingEntry[] = [],
  manualStrength: readonly ManualStrengthEntry[] = [],
): StravaPayload {
  const manualKey = JSON.stringify({ fueling: manualFueling, strength: manualStrength })
  const key = `${since ?? ''}:${manualKey}:${stamp(stravaCachePath)}:${stamp(ouraCachePath)}:${stamp(garminCachePath)}:${stamp(weatherCachePath)}:${stamp(appleCachePath)}:${stamp(coreBodyTemperatureCachePath)}`
  if (memo?.key !== key) {
    const apple = readJson<AppleCache>(appleCachePath)
    const core = parseCoreBodyTemperatureCache(readJson<unknown>(coreBodyTemperatureCachePath))
    const payload = buildPayload(
      readJson<StravaRawCache>(stravaCachePath),
      readJson<OuraCache>(ouraCachePath),
      readJson<GarminCache>(garminCachePath),
      since,
      readJson<WeatherCache>(weatherCachePath),
      ATHLETE.ftp,
    )
    applyManualFueling(payload, manualFueling)
    applyManualStrength(payload, manualStrength)
    enrichSwimMetrics(payload, apple)
    enrichRunDynamics(payload, apple)
    enrichCoreBodyTemperature(payload, core)
    memo = { key, payload }
  }
  return memo.payload
}
