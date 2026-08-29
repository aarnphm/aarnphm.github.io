import { readFileSync, statSync } from 'node:fs'
import type {
  AppleCache,
  AppleRunningDynamicsSample,
  AppleSwim,
  AppleWorkout,
} from '../plugins/stores/apple'
import type { GarminCache } from '../plugins/stores/garmin'
import type { OuraCache } from '../plugins/stores/oura'
import type { ManualFuelingEntry, ManualStrengthEntry } from '../plugins/stores/tracking'
import type { WeatherCache } from '../plugins/stores/weather'
import {
  ATHLETE,
  buildAnalytics,
  type ActivityDistributionPoint,
  type ActivitySummary,
  type AnalyticsInputs,
} from '../plugins/stores/analytics'
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
  calculateActivityExerciseLoad,
  calculateActivityIntensityFactor,
  calculateActivityTrainingEffect,
  type SwimActivityInterval,
  type StravaActivityDetail,
  type StravaPayload,
} from '../plugins/stores/strava'
import { parseWahooCache, type WahooCache } from '../plugins/stores/wahoo'
import { matchAppleRun } from './apple-run-match'
import { matchAppleSwims, matchAppleSwimTelemetry } from './apple-swim-match'
import { joinSegments, QUARTZ } from './path'
import { readStravaCacheFileSync } from './strava-cache-file'
import { swimPaceSeconds, swimStrokeRate } from './swim-metrics'
import {
  buildTriathlonDailyAnalytics,
  type TriathlonDailyAnalytics,
} from './triathlon-day-analytics'

export const stravaCachePath = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
export const ouraCachePath = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
export const garminCachePath = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
export const wahooCachePath = joinSegments(QUARTZ, '.quartz-cache', 'wahoo.json')
export const appleCachePath = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')
export const coreBodyTemperatureCachePath = joinSegments(
  QUARTZ,
  '.quartz-cache',
  'core-body-temperature.json',
)
export const weatherCachePath = joinSegments(QUARTZ, '.quartz-cache', 'weather.json')

export function enrichCalculatedIntensityFactors(
  payload: StravaPayload,
  activities: readonly Pick<ActivitySummary, 'id' | 'paceIntensityFactor'>[],
  ftp: number | null,
  lactateThresholdHr: number | null,
): void {
  const paceIntensityFactors = new Map(
    activities.flatMap(activity =>
      activity.paceIntensityFactor == null
        ? []
        : [[activity.id, activity.paceIntensityFactor] as const],
    ),
  )
  for (const detail of Object.values(payload.details))
    detail.calculatedIntensityFactor = calculateActivityIntensityFactor(
      detail,
      paceIntensityFactors.get(detail.id) ?? null,
      ftp,
      lactateThresholdHr,
    )
}

export function enrichRunPaceZones(
  payload: StravaPayload,
  distributions: {
    activities: readonly Pick<ActivityDistributionPoint, 'id' | 'paceZoneSeconds'>[]
    paceZoneBoundsSPerKm: readonly number[]
    tenKmRaceTimeS: number | null
  },
): void {
  const boundsSPerKm = distributions.paceZoneBoundsSPerKm
  const tenKmRaceTimeS = distributions.tenKmRaceTimeS
  const validReference =
    boundsSPerKm.length === 5 &&
    boundsSPerKm.every(value => Number.isFinite(value) && value > 0) &&
    tenKmRaceTimeS != null &&
    Number.isFinite(tenKmRaceTimeS) &&
    tenKmRaceTimeS > 0
  const zonesByActivity = new Map(
    distributions.activities.flatMap(activity =>
      activity.paceZoneSeconds == null ? [] : [[activity.id, activity.paceZoneSeconds] as const],
    ),
  )
  for (const detail of Object.values(payload.details)) {
    const zoneSeconds = zonesByActivity.get(detail.id)
    detail.runPaceZones =
      detail.sport === 'run' &&
      validReference &&
      zoneSeconds?.length === boundsSPerKm.length + 1 &&
      zoneSeconds.every(value => Number.isFinite(value) && value >= 0) &&
      zoneSeconds.some(value => value > 0)
        ? { zoneSeconds: [...zoneSeconds], boundsSPerKm: [...boundsSPerKm], tenKmRaceTimeS }
        : null
  }
}

export function enrichCalculatedExerciseLoads(payload: StravaPayload): void {
  for (const detail of Object.values(payload.details))
    detail.calculatedExerciseLoad = calculateActivityExerciseLoad(detail)
}

export function enrichCalculatedTrainingEffects(payload: StravaPayload): void {
  for (const detail of Object.values(payload.details))
    detail.calculatedTrainingEffect = calculateActivityTrainingEffect(detail)
}

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

function readWahooCache(): WahooCache | null {
  const value = readJson<unknown>(wahooCachePath)
  if (value == null) return null
  try {
    return parseWahooCache(value)
  } catch {
    return null
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

const SWIM_INTERVAL_COVERAGE_MIN = 0.9

const strokeTimeSwimPace = (swim: AppleSwim, intervals: SwimActivityInterval[]): number | null => {
  if (!Number.isFinite(swim.totalM) || swim.totalM <= 0) return null
  let distanceM = 0
  let strokeTimeS = 0
  for (const interval of intervals) {
    if (interval.paceSPer100m == null) continue
    distanceM += interval.distanceM
    strokeTimeS += interval.durationS
  }
  if (distanceM / swim.totalM < SWIM_INTERVAL_COVERAGE_MIN) return null
  return swimPaceSeconds(distanceM, strokeTimeS)
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
    const strokePace = swim ? strokeTimeSwimPace(swim, activity?.intervals ?? []) : null
    const activePace = swim ? swimPaceSeconds(swim.totalM, swim.activeTimeS ?? 0) : null
    const movingPace = swimPaceSeconds(detail.distanceKm * 1_000, detail.movingTimeS)
    detail.swimPaceSPer100m = strokePace ?? activePace ?? movingPace
    detail.swimPaceSource =
      strokePace != null
        ? 'stroke'
        : activePace != null
          ? 'active'
          : movingPace != null
            ? 'moving'
            : null
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
    detail.strokeRateSpm =
      matchedStrokeRate ??
      telemetryStrokeRate ??
      (detail.avgCadence != null && detail.avgCadence > 0 ? detail.avgCadence : null)
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
      paceSource: detail.swimPaceSource,
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

const APPLE_WORKOUT_START_TOLERANCE_MS = 5 * 60 * 1_000
const APPLE_HEART_RATE_TRACE_LIMIT = 140

type TimedAppleHeartRate = { elapsedS: number; heartRate: number }

function appleHeartRateSamples(
  detail: StravaActivityDetail,
  workout: AppleWorkout,
): TimedAppleHeartRate[] {
  const startMs = Date.parse(detail.start)
  if (!Number.isFinite(startMs) || detail.movingTimeS <= 0) return []
  const endMs = startMs + detail.movingTimeS * 1_000
  const samples = new Map<number, number>()
  for (const sample of workout.heartRate) {
    const timeMs = Date.parse(sample.time)
    if (
      !Number.isFinite(timeMs) ||
      timeMs < startMs ||
      timeMs > endMs ||
      !Number.isFinite(sample.bpm) ||
      sample.bpm <= 0
    )
      continue
    samples.set(Math.round((timeMs - startMs) / 100) / 10, Math.round(sample.bpm))
  }
  return [...samples]
    .map(([elapsedS, heartRate]) => ({ elapsedS, heartRate }))
    .sort((left, right) => left.elapsedS - right.elapsedS)
}

function sampleAppleHeartRate(samples: TimedAppleHeartRate[]): TimedAppleHeartRate[] {
  if (samples.length <= APPLE_HEART_RATE_TRACE_LIMIT) return samples
  const stride = Math.ceil(samples.length / APPLE_HEART_RATE_TRACE_LIMIT)
  let peakIndex = 0
  for (let index = 1; index < samples.length; index++)
    if (samples[index].heartRate > samples[peakIndex].heartRate) peakIndex = index
  const indices = new Set<number>([0, samples.length - 1, peakIndex])
  for (let index = 0; index < samples.length; index += stride) indices.add(index)
  return [...indices].sort((left, right) => left - right).map(index => samples[index])
}

function matchingAppleHeartRate(
  detail: StravaActivityDetail,
  workouts: readonly AppleWorkout[],
): { workout: AppleWorkout; samples: TimedAppleHeartRate[] } | null {
  const startMs = Date.parse(detail.start)
  if (!Number.isFinite(startMs)) return null
  const candidates = workouts
    .map(workout => ({
      workout,
      startDiffMs: Math.abs(Date.parse(workout.start) - startMs),
      samples: appleHeartRateSamples(detail, workout),
    }))
    .filter(
      candidate =>
        Number.isFinite(candidate.startDiffMs) &&
        candidate.startDiffMs <= APPLE_WORKOUT_START_TOLERANCE_MS &&
        candidate.samples.length >= 2,
    )
    .sort(
      (left, right) =>
        left.startDiffMs - right.startDiffMs ||
        right.samples.length - left.samples.length ||
        left.workout.id.localeCompare(right.workout.id),
    )
  return candidates[0] ?? null
}

export function enrichRouteLessHeartRate(payload: StravaPayload, apple: AppleCache | null): void {
  const workouts = Object.values(apple?.workouts ?? {})
  if (workouts.length === 0) return
  for (const detail of Object.values(payload.details)) {
    if (
      !detail ||
      detail.sport === 'sauna' ||
      detail.route.length >= 2 ||
      detail.heartRateTrace.filter(point => point.heartRate != null).length >= 2
    )
      continue
    const match = matchingAppleHeartRate(detail, workouts)
    if (!match) continue
    const samples = sampleAppleHeartRate(match.samples)
    detail.heartRateTrace = [
      ...(samples[0].elapsedS > 0
        ? [
            {
              distanceKm: 0,
              elapsedS: 0,
              heartRate: null,
              heatStrainIndex: null,
              coreTemperatureC: null,
              skinTemperatureC: null,
              coreTemperatureSource: null,
            },
          ]
        : []),
      ...samples.map(sample => ({
        distanceKm: 0,
        elapsedS: sample.elapsedS,
        heartRate: sample.heartRate,
        heatStrainIndex: null,
        coreTemperatureC: null,
        skinTemperatureC: null,
        coreTemperatureSource: null,
      })),
      ...(samples.at(-1)?.elapsedS !== detail.movingTimeS
        ? [
            {
              distanceKm: 0,
              elapsedS: detail.movingTimeS,
              heartRate: null,
              heatStrainIndex: null,
              coreTemperatureC: null,
              skinTemperatureC: null,
              coreTemperatureSource: null,
            },
          ]
        : []),
    ]
    detail.avgHr ??=
      match.workout.averageHeartRateBpm != null &&
      Number.isFinite(match.workout.averageHeartRateBpm) &&
      match.workout.averageHeartRateBpm > 0
        ? Math.round(match.workout.averageHeartRateBpm)
        : Math.round(
            samples.reduce((total, sample) => total + sample.heartRate, 0) / samples.length,
          )
    detail.maxHr ??= Math.max(...samples.map(sample => sample.heartRate))
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
    if (!detail) continue
    const points = detail.route.length >= 2 ? detail.route : detail.heartRateTrace
    if (points.length < 2) continue
    const durationS = Math.max(detail.movingTimeS, points.at(-1)?.elapsedS ?? 0)
    const samples = coreBodyTemperatureSamplesForWindow(detail.start, durationS, core).filter(
      sample => isUsableCoreTemperatureSample(sample),
    )
    if (samples.length === 0) continue
    for (const point of points) {
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

export type LoadedStravaPayload = StravaPayload & { dailyAnalytics: TriathlonDailyAnalytics }

export type StravaPayloadAnalyticsInputs = Pick<
  AnalyticsInputs,
  'weights' | 'events' | 'dexa' | 'vo2labs'
>

const payloadMemo = new Map<string, LoadedStravaPayload>()
let payloadMemoStamps = ''

export function loadStravaPayloadSync(
  since?: string,
  manualFueling: readonly ManualFuelingEntry[] = [],
  manualStrength: readonly ManualStrengthEntry[] = [],
  analyticsInputs: StravaPayloadAnalyticsInputs = {},
): LoadedStravaPayload {
  const manualKey = JSON.stringify({
    fueling: manualFueling,
    strength: manualStrength,
    analytics: analyticsInputs,
  })
  const stamps = `${stamp(stravaCachePath)}:${stamp(ouraCachePath)}:${stamp(garminCachePath)}:${stamp(wahooCachePath)}:${stamp(weatherCachePath)}:${stamp(appleCachePath)}:${stamp(coreBodyTemperatureCachePath)}`
  if (payloadMemoStamps !== stamps) {
    payloadMemo.clear()
    payloadMemoStamps = stamps
  }
  const key = `${since ?? ''}:${manualKey}`
  const cached = payloadMemo.get(key)
  if (cached) return cached
  const strava = readStravaCacheFileSync(stravaCachePath)
  const oura = readJson<OuraCache>(ouraCachePath)
  const garmin = readJson<GarminCache>(garminCachePath)
  const wahoo = readWahooCache()
  const apple = readJson<AppleCache>(appleCachePath)
  const core = parseCoreBodyTemperatureCache(readJson<unknown>(coreBodyTemperatureCachePath))
  const weather = readJson<WeatherCache>(weatherCachePath)
  const payload = buildPayload(
    strava,
    oura,
    garmin,
    since,
    weather,
    ATHLETE.ftp,
    undefined,
    undefined,
    wahoo,
    ATHLETE.hrMax,
  )
  applyManualFueling(payload, manualFueling)
  applyManualStrength(payload, manualStrength)
  enrichSwimMetrics(payload, apple)
  enrichRunDynamics(payload, apple)
  enrichRouteLessHeartRate(payload, apple)
  enrichCoreBodyTemperature(payload, core)
  const analytics = buildAnalytics(strava, {
    ...analyticsInputs,
    oura,
    apple,
    core,
    garmin,
    weather,
    ftp: ATHLETE.ftp,
    powerCurve: {
      sixWeeks: payload.powerCurveRef,
      year: payload.powerCurveYearRef,
      yearLabel: payload.powerCurveYear,
      criticalPower: payload.criticalPower,
      criticalPowerYear: payload.criticalPowerYear,
      ftp: ATHLETE.ftp,
      goalFtp: ATHLETE.goalFTP,
    },
    zones: payload.zones,
    activityDetails: payload.details,
    since,
  })
  enrichRunPaceZones(payload, analytics.distributions)
  enrichCalculatedIntensityFactors(payload, analytics.activities, ATHLETE.ftp, ATHLETE.lt)
  enrichCalculatedExerciseLoads(payload)
  enrichCalculatedTrainingEffects(payload)
  const ouraDetails = oura?.details ?? {}
  const loaded: LoadedStravaPayload = {
    ...payload,
    dailyAnalytics: buildTriathlonDailyAnalytics(analytics, ouraDetails, payload.details),
  }
  payloadMemo.set(key, loaded)
  return loaded
}
