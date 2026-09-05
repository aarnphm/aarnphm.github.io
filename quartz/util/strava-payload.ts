import { readFileSync, statSync } from 'node:fs'
import type {
  AppleCache,
  AppleRunningDynamicsSample,
  AppleSwim,
  AppleWorkout,
} from '../plugins/stores/apple'
import type { GarminCache, GarminSwimData } from '../plugins/stores/garmin'
import type { OuraCache } from '../plugins/stores/oura'
import type {
  ActivityTrackingEntry,
  ManualFuelingEntry,
  ManualSaunaEntry,
  ManualStrengthEntry,
} from '../plugins/stores/tracking'
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
  parseCoreBodyTemperatureCache,
  type CoreBodyTemperatureActivitySample,
  type CoreBodyTemperatureCache,
} from '../plugins/stores/core-body-temperature'
import {
  applyActivityTracking,
  applyManualFueling,
  applyManualSauna,
  applyManualStrength,
  buildPayload,
  calculateActivityExerciseLoad,
  calculateActivityIntensityFactor,
  calculateActivityTrainingEffect,
  normalizeActivityDevice,
  prefersActivityDeviceThermal,
  type ActivityAnalysisRange,
  type SwimActivityInterval,
  type StravaActivityDetail,
  type StravaPayload,
} from '../plugins/stores/strava'
import { parseWahooCache, type WahooCache } from '../plugins/stores/wahoo'
import { matchAppleRun } from './apple-run-match'
import { matchAppleSwims, matchAppleSwimTelemetry } from './apple-swim-match'
import { joinSegments, QUARTZ } from './path'
import { latestProviderSync } from './provider-sync'
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

const intervalSwimPace = (
  totalM: number,
  intervals: readonly SwimActivityInterval[],
): number | null => {
  if (!Number.isFinite(totalM) || totalM <= 0) return null
  let distanceM = 0
  let strokeTimeS = 0
  for (const interval of intervals) {
    if (interval.paceSPer100m == null) continue
    distanceM += interval.distanceM
    strokeTimeS += interval.durationS
  }
  if (distanceM / totalM < SWIM_INTERVAL_COVERAGE_MIN) return null
  return swimPaceSeconds(distanceM, strokeTimeS)
}

function garminSwimIntervals(swim: GarminSwimData | undefined): SwimActivityInterval[] {
  if (!swim) return []
  let cumulativeDistanceM = 0
  return swim.lengths.map(length => {
    cumulativeDistanceM += length.distanceM
    return {
      ...length,
      cumulativeDistanceM: Math.round(cumulativeDistanceM * 10) / 10,
      paceSPer100m: swimPaceSeconds(length.distanceM, length.durationS),
    }
  })
}

function garminSwimAnalysisRanges(swim: GarminSwimData | undefined): ActivityAnalysisRange[] {
  if (!swim) return []
  const ranges: ActivityAnalysisRange[] = []
  let cumulativeDistanceKm = 0
  for (const lap of swim.laps ?? []) {
    if (
      !Number.isFinite(lap.startElapsedS) ||
      !Number.isFinite(lap.endElapsedS) ||
      lap.startElapsedS < 0 ||
      lap.endElapsedS <= lap.startElapsedS ||
      !Number.isFinite(lap.distanceM) ||
      lap.distanceM <= 0 ||
      !Number.isFinite(lap.durationS) ||
      lap.durationS <= 0
    )
      continue
    const paceSPer100m = swimPaceSeconds(lap.distanceM, lap.durationS)
    if (paceSPer100m == null) continue
    const distanceKm = lap.distanceM / 1_000
    const startDistanceKm = cumulativeDistanceKm
    cumulativeDistanceKm += distanceKm
    ranges.push({
      kind: 'lap',
      id: `garmin-swim-lap:${ranges.length + 1}`,
      label: `Lap ${ranges.length + 1}`,
      startElapsedS: lap.startElapsedS,
      endElapsedS: lap.endElapsedS,
      startDistanceKm,
      endDistanceKm: cumulativeDistanceKm,
      durationS: lap.durationS,
      movingTimeS: lap.durationS,
      distanceKm,
      elevationGainM: swim.location === 'openWater' ? lap.elevationGainM : null,
      averageSpeedKph: 360 / paceSPer100m,
      averageHeartRate: lap.averageHeartRate,
      averageWatts: null,
      averageCadence: lap.strokeRateSpm,
    })
  }
  return ranges
}

function garminSwimStrokeDistances(swim: GarminSwimData | undefined): Record<string, number> {
  const strokes: Record<string, number> = {}
  for (const length of swim?.lengths ?? []) {
    if (!length.stroke || !Number.isFinite(length.distanceM) || length.distanceM <= 0) continue
    strokes[length.stroke] = (strokes[length.stroke] ?? 0) + length.distanceM
  }
  return strokes
}

function projectSwimHeartRateDistance(
  detail: StravaActivityDetail,
  intervals: readonly SwimActivityInterval[],
): void {
  if (
    intervals.length === 0 ||
    detail.heartRateTrace.some(point => Number.isFinite(point.distanceKm) && point.distanceKm > 0)
  )
    return
  const measured = intervals
    .filter(
      interval =>
        Number.isFinite(interval.startElapsedS) &&
        Number.isFinite(interval.endElapsedS) &&
        interval.endElapsedS > interval.startElapsedS &&
        Number.isFinite(interval.distanceM) &&
        interval.distanceM > 0 &&
        Number.isFinite(interval.cumulativeDistanceM) &&
        interval.cumulativeDistanceM >= interval.distanceM,
    )
    .toSorted((left, right) => left.startElapsedS - right.startElapsedS)
  if (measured.length === 0) return
  detail.heartRateTrace = detail.heartRateTrace.map(point => {
    if (!Number.isFinite(point.elapsedS) || point.elapsedS < 0) return point
    let completedDistanceM = 0
    for (const interval of measured) {
      const startDistanceM = Math.max(
        completedDistanceM,
        interval.cumulativeDistanceM - interval.distanceM,
      )
      if (point.elapsedS < interval.startElapsedS)
        return { ...point, distanceKm: completedDistanceM / 1_000 }
      if (point.elapsedS <= interval.endElapsedS) {
        const fraction =
          (point.elapsedS - interval.startElapsedS) /
          (interval.endElapsedS - interval.startElapsedS)
        return { ...point, distanceKm: (startDistanceM + interval.distanceM * fraction) / 1_000 }
      }
      completedDistanceM = Math.max(completedDistanceM, interval.cumulativeDistanceM)
    }
    return { ...point, distanceKm: completedDistanceM / 1_000 }
  })
}

export function enrichSwimMetrics(
  payload: StravaPayload,
  apple: AppleCache | null,
  garmin: GarminCache | null,
): void {
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
    const garminSwim = detail.garmin ? garmin?.swims?.[detail.garmin.activityId] : undefined
    const garminIntervals = garminSwimIntervals(garminSwim)
    const garminAnalysisRanges = garminSwimAnalysisRanges(garminSwim)
    const garminStrokes = garminSwimStrokeDistances(garminSwim)
    const swim = matches.get(detail.id)
    const telemetry = telemetryMatches.get(detail.id) ?? swim
    const activity = swim ? swimActivityIntervals(swim) : null
    const intervals = garminIntervals.length > 0 ? garminIntervals : (activity?.intervals ?? [])
    const intervalDistanceM =
      garminIntervals.length > 0 ? (garminSwim?.distanceM ?? 0) : (swim?.totalM ?? 0)
    const strokePace = intervalSwimPace(intervalDistanceM, intervals)
    const activePace = garminSwim
      ? swimPaceSeconds(garminSwim.distanceM ?? 0, garminSwim.activeTimeS ?? 0)
      : swim
        ? swimPaceSeconds(swim.totalM, swim.activeTimeS ?? 0)
        : null
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
    if (Object.keys(garminStrokes).length > 0) detail.strokes = garminStrokes
    else {
      const strokeDistribution = [swim, telemetry].find(
        candidate => candidate != null && Object.keys(candidate.strokes).length > 0,
      )
      if (strokeDistribution) detail.strokes = strokeDistribution.strokes
    }
    const matchedStrokeRate = swim
      ? swimStrokeRate(swim.strokeCount ?? 0, swim.strokeTimeS ?? 0)
      : null
    const telemetryStrokeRate = telemetry
      ? swimStrokeRate(telemetry.strokeCount ?? 0, telemetry.strokeTimeS ?? 0)
      : null
    detail.strokeCount =
      garminSwim?.strokeCount ??
      (matchedStrokeRate != null
        ? (swim?.strokeCount ?? null)
        : telemetryStrokeRate != null
          ? (telemetry?.strokeCount ?? null)
          : (swim?.strokeCount ?? telemetry?.strokeCount ?? null))
    detail.strokeRateSpm =
      garminSwim?.strokeRateSpm ??
      matchedStrokeRate ??
      telemetryStrokeRate ??
      (detail.avgCadence != null && detail.avgCadence > 0 ? detail.avgCadence : null)
    detail.swimDurationS = garminSwim?.elapsedTimeS ?? activity?.durationS ?? null
    detail.swimIntervals = intervals
    if (garminAnalysisRanges.length > 0)
      detail.analysisRanges = [
        ...detail.analysisRanges.filter(range => range.kind !== 'lap'),
        ...garminAnalysisRanges,
      ]
    projectSwimHeartRateDistance(detail, intervals)
    detail.swimLocation = garminSwim?.location ?? telemetry?.location ?? null
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
      point.strideLengthM ??= runningDynamicsAt(strideLengthM, pointTimeMs)
      point.groundContactTimeMs ??= runningDynamicsAt(groundContactTimeMs, pointTimeMs)
      point.verticalOscillationCm ??= runningDynamicsAt(verticalOscillationCm, pointTimeMs)
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

const APPLE_DEVICE_START_TOLERANCE_MS = 5 * 60 * 1_000

const appleWorkoutActivity = (sport: StravaActivityDetail['sport']): string | null => {
  if (sport === 'run') return 'running'
  if (sport === 'walk') return 'walking'
  if (sport === 'swim') return 'swimming'
  return null
}

const isAppleWatchUltra3 = (workout: AppleWorkout): boolean =>
  normalizeActivityDevice(workout.device) === 'apple-watch-ultra-3' ||
  normalizeActivityDevice(workout.source) === 'apple-watch-ultra-3'

function hasMatchingAppleWatchUltra3(
  detail: StravaActivityDetail,
  workouts: readonly AppleWorkout[],
): boolean {
  const activity = appleWorkoutActivity(detail.sport)
  const startMs = Date.parse(detail.start)
  if (!activity || !Number.isFinite(startMs)) return false
  const distanceM = detail.distanceKm * 1_000
  return workouts.some(workout => {
    if (workout.activity !== activity || !isAppleWatchUltra3(workout)) return false
    const startDiffMs = Math.abs(Date.parse(workout.start) - startMs)
    if (!Number.isFinite(startDiffMs) || startDiffMs > APPLE_DEVICE_START_TOLERANCE_MS) return false
    const distanceDiffM = workout.distanceM == null ? 0 : Math.abs(workout.distanceM - distanceM)
    if (distanceDiffM > Math.max(detail.sport === 'swim' ? 100 : 200, distanceM * 0.1)) return false
    const durationDiffS = Math.abs(workout.durationS - detail.movingTimeS)
    return durationDiffS <= Math.max(600, detail.elapsedTimeS * 0.2)
  })
}

export function enrichActivityDevices(payload: StravaPayload, apple: AppleCache | null): void {
  const workouts = Object.values(apple?.workouts ?? {})
  for (const detail of Object.values(payload.details)) {
    if (!detail || detail.device != null || appleWorkoutActivity(detail.sport) == null) continue
    const garminDevice = normalizeActivityDevice(detail.garmin?.sourceDevice)
    if (garminDevice) detail.device = garminDevice
    else if (hasMatchingAppleWatchUltra3(detail, workouts)) detail.device = 'apple-watch-ultra-3'
  }
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
              heatStrainSource: null,
              coreTemperatureC: null,
              coreTemperatureSource: null,
              skinTemperatureC: null,
              skinTemperatureSource: null,
            },
          ]
        : []),
      ...samples.map(sample => ({
        distanceKm: 0,
        elapsedS: sample.elapsedS,
        heartRate: sample.heartRate,
        heatStrainIndex: null,
        heatStrainSource: null,
        coreTemperatureC: null,
        coreTemperatureSource: null,
        skinTemperatureC: null,
        skinTemperatureSource: null,
      })),
      ...(samples.at(-1)?.elapsedS !== detail.movingTimeS
        ? [
            {
              distanceKm: 0,
              elapsedS: detail.movingTimeS,
              heartRate: null,
              heatStrainIndex: null,
              heatStrainSource: null,
              coreTemperatureC: null,
              coreTemperatureSource: null,
              skinTemperatureC: null,
              skinTemperatureSource: null,
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

const validCoreMetric = (metric: CoreMetric, value: number): boolean => {
  if (metric === 'heatStrainIndex') return value >= 0 && value <= 20
  if (metric === 'coreTemperatureC') return value >= 25 && value <= 45
  return value >= 0 && value <= 50
}

function coreMetricAt(
  samples: CoreBodyTemperatureActivitySample[],
  elapsedS: number,
  metric: CoreMetric,
): number | null {
  const values = samples
    .map(sample => ({ elapsedS: sample.elapsedS, value: sample[metric] }))
    .filter(
      (sample): sample is { elapsedS: number; value: number } =>
        sample.value != null &&
        Number.isFinite(sample.value) &&
        validCoreMetric(metric, sample.value),
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
      sample => sample.quality == null || sample.quality >= 2,
    )
    if (samples.length === 0) continue
    const nativeFirst = prefersActivityDeviceThermal(detail.sport)
    for (const point of points) {
      const coreTemperatureC = coreMetricAt(samples, point.elapsedS, 'coreTemperatureC')
      const skinTemperatureC = coreMetricAt(samples, point.elapsedS, 'skinTemperatureC')
      const heatStrainIndex = coreMetricAt(samples, point.elapsedS, 'heatStrainIndex')
      if (coreTemperatureC != null && (!nativeFirst || point.coreTemperatureC == null)) {
        point.coreTemperatureC = Math.round(coreTemperatureC * 100) / 100
        point.coreTemperatureSource = 'core-app'
      }
      if (skinTemperatureC != null && (!nativeFirst || point.skinTemperatureC == null)) {
        point.skinTemperatureC = Math.round(skinTemperatureC * 100) / 100
        point.skinTemperatureSource = 'core-app'
      }
      if (heatStrainIndex != null && (!nativeFirst || point.heatStrainIndex == null)) {
        point.heatStrainIndex = Math.round(heatStrainIndex * 10) / 10
        point.heatStrainSource = 'core-app'
      }
    }
  }
}

export type LoadedStravaPayload = StravaPayload & { dailyAnalytics: TriathlonDailyAnalytics }

export type StravaPayloadAnalyticsInputs = Pick<
  AnalyticsInputs,
  'weights' | 'events' | 'dexa' | 'vo2labs'
>

export interface ManualActivityTracking {
  readonly activities: readonly ActivityTrackingEntry[]
  readonly fueling: readonly ManualFuelingEntry[]
  readonly strength: readonly ManualStrengthEntry[]
  readonly sauna: readonly ManualSaunaEntry[]
}

export function applyManualActivityTracking(
  payload: StravaPayload,
  tracking: ManualActivityTracking | null | undefined,
  oura: OuraCache | null,
  weather: WeatherCache | null,
  garmin: GarminCache | null,
): void {
  applyManualFueling(payload, tracking?.fueling ?? [])
  applyManualStrength(payload, tracking?.strength ?? [])
  applyManualSauna(
    payload,
    tracking?.sauna ?? [],
    oura?.heartRate ?? [],
    undefined,
    weather,
    garmin,
  )
}

const payloadMemo = new Map<string, LoadedStravaPayload>()
let payloadMemoStamps = ''

export function loadStravaPayloadSync(
  since: string | undefined,
  manualTracking: ManualActivityTracking | null | undefined,
  analyticsInputs: StravaPayloadAnalyticsInputs = {},
): LoadedStravaPayload {
  const manualKey = JSON.stringify({
    activities: manualTracking?.activities ?? [],
    fueling: manualTracking?.fueling ?? [],
    strength: manualTracking?.strength ?? [],
    sauna: manualTracking?.sauna ?? [],
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
  const generatedAt = latestProviderSync(strava, oura, garmin, wahoo, apple, core, weather)
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
    ATHLETE.lt,
    generatedAt,
    manualTracking?.activities,
  )
  applyManualActivityTracking(payload, manualTracking, oura, weather, garmin)
  enrichActivityDevices(payload, apple)
  enrichRouteLessHeartRate(payload, apple)
  enrichSwimMetrics(payload, apple, garmin)
  enrichRunDynamics(payload, apple)
  enrichCoreBodyTemperature(payload, core)
  const analytics = buildAnalytics(
    applyActivityTracking(strava, garmin, manualTracking?.activities ?? []),
    {
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
      generatedAt,
    },
  )
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
