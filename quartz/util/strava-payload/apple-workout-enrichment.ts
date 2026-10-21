import type {
  AppleCache,
  AppleRunningDynamicsSample,
  AppleWorkout,
} from '../../plugins/stores/apple'
import {
  normalizeActivityDevice,
  type StravaActivityDetail,
  type StravaPayload,
} from '../../plugins/stores/strava'
import { matchAppleRun } from '../apple-run-match'

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
  const candidates: {
    workout: AppleWorkout
    startDiffMs: number
    samples: TimedAppleHeartRate[]
  }[] = []
  for (const workout of workouts) {
    const startDiffMs = Math.abs(Date.parse(workout.start) - startMs)
    if (!Number.isFinite(startDiffMs) || startDiffMs > APPLE_WORKOUT_START_TOLERANCE_MS) continue
    const samples = appleHeartRateSamples(detail, workout)
    if (samples.length >= 2) candidates.push({ workout, startDiffMs, samples })
  }
  candidates.sort(
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
