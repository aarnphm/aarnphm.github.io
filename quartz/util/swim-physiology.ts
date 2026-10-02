import type { StravaActivityDetail } from '../plugins/stores/strava'
import type { HeartRatePhysiologyPoint } from './heart-rate-physiology'
import { heartRateStaminaDepletionPerHour } from './heart-rate-physiology'

export const SWIM_PHYSIOLOGY_METHOD = 'garden-swim-session-v1'
export type SwimStrokeRateSource =
  | 'stream'
  | 'stream-with-average'
  | 'activity-average'
  | 'unavailable'

export interface SwimPhysiologySample {
  elapsedS: number
  distanceKm: number
  heartRate: number | null
  speedMps: number | null
  strokeRateSpm: number | null
}

export interface SwimPhysiology {
  source: 'garden-estimate'
  method: typeof SWIM_PHYSIOLOGY_METHOD
  exertionSource: 'heart-rate'
  speedBasis: 'ground-speed'
  strokeRateSource: SwimStrokeRateSource
  maxHeartRateBpm: number
  baselineHeartRateBpm: number
  baselineSpeedMps: number
  baselineStrokesPerM: number | null
  baselineSeconds: 180
  windowSeconds: 60
  coverage: number
  points: HeartRatePhysiologyPoint[]
}

const bounded = (v: number | null, min: number, max: number): v is number =>
  v != null && Number.isFinite(v) && v >= min && v <= max

export function applySwimPhysiology(
  detail: StravaActivityDetail,
  maxHeartRateBpm: number | null,
): void {
  if (detail.sport !== 'swim' || detail.swimLocation !== 'openWater' || maxHeartRateBpm == null)
    return
  const streamStrokeRate = detail.route.some(p => bounded(p.cad, 5, 100))
  const averageStrokeRate = bounded(detail.strokeRateSpm, 5, 100) ? detail.strokeRateSpm : null
  const strokeRateSource = streamStrokeRate
    ? averageStrokeRate != null && detail.route.some(p => !bounded(p.cad, 5, 100))
      ? 'stream-with-average'
      : 'stream'
    : averageStrokeRate != null
      ? 'activity-average'
      : 'unavailable'
  const samples = detail.route
    .filter(p => p.elapsedS <= detail.elapsedTimeS)
    .map((point, i, route) => {
      const previous = route[i - 1]
      const duration = previous ? point.elapsedS - previous.elapsedS : 0
      return {
        elapsedS: point.elapsedS,
        distanceKm: point.d,
        heartRate: point.hr > 0 ? point.hr : null,
        speedMps:
          previous && duration > 0 && duration <= 120
            ? (1000 * (point.d - previous.d)) / duration
            : null,
        strokeRateSpm: streamStrokeRate
          ? bounded(point.cad, 5, 100)
            ? point.cad
            : averageStrokeRate
          : averageStrokeRate,
      }
    })
  detail.swimPhysiology = estimateSwimPhysiology(samples, maxHeartRateBpm, strokeRateSource)
  // Missing swimming telemetry stays missing instead of reverting to an HR-only condition proxy.
  detail.heartRatePhysiology = null
}

export function estimateSwimPhysiology(
  samples: readonly SwimPhysiologySample[],
  maxHeartRateBpm: number,
  strokeRateSource: SwimStrokeRateSource,
): SwimPhysiology | null {
  if (!bounded(maxHeartRateBpm, 100, 240) || samples.length < 2) return null
  const intervals: {
    start: number
    end: number
    hr: number
    speed: number
    rate: number | null
  }[] = []
  const points: HeartRatePhysiologyPoint[] = []
  let stamina = 100
  let observedSeconds = 0
  let baselineSeconds = 0
  let baselineHr = 0
  let baselineSpeed = 0
  let baselineStrokeRate = 0
  let baselineStrokeSeconds = 0
  let baseline: { hr: number; speed: number; strokesPerM: number | null } | null = null
  let windowStart = 0
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i]
    const previous = samples[i - 1]
    if (
      !Number.isFinite(sample.elapsedS) ||
      sample.elapsedS < 0 ||
      !Number.isFinite(sample.distanceKm) ||
      sample.distanceKm < 0 ||
      (previous &&
        (sample.elapsedS <= previous.elapsedS || sample.distanceKm < previous.distanceKm))
    )
      return null
    const duration = previous ? sample.elapsedS - previous.elapsedS : 0
    const covered =
      previous &&
      duration <= 120 &&
      bounded(previous.heartRate, 35, 240) &&
      bounded(sample.heartRate, 35, 240) &&
      bounded(sample.speedMps, 100 / 360, 100 / 45)
    if (
      covered &&
      previous &&
      sample.speedMps != null &&
      previous.heartRate != null &&
      sample.heartRate != null
    ) {
      const hr = (previous.heartRate + sample.heartRate) / 2
      const rate = bounded(sample.strokeRateSpm, 5, 100) ? sample.strokeRateSpm : null
      const interval = {
        start: previous.elapsedS,
        end: sample.elapsedS,
        hr,
        speed: sample.speedMps,
        rate,
      }
      intervals.push(interval)
      observedSeconds += duration
      // Skip the first minute, then use three observed minutes as the session reference.
      const baselineDuration = Math.min(
        180 - baselineSeconds,
        Math.max(0, sample.elapsedS - Math.max(60, previous.elapsedS)),
      )
      baselineHr += hr * baselineDuration
      baselineSpeed += sample.speedMps * baselineDuration
      if (rate != null) {
        baselineStrokeRate += rate * baselineDuration
        baselineStrokeSeconds += baselineDuration
      }
      baselineSeconds += baselineDuration
      if (baselineSeconds >= 180 && !baseline)
        baseline = {
          hr: baselineHr / 180,
          speed: baselineSpeed / 180,
          strokesPerM:
            baselineStrokeSeconds >= 144
              ? baselineStrokeRate / baselineStrokeSeconds / 60 / (baselineSpeed / 180)
              : null,
        }
    }
    while (windowStart < intervals.length && intervals[windowStart].end <= sample.elapsedS - 60)
      windowStart++
    let seconds = 0,
      hrSum = 0,
      speedSum = 0,
      rateSum = 0,
      rateSeconds = 0
    for (let j = windowStart; j < intervals.length; j++) {
      const interval = intervals[j]
      const overlap = interval.end - Math.max(interval.start, sample.elapsedS - 60)
      seconds += overlap
      hrSum += interval.hr * overlap
      speedSum += interval.speed * overlap
      if (interval.rate != null) {
        rateSum += interval.rate * overlap
        rateSeconds += overlap
      }
    }
    let condition: number | null = null
    if (covered && seconds > 0) {
      const speed = speedSum / seconds,
        hr = hrSum / seconds
      const strokeCost =
        baseline?.strokesPerM != null && rateSeconds >= 0.8 * seconds
          ? Math.max(1, rateSum / rateSeconds / 60 / speed / baseline.strokesPerM)
          : 1
      const demand = baseline ? Math.max(1, (speed / baseline.speed) ** 3) : 1
      stamina = Math.max(
        0,
        stamina -
          (heartRateStaminaDepletionPerHour(hr, maxHeartRateBpm) * demand * strokeCost * duration) /
            3600,
      )
      if (baseline && seconds >= 48)
        condition = Math.max(
          -20,
          Math.min(20, 100 * (speed / baseline.speed / (hr / baseline.hr) / strokeCost - 1)),
        )
    }
    const available = i === 0 ? bounded(sample.heartRate, 35, 240) : Boolean(covered)
    points.push({
      elapsedS: sample.elapsedS,
      distanceKm: sample.distanceKm,
      stamina: available ? stamina : null,
      potentialStamina: available ? stamina : null,
      performanceCondition: condition,
    })
  }
  if (!baseline) return null
  return {
    source: 'garden-estimate',
    method: SWIM_PHYSIOLOGY_METHOD,
    exertionSource: 'heart-rate',
    speedBasis: 'ground-speed',
    strokeRateSource,
    maxHeartRateBpm,
    baselineHeartRateBpm: baseline.hr,
    baselineSpeedMps: baseline.speed,
    baselineStrokesPerM: baseline.strokesPerM,
    baselineSeconds: 180,
    windowSeconds: 60,
    coverage: observedSeconds / (samples[samples.length - 1].elapsedS - samples[0].elapsedS),
    points,
  }
}
