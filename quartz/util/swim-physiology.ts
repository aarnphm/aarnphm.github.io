import type { StravaActivityDetail } from '../plugins/stores/strava'
import type { HeartRatePhysiologyPoint } from './heart-rate-physiology'
import { staminaDepletionPerHour } from './heart-rate-physiology'
import { swimPaceSeconds } from './swim-metrics'

export const SWIM_PHYSIOLOGY_METHOD = 'garden-swim-session-v2'
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
  speedBasis: 'ground-speed' | 'pool-length'
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

function poolSamples(detail: StravaActivityDetail): SwimPhysiologySample[] {
  const lengths: {
    start: number
    end: number
    startDistanceM: number
    distanceM: number
    speed: number
    rate: number | null
  }[] = []
  for (const interval of detail.swimIntervals) {
    const previous = lengths.at(-1)
    // Pool start timestamps can be rounded to seconds while length durations retain fractions.
    const start = Math.max(previous?.end ?? 0, interval.startElapsedS)
    if (
      !Number.isFinite(start) ||
      (previous && previous.end - interval.startElapsedS > 1) ||
      !Number.isFinite(interval.endElapsedS) ||
      interval.endElapsedS <= start ||
      interval.endElapsedS > detail.elapsedTimeS ||
      swimPaceSeconds(interval.distanceM, interval.durationS) == null ||
      !Number.isFinite(interval.cumulativeDistanceM) ||
      interval.cumulativeDistanceM < interval.distanceM
    )
      continue
    lengths.push({
      start,
      end: interval.endElapsedS,
      startDistanceM: Math.max(
        previous ? previous.startDistanceM + previous.distanceM : 0,
        interval.cumulativeDistanceM - interval.distanceM,
      ),
      distanceM: interval.distanceM,
      speed: interval.distanceM / interval.durationS,
      rate: interval.strokeRateSpm,
    })
  }
  const hr = detail.heartRateTrace
    .filter(
      p => Number.isFinite(p.elapsedS) && p.elapsedS >= 0 && p.elapsedS <= detail.elapsedTimeS,
    )
    .sort((a, b) => a.elapsedS - b.elapsedS)
  if (lengths.length === 0 || hr.length < 2) return []
  const times = [
    ...new Set([...hr.map(p => p.elapsedS), ...lengths.flatMap(p => [p.start, p.end])]),
  ].sort((a, b) => a - b)
  let hrIndex = 0
  let lengthIndex = 0
  return times.map((elapsedS, i) => {
    while (hrIndex < hr.length - 1 && hr[hrIndex].elapsedS < elapsedS) hrIndex++
    const next = hr[hrIndex]
    const previous = hr[hrIndex - 1]
    let heartRate: number | null = null
    if (next.elapsedS === elapsedS && bounded(next.heartRate, 35, 240)) heartRate = next.heartRate
    else if (
      previous &&
      next.elapsedS > elapsedS &&
      next.elapsedS - previous.elapsedS <= 120 &&
      bounded(previous.heartRate, 35, 240) &&
      bounded(next.heartRate, 35, 240)
    )
      heartRate =
        previous.heartRate +
        ((next.heartRate - previous.heartRate) * (elapsedS - previous.elapsedS)) /
          (next.elapsedS - previous.elapsedS)
    const midpoint = i > 0 ? (times[i - 1] + elapsedS) / 2 : elapsedS
    while (lengthIndex < lengths.length - 1 && lengths[lengthIndex].end < midpoint) lengthIndex++
    const length = lengths[lengthIndex]
    const active = midpoint >= length.start && midpoint <= length.end
    const fraction = Math.max(
      0,
      Math.min(1, (elapsedS - length.start) / (length.end - length.start)),
    )
    const completed = lengths[lengthIndex - 1]
    const distanceM =
      elapsedS < length.start
        ? completed
          ? completed.startDistanceM + completed.distanceM
          : 0
        : length.startDistanceM + length.distanceM * fraction
    return {
      elapsedS,
      distanceKm: distanceM / 1000,
      heartRate,
      speedMps: active ? length.speed : null,
      strokeRateSpm: active ? length.rate : null,
    }
  })
}

export function applySwimPhysiology(
  detail: StravaActivityDetail,
  maxHeartRateBpm: number | null,
): void {
  if (detail.sport !== 'swim' || maxHeartRateBpm == null) return
  const pool =
    detail.swimLocation === 'pool' || (detail.route.length < 2 && detail.swimIntervals.length > 0)
  if (!pool && detail.swimLocation !== 'openWater') return
  const streamStrokeRate = pool
    ? detail.swimIntervals.some(p => bounded(p.strokeRateSpm, 5, 100))
    : detail.route.some(p => bounded(p.cad, 5, 100))
  const averageStrokeRate = bounded(detail.strokeRateSpm, 5, 100) ? detail.strokeRateSpm : null
  const strokeRateSource = streamStrokeRate
    ? averageStrokeRate != null &&
      (pool
        ? detail.swimIntervals.some(p => !bounded(p.strokeRateSpm, 5, 100))
        : detail.route.some(p => !bounded(p.cad, 5, 100)))
      ? 'stream-with-average'
      : 'stream'
    : averageStrokeRate != null
      ? 'activity-average'
      : 'unavailable'
  const samples = pool
    ? poolSamples(detail).map(p => ({
        ...p,
        strokeRateSpm: bounded(p.strokeRateSpm, 5, 100) ? p.strokeRateSpm : averageStrokeRate,
      }))
    : detail.route
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
  detail.swimPhysiology = estimateSwimPhysiology(
    samples,
    maxHeartRateBpm,
    strokeRateSource,
    pool ? 'pool-length' : 'ground-speed',
  )
  // Pool swims retain the HR fallback when length telemetry cannot support the combined model.
  if (detail.swimPhysiology || !pool) detail.heartRatePhysiology = null
}

export function estimateSwimPhysiology(
  samples: readonly SwimPhysiologySample[],
  maxHeartRateBpm: number,
  strokeRateSource: SwimStrokeRateSource,
  speedBasis: SwimPhysiology['speedBasis'] = 'ground-speed',
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
          (staminaDepletionPerHour('swim', hr, maxHeartRateBpm) * demand * strokeCost * duration) /
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
    speedBasis,
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
