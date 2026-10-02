import type { StravaActivityDetail } from '../plugins/stores/strava'

export const HEART_RATE_PHYSIOLOGY_METHOD = 'garden-hr-session-v1'

export interface HeartRatePhysiologyPoint {
  elapsedS: number
  distanceKm: number
  stamina: number | null
  potentialStamina: number | null
  performanceCondition: number | null
}

export interface HeartRatePhysiology {
  source: 'garden-estimate'
  method: typeof HEART_RATE_PHYSIOLOGY_METHOD
  maxHeartRateBpm: number
  baselineHeartRateBpm: number
  windowSeconds: 60
  points: HeartRatePhysiologyPoint[]
}

interface HeartRateSample {
  elapsedS: number
  distanceKm: number
  heartRate: number | null
}

export function applyHeartRatePhysiology(
  detail: StravaActivityDetail,
  maxHeartRateBpm: number | null,
): void {
  if (maxHeartRateBpm == null) return
  if (detail.deviceWatts && detail.route.some(point => point.w > 0)) return
  const samples =
    detail.sport === 'walk' && detail.heartRateTrace.length >= 2
      ? detail.heartRateTrace
      : detail.route.length >= 2
        ? detail.route.map(point => ({
            elapsedS: point.elapsedS,
            distanceKm: point.d,
            heartRate: point.hr > 0 ? point.hr : null,
          }))
        : detail.heartRateTrace
  detail.heartRatePhysiology = estimateHeartRatePhysiology(
    samples.filter(sample => sample.elapsedS <= detail.elapsedTimeS),
    maxHeartRateBpm,
  )
}

// The HR term used by Garden's cycling stamina model, expressed as percentage points/hour.
export const heartRateStaminaDepletionPerHour = (heartRate: number, maxHeartRate: number): number =>
  98.85 * (heartRate / maxHeartRate) ** 10

export function estimateHeartRatePhysiology(
  samples: readonly HeartRateSample[],
  maxHeartRateBpm: number,
): HeartRatePhysiology | null {
  if (!Number.isFinite(maxHeartRateBpm) || maxHeartRateBpm < 100 || maxHeartRateBpm > 240)
    return null
  if (samples.length < 2) return null
  const valid = (value: number | null): value is number =>
    value != null && Number.isFinite(value) && value >= 35 && value <= 240
  const intervals: { start: number; end: number; hr: number }[] = []
  const points: HeartRatePhysiologyPoint[] = []
  let stamina = 100
  let baselineHeartRateBpm: number | null = null
  let windowStart = 0
  let observedSeconds = 0
  let baselineSum = 0
  let baselineSeconds = 0
  for (let index = 0; index < samples.length; index++) {
    const sample = samples[index]
    const previous = samples[index - 1]
    if (
      !Number.isFinite(sample.elapsedS) ||
      sample.elapsedS < 0 ||
      !Number.isFinite(sample.distanceKm) ||
      sample.distanceKm < 0 ||
      (previous && sample.elapsedS <= previous.elapsedS)
    )
      return null
    const duration = previous ? sample.elapsedS - previous.elapsedS : 0
    const covered =
      previous && duration <= 120 && valid(previous.heartRate) && valid(sample.heartRate)
    if (covered && valid(previous.heartRate) && valid(sample.heartRate)) {
      const hr = (previous.heartRate + sample.heartRate) / 2
      observedSeconds += duration
      stamina = Math.max(
        0,
        stamina - (heartRateStaminaDepletionPerHour(hr, maxHeartRateBpm) * duration) / 3600,
      )
      intervals.push({ start: previous.elapsedS, end: sample.elapsedS, hr })
      const baselineDuration = Math.max(0, Math.min(60 - baselineSeconds, duration))
      baselineSum += hr * baselineDuration
      baselineSeconds += baselineDuration
      if (baselineSeconds >= 60 && baselineHeartRateBpm == null)
        baselineHeartRateBpm = baselineSum / baselineSeconds
    }
    while (windowStart < intervals.length && intervals[windowStart].end <= sample.elapsedS - 60)
      windowStart++
    let weightedHr = 0
    let seconds = 0
    for (let offset = windowStart; offset < intervals.length; offset++) {
      const interval = intervals[offset]
      const overlap = interval.end - Math.max(interval.start, sample.elapsedS - 60)
      weightedHr += interval.hr * overlap
      seconds += overlap
    }
    const available = valid(sample.heartRate) && (index === 0 || covered)
    points.push({
      elapsedS: sample.elapsedS,
      distanceKm: sample.distanceKm,
      stamina: available ? stamina : null,
      potentialStamina: available ? stamina : null,
      // A within-session HR change proxy. No pace, power, HRV, or fitness baseline is inferred.
      performanceCondition:
        available && baselineHeartRateBpm != null && seconds >= 48
          ? Math.max(
              -20,
              Math.min(20, (100 * (baselineHeartRateBpm - weightedHr / seconds)) / maxHeartRateBpm),
            )
          : null,
    })
  }
  const span = samples[samples.length - 1].elapsedS - samples[0].elapsedS
  if (baselineHeartRateBpm == null || observedSeconds < span * 0.8) return null
  return {
    source: 'garden-estimate',
    method: HEART_RATE_PHYSIOLOGY_METHOD,
    maxHeartRateBpm,
    baselineHeartRateBpm,
    windowSeconds: 60,
    points,
  }
}
