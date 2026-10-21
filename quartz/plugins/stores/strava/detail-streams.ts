import type { GarminStreams } from '../garmin'
import { sumPrefix } from './efforts-and-curves'
import { nativeThermalAt, type ActivityMetricSamples } from './metric-samples'
import {
  ROUTE_POINTS,
  type ActivityHeartRate,
  type ActivityHeartRateTracePoint,
  type ActivityKind,
  type StravaStreams,
} from './raw-and-kinds'
import {
  nearestElapsedIndex,
  round,
  sampleIndicesWithRequired,
  timedMetricAt,
  timedNullableMetricAt,
  timedStreamAlignment,
  type TimedNullableMetricSample,
} from './stream-primitives'

export const GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD = 'garden-cycling-performance-condition-v1'

export interface CyclingPerformanceConditionSample {
  elapsedS: number
  value: number
}

export interface CyclingPerformanceConditionEstimate {
  method: typeof GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD
  ftpWatts: number
  lactateThresholdHeartRateBpm: number
  restingHeartRateBpm: number
  windowSeconds: number
  samples: CyclingPerformanceConditionSample[]
}

export type ActivityPerformanceConditionTrace =
  | { source: 'garmin'; method: 'garmin-native' }
  | ({ source: 'garden-estimate' } & Omit<CyclingPerformanceConditionEstimate, 'samples'>)

const CYCLING_PERFORMANCE_CONDITION_WINDOW_S = 6 * 60
const CYCLING_PERFORMANCE_CONDITION_POWER_WINDOW_S = 30
const CYCLING_PERFORMANCE_CONDITION_SAMPLE_STEP_S = 15
const CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE = 0.85

export function calculateCyclingPerformanceCondition(
  watts: ArrayLike<number>,
  wattsObserved: ArrayLike<number>,
  heartRate: ArrayLike<number>,
  ftpWatts: number | null,
  lactateThresholdHeartRateBpm: number | null,
  restingHeartRateBpm: number | null,
): CyclingPerformanceConditionEstimate | null {
  const length = Math.min(watts.length, wattsObserved.length, heartRate.length)
  if (
    length <= CYCLING_PERFORMANCE_CONDITION_WINDOW_S ||
    ftpWatts == null ||
    !Number.isFinite(ftpWatts) ||
    ftpWatts <= 0 ||
    lactateThresholdHeartRateBpm == null ||
    !Number.isFinite(lactateThresholdHeartRateBpm) ||
    restingHeartRateBpm == null ||
    !Number.isFinite(restingHeartRateBpm) ||
    restingHeartRateBpm <= 0 ||
    lactateThresholdHeartRateBpm <= restingHeartRateBpm
  )
    return null

  const powerSum = new Float64Array(length + 1)
  const powerCount = new Uint32Array(length + 1)
  const heartRateSum = new Float64Array(length + 1)
  const heartRateCount = new Uint32Array(length + 1)
  for (let second = 0; second < length; second++) {
    const power = watts[second]
    const powerIsObserved = wattsObserved[second] > 0 && Number.isFinite(power)
    const bpm = heartRate[second]
    const heartRateIsObserved = Number.isFinite(bpm) && bpm > restingHeartRateBpm
    powerSum[second + 1] = powerSum[second] + (powerIsObserved ? Math.max(0, power) : 0)
    powerCount[second + 1] = powerCount[second] + (powerIsObserved ? 1 : 0)
    heartRateSum[second + 1] = heartRateSum[second] + (heartRateIsObserved ? bpm : 0)
    heartRateCount[second + 1] = heartRateCount[second] + (heartRateIsObserved ? 1 : 0)
  }

  const rollingPowerFourth = new Float64Array(length)
  const rollingPowerValid = new Uint8Array(length)
  for (let second = 0; second < length; second++) {
    const start = Math.max(0, second - CYCLING_PERFORMANCE_CONDITION_POWER_WINDOW_S + 1)
    const span = second - start + 1
    const observed = powerCount[second + 1] - powerCount[start]
    if (observed / span < CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE) continue
    const averageWatts = (powerSum[second + 1] - powerSum[start]) / observed
    rollingPowerFourth[second] = averageWatts ** 4
    rollingPowerValid[second] = 1
  }
  const rollingPowerFourthSum = sumPrefix(rollingPowerFourth)
  const rollingPowerCount = new Uint32Array(length + 1)
  for (let second = 0; second < length; second++)
    rollingPowerCount[second + 1] = rollingPowerCount[second] + rollingPowerValid[second]

  const thresholdHeartRateReserve = lactateThresholdHeartRateBpm - restingHeartRateBpm
  const samples: CyclingPerformanceConditionSample[] = []
  for (
    let second = CYCLING_PERFORMANCE_CONDITION_WINDOW_S;
    second < length;
    second += CYCLING_PERFORMANCE_CONDITION_SAMPLE_STEP_S
  ) {
    const start = second - CYCLING_PERFORMANCE_CONDITION_WINDOW_S + 1
    const powerSamples = rollingPowerCount[second + 1] - rollingPowerCount[start]
    const heartRateSamples = heartRateCount[second + 1] - heartRateCount[start]
    if (
      powerSamples / CYCLING_PERFORMANCE_CONDITION_WINDOW_S <
        CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE ||
      heartRateSamples / CYCLING_PERFORMANCE_CONDITION_WINDOW_S <
        CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE
    )
      continue
    const normalizedPower = Math.pow(
      (rollingPowerFourthSum[second + 1] - rollingPowerFourthSum[start]) / powerSamples,
      0.25,
    )
    const averageHeartRate = (heartRateSum[second + 1] - heartRateSum[start]) / heartRateSamples
    const powerIntensity = normalizedPower / ftpWatts
    const heartRateIntensity = (averageHeartRate - restingHeartRateBpm) / thresholdHeartRateReserve
    if (
      !Number.isFinite(powerIntensity) ||
      !Number.isFinite(heartRateIntensity) ||
      powerIntensity < 0.3 ||
      heartRateIntensity < 0.3 ||
      heartRateIntensity > 1.5
    )
      continue
    const value = Math.min(20, Math.max(-20, 100 * (powerIntensity / heartRateIntensity - 1)))
    samples.push({ elapsedS: second, value: round(value, 1) })
  }
  if (samples.length < 2) return null
  return {
    method: GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD,
    ftpWatts,
    lactateThresholdHeartRateBpm,
    restingHeartRateBpm,
    windowSeconds: CYCLING_PERFORMANCE_CONDITION_WINDOW_S,
    samples,
  }
}

const PERFORMANCE_CONDITION_SAMPLE_MAX_DISTANCE_S = 15

export function performanceConditionAt(
  samples: TimedNullableMetricSample[],
  elapsedS: number,
): number | null {
  const value = timedNullableMetricAt(
    samples,
    elapsedS,
    PERFORMANCE_CONDITION_SAMPLE_MAX_DISTANCE_S,
  )
  return value == null ? null : round(value, 1)
}

export function estimatedPerformanceConditionAt(
  samples: CyclingPerformanceConditionSample[],
  elapsedS: number,
): number | null {
  if (samples.length === 0 || elapsedS < samples[0].elapsedS) return null
  const value = timedMetricAt(samples, elapsedS)
  return value == null ? null : round(value, 1)
}

function nativeThermalTimeline(samples: ActivityMetricSamples): number[] {
  const elapsed = new Set<number>()
  for (const sample of [
    ...samples.heatStrainIndex,
    ...samples.coreTemperatureC,
    ...samples.skinTemperatureC,
  ])
    elapsed.add(sample.elapsedS)
  return [...elapsed].sort((left, right) => left - right)
}

export function projectHeartRateTrace(
  sport: ActivityKind,
  streams: StravaStreams | GarminStreams | undefined,
  heartRate: ActivityHeartRate,
  metricSamples: ActivityMetricSamples,
  boundaryElapsedS: number[] = [],
): ActivityHeartRateTracePoint[] {
  const time = streams?.time?.length ? streams.time : nativeThermalTimeline(metricSamples)
  if (time.length < 2) return []
  const values =
    heartRate.stream.length === time.length
      ? heartRate.stream
      : Array.from({ length: time.length }, () => 0)
  let previousTime = -Infinity
  for (const elapsedS of time) {
    if (!Number.isFinite(elapsedS) || elapsedS < 0 || elapsedS < previousTime) return []
    previousTime = elapsedS
  }
  if (time[time.length - 1] <= time[0]) return []
  const distanceAlignment =
    sport === 'swim' || sport === 'walk' ? timedStreamAlignment(streams) : null
  if (sport === 'swim' && !distanceAlignment) return []
  const available = values.map(value => Number.isFinite(value) && value > 0)
  const hasThermal =
    metricSamples.heatStrainIndex.some(sample => sample.value != null) ||
    metricSamples.coreTemperatureC.some(sample => sample.value != null) ||
    metricSamples.skinTemperatureC.some(sample => sample.value != null)
  if (available.filter(Boolean).length < 2 && !hasThermal) return []
  const required = boundaryElapsedS.map(elapsedS => nearestElapsedIndex(time, elapsedS))
  let peakIndex = -1
  for (let index = 1; index < available.length; index++)
    if (available[index] !== available[index - 1]) required.push(index - 1, index)
  for (let index = 0; index < values.length; index++)
    if (available[index] && (peakIndex < 0 || values[index] > values[peakIndex])) peakIndex = index
  if (peakIndex >= 0) required.push(peakIndex)
  return sampleIndicesWithRequired(0, values.length - 1, ROUTE_POINTS, required).map(index => {
    const elapsedS = time[index]
    return {
      distanceKm: distanceAlignment ? round(distanceAlignment.distance[index] / 1_000, 3) : 0,
      elapsedS: round(elapsedS, 3),
      heartRate: available[index] ? Math.round(values[index]) : null,
      ...nativeThermalAt(metricSamples, elapsedS),
    }
  })
}
