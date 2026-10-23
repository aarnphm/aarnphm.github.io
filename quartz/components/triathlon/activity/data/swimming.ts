import { HEART_RATE_PHYSIOLOGY_METHOD } from '../../../../util/heart-rate-physiology'
import { SWIM_PHYSIOLOGY_METHOD } from '../../../../util/swim-physiology'
import { SWIM_POWER_REFERENCE_PACE_S_PER_100M } from '../../../../util/swim-power'
import { isRecord } from '../../../../util/type-guards'
import { bounded, finite, nullableBounded } from './predicates'

export const isHeartRatePhysiology = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garden-estimate' ||
    value.method !== HEART_RATE_PHYSIOLOGY_METHOD ||
    !bounded(value.maxHeartRateBpm, 100, 240) ||
    !bounded(value.baselineHeartRateBpm, 35, 240) ||
    value.windowSeconds !== 60 ||
    !Array.isArray(value.points) ||
    value.points.length < 2
  )
    return false
  return isPhysiologyPoints(value.points, elapsedTimeS)
}

const isPhysiologyPoints = (points: unknown, elapsedTimeS: number): boolean => {
  if (!Array.isArray(points) || points.length < 2) return false
  let previous = -1
  let distance = 0
  return points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previous ||
      !finite(point.distanceKm) ||
      point.distanceKm < distance ||
      !nullableBounded(point.stamina, 0, 100) ||
      !nullableBounded(point.potentialStamina, 0, 100) ||
      !nullableBounded(point.performanceCondition, -20, 20)
    )
      return false
    previous = point.elapsedS
    distance = point.distanceKm
    return true
  })
}

export const isSwimPhysiology = (value: unknown, elapsedTimeS: number): boolean =>
  value == null ||
  (isRecord(value) &&
    value.source === 'garden-estimate' &&
    value.method === SWIM_PHYSIOLOGY_METHOD &&
    value.exertionSource === 'heart-rate' &&
    (value.speedBasis === 'ground-speed' || value.speedBasis === 'pool-length') &&
    (value.strokeRateSource === 'stream' ||
      value.strokeRateSource === 'stream-with-average' ||
      value.strokeRateSource === 'activity-average' ||
      value.strokeRateSource === 'unavailable') &&
    bounded(value.maxHeartRateBpm, 100, 240) &&
    bounded(value.baselineHeartRateBpm, 35, 240) &&
    bounded(value.baselineSpeedMps, 100 / 360, 100 / 45) &&
    nullableBounded(value.baselineStrokesPerM, 0.001, 10) &&
    value.baselineSeconds === 180 &&
    value.windowSeconds === 60 &&
    bounded(value.coverage, 0, 1) &&
    isPhysiologyPoints(value.points, elapsedTimeS))

export const isSwimPower = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garden-estimate' ||
    !(
      (value.speedBasis === 'pool-length' &&
        value.method === 'freestyle-drag-index-v1' &&
        (value.inputSource === 'garmin' ||
          value.inputSource === 'apple' ||
          value.inputSource === 'reconstructed')) ||
      (value.speedBasis === 'ground-speed' &&
        value.method === 'open-water-drag-index-v1' &&
        value.inputSource === 'route')
    ) ||
    value.referencePaceSPer100m !== SWIM_POWER_REFERENCE_PACE_S_PER_100M ||
    !bounded(value.averageIndex, 0.001, 4000) ||
    !finite(value.activeTimeS) ||
    value.activeTimeS <= 0 ||
    !finite(value.distanceM) ||
    value.distanceM <= 0 ||
    !finite(value.validIntervalCount) ||
    !Number.isSafeInteger(value.validIntervalCount) ||
    value.validIntervalCount <= 0 ||
    !finite(value.excludedIntervalCount) ||
    !Number.isSafeInteger(value.excludedIntervalCount) ||
    value.excludedIntervalCount < 0 ||
    !Array.isArray(value.curve) ||
    !Array.isArray(value.histogramS) ||
    value.histogramS.length === 0 ||
    value.histogramS.length > 160
  )
    return false
  let histogramTimeS = 0
  for (const seconds of value.histogramS) {
    if (!finite(seconds) || seconds < 0) return false
    histogramTimeS += seconds
  }
  if (Math.abs(histogramTimeS - value.activeTimeS) > 0.5) return false
  let previous = 0
  return value.curve.every((p: unknown) => {
    if (
      !isRecord(p) ||
      !bounded(p.durationS, 60, elapsedTimeS + 1) ||
      p.durationS <= previous ||
      !bounded(p.index, 0.001, 4000) ||
      !bounded(p.startElapsedS, 0, elapsedTimeS + 1) ||
      !bounded(p.endElapsedS, p.startElapsedS, elapsedTimeS + 1)
    )
      return false
    previous = p.durationS
    return true
  })
}
