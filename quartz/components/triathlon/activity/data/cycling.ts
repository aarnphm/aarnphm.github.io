import { GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD } from '../../../../plugins/stores/strava'
import { CYCLING_POWER_MAX_POINTS } from '../../../../util/cycling-power'
import { GARDEN_CYCLING_STAMINA_METHOD } from '../../../../util/cycling-stamina'
import {
  STAMINA_LEDGER_METHOD,
  STAMINA_LEDGER_OFFSET_METHOD,
} from '../../../../util/stamina-ledger'
import { isRecord } from '../../../../util/type-guards'
import { bounded, finite, isElapsedPauses, nullableBounded, nullableFinite } from './predicates'

export const isStaminaTrace = (value: unknown): boolean => {
  if (value === null) return true
  if (!isRecord(value)) return false
  if (value.source === 'garmin')
    return (
      value.method === 'garmin-native' && value.ftpWatts === null && value.maxHeartRateBpm === null
    )
  if (value.method === STAMINA_LEDGER_OFFSET_METHOD)
    return (
      value.source === 'garden-estimate' &&
      value.ftpWatts === null &&
      value.maxHeartRateBpm === null &&
      finite(value.garminOffset) &&
      value.garminOffset > 0
    )
  return (
    value.source === 'garden-estimate' &&
    value.method === GARDEN_CYCLING_STAMINA_METHOD &&
    typeof value.ftpWatts === 'number' &&
    Number.isFinite(value.ftpWatts) &&
    value.ftpWatts > 0 &&
    typeof value.maxHeartRateBpm === 'number' &&
    Number.isFinite(value.maxHeartRateBpm) &&
    value.maxHeartRateBpm > 0
  )
}

export const isStaminaLedger = (value: unknown): boolean =>
  value == null ||
  (isRecord(value) &&
    value.method === STAMINA_LEDGER_METHOD &&
    typeof value.garminVisible === 'boolean' &&
    [value.fatigueDeficit, value.recoveryHours, value.recoveryFloor, value.garminOffset].every(
      item => finite(item) && item >= 0,
    ))

export const isPerformanceConditionTrace = (value: unknown): boolean => {
  if (!isRecord(value)) return false
  if (value.source === 'garmin') return value.method === 'garmin-native'
  return (
    value.source === 'garden-estimate' &&
    value.method === GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD &&
    finite(value.ftpWatts) &&
    value.ftpWatts > 0 &&
    finite(value.lactateThresholdHeartRateBpm) &&
    finite(value.restingHeartRateBpm) &&
    value.restingHeartRateBpm > 0 &&
    value.lactateThresholdHeartRateBpm > value.restingHeartRateBpm &&
    value.windowSeconds === 360
  )
}

export const isCyclingTorqueTrace = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'wahoo' ||
    value.method !== 'power-cadence-v1' ||
    !isRecord(value.summary) ||
    value.summary.source !== 'wahoo' ||
    value.summary.method !== 'power-cadence-v1' ||
    !bounded(value.summary.coverage, 0, 1) ||
    !bounded(value.summary.observedSeconds, 0, elapsedTimeS) ||
    !finite(value.summary.averageTorqueNm) ||
    value.summary.averageTorqueNm < 0 ||
    !finite(value.summary.averageCadenceRpm) ||
    value.summary.averageCadenceRpm <= 0 ||
    !Array.isArray(value.points) ||
    value.points.length < 2 ||
    !Array.isArray(value.cells)
  )
    return false
  let previous = -1
  let distance = 0
  return (
    value.points.every((point: unknown) => {
      if (
        !isRecord(point) ||
        !bounded(point.elapsedS, 0, elapsedTimeS) ||
        point.elapsedS <= previous ||
        !finite(point.distanceKm) ||
        point.distanceKm < distance ||
        !finite(point.durationS) ||
        point.durationS <= 0 ||
        !nullableBounded(point.torqueNm, 0, Number.MAX_VALUE) ||
        !nullableBounded(point.cadenceRpm, 0, Number.MAX_VALUE) ||
        !nullableBounded(point.watts, 0, Number.MAX_VALUE)
      )
        return false
      previous = point.elapsedS
      distance = point.distanceKm
      return true
    }) &&
    value.cells.every(
      (cell: unknown) =>
        isRecord(cell) &&
        finite(cell.cadenceRpm) &&
        cell.cadenceRpm >= 0 &&
        finite(cell.torqueNm) &&
        cell.torqueNm >= 0 &&
        finite(cell.seconds) &&
        cell.seconds > 0,
    )
  )
}

export const isCyclingIntensityTrace = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'wahoo' ||
    value.method !== 'cumulative-power-30s-v1' ||
    !(finite(value.ftpWatts) && value.ftpWatts > 0) ||
    !(value.ftpSource === 'wahoo-summary' || value.ftpSource === 'athlete') ||
    !Array.isArray(value.points) ||
    value.points.length < 2 ||
    value.points.length > 322
  )
    return false
  let previousElapsed = -1
  let previousDistance = 0
  return value.points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previousElapsed ||
      !finite(point.distanceKm) ||
      point.distanceKm < previousDistance ||
      !nullableBounded(point.intensityFactor, 0, Number.MAX_VALUE)
    )
      return false
    previousElapsed = point.elapsedS
    previousDistance = point.distanceKm
    return true
  })
}

export const isCyclingPowerTrace = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    (value.source !== 'wahoo' && value.source !== 'strava') ||
    (value.terrainSource !== 'wahoo' &&
      value.terrainSource !== 'strava' &&
      value.terrainSource !== 'garmin') ||
    value.method !== 'recorded-power-average-v1' ||
    !Array.isArray(value.points) ||
    value.points.length < 2 ||
    value.points.length > CYCLING_POWER_MAX_POINTS ||
    !isElapsedPauses(value.pauses, elapsedTimeS)
  )
    return false
  let previousElapsed = -1
  let previousDistance = 0
  return value.points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previousElapsed ||
      !finite(point.distanceKm) ||
      point.distanceKm < previousDistance ||
      !nullableFinite(point.elevationM) ||
      !nullableBounded(point.power30sWatts, 0, Number.MAX_VALUE) ||
      !nullableBounded(point.power5mWatts, 0, Number.MAX_VALUE) ||
      !nullableBounded(point.cumulativePowerWatts, 0, Number.MAX_VALUE)
    )
      return false
    previousElapsed = point.elapsedS
    previousDistance = point.distanceKm
    return true
  })
}
