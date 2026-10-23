import {
  RUN_GAP_MAX_MPS,
  RUN_HEART_COST_MAX_POINTS,
  RUN_HEART_RATE_LAG_S,
  RUN_HEART_RATE_MAX_BPM,
  RUN_HEART_RATE_MIN_BPM,
} from '../../../../util/run-mechanics'
import { isRecord } from '../../../../util/type-guards'
import { WALK_POWER_METHOD, WALK_POWER_WINDOW_SECONDS } from '../../../../util/walk-power'
import { bounded, finite, nullableBounded, nullableFinite } from './predicates'

export const isWalkPower = (value: unknown, elapsedTimeS: number, date: unknown): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garden-estimate' ||
    (value.inputSource !== 'strava' && value.inputSource !== 'garmin') ||
    value.method !== WALK_POWER_METHOD ||
    value.kind !== 'net-metabolic' ||
    value.windowSeconds !== WALK_POWER_WINDOW_SECONDS ||
    !isRecord(value.weight) ||
    !finite(value.weight.kg) ||
    value.weight.kg <= 0 ||
    value.weight.date !== date ||
    value.weight.source !== 'garmin' ||
    !finite(value.averageWatts) ||
    value.averageWatts < 0 ||
    !Array.isArray(value.points) ||
    value.points.length < 3
  )
    return false
  let previous = -1
  let distance = 0
  let usable = 0
  const valid = value.points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previous ||
      !finite(point.distanceKm) ||
      point.distanceKm < distance ||
      !(point.watts === null || (finite(point.watts) && point.watts >= 0))
    )
      return false
    previous = point.elapsedS
    distance = point.distanceKm
    if (point.watts != null) usable++
    return true
  })
  return valid && usable >= 2
}

const isRunStrideCentroid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    bounded(value.cadenceSpm, 100, 260) &&
    bounded(value.strideM, 0.2, 3) &&
    finite(value.seconds) &&
    value.seconds > 0)

export const isRunStrideCadenceTrace = (value: unknown, elapsedTimeS: number): boolean =>
  value == null ||
  (isRecord(value) &&
    value.source === 'strava' &&
    value.method === 'speed-cadence-v1' &&
    bounded(value.seconds, 0, elapsedTimeS + 60) &&
    Array.isArray(value.cells) &&
    value.cells.length > 0 &&
    value.cells.every(
      (cell: unknown) =>
        isRecord(cell) &&
        bounded(cell.cadenceSpm, 100, 260) &&
        bounded(cell.strideM, 0.2, 3) &&
        finite(cell.seconds) &&
        cell.seconds > 0,
    ) &&
    isRunStrideCentroid(value.early) &&
    isRunStrideCentroid(value.late))

export const isRunHeartCostTrace = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'strava' ||
    value.method !== 'gap-per-beat-v1' ||
    value.heartRateLagS !== RUN_HEART_RATE_LAG_S ||
    !Array.isArray(value.points) ||
    value.points.length < 2 ||
    value.points.length > RUN_HEART_COST_MAX_POINTS
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
      !nullableBounded(point.gapSpeed30sMps, 0, RUN_GAP_MAX_MPS) ||
      !nullableBounded(point.heartRate30sBpm, RUN_HEART_RATE_MIN_BPM, RUN_HEART_RATE_MAX_BPM) ||
      !nullableBounded(point.gapSpeed5mMps, 0, RUN_GAP_MAX_MPS) ||
      !nullableBounded(point.heartRate5mBpm, RUN_HEART_RATE_MIN_BPM, RUN_HEART_RATE_MAX_BPM) ||
      !nullableBounded(
        point.cumulativeMetersPerBeat,
        0,
        (RUN_GAP_MAX_MPS * 60) / RUN_HEART_RATE_MIN_BPM,
      ) ||
      (point.gapSpeed30sMps == null) !== (point.heartRate30sBpm == null) ||
      (point.gapSpeed5mMps == null) !== (point.heartRate5mBpm == null)
    )
      return false
    previousElapsed = point.elapsedS
    previousDistance = point.distanceKm
    return true
  })
}

export const isRunWalk = (value: unknown): boolean => {
  if (value === null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garmin' ||
    !bounded(value.elapsedTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !bounded(value.runTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !bounded(value.walkTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !bounded(value.idleTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !Array.isArray(value.segments) ||
    value.segments.length === 0
  )
    return false
  let elapsedTimeS = 0
  let runTimeS = 0
  let walkTimeS = 0
  let idleTimeS = 0
  for (const segment of value.segments) {
    if (
      !isRecord(segment) ||
      (segment.state !== 'run' && segment.state !== 'walk' && segment.state !== 'idle') ||
      !finite(segment.startElapsedS) ||
      !finite(segment.endElapsedS) ||
      Math.abs(segment.startElapsedS - elapsedTimeS) > 0.002 ||
      segment.endElapsedS <= segment.startElapsedS
    )
      return false
    const durationS = segment.endElapsedS - segment.startElapsedS
    elapsedTimeS = segment.endElapsedS
    if (segment.state === 'run') runTimeS += durationS
    else if (segment.state === 'walk') walkTimeS += durationS
    else idleTimeS += durationS
  }
  return (
    Math.abs(value.elapsedTimeS - elapsedTimeS) <= 0.002 &&
    Math.abs(value.runTimeS - runTimeS) <= 0.01 &&
    Math.abs(value.walkTimeS - walkTimeS) <= 0.01 &&
    Math.abs(value.idleTimeS - idleTimeS) <= 0.01
  )
}

export const isRunBestEfforts = (value: unknown): boolean => {
  if (value == null) return true
  return (
    isRecord(value) &&
    value.distanceSource === 'calculated' &&
    value.weightKg === null &&
    value.weightDate === null &&
    Array.isArray(value.power) &&
    value.power.length === 0 &&
    Array.isArray(value.climbs) &&
    value.climbs.length === 0 &&
    Array.isArray(value.distance) &&
    value.distance.every(
      (effort: unknown) =>
        isRecord(effort) &&
        typeof effort.label === 'string' &&
        finite(effort.targetDistanceM) &&
        effort.targetDistanceM > 0 &&
        finite(effort.elapsedTimeS) &&
        effort.elapsedTimeS > 0 &&
        finite(effort.averageSpeedKph) &&
        effort.averageSpeedKph > 0 &&
        (effort.averageHeartRate === null ||
          (finite(effort.averageHeartRate) && effort.averageHeartRate > 0)) &&
        nullableFinite(effort.elevationDeltaM),
    )
  )
}
