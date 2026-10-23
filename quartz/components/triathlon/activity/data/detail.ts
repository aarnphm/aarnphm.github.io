import {
  isActivityDevice,
  isActivityKind,
  type StravaActivityDetail,
} from '../../../../plugins/stores/strava'
import { emptyWahooMetrics } from '../../../../plugins/stores/wahoo'
import { STAMINA_LEDGER_OFFSET_METHOD } from '../../../../util/stamina-ledger'
import { isRecord } from '../../../../util/type-guards'
import {
  isCyclingIntensityTrace,
  isCyclingPowerTrace,
  isCyclingTorqueTrace,
  isPerformanceConditionTrace,
  isStaminaLedger,
  isStaminaTrace,
} from './cycling'
import { isActivityAnalyses } from './environment'
import { finite, nullableBounded, optionalNullableBounded } from './predicates'
import {
  isRunBestEfforts,
  isRunHeartCostTrace,
  isRunStrideCadenceTrace,
  isRunWalk,
  isWalkPower,
} from './running'
import { isHeartRatePhysiology, isSwimPhysiology, isSwimPower } from './swimming'

const isWahooVerification = (value: unknown): boolean => {
  if (value === undefined) return true
  if (!isRecord(value) || !isRecord(value.metrics) || !isRecord(value.summarySources)) return false
  const metrics = value.metrics
  return (
    typeof value.activityId === 'string' &&
    (value.fitPath === null ||
      (typeof value.fitPath === 'string' &&
        value.fitPath.startsWith('triathlon/wahoo/') &&
        value.fitPath.toLowerCase().endsWith('.fit'))) &&
    typeof value.sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test(value.sha256) &&
    (value.sourceDevice === null || typeof value.sourceDevice === 'string') &&
    finite(value.startOffsetS) &&
    (value.distanceM === null || finite(value.distanceM)) &&
    [null, 'strava', 'garmin'].includes(value.streamFallback as string | null) &&
    Object.keys(emptyWahooMetrics()).every(key => metrics[key] === null || finite(metrics[key])) &&
    Object.values(value.summarySources).every(source => source === 'wahoo')
  )
}

const isActivitySources = (value: unknown): boolean =>
  value === undefined ||
  (Array.isArray(value) &&
    value.every(
      source =>
        isRecord(source) &&
        (source.provider === 'strava' ||
          source.provider === 'garmin' ||
          source.provider === 'wahoo') &&
        typeof source.activityId === 'string' &&
        (source.name === null || typeof source.name === 'string') &&
        (source.fileName === null || typeof source.fileName === 'string'),
    ))

const isActivityMoves = (value: unknown): boolean =>
  value === undefined ||
  (isRecord(value) &&
    value.source === 'manual' &&
    Array.isArray(value.entries) &&
    value.entries.length > 0 &&
    value.entries.every(
      move =>
        isRecord(move) &&
        typeof move.name === 'string' &&
        move.name.trim().length > 0 &&
        Array.isArray(move.sets) &&
        move.sets.length > 0 &&
        move.sets.every(
          set =>
            isRecord(set) &&
            typeof set.repetitions === 'number' &&
            Number.isSafeInteger(set.repetitions) &&
            set.repetitions > 0 &&
            typeof set.perSide === 'boolean',
        ),
    ))

const isThermalSource = (value: unknown): boolean =>
  value === null || value === 'core-app' || value === 'core-fit'

const isThermalPoint = (value: unknown): boolean =>
  isRecord(value) &&
  nullableBounded(value.heatStrainIndex, 0, 20) &&
  isThermalSource(value.heatStrainSource) &&
  (value.heatStrainIndex == null) === (value.heatStrainSource === null) &&
  nullableBounded(value.coreTemperatureC, 25, 45) &&
  isThermalSource(value.coreTemperatureSource) &&
  (value.coreTemperatureC == null) === (value.coreTemperatureSource === null) &&
  nullableBounded(value.skinTemperatureC, 0, 50) &&
  isThermalSource(value.skinTemperatureSource) &&
  (value.skinTemperatureC == null) === (value.skinTemperatureSource === null)

const isThermalTrace = (value: unknown): boolean =>
  Array.isArray(value) && value.every(isThermalPoint)

const isRouteMetricPoint = (value: unknown): boolean =>
  isRecord(value) &&
  optionalNullableBounded(value.performanceCondition, -20, 20) &&
  optionalNullableBounded(value.strideLengthM, 0.2, 3) &&
  optionalNullableBounded(value.verticalRatioPct, 0, 50) &&
  optionalNullableBounded(value.verticalOscillationCm, 1, 30) &&
  optionalNullableBounded(value.groundContactBalanceLeftPct, 0, 100) &&
  optionalNullableBounded(value.groundContactTimeMs, 50, 1_000) &&
  optionalNullableBounded(value.stepSpeedLossMps, 0, 5) &&
  optionalNullableBounded(value.stepSpeedLossPct, 0, 100) &&
  optionalNullableBounded(value.impactLoadFactor, 0, 10)

const isRouteTrace = (value: unknown): boolean =>
  Array.isArray(value) && value.every(point => isThermalPoint(point) && isRouteMetricPoint(point))

export const isActivityDetail = (value: unknown): value is StravaActivityDetail => {
  if (
    !isRecord(value) ||
    typeof value.id !== 'number' ||
    !/^\d{4}-\d{2}-\d{2}$/.test(typeof value.date === 'string' ? value.date : '') ||
    !isActivityKind(value.sport) ||
    !(value.commute === undefined || typeof value.commute === 'boolean') ||
    (value.sport === 'run' && !isRunBestEfforts(value.bestEfforts)) ||
    !isWahooVerification(value.wahoo) ||
    !isActivitySources(value.sources) ||
    !(
      value.powerCurveWeight === undefined ||
      (isRecord(value.powerCurveWeight) &&
        finite(value.powerCurveWeight.kg) &&
        value.powerCurveWeight.kg > 0 &&
        typeof value.powerCurveWeight.date === 'string' &&
        /^\d{4}-\d{2}-\d{2}$/.test(value.powerCurveWeight.date) &&
        typeof value.date === 'string' &&
        value.powerCurveWeight.date <= value.date &&
        value.powerCurveWeight.source === 'garmin')
    ) ||
    !(
      value.equipment === undefined ||
      (isRecord(value.equipment) &&
        typeof value.equipment.id === 'string' &&
        value.equipment.id.trim().length > 0 &&
        (value.equipment.name === null ||
          (typeof value.equipment.name === 'string' && value.equipment.name.trim().length > 0)) &&
        value.equipment.source === 'strava')
    ) ||
    !isActivityMoves(value.moves) ||
    !(
      value.computerOverride === undefined ||
      (value.sport === 'bike' &&
        typeof value.computerOverride === 'string' &&
        value.computerOverride.trim().length > 0)
    ) ||
    !(value.device === null || isActivityDevice(value.device)) ||
    !isStaminaTrace(value.staminaTrace) ||
    !isStaminaLedger(value.staminaLedger) ||
    (isRecord(value.staminaTrace) &&
      ((value.staminaTrace.source === 'garden-estimate' &&
        (value.staminaTrace.method === STAMINA_LEDGER_OFFSET_METHOD
          ? value.sport !== 'bike' && value.sport !== 'run'
          : value.sport !== 'bike')) ||
        (value.staminaTrace.source === 'garmin' &&
          value.sport !== 'bike' &&
          value.sport !== 'run'))) ||
    !(
      value.performanceConditionTrace === null ||
      isPerformanceConditionTrace(value.performanceConditionTrace)
    ) ||
    (isRecord(value.performanceConditionTrace) &&
      ((value.performanceConditionTrace.source === 'garden-estimate' && value.sport !== 'bike') ||
        (value.sport !== 'bike' && value.sport !== 'run'))) ||
    !finite(value.elapsedTimeS) ||
    !(
      value.swimProvenance === undefined ||
      (value.sport === 'swim' &&
        isRecord(value.swimProvenance) &&
        value.swimProvenance.source === 'athlete-estimate' &&
        typeof value.swimProvenance.note === 'string' &&
        Number.isSafeInteger(value.swimProvenance.trainingPeaksWorkoutId))
    ) ||
    !isSwimPhysiology(value.swimPhysiology, value.elapsedTimeS) ||
    !isSwimPower(value.swimPower, value.elapsedTimeS) ||
    ((value.swimPhysiology != null || value.swimPower != null) && value.sport !== 'swim') ||
    !isHeartRatePhysiology(value.heartRatePhysiology, value.elapsedTimeS) ||
    !isWalkPower(value.walkPower, value.elapsedTimeS, value.date) ||
    (value.walkPower != null && value.sport !== 'walk') ||
    value.elapsedTimeS < 0 ||
    value.elapsedTimeS > Number.MAX_SAFE_INTEGER ||
    !isCyclingTorqueTrace(value.cyclingTorque, value.elapsedTimeS) ||
    (value.cyclingTorque != null && (value.sport !== 'bike' || value.wahoo === undefined)) ||
    !isCyclingIntensityTrace(value.cyclingIntensityTrace, value.elapsedTimeS) ||
    (value.cyclingIntensityTrace != null &&
      (value.sport !== 'bike' || value.wahoo === undefined)) ||
    !isCyclingPowerTrace(value.cyclingPowerTrace, value.elapsedTimeS) ||
    (isRecord(value.cyclingPowerTrace) &&
      (value.sport !== 'bike' ||
        (value.cyclingPowerTrace.source === 'wahoo' && value.wahoo === undefined) ||
        (value.cyclingPowerTrace.source === 'strava' && value.deviceWatts !== true))) ||
    !isRunStrideCadenceTrace(value.runStrideCadence, value.elapsedTimeS) ||
    !isRunHeartCostTrace(value.runHeartCost, value.elapsedTimeS) ||
    ((value.runStrideCadence != null || value.runHeartCost != null) && value.sport !== 'run') ||
    !nullableBounded(value.deviceTemperatureC, -90, 100) ||
    !nullableBounded(value.ambientTemperatureC, -90, 70) ||
    !isRouteTrace(value.route) ||
    !isThermalTrace(value.heartRateTrace) ||
    !isRunWalk(value.runWalk) ||
    (value.runWalk !== null && value.sport !== 'run')
  )
    return false
  return isActivityAnalyses(value.analyses, value.id, value.elapsedTimeS, value.start)
}
