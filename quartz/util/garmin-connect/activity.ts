import {
  emptyGarminFueling,
  emptyGarminMetrics,
  hasGarminFueling,
  hasGarminMetrics,
  normalizeGarminSport,
  type GarminActivity,
  type GarminRunningDynamicsSummary,
} from '../../plugins/stores/garmin'
import { isRecord, type UnknownRecord } from '../type-guards'
import { firstNumber, firstString, normalizeDate, positive, rounded, roundedFloat } from './scalars'

const RECORD_KEYS = [
  'activity',
  'activityDTO',
  'activityDetail',
  'activityDetailDTO',
  'activitySummary',
  'details',
  'fueling',
  'hydration',
  'metadataDTO',
  'nutrition',
  'summary',
  'summaryDTO',
]

const ID_KEYS = ['garminActivityId', 'activityId', 'activityIdStr', 'id', 'summaryId', 'uuid']
const NAME_KEYS = ['activityName', 'name', 'title']
const SPORT_KEYS = ['activityType', 'eventType', 'sport', 'sportType', 'type']
const SPORT_NESTED_KEYS = ['name', 'type', 'typeKey']
const START_UTC_KEYS = [
  'beginTimestamp',
  'startDate',
  'startTime',
  'startTimeGMT',
  'startTimeGmt',
  'startTimeInSeconds',
  'startedAt',
]
const START_LOCAL_KEYS = ['startDateLocal', 'startLocal', 'startTimeLocal', 'startedAtLocal']
const DISTANCE_M_KEYS = ['distance', 'distanceInMeters', 'distanceM', 'distanceMeters']
const DISTANCE_KM_KEYS = ['distanceKm', 'distanceKilometers']
const MOVING_S_KEYS = ['movingDuration', 'movingDurationS', 'movingTimeS']
const MOVING_MS_KEYS = ['movingDurationMs', 'movingTimeMs']
const ELAPSED_S_KEYS = [
  'elapsedDuration',
  'elapsedDurationS',
  'elapsedTimeS',
  'duration',
  'durationInSeconds',
  'durationS',
]
const ELAPSED_MS_KEYS = ['elapsedDurationMs', 'elapsedTimeMs', 'durationMs']
const DEVICE_KEYS = [
  'activityDeviceName',
  'deviceDisplayName',
  'deviceModel',
  'deviceName',
  'sourceDevice',
]

const TOTAL_CALORIES_KEYS = [
  'activeKilocalories',
  'calories',
  'caloriesBurned',
  'kilocalories',
  'totalCalories',
]
const METABOLIC_CALORIES_KEYS = ['bmrCalories', 'metabolicCalories']
const AVG_HR_KEYS = ['averageHR', 'averageHeartRate', 'averageHeartRateInBeatsPerMinute', 'avgHr']
const MAX_HR_KEYS = ['maxHR', 'maxHeartRate', 'maxHeartRateInBeatsPerMinute']
const AVG_POWER_KEYS = ['averagePower', 'avgPower']
const NORMALIZED_POWER_KEYS = ['normalizedPower', 'weightedAverageWatts']
const MAX_POWER_KEYS = ['maxPower', 'maxPowerInWatts']
const AVG_CADENCE_KEYS = ['averageBikeCadence', 'averageSwimCadence', 'avgCadence']
const ASCENT_M_KEYS = ['elevationGain', 'totalAscent', 'totalAscentM', 'totalElevationGain']
const DESCENT_M_KEYS = ['elevationLoss', 'totalDescent', 'totalDescentM', 'totalElevationLoss']
const WORK_KJ_KEYS = ['kilojoules', 'totalWorkKJ']
const WORK_KCAL_KEYS = ['totalWork']
const TSS_KEYS = ['trainingStressScore', 'tss']
const IF_KEYS = ['intensityFactor']
const AEROBIC_TRAINING_EFFECT_KEYS = ['aerobicTrainingEffect', 'trainingEffect']
const ANAEROBIC_TRAINING_EFFECT_KEYS = ['anaerobicTrainingEffect']
const EXERCISE_LOAD_KEYS = ['activityTrainingLoad', 'exerciseLoad']
const TRAINING_EFFECT_LABEL_KEYS = ['trainingEffectLabel']
const AEROBIC_TRAINING_EFFECT_MESSAGE_KEYS = ['aerobicTrainingEffectMessage']
const ANAEROBIC_TRAINING_EFFECT_MESSAGE_KEYS = ['anaerobicTrainingEffectMessage']
const AVG_RESPIRATION_RATE_KEYS = ['avgRespirationRate', 'averageRespirationRate']
const AVG_STRIDE_LENGTH_CM_KEYS = ['strideLength', 'averageStrideLength']
const AVG_VERTICAL_RATIO_PCT_KEYS = ['verticalRatio', 'averageVerticalRatio']
const AVG_VERTICAL_OSCILLATION_CM_KEYS = ['verticalOscillation', 'averageVerticalOscillation']
const AVG_GROUND_CONTACT_BALANCE_LEFT_PCT_KEYS = [
  'groundContactBalanceLeft',
  'averageGroundContactBalanceLeft',
]
const AVG_GROUND_CONTACT_TIME_MS_KEYS = ['groundContactTime', 'averageGroundContactTime']
const AVG_STEP_SPEED_LOSS_MPS_KEYS = ['stepSpeedLoss', 'averageStepSpeedLoss']
const AVG_STEP_SPEED_LOSS_PCT_KEYS = ['stepSpeedLossPercent', 'averageStepSpeedLossPercent']
const IMPACT_LOAD_M_KEYS = ['impactLoad']
const KJ_PER_KCAL = 4.184

const CALORIES_CONSUMED_KEYS = [
  'caloriesConsumed',
  'caloriesConsumedInKcal',
  'caloriesConsumedKcal',
  'caloriesIntake',
  'caloriesIntakeKcal',
  'consumedCalories',
  'nutritionCalories',
]
const CARBS_KEYS = [
  'carbIntakeG',
  'carbohydrateIntakeG',
  'carbohydratesConsumed',
  'carbohydratesConsumedG',
  'carbsConsumed',
  'carbsConsumedG',
  'consumedCarbs',
]
const CARBS_RECOMMENDED_KEYS = [
  'carbohydratesRecommendedG',
  'carbsRecommendedG',
  'recommendedCarbohydratesG',
  'recommendedCarbsG',
]
const FLUID_ML_KEYS = [
  'fluidConsumedInMl',
  'fluidConsumedMl',
  'fluidIntakeInMl',
  'fluidIntakeMl',
  'fluidMl',
  'hydrationMl',
  'waterConsumedMl',
  'waterIntakeMl',
]
const FLUID_L_KEYS = ['fluidConsumedL', 'fluidIntakeL', 'fluidL', 'fluidLiters', 'waterL']
const FLUID_OZ_KEYS = ['fluidConsumedOz', 'fluidIntakeOz', 'fluidOunces', 'fluidOz', 'waterOz']
const FLUID_RECOMMENDED_ML_KEYS = [
  'fluidRecommendedMl',
  'recommendedFluidMl',
  'recommendedHydrationMl',
  'recommendedWaterMl',
]
const FLUID_RECOMMENDED_L_KEYS = ['fluidRecommendedL', 'recommendedFluidL', 'recommendedWaterL']
const FLUID_RECOMMENDED_OZ_KEYS = ['fluidRecommendedOz', 'recommendedFluidOz', 'recommendedWaterOz']
const SWEAT_ML_KEYS = ['estimatedSweatLossMl', 'sweatLoss', 'sweatLossInMl', 'sweatLossMl']
const SWEAT_L_KEYS = ['estimatedSweatLossL', 'sweatLossL']
const SWEAT_OZ_KEYS = ['estimatedSweatLossOz', 'sweatLossOz']

function roundedNonnegativeFloat(value: number | null, dp: number): number | null {
  if (value == null || !Number.isFinite(value) || value < 0) return null
  const factor = 10 ** dp
  return Math.round(value * factor) / factor
}

function collectRecords(root: UnknownRecord): UnknownRecord[] {
  const out: UnknownRecord[] = []
  const queue: UnknownRecord[] = [root]
  const seen = new Set<UnknownRecord>()
  for (let i = 0; i < queue.length; i++) {
    const record = queue[i]
    if (seen.has(record)) continue
    seen.add(record)
    out.push(record)
    for (const key of RECORD_KEYS) {
      const child = record[key]
      if (isRecord(child)) queue.push(child)
    }
  }
  return out
}

function firstDateValue(
  records: readonly UnknownRecord[],
  keys: readonly string[],
): string | number | null {
  for (const record of records) {
    for (const key of keys) {
      const value = record[key]
      if (typeof value === 'string' && value.trim()) return value.trim()
      if (typeof value === 'number' && Number.isFinite(value)) return value
    }
  }
  return null
}

function firstSport(records: readonly UnknownRecord[]): string | null {
  const direct = firstString(records, SPORT_KEYS)
  if (direct) return direct
  for (const record of records) {
    for (const key of SPORT_KEYS) {
      const child = record[key]
      if (!isRecord(child)) continue
      const nested = firstString([child], SPORT_NESTED_KEYS)
      if (nested) return nested
    }
  }
  return null
}

function normalizeLocalDate(value: string | null, fallback: string): string {
  if (!value) return fallback
  return value.trim().replace(' ', 'T')
}

function activityStartDate(records: readonly UnknownRecord[]): string | null {
  return normalizeDate(
    firstDateValue(records, START_UTC_KEYS) ?? firstDateValue(records, START_LOCAL_KEYS),
  )
}

export function garminConnectActivityStartDate(record: UnknownRecord): string | null {
  return activityStartDate(collectRecords(record))
}

function ml(
  records: readonly UnknownRecord[],
  mlKeys: readonly string[],
  literKeys: readonly string[],
  ounceKeys: readonly string[],
): number | null {
  const direct = rounded(firstNumber(records, mlKeys))
  if (direct != null) return direct
  const liters = positive(firstNumber(records, literKeys))
  if (liters != null) return Math.round(liters * 1000)
  const ounces = positive(firstNumber(records, ounceKeys))
  return ounces == null ? null : Math.round(ounces * 29.5735)
}

export function activityId(record: UnknownRecord): string | null {
  const id = firstString([record], ID_KEYS)
  return id?.trim() || null
}

function hasGarminActivityData(activity: GarminActivity): boolean {
  return (
    hasGarminFueling(activity.fueling) ||
    hasGarminMetrics(activity.metrics) ||
    activity.runningDynamics != null ||
    activity.distanceM != null ||
    activity.movingTimeS != null ||
    activity.elapsedTimeS != null
  )
}

function garminRunningDynamicsSummary(
  records: readonly UnknownRecord[],
): GarminRunningDynamicsSummary | null {
  const summary: GarminRunningDynamicsSummary = {
    source: 'garmin',
    averageRespirationRate: roundedNonnegativeFloat(
      firstNumber(records, AVG_RESPIRATION_RATE_KEYS),
      2,
    ),
    averageStrideLengthCm: roundedNonnegativeFloat(
      firstNumber(records, AVG_STRIDE_LENGTH_CM_KEYS),
      2,
    ),
    averageVerticalRatioPct: roundedNonnegativeFloat(
      firstNumber(records, AVG_VERTICAL_RATIO_PCT_KEYS),
      2,
    ),
    averageVerticalOscillationCm: roundedNonnegativeFloat(
      firstNumber(records, AVG_VERTICAL_OSCILLATION_CM_KEYS),
      2,
    ),
    averageGroundContactBalanceLeftPct: roundedNonnegativeFloat(
      firstNumber(records, AVG_GROUND_CONTACT_BALANCE_LEFT_PCT_KEYS),
      2,
    ),
    averageGroundContactTimeMs: roundedNonnegativeFloat(
      firstNumber(records, AVG_GROUND_CONTACT_TIME_MS_KEYS),
      2,
    ),
    averageStepSpeedLossMps: roundedNonnegativeFloat(
      firstNumber(records, AVG_STEP_SPEED_LOSS_MPS_KEYS),
      4,
    ),
    averageStepSpeedLossPct: roundedNonnegativeFloat(
      firstNumber(records, AVG_STEP_SPEED_LOSS_PCT_KEYS),
      2,
    ),
    impactLoadM: roundedNonnegativeFloat(firstNumber(records, IMPACT_LOAD_M_KEYS), 1),
  }
  return Object.entries(summary).some(([key, value]) => key !== 'source' && value != null)
    ? summary
    : null
}

export function garminConnectActivity(
  detail: UnknownRecord | null,
  fallback: UnknownRecord,
  index: number,
): GarminActivity | null {
  const records = detail
    ? [...collectRecords(detail), ...collectRecords(fallback)]
    : collectRecords(fallback)
  const localRaw = firstString(records, START_LOCAL_KEYS)
  const startDate = activityStartDate(records)
  if (!startDate) return null

  const sourceDevice = firstString(records, DEVICE_KEYS)
  const distanceM =
    rounded(firstNumber(records, DISTANCE_M_KEYS)) ??
    rounded((firstNumber(records, DISTANCE_KM_KEYS) ?? 0) * 1000)
  const movingTimeS =
    rounded(firstNumber(records, MOVING_S_KEYS)) ??
    rounded((firstNumber(records, MOVING_MS_KEYS) ?? 0) / 1000)
  const elapsedTimeS =
    rounded(firstNumber(records, ELAPSED_S_KEYS)) ??
    rounded((firstNumber(records, ELAPSED_MS_KEYS) ?? 0) / 1000)

  const metrics = emptyGarminMetrics()
  metrics.totalCalories = rounded(firstNumber(records, TOTAL_CALORIES_KEYS))
  metrics.metabolicCalories = rounded(firstNumber(records, METABOLIC_CALORIES_KEYS))
  metrics.avgHeartRate = rounded(firstNumber(records, AVG_HR_KEYS))
  metrics.maxHeartRate = rounded(firstNumber(records, MAX_HR_KEYS))
  metrics.avgPower = rounded(firstNumber(records, AVG_POWER_KEYS))
  metrics.normalizedPower = rounded(firstNumber(records, NORMALIZED_POWER_KEYS))
  metrics.maxPower = rounded(firstNumber(records, MAX_POWER_KEYS))
  const cadence = firstNumber(records, AVG_CADENCE_KEYS)
  const runCadence = firstNumber(records, ['averageRunCadence'])
  // Run cadence counts both feet; the shared activity contract uses cycles per minute.
  metrics.avgCadence =
    cadence != null ? rounded(cadence) : runCadence != null ? roundedFloat(runCadence / 2, 1) : null
  metrics.totalAscentM = rounded(firstNumber(records, ASCENT_M_KEYS))
  metrics.totalDescentM = rounded(firstNumber(records, DESCENT_M_KEYS))
  metrics.totalWorkKJ =
    roundedFloat(firstNumber(records, WORK_KJ_KEYS), 1) ??
    roundedFloat((firstNumber(records, WORK_KCAL_KEYS) ?? 0) * KJ_PER_KCAL, 1)
  metrics.trainingStressScore = roundedFloat(firstNumber(records, TSS_KEYS), 1)
  metrics.intensityFactor = roundedFloat(firstNumber(records, IF_KEYS), 3)
  metrics.aerobicTrainingEffect = roundedNonnegativeFloat(
    firstNumber(records, AEROBIC_TRAINING_EFFECT_KEYS),
    1,
  )
  metrics.anaerobicTrainingEffect = roundedNonnegativeFloat(
    firstNumber(records, ANAEROBIC_TRAINING_EFFECT_KEYS),
    1,
  )
  metrics.exerciseLoad = roundedNonnegativeFloat(firstNumber(records, EXERCISE_LOAD_KEYS), 1)
  metrics.trainingEffectLabel = firstString(records, TRAINING_EFFECT_LABEL_KEYS)
  metrics.aerobicTrainingEffectMessage = firstString(records, AEROBIC_TRAINING_EFFECT_MESSAGE_KEYS)
  metrics.anaerobicTrainingEffectMessage = firstString(
    records,
    ANAEROBIC_TRAINING_EFFECT_MESSAGE_KEYS,
  )

  const fueling = emptyGarminFueling(sourceDevice)
  fueling.caloriesConsumed = rounded(firstNumber(records, CALORIES_CONSUMED_KEYS))
  fueling.carbsConsumedG = rounded(firstNumber(records, CARBS_KEYS))
  fueling.fluidMl = ml(records, FLUID_ML_KEYS, FLUID_L_KEYS, FLUID_OZ_KEYS)
  fueling.carbsRecommendedG = rounded(firstNumber(records, CARBS_RECOMMENDED_KEYS))
  fueling.fluidRecommendedMl = ml(
    records,
    FLUID_RECOMMENDED_ML_KEYS,
    FLUID_RECOMMENDED_L_KEYS,
    FLUID_RECOMMENDED_OZ_KEYS,
  )
  fueling.sweatLossMl = ml(records, SWEAT_ML_KEYS, SWEAT_L_KEYS, SWEAT_OZ_KEYS)

  const activity: GarminActivity = {
    id: `connect:${firstString(records, ID_KEYS) ?? `${startDate}:${index}`}`,
    name: firstString(records, NAME_KEYS),
    sport: normalizeGarminSport(firstSport(records)),
    startDate,
    startDateLocal: normalizeLocalDate(localRaw, startDate),
    distanceM,
    movingTimeS,
    elapsedTimeS,
    sourceDevice,
    sourceFile: null,
    metrics,
    runningDynamics: garminRunningDynamicsSummary(records),
    fueling,
  }
  return hasGarminActivityData(activity) ? activity : null
}
