import {
  emptyGarminFueling,
  emptyGarminMetrics,
  type GarminActivity,
  hasGarminFueling,
  hasGarminMetrics,
  normalizeGarminSport,
} from '../plugins/stores/garmin'
import { isRecord, readNumber, readString, type UnknownRecord } from './type-guards'

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
  'duration',
  'durationInSeconds',
  'durationS',
  'elapsedDuration',
  'elapsedDurationS',
  'elapsedTimeS',
]
const ELAPSED_MS_KEYS = ['durationMs', 'elapsedDurationMs', 'elapsedTimeMs']
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
const AVG_CADENCE_KEYS = [
  'averageBikeCadence',
  'averageRunCadence',
  'averageSwimCadence',
  'avgCadence',
]
const ASCENT_M_KEYS = ['elevationGain', 'totalAscent', 'totalAscentM', 'totalElevationGain']
const DESCENT_M_KEYS = ['elevationLoss', 'totalDescent', 'totalDescentM', 'totalElevationLoss']
const WORK_KJ_KEYS = ['kilojoules', 'totalWorkKJ']
const WORK_J_KEYS = ['totalWork']
const TSS_KEYS = ['trainingStressScore', 'tss']
const IF_KEYS = ['intensityFactor']

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

export interface GarminConnectActivityListItem {
  id: string
  record: UnknownRecord
}

function numeric(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value !== 'string') return null
  const parsed = Number(value.replace(/,/g, '').trim())
  return Number.isFinite(parsed) ? parsed : null
}

function positive(value: number | null): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? value : null
}

function rounded(value: number | null): number | null {
  const n = positive(value)
  return n == null ? null : Math.round(n)
}

function roundedFloat(value: number | null, dp: number): number | null {
  const n = positive(value)
  if (n == null) return null
  const factor = 10 ** dp
  return Math.round(n * factor) / factor
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

function firstNumber(records: readonly UnknownRecord[], keys: readonly string[]): number | null {
  for (const record of records) {
    for (const key of keys) {
      const value = readNumber(record, key) ?? numeric(record[key])
      if (value != null) return value
    }
  }
  return null
}

function firstString(records: readonly UnknownRecord[], keys: readonly string[]): string | null {
  for (const record of records) {
    for (const key of keys) {
      const value = readString(record, key)
      if (value?.trim()) return value.trim()
      const n = readNumber(record, key)
      if (n != null) return String(n)
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

function normalizeDate(value: string | number | Date | null): string | null {
  if (!value) return null
  if (value instanceof Date) return Number.isFinite(value.valueOf()) ? value.toISOString() : null
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null
    const ms = value > 1_000_000_000_000 ? value : value * 1000
    return new Date(ms).toISOString()
  }
  const trimmed = value.trim().replace(' ', 'T')
  if (!trimmed) return null
  const zoned = /(?:Z|[+-]\d{2}:?\d{2})$/.test(trimmed) ? trimmed : `${trimmed}Z`
  const ms = Date.parse(zoned)
  return Number.isFinite(ms) ? new Date(ms).toISOString() : null
}

function normalizeLocalDate(value: string | null, fallback: string): string {
  if (!value) return fallback
  return value.trim().replace(' ', 'T')
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

function activityId(record: UnknownRecord): string | null {
  const id = firstString([record], ID_KEYS)
  return id?.trim() || null
}

function hasGarminActivityData(activity: GarminActivity): boolean {
  return (
    hasGarminFueling(activity.fueling) ||
    hasGarminMetrics(activity.metrics) ||
    activity.distanceM != null ||
    activity.movingTimeS != null ||
    activity.elapsedTimeS != null
  )
}

function recordsFromJson(raw: unknown): UnknownRecord[] {
  if (Array.isArray(raw)) return raw.filter(isRecord)
  if (!isRecord(raw)) return []
  if (Array.isArray(raw.activities)) return raw.activities.filter(isRecord)
  if (Array.isArray(raw.data)) return raw.data.filter(isRecord)
  if (isRecord(raw.data) && raw.data.searchActivitiesScalar != null)
    return recordsFromJson(graphqlScalar(raw.data.searchActivitiesScalar))
  if (isRecord(raw.data) && Array.isArray(raw.data.activities))
    return raw.data.activities.filter(isRecord)
  return []
}

function graphqlScalar(raw: unknown): unknown {
  if (typeof raw !== 'string') return raw
  try {
    return JSON.parse(raw) as unknown
  } catch {
    return null
  }
}

export function garminConnectActivities(raw: unknown): GarminConnectActivityListItem[] {
  const out: GarminConnectActivityListItem[] = []
  const seen = new Set<string>()
  for (const record of recordsFromJson(raw)) {
    const id = activityId(record)
    if (!id || seen.has(id)) continue
    seen.add(id)
    out.push({ id, record })
  }
  return out
}

export function garminConnectActivity(
  detail: UnknownRecord | null,
  fallback: UnknownRecord,
  index: number,
): GarminActivity | null {
  const records = detail
    ? [...collectRecords(detail), ...collectRecords(fallback)]
    : collectRecords(fallback)
  const utcRaw = firstString(records, START_UTC_KEYS)
  const localRaw = firstString(records, START_LOCAL_KEYS)
  const startDate = normalizeDate(utcRaw ?? localRaw)
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
  metrics.avgCadence = rounded(firstNumber(records, AVG_CADENCE_KEYS))
  metrics.totalAscentM = rounded(firstNumber(records, ASCENT_M_KEYS))
  metrics.totalDescentM = rounded(firstNumber(records, DESCENT_M_KEYS))
  metrics.totalWorkKJ =
    roundedFloat(firstNumber(records, WORK_KJ_KEYS), 1) ??
    roundedFloat((firstNumber(records, WORK_J_KEYS) ?? 0) / 1000, 1)
  metrics.trainingStressScore = roundedFloat(firstNumber(records, TSS_KEYS), 1)
  metrics.intensityFactor = roundedFloat(firstNumber(records, IF_KEYS), 3)

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
    fueling,
  }
  return hasGarminActivityData(activity) ? activity : null
}
