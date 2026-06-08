import fs from 'node:fs/promises'
import {
  emptyGarminFueling,
  type GarminActivity,
  type GarminCache,
  hasGarminFueling,
  normalizeGarminSport,
} from '../plugins/stores/garmin'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, readNumber, readString, type UnknownRecord } from '../util/type-guards'

const CACHE_VERSION = 1
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
const defaultInputDir = joinSegments(QUARTZ, '.quartz-cache', 'garmin-fueling')

const RECORD_KEYS = [
  'activity',
  'activityDTO',
  'activityDetail',
  'activityDetailDTO',
  'activitySummary',
  'fueling',
  'hydration',
  'metadataDTO',
  'nutrition',
  'summary',
  'summaryDTO',
]

const ID_KEYS = ['garminActivityId', 'activityId', 'activityIdStr', 'id', 'uuid']
const NAME_KEYS = ['activityName', 'name', 'title']
const SPORT_KEYS = ['activityType', 'eventType', 'sport', 'sportType', 'type']
const SPORT_NESTED_KEYS = ['name', 'type', 'typeKey']
const START_UTC_KEYS = [
  'beginTimestamp',
  'startDate',
  'startTime',
  'startTimeGMT',
  'startTimeGmt',
  'startedAt',
]
const START_LOCAL_KEYS = ['startDateLocal', 'startLocal', 'startTimeLocal', 'startedAtLocal']
const DISTANCE_M_KEYS = ['distance', 'distanceInMeters', 'distanceM', 'distanceMeters']
const DISTANCE_KM_KEYS = ['distanceKm', 'distanceKilometers']
const MOVING_S_KEYS = ['movingDuration', 'movingDurationS', 'movingTimeS']
const MOVING_MS_KEYS = ['movingDurationMs', 'movingTimeMs']
const ELAPSED_S_KEYS = [
  'duration',
  'durationS',
  'elapsedDuration',
  'elapsedDurationS',
  'elapsedTimeS',
]
const ELAPSED_MS_KEYS = ['durationMs', 'elapsedDurationMs', 'elapsedTimeMs']
const DEVICE_KEYS = ['activityDeviceName', 'deviceModel', 'deviceName', 'sourceDevice']

const CALORIES_KEYS = [
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

function numeric(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value !== 'string') return null
  const parsed = Number(value.replace(/,/g, '').trim())
  return Number.isFinite(parsed) ? parsed : null
}

function positive(value: number | null): number | null {
  return value != null && value > 0 ? value : null
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

function normalizeDate(value: string | null): string | null {
  if (!value) return null
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

function rounded(value: number | null): number | null {
  const n = positive(value)
  return n == null ? null : Math.round(n)
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

function activityRecords(raw: unknown): UnknownRecord[] {
  if (Array.isArray(raw)) return raw.filter(isRecord)
  if (!isRecord(raw)) return []
  if (Array.isArray(raw.activities)) return raw.activities.filter(isRecord)
  if (isRecord(raw.activities)) return Object.values(raw.activities).filter(isRecord)
  if (Array.isArray(raw.data)) return raw.data.filter(isRecord)
  if (isRecord(raw.data)) return Object.values(raw.data).filter(isRecord)
  return [raw]
}

function toActivity(record: UnknownRecord, index: number): GarminActivity | null {
  const records = collectRecords(record)
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
  const fueling = emptyGarminFueling(sourceDevice)
  fueling.caloriesConsumed = rounded(firstNumber(records, CALORIES_KEYS))
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
  if (!hasGarminFueling(fueling)) return null
  return {
    id: firstString(records, ID_KEYS) ?? `${startDate}:${index}`,
    name: firstString(records, NAME_KEYS),
    sport: normalizeGarminSport(firstSport(records)),
    startDate,
    startDateLocal: normalizeLocalDate(localRaw, startDate),
    distanceM,
    movingTimeS,
    elapsedTimeS,
    sourceDevice,
    fueling,
  }
}

async function inputFiles(): Promise<string[]> {
  const file = process.env.GARMIN_FUELING_FILE
  if (file?.trim()) return [file.trim()]
  const dir = process.env.GARMIN_FUELING_DIR?.trim() || defaultInputDir
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true })
    return entries
      .filter(entry => entry.isFile() && entry.name.endsWith('.json'))
      .map(entry => joinSegments(dir, entry.name))
      .sort()
  } catch {
    return []
  }
}

async function main(): Promise<void> {
  const files = await inputFiles()
  if (files.length === 0) {
    console.log(
      `[garmin] no JSON input found (set GARMIN_FUELING_FILE or GARMIN_FUELING_DIR, default ${defaultInputDir})`,
    )
    return
  }

  const activities: Record<string, GarminActivity> = {}
  let index = 0
  for (const file of files) {
    const raw: unknown = JSON.parse(await fs.readFile(file, 'utf8'))
    for (const record of activityRecords(raw)) {
      const activity = toActivity(record, index++)
      if (activity) activities[activity.id] = activity
    }
  }

  const sorted: Record<string, GarminActivity> = {}
  for (const activity of Object.values(activities).sort((a, b) =>
    a.startDate.localeCompare(b.startDate),
  )) {
    sorted[activity.id] = activity
  }
  const cache: GarminCache = { version: CACHE_VERSION, lastSync: Date.now(), activities: sorted }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  console.log(`[garmin] wrote ${Object.keys(sorted).length} fueling activities → ${cacheFile}`)
}

main().catch(err => {
  console.error(`[garmin] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
