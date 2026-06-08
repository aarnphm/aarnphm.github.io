import {
  Decoder,
  Stream,
  type FileIdMesg,
  type FitMessages,
  type SessionMesg,
} from '@garmin/fitsdk'
import fs from 'node:fs/promises'
import { basename, extname } from 'node:path'
import {
  emptyGarminFueling,
  emptyGarminMetrics,
  type GarminActivity,
  type GarminCache,
  hasGarminMetrics,
  hasGarminFueling,
  normalizeGarminSport,
} from '../plugins/stores/garmin'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, readNumber, readString, type UnknownRecord } from '../util/type-guards'

const CACHE_VERSION = 1
const FIT_EPOCH_MS = Date.UTC(1989, 11, 31)
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
const defaultInputDir = joinSegments(QUARTZ, '.quartz-cache', 'garmin-fueling')
const defaultFitDir = joinSegments(QUARTZ, '.quartz-cache', 'garmin-fit')

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

function normalizeDate(value: string | number | Date | null): string | null {
  if (!value) return null
  if (value instanceof Date) return Number.isFinite(value.valueOf()) ? value.toISOString() : null
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null
    const ms = value > 1_000_000_000_000 ? value : FIT_EPOCH_MS + value * 1000
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

function hasGarminActivityData(activity: GarminActivity): boolean {
  return (
    hasGarminFueling(activity.fueling) ||
    hasGarminMetrics(activity.metrics) ||
    activity.distanceM != null ||
    activity.movingTimeS != null ||
    activity.elapsedTimeS != null
  )
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
  const activity: GarminActivity = {
    id: firstString(records, ID_KEYS) ?? `${startDate}:${index}`,
    name: firstString(records, NAME_KEYS),
    sport: normalizeGarminSport(firstSport(records)),
    startDate,
    startDateLocal: normalizeLocalDate(localRaw, startDate),
    distanceM,
    movingTimeS,
    elapsedTimeS,
    sourceDevice,
    sourceFile: null,
    metrics: emptyGarminMetrics(),
    fueling,
  }
  return hasGarminActivityData(activity) ? activity : null
}

function productName(value: string | number | undefined): string | null {
  if (value == null) return null
  const clean = String(value).trim()
  if (!clean) return null
  const edge = clean.match(/^edge(\d+)$/i)
  if (edge) return `Edge ${edge[1]}`
  return clean
}

function sourceDevice(fileId: FileIdMesg | undefined, messages: FitMessages): string | null {
  return (
    productName(fileId?.productName) ??
    productName(fileId?.garminProduct) ??
    productName(messages.deviceInfoMesgs?.find(device => device.garminProduct)?.garminProduct) ??
    productName(fileId?.product)
  )
}

function isFitActivity(messages: FitMessages): boolean {
  const type = messages.fileIdMesgs?.[0]?.type
  return type === 'activity' || type === 4
}

function fitStart(session: SessionMesg, fileId: FileIdMesg | undefined): string | null {
  return normalizeDate(session.startTime ?? session.timestamp ?? fileId?.timeCreated ?? null)
}

function toFitActivity(file: string, messages: FitMessages): GarminActivity | null {
  if (!isFitActivity(messages)) return null
  const session = messages.sessionMesgs?.[0]
  if (!session) return null
  const fileId = messages.fileIdMesgs?.[0]
  const startDate = fitStart(session, fileId)
  if (!startDate) return null
  const device = sourceDevice(fileId, messages)
  const sourceFile = basename(file)
  const metrics = emptyGarminMetrics()
  metrics.totalCalories = rounded(session.totalCalories ?? null)
  metrics.metabolicCalories = rounded(session.metabolicCalories ?? null)
  metrics.avgHeartRate = rounded(session.avgHeartRate ?? null)
  metrics.maxHeartRate = rounded(session.maxHeartRate ?? null)
  metrics.avgPower = rounded(session.avgPower ?? null)
  metrics.normalizedPower = rounded(session.normalizedPower ?? null)
  metrics.maxPower = rounded(session.maxPower ?? null)
  metrics.avgCadence = rounded(session.avgCadence ?? null)
  metrics.totalAscentM = rounded(session.totalAscent ?? null)
  metrics.totalDescentM = rounded(session.totalDescent ?? null)
  metrics.totalWorkKJ = roundedFloat(session.totalWork != null ? session.totalWork / 1000 : null, 1)
  metrics.trainingStressScore = roundedFloat(session.trainingStressScore ?? null, 1)
  metrics.intensityFactor = roundedFloat(session.intensityFactor ?? null, 3)
  const activity: GarminActivity = {
    id: `fit:${sourceFile.replace(/\.fit$/i, '')}`,
    name: session.sportProfileName ?? sourceFile,
    sport: normalizeGarminSport(typeof session.sport === 'string' ? session.sport : null),
    startDate,
    startDateLocal: startDate,
    distanceM: rounded(session.totalDistance ?? null),
    movingTimeS: rounded(session.totalTimerTime ?? null),
    elapsedTimeS: rounded(session.totalElapsedTime ?? null),
    sourceDevice: device,
    sourceFile,
    metrics,
    fueling: emptyGarminFueling(device),
  }
  return hasGarminActivityData(activity) ? activity : null
}

async function fitActivity(file: string): Promise<GarminActivity | null> {
  const bytes = await fs.readFile(file)
  const decoder = new Decoder(Stream.fromByteArray(bytes))
  const result = decoder.read()
  if (result.errors.length > 0) {
    const summary = result.errors.map(error => error.message).join('; ')
    console.warn(`[garmin] decoded ${file} with ${result.errors.length} FIT error(s): ${summary}`)
  }
  return toFitActivity(file, result.messages)
}

async function filesInDir(dir: string, extension: string): Promise<string[]> {
  const out: string[] = []
  async function walk(current: string): Promise<void> {
    const entries = await fs.readdir(current, { withFileTypes: true })
    for (const entry of entries) {
      const file = joinSegments(current, entry.name)
      if (entry.isDirectory()) await walk(file)
      else if (entry.isFile() && extname(entry.name).toLowerCase() === extension) out.push(file)
    }
  }
  try {
    await walk(dir)
  } catch {
    return []
  }
  return out.sort()
}

async function jsonInputFiles(): Promise<string[]> {
  const file = process.env.GARMIN_FUELING_FILE
  if (file?.trim()) return [file.trim()]
  const dir = process.env.GARMIN_FUELING_DIR?.trim() || defaultInputDir
  return filesInDir(dir, '.json')
}

async function fitInputFiles(): Promise<string[]> {
  const file = process.env.GARMIN_FIT_FILE
  if (file?.trim()) return [file.trim()]
  const dir = process.env.GARMIN_FIT_DIR?.trim() || defaultFitDir
  return filesInDir(dir, '.fit')
}

async function main(): Promise<void> {
  const jsonFiles = await jsonInputFiles()
  const fitFiles = await fitInputFiles()
  if (jsonFiles.length === 0 && fitFiles.length === 0) {
    console.log(
      `[garmin] no input found (set GARMIN_FIT_FILE/GARMIN_FIT_DIR or GARMIN_FUELING_FILE/GARMIN_FUELING_DIR; defaults ${defaultFitDir} and ${defaultInputDir})`,
    )
    return
  }

  const activities: Record<string, GarminActivity> = {}
  let fitCount = 0
  for (const file of fitFiles) {
    const activity = await fitActivity(file)
    if (activity) {
      activities[activity.id] = activity
      fitCount++
    }
  }

  let index = 0
  for (const file of jsonFiles) {
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
  const fuelingCount = Object.values(sorted).filter(activity =>
    hasGarminFueling(activity.fueling),
  ).length
  console.log(
    `[garmin] wrote ${Object.keys(sorted).length} activities (${fitCount} FIT, ${fuelingCount} fueling) → ${cacheFile}`,
  )
}

main().catch(err => {
  console.error(`[garmin] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
