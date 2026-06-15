import {
  emptyGarminFueling,
  emptyGarminMetrics,
  type GarminActivity,
  type GarminStreams,
  type GarminVo2Day,
  type GarminWeightSample,
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
const METRIC_KEYS = {
  altitude: 'directElevation',
  cadence: 'directBikeCadence',
  distance: 'sumDistance',
  heartRate: 'directHeartRate',
  latitude: 'directLatitude',
  longitude: 'directLongitude',
  power: 'directPower',
}

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

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : numeric(value)
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

function vo2Of(value: unknown): number | null {
  if (!isRecord(value)) return null
  return finite(value.vo2MaxPreciseValue) ?? finite(value.vo2MaxValue)
}

export function garminConnectVo2(raw: unknown): GarminVo2Day[] {
  if (!Array.isArray(raw)) return []
  const out: GarminVo2Day[] = []
  for (const item of raw) {
    if (!isRecord(item)) continue
    const generic = isRecord(item.generic) ? item.generic : null
    const cycling = isRecord(item.cycling) ? item.cycling : null
    const date =
      readString(item, 'calendarDate') ??
      (generic ? readString(generic, 'calendarDate') : null) ??
      (cycling ? readString(cycling, 'calendarDate') : null)
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    const g = vo2Of(generic)
    const c = vo2Of(cycling)
    if (g == null && c == null) continue
    out.push({ date, generic: g, cycling: c })
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}

const kgOf = (value: unknown): number | null => {
  const n = finite(value)
  if (n == null || n <= 0) return null
  return Math.round((n > 400 ? n / 1000 : n) * 100) / 100
}

const pctOf = (value: unknown): number | null => {
  const n = finite(value)
  return n != null && n > 0 && n <= 100 ? Math.round(n * 10) / 10 : null
}

const isoDayOf = (record: UnknownRecord, keys: string[]): string | null => {
  for (const key of keys) {
    const v = record[key]
    if (typeof v === 'string' && /^\d{4}-\d{2}-\d{2}/.test(v)) return v.slice(0, 10)
    if (typeof v === 'number' && v > 1_000_000_000_000)
      return new Date(v).toISOString().slice(0, 10)
  }
  return null
}

const tsOf = (record: UnknownRecord): number | null => {
  for (const key of ['timestampGMT', 'weighInTimestampGMT', 'date', 'samplePk']) {
    if (key === 'samplePk') break
    const v = finite(record[key])
    if (v != null && v > 1_000_000_000_000) return v
  }
  return null
}

export function garminConnectWeightSamples(raw: unknown): GarminWeightSample[] {
  const out: GarminWeightSample[] = []
  const push = (m: UnknownRecord, dayHint: string | null): void => {
    const ts = tsOf(m)
    const date =
      dayHint ??
      isoDayOf(m, ['calendarDate', 'summaryDate', 'date', 'weightDate']) ??
      (ts != null ? new Date(ts).toISOString().slice(0, 10) : null)
    if (!date) return
    const sample: GarminWeightSample = {
      ts: ts ?? Date.parse(`${date}T12:00:00.000Z`),
      date,
      weightKg: kgOf(m.weight),
      bmi: pctOf(m.bmi),
      bodyFatPct: pctOf(m.bodyFat),
      bodyWaterPct: pctOf(m.bodyWater),
      muscleMassKg: kgOf(m.muscleMass),
      boneMassKg: kgOf(m.boneMass),
    }
    if (
      sample.weightKg == null &&
      sample.bmi == null &&
      sample.bodyFatPct == null &&
      sample.bodyWaterPct == null &&
      sample.muscleMassKg == null &&
      sample.boneMassKg == null
    )
      return
    out.push(sample)
  }
  const summaries =
    isRecord(raw) && Array.isArray(raw.dailyWeightSummaries) ? raw.dailyWeightSummaries : []
  for (const sum of summaries) {
    if (!isRecord(sum)) continue
    const date = isoDayOf(sum, ['summaryDate', 'calendarDate'])
    const metrics = Array.isArray(sum.allWeightMetrics) ? sum.allWeightMetrics : null
    if (metrics && metrics.length) {
      for (const m of metrics)
        if (isRecord(m)) push(m, date ?? isoDayOf(m, ['calendarDate', 'date']))
    } else if (isRecord(sum.latestWeight)) {
      push(sum.latestWeight, date)
    }
  }
  if (!out.length) {
    const list = isRecord(raw) && Array.isArray(raw.dateWeightList) ? raw.dateWeightList : raw
    if (Array.isArray(list)) for (const m of list) if (isRecord(m)) push(m, null)
  }
  return out.sort((a, b) => a.ts - b.ts)
}

function metricIndex(detail: UnknownRecord, key: string): number | null {
  const descriptors = detail.metricDescriptors
  if (!Array.isArray(descriptors)) return null
  for (let i = 0; i < descriptors.length; i++) {
    const descriptor = descriptors[i]
    if (isRecord(descriptor) && readString(descriptor, 'key') === key) return i
  }
  return null
}

function metricValue(row: UnknownRecord, index: number | null): number | null {
  if (index == null) return null
  const metrics = row.metrics
  if (!Array.isArray(metrics)) return null
  return finite(metrics[index])
}

function validLatLng(lat: number, lng: number): boolean {
  return lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180
}

function hasStreamData(streams: GarminStreams): boolean {
  return (
    streams.latlng.length >= 2 ||
    streams.altitude.length > 0 ||
    streams.distance.length > 0 ||
    (streams.watts?.some(value => value > 0) ?? false) ||
    (streams.heartrate?.some(value => value > 0) ?? false) ||
    (streams.cadence?.some(value => value > 0) ?? false)
  )
}

function polylineStreams(detail: UnknownRecord): GarminStreams | null {
  const geo = detail.geoPolylineDTO
  if (!isRecord(geo) || !Array.isArray(geo.polyline)) return null

  const streams: GarminStreams = { latlng: [], altitude: [], distance: [] }
  let distance = 0
  for (const item of geo.polyline) {
    if (!isRecord(item)) continue
    const lat = finite(item.lat)
    const lng = finite(item.lon)
    if (lat == null || lng == null || !validLatLng(lat, lng)) continue
    distance =
      finite(item.distanceInMeters) ?? distance + (finite(item.distanceFromPreviousPoint) ?? 0)
    streams.latlng.push([lat, lng])
    streams.altitude.push(finite(item.altitude) ?? 0)
    streams.distance.push(distance)
  }
  return hasStreamData(streams) ? streams : null
}

export function garminConnectStreams(detail: UnknownRecord | null): GarminStreams | null {
  if (!detail || !Array.isArray(detail.activityDetailMetrics))
    return detail ? polylineStreams(detail) : null

  const indices = {
    altitude: metricIndex(detail, METRIC_KEYS.altitude),
    cadence: metricIndex(detail, METRIC_KEYS.cadence),
    distance: metricIndex(detail, METRIC_KEYS.distance),
    heartRate: metricIndex(detail, METRIC_KEYS.heartRate),
    latitude: metricIndex(detail, METRIC_KEYS.latitude),
    longitude: metricIndex(detail, METRIC_KEYS.longitude),
    power: metricIndex(detail, METRIC_KEYS.power),
  }
  const streams: GarminStreams = {
    latlng: [],
    altitude: [],
    distance: [],
    watts: [],
    heartrate: [],
    cadence: [],
  }

  const hasLocationMetrics = indices.latitude != null && indices.longitude != null
  let lastDistance = 0
  for (const item of detail.activityDetailMetrics) {
    if (!isRecord(item)) continue
    const lat = metricValue(item, indices.latitude)
    const lng = metricValue(item, indices.longitude)
    const hasLocation = lat != null && lng != null && validLatLng(lat, lng)
    if (hasLocationMetrics && !hasLocation) continue

    const distance = metricValue(item, indices.distance)
    if (distance != null) lastDistance = distance
    if (lat != null && lng != null && validLatLng(lat, lng)) streams.latlng.push([lat, lng])
    streams.altitude.push(metricValue(item, indices.altitude) ?? 0)
    streams.distance.push(lastDistance)
    streams.watts?.push(metricValue(item, indices.power) ?? 0)
    streams.heartrate?.push(metricValue(item, indices.heartRate) ?? 0)
    streams.cadence?.push(metricValue(item, indices.cadence) ?? 0)
  }

  if (streams.latlng.length < 2) {
    const polyline = polylineStreams(detail)
    if (polyline) {
      streams.latlng = polyline.latlng
      streams.altitude = polyline.altitude
      streams.distance = polyline.distance
    }
  }

  return hasStreamData(streams) ? streams : null
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
