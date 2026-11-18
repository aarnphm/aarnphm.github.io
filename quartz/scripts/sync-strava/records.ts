import type {
  RawStravaActivity,
  RawStravaActivityDetail,
  RawStravaAnalysisRange,
  RawStravaRunSplit,
  StravaGear,
} from '../../plugins/stores/strava'
import { stravaId, stravaObjectId } from '../../util/strava-id'
import { isRecord, readNumber, readString } from '../../util/type-guards'

export function mapActivity(raw: Record<string, unknown>): RawStravaActivity {
  const deviceName = readString(raw, 'device_name')?.trim()
  return {
    id: stravaObjectId(raw),
    name: String(raw.name ?? ''),
    sportType: String(raw.sport_type ?? raw.type ?? ''),
    ...(typeof raw.trainer === 'boolean' ? { trainer: raw.trainer } : {}),
    ...(typeof raw.commute === 'boolean' ? { commute: raw.commute } : {}),
    distance: Number(raw.distance ?? 0),
    movingTime: Number(raw.moving_time ?? 0),
    elapsedTime: Number(raw.elapsed_time ?? 0),
    totalElevationGain: Number(raw.total_elevation_gain ?? 0),
    startDate: String(raw.start_date ?? ''),
    startDateLocal: String(raw.start_date_local ?? raw.start_date ?? ''),
    averageSpeed: Number(raw.average_speed ?? 0),
    maxSpeed: nullableNumber(raw, 'max_speed') ?? undefined,
    averageHeartrate:
      raw.average_heartrate === undefined ? undefined : Number(raw.average_heartrate),
    maxHeartrate: raw.max_heartrate === undefined ? undefined : Number(raw.max_heartrate),
    averageWatts: raw.average_watts === undefined ? undefined : Number(raw.average_watts),
    weightedAverageWatts:
      raw.weighted_average_watts === undefined ? undefined : Number(raw.weighted_average_watts),
    maxWatts: raw.max_watts === undefined ? undefined : Number(raw.max_watts),
    kilojoules: raw.kilojoules === undefined ? undefined : Number(raw.kilojoules),
    deviceWatts: raw.device_watts === undefined ? undefined : Boolean(raw.device_watts),
    averageCadence: raw.average_cadence === undefined ? undefined : Number(raw.average_cadence),
    sufferScore: raw.suffer_score === undefined ? undefined : Number(raw.suffer_score),
    averageTemp: raw.average_temp === undefined ? undefined : Number(raw.average_temp),
    ...(deviceName ? { deviceName } : {}),
    ...(raw.gear_id === null || typeof raw.gear_id === 'string'
      ? { gearId: readString(raw, 'gear_id')?.trim() || null }
      : {}),
  }
}

export function parseGear(value: unknown): StravaGear | null {
  if (!isRecord(value)) return null
  const id = readString(value, 'id')?.trim()
  if (!id) return null
  const brandName = readString(value, 'brand_name')?.trim() || null
  const modelName = readString(value, 'model_name')?.trim() || null
  const distance = nullableNumber(value, 'distance')
  return {
    id,
    name:
      readString(value, 'name')?.trim() || [brandName, modelName].filter(Boolean).join(' ') || null,
    brandName,
    modelName,
    ...(value.description === null || typeof value.description === 'string'
      ? { description: readString(value, 'description')?.trim() || null }
      : {}),
    distanceM: distance != null && distance >= 0 ? distance : null,
  }
}

function nullableNumber(record: Record<string, unknown>, key: string): number | null {
  const value = readNumber(record, key)
  return value != null && Number.isFinite(value) ? value : null
}

function parseAnalysisRange(
  value: unknown,
  fallbackKind: 'lap' | 'segment',
  index: number,
): RawStravaAnalysisRange | null {
  if (!isRecord(value)) return null
  const elapsedTime = nullableNumber(value, 'elapsed_time')
  const distance = nullableNumber(value, 'distance')
  if (elapsedTime == null || elapsedTime < 0 || distance == null || distance < 0) return null
  const id =
    (typeof value.id_str === 'string' ? stravaId(value.id_str) : null) ??
    stravaId(value.id) ??
    `${fallbackKind}-${index + 1}`
  const name =
    readString(value, 'name')?.trim() ||
    `${fallbackKind === 'lap' ? 'Lap' : 'Segment'} ${index + 1}`
  return {
    id,
    name,
    elapsedTime,
    movingTime: nullableNumber(value, 'moving_time') ?? elapsedTime,
    startDate: readString(value, 'start_date') ?? null,
    distance,
    startIndex: nullableNumber(value, 'start_index'),
    endIndex: nullableNumber(value, 'end_index'),
    totalElevationGain: nullableNumber(value, 'total_elevation_gain'),
    averageSpeed: nullableNumber(value, 'average_speed'),
    averageHeartrate: nullableNumber(value, 'average_heartrate'),
    averageWatts: nullableNumber(value, 'average_watts'),
    averageCadence: nullableNumber(value, 'average_cadence'),
  }
}

function parseAnalysisRanges(value: unknown, kind: 'lap' | 'segment'): RawStravaAnalysisRange[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item, index) => {
    const range = parseAnalysisRange(item, kind, index)
    return range ? [range] : []
  })
}

function parseRunSplit(value: unknown, index: number): RawStravaRunSplit | null {
  if (!isRecord(value)) return null
  const distance = nullableNumber(value, 'distance')
  const elapsedTime = nullableNumber(value, 'elapsed_time')
  const movingTime = nullableNumber(value, 'moving_time') ?? elapsedTime
  if (
    distance == null ||
    distance <= 0 ||
    elapsedTime == null ||
    elapsedTime <= 0 ||
    movingTime == null ||
    movingTime <= 0
  )
    return null
  const averageSpeed = nullableNumber(value, 'average_speed') ?? distance / movingTime
  if (!Number.isFinite(averageSpeed) || averageSpeed <= 0) return null
  return {
    split: Math.max(1, Math.round(nullableNumber(value, 'split') ?? index + 1)),
    distance,
    elapsedTime,
    movingTime,
    averageSpeed,
    averageHeartrate: nullableNumber(value, 'average_heartrate'),
    elevationDifference: nullableNumber(value, 'elevation_difference'),
    paceZone: nullableNumber(value, 'pace_zone'),
  }
}

export function parseRunSplits(value: unknown): RawStravaRunSplit[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item, index) => {
    const split = parseRunSplit(item, index)
    return split ? [split] : []
  })
}

export function parseActivityDetail(
  data: unknown,
  id: string,
  athleteId: string | null,
): RawStravaActivityDetail {
  if (!isRecord(data) || stravaObjectId(data) !== id)
    throw new Error('Strava detail ID does not match requested activity')
  if (data.athlete !== undefined) {
    if (!isRecord(data.athlete)) throw new Error('Invalid Strava detail athlete ID')
    const nestedId = stravaObjectId(data.athlete)
    if (athleteId !== null && nestedId !== athleteId)
      throw new Error('Strava detail athlete ID changed')
  }
  return {
    description: readString(data, 'description') ?? null,
    fetchedAt: Date.now(),
    calories: nullableNumber(data, 'calories'),
    laps: parseAnalysisRanges(data.laps, 'lap'),
    segmentEfforts: parseAnalysisRanges(data.segment_efforts, 'segment'),
    splitsMetric: parseRunSplits(data.splits_metric),
    splitsStandard: parseRunSplits(data.splits_standard),
  }
}
