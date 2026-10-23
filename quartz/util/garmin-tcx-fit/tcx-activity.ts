import { XMLParser } from 'fast-xml-parser'
import { isRecord, type UnknownRecord } from '../type-guards'

export type GarminTcxFitSport = 'running' | 'cycling' | 'swimming' | 'generic'

export interface TcxTrackpoint {
  timestamp: Date
  latitudeDegrees: number | null
  longitudeDegrees: number | null
  altitudeMeters: number | null
  distanceMeters: number | null
  heartRateBpm: number | null
  cadenceRpm: number | null
  powerWatts: number | null
  temperatureCelsius: number | null
  speedMetersPerSecond: number | null
}

export interface TcxLap {
  startTime: Date
  totalTimeSeconds: number | null
  distanceMeters: number | null
  calories: number | null
  trackpoints: TcxTrackpoint[]
}

export interface TcxActivity {
  sport: GarminTcxFitSport
  startTime: Date
  endTime: Date
  laps: TcxLap[]
  trackpoints: TcxTrackpoint[]
}

const MAX_TCX_BYTES = 25 * 1024 * 1024

function requiredRecord(value: unknown, label: string): UnknownRecord {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return value
}

function values(value: unknown): unknown[] {
  if (value == null) return []
  return Array.isArray(value) ? value : [value]
}

function xmlString(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim().length === 0)
    throw new Error(`${label} must be a nonempty string`)
  return value.trim()
}

function xmlNumber(value: unknown, label: string): number | null {
  if (value == null) return null
  const parsed =
    typeof value === 'number'
      ? value
      : typeof value === 'string' && value.trim().length > 0
        ? Number(value)
        : Number.NaN
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be finite`)
  return parsed
}

function boundedNumber(
  value: unknown,
  label: string,
  minimum: number,
  maximum: number,
): number | null {
  const parsed = xmlNumber(value, label)
  if (parsed != null && (parsed < minimum || parsed > maximum))
    throw new Error(`${label} must be between ${minimum} and ${maximum}`)
  return parsed
}

function nonnegativeNumber(value: unknown, label: string): number | null {
  const parsed = xmlNumber(value, label)
  if (parsed != null && parsed < 0) throw new Error(`${label} must be nonnegative`)
  return parsed
}

function xmlDate(value: unknown, label: string): Date {
  const text = xmlString(value, label)
  const milliseconds = Date.parse(text)
  if (!Number.isFinite(milliseconds)) throw new Error(`${label} must be an ISO timestamp`)
  return new Date(milliseconds)
}

function firstRecord(value: unknown): UnknownRecord | null {
  const first = values(value)[0]
  return isRecord(first) ? first : null
}

function sportFromTcx(value: unknown): GarminTcxFitSport {
  switch (xmlString(value, 'TCX Activity.Sport')) {
    case 'Running':
      return 'running'
    case 'Biking':
      return 'cycling'
    case 'Swimming':
      return 'swimming'
    case 'Other':
      return 'generic'
    default:
      throw new Error('TCX Activity.Sport is unsupported')
  }
}

function parseTrackpoint(value: unknown, label: string): TcxTrackpoint {
  const record = requiredRecord(value, label)
  const position = isRecord(record.Position) ? record.Position : null
  const heartRate = firstRecord(record.HeartRateBpm)
  const extensions = firstRecord(record.Extensions)
  const tpx = firstRecord(extensions?.TPX)
  const cadence = boundedNumber(record.Cadence, `${label}.Cadence`, 0, 254)
  const runCadence = boundedNumber(tpx?.RunCadence, `${label}.Extensions.TPX.RunCadence`, 0, 254)
  return {
    timestamp: xmlDate(record.Time, `${label}.Time`),
    latitudeDegrees: boundedNumber(
      position?.LatitudeDegrees,
      `${label}.Position.LatitudeDegrees`,
      -90,
      90,
    ),
    longitudeDegrees: boundedNumber(
      position?.LongitudeDegrees,
      `${label}.Position.LongitudeDegrees`,
      -180,
      180,
    ),
    altitudeMeters: xmlNumber(record.AltitudeMeters, `${label}.AltitudeMeters`),
    distanceMeters: nonnegativeNumber(record.DistanceMeters, `${label}.DistanceMeters`),
    heartRateBpm: boundedNumber(heartRate?.Value, `${label}.HeartRateBpm.Value`, 1, 254),
    cadenceRpm: cadence ?? runCadence,
    powerWatts: boundedNumber(tpx?.Watts, `${label}.Extensions.TPX.Watts`, 0, 65_534),
    temperatureCelsius: boundedNumber(tpx?.Temp, `${label}.Extensions.TPX.Temp`, -127, 127),
    speedMetersPerSecond: nonnegativeNumber(tpx?.Speed, `${label}.Extensions.TPX.Speed`),
  }
}

function parseLap(value: unknown, index: number): TcxLap {
  const label = `TCX Activity.Lap[${index}]`
  const record = requiredRecord(value, label)
  const startTime = xmlDate(record['@_StartTime'], `${label}.StartTime`)
  const trackpoints: TcxTrackpoint[] = []
  for (const [trackIndex, trackValue] of values(record.Track).entries()) {
    const track = requiredRecord(trackValue, `${label}.Track[${trackIndex}]`)
    for (const [pointIndex, pointValue] of values(track.Trackpoint).entries())
      trackpoints.push(
        parseTrackpoint(pointValue, `${label}.Track[${trackIndex}].Trackpoint[${pointIndex}]`),
      )
  }
  trackpoints.sort((left, right) => left.timestamp.getTime() - right.timestamp.getTime())
  return {
    startTime,
    totalTimeSeconds: nonnegativeNumber(record.TotalTimeSeconds, `${label}.TotalTimeSeconds`),
    distanceMeters: nonnegativeNumber(record.DistanceMeters, `${label}.DistanceMeters`),
    calories: nonnegativeNumber(record.Calories, `${label}.Calories`),
    trackpoints,
  }
}

export function parseTcxActivity(bytes: Uint8Array): TcxActivity {
  if (bytes.byteLength === 0) throw new Error('Garmin activity TCX is empty')
  if (bytes.byteLength > MAX_TCX_BYTES)
    throw new Error(`Garmin activity TCX exceeds ${MAX_TCX_BYTES} bytes`)
  const xml = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  if (/<!DOCTYPE/i.test(xml)) throw new Error('Garmin activity TCX must not contain a doctype')
  const parsed: unknown = new XMLParser({
    ignoreAttributes: false,
    removeNSPrefix: true,
    parseTagValue: false,
    parseAttributeValue: false,
    trimValues: true,
  }).parse(xml)
  const document = requiredRecord(parsed, 'TCX document')
  const database = requiredRecord(document.TrainingCenterDatabase, 'TCX TrainingCenterDatabase')
  const activities = requiredRecord(database.Activities, 'TCX Activities')
  const activityValues = values(activities.Activity)
  if (activityValues.length !== 1)
    throw new Error(
      `Garmin activity TCX must contain exactly one activity, found ${activityValues.length}`,
    )
  const activity = requiredRecord(activityValues[0], 'TCX Activity')
  const laps = values(activity.Lap).map(parseLap)
  if (laps.length === 0) throw new Error('Garmin activity TCX contains no laps')
  const trackpoints = laps.flatMap(lap => lap.trackpoints)
  if (trackpoints.length === 0) throw new Error('Garmin activity TCX contains no trackpoints')
  trackpoints.sort((left, right) => left.timestamp.getTime() - right.timestamp.getTime())
  const activityId = activity.Id == null ? null : xmlDate(activity.Id, 'TCX Activity.Id')
  const startTime = new Date(
    Math.min(
      activityId?.getTime() ?? Number.POSITIVE_INFINITY,
      ...laps.map(lap => lap.startTime.getTime()),
      trackpoints[0].timestamp.getTime(),
    ),
  )
  const finalLap = laps[laps.length - 1]
  const finalLapEnd =
    finalLap.trackpoints.at(-1)?.timestamp.getTime() ??
    finalLap.startTime.getTime() + (finalLap.totalTimeSeconds ?? 0) * 1000
  const endTime = new Date(Math.max(trackpoints.at(-1)?.timestamp.getTime() ?? 0, finalLapEnd))
  if (endTime.getTime() <= startTime.getTime())
    throw new Error('Garmin activity TCX duration must be positive')
  return { sport: sportFromTcx(activity['@_Sport']), startTime, endTime, laps, trackpoints }
}
