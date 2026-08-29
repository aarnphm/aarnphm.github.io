import {
  Encoder,
  Profile,
  type ActivityMesg,
  type DeviceInfoMesg,
  type EventMesg,
  type FileIdMesg,
  type LapMesg,
  type RecordMesg,
  type SessionMesg,
} from '@garmin/fitsdk'
import { XMLParser } from 'fast-xml-parser'
import { validateGarminFit, type GarminFitEncoding } from './garmin-fit'
import { isRecord, type UnknownRecord } from './type-guards'

type GarminTcxFitSport = 'running' | 'cycling' | 'swimming' | 'generic'

interface TcxTrackpoint {
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

interface TcxLap {
  startTime: Date
  totalTimeSeconds: number | null
  distanceMeters: number | null
  calories: number | null
  trackpoints: TcxTrackpoint[]
}

interface TcxActivity {
  sport: GarminTcxFitSport
  startTime: Date
  endTime: Date
  laps: TcxLap[]
  trackpoints: TcxTrackpoint[]
}

interface TcxStatistics {
  averageHeartRate: number | null
  maximumHeartRate: number | null
  averageCadence: number | null
  maximumCadence: number | null
  averagePower: number | null
  maximumPower: number | null
  averageTemperature: number | null
  maximumTemperature: number | null
  averageAltitude: number | null
  minimumAltitude: number | null
  maximumAltitude: number | null
  maximumSpeed: number | null
  totalAscent: number | null
  totalDescent: number | null
}

const MAX_TCX_BYTES = 25 * 1024 * 1024
const SEMICIRCLES_PER_DEGREE = 2 ** 31 / 180

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

function parseTcxActivity(bytes: Uint8Array): TcxActivity {
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

function uint8(value: number): number {
  return Math.max(0, Math.min(254, Math.round(value)))
}

function uint16(value: number): number {
  return Math.max(0, Math.min(65_534, Math.round(value)))
}

function sint8(value: number): number {
  return Math.max(-127, Math.min(127, Math.round(value)))
}

function semicircles(degrees: number): number {
  return Math.max(
    -2_147_483_648,
    Math.min(2_147_483_647, Math.round(degrees * SEMICIRCLES_PER_DEGREE)),
  )
}

function hashSourceId(sourceId: string | number): number {
  const text = String(sourceId)
  let hash = 2_166_136_261
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 16_777_619)
  }
  return hash >>> 0 || 1
}

function finiteValues(
  trackpoints: readonly TcxTrackpoint[],
  read: (trackpoint: TcxTrackpoint) => number | null,
): number[] {
  const result: number[] = []
  for (const trackpoint of trackpoints) {
    const value = read(trackpoint)
    if (value != null) result.push(value)
  }
  return result
}

function average(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length
}

function minimum(values: readonly number[]): number | null {
  return values.length === 0 ? null : Math.min(...values)
}

function maximum(values: readonly number[]): number | null {
  return values.length === 0 ? null : Math.max(...values)
}

function pointSpeed(trackpoints: readonly TcxTrackpoint[], index: number): number | null {
  const point = trackpoints[index]
  if (point.speedMetersPerSecond != null) return point.speedMetersPerSecond
  if (index === 0 || point.distanceMeters == null) return null
  const previous = trackpoints[index - 1]
  if (previous.distanceMeters == null) return null
  const seconds = (point.timestamp.getTime() - previous.timestamp.getTime()) / 1000
  const distance = point.distanceMeters - previous.distanceMeters
  return seconds > 0 && distance >= 0 ? distance / seconds : null
}

function statistics(trackpoints: readonly TcxTrackpoint[]): TcxStatistics {
  const heartRates = finiteValues(trackpoints, point => point.heartRateBpm)
  const cadences = finiteValues(trackpoints, point => point.cadenceRpm)
  const powers = finiteValues(trackpoints, point => point.powerWatts)
  const temperatures = finiteValues(trackpoints, point => point.temperatureCelsius)
  const altitudes = finiteValues(trackpoints, point => point.altitudeMeters)
  const speeds = trackpoints
    .map((_, index) => pointSpeed(trackpoints, index))
    .filter((speed): speed is number => speed != null)
  let totalAscent = 0
  let totalDescent = 0
  for (let index = 1; index < altitudes.length; index++) {
    const change = altitudes[index] - altitudes[index - 1]
    if (change > 0) totalAscent += change
    else totalDescent -= change
  }
  return {
    averageHeartRate: average(heartRates),
    maximumHeartRate: maximum(heartRates),
    averageCadence: average(cadences),
    maximumCadence: maximum(cadences),
    averagePower: average(powers),
    maximumPower: maximum(powers),
    averageTemperature: average(temperatures),
    maximumTemperature: maximum(temperatures),
    averageAltitude: average(altitudes),
    minimumAltitude: minimum(altitudes),
    maximumAltitude: maximum(altitudes),
    maximumSpeed: maximum(speeds),
    totalAscent: altitudes.length > 1 ? totalAscent : null,
    totalDescent: altitudes.length > 1 ? totalDescent : null,
  }
}

function gpsBounds(
  trackpoints: readonly TcxTrackpoint[],
): {
  firstLatitude: number
  firstLongitude: number
  lastLatitude: number
  lastLongitude: number
  northeastLatitude: number
  northeastLongitude: number
  southwestLatitude: number
  southwestLongitude: number
} | null {
  const positioned = trackpoints.filter(
    (point): point is TcxTrackpoint & { latitudeDegrees: number; longitudeDegrees: number } =>
      point.latitudeDegrees != null && point.longitudeDegrees != null,
  )
  if (positioned.length === 0) return null
  const first = positioned[0]
  const last = positioned[positioned.length - 1]
  const latitudes = positioned.map(point => point.latitudeDegrees)
  const longitudes = positioned.map(point => point.longitudeDegrees)
  return {
    firstLatitude: semicircles(first.latitudeDegrees),
    firstLongitude: semicircles(first.longitudeDegrees),
    lastLatitude: semicircles(last.latitudeDegrees),
    lastLongitude: semicircles(last.longitudeDegrees),
    northeastLatitude: semicircles(Math.max(...latitudes)),
    northeastLongitude: semicircles(Math.max(...longitudes)),
    southwestLatitude: semicircles(Math.min(...latitudes)),
    southwestLongitude: semicircles(Math.min(...longitudes)),
  }
}

function activityDistance(activity: TcxActivity): number | null {
  const lapDistances = activity.laps.map(lap => lap.distanceMeters)
  if (lapDistances.every((distance): distance is number => distance != null))
    return lapDistances.reduce((sum, distance) => sum + distance, 0)
  return maximum(finiteValues(activity.trackpoints, point => point.distanceMeters))
}

function lapElapsedSeconds(lap: TcxLap): number {
  const observed = lap.trackpoints.at(-1)
  const observedSeconds = observed
    ? (observed.timestamp.getTime() - lap.startTime.getTime()) / 1000
    : 0
  return Math.max(observedSeconds, lap.totalTimeSeconds ?? 0)
}

function activityTimerSeconds(activity: TcxActivity): number {
  const seconds = activity.laps.reduce(
    (sum, lap) => sum + (lap.totalTimeSeconds ?? lapElapsedSeconds(lap)),
    0,
  )
  return seconds > 0 ? seconds : (activity.endTime.getTime() - activity.startTime.getTime()) / 1000
}

function applyStatistics(target: LapMesg | SessionMesg, values: TcxStatistics): void {
  if (values.averageHeartRate != null) target.avgHeartRate = uint8(values.averageHeartRate)
  if (values.maximumHeartRate != null) target.maxHeartRate = uint8(values.maximumHeartRate)
  if (values.averageCadence != null) target.avgCadence = uint8(values.averageCadence)
  if (values.maximumCadence != null) target.maxCadence = uint8(values.maximumCadence)
  if (values.averagePower != null) target.avgPower = uint16(values.averagePower)
  if (values.maximumPower != null) target.maxPower = uint16(values.maximumPower)
  if (values.averageTemperature != null) target.avgTemperature = sint8(values.averageTemperature)
  if (values.maximumTemperature != null) target.maxTemperature = sint8(values.maximumTemperature)
  if (values.averageAltitude != null) target.avgAltitude = values.averageAltitude
  if (values.minimumAltitude != null) target.minAltitude = values.minimumAltitude
  if (values.maximumAltitude != null) target.maxAltitude = values.maximumAltitude
  if (values.totalAscent != null) target.totalAscent = uint16(values.totalAscent)
  if (values.totalDescent != null) target.totalDescent = uint16(values.totalDescent)
}

function recordMessage(trackpoints: readonly TcxTrackpoint[], index: number): RecordMesg {
  const point = trackpoints[index]
  const message: RecordMesg = { timestamp: point.timestamp }
  if (point.latitudeDegrees != null) message.positionLat = semicircles(point.latitudeDegrees)
  if (point.longitudeDegrees != null) message.positionLong = semicircles(point.longitudeDegrees)
  if (point.altitudeMeters != null) {
    message.altitude = point.altitudeMeters
    message.enhancedAltitude = point.altitudeMeters
  }
  if (point.distanceMeters != null) message.distance = point.distanceMeters
  if (point.heartRateBpm != null) message.heartRate = uint8(point.heartRateBpm)
  if (point.cadenceRpm != null) message.cadence = uint8(point.cadenceRpm)
  if (point.powerWatts != null) message.power = uint16(point.powerWatts)
  if (point.temperatureCelsius != null) message.temperature = sint8(point.temperatureCelsius)
  const speed = pointSpeed(trackpoints, index)
  if (speed != null) {
    message.speed = speed
    message.enhancedSpeed = speed
  }
  return message
}

function lapMessage(
  lap: TcxLap,
  index: number,
  lapCount: number,
  sport: GarminTcxFitSport,
): LapMesg {
  const elapsedSeconds = lapElapsedSeconds(lap)
  const timerSeconds = lap.totalTimeSeconds ?? elapsedSeconds
  const endTime = new Date(lap.startTime.getTime() + elapsedSeconds * 1000)
  const values = statistics(lap.trackpoints)
  const distance =
    lap.distanceMeters ?? maximum(finiteValues(lap.trackpoints, point => point.distanceMeters))
  const averageSpeed = distance != null && timerSeconds > 0 ? distance / timerSeconds : null
  const message: LapMesg = {
    messageIndex: index,
    timestamp: endTime,
    event: 'lap',
    eventType: 'stop',
    startTime: lap.startTime,
    sport,
    totalElapsedTime: elapsedSeconds,
    totalTimerTime: timerSeconds,
    totalDistance: distance ?? undefined,
    totalCalories: lap.calories == null ? undefined : uint16(lap.calories),
    avgSpeed: averageSpeed ?? undefined,
    enhancedAvgSpeed: averageSpeed ?? undefined,
    maxSpeed: values.maximumSpeed ?? undefined,
    enhancedMaxSpeed: values.maximumSpeed ?? undefined,
    intensity: 'active',
    lapTrigger: index === lapCount - 1 ? 'sessionEnd' : 'manual',
    totalMovingTime: timerSeconds,
  }
  const bounds = gpsBounds(lap.trackpoints)
  if (bounds) {
    message.startPositionLat = bounds.firstLatitude
    message.startPositionLong = bounds.firstLongitude
    message.endPositionLat = bounds.lastLatitude
    message.endPositionLong = bounds.lastLongitude
  }
  applyStatistics(message, values)
  return message
}

export function encodeGarminTcxActivityFit(
  tcx: Uint8Array,
  sourceId: string | number,
): GarminFitEncoding {
  const activity = parseTcxActivity(tcx)
  const elapsedSeconds = (activity.endTime.getTime() - activity.startTime.getTime()) / 1000
  const timerSeconds = activityTimerSeconds(activity)
  const distance = activityDistance(activity)
  const values = statistics(activity.trackpoints)
  const averageSpeed = distance != null && timerSeconds > 0 ? distance / timerSeconds : null
  const calories = activity.laps.map(lap => lap.calories)
  const serialNumber = hashSourceId(sourceId)
  const encoder = new Encoder()
  const fileId: FileIdMesg = {
    type: 'activity',
    manufacturer: 'development',
    product: 0,
    serialNumber,
    timeCreated: activity.startTime,
  }
  const deviceInfo: DeviceInfoMesg = {
    timestamp: activity.startTime,
    deviceIndex: 'creator',
    manufacturer: 'development',
    product: 0,
    serialNumber,
    productName: 'Garmin TCX Bridge',
    softwareVersion: 1,
  }
  const timerStart: EventMesg = {
    timestamp: activity.startTime,
    event: 'timer',
    eventType: 'start',
  }
  encoder.onMesg(Profile.MesgNum.FILE_ID, fileId)
  encoder.onMesg(Profile.MesgNum.DEVICE_INFO, deviceInfo)
  encoder.onMesg(Profile.MesgNum.EVENT, timerStart)
  for (let index = 0; index < activity.trackpoints.length; index++)
    encoder.onMesg(Profile.MesgNum.RECORD, recordMessage(activity.trackpoints, index))
  for (const [index, lap] of activity.laps.entries())
    encoder.onMesg(
      Profile.MesgNum.LAP,
      lapMessage(lap, index, activity.laps.length, activity.sport),
    )
  const timerStop: EventMesg = { timestamp: activity.endTime, event: 'timer', eventType: 'stop' }
  const session: SessionMesg = {
    messageIndex: 0,
    timestamp: activity.endTime,
    event: 'session',
    eventType: 'stop',
    startTime: activity.startTime,
    sport: activity.sport,
    totalElapsedTime: elapsedSeconds,
    totalTimerTime: timerSeconds,
    totalDistance: distance ?? undefined,
    totalCalories: calories.some(value => value != null)
      ? uint16(calories.reduce((sum, value) => sum + (value ?? 0), 0))
      : undefined,
    avgSpeed: averageSpeed ?? undefined,
    enhancedAvgSpeed: averageSpeed ?? undefined,
    maxSpeed: values.maximumSpeed ?? undefined,
    enhancedMaxSpeed: values.maximumSpeed ?? undefined,
    firstLapIndex: 0,
    numLaps: activity.laps.length,
    trigger: 'activityEnd',
    totalMovingTime: timerSeconds,
  }
  const bounds = gpsBounds(activity.trackpoints)
  if (bounds) {
    session.startPositionLat = bounds.firstLatitude
    session.startPositionLong = bounds.firstLongitude
    session.endPositionLat = bounds.lastLatitude
    session.endPositionLong = bounds.lastLongitude
    session.necLat = bounds.northeastLatitude
    session.necLong = bounds.northeastLongitude
    session.swcLat = bounds.southwestLatitude
    session.swcLong = bounds.southwestLongitude
  }
  applyStatistics(session, values)
  const activityMessage: ActivityMesg = {
    timestamp: activity.endTime,
    totalTimerTime: timerSeconds,
    numSessions: 1,
    type: 'manual',
    event: 'activity',
    eventType: 'stop',
  }
  encoder.onMesg(Profile.MesgNum.EVENT, timerStop)
  encoder.onMesg(Profile.MesgNum.SESSION, session)
  encoder.onMesg(Profile.MesgNum.ACTIVITY, activityMessage)
  const bytes = encoder.close()
  const validation = validateGarminFit(bytes)
  if (!validation.valid)
    throw new Error(`Garmin TCX conversion produced invalid FIT: ${validation.errors.join('; ')}`)
  return { bytes, validation }
}
