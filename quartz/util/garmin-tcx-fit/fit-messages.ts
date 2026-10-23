import type { LapMesg, RecordMesg, SessionMesg } from '@garmin/fitsdk'
import type { GarminTcxFitSport, TcxActivity, TcxLap, TcxTrackpoint } from './tcx-activity'

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

const SEMICIRCLES_PER_DEGREE = 2 ** 31 / 180

function uint8(value: number): number {
  return Math.max(0, Math.min(254, Math.round(value)))
}

export function uint16(value: number): number {
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

export function statistics(trackpoints: readonly TcxTrackpoint[]): TcxStatistics {
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

export function gpsBounds(
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

export function activityDistance(activity: TcxActivity): number | null {
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

export function activityTimerSeconds(activity: TcxActivity): number {
  const seconds = activity.laps.reduce(
    (sum, lap) => sum + (lap.totalTimeSeconds ?? lapElapsedSeconds(lap)),
    0,
  )
  return seconds > 0 ? seconds : (activity.endTime.getTime() - activity.startTime.getTime()) / 1000
}

export function applyStatistics(target: LapMesg | SessionMesg, values: TcxStatistics): void {
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

export function recordMessage(trackpoints: readonly TcxTrackpoint[], index: number): RecordMesg {
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

export function lapMessage(
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
