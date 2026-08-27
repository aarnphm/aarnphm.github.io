import { Decoder, Stream, type FitMessages, type RecordMesg } from '@garmin/fitsdk'
import { createHash } from 'node:crypto'
import type { WahooMetrics, WahooStreams } from '../plugins/stores/wahoo'
import type { WahooCyclingDynamics, WahooGearShift } from '../plugins/stores/wahoo'
import { emptyWahooMetrics } from '../plugins/stores/wahoo'
import { fitCyclingDynamics, fitGearShifts } from './garmin-fit'

const SEMICIRCLES_PER_DEGREE = 2 ** 31 / 180

export interface WahooFitData {
  startDate: string
  sport: string | null
  sourceDevice: string | null
  distanceM: number | null
  movingTimeS: number | null
  elapsedTimeS: number | null
  metrics: WahooMetrics
  streams: WahooStreams
  gearShifts: WahooGearShift[]
  cyclingDynamics: WahooCyclingDynamics
  profileVersion: string
}

function finite(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) ? value : null
}

function nonnegative(value: number | null | undefined): number | null {
  const parsed = finite(value)
  return parsed != null && parsed >= 0 ? parsed : null
}

function positive(value: number | null | undefined): number | null {
  const parsed = finite(value)
  return parsed != null && parsed > 0 ? parsed : null
}

function timestamp(value: Date | number | 'min' | undefined): Date | null {
  if (!(value instanceof Date) || !Number.isFinite(value.getTime())) return null
  return value
}

function text(value: string | number | undefined): string | null {
  if (typeof value === 'string') return value.trim() || null
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return null
}

function decodeMessages(bytes: Uint8Array): { messages: FitMessages; profileVersion: string } {
  if (!Decoder.isFIT(Stream.fromByteArray(bytes))) throw new Error('Wahoo workout file is not FIT')
  if (!new Decoder(Stream.fromByteArray(bytes)).checkIntegrity())
    throw new Error('Wahoo workout FIT integrity check failed')
  const decoded = new Decoder(Stream.fromByteArray(bytes)).read({
    expandSubFields: true,
    expandComponents: true,
  })
  if (decoded.errors.length > 0)
    throw new Error(decoded.errors.map(error => error.message).join('; '))
  return {
    messages: decoded.messages,
    profileVersion: `${decoded.profileVersion.major}.${decoded.profileVersion.minor}`,
  }
}

function recordTimestamp(record: RecordMesg): number {
  return timestamp(record.timestamp)?.getTime() ?? Number.POSITIVE_INFINITY
}

function coordinate(value: number | undefined, minimum: number, maximum: number): number | null {
  if (value == null || !Number.isFinite(value)) return null
  const degrees = value / SEMICIRCLES_PER_DEGREE
  return degrees >= minimum && degrees <= maximum ? degrees : null
}

function rightBalance(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isInteger(value)) return null
  const percent = value & 0x7f
  if (percent < 0 || percent > 100) return null
  return (value & 0x80) !== 0 ? percent : 100 - percent
}

function deviceName(messages: FitMessages): string | null {
  const fileName = text(messages.fileIdMesgs?.[0]?.productName)
  if (fileName) return fileName
  for (const device of messages.deviceInfoMesgs ?? []) {
    const name = text(device.productName) ?? text(device.manufacturer)
    if (name) return name
  }
  return text(messages.fileIdMesgs?.[0]?.manufacturer)
}

function streamsFor(messages: FitMessages, startMs: number): WahooStreams {
  const records = [...(messages.recordMesgs ?? [])]
    .filter(record => timestamp(record.timestamp) != null)
    .sort((left, right) => recordTimestamp(left) - recordTimestamp(right))
  const streams: WahooStreams = {
    timestamps: [],
    time: [],
    latlng: [],
    altitude: [],
    distance: [],
    watts: [],
    rightBalance: [],
    heartrate: [],
    cadence: [],
    speed: [],
    temperature: [],
    respiration: [],
  }
  for (const record of records) {
    const date = timestamp(record.timestamp)
    if (!date) continue
    const latitude = coordinate(record.positionLat, -90, 90)
    const longitude = coordinate(record.positionLong, -180, 180)
    streams.timestamps.push(date.toISOString())
    streams.time.push(Math.max(0, (date.getTime() - startMs) / 1000))
    streams.latlng.push(latitude != null && longitude != null ? [latitude, longitude] : null)
    streams.altitude.push(finite(record.enhancedAltitude ?? record.altitude))
    streams.distance.push(nonnegative(record.distance))
    streams.watts.push(nonnegative(record.power))
    streams.rightBalance.push(rightBalance(record.leftRightBalance))
    streams.heartrate.push(positive(record.heartRate))
    streams.cadence.push(nonnegative(record.cadence))
    streams.speed.push(nonnegative(record.enhancedSpeed ?? record.speed))
    streams.temperature.push(finite(record.temperature))
    streams.respiration.push(positive(record.respirationRate))
  }
  return streams
}

function sessionMetrics(messages: FitMessages): WahooMetrics {
  const session = messages.sessionMesgs?.[0]
  const metrics = emptyWahooMetrics()
  if (!session) return metrics
  metrics.totalCalories = nonnegative(session.totalCalories)
  metrics.avgHeartRate = positive(session.avgHeartRate)
  metrics.maxHeartRate = positive(session.maxHeartRate)
  metrics.avgPower = nonnegative(session.avgPower)
  metrics.normalizedPower = nonnegative(session.normalizedPower)
  metrics.maxPower = nonnegative(session.maxPower)
  metrics.avgCadence = nonnegative(session.avgCadence ?? session.avgRunningCadence)
  metrics.totalAscentM = nonnegative(session.totalAscent)
  metrics.totalDescentM = nonnegative(session.totalDescent)
  const workJ = nonnegative(session.totalWork)
  metrics.totalWorkKJ = workJ == null ? null : workJ / 1000
  metrics.trainingStressScore = nonnegative(session.trainingStressScore)
  metrics.intensityFactor = nonnegative(session.intensityFactor)
  metrics.avgSpeedMps = nonnegative(session.enhancedAvgSpeed ?? session.avgSpeed)
  metrics.maxSpeedMps = nonnegative(session.enhancedMaxSpeed ?? session.maxSpeed)
  metrics.avgTemperatureC = finite(session.avgTemperature)
  return metrics
}

export function wahooFitSha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

export function decodeWahooFit(bytes: Uint8Array): WahooFitData {
  const { messages, profileVersion } = decodeMessages(bytes)
  const file = messages.fileIdMesgs?.[0]
  if (!file || file.type !== 'activity') throw new Error('Wahoo FIT file must be an activity')
  const session = messages.sessionMesgs?.[0]
  if (!session) throw new Error('Wahoo FIT file has no session')
  const start = timestamp(session.startTime) ?? timestamp(file.timeCreated)
  if (!start) throw new Error('Wahoo FIT file has no valid start time')
  return {
    startDate: start.toISOString(),
    sport: text(session.sport),
    sourceDevice: deviceName(messages),
    distanceM: nonnegative(session.totalDistance),
    movingTimeS: nonnegative(session.totalMovingTime ?? session.totalTimerTime),
    elapsedTimeS: nonnegative(session.totalElapsedTime),
    metrics: sessionMetrics(messages),
    streams: streamsFor(messages, start.getTime()),
    gearShifts: fitGearShifts(messages),
    cyclingDynamics: fitCyclingDynamics(messages),
    profileVersion,
  }
}
