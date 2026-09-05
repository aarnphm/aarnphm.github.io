import {
  Decoder,
  Stream,
  type FitMessages,
  type RecordMesg,
  type SegmentLapMesg,
} from '@garmin/fitsdk'
import { createHash } from 'node:crypto'
import type {
  WahooCyclingDynamics,
  WahooGearShift,
  WahooMetrics,
  WahooStreams,
  WahooSummitSegment,
  WahooSweatLoss,
} from '../plugins/stores/wahoo'
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
  sweatLoss: WahooSweatLoss
  gearShifts: WahooGearShift[]
  cyclingDynamics: WahooCyclingDynamics
  summitSegments: WahooSummitSegment[]
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

interface DecodedMessages {
  messages: FitMessages
  developerFields: ReadonlyMap<number, DeveloperField>
  profileVersion: string
}

interface DeveloperField {
  name: string
  offset: number
  scale: number
}

function fieldName(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
}

function decodeMessages(bytes: Uint8Array): DecodedMessages {
  if (!Decoder.isFIT(Stream.fromByteArray(bytes))) throw new Error('Wahoo workout file is not FIT')
  if (!new Decoder(Stream.fromByteArray(bytes)).checkIntegrity())
    throw new Error('Wahoo workout FIT integrity check failed')
  const developerFields = new Map<number, DeveloperField>()
  const decoded = new Decoder(Stream.fromByteArray(bytes)).read({
    expandSubFields: true,
    expandComponents: true,
    fieldDescriptionListener: (key, _developer, field) => {
      const name = text(field.fieldName)
      if (name)
        developerFields.set(key, {
          name: fieldName(name),
          offset: finite(field.offset) ?? 0,
          scale: positive(field.scale) ?? 1,
        })
    },
  })
  if (decoded.errors.length > 0)
    throw new Error(decoded.errors.map(error => error.message).join('; '))
  return {
    messages: decoded.messages,
    developerFields,
    profileVersion: `${decoded.profileVersion.major}.${decoded.profileVersion.minor}`,
  }
}

function recordTimestamp(record: RecordMesg): number {
  return timestamp(record.timestamp)?.getTime() ?? Number.POSITIVE_INFINITY
}

function sortedRecords(messages: FitMessages): RecordMesg[] {
  return [...(messages.recordMesgs ?? [])]
    .filter(record => timestamp(record.timestamp) != null)
    .sort((left, right) => recordTimestamp(left) - recordTimestamp(right))
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

function developerNumber(
  record: RecordMesg,
  fields: ReadonlyMap<number, DeveloperField>,
  acceptedNames: ReadonlySet<string>,
): number | null {
  for (const [key, value] of Object.entries(record.developerFields ?? {})) {
    const field = fields.get(Number(key))
    if (!field || !acceptedNames.has(field.name)) continue
    if (typeof value === 'number') return finite(value / field.scale - field.offset)
    if (!Array.isArray(value)) continue
    for (let index = value.length - 1; index >= 0; index--) {
      const item = value[index]
      if (typeof item === 'number') return finite(item / field.scale - field.offset)
    }
  }
  return null
}

const BREATH_RATE_FIELDS = new Set(['tyme_breath_rate', 'breathing_rate', 'respiration_rate'])
const MINUTE_VENTILATION_FIELDS = new Set(['tyme_minute_volume', 'minute_ventilation'])
const TIDAL_VOLUME_FIELDS = new Set(['tyme_tidal_volume', 'tidal_volume'])
const FLUID_LOSS_FIELDS = new Set(['fluid_loss_ml', 'fluid_loss'])
const SODIUM_LOSS_FIELDS = new Set(['sodium_loss_mg', 'sodium_loss'])
const HEAT_STRAIN_FIELDS = new Set(['heat_strain_index'])
const SKIN_TEMPERATURE_FIELDS = new Set(['skin_temperature'])

function streamsFor(
  records: readonly RecordMesg[],
  developerFields: ReadonlyMap<number, DeveloperField>,
  startMs: number,
): WahooStreams {
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
    muscleOxygenPercent: [],
    totalHemoglobinConcentration: [],
    heatStrainIndex: [],
    coreTemperatureC: [],
    skinTemperatureC: [],
    minuteVentilation: [],
    tidalVolume: [],
    fluidLossMl: [],
    sodiumLossMg: [],
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
    streams.respiration.push(
      positive(
        record.enhancedRespirationRate ??
          record.respirationRate ??
          developerNumber(record, developerFields, BREATH_RATE_FIELDS),
      ),
    )
    streams.muscleOxygenPercent.push(nonnegative(record.saturatedHemoglobinPercent))
    streams.totalHemoglobinConcentration.push(nonnegative(record.totalHemoglobinConc))
    streams.heatStrainIndex.push(
      nonnegative(developerNumber(record, developerFields, HEAT_STRAIN_FIELDS)),
    )
    streams.coreTemperatureC.push(finite(record.coreTemperature))
    streams.skinTemperatureC.push(
      finite(developerNumber(record, developerFields, SKIN_TEMPERATURE_FIELDS)),
    )
    streams.minuteVentilation.push(
      nonnegative(developerNumber(record, developerFields, MINUTE_VENTILATION_FIELDS)),
    )
    streams.tidalVolume.push(
      nonnegative(developerNumber(record, developerFields, TIDAL_VOLUME_FIELDS)),
    )
    streams.fluidLossMl.push(
      nonnegative(developerNumber(record, developerFields, FLUID_LOSS_FIELDS)),
    )
    streams.sodiumLossMg.push(
      nonnegative(developerNumber(record, developerFields, SODIUM_LOSS_FIELDS)),
    )
  }
  return streams
}

function summitFeature(uuid: string): WahooSummitSegment['feature'] | null {
  if (uuid.startsWith('WAHOO_ON_ROUTE_CLIMB-')) return 'summit-segment'
  if (uuid.startsWith('WAHOO_OFF_ROUTE_CLIMB-')) return 'summit-freeride'
  return null
}

function average(values: readonly (number | null)[]): number | null {
  const finiteValues = values.filter((value): value is number => value != null)
  if (finiteValues.length === 0) return null
  return finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length
}

function recordDistance(records: readonly RecordMesg[]): number | null {
  const distances = records
    .map(record => nonnegative(record.distance))
    .filter((value): value is number => value != null)
  if (distances.length < 2) return null
  const first = distances[0]
  const last = distances[distances.length - 1]
  return nonnegative(last - first)
}

function recordElevations(records: readonly RecordMesg[]): number[] {
  return records
    .map(record => finite(record.enhancedAltitude ?? record.altitude))
    .filter((value): value is number => value != null)
}

function recordElevationGain(records: readonly RecordMesg[]): number | null {
  const elevations = recordElevations(records)
  if (elevations.length < 2) return null
  let gain = 0
  for (let index = 1; index < elevations.length; index++)
    gain += Math.max(0, elevations[index] - elevations[index - 1])
  return gain
}

function segmentDates(
  segment: SegmentLapMesg,
): { start: Date; end: Date; durationS: number } | null {
  const fitDurationS = positive(segment.totalTimerTime) ?? positive(segment.totalElapsedTime)
  let start = timestamp(segment.startTime)
  let end = timestamp(segment.timestamp)
  if (!start && end && fitDurationS != null) start = new Date(end.getTime() - fitDurationS * 1000)
  if (!end && start && fitDurationS != null) end = new Date(start.getTime() + fitDurationS * 1000)
  if (!start || !end || end.getTime() <= start.getTime()) return null
  return { start, end, durationS: fitDurationS ?? (end.getTime() - start.getTime()) / 1000 }
}

function summitSegment(
  segment: SegmentLapMesg,
  records: readonly RecordMesg[],
): WahooSummitSegment | null {
  const uuid = text(segment.uuid)
  if (!uuid) return null
  const feature = summitFeature(uuid)
  if (!feature) return null
  const dates = segmentDates(segment)
  if (!dates) return null
  const enclosed = records.filter(record => {
    const time = recordTimestamp(record)
    return time >= dates.start.getTime() && time <= dates.end.getTime()
  })
  const distanceM = positive(segment.totalDistance) ?? recordDistance(enclosed)
  if (distanceM == null || distanceM <= 0 || dates.durationS <= 0) return null
  const elevationGainM = nonnegative(segment.totalAscent) ?? recordElevationGain(enclosed)
  const avgSpeedMps = nonnegative(segment.avgSpeed) ?? distanceM / dates.durationS
  return {
    feature,
    uuid,
    name: text(segment.name),
    startDate: dates.start.toISOString(),
    endDate: dates.end.toISOString(),
    distanceM,
    durationS: dates.durationS,
    elevationGainM,
    avgGradePct:
      finite(segment.avgGrade) ??
      (elevationGainM != null && distanceM > 0 ? (elevationGainM / distanceM) * 100 : null),
    avgSpeedMps,
    avgHeartRate:
      positive(segment.avgHeartRate) ?? average(enclosed.map(record => positive(record.heartRate))),
    avgPower:
      nonnegative(segment.avgPower) ?? average(enclosed.map(record => nonnegative(record.power))),
    avgCadence:
      nonnegative(segment.avgCadence) ??
      average(enclosed.map(record => nonnegative(record.cadence))),
  }
}

function preferSummitSegment(
  current: WahooSummitSegment,
  candidate: WahooSummitSegment,
): WahooSummitSegment {
  const endDiff = Date.parse(candidate.endDate) - Date.parse(current.endDate)
  if (endDiff !== 0) return endDiff > 0 ? candidate : current
  if (candidate.durationS !== current.durationS)
    return candidate.durationS > current.durationS ? candidate : current
  if (candidate.distanceM !== current.distanceM)
    return candidate.distanceM > current.distanceM ? candidate : current
  return candidate
}

function summitSegmentsFor(
  messages: FitMessages,
  records: readonly RecordMesg[],
): WahooSummitSegment[] {
  const segments = new Map<string, WahooSummitSegment>()
  for (const value of messages.segmentLapMesgs ?? []) {
    const segment = summitSegment(value, records)
    if (!segment) continue
    const key = JSON.stringify([
      segment.feature,
      segment.uuid,
      segment.name,
      segment.startDate,
      finite(value.startPositionLat),
      finite(value.startPositionLong),
    ])
    const current = segments.get(key)
    segments.set(key, current ? preferSummitSegment(current, segment) : segment)
  }
  return [...segments.values()].sort(
    (left, right) =>
      left.startDate.localeCompare(right.startDate) ||
      left.endDate.localeCompare(right.endDate) ||
      left.uuid.localeCompare(right.uuid),
  )
}

function finalCumulativeValue(values: readonly (number | null)[]): number | null {
  for (let index = values.length - 1; index >= 0; index--) {
    const value = values[index]
    if (value != null && Number.isFinite(value) && value >= 0) return value
  }
  return null
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
  const { messages, developerFields, profileVersion } = decodeMessages(bytes)
  const file = messages.fileIdMesgs?.[0]
  if (!file || file.type !== 'activity') throw new Error('Wahoo FIT file must be an activity')
  const session = messages.sessionMesgs?.[0]
  if (!session) throw new Error('Wahoo FIT file has no session')
  const start = timestamp(session.startTime) ?? timestamp(file.timeCreated)
  if (!start) throw new Error('Wahoo FIT file has no valid start time')
  const records = sortedRecords(messages)
  const streams = streamsFor(records, developerFields, start.getTime())
  return {
    startDate: start.toISOString(),
    sport: text(session.sport),
    sourceDevice: deviceName(messages),
    distanceM: nonnegative(session.totalDistance),
    movingTimeS: nonnegative(session.totalMovingTime ?? session.totalTimerTime),
    elapsedTimeS: nonnegative(session.totalElapsedTime),
    metrics: sessionMetrics(messages),
    streams,
    sweatLoss: {
      fluidMl: finalCumulativeValue(streams.fluidLossMl),
      sodiumMg: finalCumulativeValue(streams.sodiumLossMg),
    },
    gearShifts: fitGearShifts(messages),
    cyclingDynamics: fitCyclingDynamics(messages),
    summitSegments: summitSegmentsFor(messages, records),
    profileVersion,
  }
}
