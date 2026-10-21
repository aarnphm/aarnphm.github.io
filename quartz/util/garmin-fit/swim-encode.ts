import {
  Encoder,
  Profile,
  type ActivityMesg,
  type DeviceInfoMesg,
  type EventMesg,
  type FileIdMesg,
  type LapMesg,
  type LengthMesg,
  type RecordMesg,
  type SessionMesg,
} from '@garmin/fitsdk'
import {
  DEFAULT_POOL_LENGTH_METERS,
  finiteValues,
  normalizedSamples,
  poolLengthSamples,
  poolLengthStatistics,
  poolSummaryStroke,
  preparedPoolLengths,
  prepareSamples,
  scalePoolDistances,
  statistics,
  totalStrokes,
  validateInput,
  type GarminPoolSwimInput,
  type GarminSwimFitInput,
  type GarminSwimSample,
  type PreparedPoolLength,
  type PreparedSample,
  type SwimStatistics,
} from './swim-prepare'
import { validateGarminFit, type GarminFitEncoding } from './validate'

const SEMICIRCLES_PER_DEGREE = 2 ** 31 / 180

function sampleDate(startTime: Date, elapsedSeconds: number): Date {
  return new Date(startTime.getTime() + Math.round(elapsedSeconds * 1000))
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

function recordMesg(input: GarminSwimFitInput, sample: PreparedSample): RecordMesg {
  const message: RecordMesg = {
    timestamp: sampleDate(input.startTime, sample.elapsedSeconds),
    distance: sample.distanceMeters,
    speed: sample.speedMetersPerSecond,
    enhancedSpeed: sample.speedMetersPerSecond,
  }
  if (input.kind === 'openWater') {
    if (sample.latitudeDegrees != null) message.positionLat = semicircles(sample.latitudeDegrees)
    if (sample.longitudeDegrees != null) message.positionLong = semicircles(sample.longitudeDegrees)
  }
  if (sample.altitudeMeters != null) {
    message.altitude = sample.altitudeMeters
    message.enhancedAltitude = sample.altitudeMeters
  }
  if (sample.heartRateBpm != null) message.heartRate = uint8(sample.heartRateBpm)
  if (sample.cadenceRpm != null) message.cadence = uint8(sample.cadenceRpm)
  if (sample.powerWatts != null) message.power = uint16(sample.powerWatts)
  if (sample.temperatureCelsius != null) message.temperature = sint8(sample.temperatureCelsius)
  return message
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

function writeHeaderMessages(
  encoder: Encoder,
  input: GarminSwimFitInput,
  serialNumber: number,
): void {
  const fileId: FileIdMesg = {
    type: 'activity',
    manufacturer: 'development',
    product: 0,
    serialNumber,
    timeCreated: input.startTime,
  }
  const deviceInfo: DeviceInfoMesg = {
    timestamp: input.startTime,
    deviceIndex: 'creator',
    manufacturer: 'development',
    product: 0,
    serialNumber,
    productName: 'Strava Swim Backfill',
    softwareVersion: 1,
  }
  const timerStart: EventMesg = { timestamp: input.startTime, event: 'timer', eventType: 'start' }
  encoder.onMesg(Profile.MesgNum.FILE_ID, fileId)
  encoder.onMesg(Profile.MesgNum.DEVICE_INFO, deviceInfo)
  encoder.onMesg(Profile.MesgNum.EVENT, timerStart)
}

function writeRecord(encoder: Encoder, input: GarminSwimFitInput, sample: PreparedSample): void {
  encoder.onMesg(Profile.MesgNum.RECORD, recordMesg(input, sample))
}

function writePoolTimeline(
  encoder: Encoder,
  input: GarminPoolSwimInput,
  samples: readonly PreparedSample[],
  lengths: readonly PreparedSample[],
  poolLengthMeters: number,
  strokeCount: number,
): void {
  writeRecord(encoder, input, samples[0])
  const totalLengthElapsed = lengths[lengths.length - 1]?.elapsedSeconds ?? input.elapsedTimeSeconds
  const timerScale = input.timerTimeSeconds / totalLengthElapsed
  let lengthIndex = 0
  let previousLengthElapsed = 0
  let previousCumulativeStrokes = 0
  for (let sampleIndex = 1; sampleIndex < samples.length; sampleIndex++) {
    const sample = samples[sampleIndex]
    let wroteSample = false
    while (
      lengthIndex < lengths.length &&
      lengths[lengthIndex].elapsedSeconds <= sample.elapsedSeconds
    ) {
      const lengthSample = lengths[lengthIndex]
      const elapsedDuration = lengthSample.elapsedSeconds - previousLengthElapsed
      const timerDuration = elapsedDuration * timerScale
      const cumulativeStrokes = Math.round((strokeCount * (lengthIndex + 1)) / lengths.length)
      const lengthStrokes = cumulativeStrokes - previousCumulativeStrokes
      const length: LengthMesg = {
        messageIndex: lengthIndex,
        timestamp: sampleDate(input.startTime, lengthSample.elapsedSeconds),
        event: 'length',
        eventType: 'stop',
        startTime: sampleDate(input.startTime, previousLengthElapsed),
        totalElapsedTime: elapsedDuration,
        totalTimerTime: timerDuration,
        totalStrokes: lengthStrokes,
        avgSpeed: poolLengthMeters / timerDuration,
        swimStroke: input.swimStroke ?? 'freestyle',
        avgSwimmingCadence: uint8((lengthStrokes * 60) / timerDuration),
        lengthType: 'active',
      }
      encoder.onMesg(Profile.MesgNum.LENGTH, length)
      writeRecord(encoder, input, lengthSample)
      wroteSample =
        lengthSample.elapsedSeconds === sample.elapsedSeconds &&
        lengthSample.distanceMeters === sample.distanceMeters
      previousLengthElapsed = lengthSample.elapsedSeconds
      previousCumulativeStrokes = cumulativeStrokes
      lengthIndex++
    }
    if (!wroteSample) writeRecord(encoder, input, sample)
  }
}

function writeExplicitPoolTimeline(
  encoder: Encoder,
  input: GarminPoolSwimInput,
  samples: readonly PreparedSample[],
  lengths: readonly PreparedPoolLength[],
): void {
  let sampleIndex = 0
  for (let lengthIndex = 0; lengthIndex < lengths.length; lengthIndex++) {
    const length = lengths[lengthIndex]
    while (
      sampleIndex < samples.length &&
      samples[sampleIndex].elapsedSeconds <= length.endElapsedSeconds
    ) {
      writeRecord(encoder, input, samples[sampleIndex])
      sampleIndex++
    }
    const elapsedDuration = length.endElapsedSeconds - length.startElapsedSeconds
    const strokeTime = length.strokeTimeSeconds ?? elapsedDuration
    const message: LengthMesg = {
      messageIndex: lengthIndex,
      timestamp: sampleDate(input.startTime, length.endElapsedSeconds),
      event: 'length',
      eventType: 'stop',
      startTime: sampleDate(input.startTime, length.startElapsedSeconds),
      totalElapsedTime: elapsedDuration,
      totalTimerTime: elapsedDuration,
      totalStrokes: length.totalStrokes,
      avgSpeed: length.distanceMeters / elapsedDuration,
      swimStroke: length.swimStroke ?? input.swimStroke ?? 'mixed',
      avgSwimmingCadence:
        length.totalStrokes > 0 ? uint8((length.totalStrokes * 60) / strokeTime) : undefined,
      lengthType: 'active',
    }
    encoder.onMesg(Profile.MesgNum.LENGTH, message)
  }
  while (sampleIndex < samples.length) {
    writeRecord(encoder, input, samples[sampleIndex])
    sampleIndex++
  }
}

function applyStatistics(target: LapMesg | SessionMesg, values: SwimStatistics): void {
  if (values.avgHeartRate != null) target.avgHeartRate = uint8(values.avgHeartRate)
  if (values.maxHeartRate != null) target.maxHeartRate = uint8(values.maxHeartRate)
  if (values.avgCadence != null) target.avgCadence = uint8(values.avgCadence)
  if (values.maxCadence != null) target.maxCadence = uint8(values.maxCadence)
  if (values.avgPower != null) target.avgPower = uint16(values.avgPower)
  if (values.maxPower != null) target.maxPower = uint16(values.maxPower)
  if (values.avgTemperature != null) target.avgTemperature = sint8(values.avgTemperature)
  if (values.maxTemperature != null) target.maxTemperature = sint8(values.maxTemperature)
  if (values.avgAltitude != null) target.avgAltitude = values.avgAltitude
  if (values.minAltitude != null) target.minAltitude = values.minAltitude
  if (values.maxAltitude != null) target.maxAltitude = values.maxAltitude
}

function gpsBounds(samples: readonly GarminSwimSample[]): {
  firstLat: number
  firstLong: number
  lastLat: number
  lastLong: number
  northEastLat: number
  northEastLong: number
  southWestLat: number
  southWestLong: number
} {
  const latitudes = finiteValues(samples, sample => sample.latitudeDegrees)
  const longitudes = finiteValues(samples, sample => sample.longitudeDegrees)
  return {
    firstLat: semicircles(latitudes[0]),
    firstLong: semicircles(longitudes[0]),
    lastLat: semicircles(latitudes[latitudes.length - 1]),
    lastLong: semicircles(longitudes[longitudes.length - 1]),
    northEastLat: semicircles(Math.max(...latitudes)),
    northEastLong: semicircles(Math.max(...longitudes)),
    southWestLat: semicircles(Math.min(...latitudes)),
    southWestLong: semicircles(Math.min(...longitudes)),
  }
}

function writeSummaryMessages(
  encoder: Encoder,
  input: GarminSwimFitInput,
  samples: readonly PreparedSample[],
  distanceMeters: number,
  strokeCount: number,
  lengthCount: number,
  poolLengthMeters: number | undefined,
  poolLengths: readonly PreparedPoolLength[] | undefined,
): void {
  const endTime = sampleDate(input.startTime, input.elapsedTimeSeconds)
  const values = statistics(samples)
  if (poolLengths) Object.assign(values, poolLengthStatistics(poolLengths))
  const avgSpeed = distanceMeters / input.timerTimeSeconds
  const timerStop: EventMesg = { timestamp: endTime, event: 'timer', eventType: 'stop' }
  const lap: LapMesg = {
    messageIndex: 0,
    timestamp: endTime,
    event: 'lap',
    eventType: 'stop',
    startTime: input.startTime,
    totalElapsedTime: input.elapsedTimeSeconds,
    totalTimerTime: input.timerTimeSeconds,
    totalDistance: distanceMeters,
    totalCycles: strokeCount,
    totalCalories: input.calories == null ? undefined : uint16(input.calories),
    avgSpeed,
    maxSpeed: values.maxSpeed,
    enhancedAvgSpeed: avgSpeed,
    enhancedMaxSpeed: values.maxSpeed,
    intensity: 'active',
    lapTrigger: 'sessionEnd',
    sport: 'swimming',
    subSport: input.kind === 'pool' ? 'lapSwimming' : 'openWater',
    totalMovingTime: input.timerTimeSeconds,
  }
  const session: SessionMesg = {
    messageIndex: 0,
    timestamp: endTime,
    event: 'session',
    eventType: 'stop',
    startTime: input.startTime,
    sport: 'swimming',
    subSport: input.kind === 'pool' ? 'lapSwimming' : 'openWater',
    totalElapsedTime: input.elapsedTimeSeconds,
    totalTimerTime: input.timerTimeSeconds,
    totalDistance: distanceMeters,
    totalCycles: strokeCount,
    totalCalories: input.calories == null ? undefined : uint16(input.calories),
    avgSpeed,
    maxSpeed: values.maxSpeed,
    enhancedAvgSpeed: avgSpeed,
    enhancedMaxSpeed: values.maxSpeed,
    firstLapIndex: 0,
    numLaps: 1,
    trigger: 'activityEnd',
    totalMovingTime: input.timerTimeSeconds,
  }
  if (strokeCount > 0) {
    lap.avgStrokeDistance = distanceMeters / strokeCount
    session.avgStrokeDistance = distanceMeters / strokeCount
  }
  if (input.kind === 'pool' && poolLengthMeters != null) {
    const swimStroke = poolSummaryStroke(input, poolLengths)
    lap.firstLengthIndex = 0
    lap.numLengths = lengthCount
    lap.numActiveLengths = lengthCount
    lap.swimStroke = swimStroke
    session.numLengths = lengthCount
    session.numActiveLengths = lengthCount
    session.swimStroke = swimStroke
    session.poolLength = poolLengthMeters
    session.poolLengthUnit = 'metric'
  } else {
    const bounds = gpsBounds(samples)
    lap.startPositionLat = bounds.firstLat
    lap.startPositionLong = bounds.firstLong
    lap.endPositionLat = bounds.lastLat
    lap.endPositionLong = bounds.lastLong
    session.startPositionLat = bounds.firstLat
    session.startPositionLong = bounds.firstLong
    session.endPositionLat = bounds.lastLat
    session.endPositionLong = bounds.lastLong
    session.necLat = bounds.northEastLat
    session.necLong = bounds.northEastLong
    session.swcLat = bounds.southWestLat
    session.swcLong = bounds.southWestLong
  }
  applyStatistics(lap, values)
  applyStatistics(session, values)
  const activity: ActivityMesg = {
    timestamp: endTime,
    totalTimerTime: input.timerTimeSeconds,
    numSessions: 1,
    type: 'manual',
    event: 'activity',
    eventType: 'stop',
  }
  encoder.onMesg(Profile.MesgNum.EVENT, timerStop)
  encoder.onMesg(Profile.MesgNum.LAP, lap)
  encoder.onMesg(Profile.MesgNum.SESSION, session)
  encoder.onMesg(Profile.MesgNum.ACTIVITY, activity)
}

export function encodeGarminSwimFit(input: GarminSwimFitInput): GarminFitEncoding {
  validateInput(input)
  let samples = normalizedSamples(input)
  let distanceMeters = input.distanceMeters
  let poolLengthMeters: number | undefined
  let poolLengths: PreparedPoolLength[] | undefined
  let lengthCount = 0
  if (input.kind === 'pool') {
    poolLengthMeters = input.poolLengthMeters ?? DEFAULT_POOL_LENGTH_METERS
    poolLengths = input.lengths?.length ? preparedPoolLengths(input.lengths) : undefined
    lengthCount = poolLengths?.length ?? Math.round(input.distanceMeters / poolLengthMeters)
    if (lengthCount < 1) throw new Error('pool swim distance must contain at least one length')
    if (poolLengths) {
      distanceMeters = poolLengths.reduce((sum, length) => sum + length.distanceMeters, 0)
    } else {
      distanceMeters = lengthCount * poolLengthMeters
      samples = scalePoolDistances(samples, input.distanceMeters, distanceMeters)
    }
  }
  const prepared = prepareSamples(samples)
  const strokeCount =
    poolLengths?.reduce((sum, length) => sum + length.totalStrokes, 0) ??
    totalStrokes(input, prepared)
  const encoder = new Encoder()
  writeHeaderMessages(encoder, input, hashSourceId(input.sourceId))
  if (input.kind === 'pool' && poolLengthMeters != null) {
    if (poolLengths) {
      writeExplicitPoolTimeline(encoder, input, prepared, poolLengths)
    } else {
      const lengths = poolLengthSamples(prepared, poolLengthMeters, lengthCount)
      writePoolTimeline(encoder, input, prepared, lengths, poolLengthMeters, strokeCount)
    }
  } else {
    for (const sample of prepared) writeRecord(encoder, input, sample)
  }
  writeSummaryMessages(
    encoder,
    input,
    prepared,
    distanceMeters,
    strokeCount,
    lengthCount,
    poolLengthMeters,
    poolLengths,
  )
  const bytes = encoder.close()
  return { bytes, validation: validateGarminFit(bytes) }
}
