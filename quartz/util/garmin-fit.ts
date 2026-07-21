import {
  Decoder,
  Encoder,
  Profile,
  Stream,
  type ActivityMesg,
  type DeviceInfoMesg,
  type EventMesg,
  type FileIdMesg,
  type LapMesg,
  type LengthMesg,
  type RecordMesg,
  type SessionMesg,
} from '@garmin/fitsdk'

export type GarminSwimStroke =
  | 'freestyle'
  | 'backstroke'
  | 'breaststroke'
  | 'butterfly'
  | 'drill'
  | 'mixed'
  | 'im'

export interface GarminSwimSample {
  elapsedSeconds: number
  distanceMeters: number
  latitudeDegrees?: number
  longitudeDegrees?: number
  altitudeMeters?: number
  heartRateBpm?: number
  cadenceRpm?: number
  powerWatts?: number
  temperatureCelsius?: number
}

export interface GarminPoolSwimLength {
  startElapsedSeconds: number
  endElapsedSeconds: number
  distanceMeters: number
  totalStrokes?: number
  strokeTimeSeconds?: number
  swimStroke?: GarminSwimStroke
}

export interface GarminSwimFitBaseInput {
  sourceId: string | number
  name: string
  startTime: Date
  distanceMeters: number
  elapsedTimeSeconds: number
  timerTimeSeconds: number
  samples: readonly GarminSwimSample[]
  calories?: number
  totalStrokes?: number
  swimStroke?: GarminSwimStroke
}

export interface GarminPoolSwimInput extends GarminSwimFitBaseInput {
  kind: 'pool'
  poolLengthMeters?: number
  lengths?: readonly GarminPoolSwimLength[]
}

export interface GarminOpenWaterSwimInput extends GarminSwimFitBaseInput {
  kind: 'openWater'
}

export type GarminSwimFitInput = GarminPoolSwimInput | GarminOpenWaterSwimInput

export interface GarminFitMessageCounts {
  fileIds: number
  deviceInfos: number
  events: number
  records: number
  lengths: number
  laps: number
  sessions: number
  activities: number
}

export interface GarminFitValidation {
  valid: boolean
  isFit: boolean
  integrity: boolean
  errors: readonly string[]
  counts: GarminFitMessageCounts
}

export interface GarminFitEncoding {
  bytes: Uint8Array
  validation: GarminFitValidation
}

interface PreparedSample extends GarminSwimSample {
  speedMetersPerSecond: number
}

interface PreparedPoolLength extends GarminPoolSwimLength {
  totalStrokes: number
}

interface SwimStatistics {
  avgHeartRate?: number
  maxHeartRate?: number
  avgCadence?: number
  maxCadence?: number
  avgPower?: number
  maxPower?: number
  avgTemperature?: number
  maxTemperature?: number
  avgAltitude?: number
  minAltitude?: number
  maxAltitude?: number
  maxSpeed: number
}

const DEFAULT_POOL_LENGTH_METERS = 25
const SEMICIRCLES_PER_DEGREE = 2 ** 31 / 180

function requireFinite(value: number, label: string): void {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`)
}

function requireOptionalFinite(value: number | undefined, label: string): void {
  if (value != null) requireFinite(value, label)
}

function validateInput(input: GarminSwimFitInput): void {
  if (!input.name.trim()) throw new Error('name must not be empty')
  if (typeof input.sourceId === 'string' && !input.sourceId.trim())
    throw new Error('sourceId must not be empty')
  if (!Number.isFinite(input.startTime.getTime())) throw new Error('startTime must be valid')
  requireFinite(input.distanceMeters, 'distanceMeters')
  requireFinite(input.elapsedTimeSeconds, 'elapsedTimeSeconds')
  requireFinite(input.timerTimeSeconds, 'timerTimeSeconds')
  if (input.distanceMeters <= 0) throw new Error('distanceMeters must be positive')
  if (input.elapsedTimeSeconds <= 0) throw new Error('elapsedTimeSeconds must be positive')
  if (input.timerTimeSeconds <= 0 || input.timerTimeSeconds > input.elapsedTimeSeconds)
    throw new Error('timerTimeSeconds must be positive and no greater than elapsedTimeSeconds')
  if (input.samples.length < 2) throw new Error('samples must contain at least two points')
  requireOptionalFinite(input.calories, 'calories')
  requireOptionalFinite(input.totalStrokes, 'totalStrokes')
  if (input.calories != null && input.calories < 0) throw new Error('calories must be nonnegative')
  if (input.totalStrokes != null && input.totalStrokes < 0)
    throw new Error('totalStrokes must be nonnegative')
  if (input.kind === 'pool') {
    const poolLength = input.poolLengthMeters ?? DEFAULT_POOL_LENGTH_METERS
    requireFinite(poolLength, 'poolLengthMeters')
    if (poolLength <= 0) throw new Error('poolLengthMeters must be positive')
    let previousEnd = 0
    let lengthDistance = 0
    for (let index = 0; index < (input.lengths?.length ?? 0); index++) {
      const length = input.lengths?.[index]
      if (!length) continue
      requireFinite(length.startElapsedSeconds, `lengths[${index}].startElapsedSeconds`)
      requireFinite(length.endElapsedSeconds, `lengths[${index}].endElapsedSeconds`)
      requireFinite(length.distanceMeters, `lengths[${index}].distanceMeters`)
      requireOptionalFinite(length.totalStrokes, `lengths[${index}].totalStrokes`)
      requireOptionalFinite(length.strokeTimeSeconds, `lengths[${index}].strokeTimeSeconds`)
      if (
        length.startElapsedSeconds < previousEnd ||
        length.endElapsedSeconds <= length.startElapsedSeconds ||
        length.endElapsedSeconds > input.elapsedTimeSeconds
      )
        throw new Error(`lengths[${index}] has an invalid activity interval`)
      if (Math.abs(length.distanceMeters - poolLength) > Math.max(0.5, poolLength * 0.02))
        throw new Error(`lengths[${index}].distanceMeters does not match poolLengthMeters`)
      if (length.totalStrokes != null && length.totalStrokes < 0)
        throw new Error(`lengths[${index}].totalStrokes must be nonnegative`)
      if (
        length.strokeTimeSeconds != null &&
        (length.strokeTimeSeconds <= 0 ||
          length.strokeTimeSeconds > length.endElapsedSeconds - length.startElapsedSeconds)
      )
        throw new Error(`lengths[${index}].strokeTimeSeconds is outside the length duration`)
      previousEnd = length.endElapsedSeconds
      lengthDistance += length.distanceMeters
    }
    if (
      input.lengths &&
      input.lengths.length > 0 &&
      Math.abs(lengthDistance - input.distanceMeters) > Math.max(1, input.distanceMeters * 0.01)
    )
      throw new Error('length distance does not match distanceMeters')
  }

  let previousElapsed = -1
  let previousDistance = -1
  for (let index = 0; index < input.samples.length; index++) {
    const sample = input.samples[index]
    requireFinite(sample.elapsedSeconds, `samples[${index}].elapsedSeconds`)
    requireFinite(sample.distanceMeters, `samples[${index}].distanceMeters`)
    requireOptionalFinite(sample.latitudeDegrees, `samples[${index}].latitudeDegrees`)
    requireOptionalFinite(sample.longitudeDegrees, `samples[${index}].longitudeDegrees`)
    requireOptionalFinite(sample.altitudeMeters, `samples[${index}].altitudeMeters`)
    requireOptionalFinite(sample.heartRateBpm, `samples[${index}].heartRateBpm`)
    requireOptionalFinite(sample.cadenceRpm, `samples[${index}].cadenceRpm`)
    requireOptionalFinite(sample.powerWatts, `samples[${index}].powerWatts`)
    requireOptionalFinite(sample.temperatureCelsius, `samples[${index}].temperatureCelsius`)
    if (sample.elapsedSeconds < 0 || sample.elapsedSeconds > input.elapsedTimeSeconds)
      throw new Error(`samples[${index}].elapsedSeconds is outside the activity duration`)
    if (sample.distanceMeters < 0 || sample.distanceMeters > input.distanceMeters)
      throw new Error(`samples[${index}].distanceMeters is outside the activity distance`)
    if (sample.elapsedSeconds <= previousElapsed)
      throw new Error('sample elapsedSeconds values must be strictly increasing')
    if (sample.distanceMeters < previousDistance)
      throw new Error('sample distanceMeters values must be nondecreasing')
    if ((sample.latitudeDegrees == null) !== (sample.longitudeDegrees == null))
      throw new Error(`samples[${index}] must provide both latitudeDegrees and longitudeDegrees`)
    if (
      sample.latitudeDegrees != null &&
      (sample.latitudeDegrees < -90 || sample.latitudeDegrees > 90)
    )
      throw new Error(`samples[${index}].latitudeDegrees is outside -90 to 90`)
    if (
      sample.longitudeDegrees != null &&
      (sample.longitudeDegrees < -180 || sample.longitudeDegrees > 180)
    )
      throw new Error(`samples[${index}].longitudeDegrees is outside -180 to 180`)
    if (input.kind === 'openWater' && sample.latitudeDegrees == null)
      throw new Error(`samples[${index}] is missing open-water GPS coordinates`)
    previousElapsed = sample.elapsedSeconds
    previousDistance = sample.distanceMeters
  }
}

function copySample(
  sample: GarminSwimSample,
  elapsedSeconds: number,
  distanceMeters: number,
): GarminSwimSample {
  return { ...sample, elapsedSeconds, distanceMeters }
}

function normalizedSamples(input: GarminSwimFitInput): GarminSwimSample[] {
  const samples = input.samples.map(sample => ({ ...sample }))
  const first = samples[0]
  if (first.elapsedSeconds > 0) samples.unshift(copySample(first, 0, 0))
  else if (first.distanceMeters !== 0)
    throw new Error('the first sample must start at zero distance when elapsedSeconds is zero')

  const lastIndex = samples.length - 1
  const last = samples[lastIndex]
  if (last.elapsedSeconds < input.elapsedTimeSeconds) {
    samples.push(copySample(last, input.elapsedTimeSeconds, input.distanceMeters))
  } else if (last.distanceMeters !== input.distanceMeters) {
    samples[lastIndex] = copySample(last, last.elapsedSeconds, input.distanceMeters)
  }
  return samples
}

function scalePoolDistances(
  samples: readonly GarminSwimSample[],
  inputDistance: number,
  outputDistance: number,
): GarminSwimSample[] {
  const scale = outputDistance / inputDistance
  return samples.map(sample => ({ ...sample, distanceMeters: sample.distanceMeters * scale }))
}

function segmentSpeed(left: GarminSwimSample, right: GarminSwimSample): number {
  return Math.max(
    0,
    (right.distanceMeters - left.distanceMeters) / (right.elapsedSeconds - left.elapsedSeconds),
  )
}

function prepareSamples(samples: readonly GarminSwimSample[]): PreparedSample[] {
  return samples.map((sample, index) => {
    const left = index === 0 ? sample : samples[index - 1]
    const right = index === 0 ? samples[1] : sample
    return { ...sample, speedMetersPerSecond: segmentSpeed(left, right) }
  })
}

function interpolateOptional(
  left: number | undefined,
  right: number | undefined,
  ratio: number,
): number | undefined {
  if (left != null && right != null) return left + (right - left) * ratio
  return right ?? left
}

function interpolateSample(
  left: PreparedSample,
  right: PreparedSample,
  distanceMeters: number,
): PreparedSample {
  const distanceDelta = right.distanceMeters - left.distanceMeters
  const rawRatio = distanceDelta > 0 ? (distanceMeters - left.distanceMeters) / distanceDelta : 1
  const ratio = Number.isFinite(rawRatio) ? rawRatio : 1
  return {
    elapsedSeconds: left.elapsedSeconds + (right.elapsedSeconds - left.elapsedSeconds) * ratio,
    distanceMeters,
    speedMetersPerSecond: segmentSpeed(left, right),
    latitudeDegrees: interpolateOptional(left.latitudeDegrees, right.latitudeDegrees, ratio),
    longitudeDegrees: interpolateOptional(left.longitudeDegrees, right.longitudeDegrees, ratio),
    altitudeMeters: interpolateOptional(left.altitudeMeters, right.altitudeMeters, ratio),
    heartRateBpm: interpolateOptional(left.heartRateBpm, right.heartRateBpm, ratio),
    cadenceRpm: interpolateOptional(left.cadenceRpm, right.cadenceRpm, ratio),
    powerWatts: interpolateOptional(left.powerWatts, right.powerWatts, ratio),
    temperatureCelsius: interpolateOptional(
      left.temperatureCelsius,
      right.temperatureCelsius,
      ratio,
    ),
  }
}

function poolLengthSamples(
  samples: readonly PreparedSample[],
  poolLengthMeters: number,
  lengthCount: number,
): PreparedSample[] {
  const lengths: PreparedSample[] = []
  let segmentIndex = 1
  for (let lengthIndex = 1; lengthIndex <= lengthCount; lengthIndex++) {
    const distance = lengthIndex * poolLengthMeters
    while (segmentIndex < samples.length - 1 && samples[segmentIndex].distanceMeters < distance)
      segmentIndex++
    lengths.push(interpolateSample(samples[segmentIndex - 1], samples[segmentIndex], distance))
  }
  return lengths
}

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

function finiteValues(
  samples: readonly GarminSwimSample[],
  read: (sample: GarminSwimSample) => number | undefined,
): number[] {
  const values: number[] = []
  for (const sample of samples) {
    const value = read(sample)
    if (value != null) values.push(value)
  }
  return values
}

function mean(values: readonly number[]): number | undefined {
  if (values.length === 0) return undefined
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function maximum(values: readonly number[]): number | undefined {
  return values.length > 0 ? Math.max(...values) : undefined
}

function minimum(values: readonly number[]): number | undefined {
  return values.length > 0 ? Math.min(...values) : undefined
}

function statistics(samples: readonly PreparedSample[]): SwimStatistics {
  const heartRates = finiteValues(samples, sample => sample.heartRateBpm)
  const cadences = finiteValues(samples, sample => sample.cadenceRpm)
  const powers = finiteValues(samples, sample => sample.powerWatts)
  const temperatures = finiteValues(samples, sample => sample.temperatureCelsius)
  const altitudes = finiteValues(samples, sample => sample.altitudeMeters)
  return {
    avgHeartRate: mean(heartRates),
    maxHeartRate: maximum(heartRates),
    avgCadence: mean(cadences),
    maxCadence: maximum(cadences),
    avgPower: mean(powers),
    maxPower: maximum(powers),
    avgTemperature: mean(temperatures),
    maxTemperature: maximum(temperatures),
    avgAltitude: mean(altitudes),
    minAltitude: minimum(altitudes),
    maxAltitude: maximum(altitudes),
    maxSpeed: Math.max(...samples.map(sample => sample.speedMetersPerSecond)),
  }
}

function totalStrokes(input: GarminSwimFitInput, samples: readonly GarminSwimSample[]): number {
  if (input.totalStrokes != null) return Math.round(input.totalStrokes)
  let strokes = 0
  for (let index = 1; index < samples.length; index++) {
    const left = samples[index - 1]
    const right = samples[index]
    const cadence = mean(
      [left.cadenceRpm, right.cadenceRpm].filter((value): value is number => value != null),
    )
    if (cadence != null) strokes += (cadence * (right.elapsedSeconds - left.elapsedSeconds)) / 60
  }
  return Math.round(strokes)
}

function preparedPoolLengths(lengths: readonly GarminPoolSwimLength[]): PreparedPoolLength[] {
  const target = Math.round(
    lengths.reduce((sum, length) => sum + Math.max(0, length.totalStrokes ?? 0), 0),
  )
  const prepared = lengths.map(length => ({
    ...length,
    totalStrokes: Math.floor(Math.max(0, length.totalStrokes ?? 0)),
  }))
  const assigned = prepared.reduce((sum, length) => sum + length.totalStrokes, 0)
  const order = lengths
    .map((length, index) => ({ index, remainder: Math.max(0, length.totalStrokes ?? 0) % 1 }))
    .sort((left, right) => right.remainder - left.remainder || left.index - right.index)
  for (let offset = 0; offset < target - assigned; offset++) {
    const index = order[offset]?.index
    if (index != null && prepared[index]) prepared[index].totalStrokes++
  }
  return prepared
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

function poolLengthStatistics(
  lengths: readonly PreparedPoolLength[],
): Pick<SwimStatistics, 'avgCadence' | 'maxCadence' | 'maxSpeed'> {
  const cadenceLengths = lengths.filter(
    length =>
      length.totalStrokes > 0 &&
      (length.strokeTimeSeconds ?? length.endElapsedSeconds - length.startElapsedSeconds) > 0,
  )
  const totalStrokes = cadenceLengths.reduce((sum, length) => sum + length.totalStrokes, 0)
  const totalStrokeTime = cadenceLengths.reduce(
    (sum, length) =>
      sum + (length.strokeTimeSeconds ?? length.endElapsedSeconds - length.startElapsedSeconds),
    0,
  )
  const cadences = cadenceLengths.map(
    length =>
      (length.totalStrokes * 60) /
      (length.strokeTimeSeconds ?? length.endElapsedSeconds - length.startElapsedSeconds),
  )
  return {
    avgCadence: totalStrokeTime > 0 ? (totalStrokes * 60) / totalStrokeTime : undefined,
    maxCadence: maximum(cadences),
    maxSpeed: Math.max(
      ...lengths.map(
        length => length.distanceMeters / (length.endElapsedSeconds - length.startElapsedSeconds),
      ),
    ),
  }
}

function poolSummaryStroke(
  input: GarminPoolSwimInput,
  lengths: readonly PreparedPoolLength[] | undefined,
): GarminSwimStroke {
  if (!lengths) return input.swimStroke ?? 'freestyle'
  const strokes = new Set(
    lengths
      .map(length => length.swimStroke)
      .filter((stroke): stroke is GarminSwimStroke => stroke != null),
  )
  const onlyStroke = strokes.size === 1 ? strokes.values().next().value : undefined
  if (onlyStroke) return onlyStroke
  return 'mixed'
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

function emptyCounts(): GarminFitMessageCounts {
  return {
    fileIds: 0,
    deviceInfos: 0,
    events: 0,
    records: 0,
    lengths: 0,
    laps: 0,
    sessions: 0,
    activities: 0,
  }
}

export function validateGarminFit(bytes: Uint8Array): GarminFitValidation {
  const errors: string[] = []
  let isFit = false
  let integrity = false
  let counts = emptyCounts()
  try {
    isFit = Decoder.isFIT(Stream.fromByteArray(bytes))
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error))
  }
  if (!isFit) errors.push('input is not a FIT file')
  if (isFit) {
    try {
      integrity = new Decoder(Stream.fromByteArray(bytes)).checkIntegrity()
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error))
    }
    if (!integrity) errors.push('FIT integrity check failed')
    try {
      const decoded = new Decoder(Stream.fromByteArray(bytes)).read()
      errors.push(...decoded.errors.map(error => error.message))
      counts = {
        fileIds: decoded.messages.fileIdMesgs?.length ?? 0,
        deviceInfos: decoded.messages.deviceInfoMesgs?.length ?? 0,
        events: decoded.messages.eventMesgs?.length ?? 0,
        records: decoded.messages.recordMesgs?.length ?? 0,
        lengths: decoded.messages.lengthMesgs?.length ?? 0,
        laps: decoded.messages.lapMesgs?.length ?? 0,
        sessions: decoded.messages.sessionMesgs?.length ?? 0,
        activities: decoded.messages.activityMesgs?.length ?? 0,
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error))
    }
  }
  const complete =
    counts.fileIds === 1 &&
    counts.deviceInfos >= 1 &&
    counts.events >= 2 &&
    counts.records >= 1 &&
    counts.laps >= 1 &&
    counts.sessions >= 1 &&
    counts.activities === 1
  if (isFit && integrity && !complete) errors.push('FIT activity messages are incomplete')
  return {
    valid: isFit && integrity && complete && errors.length === 0,
    isFit,
    integrity,
    errors,
    counts,
  }
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
