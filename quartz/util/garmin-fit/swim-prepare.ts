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

export interface PreparedSample extends GarminSwimSample {
  speedMetersPerSecond: number
}

export interface PreparedPoolLength extends GarminPoolSwimLength {
  totalStrokes: number
}

export interface SwimStatistics {
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

export const DEFAULT_POOL_LENGTH_METERS = 25

function requireFinite(value: number, label: string): void {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`)
}

function requireOptionalFinite(value: number | undefined, label: string): void {
  if (value != null) requireFinite(value, label)
}

export function validateInput(input: GarminSwimFitInput): void {
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

export function normalizedSamples(input: GarminSwimFitInput): GarminSwimSample[] {
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

export function scalePoolDistances(
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

export function prepareSamples(samples: readonly GarminSwimSample[]): PreparedSample[] {
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

export function poolLengthSamples(
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

export function finiteValues(
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

export function statistics(samples: readonly PreparedSample[]): SwimStatistics {
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

export function totalStrokes(
  input: GarminSwimFitInput,
  samples: readonly GarminSwimSample[],
): number {
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

export function preparedPoolLengths(
  lengths: readonly GarminPoolSwimLength[],
): PreparedPoolLength[] {
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

export function poolLengthStatistics(
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

export function poolSummaryStroke(
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
