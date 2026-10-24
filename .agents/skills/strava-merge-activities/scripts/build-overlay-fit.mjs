import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const argv = process.argv.slice(2)

function option(name, required = false) {
  const index = argv.indexOf(`--${name}`)
  const value = index >= 0 ? argv[index + 1] : undefined
  if (required && (!value || value.startsWith('--'))) throw new Error(`missing --${name}`)
  return value
}

function flag(name) {
  return argv.includes(`--${name}`)
}

function activityId(value, name) {
  const match = value?.match(/(?:activities\/)?(\d{8,})/)
  if (!match) throw new Error(`invalid --${name}`)
  return match[1]
}

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

function mean(values) {
  return values.length > 0
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : undefined
}

function maximum(values) {
  return values.length > 0 ? Math.max(...values) : undefined
}

function minimum(values) {
  return values.length > 0 ? Math.min(...values) : undefined
}

function uint8(value) {
  return Math.max(0, Math.min(254, Math.round(value)))
}

function uint16(value) {
  return Math.max(0, Math.min(65_534, Math.round(value)))
}

function semicircles(value) {
  return Math.max(-2_147_483_648, Math.min(2_147_483_647, Math.round((value * 2 ** 31) / 180)))
}

function streamField(stream, field) {
  if (!stream || !Array.isArray(stream.time) || stream.time.length < 2) return undefined
  const values = stream[field]
  if (!Array.isArray(values) || values.length < 2) return undefined
  if (values.length !== stream.time.length) throw new Error(`${field} length does not match time`)
  return values
}

function validateStream(stream, id) {
  if (!stream) return
  if (!Array.isArray(stream.time) || stream.time.length < 2)
    throw new Error(`activity ${id} has an invalid time stream`)
  if (
    stream.time.some(
      (value, index) => !finite(value) || (index > 0 && value <= stream.time[index - 1]),
    )
  )
    throw new Error(`activity ${id} time stream is not strictly increasing`)
  for (const field of ['latlng', 'altitude', 'distance', 'watts', 'heartrate', 'cadence'])
    streamField(stream, field)
}

function bracket(times, elapsedSeconds) {
  if (elapsedSeconds <= times[0]) return [0, 0, 0]
  const lastIndex = times.length - 1
  if (elapsedSeconds >= times[lastIndex]) return [lastIndex, lastIndex, 0]
  let low = 0
  let high = times.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (times[middle] < elapsedSeconds) low = middle + 1
    else high = middle
  }
  const left = low - 1
  const right = low
  const ratio = (elapsedSeconds - times[left]) / (times[right] - times[left])
  return [left, right, ratio]
}

function sampleNumber(stream, field, elapsedSeconds) {
  const values = streamField(stream, field)
  if (!values) return undefined
  const [left, right, ratio] = bracket(stream.time, elapsedSeconds)
  const leftValue = values[left]
  const rightValue = values[right]
  if (finite(leftValue) && finite(rightValue)) return leftValue + (rightValue - leftValue) * ratio
  if (finite(rightValue)) return rightValue
  if (finite(leftValue)) return leftValue
  return undefined
}

function sampleLatLng(stream, elapsedSeconds) {
  const values = streamField(stream, 'latlng')
  if (!values) return undefined
  const [left, right, ratio] = bracket(stream.time, elapsedSeconds)
  const leftValue = values[left]
  const rightValue = values[right]
  const valid = value =>
    Array.isArray(value) && value.length === 2 && finite(value[0]) && finite(value[1])
  if (valid(leftValue) && valid(rightValue))
    return [
      leftValue[0] + (rightValue[0] - leftValue[0]) * ratio,
      leftValue[1] + (rightValue[1] - leftValue[1]) * ratio,
    ]
  if (valid(rightValue)) return rightValue
  if (valid(leftValue)) return leftValue
  return undefined
}

function activityDate(activity, id) {
  const date = new Date(activity.startDate)
  if (!Number.isFinite(date.getTime())) throw new Error(`activity ${id} has an invalid start date`)
  return date
}

function outputAvailable(filePath, force) {
  return fs
    .access(filePath)
    .then(() => {
      if (!force) throw new Error(`${filePath} already exists; choose a new path or pass --force`)
    })
    .catch(error => {
      if (error?.code !== 'ENOENT') throw error
    })
}

const workspace = path.resolve(option('workspace', true))
const cachePath = path.resolve(
  option('cache') ?? path.join(workspace, 'quartz/.quartz-cache/strava.json'),
)
const sourceId = activityId(option('source', true), 'source')
const destinationId = activityId(option('destination', true), 'destination')
const outputPath = path.resolve(option('output', true))
const backupPath = path.resolve(option('backup', true))
const productName = option('product') ?? 'Strava Merge'
const force = flag('force')

if (sourceId === destinationId) throw new Error('source and destination must differ')
await Promise.all([outputAvailable(outputPath, force), outputAvailable(backupPath, force)])

const cache = JSON.parse(await fs.readFile(cachePath, 'utf8'))
const source = cache.activities?.[sourceId]
const destination = cache.activities?.[destinationId]
if (!source) throw new Error(`source activity ${sourceId} is missing from the cache`)
if (!destination) throw new Error(`destination activity ${destinationId} is missing from the cache`)

const sourceStream = cache.streams?.[sourceId]
const destinationStream = cache.streams?.[destinationId]
validateStream(sourceStream, sourceId)
validateStream(destinationStream, destinationId)

const sourceFields = ['latlng', 'altitude', 'distance', 'watts', 'heartrate', 'cadence'].filter(
  field => streamField(sourceStream, field) != null,
)
if (sourceFields.length === 0)
  throw new Error(`source activity ${sourceId} has no standard telemetry`)

const destinationStart = activityDate(destination, destinationId)
const sourceStart = activityDate(source, sourceId)
const duration = Math.round(destination.elapsedTime)
const timerTime = Math.round(destination.movingTime > 0 ? destination.movingTime : duration)
const sourceDuration = Math.round(source.elapsedTime)
if (duration <= 0 || timerTime <= 0 || sourceDuration <= 0)
  throw new Error('activities must have positive durations')

const sourceOffsetSeconds = (sourceStart.getTime() - destinationStart.getTime()) / 1_000
const overlapStart = Math.max(0, sourceOffsetSeconds)
const overlapEnd = Math.min(duration, sourceOffsetSeconds + sourceDuration)
const overlapSeconds = Math.max(0, overlapEnd - overlapStart)
const overlapRatio = overlapSeconds / Math.min(duration, sourceDuration)
const destinationOverlapRatio = overlapSeconds / duration
if (overlapRatio < 0.5)
  throw new Error(`activity overlap ${overlapRatio.toFixed(3)} is below the 0.5 overlay threshold`)

const sourceTime = elapsedSeconds => elapsedSeconds - sourceOffsetSeconds

const spans = new Map(
  ['latlng', 'altitude', 'distance', 'watts', 'heartrate', 'cadence'].map(field => [
    field,
    {
      source: streamField(sourceStream, field)
        ? [sourceStream.time[0], sourceStream.time.at(-1)]
        : undefined,
      destination: streamField(destinationStream, field)
        ? [destinationStream.time[0], destinationStream.time.at(-1)]
        : undefined,
    },
  ]),
)

function measured(field, elapsedSeconds) {
  const span = spans.get(field)
  if (!span) return false
  if (
    span.destination &&
    elapsedSeconds >= span.destination[0] &&
    elapsedSeconds <= span.destination[1]
  )
    return true
  const offset = sourceTime(elapsedSeconds)
  return span.source != null && offset >= span.source[0] && offset <= span.source[1]
}

function measuredValues(recordField, field) {
  return records.flatMap((record, index) =>
    finite(record[recordField]) && measured(field, index) ? [record[recordField]] : [],
  )
}

function sampleField(field, elapsedSeconds) {
  const sources =
    field === 'heartrate'
      ? [
          [sourceStream, sourceTime(elapsedSeconds)],
          [destinationStream, elapsedSeconds],
        ]
      : [
          [destinationStream, elapsedSeconds],
          [sourceStream, sourceTime(elapsedSeconds)],
        ]
  for (const [stream, streamTime] of sources) {
    const value =
      field === 'latlng'
        ? sampleLatLng(stream, streamTime)
        : sampleNumber(stream, field, streamTime)
    if (value != null) return value
  }
  return undefined
}

const records = []
let lastDistance = 0
for (let elapsedSeconds = 0; elapsedSeconds <= duration; elapsedSeconds++) {
  const sampledDistance = sampleField('distance', elapsedSeconds)
  const distance = finite(sampledDistance) ? Math.max(lastDistance, sampledDistance) : lastDistance
  const speed = elapsedSeconds > 0 ? Math.max(0, distance - lastDistance) : 0
  const heartRate = sampleField('heartrate', elapsedSeconds)
  const cadence = sampleField('cadence', elapsedSeconds)
  const power = sampleField('watts', elapsedSeconds)
  const altitude = sampleField('altitude', elapsedSeconds)
  const latlng = sampleField('latlng', elapsedSeconds)
  records.push({
    timestamp: new Date(destinationStart.getTime() + elapsedSeconds * 1_000),
    distance,
    speed,
    heartRate: finite(heartRate) ? uint8(heartRate) : undefined,
    cadence: finite(cadence) ? uint8(cadence) : undefined,
    power: finite(power) ? uint16(power) : undefined,
    altitude: finite(altitude) ? altitude : undefined,
    latlng,
  })
  lastDistance = distance
}

const values = field => records.flatMap(record => (finite(record[field]) ? [record[field]] : []))
const heartRates = values('heartRate')
const cadences = values('cadence')
const powers = values('power')
const altitudes = values('altitude')
const speeds = values('speed')
const measuredHeartRates = measuredValues('heartRate', 'heartrate')
const measuredCadences = measuredValues('cadence', 'cadence')
const measuredPowers = measuredValues('power', 'watts')
const measuredAltitudes = measuredValues('altitude', 'altitude')
const totalDistance = destination.distance > 0 ? destination.distance : records.at(-1).distance
const calories = destination.calories > 0 ? destination.calories : Math.max(0, source.calories ?? 0)
const derivedAscent = records.reduce((sum, record, index) => {
  if (index === 0 || !finite(record.altitude) || !finite(records[index - 1].altitude)) return sum
  return sum + Math.max(0, record.altitude - records[index - 1].altitude)
}, 0)
const totalAscent =
  destination.totalElevationGain > 0 ? destination.totalElevationGain : derivedAscent
const end = new Date(destinationStart.getTime() + duration * 1_000)
const averageHeartRate = mean(heartRates)
const maximumHeartRate = maximum(heartRates)
const averageCadence = mean(cadences)
const maximumCadence = maximum(cadences)
const averagePower = mean(powers)
const maximumPower = maximum(powers)
const averageSpeed = timerTime > 0 ? totalDistance / timerTime : 0
const maximumSpeed = maximum(speeds) ?? 0

const sdkPath = path.join(workspace, 'node_modules/@garmin/fitsdk/src/index.js')
const { Decoder, Encoder, Profile, Stream } = await import(pathToFileURL(sdkPath).href)
const encoder = new Encoder()
const serialNumber = Number(BigInt(destinationId) % 4_294_967_295n)

encoder.onMesg(Profile.MesgNum.FILE_ID, {
  type: 'activity',
  manufacturer: 'development',
  product: 0,
  serialNumber,
  timeCreated: destinationStart,
  productName,
})
encoder.onMesg(Profile.MesgNum.DEVICE_INFO, {
  timestamp: destinationStart,
  deviceIndex: 'creator',
  manufacturer: 'development',
  product: 0,
  serialNumber,
  productName,
  softwareVersion: 1,
})
encoder.onMesg(Profile.MesgNum.EVENT, {
  timestamp: destinationStart,
  event: 'timer',
  eventType: 'start',
})
for (const record of records) {
  const message = {
    timestamp: record.timestamp,
    distance: record.distance,
    speed: record.speed,
    enhancedSpeed: record.speed,
  }
  if (record.heartRate != null) message.heartRate = record.heartRate
  if (record.cadence != null) message.cadence = record.cadence
  if (record.power != null) message.power = record.power
  if (record.altitude != null) {
    message.altitude = record.altitude
    message.enhancedAltitude = record.altitude
  }
  if (record.latlng) {
    message.positionLat = semicircles(record.latlng[0])
    message.positionLong = semicircles(record.latlng[1])
  }
  encoder.onMesg(Profile.MesgNum.RECORD, message)
}
encoder.onMesg(Profile.MesgNum.EVENT, { timestamp: end, event: 'timer', eventType: 'stopAll' })

const sharedSummary = {
  timestamp: end,
  startTime: destinationStart,
  totalElapsedTime: duration,
  totalTimerTime: timerTime,
  totalDistance,
  totalCalories: Math.round(calories),
  totalAscent: Math.round(totalAscent),
  avgSpeed: averageSpeed,
  maxSpeed: maximumSpeed,
  sport: 'training',
  subSport: 'generic',
}
if (averageHeartRate != null) sharedSummary.avgHeartRate = Math.round(averageHeartRate)
if (maximumHeartRate != null) sharedSummary.maxHeartRate = Math.round(maximumHeartRate)
if (averageCadence != null) sharedSummary.avgCadence = Math.round(averageCadence)
if (maximumCadence != null) sharedSummary.maxCadence = Math.round(maximumCadence)
if (averagePower != null) {
  sharedSummary.avgPower = Math.round(averagePower)
  sharedSummary.totalWork = Math.round(averagePower * timerTime)
}
if (maximumPower != null) sharedSummary.maxPower = Math.round(maximumPower)

encoder.onMesg(Profile.MesgNum.LAP, {
  ...sharedSummary,
  messageIndex: 0,
  event: 'lap',
  eventType: 'stop',
  lapTrigger: 'sessionEnd',
  intensity: 'active',
})
encoder.onMesg(Profile.MesgNum.SESSION, {
  ...sharedSummary,
  messageIndex: 0,
  firstLapIndex: 0,
  numLaps: 1,
  event: 'lap',
  eventType: 'stop',
  trigger: 'activityEnd',
})
encoder.onMesg(Profile.MesgNum.ACTIVITY, {
  timestamp: end,
  totalTimerTime: timerTime,
  numSessions: 1,
  type: 'manual',
  event: 'activity',
  eventType: 'stop',
})

const bytes = encoder.close()
const decoder = new Decoder(Stream.fromBuffer(bytes))
if (!decoder.isFIT()) throw new Error('generated output is not a FIT file')
if (!decoder.checkIntegrity()) throw new Error('generated FIT failed CRC integrity')
const decoded = decoder.read()
if (decoded.errors.length > 0) throw new Error(decoded.errors.join('\n'))

const decodedRecords = decoded.messages.recordMesgs ?? []
const decodedSession = decoded.messages.sessionMesgs?.[0]
const decodedActivity = decoded.messages.activityMesgs?.[0]
const decodedTimestamps = decodedRecords.flatMap(record =>
  record.timestamp instanceof Date ? [record.timestamp.getTime()] : [],
)
if (!decodedSession || !decodedActivity) throw new Error('generated FIT lacks session or activity')
if (decodedRecords.length !== duration + 1) throw new Error('decoded record count mismatch')
if (
  !decodedTimestamps.every(
    (timestamp, index) => index === 0 || timestamp > decodedTimestamps[index - 1],
  )
)
  throw new Error('decoded timestamps are not strictly increasing')
if (
  !(decodedSession.startTime instanceof Date) ||
  decodedSession.startTime.getTime() !== destinationStart.getTime()
)
  throw new Error('decoded start time mismatch')
if (decodedTimestamps.at(-1) !== end.getTime()) throw new Error('decoded end time mismatch')
if (decodedSession.totalElapsedTime !== duration) throw new Error('decoded elapsed time mismatch')
if (decodedSession.totalTimerTime !== timerTime) throw new Error('decoded timer time mismatch')
if (Math.abs(decodedSession.totalDistance - totalDistance) > 0.01)
  throw new Error('decoded distance mismatch')
if (decodedSession.totalCalories !== Math.round(calories))
  throw new Error('decoded calories mismatch')
if (heartRates.length > 0) {
  const decodedHeartRates = decodedRecords.filter(record => finite(record.heartRate))
  if (decodedHeartRates.length !== heartRates.length)
    throw new Error('decoded heart-rate count mismatch')
}

const sha256 = createHash('sha256').update(bytes).digest('hex')
const sourceFirstTime = sourceStream.time[0]
const sourceLastTime = sourceStream.time.at(-1)
const leadingBoundaryFillSeconds = Math.max(0, Math.ceil(sourceOffsetSeconds + sourceFirstTime))
const trailingBoundaryFillSeconds = Math.max(
  0,
  Math.ceil(duration - sourceOffsetSeconds - sourceLastTime),
)
const report = {
  sourceId,
  destinationId,
  outputPath,
  backupPath,
  productName,
  start: destinationStart,
  end,
  elapsedTime: duration,
  timerTime,
  distance: totalDistance,
  calories: Math.round(calories),
  overlapRatio,
  destinationOverlapRatio,
  sourceOffsetSeconds,
  leadingBoundaryFillSeconds,
  trailingBoundaryFillSeconds,
  records: decodedRecords.length,
  heartRate: {
    emitted: heartRates.length,
    measured: measuredHeartRates.length,
    boundaryFilled: heartRates.length - measuredHeartRates.length,
    average: mean(measuredHeartRates),
    minimum: minimum(measuredHeartRates),
    maximum: maximum(measuredHeartRates),
  },
  cadence: {
    emitted: cadences.length,
    measured: measuredCadences.length,
    boundaryFilled: cadences.length - measuredCadences.length,
    average: mean(measuredCadences),
    maximum: maximum(measuredCadences),
  },
  power: {
    emitted: powers.length,
    measured: measuredPowers.length,
    boundaryFilled: powers.length - measuredPowers.length,
    average: mean(measuredPowers),
    maximum: maximum(measuredPowers),
  },
  altitude: {
    emitted: altitudes.length,
    measured: measuredAltitudes.length,
    boundaryFilled: altitudes.length - measuredAltitudes.length,
    minimum: minimum(measuredAltitudes),
    maximum: maximum(measuredAltitudes),
  },
  positionCount: records.filter(record => record.latlng != null).length,
  integrity: true,
  decodeErrors: decoded.errors,
  sha256,
}

await Promise.all([
  fs.mkdir(path.dirname(outputPath), { recursive: true }),
  fs.mkdir(path.dirname(backupPath), { recursive: true }),
])
await Promise.all([
  fs.writeFile(outputPath, Buffer.from(bytes)),
  fs.writeFile(
    backupPath,
    `${JSON.stringify(
      {
        createdAt: new Date().toISOString(),
        report,
        source: {
          activity: source,
          detail: cache.activityDetails?.[sourceId] ?? null,
          stream: sourceStream,
        },
        destination: {
          activity: destination,
          detail: cache.activityDetails?.[destinationId] ?? null,
          stream: destinationStream ?? null,
        },
      },
      null,
      2,
    )}\n`,
  ),
])

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
