import {
  Decoder,
  Encoder,
  Profile,
  Stream,
  Utils,
  type DeviceInfoMesg,
  type FileIdMesg,
  type MonitoringHrDataMesg,
  type MonitoringInfoMesg,
  type MonitoringMesg,
  type SleepLevelMesg,
} from '@garmin/fitsdk'

export type GarminSleepLevel = 'unmeasurable' | 'awake' | 'light' | 'deep' | 'rem'

export interface GarminSleepStage {
  startTime: Date
  level: GarminSleepLevel
}

export interface GarminWellnessHeartRateSample {
  time: Date
  heartRateBpm: number
}

export interface GarminWellnessFitInput {
  serialNumber: number
  productName: string
  sleepStart: Date
  sleepEnd: Date
  localOffsetMinutes: number
  stages: readonly GarminSleepStage[]
  heartRate?: readonly GarminWellnessHeartRateSample[]
  restingHeartRate?: number
}

export interface GarminWellnessFitMessageCounts {
  fileIds: number
  deviceInfos: number
  monitoringInfos: number
  sleepLevels: number
  monitorings: number
  monitoringHrData: number
}

export interface GarminWellnessFitValidation {
  valid: boolean
  isFit: boolean
  integrity: boolean
  errors: readonly string[]
  counts: GarminWellnessFitMessageCounts
}

export interface GarminWellnessFitEncoding {
  bytes: Uint8Array
  validation: GarminWellnessFitValidation
}

const MIN_HEART_RATE_BPM = 20
const MAX_HEART_RATE_BPM = 250
const MAX_LOCAL_OFFSET_MINUTES = 14 * 60

function requireTime(value: Date, label: string): number {
  const ms = value.valueOf()
  if (!Number.isFinite(ms)) throw new Error(`${label} must be a valid date`)
  return ms
}

function validateInput(input: GarminWellnessFitInput): void {
  if (!Number.isSafeInteger(input.serialNumber) || input.serialNumber <= 0)
    throw new Error('serialNumber must be a positive integer')
  if (!input.productName.trim()) throw new Error('productName must not be empty')
  const start = requireTime(input.sleepStart, 'sleepStart')
  const end = requireTime(input.sleepEnd, 'sleepEnd')
  if (end <= start) throw new Error('sleepEnd must be after sleepStart')
  if (
    !Number.isInteger(input.localOffsetMinutes) ||
    Math.abs(input.localOffsetMinutes) > MAX_LOCAL_OFFSET_MINUTES
  )
    throw new Error('localOffsetMinutes must be a whole-minute UTC offset')
  if (input.stages.length === 0) throw new Error('stages must not be empty')
  let previous = -Infinity
  for (const stage of input.stages) {
    const ms = requireTime(stage.startTime, 'stage startTime')
    if (ms < start || ms > end) throw new Error('stage startTime must fall inside the sleep window')
    if (ms <= previous) throw new Error('stages must be strictly increasing')
    previous = ms
  }
  for (const sample of input.heartRate ?? []) {
    const ms = requireTime(sample.time, 'heart rate time')
    if (ms < start || ms > end)
      throw new Error('heart rate samples must fall inside the sleep window')
    if (
      !Number.isFinite(sample.heartRateBpm) ||
      sample.heartRateBpm < MIN_HEART_RATE_BPM ||
      sample.heartRateBpm > MAX_HEART_RATE_BPM
    )
      throw new Error(`heart rate ${sample.heartRateBpm} is out of range`)
  }
  if (
    input.restingHeartRate != null &&
    (!Number.isFinite(input.restingHeartRate) ||
      input.restingHeartRate < MIN_HEART_RATE_BPM ||
      input.restingHeartRate > MAX_HEART_RATE_BPM)
  )
    throw new Error(`resting heart rate ${input.restingHeartRate} is out of range`)
}

function localTimestamp(time: Date, localOffsetMinutes: number): number {
  return Utils.convertDateToDateTime(new Date(time.valueOf() + localOffsetMinutes * 60_000))
}

export function collapseSleepStages(
  stages: readonly GarminSleepStage[],
): readonly GarminSleepStage[] {
  const out: GarminSleepStage[] = []
  for (const stage of stages) {
    if (out[out.length - 1]?.level === stage.level) continue
    out.push(stage)
  }
  return out
}

function writeHeaderMessages(encoder: Encoder, input: GarminWellnessFitInput): void {
  const fileId: FileIdMesg = {
    type: 'monitoringB',
    manufacturer: 'development',
    product: 0,
    serialNumber: input.serialNumber,
    timeCreated: input.sleepEnd,
  }
  const deviceInfo: DeviceInfoMesg = {
    timestamp: input.sleepStart,
    deviceIndex: 'creator',
    manufacturer: 'development',
    product: 0,
    serialNumber: input.serialNumber,
    productName: input.productName,
    softwareVersion: 1,
  }
  const monitoringInfo: MonitoringInfoMesg = {
    timestamp: input.sleepStart,
    localTimestamp: localTimestamp(input.sleepStart, input.localOffsetMinutes),
    activityType: ['generic'],
  }
  encoder.onMesg(Profile.MesgNum.FILE_ID, fileId)
  encoder.onMesg(Profile.MesgNum.DEVICE_INFO, deviceInfo)
  encoder.onMesg(Profile.MesgNum.MONITORING_INFO, monitoringInfo)
}

function writeSleepLevels(encoder: Encoder, input: GarminWellnessFitInput): void {
  const stages = collapseSleepStages(input.stages)
  for (const stage of stages) {
    const mesg: SleepLevelMesg = { timestamp: stage.startTime, sleepLevel: stage.level }
    encoder.onMesg(Profile.MesgNum.SLEEP_LEVEL, mesg)
  }
  if (stages[stages.length - 1].startTime.valueOf() >= input.sleepEnd.valueOf()) return
  const wake: SleepLevelMesg = { timestamp: input.sleepEnd, sleepLevel: 'awake' }
  encoder.onMesg(Profile.MesgNum.SLEEP_LEVEL, wake)
}

function writeHeartRate(encoder: Encoder, input: GarminWellnessFitInput): void {
  for (const sample of input.heartRate ?? []) {
    const mesg: MonitoringMesg = {
      timestamp: sample.time,
      localTimestamp: localTimestamp(sample.time, input.localOffsetMinutes),
      activityType: 'generic',
      heartRate: Math.round(sample.heartRateBpm),
    }
    encoder.onMesg(Profile.MesgNum.MONITORING, mesg)
  }
  if (input.restingHeartRate == null) return
  const resting = Math.round(input.restingHeartRate)
  const mesg: MonitoringHrDataMesg = {
    timestamp: input.sleepEnd,
    restingHeartRate: resting,
    currentDayRestingHeartRate: resting,
  }
  encoder.onMesg(Profile.MesgNum.MONITORING_HR_DATA, mesg)
}

function emptyCounts(): GarminWellnessFitMessageCounts {
  return {
    fileIds: 0,
    deviceInfos: 0,
    monitoringInfos: 0,
    sleepLevels: 0,
    monitorings: 0,
    monitoringHrData: 0,
  }
}

export function validateGarminWellnessFit(bytes: Uint8Array): GarminWellnessFitValidation {
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
        monitoringInfos: decoded.messages.monitoringInfoMesgs?.length ?? 0,
        sleepLevels: decoded.messages.sleepLevelMesgs?.length ?? 0,
        monitorings: decoded.messages.monitoringMesgs?.length ?? 0,
        monitoringHrData: decoded.messages.monitoringHrDataMesgs?.length ?? 0,
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error))
    }
  }
  const complete =
    counts.fileIds === 1 &&
    counts.deviceInfos >= 1 &&
    counts.monitoringInfos === 1 &&
    counts.sleepLevels >= 2
  if (isFit && integrity && !complete) errors.push('FIT wellness messages are incomplete')
  return {
    valid: isFit && integrity && complete && errors.length === 0,
    isFit,
    integrity,
    errors,
    counts,
  }
}

export function encodeGarminWellnessFit(input: GarminWellnessFitInput): GarminWellnessFitEncoding {
  validateInput(input)
  const encoder = new Encoder()
  writeHeaderMessages(encoder, input)
  writeSleepLevels(encoder, input)
  writeHeartRate(encoder, input)
  const bytes = encoder.close()
  return { bytes, validation: validateGarminWellnessFit(bytes) }
}
