import type {
  GarminOvernightSleep,
  GarminSleepNap,
  GarminSleepScoreFactorName,
  GarminSleepStageInterval,
  GarminSleepSummary,
} from '../../plugins/stores/garmin-sleep'
import { isRecord, readString, type UnknownRecord } from '../type-guards'
import { normalizeDate, positive } from './scalars'

const stageCodes: Readonly<Record<number, GarminSleepStageInterval['stage'] | undefined>> = {
  0: 'deep',
  1: 'light',
  2: 'rem',
  3: 'awake',
}

const factorNames: GarminSleepScoreFactorName[] = [
  'totalDuration',
  'stress',
  'awakeCount',
  'restlessness',
  'lightPercentage',
  'deepPercentage',
  'remPercentage',
]

function sleepMetric(value: unknown, min: number, max = Infinity): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max
    ? value
    : null
}

function sleepTimestamp(value: unknown): number | null {
  const timestamp = sleepMetric(value, 1)
  return timestamp != null && Number.isFinite(new Date(timestamp).getTime()) ? timestamp : null
}

function garminGmtTimestamp(value: unknown): number | null {
  // Garmin Connect returns these GMT strings without a zone suffix.
  if (typeof value === 'string') {
    const normalized = normalizeDate(value)
    return normalized ? sleepTimestamp(Date.parse(normalized)) : null
  }
  return sleepTimestamp(value)
}

function validUtcOffsetMinutes(value: unknown): number | undefined {
  const minutes = sleepMetric(value, -14 * 60, 14 * 60)
  return minutes != null && Number.isInteger(minutes) ? minutes : undefined
}

function sleepBodyBatteryAt(raw: unknown, timestamp: number | null): number | null {
  if (!Array.isArray(raw) || timestamp == null) return null
  // Garmin samples these values every three minutes, including the epoch before sleep starts.
  let closestDistance = 5 * 60 * 1000
  let closest: number | null = null
  for (const sample of raw) {
    if (!isRecord(sample)) continue
    const time = sleepTimestamp(sample.startGMT)
    const value = sleepMetric(sample.value, 0, 100)
    if (time == null || value == null) continue
    const distance = Math.abs(time - timestamp)
    if (distance <= closestDistance) {
      closestDistance = distance
      closest = value
    }
  }
  return closest
}

function overnightSamples(
  raw: unknown,
  start: number,
  end: number,
  timestampKey: string,
  valueKey: string,
): [number, number | null][] {
  if (!Array.isArray(raw)) return []
  const samples = new Map<number, number | null>()
  for (const sample of raw) {
    if (!isRecord(sample)) continue
    const timestamp = sleepTimestamp(sample[timestampKey])
    if (timestamp == null || timestamp < start || timestamp > end) continue
    const value = positive(sleepMetric(sample[valueKey], 0))
    if (samples.get(timestamp) == null || value != null) samples.set(timestamp, value)
  }
  const readings = [...samples].sort(([left], [right]) => left - right)
  return readings.some(([, value]) => value != null) ? readings : []
}

function overnightStages(raw: unknown, start: number, end: number): GarminSleepStageInterval[] {
  if (!Array.isArray(raw)) return []
  const intervals: GarminSleepStageInterval[] = []
  for (const row of raw) {
    if (!isRecord(row) || typeof row.activityLevel !== 'number') continue
    const stage = stageCodes[row.activityLevel]
    const rowStart = garminGmtTimestamp(row.startGMT)
    const rowEnd = garminGmtTimestamp(row.endGMT)
    if (!stage || rowStart == null || rowEnd == null) continue
    const startTime = Math.max(rowStart, start)
    const endTime = Math.min(rowEnd, end)
    if (endTime > startTime) intervals.push({ startTime, endTime, stage })
  }
  const stages: GarminSleepStageInterval[] = []
  for (const interval of intervals.sort((left, right) => left.startTime - right.startTime)) {
    const previous = stages.at(-1)
    if (!previous || interval.startTime >= previous.endTime) stages.push(interval)
  }
  return stages
}

function garminNaps(raw: unknown, day: string): GarminSleepNap[] {
  if (!Array.isArray(raw)) return []
  const naps = new Map<string, GarminSleepNap>()
  for (const row of raw) {
    if (!isRecord(row) || row.calendarDate !== day) continue
    const startTime = garminGmtTimestamp(row.napStartTimestampGMT)
    const endTime = garminGmtTimestamp(row.napEndTimestampGMT)
    const durationS = positive(sleepMetric(row.napTimeSec, 0))
    if (startTime == null || endTime == null || endTime <= startTime || durationS == null) continue
    const id = `garmin-nap-${startTime}-${endTime}`
    const startUtcOffsetMinutes = validUtcOffsetMinutes(row.napStartTimeOffset)
    const endUtcOffsetMinutes = validUtcOffsetMinutes(row.napEndTimeOffset)
    naps.set(id, {
      id,
      startTime,
      endTime,
      ...(startUtcOffsetMinutes == null ? {} : { startUtcOffsetMinutes }),
      ...(endUtcOffsetMinutes == null ? {} : { endUtcOffsetMinutes }),
      durationS,
      providerSource: sleepMetric(row.napSource, 0),
      feedback: readString(row, 'napFeedback') ?? null,
    })
  }
  return [...naps.values()].sort((left, right) => left.startTime - right.startTime)
}

function garminOvernightSleep(
  raw: UnknownRecord,
  dto: UnknownRecord,
  start: number,
  end: number,
): GarminOvernightSleep | undefined {
  const totalSleepS = positive(sleepMetric(dto.sleepTimeSeconds, 0))
  if (dto.sleepFromDevice !== true || totalSleepS == null) return undefined
  const scores = isRecord(dto.sleepScores) ? dto.sleepScores : {}
  const overall = isRecord(scores.overall) ? scores.overall : {}
  const sleep: GarminOvernightSleep = {
    totalSleepS,
    deepS: sleepMetric(dto.deepSleepSeconds, 0),
    lightS: sleepMetric(dto.lightSleepSeconds, 0),
    remS: sleepMetric(dto.remSleepSeconds, 0),
    awakeS: sleepMetric(dto.awakeSleepSeconds, 0),
    unmeasurableS: sleepMetric(dto.unmeasurableSleepSeconds, 0),
    awakeCount: sleepMetric(dto.awakeCount, 0),
    sleepScore: sleepMetric(overall.value, 0, 100),
    sleepScoreQualifier: readString(overall, 'qualifierKey') ?? null,
    averageHeartRateBpm: positive(sleepMetric(dto.avgHeartRate, 0)),
    restingHeartRateBpm: positive(sleepMetric(raw.restingHeartRate, 0)),
    averageHrvMs: positive(sleepMetric(raw.avgOvernightHrv, 0)),
    skinTempDeviationC: sleepMetric(raw.avgSkinTempDeviationC, -Infinity),
  }
  const scoreFactors: NonNullable<GarminOvernightSleep['scoreFactors']> = {}
  for (const name of factorNames) {
    const factor = scores[name]
    if (!isRecord(factor)) continue
    const qualifier = readString(factor, 'qualifierKey') ?? null
    const value = name.endsWith('Percentage') ? sleepMetric(factor.value, 0, 100) : null
    if (qualifier != null || value != null) scoreFactors[name] = { qualifier, value }
  }
  if (Object.keys(scoreFactors).length) sleep.scoreFactors = scoreFactors
  const stages = overnightStages(raw.sleepLevels, start, end)
  if (stages.length) sleep.stages = stages
  const heartRate = overnightSamples(raw.sleepHeartRate, start, end, 'startGMT', 'value')
  if (heartRate.length) sleep.heartRate = heartRate.map(([timestamp, bpm]) => ({ timestamp, bpm }))
  const hrv = overnightSamples(raw.hrvData, start, end, 'startGMT', 'value')
  if (hrv.length) sleep.hrv = hrv.map(([timestamp, ms]) => ({ timestamp, ms }))
  return sleep
}

function sleepUtcOffsetMinutes(dto: UnknownRecord, start: number, end: number): number | null {
  const localStart = sleepTimestamp(dto.sleepStartTimestampLocal)
  if (localStart == null) return null
  const offset = validUtcOffsetMinutes((localStart - start) / 60_000)
  if (offset == null) return null
  const localEnd = sleepTimestamp(dto.sleepEndTimestampLocal)
  // A single offset cannot describe a night crossing a clock change.
  if (localEnd != null && (localEnd - end) / 60_000 !== offset) return null
  return offset
}

export function garminConnectSleep(raw: unknown, day: string): GarminSleepSummary | null {
  if (!isRecord(raw) || !isRecord(raw.dailySleepDTO)) return null
  const original = raw.dailySleepDTO
  if (original.calendarDate !== day) throw new Error(`Garmin sleep response does not match ${day}`)
  const naps = garminNaps(original.dailyNapDTOS, day)
  const manual =
    original.sleepFromDevice === false ||
    readString(original, 'sleepWindowConfirmationType')?.toLowerCase() === 'manually_confirmed'
  if (manual && naps.length === 0) return null
  const dto = manual ? {} : original
  const overnight = manual ? {} : raw
  const start = sleepTimestamp(dto.sleepStartTimestampGMT)
  const end = sleepTimestamp(dto.sleepEndTimestampGMT)
  const validWindow = start != null && end != null && end > start
  const summary: GarminSleepSummary = {
    source: 'garmin',
    date: day,
    startTime: validWindow ? new Date(start).toISOString() : null,
    endTime: validWindow ? new Date(end).toISOString() : null,
    averageBreathsPerMinute: positive(sleepMetric(dto.averageRespirationValue, 0)),
    lowestBreathsPerMinute: positive(sleepMetric(dto.lowestRespirationValue, 0)),
    highestBreathsPerMinute: positive(sleepMetric(dto.highestRespirationValue, 0)),
    averageSpO2: positive(sleepMetric(dto.averageSpO2Value, 0, 100)),
    lowestSpO2: positive(sleepMetric(dto.lowestSpO2Value, 0, 100)),
    bodyBatteryStart: validWindow ? sleepBodyBatteryAt(overnight.sleepBodyBattery, start) : null,
    bodyBatteryEnd: validWindow ? sleepBodyBatteryAt(overnight.sleepBodyBattery, end) : null,
    bodyBatteryChange: sleepMetric(overnight.bodyBatteryChange, -100, 100),
    averageStress: sleepMetric(dto.avgSleepStress, 0, 100),
    restlessMoments: sleepMetric(overnight.restlessMomentsCount, 0),
  }
  const hasMeasurements = Object.values(summary).some(value => typeof value === 'number')
  if (validWindow) {
    const respiration = overnightSamples(
      overnight.wellnessEpochRespirationDataDTOList,
      start,
      end,
      'startTimeGMT',
      'respirationValue',
    )
    if (respiration.length)
      summary.respiration = respiration.map(([timestamp, breathsPerMinute]) => ({
        timestamp,
        breathsPerMinute,
      }))
    const utcOffsetMinutes = sleepUtcOffsetMinutes(dto, start, end)
    if (utcOffsetMinutes != null) summary.utcOffsetMinutes = utcOffsetMinutes
    const sleep = garminOvernightSleep(overnight, dto, start, end)
    if (sleep) summary.sleep = sleep
  }
  if (naps.length) summary.naps = naps
  return summary.sleep || summary.naps || summary.respiration || hasMeasurements ? summary : null
}
