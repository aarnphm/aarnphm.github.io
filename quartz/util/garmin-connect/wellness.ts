import type {
  GarminLactateThresholdValue,
  GarminRunningLactateThreshold,
  GarminSleepRespirationSample,
  GarminSleepSummary,
  GarminVo2Day,
} from '../../plugins/stores/garmin'
import { isRecord, readString, type UnknownRecord } from '../type-guards'
import { finite, positive } from './scalars'

function sleepMetric(value: unknown, min: number, max = Infinity): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max
    ? value
    : null
}

function sleepTimestamp(value: unknown): number | null {
  const timestamp = sleepMetric(value, 1)
  return timestamp != null && Number.isFinite(new Date(timestamp).getTime()) ? timestamp : null
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

function sleepRespiration(
  raw: unknown,
  start: number,
  end: number,
): GarminSleepRespirationSample[] {
  if (!Array.isArray(raw)) return []
  const samples = new Map<number, GarminSleepRespirationSample>()
  for (const sample of raw) {
    if (!isRecord(sample)) continue
    const timestamp = sleepTimestamp(sample.startTimeGMT)
    if (timestamp == null || timestamp < start || timestamp > end) continue
    const breathsPerMinute = positive(sleepMetric(sample.respirationValue, 0))
    if (samples.get(timestamp)?.breathsPerMinute == null || breathsPerMinute != null)
      samples.set(timestamp, { timestamp, breathsPerMinute })
  }
  return [...samples.values()].sort((left, right) => left.timestamp - right.timestamp)
}

function sleepUtcOffsetMinutes(dto: UnknownRecord, start: number, end: number): number | null {
  const localStart = sleepTimestamp(dto.sleepStartTimestampLocal)
  if (localStart == null) return null
  const offset = (localStart - start) / 60_000
  if (!Number.isInteger(offset) || offset < -14 * 60 || offset > 14 * 60) return null
  const localEnd = sleepTimestamp(dto.sleepEndTimestampLocal)
  // A single offset cannot describe a night crossing a clock change.
  if (localEnd != null && (localEnd - end) / 60_000 !== offset) return null
  return offset
}

export function garminConnectSleep(raw: unknown, day: string): GarminSleepSummary | null {
  if (!isRecord(raw) || !isRecord(raw.dailySleepDTO)) return null
  const dto = raw.dailySleepDTO
  if (dto.calendarDate !== day) throw new Error(`Garmin sleep response does not match ${day}`)
  if (dto.sleepFromDevice === false) return null
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
    bodyBatteryStart: validWindow ? sleepBodyBatteryAt(raw.sleepBodyBattery, start) : null,
    bodyBatteryEnd: validWindow ? sleepBodyBatteryAt(raw.sleepBodyBattery, end) : null,
    bodyBatteryChange: sleepMetric(raw.bodyBatteryChange, -100, 100),
    averageStress: sleepMetric(dto.avgSleepStress, 0, 100),
    restlessMoments: sleepMetric(raw.restlessMomentsCount, 0),
  }
  const hasMeasurements = Object.values(summary).some(value => typeof value === 'number')
  if (validWindow) {
    const respiration = sleepRespiration(raw.wellnessEpochRespirationDataDTOList, start, end)
    if (respiration.some(sample => sample.breathsPerMinute != null))
      summary.respiration = respiration
    const utcOffsetMinutes = sleepUtcOffsetMinutes(dto, start, end)
    if (utcOffsetMinutes != null) summary.utcOffsetMinutes = utcOffsetMinutes
  }
  return summary.respiration || hasMeasurements ? summary : null
}

export function garminConnectRunningLactateThreshold(
  raw: unknown,
): GarminRunningLactateThreshold | null {
  if (!Array.isArray(raw)) throw new Error('Invalid Garmin lactate threshold response')
  const result: GarminRunningLactateThreshold = { speedMps: null, heartRateBpm: null }
  const records = raw
    .filter(isRecord)
    .sort((left, right) =>
      (readString(left, 'calendarDate') ?? '').localeCompare(
        readString(right, 'calendarDate') ?? '',
      ),
    )
  for (const item of records) {
    const timestamp = readString(item, 'calendarDate')
    const date = timestamp?.slice(0, 10)
    if (
      !date ||
      !/^\d{4}-\d{2}-\d{2}$/.test(date) ||
      !Number.isFinite(Date.parse(date)) ||
      new Date(date).toISOString().slice(0, 10) !== date
    )
      continue
    // This endpoint stores speed in decametres/s; 0.375 corresponds to 4:27/km.
    const speed = positive(finite(item.speed))
    const heartRate = positive(finite(item.heartRate)) ?? positive(finite(item.hearRate))
    if (speed != null && (!result.speedMps || date >= result.speedMps.date))
      result.speedMps = { value: speed * 10, date }
    if (heartRate != null && (!result.heartRateBpm || date >= result.heartRateBpm.date))
      result.heartRateBpm = { value: heartRate, date }
  }
  return result.speedMps || result.heartRateBpm ? result : null
}

export function garminConnectLactateThresholdHistory(
  raw: unknown,
  metric: 'speedMps' | 'heartRateBpm',
): GarminLactateThresholdValue[] {
  if (!Array.isArray(raw)) throw new Error('Invalid Garmin lactate threshold history response')
  const days = new Map<string, GarminLactateThresholdValue>()
  for (const item of raw) {
    if (!isRecord(item) || item.series !== 'running') continue
    const date = readString(item, 'updatedDate') ?? readString(item, 'from')
    const value = positive(finite(item.value))
    if (
      !date ||
      !/^\d{4}-\d{2}-\d{2}$/.test(date) ||
      !Number.isFinite(Date.parse(date)) ||
      new Date(date).toISOString().slice(0, 10) !== date ||
      value == null
    )
      continue
    days.set(date, { date, value: metric === 'speedMps' ? value * 10 : value })
  }
  return [...days.values()].sort((left, right) => left.date.localeCompare(right.date))
}

function vo2Of(value: unknown): number | null {
  if (!isRecord(value)) return null
  return finite(value.vo2MaxPreciseValue) ?? finite(value.vo2MaxValue)
}

export function garminConnectVo2(raw: unknown): GarminVo2Day[] {
  if (!Array.isArray(raw)) return []
  const out: GarminVo2Day[] = []
  for (const item of raw) {
    if (!isRecord(item)) continue
    const generic = isRecord(item.generic) ? item.generic : null
    const cycling = isRecord(item.cycling) ? item.cycling : null
    const date =
      readString(item, 'calendarDate') ??
      (generic ? readString(generic, 'calendarDate') : null) ??
      (cycling ? readString(cycling, 'calendarDate') : null)
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    const g = vo2Of(generic)
    const c = vo2Of(cycling)
    if (g == null && c == null) continue
    out.push({ date, generic: g, cycling: c })
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}
