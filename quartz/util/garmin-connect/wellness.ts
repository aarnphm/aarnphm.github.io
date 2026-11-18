import type {
  GarminLactateThresholdValue,
  GarminRunningLactateThreshold,
  GarminVo2Day,
} from '../../plugins/stores/garmin'
import { isRecord, readString } from '../type-guards'
import { finite, positive } from './scalars'

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
