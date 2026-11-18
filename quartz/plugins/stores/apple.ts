import type { AppleSwim } from './apple-swims'
import { parseAppleJsonSwims } from './apple-swim-json'
export * from './apple-swims'
import type { AppleSleepDay } from './apple-sleep'
import { isRecord, readNumber, readString } from '../../util/type-guards'

export interface AppleDaily {
  date: string
  burnKcal: number | null
  activeKcal: number | null
  intakeKcal: number | null
  weightKg: number | null
  vo2max: number | null
}

export interface AppleHeartRateSample {
  time: string
  bpm: number
}

export interface AppleRunningDynamicsSample {
  time: string
  value: number
}

export interface AppleWorkout {
  id: string
  activity: string
  start: string
  end: string
  durationS: number
  elapsedTimeS?: number
  distanceM?: number
  activeEnergyKcal?: number
  averageHeartRateBpm?: number
  averageRunningPowerW?: number
  averageCadenceSpm?: number
  lapCount?: number
  source?: string
  device?: string
  gpxFile?: string
  heartRate: AppleHeartRateSample[]
  strideLengthM?: AppleRunningDynamicsSample[]
  groundContactTimeMs?: AppleRunningDynamicsSample[]
  verticalOscillationCm?: AppleRunningDynamicsSample[]
}

export interface AppleCache {
  version?: number
  lastSync: number
  days: Record<string, AppleDaily>
  sleep?: Record<string, AppleSleepDay>
  swims?: Record<string, AppleSwim>
  workouts?: Record<string, AppleWorkout>
}

export interface AppleJsonEntries {
  days: AppleDaily[]
  swims: AppleSwim[]
  workouts: AppleWorkout[]
}

export interface AppleRecord {
  date: string
  kind: 'active' | 'basal' | 'intake' | 'weight' | 'vo2max'
  value: number
  unit: string
  source: string
}

const LB_TO_KG = 0.45359237

const KIND_BY_TYPE: Record<string, AppleRecord['kind']> = {
  HKQuantityTypeIdentifierActiveEnergyBurned: 'active',
  HKQuantityTypeIdentifierBasalEnergyBurned: 'basal',
  HKQuantityTypeIdentifierDietaryEnergyConsumed: 'intake',
  HKQuantityTypeIdentifierBodyMass: 'weight',
  HKQuantityTypeIdentifierVO2Max: 'vo2max',
}

export function matchAppleRecord(line: string): AppleRecord | null {
  if (!line.includes('<Record')) return null
  const type = /type="(HKQuantityTypeIdentifier\w+)"/.exec(line)?.[1]
  const kind = type ? KIND_BY_TYPE[type] : undefined
  if (!kind) return null
  const date = /startDate="(\d{4}-\d{2}-\d{2})/.exec(line)?.[1]
  const value = /value="([\d.]+)"/.exec(line)?.[1]
  if (!date || value === undefined) return null
  const unit = /unit="([^"]+)"/.exec(line)?.[1] ?? ''
  const source = /sourceName="([^"]*)"/.exec(line)?.[1] ?? ''
  return { date, kind, value: Number(value), unit, source }
}

const toKg = (value: number, unit: string): number =>
  unit.toLowerCase().startsWith('lb') ? value * LB_TO_KG : value

interface SourceAgg {
  sum: number
  count: number
  last: number
  unit: string
}

// Apple exports the same metric from several sources (Watch + iPhone + Oura + Strava),
// so summing every record triple-counts. Pick ONE source per day+metric — prefer the
// Apple Watch, else the source with the most samples — and use only its records.
function pickSource(sources: Map<string, SourceAgg>): SourceAgg | null {
  let watch: SourceAgg | null = null
  let best: SourceAgg | null = null
  for (const [name, agg] of sources) {
    if (/watch/i.test(name) && (!watch || agg.count > watch.count)) watch = agg
    if (!best || agg.count > best.count) best = agg
  }
  return watch ?? best
}

export function aggregateAppleRecords(records: AppleRecord[]): AppleDaily[] {
  const byDay = new Map<string, Map<AppleRecord['kind'], Map<string, SourceAgg>>>()
  for (const r of records) {
    let kinds = byDay.get(r.date)
    if (!kinds) {
      kinds = new Map()
      byDay.set(r.date, kinds)
    }
    let sources = kinds.get(r.kind)
    if (!sources) {
      sources = new Map()
      kinds.set(r.kind, sources)
    }
    const agg = sources.get(r.source) ?? { sum: 0, count: 0, last: 0, unit: r.unit }
    agg.sum += r.value
    agg.count += 1
    agg.last = r.value
    agg.unit = r.unit
    sources.set(r.source, agg)
  }
  const out: AppleDaily[] = []
  for (const [date, kinds] of byDay) {
    const at = (k: AppleRecord['kind']): SourceAgg | null => {
      const sources = kinds.get(k)
      return sources ? pickSource(sources) : null
    }
    const active = at('active')
    const basal = at('basal')
    const intake = at('intake')
    const weight = at('weight')
    const vo2max = at('vo2max')
    const activeKcal = active ? Math.round(active.sum) : null
    const basalKcal = basal ? Math.round(basal.sum) : null
    const burnKcal =
      activeKcal != null || basalKcal != null ? (activeKcal ?? 0) + (basalKcal ?? 0) : null
    out.push({
      date,
      activeKcal,
      burnKcal,
      intakeKcal: intake ? Math.round(intake.sum) : null,
      weightKg: weight ? Math.round(toKg(weight.last, weight.unit) * 10) / 10 : null,
      vo2max: vo2max ? Math.round(vo2max.last * 10) / 10 : null,
    })
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}

function parseAppleJsonDays(raw: unknown): AppleDaily[] {
  const days = isRecord(raw) ? raw.days : undefined
  if (!Array.isArray(days)) return []
  const out: AppleDaily[] = []
  for (const d of days) {
    if (!isRecord(d)) continue
    const date = readString(d, 'date')?.slice(0, 10) ?? null
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    const active = readNumber(d, 'activeKcal') ?? null
    const basal = readNumber(d, 'basalKcal') ?? null
    const burn =
      readNumber(d, 'burnKcal') ??
      (active != null || basal != null ? (active ?? 0) + (basal ?? 0) : null)
    const lbs = readNumber(d, 'weightLbs') ?? null
    const intake = readNumber(d, 'intakeKcal') ?? null
    const weightKg = readNumber(d, 'weightKg') ?? null
    const vo2max = readNumber(d, 'vo2max') ?? null
    out.push({
      date,
      activeKcal: active != null ? Math.round(active) : null,
      burnKcal: burn != null ? Math.round(burn) : null,
      intakeKcal: intake != null ? Math.round(intake) : null,
      weightKg:
        weightKg != null
          ? Math.round(weightKg * 10) / 10
          : lbs != null
            ? Math.round(lbs * LB_TO_KG * 10) / 10
            : null,
      vo2max: vo2max != null ? Math.round(vo2max * 10) / 10 : null,
    })
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}

function parseIsoSecond(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const time = Date.parse(value)
  if (!Number.isFinite(time)) return null
  return new Date(time).toISOString().replace('.000Z', 'Z')
}

function parseRunningDynamicsSamples(
  raw: unknown,
  minimum: number,
  maximum: number,
): AppleRunningDynamicsSample[] {
  if (!Array.isArray(raw)) return []
  const samples: AppleRunningDynamicsSample[] = []
  for (const sample of raw) {
    if (!isRecord(sample)) continue
    const time = parseIsoSecond(sample.time)
    const value = readNumber(sample, 'value')
    if (!time || value == null || value < minimum || value > maximum) continue
    samples.push({ time, value })
  }
  return samples.sort((a, b) => a.time.localeCompare(b.time))
}

function parseAppleJsonWorkouts(raw: unknown): AppleWorkout[] {
  const workouts = isRecord(raw) ? raw.workouts : undefined
  if (!Array.isArray(workouts)) return []
  const out: AppleWorkout[] = []
  for (const workout of workouts) {
    if (!isRecord(workout)) continue
    const id = readString(workout, 'id')
    const activity = readString(workout, 'activity')
    const start = parseIsoSecond(workout.start)
    const end = parseIsoSecond(workout.end)
    const durationS = readNumber(workout, 'durationS')
    const elapsedTimeS = readNumber(workout, 'elapsedTimeS')
    const distanceM = readNumber(workout, 'distanceM')
    const activeEnergyKcal = readNumber(workout, 'activeEnergyKcal')
    const averageHeartRateBpm = readNumber(workout, 'averageHeartRateBpm')
    const averageRunningPowerW = readNumber(workout, 'averageRunningPowerW')
    const averageCadenceSpm = readNumber(workout, 'averageCadenceSpm')
    const lapCount = readNumber(workout, 'lapCount')
    const source = readString(workout, 'source')?.trim()
    const device = readString(workout, 'device')?.trim()
    const gpxFileValue = readString(workout, 'gpxFile')?.trim()
    const gpxFile =
      gpxFileValue && /^GPX\/[0-9A-F-]+\.gpx$/i.test(gpxFileValue) ? gpxFileValue : null
    const rawHeartRate = Array.isArray(workout.heartRate) ? workout.heartRate : []
    if (!id || !activity || !start || !end || durationS == null) continue
    const heartRate: AppleHeartRateSample[] = []
    for (const sample of rawHeartRate) {
      if (!isRecord(sample)) continue
      const time = parseIsoSecond(sample.time)
      const bpm = readNumber(sample, 'bpm')
      if (!time || bpm == null || bpm <= 0) continue
      heartRate.push({ time, bpm: Math.round(bpm) })
    }
    const parsed: AppleWorkout = {
      id,
      activity,
      start,
      end,
      durationS: Math.round(durationS),
      heartRate: heartRate.sort((a, b) => a.time.localeCompare(b.time)),
      strideLengthM: parseRunningDynamicsSamples(workout.strideLengthM, 0.2, 3),
      groundContactTimeMs: parseRunningDynamicsSamples(workout.groundContactTimeMs, 50, 1_000),
      verticalOscillationCm: parseRunningDynamicsSamples(workout.verticalOscillationCm, 1, 30),
    }
    if (elapsedTimeS != null && elapsedTimeS >= 0) parsed.elapsedTimeS = Math.round(elapsedTimeS)
    if (distanceM != null && distanceM >= 0) parsed.distanceM = Math.round(distanceM * 10) / 10
    if (activeEnergyKcal != null && activeEnergyKcal >= 0)
      parsed.activeEnergyKcal = Math.round(activeEnergyKcal * 10) / 10
    if (averageHeartRateBpm != null && averageHeartRateBpm > 0)
      parsed.averageHeartRateBpm = Math.round(averageHeartRateBpm)
    if (averageRunningPowerW != null && averageRunningPowerW > 0)
      parsed.averageRunningPowerW = Math.round(averageRunningPowerW)
    if (averageCadenceSpm != null && averageCadenceSpm > 0)
      parsed.averageCadenceSpm = Math.round(averageCadenceSpm)
    if (lapCount != null && lapCount > 0) parsed.lapCount = Math.round(lapCount)
    if (source) parsed.source = source
    if (device) parsed.device = device
    if (gpxFile) parsed.gpxFile = gpxFile
    out.push(parsed)
  }
  return out.sort((a, b) => a.start.localeCompare(b.start))
}

export function parseAppleJson(raw: unknown): AppleJsonEntries {
  return {
    days: parseAppleJsonDays(raw),
    swims: parseAppleJsonSwims(raw),
    workouts: parseAppleJsonWorkouts(raw),
  }
}

export function mergeAppleDay(prev: AppleDaily | undefined, next: AppleDaily): AppleDaily {
  if (!prev) return next
  return {
    date: next.date,
    burnKcal: next.burnKcal ?? prev.burnKcal,
    activeKcal: next.activeKcal ?? prev.activeKcal,
    intakeKcal: next.intakeKcal ?? prev.intakeKcal,
    weightKg: next.weightKg ?? prev.weightKg,
    vo2max: next.vo2max ?? prev.vo2max,
  }
}

export function latestAppleDate(days: Record<string, AppleDaily>): string | null {
  let latest: string | null = null
  for (const day of Object.values(days)) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(day.date)) continue
    if (!latest || day.date > latest) latest = day.date
  }
  return latest
}
