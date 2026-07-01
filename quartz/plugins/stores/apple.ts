import { isRecord, readNumber, readString } from '../../util/type-guards'

export interface AppleDaily {
  date: string
  burnKcal: number | null
  activeKcal: number | null
  intakeKcal: number | null
  weightKg: number | null
  vo2max: number | null
}

export interface AppleSwim {
  date: string
  totalM: number
  laps: number
  strokes: Record<string, number>
}

export interface AppleHeartRateSample {
  time: string
  bpm: number
}

export interface AppleWorkout {
  id: string
  activity: string
  start: string
  end: string
  durationS: number
  heartRate: AppleHeartRateSample[]
}

export interface AppleCache {
  version?: number
  lastSync: number
  days: Record<string, AppleDaily>
  swims?: Record<string, AppleSwim>
  workouts?: Record<string, AppleWorkout>
}

export interface AppleJsonEntries {
  days: AppleDaily[]
  swims: AppleSwim[]
  workouts: AppleWorkout[]
}

export const SWIM_STROKES = [
  'freestyle',
  'breaststroke',
  'backstroke',
  'butterfly',
  'mixed',
  'kickboard',
] as const
export type SwimStroke = (typeof SWIM_STROKES)[number]

export const STROKE_LABEL: Record<SwimStroke, string> = {
  freestyle: 'freestyle',
  breaststroke: 'breast',
  backstroke: 'back',
  butterfly: 'fly',
  mixed: 'mixed',
  kickboard: 'kick',
}

const STROKE_BY_VALUE: Record<string, SwimStroke> = {
  '1': 'mixed',
  '2': 'freestyle',
  '3': 'backstroke',
  '4': 'breaststroke',
  '5': 'butterfly',
  '6': 'kickboard',
}

const SWIM_DIST_UNIT: Record<string, number> = { m: 1, km: 1000, mi: 1609.344, yd: 0.9144 }

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

export function matchSwimDistance(line: string): { start: string; meters: number } | null {
  if (!line.includes('HKQuantityTypeIdentifierDistanceSwimming')) return null
  const start = /startDate="([^"]+)"/.exec(line)?.[1]
  const value = /\svalue="([\d.]+)"/.exec(line)?.[1]
  if (!start || value === undefined) return null
  const unit = /\sunit="([^"]+)"/.exec(line)?.[1] ?? 'm'
  return { start, meters: Number(value) * (SWIM_DIST_UNIT[unit.toLowerCase()] ?? 1) }
}

export function matchSwimStrokeOpen(line: string): string | null {
  if (!line.includes('HKQuantityTypeIdentifierSwimmingStrokeCount')) return null
  if (line.trimEnd().endsWith('/>')) return null
  return /startDate="([^"]+)"/.exec(line)?.[1] ?? null
}

export function matchStrokeStyle(line: string): SwimStroke | null {
  const v = /key="HKSwimmingStrokeStyle" value="(\d+)"/.exec(line)?.[1]
  return v ? (STROKE_BY_VALUE[v] ?? null) : null
}

function poolLengthByDate(distByStart: Map<string, number>): Map<string, number> {
  const groups = new Map<string, number[]>()
  for (const [start, m] of distByStart) {
    const date = start.slice(0, 10)
    const arr = groups.get(date)
    if (arr) arr.push(m)
    else groups.set(date, [m])
  }
  const out = new Map<string, number>()
  for (const [date, arr] of groups) {
    arr.sort((a, b) => a - b)
    out.set(date, arr[arr.length >> 1])
  }
  return out
}

export function aggregateSwimLaps(
  strokeLaps: { start: string; stroke: SwimStroke }[],
  distByStart: Map<string, number>,
): AppleSwim[] {
  const poolByDate = poolLengthByDate(distByStart)
  const byDate = new Map<string, AppleSwim>()
  for (const lap of strokeLaps) {
    const date = lap.start.slice(0, 10)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    const meters = distByStart.get(lap.start) ?? poolByDate.get(date) ?? 25
    let sw = byDate.get(date)
    if (!sw) {
      sw = { date, totalM: 0, laps: 0, strokes: {} }
      byDate.set(date, sw)
    }
    sw.strokes[lap.stroke] = (sw.strokes[lap.stroke] ?? 0) + meters
    sw.totalM += meters
    sw.laps += 1
  }
  return [...byDate.values()]
    .map(sw => {
      const strokes: Record<string, number> = {}
      for (const k of Object.keys(sw.strokes)) strokes[k] = Math.round(sw.strokes[k])
      return { date: sw.date, laps: sw.laps, totalM: Math.round(sw.totalM), strokes }
    })
    .sort((a, b) => a.date.localeCompare(b.date))
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
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

function parseAppleJsonSwims(raw: unknown): AppleSwim[] {
  const swims = isRecord(raw) ? raw.swims : undefined
  if (!Array.isArray(swims)) return []
  const out: AppleSwim[] = []
  for (const swim of swims) {
    if (!isRecord(swim)) continue
    const date = readString(swim, 'date')?.slice(0, 10) ?? null
    const totalM = readNumber(swim, 'totalM')
    const laps = readNumber(swim, 'laps')
    const rawStrokes = isRecord(swim.strokes) ? swim.strokes : null
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date) || totalM == null || laps == null || !rawStrokes)
      continue
    const strokes: Record<string, number> = {}
    for (const [stroke, meters] of Object.entries(rawStrokes)) {
      const rounded = num(meters)
      if (rounded != null && rounded > 0) strokes[stroke] = Math.round(rounded)
    }
    if (Object.keys(strokes).length === 0) continue
    out.push({ date, totalM: Math.round(totalM), laps: Math.round(laps), strokes })
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}

function parseIsoSecond(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const time = Date.parse(value)
  if (!Number.isFinite(time)) return null
  return new Date(time).toISOString().replace('.000Z', 'Z')
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
    if (heartRate.length === 0) continue
    out.push({
      id,
      activity,
      start,
      end,
      durationS: Math.round(durationS),
      heartRate: heartRate.sort((a, b) => a.time.localeCompare(b.time)),
    })
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
