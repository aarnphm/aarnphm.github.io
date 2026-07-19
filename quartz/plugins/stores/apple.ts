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
  id: string | null
  date: string
  start: string | null
  end: string | null
  totalM: number
  laps: number
  activeTimeS: number | null
  strokeCount: number | null
  strokeTimeS: number | null
  strokes: Record<string, number>
  intervals?: AppleSwimInterval[]
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

export interface AppleSwimInterval {
  start: string
  end: string
  distanceM: number
  startElapsedS?: number | null
  endElapsedS?: number | null
  durationS?: number | null
  strokeCount: number | null
  strokeTimeS: number | null
  stroke: SwimStroke | null
}

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
const SWIM_DURATION_UNIT: Record<string, number> = { s: 1, sec: 1, min: 60, h: 3600, hr: 3600 }

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

export interface AppleSwimDistanceRecord {
  start: string
  end: string
  meters: number
}

export function matchSwimDistance(line: string): AppleSwimDistanceRecord | null {
  if (!line.includes('HKQuantityTypeIdentifierDistanceSwimming')) return null
  const start = /startDate="([^"]+)"/.exec(line)?.[1]
  const end = /endDate="([^"]+)"/.exec(line)?.[1] ?? start
  const value = /\svalue="([\d.]+)"/.exec(line)?.[1]
  const unit = /\sunit="([^"]+)"/.exec(line)?.[1]?.toLowerCase()
  const factor = unit ? SWIM_DIST_UNIT[unit] : undefined
  if (!start || !end || value === undefined || factor === undefined) return null
  const meters = Number(value) * factor
  return Number.isFinite(meters) && meters > 0 ? { start, end, meters } : null
}

export interface AppleSwimStrokeRecord {
  start: string
  end: string
  count: number
}

export interface AppleSwimLap extends AppleSwimStrokeRecord {
  stroke: SwimStroke | null
}

export interface AppleSwimWorkoutWindow {
  id: string
  start: string
  end: string
  totalM: number | null
  activeTimeS: number | null
}

export function matchSwimStrokeOpen(line: string): AppleSwimStrokeRecord | null {
  if (!line.includes('HKQuantityTypeIdentifierSwimmingStrokeCount')) return null
  const start = /startDate="([^"]+)"/.exec(line)?.[1]
  const end = /endDate="([^"]+)"/.exec(line)?.[1]
  const count = /\svalue="([\d.]+)"/.exec(line)?.[1]
  if (!start || !end || count === undefined) return null
  return { start, end, count: Number(count) }
}

export function matchSwimWorkout(line: string): AppleSwimWorkoutWindow | null {
  if (!line.includes('<Workout ') || !line.includes('HKWorkoutActivityTypeSwimming')) return null
  const start = /startDate="([^"]+)"/.exec(line)?.[1]
  const end = /endDate="([^"]+)"/.exec(line)?.[1]
  if (!start || !end) return null
  const distance = /\stotalDistance="([\d.]+)"/.exec(line)?.[1]
  const distanceUnit = /\stotalDistanceUnit="([^"]+)"/.exec(line)?.[1]?.toLowerCase()
  const distanceFactor = distanceUnit ? SWIM_DIST_UNIT[distanceUnit] : undefined
  const totalM =
    distance !== undefined && distanceFactor !== undefined
      ? Number(distance) * distanceFactor
      : null
  const duration = /\sduration="([\d.]+)"/.exec(line)?.[1]
  const durationUnit = /\sdurationUnit="([^"]+)"/.exec(line)?.[1]?.toLowerCase()
  const durationFactor = durationUnit ? SWIM_DURATION_UNIT[durationUnit] : undefined
  const activeTimeS =
    duration !== undefined && durationFactor !== undefined
      ? Number(duration) * durationFactor
      : null
  return {
    id: `xml:${start}|${end}`,
    start,
    end,
    totalM: totalM != null && Number.isFinite(totalM) && totalM > 0 ? totalM : null,
    activeTimeS:
      activeTimeS != null && Number.isFinite(activeTimeS) && activeTimeS > 0 ? activeTimeS : null,
  }
}

export function matchStrokeStyle(line: string): SwimStroke | null {
  const v = /key="HKSwimmingStrokeStyle" value="(\d+)"/.exec(line)?.[1]
  return v ? (STROKE_BY_VALUE[v] ?? null) : null
}

interface TimedSwimWorkout extends AppleSwimWorkoutWindow {
  startMs: number
  endMs: number
}

function timedSwimWorkouts(workouts: AppleSwimWorkoutWindow[]): TimedSwimWorkout[] {
  return workouts
    .map(workout => ({
      ...workout,
      startMs: Date.parse(workout.start),
      endMs: Date.parse(workout.end),
    }))
    .filter(workout => Number.isFinite(workout.startMs) && Number.isFinite(workout.endMs))
}

function findSwimWorkout(
  start: string,
  end: string,
  workouts: TimedSwimWorkout[],
): TimedSwimWorkout | null {
  const startMs = Date.parse(start)
  const endMs = Date.parse(end)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return null
  const containing = workouts.filter(
    workout => workout.startMs <= startMs && workout.endMs >= endMs,
  )
  const candidates =
    containing.length > 0
      ? containing
      : workouts.filter(workout => workout.startMs < endMs && workout.endMs > startMs)
  return (
    candidates.sort(
      (a, b) => a.endMs - a.startMs - (b.endMs - b.startMs) || a.id.localeCompare(b.id),
    )[0] ?? null
  )
}

function unionDurationS(intervals: { start: string; end: string }[]): number {
  const parsed = intervals
    .map(interval => ({ start: Date.parse(interval.start), end: Date.parse(interval.end) }))
    .filter(interval => Number.isFinite(interval.start) && interval.end > interval.start)
    .sort((a, b) => a.start - b.start || b.end - a.end)
  let total = 0
  let coveredUntil = -Infinity
  for (const interval of parsed) {
    total += Math.max(0, interval.end - Math.max(interval.start, coveredUntil))
    coveredUntil = Math.max(coveredUntil, interval.end)
  }
  return total / 1000
}

function unionStrokeTotals(laps: AppleSwimLap[]): { count: number; timeS: number } {
  const intervals = laps
    .filter(lap => lap.count > 0 && lap.stroke !== 'kickboard')
    .map(lap => ({
      count: Math.max(0, lap.count),
      start: Date.parse(lap.start),
      end: Date.parse(lap.end),
    }))
    .filter(interval => Number.isFinite(interval.start) && interval.end > interval.start)
    .sort((a, b) => a.start - b.start || b.end - a.end || b.count - a.count)
  let count = 0
  let timeMs = 0
  let coveredUntil = -Infinity
  for (const interval of intervals) {
    const durationMs = interval.end - interval.start
    const uncoveredMs = Math.max(0, interval.end - Math.max(interval.start, coveredUntil))
    count += interval.count * (uncoveredMs / durationMs)
    timeMs += uncoveredMs
    coveredUntil = Math.max(coveredUntil, interval.end)
  }
  return { count, timeS: timeMs / 1000 }
}

function intervalStrokeTotals(
  distance: AppleSwimDistanceRecord,
  laps: AppleSwimLap[],
): { count: number | null; timeS: number | null; stroke: SwimStroke | null } {
  const start = Date.parse(distance.start)
  const end = Date.parse(distance.end)
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start)
    return { count: null, timeS: null, stroke: null }
  const overlaps = laps
    .map(lap => {
      const lapStart = Date.parse(lap.start)
      const lapEnd = Date.parse(lap.end)
      const overlapMs = Math.max(0, Math.min(end, lapEnd) - Math.max(start, lapStart))
      return { lap, start: lapStart, end: lapEnd, overlapMs }
    })
    .filter(
      overlap =>
        Number.isFinite(overlap.start) &&
        Number.isFinite(overlap.end) &&
        overlap.end > overlap.start &&
        overlap.overlapMs > 0,
    )
    .sort(
      (a, b) =>
        b.overlapMs - a.overlapMs ||
        a.end - a.start - (b.end - b.start) ||
        b.lap.count - a.lap.count,
    )
  const stroke = overlaps[0]?.lap.stroke ?? null
  if (stroke === 'kickboard') return { count: null, timeS: null, stroke }
  const rateIntervals = overlaps.filter(
    overlap => overlap.lap.count > 0 && overlap.lap.stroke !== 'kickboard',
  )
  const exact = rateIntervals.find(interval => interval.start === start && interval.end === end)
  if (exact) return { count: exact.lap.count, timeS: (end - start) / 1000, stroke }
  const boundaries = new Set([start, end])
  for (const interval of rateIntervals) {
    boundaries.add(Math.max(start, interval.start))
    boundaries.add(Math.min(end, interval.end))
  }
  const sortedBoundaries = [...boundaries].sort((a, b) => a - b)
  let count = 0
  let timeMs = 0
  for (let index = 0; index < sortedBoundaries.length - 1; index++) {
    const segmentStart = sortedBoundaries[index]
    const segmentEnd = sortedBoundaries[index + 1]
    const segmentMs = segmentEnd - segmentStart
    if (segmentMs <= 0) continue
    const interval = rateIntervals
      .filter(candidate => candidate.start <= segmentStart && candidate.end >= segmentEnd)
      .sort(
        (a, b) =>
          a.end - a.start - (b.end - b.start) || a.start - b.start || b.lap.count - a.lap.count,
      )[0]
    if (!interval) continue
    count += (interval.lap.count / (interval.end - interval.start)) * segmentMs
    timeMs += segmentMs
  }
  return { count: count > 0 ? count : null, timeS: timeMs > 0 ? timeMs / 1000 : null, stroke }
}

interface SwimAggregate {
  id: string
  date: string
  start: string
  end: string
  workoutTotalM: number | null
  workoutActiveTimeS: number | null
  distances: AppleSwimDistanceRecord[]
  strokeLaps: AppleSwimLap[]
}

export function aggregateSwimLaps(
  strokeLaps: AppleSwimLap[],
  distanceLaps: AppleSwimDistanceRecord[],
  workoutWindows: AppleSwimWorkoutWindow[] = [],
): AppleSwim[] {
  const workouts = timedSwimWorkouts(workoutWindows)
  const uniqueLaps = new Map<string, AppleSwimLap>()
  for (const lap of strokeLaps) {
    const key = `${lap.start}\u0000${lap.end}`
    const existing = uniqueLaps.get(key)
    if (
      !existing ||
      lap.count > existing.count ||
      (lap.count === existing.count && !existing.stroke && lap.stroke !== null)
    ) {
      uniqueLaps.set(key, lap)
    }
  }
  const aggregates = new Map<string, SwimAggregate>()
  for (const workout of workouts) {
    const date = workout.start.slice(0, 10)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    aggregates.set(workout.id, {
      id: workout.id,
      date,
      start: workout.start,
      end: workout.end,
      workoutTotalM: workout.totalM,
      workoutActiveTimeS: workout.activeTimeS,
      distances: [],
      strokeLaps: [],
    })
  }
  const uniqueDistances = new Map<string, AppleSwimDistanceRecord>()
  for (const distance of distanceLaps) {
    if (!Number.isFinite(distance.meters) || distance.meters <= 0) continue
    const key = `${distance.start}\u0000${distance.end}`
    const existing = uniqueDistances.get(key)
    if (!existing || distance.meters > existing.meters) uniqueDistances.set(key, distance)
  }
  for (const distance of uniqueDistances.values()) {
    const date = distance.start.slice(0, 10)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    const workout = findSwimWorkout(distance.start, distance.end, workouts)
    const key = workout?.id ?? `xml:unassigned:${date}`
    let aggregate = aggregates.get(key)
    if (!aggregate) {
      aggregate = {
        id: key,
        date,
        start: workout?.start ?? distance.start,
        end: workout?.end ?? distance.end,
        workoutTotalM: workout?.totalM ?? null,
        workoutActiveTimeS: workout?.activeTimeS ?? null,
        distances: [],
        strokeLaps: [],
      }
      aggregates.set(key, aggregate)
    }
    if (!workout) {
      if (Date.parse(distance.start) < Date.parse(aggregate.start)) aggregate.start = distance.start
      if (Date.parse(distance.end) > Date.parse(aggregate.end)) aggregate.end = distance.end
    }
    aggregate.distances.push(distance)
  }
  for (const lap of uniqueLaps.values()) {
    const date = lap.start.slice(0, 10)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    const workout = findSwimWorkout(lap.start, lap.end, workouts)
    const aggregate = aggregates.get(workout?.id ?? `xml:unassigned:${date}`)
    if (aggregate) aggregate.strokeLaps.push(lap)
  }
  return [...aggregates.values()]
    .map(aggregate => {
      const totalM =
        aggregate.workoutTotalM ??
        aggregate.distances.reduce((total, distance) => total + distance.meters, 0)
      if (totalM <= 0) return null
      const strokeTotals = unionStrokeTotals(aggregate.strokeLaps)
      const activeTimeFromDistances = unionDurationS(aggregate.distances)
      const activeTimeS = aggregate.workoutActiveTimeS ?? activeTimeFromDistances
      const strokeByStart = new Map<string, SwimStroke>()
      for (const lap of aggregate.strokeLaps) {
        if (lap.stroke && !strokeByStart.has(lap.start)) strokeByStart.set(lap.start, lap.stroke)
      }
      const strokes: Record<string, number> = {}
      for (const distance of aggregate.distances) {
        const stroke = strokeByStart.get(distance.start)
        if (stroke) strokes[stroke] = (strokes[stroke] ?? 0) + distance.meters
      }
      for (const stroke of Object.keys(strokes)) {
        strokes[stroke] = Math.round(strokes[stroke])
      }
      const intervals = aggregate.distances
        .slice()
        .sort((a, b) => a.start.localeCompare(b.start) || a.end.localeCompare(b.end))
        .map(distance => {
          const start = parseIsoSecond(distance.start)
          const end = parseIsoSecond(distance.end)
          const stroke = intervalStrokeTotals(distance, aggregate.strokeLaps)
          if (!start || !end || Date.parse(end) <= Date.parse(start)) return null
          return {
            start,
            end,
            distanceM: Math.round(distance.meters * 10) / 10,
            startElapsedS:
              Math.round(
                Math.max(0, (Date.parse(distance.start) - Date.parse(aggregate.start)) / 1000) * 10,
              ) / 10,
            endElapsedS:
              Math.round(
                Math.max(0, (Date.parse(distance.end) - Date.parse(aggregate.start)) / 1000) * 10,
              ) / 10,
            durationS: Math.round(((Date.parse(end) - Date.parse(start)) / 1000) * 10) / 10,
            strokeCount: stroke.count != null ? Math.round(stroke.count * 10) / 10 : null,
            strokeTimeS: stroke.timeS != null ? Math.round(stroke.timeS * 10) / 10 : null,
            stroke: stroke.stroke,
          }
        })
        .filter(interval => interval !== null)
      return {
        id: aggregate.id,
        date: aggregate.date,
        start: parseIsoSecond(aggregate.start),
        end: parseIsoSecond(aggregate.end),
        laps: aggregate.distances.length,
        totalM: Math.round(totalM),
        activeTimeS: activeTimeS > 0 ? Math.round(activeTimeS) : null,
        strokeCount: strokeTotals.count > 0 ? Math.round(strokeTotals.count) : null,
        strokeTimeS: strokeTotals.timeS > 0 ? Math.round(strokeTotals.timeS) : null,
        strokes,
        intervals,
      }
    })
    .filter(swim => swim !== null)
    .sort(
      (a, b) =>
        (a.start ?? a.date).localeCompare(b.start ?? b.date) ||
        (a.id ?? '').localeCompare(b.id ?? ''),
    )
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
    const idValue = readString(swim, 'id')?.trim()
    const id = idValue ? idValue : null
    const date = readString(swim, 'date')?.slice(0, 10) ?? null
    const start = parseIsoSecond(swim.start)
    const end = parseIsoSecond(swim.end)
    const totalM = readNumber(swim, 'totalM')
    const laps = readNumber(swim, 'laps')
    const activeTimeS = num(swim.activeTimeS)
    const strokeCount = num(swim.strokeCount)
    const strokeTimeS = num(swim.strokeTimeS)
    const rawStrokes = isRecord(swim.strokes) ? swim.strokes : null
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date) || totalM == null || laps == null || !rawStrokes)
      continue
    const strokes: Record<string, number> = {}
    for (const [stroke, meters] of Object.entries(rawStrokes)) {
      const rounded = num(meters)
      if (rounded != null && rounded > 0) strokes[stroke] = Math.round(rounded)
    }
    const intervals: AppleSwimInterval[] = []
    const rawIntervals = Array.isArray(swim.intervals) ? swim.intervals : []
    for (const interval of rawIntervals) {
      if (!isRecord(interval)) continue
      const intervalStart = parseIsoSecond(interval.start)
      const intervalEnd = parseIsoSecond(interval.end)
      const distanceM = num(interval.distanceM)
      const startElapsedS = num(interval.startElapsedS)
      const endElapsedS = num(interval.endElapsedS)
      const durationS = num(interval.durationS)
      const intervalStrokeCount = num(interval.strokeCount)
      const intervalStrokeTimeS = num(interval.strokeTimeS)
      const strokeName = readString(interval, 'stroke')
      const stroke = SWIM_STROKES.find(candidate => candidate === strokeName) ?? null
      if (
        !intervalStart ||
        !intervalEnd ||
        Date.parse(intervalEnd) <= Date.parse(intervalStart) ||
        distanceM == null ||
        distanceM <= 0
      )
        continue
      intervals.push({
        start: intervalStart,
        end: intervalEnd,
        distanceM: Math.round(distanceM * 10) / 10,
        startElapsedS:
          startElapsedS != null && startElapsedS >= 0 ? Math.round(startElapsedS * 10) / 10 : null,
        endElapsedS:
          endElapsedS != null && endElapsedS > 0 ? Math.round(endElapsedS * 10) / 10 : null,
        durationS:
          durationS != null && durationS > 0
            ? Math.round(durationS * 10) / 10
            : Math.round(((Date.parse(intervalEnd) - Date.parse(intervalStart)) / 1000) * 10) / 10,
        strokeCount:
          intervalStrokeCount != null && intervalStrokeCount >= 0
            ? Math.round(intervalStrokeCount * 10) / 10
            : null,
        strokeTimeS:
          intervalStrokeTimeS != null && intervalStrokeTimeS > 0
            ? Math.round(intervalStrokeTimeS * 10) / 10
            : null,
        stroke,
      })
    }
    intervals.sort((a, b) => a.start.localeCompare(b.start) || a.end.localeCompare(b.end))
    out.push({
      id,
      date,
      start,
      end,
      totalM: Math.round(totalM),
      laps: Math.round(laps),
      activeTimeS: activeTimeS != null && activeTimeS >= 0 ? Math.round(activeTimeS) : null,
      strokeCount: strokeCount != null && strokeCount >= 0 ? Math.round(strokeCount) : null,
      strokeTimeS: strokeTimeS != null && strokeTimeS >= 0 ? Math.round(strokeTimeS) : null,
      strokes,
      intervals,
    })
  }
  return out.sort(
    (a, b) =>
      (a.start ?? a.date).localeCompare(b.start ?? b.date) ||
      (a.id ?? '').localeCompare(b.id ?? ''),
  )
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
