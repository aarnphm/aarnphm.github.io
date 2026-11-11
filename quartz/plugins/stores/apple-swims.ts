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
  location: SwimLocation | null
  waterTemperatureC: number | null
}

export type SwimLocation = 'pool' | 'openWater'

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
        location: null,
        waterTemperatureC: null,
      }
    })
    .filter(swim => swim !== null)
    .sort(
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
