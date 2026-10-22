import { isRecord } from '../../util/type-guards'
import {
  timedValue,
  type HeartRateSample,
  type HeartRateTimestampRepair,
  type MultisportActivity,
  type MultisportActivityBase,
  type MultisportDiscipline,
  type MultisportWorkout,
  type SportActivity,
  type TimedValue,
  type TransitionActivity,
} from './conversion-model'

function requiredString(record: Record<string, unknown>, key: string): string {
  const value = record[key]
  if (typeof value !== 'string' || value.trim().length === 0)
    throw new Error(`${key} must be a nonempty string`)
  return value.trim()
}

function optionalString(record: Record<string, unknown>, key: string): string | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'string' || value.trim().length === 0)
    throw new Error(`${key} must be a nonempty string when present`)
  return value.trim()
}

function requiredNumber(record: Record<string, unknown>, key: string): number {
  const value = record[key]
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${key} must be a finite number`)
  return value
}

function optionalNumber(record: Record<string, unknown>, key: string): number | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${key} must be a finite number when present`)
  return value
}

function nonnegative(value: number, key: string): number {
  if (value < 0) throw new Error(`${key} must be nonnegative`)
  return value
}

function positiveOptional(value: number | null, key: string): number | null {
  if (value == null) return null
  if (value <= 0) throw new Error(`${key} must be positive when present`)
  return value
}

function nonnegativeOptional(value: number | null, key: string): number | null {
  if (value == null) return null
  return nonnegative(value, key)
}

function summaryOptional(
  value: number | null,
  key: string,
  discipline: MultisportDiscipline | 'transition',
): number | null {
  return discipline === 'transition'
    ? nonnegativeOptional(value, key)
    : positiveOptional(value, key)
}

function disciplineForActivity(activity: string): MultisportDiscipline | 'transition' | null {
  switch (activity) {
    case 'swim':
    case 'swimming':
      return 'swim'
    case 'bike':
    case 'cycling':
      return 'bike'
    case 'run':
    case 'running':
      return 'run'
    case 'transition':
      return 'transition'
    default:
      return null
  }
}

function parseActivity(raw: unknown, index: number): MultisportActivity {
  if (!isRecord(raw)) throw new Error(`activities[${index}] must be an object`)
  const prefix = `activities[${index}]`
  const id = requiredString(raw, 'id')
  const activity = requiredString(raw, 'activity')
  const discipline = disciplineForActivity(activity)
  if (!discipline)
    throw new Error(`${prefix}.activity must be swimming, transition, cycling, or running`)
  const start = timedValue(requiredString(raw, 'start'), `${prefix}.start`)
  const end = timedValue(requiredString(raw, 'end'), `${prefix}.end`)
  if (end.timeMs <= start.timeMs) throw new Error(`${prefix}.end must be after start`)
  const durationS = nonnegative(requiredNumber(raw, 'durationS'), `${prefix}.durationS`)
  const elapsedTimeS = nonnegative(requiredNumber(raw, 'elapsedTimeS'), `${prefix}.elapsedTimeS`)
  const distanceM = nonnegativeOptional(optionalNumber(raw, 'distanceM'), `${prefix}.distanceM`)
  const activeEnergyKcal = nonnegativeOptional(
    optionalNumber(raw, 'activeEnergyKcal'),
    `${prefix}.activeEnergyKcal`,
  )
  const averageHeartRateBpm = summaryOptional(
    optionalNumber(raw, 'averageHeartRateBpm'),
    `${prefix}.averageHeartRateBpm`,
    discipline,
  )
  const averagePowerW = summaryOptional(
    optionalNumber(raw, 'averagePowerW'),
    `${prefix}.averagePowerW`,
    discipline,
  )
  const averageCadencePerMinute = summaryOptional(
    optionalNumber(raw, 'averageCadencePerMinute'),
    `${prefix}.averageCadencePerMinute`,
    discipline,
  )
  const lapCount = summaryOptional(
    optionalNumber(raw, 'lapCount'),
    `${prefix}.lapCount`,
    discipline,
  )
  const base: MultisportActivityBase = {
    id,
    activity,
    start,
    end,
    durationS,
    elapsedTimeS,
    distanceM,
    activeEnergyKcal,
    averageHeartRateBpm,
    averagePowerW,
    averageCadencePerMinute,
    lapCount,
  }
  return discipline === 'transition'
    ? { ...base, kind: 'transition', discipline }
    : { ...base, kind: 'sport', discipline }
}

function parseHeartRate(raw: unknown): {
  samples: HeartRateSample[]
  repairs: HeartRateTimestampRepair[]
} {
  if (raw == null) return { samples: [], repairs: [] }
  if (!Array.isArray(raw)) throw new Error('heartRate must be an array')
  const parsed: { timed: TimedValue; bpm: number; sourceIndex: number }[] = []
  for (const [index, value] of raw.entries()) {
    if (!isRecord(value)) throw new Error(`heartRate[${index}] must be an object`)
    const timed = timedValue(requiredString(value, 'time'), `heartRate[${index}].time`)
    const bpm = requiredNumber(value, 'bpm')
    if (bpm <= 0 || bpm > 300) throw new Error(`heartRate[${index}].bpm is invalid`)
    parsed.push({ timed, bpm: Math.round(bpm), sourceIndex: index })
  }
  parsed.sort(
    (left, right) => left.timed.timeMs - right.timed.timeMs || left.sourceIndex - right.sourceIndex,
  )
  const reserved = new Set(parsed.map(sample => sample.timed.timeMs))
  const used = new Set<number>()
  const samples: HeartRateSample[] = []
  const repairs: HeartRateTimestampRepair[] = []
  for (const sample of parsed) {
    let outputTimeMs = sample.timed.timeMs
    if (used.has(outputTimeMs)) {
      outputTimeMs += 1
      while (reserved.has(outputTimeMs) || used.has(outputTimeMs)) outputTimeMs += 1
    }
    used.add(outputTimeMs)
    const outputIso = new Date(outputTimeMs).toISOString()
    samples.push({
      iso: outputIso,
      timeMs: outputTimeMs,
      bpm: sample.bpm,
      sourceIso: sample.timed.iso,
      sourceTimeMs: sample.timed.timeMs,
    })
    if (outputTimeMs !== sample.timed.timeMs)
      repairs.push({
        sourceTimestamp: sample.timed.iso,
        outputTimestamp: outputIso,
        bpm: sample.bpm,
        reason: 'duplicate legacy parent heart-rate timestamp',
        provenance: 'converter:stable-millisecond-disambiguation',
      })
  }
  samples.sort((left, right) => left.timeMs - right.timeMs)
  return { samples, repairs }
}

export function parseWorkout(raw: unknown): MultisportWorkout {
  if (!isRecord(raw)) throw new Error('workout must be an object')
  const id = requiredString(raw, 'id')
  const activity = requiredString(raw, 'activity')
  if (activity !== 'swimBikeRun') throw new Error(`workout ${id} is not swimBikeRun`)
  const start = timedValue(requiredString(raw, 'start'), 'workout.start')
  const end = timedValue(requiredString(raw, 'end'), 'workout.end')
  if (end.timeMs <= start.timeMs) throw new Error('workout.end must be after start')
  const gpxFile = optionalString(raw, 'gpxFile')
  if (!gpxFile) throw new Error(`workout ${id} has no gpxFile`)
  if (!Array.isArray(raw.activities)) throw new Error('activities must be an array')
  const activities = raw.activities.map(parseActivity)
  for (const [index, segment] of activities.entries()) {
    if (segment.start.timeMs < start.timeMs || segment.end.timeMs > end.timeMs)
      throw new Error(`${segment.discipline} activity is outside the parent workout`)
    const previous = activities[index - 1]
    if (previous && previous.end.timeMs > segment.start.timeMs)
      throw new Error(`${previous.discipline} and ${segment.discipline} activities overlap`)
  }
  const sports = activities.filter(isSportActivity)
  const first = sports[0]
  const second = sports[1]
  const third = sports[2]
  if (!first || !second || !third || sports.length !== 3)
    throw new Error('multisport workout must contain exactly one swim, bike, and run')
  const legs: [SportActivity, SportActivity, SportActivity] = [first, second, third]
  if (
    legs[0].discipline !== 'swim' ||
    legs[1].discipline !== 'bike' ||
    legs[2].discipline !== 'run'
  )
    throw new Error('sport activities must be ordered swimming, cycling, running')
  const transitions = activities.filter(isTransitionActivity)
  for (const transition of transitions) {
    const previousSport = [...sports]
      .reverse()
      .find(sport => sport.end.timeMs <= transition.start.timeMs)
    const nextSport = sports.find(sport => sport.start.timeMs >= transition.end.timeMs)
    if (!previousSport || !nextSport)
      throw new Error('transition activities must occur between sport activities')
  }
  for (let index = 0; index < sports.length - 1; index++) {
    const previous = sports[index]
    const next = sports[index + 1]
    if (!previous || !next) continue
    const between = transitions.filter(
      transition =>
        transition.start.timeMs >= previous.end.timeMs &&
        transition.end.timeMs <= next.start.timeMs,
    )
    if (between.length > 1)
      throw new Error(`${previous.discipline} and ${next.discipline} contain multiple transitions`)
  }
  const heartRate = parseHeartRate(raw.heartRate)
  return {
    id,
    activity,
    start,
    end,
    gpxFile,
    heartRate: heartRate.samples,
    heartRateTimestampRepairs: heartRate.repairs,
    activities,
    legs,
    transitions,
  }
}

function isSportActivity(activity: MultisportActivity): activity is SportActivity {
  return activity.kind === 'sport'
}

function isTransitionActivity(activity: MultisportActivity): activity is TransitionActivity {
  return activity.kind === 'transition'
}

export function workoutRecord(document: unknown, workoutId: string): Record<string, unknown> {
  if (!isRecord(document) || !Array.isArray(document.workouts))
    throw new Error('input must contain a workouts array')
  const matches = document.workouts.filter(value => isRecord(value) && value.id === workoutId)
  if (matches.length === 0) throw new Error(`workout ${workoutId} was not found`)
  if (matches.length > 1) throw new Error(`workout ${workoutId} is duplicated`)
  const match = matches[0]
  if (!isRecord(match)) throw new Error(`workout ${workoutId} must be an object`)
  return match
}
