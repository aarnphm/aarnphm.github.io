import type { ActivityKind, GarminVerification, RawStravaActivity } from '../plugins/stores/strava'
import {
  isTrainingPeaksCalendarDay,
  type TrainingPeaksCalendar,
  type TrainingPeaksCalendarActivityLink,
  type TrainingPeaksCalendarPeak,
  type TrainingPeaksCalendarWorkout,
} from './trainingpeaks-calendar'

export type TrainingPeaksLinkActivity = Pick<
  RawStravaActivity,
  'id' | 'name' | 'startDate' | 'startDateLocal' | 'distance' | 'movingTime' | 'elapsedTime'
> & {
  sport: ActivityKind
  peaks?: TrainingPeaksCalendarPeak[]
  garmin?: Pick<
    GarminVerification,
    'startDate' | 'distanceM' | 'movingTimeS' | 'elapsedTimeS'
  > | null
}

interface Recording {
  date: string
  clock: number
  localStartMs: number
  distance: number | null
  durations: number[]
}

interface Candidate {
  activity: TrainingPeaksLinkActivity
  date: string
  recordings: Recording[]
}

interface Proposal {
  candidates: Candidate[]
  match: TrainingPeaksCalendarActivityLink['match']
}

function clockSeconds(value: string): number | null {
  const match = /^(\d{2}):(\d{2}):(\d{2})$/.exec(value)
  if (!match || Number(match[1]) > 23 || Number(match[2]) > 59 || Number(match[3]) > 59) return null
  return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3])
}

function localStart(value: string): Pick<Recording, 'date' | 'clock' | 'localStartMs'> | null {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$/.exec(
    value,
  )
  if (!match || !isTrainingPeaksCalendarDay(match[1])) return null
  const clock = clockSeconds(match[2])
  return clock === null
    ? null
    : { date: match[1], clock, localStartMs: Date.parse(`${match[1]}T${match[2]}.000Z`) }
}

function nonnegative(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value >= 0 ? value : null
}

function positive(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value) && value > 0
}

function candidate(activity: TrainingPeaksLinkActivity): Candidate | null {
  if (!Number.isSafeInteger(activity.id) || activity.id <= 0) return null
  const local = localStart(activity.startDateLocal)
  if (!local) return null
  const recordings: Recording[] = [
    {
      ...local,
      distance: nonnegative(activity.distance),
      durations: [activity.movingTime, activity.elapsedTime].filter(positive),
    },
  ]
  if (activity.garmin) {
    const offset = Date.parse(activity.garmin.startDate) - Date.parse(activity.startDate)
    if (Number.isFinite(offset)) {
      // Preserve the source's wall-clock offset, including recordings made while travelling.
      const garminLocal = localStart(new Date(local.localStartMs + offset).toISOString())
      if (garminLocal)
        recordings.push({
          ...garminLocal,
          distance: nonnegative(activity.garmin.distanceM),
          durations: [activity.garmin.movingTimeS, activity.garmin.elapsedTimeS].filter(positive),
        })
    }
  }
  return { activity, date: local.date, recordings }
}

function sportMatches(
  workout: TrainingPeaksCalendarWorkout,
  activity: TrainingPeaksLinkActivity,
): boolean {
  if (workout.sport !== 'other') return workout.sport === activity.sport
  return (
    activity.sport === 'strength' ||
    activity.sport === 'walk' ||
    activity.sport === 'yoga' ||
    activity.sport === 'treatment' ||
    activity.sport === 'sauna'
  )
}

function sourceIds(description: string): Set<string> {
  const ids = new Set<string>()
  for (const match of description.matchAll(/https?:\/\/[^\s<>]+/g)) {
    const value = match[0].replace(/[),.;\]'"]+$/, '')
    if (!URL.canParse(value)) continue
    const url = new URL(value)
    if (url.hostname !== 'strava.com' && url.hostname !== 'www.strava.com') continue
    const path = /^\/activities\/([1-9]\d*)\/?$/.exec(url.pathname)
    if (path) ids.add(path[1])
  }
  return ids
}

function recordingMatches(
  workout: TrainingPeaksCalendarWorkout,
  recording: Recording,
  explicit: boolean,
): boolean {
  if (recording.date !== workout.date) return false
  const clock = workout.startTime === null ? null : clockSeconds(workout.startTime)
  if (workout.startTime !== null && clock === null) return false
  if (clock !== null && Math.abs(clock - recording.clock) > 120) return false

  const duration = workout.actual.durationSeconds
  let durationNear = false
  let durationInRange = false
  if (positive(duration) && recording.durations.length > 0) {
    const tolerance = Math.max(30, duration * 0.02)
    const minimum = Math.min(...recording.durations)
    const maximum = Math.max(...recording.durations)
    // Providers count pauses differently; keep each device's own moving-to-elapsed interval.
    durationInRange = duration >= minimum - tolerance && duration <= maximum + tolerance
    if (!durationInRange) return false
    durationNear = recording.durations.some(value => Math.abs(duration - value) <= tolerance)
  }

  const distance = nonnegative(workout.actual.distanceMeters)
  let positiveDistanceMatches = false
  if (distance !== null && recording.distance !== null) {
    if (Math.abs(distance - recording.distance) > Math.max(50, distance * 0.02)) return false
    positiveDistanceMatches = distance > 0 && recording.distance > 0
  }
  if (explicit) return true
  if (clock !== null) return durationInRange || positiveDistanceMatches
  return durationNear && positiveDistanceMatches
}

/** Supply only activity records whose details are emitted to the local /triathlon/on pages. */
export function linkTrainingPeaksCalendarActivities(
  calendar: TrainingPeaksCalendar,
  activities: readonly TrainingPeaksLinkActivity[],
): TrainingPeaksCalendar {
  const activityCounts = new Map<number, number>()
  for (const activity of activities)
    activityCounts.set(activity.id, (activityCounts.get(activity.id) ?? 0) + 1)
  const byDate = new Map<string, Candidate[]>()
  for (const activity of activities) {
    if (activityCounts.get(activity.id) !== 1) continue
    const item = candidate(activity)
    if (!item) continue
    const day = byDate.get(item.date) ?? []
    day.push(item)
    byDate.set(item.date, day)
  }

  const proposals = new Map<string, Proposal>()
  const contenders = new Map<number, number>()
  for (const workout of calendar.workouts) {
    if (workout.status !== 'completed') continue
    const referenced = sourceIds(workout.description)
    if (referenced.size > 1) continue
    const explicit = referenced.size === 1
    const candidates = (byDate.get(workout.date) ?? []).filter(
      item =>
        (!explicit || referenced.has(String(item.activity.id))) &&
        sportMatches(workout, item.activity) &&
        item.recordings.some(recording => recordingMatches(workout, recording, explicit)),
    )
    proposals.set(workout.id, { candidates, match: explicit ? 'source-id' : 'recording' })
    for (const item of candidates)
      contenders.set(item.activity.id, (contenders.get(item.activity.id) ?? 0) + 1)
  }

  return {
    ...calendar,
    workouts: calendar.workouts.map(workout => {
      const unlinked = { ...workout }
      delete unlinked.activity
      const proposal = proposals.get(workout.id)
      const target = proposal?.candidates.length === 1 ? proposal.candidates[0] : null
      if (!target || contenders.get(target.activity.id) !== 1 || !proposal) return unlinked
      return {
        ...unlinked,
        activity: {
          id: target.activity.id,
          date: target.date,
          title: target.activity.name,
          match: proposal.match,
          ...(target.activity.peaks ? { peaks: target.activity.peaks } : {}),
        },
      }
    }),
  }
}
