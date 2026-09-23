import type { StravaRawCache } from '../plugins/stores/strava'
import type { ManualSaunaEntry } from '../plugins/stores/tracking'
import { TRAININGPEAKS_PLANNED_FIELDS, type TrainingPeaksWorkout } from './trainingpeaks-api'

export interface TrainingPeaksSaunaSource {
  stravaId: number
  date: string
  startTime: string
  durationS: number
  title: string
  description: string
}

export interface TrainingPeaksWorkoutSummary {
  id: string
  date: string
  startTime: string
  durationS: number
  title: string
  sport: string
  distance: number | null
  planned: boolean
}

export function selectTrainingPeaksSaunaSources(
  cache: StravaRawCache,
  sauna: readonly ManualSaunaEntry[],
): TrainingPeaksSaunaSource[] {
  const linked = new Map(sauna.map(entry => [entry.stravaActivityId, entry]))
  return Object.values(cache.activities)
    .flatMap(activity => {
      const entry = linked.get(activity.id)
      const description =
        cache.activityDetails?.[activity.id]?.description?.replace(/\r\n?/g, '\n').trim() ?? ''
      const text = `${activity.name}\n${description}`
      const stationary = ['Workout', 'PhysicalTherapy', 'Yoga'].includes(activity.sportType)
      const explicitSauna = /\bsauna\b|\bpassive heat\b/i.test(text)
      const othershipHeat = /\bothership\b/i.test(text) && /\bHTL\s+\d/i.test(text)
      if (!entry && !(stationary && activity.distance <= 100 && (explicitSauna || othershipHeat)))
        return []
      const start = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})/.exec(activity.startDateLocal)
      if (!start || !Number.isFinite(activity.elapsedTime) || activity.elapsedTime <= 0) return []
      const details = description || (entry ? saunaDescription(entry) : '')
      const name = activity.name.trim()
      if (!name) return []
      return [
        {
          stravaId: activity.id,
          date: start[1],
          startTime: start[2],
          durationS: activity.elapsedTime,
          title: /^sauna\b/i.test(name) ? name : `Sauna - ${name}`,
          description: [
            'Sauna / passive heat session.',
            details,
            `Strava: https://www.strava.com/activities/${activity.id}`,
          ]
            .filter(Boolean)
            .join('\n\n'),
        },
      ]
    })
    .sort((a, b) => a.date.localeCompare(b.date) || a.startTime.localeCompare(b.startTime))
}

function saunaDescription(entry: ManualSaunaEntry): string {
  return [
    entry.location?.name,
    `Sauna temperature: ${entry.temperatureC} °C`,
    `Relative humidity: ${entry.humidityPct}%`,
    `Session duration: ${entry.durationS / 60} min`,
    `Cool-down: ${entry.cooldown}`,
    entry.heatTrainingLoad == null ? null : `HTL: ${entry.heatTrainingLoad}`,
  ]
    .filter(Boolean)
    .join('\n')
}

function clockSeconds(value: string): number | null {
  const match = /^(\d{2}):(\d{2}):(\d{2})$/.exec(value)
  if (!match || Number(match[1]) >= 24 || Number(match[2]) >= 60 || Number(match[3]) >= 60)
    return null
  return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3])
}

export function trainingPeaksStartMatches(
  source: TrainingPeaksSaunaSource,
  value: string,
): boolean {
  const sourceSeconds = clockSeconds(source.startTime)
  const workoutSeconds = clockSeconds(value)
  return (
    sourceSeconds != null &&
    workoutSeconds != null &&
    Math.abs(sourceSeconds - workoutSeconds) <= 120
  )
}

export function trainingPeaksSaunaCandidates(
  source: TrainingPeaksSaunaSource,
  workouts: readonly TrainingPeaksWorkoutSummary[],
): TrainingPeaksWorkoutSummary[] {
  return workouts.filter(
    workout =>
      workout.date === source.date &&
      trainingPeaksStartMatches(source, workout.startTime) &&
      workout.sport === 'Other' &&
      !workout.planned &&
      (workout.distance == null || workout.distance === 0) &&
      Math.abs(workout.durationS - source.durationS) <= Math.max(30, source.durationS * 0.01),
  )
}

export function trainingPeaksWorkoutSummary(
  workout: TrainingPeaksWorkout,
): TrainingPeaksWorkoutSummary {
  return {
    id: String(workout.workoutId),
    date: workout.workoutDay.slice(0, 10),
    startTime: workout.startTime?.match(/T(\d{2}:\d{2}:\d{2})/)?.[1] ?? '',
    durationS: Math.round((workout.totalTime ?? 0) * 3600),
    title: workout.title,
    sport: workout.workoutTypeValueId === 100 ? 'Other' : '',
    distance: workout.distance,
    planned:
      workout.totalTime == null ||
      workout.totalTime <= 0 ||
      workout.startTimePlanned != null ||
      TRAININGPEAKS_PLANNED_FIELDS.some(field => workout[field] != null),
  }
}

export function trainingPeaksSaunaDescription(
  source: TrainingPeaksSaunaSource,
  existing: string,
): string {
  const normalized = existing.replace(/\r\n?/g, '\n').trim()
  const marker = `Strava: https://www.strava.com/activities/${source.stravaId}`
  if (!normalized || normalized === source.description) return source.description
  // Keep existing notes outside the section owned by this sync.
  const start = normalized.indexOf('Sauna / passive heat session.')
  const end = normalized.indexOf(marker, start)
  if (start >= 0 && end >= start) {
    return [
      normalized.slice(0, start).trim(),
      source.description,
      normalized.slice(end + marker.length).trim(),
    ]
      .filter(Boolean)
      .join('\n\n')
  }
  const sourceDetails = source.description.split('\n\n').slice(1, -1).join('\n\n')
  if (normalized === sourceDetails) return source.description
  return `${normalized}\n\n${source.description}`
}
