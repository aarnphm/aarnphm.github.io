import type { StravaRawCache } from '../plugins/stores/strava'
import type { ManualSaunaEntry } from '../plugins/stores/tracking'
import {
  TRAININGPEAKS_PLANNED_FIELDS,
  type TrainingPeaksStrengthWorkout,
  type TrainingPeaksWorkout,
} from './trainingpeaks-api'

export interface TrainingPeaksTitleSource {
  stravaId: number
  date: string
  startTime: string
  durationS: number
  movingTimeS: number
  distance: number
  sportType: string
  title: string
  description?: string
}

export interface TrainingPeaksWorkoutSummary {
  id: string
  date: string
  startTime: string
  durationS: number
  title: string
  endTime?: string
  workoutTypeId: number
  distance: number | null
  completed: boolean
}

export function trainingPeaksSupportedTitle(title: string): string {
  // The fitness API removes astral characters. Drop their full graphemes to avoid dangling joiners.
  const segments = new Intl.Segmenter('und', { granularity: 'grapheme' }).segment(title)
  return Array.from(segments, ({ segment }) =>
    /[\u{10000}-\u{10FFFF}]/u.test(segment) ? '' : segment,
  )
    .join('')
    .trim()
}

type TitleProtection = 'Planned workout' | 'Workout has authored instructions' | null

function hasAuthoredInstructions(text: string | null): boolean {
  // Exclude only the description section owned by the existing sauna sync.
  return Boolean(
    text
      ?.replace(
        /Sauna \/ passive heat session\.[\s\S]*?Strava: https:\/\/www\.strava\.com\/activities\/\d+/g,
        '',
      )
      .trim(),
  )
}

export function trainingPeaksTitleProtection(workout: TrainingPeaksWorkout): TitleProtection {
  if (
    workout.startTimePlanned != null ||
    TRAININGPEAKS_PLANNED_FIELDS.some(field => workout[field] != null) ||
    workout.structure != null
  )
    return 'Planned workout'
  return hasAuthoredInstructions(workout.description) ? 'Workout has authored instructions' : null
}

export function trainingPeaksStrengthTitleProtection(
  workout: TrainingPeaksStrengthWorkout,
): TitleProtection {
  if (
    workout.hasPrescribedData === true ||
    [
      'prescribedStartTime',
      'prescribedDurationInSeconds',
      'prescribedTss',
      'prescribedIntensityFactor',
    ].some(field => workout[field] != null) ||
    (typeof workout.complianceState === 'string' && workout.complianceState !== 'Unplanned') ||
    (Array.isArray(workout.blocks) && workout.blocks.length > 0) ||
    (Array.isArray(workout.sequenceSummary) && workout.sequenceSummary.length > 0)
  )
    return 'Planned workout'
  return hasAuthoredInstructions(workout.instructions) ? 'Workout has authored instructions' : null
}

export function selectTrainingPeaksTitleSources(
  cache: StravaRawCache,
  sauna: readonly ManualSaunaEntry[],
): TrainingPeaksTitleSource[] {
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
      const isSauna =
        entry || (stationary && activity.distance <= 100 && (explicitSauna || othershipHeat))
      const start = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})/.exec(activity.startDateLocal)
      if (!start || !Number.isFinite(activity.elapsedTime) || activity.elapsedTime <= 0) return []
      const details = description || (entry ? saunaDescription(entry) : '')
      if (!activity.name.trim()) return []
      return [
        {
          stravaId: activity.id,
          date: start[1],
          startTime: start[2],
          durationS: activity.elapsedTime,
          movingTimeS: activity.movingTime > 0 ? activity.movingTime : activity.elapsedTime,
          distance: activity.distance,
          sportType: activity.sportType,
          title: activity.name,
          ...(isSauna
            ? {
                description: [
                  'Sauna / passive heat session.',
                  details,
                  `Strava: https://www.strava.com/activities/${activity.id}`,
                ]
                  .filter(Boolean)
                  .join('\n\n'),
              }
            : {}),
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
  source: Pick<TrainingPeaksTitleSource, 'startTime'>,
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

function sportMatches(sport: string, workoutTypeId: number): boolean {
  // Other is also used for imported strength, walking, and triathlon transitions.
  if (workoutTypeId === 100) return true
  if (sport === 'Swim') return workoutTypeId === 1
  if (
    [
      'Ride',
      'VirtualRide',
      'EBikeRide',
      'MountainBikeRide',
      'GravelRide',
      'EMountainBikeRide',
      'Handcycle',
      'Velomobile',
    ].includes(sport)
  )
    return workoutTypeId === 2 || workoutTypeId === 8
  if (['Run', 'TrailRun', 'VirtualRun'].includes(sport)) return workoutTypeId === 3
  if (['Walk', 'Hike'].includes(sport)) return workoutTypeId === 13
  if (
    [
      'WeightTraining',
      'PhysicalTherapy',
      'Yoga',
      'Pilates',
      'Workout',
      'Crossfit',
      'RockClimbing',
    ].includes(sport)
  )
    return workoutTypeId === 9 || workoutTypeId === 29
  if (sport === 'NordicSki') return workoutTypeId === 11
  if (['Rowing', 'VirtualRow'].includes(sport)) return workoutTypeId === 12
  return false
}

function sameStationaryFinish(
  source: TrainingPeaksTitleSource,
  workout: TrainingPeaksWorkoutSummary,
): boolean {
  if (workout.workoutTypeId !== 29 || source.distance !== 0 || !workout.endTime) return false
  const start = clockSeconds(source.startTime)
  const recordedStart = clockSeconds(workout.startTime)
  const recordedEnd = clockSeconds(workout.endTime)
  if (start == null || recordedStart == null || recordedEnd == null) return false
  return (
    Math.abs(start - recordedStart) <= 300 &&
    Math.abs(((start + source.durationS) % 86400) - recordedEnd) <= 15 &&
    Math.min(source.durationS, workout.durationS) / Math.max(source.durationS, workout.durationS) >=
      0.7
  )
}

export function trainingPeaksTitleCandidates(
  source: TrainingPeaksTitleSource,
  workouts: readonly TrainingPeaksWorkoutSummary[],
): TrainingPeaksWorkoutSummary[] {
  return workouts.filter(workout => {
    const sameFinish = sameStationaryFinish(source, workout)
    if (
      workout.date !== source.date ||
      (!trainingPeaksStartMatches(source, workout.startTime) && !sameFinish) ||
      !sportMatches(source.sportType, workout.workoutTypeId) ||
      !workout.completed
    )
      return false
    const distanceTolerance = Math.max(100, source.distance * 0.1)
    if (
      workout.distance != null &&
      Math.abs(workout.distance - source.distance) > distanceTolerance
    )
      return false
    const shortest = Math.min(source.movingTimeS, source.durationS)
    const longest = Math.max(source.movingTimeS, source.durationS)
    const tolerance = Math.max(120, shortest * 0.15)
    const durationMatches =
      workout.durationS >= shortest - tolerance && workout.durationS <= longest + tolerance
    // Pool active time can exclude rests that Strava counts as moving time.
    const sameSwimRecording =
      source.sportType === 'Swim' &&
      workout.workoutTypeId === 1 &&
      source.distance > 0 &&
      workout.distance != null &&
      workout.distance > 0 &&
      Math.abs(workout.distance - source.distance) <= Math.max(50, source.distance * 0.02) &&
      Math.abs(
        (clockSeconds(source.startTime) ?? Infinity) -
          (clockSeconds(workout.startTime) ?? -Infinity),
      ) <= 2
    return durationMatches || sameSwimRecording || sameFinish
  })
}

export function trainingPeaksStrengthWorkoutSummary(
  workout: TrainingPeaksStrengthWorkout,
): TrainingPeaksWorkoutSummary {
  return {
    id: `strength:${workout.id}`,
    date: workout.startDateTime?.slice(0, 10) ?? workout.prescribedDate,
    startTime: workout.startDateTime?.match(/T(\d{2}:\d{2}:\d{2})/)?.[1] ?? '',
    endTime: workout.completedDateTime?.match(/T(\d{2}:\d{2}:\d{2})/)?.[1] ?? '',
    durationS: workout.executedDurationInSeconds ?? 0,
    title: workout.title,
    workoutTypeId: 29,
    distance: null,
    completed: workout.completedDateTime != null && (workout.executedDurationInSeconds ?? 0) > 0,
  }
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
    workoutTypeId: workout.workoutTypeValueId,
    distance: workout.distance,
    completed: workout.totalTime != null && workout.totalTime > 0,
  }
}

export function trainingPeaksSaunaDescription(
  source: { stravaId: number; description: string },
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
