import type { StravaPayload, StravaRawCache } from '../plugins/stores/strava'
import type { TrainingPeaksCalendar } from './trainingpeaks-calendar'
import { inferStrengthPrescription, normalizeStrengthRoutineTitle } from './strength-prescription'
import { linkTrainingPeaksCalendarActivities } from './trainingpeaks-activity-links'

export function enrichTrainingPeaksStrength(
  payload: StravaPayload,
  cache: StravaRawCache | null,
  calendar: TrainingPeaksCalendar | null | undefined,
): void {
  if (!cache || !calendar) return
  const activities = Object.values(cache.activities).flatMap(activity => {
    const detail = payload.details[String(activity.id)]
    return detail?.sport === 'strength'
      ? [{ ...activity, sport: detail.sport, garmin: detail.garmin }]
      : []
  })
  const linked = linkTrainingPeaksCalendarActivities(calendar, activities)
  const proposals = linked.workouts.flatMap(workout => {
    if (workout.sport !== 'strength') return []
    let id = workout.activity?.id
    if (workout.status === 'planned') {
      const matches = activities.filter(
        activity =>
          activity.startDateLocal.slice(0, 10) === workout.date &&
          normalizeStrengthRoutineTitle(activity.name) ===
            normalizeStrengthRoutineTitle(workout.title),
      )
      if (matches.length !== 1) return []
      id = matches[0].id
    }
    if (id == null) return []
    const strength = inferStrengthPrescription(workout.title, workout.description)
    return strength?.prescription
      ? [{ id, workout, strength, prescription: strength.prescription }]
      : []
  })
  for (const activity of activities) {
    const candidates = proposals.filter(proposal => proposal.id === activity.id)
    const descriptions = candidates.filter(
      proposal => proposal.strength.source === 'strava-description',
    )
    const preferred = descriptions.length ? descriptions : candidates
    if (preferred.length !== 1) continue
    const candidate = preferred[0]
    const detail = payload.details[String(activity.id)]
    if (
      detail.strength?.source === 'manual' ||
      (candidate.strength.source !== 'strava-description' &&
        detail.strength?.source === 'strava-description')
    )
      continue
    detail.strength = {
      ...candidate.strength,
      ...(detail.strength?.totalsSource === 'manual'
        ? {
            volumeKg: detail.strength.volumeKg,
            totalSets: detail.strength.totalSets,
            totalReps: detail.strength.totalReps,
            totalsSource: detail.strength.totalsSource,
          }
        : {}),
      source:
        candidate.strength.source === 'strava-description'
          ? 'trainingpeaks-description'
          : 'trainingpeaks-routine',
      prescription: { ...candidate.prescription, trainingPeaksWorkoutId: candidate.workout.id },
    }
  }
}
