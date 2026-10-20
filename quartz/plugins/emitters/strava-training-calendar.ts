import type { LoadedStravaData } from '../../util/strava-payload'
import type { TrainingPeaksCalendar } from '../../util/trainingpeaks-calendar'
import { linkTrainingPeaksCalendarActivities } from '../../util/trainingpeaks-activity-links'
import { trainingPeaksActivityPeaks } from '../../util/trainingpeaks-calendar-peaks'
import { trainingPeaksLocalZones } from '../../util/trainingpeaks-calendar-zones'

type StravaActivity = NonNullable<LoadedStravaData['trackedCache']>['activities'][string]

/**
 * Links the imported TrainingPeaks calendar to the recorded activities. The calendar is sealed
 * before it is written; `plannedActivityIds` names only the activities that have a planned
 * workout, so pages can render the planned tab before the calendar is unlocked.
 */
export function linkTrainingCalendar(
  imported: TrainingPeaksCalendar | null,
  activities: readonly StravaActivity[],
  { payload, analytics, sources, wahooActivities }: LoadedStravaData,
) {
  if (!imported) return { calendar: null, plannedActivityIds: [] }
  const linked = linkTrainingPeaksCalendarActivities(
    imported,
    activities.flatMap(activity => {
      const detail = payload.details[String(activity.id)]
      return detail
        ? [
            {
              ...activity,
              sport: detail.sport,
              garmin: detail.garmin,
              wahoo: detail.wahoo ? wahooActivities[detail.wahoo.activityId] : undefined,
              peaks: trainingPeaksActivityPeaks({
                strava: sources.strava?.streams?.[String(activity.id)],
                garmin: detail.garmin
                  ? sources.garmin?.streams?.[detail.garmin.activityId]
                  : undefined,
                powerCurve: detail.powerCurve,
              }),
            },
          ]
        : []
    }),
  )
  return {
    calendar: { ...linked, localZones: trainingPeaksLocalZones(analytics, payload.zones.ftp) },
    plannedActivityIds: linked.workouts.flatMap(workout =>
      workout.activity ? [workout.activity.id] : [],
    ),
  }
}
