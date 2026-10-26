import { isActivityKind, type StravaActivityDetail } from '../plugins/stores/strava'
import { isRecord } from './type-guards'

export type TimelineActivity = Pick<
  StravaActivityDetail,
  'sport' | 'name' | 'distanceKm' | 'movingTimeS'
> &
  Partial<Pick<StravaActivityDetail, 'elapsedTimeS'>>

export function readTimelineActivities(source: string | undefined): TimelineActivity[] {
  if (!source) return []
  const value: unknown = JSON.parse(source)
  if (
    !Array.isArray(value) ||
    !value.every(
      (activity): activity is TimelineActivity =>
        isRecord(activity) &&
        isActivityKind(activity.sport) &&
        typeof activity.name === 'string' &&
        typeof activity.distanceKm === 'number' &&
        Number.isFinite(activity.distanceKm) &&
        activity.distanceKm >= 0 &&
        typeof activity.movingTimeS === 'number' &&
        Number.isFinite(activity.movingTimeS) &&
        activity.movingTimeS >= 0 &&
        (activity.elapsedTimeS === undefined ||
          (typeof activity.elapsedTimeS === 'number' && Number.isFinite(activity.elapsedTimeS))),
    )
  )
    throw new Error('invalid timeline activities')
  return value
}
