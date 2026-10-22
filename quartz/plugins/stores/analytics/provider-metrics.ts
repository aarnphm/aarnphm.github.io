import type { RawStravaActivity, StravaActivityDetail } from '../strava'

export function activityWithProviderMetrics(
  activity: RawStravaActivity,
  detail: StravaActivityDetail | undefined,
): RawStravaActivity {
  if (!detail) return activity
  return {
    ...activity,
    averageHeartrate: activity.averageHeartrate ?? detail.avgHr ?? undefined,
    maxHeartrate: activity.maxHeartrate ?? detail.maxHr ?? undefined,
    averageCadence: activity.averageCadence ?? detail.avgCadence ?? undefined,
    averageWatts: activity.averageWatts ?? detail.avgWatts ?? undefined,
    weightedAverageWatts: activity.weightedAverageWatts ?? detail.npWatts ?? undefined,
    calories: activity.calories ?? detail.calories ?? undefined,
    deviceWatts: activity.averageWatts == null ? detail.deviceWatts : activity.deviceWatts,
  }
}
