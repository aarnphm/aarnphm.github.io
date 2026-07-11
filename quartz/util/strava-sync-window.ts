import type { RawStravaActivity } from '../plugins/stores/strava'

const DAY_SECONDS = 86_400

export const DEFAULT_STRAVA_REFRESH_WINDOW_DAYS = 30

export interface StravaActivityReconciliation {
  activities: Record<string, RawStravaActivity>
  removedIds: string[]
}

export function stravaFetchAfter(
  lastActivityStart: number | null | undefined,
  stale: boolean,
  refreshWindowDays = DEFAULT_STRAVA_REFRESH_WINDOW_DAYS,
): number {
  if (stale) return 0
  if (lastActivityStart == null || !Number.isFinite(lastActivityStart) || lastActivityStart <= 0)
    return 0
  const days = Math.max(0, refreshWindowDays)
  return Math.max(0, Math.floor(lastActivityStart - days * DAY_SECONDS))
}

export function reconcileStravaActivities(
  previous: Record<string, RawStravaActivity> | undefined,
  fetched: readonly RawStravaActivity[],
  after: number,
): StravaActivityReconciliation {
  const activities: Record<string, RawStravaActivity> = {}
  if (after > 0) {
    for (const [id, activity] of Object.entries(previous ?? {})) {
      const started = Math.floor(Date.parse(activity.startDate) / 1000)
      if (!Number.isFinite(started) || started <= after) activities[id] = activity
    }
  }
  for (const activity of fetched) {
    const id = String(activity.id)
    const cached = previous?.[id]
    activities[id] = { ...activity, calories: activity.calories ?? cached?.calories }
  }
  return {
    activities,
    removedIds: Object.keys(previous ?? {}).filter(id => activities[id] === undefined),
  }
}
