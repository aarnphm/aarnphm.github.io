import type { RawStravaActivity } from '../plugins/stores/strava'
import type { WahooActivity } from '../plugins/stores/wahoo'

export interface RecordingWindow {
  startDate: string
  elapsedTimeS: number | null
  distanceM: number | null
}

export interface RouvyWahooPlan {
  canonical: RawStravaActivity
  wahoo: WahooActivity
  duplicate: RawStravaActivity | null
}

export function isRouvyActivity(activity: RawStravaActivity): boolean {
  return activity.sportType === 'VirtualRide' && /rouvy/i.test(activity.deviceName ?? '')
}

function coversRouvy(activity: RawStravaActivity, source: RecordingWindow): boolean {
  const start = Date.parse(activity.startDate)
  const nativeStart = Date.parse(source.startDate)
  const duration = source.elapsedTimeS ?? 0
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(nativeStart) ||
    duration <= 0 ||
    activity.elapsedTime <= 0
  )
    return false
  const overlap =
    Math.min(start + activity.elapsedTime * 1000, nativeStart + duration * 1000) -
    Math.max(start, nativeStart)
  return (
    Math.abs(start - nativeStart) <= 120_000 &&
    Math.abs(duration - activity.elapsedTime) <= 120 &&
    overlap / (activity.elapsedTime * 1000) >= 0.98
  )
}

export function isWahooStravaDuplicate(
  activity: RawStravaActivity,
  source: RecordingWindow,
): boolean {
  return (
    /wahoo|elemnt/i.test(activity.deviceName ?? '') &&
    (activity.sportType === 'Ride' || activity.sportType === 'VirtualRide') &&
    Math.abs(Date.parse(activity.startDate) - Date.parse(source.startDate)) <= 2000 &&
    source.elapsedTimeS != null &&
    Math.abs(activity.elapsedTime - source.elapsedTimeS) <= 5 &&
    source.distanceM != null &&
    Math.abs(activity.distance - source.distanceM) <= 50
  )
}

export function planRouvyWahooPairs(
  activities: readonly RawStravaActivity[],
  recordings: readonly WahooActivity[],
  since: string,
): RouvyWahooPlan[] {
  const routes = activities.filter(isRouvyActivity)
  const candidates = routes.flatMap(canonical =>
    recordings
      .filter(source => source.sport === 'bike' && coversRouvy(canonical, source))
      .map(wahoo => ({ canonical, wahoo })),
  )
  return candidates.flatMap(pair => {
    if (
      pair.canonical.startDateLocal.slice(0, 10) < since ||
      candidates.filter(other => other.canonical.id === pair.canonical.id).length !== 1 ||
      candidates.filter(other => other.wahoo.id === pair.wahoo.id).length !== 1
    )
      return []
    const duplicates = activities.filter(
      activity => activity.id !== pair.canonical.id && isWahooStravaDuplicate(activity, pair.wahoo),
    )
    return [{ ...pair, duplicate: duplicates.length === 1 ? duplicates[0] : null }]
  })
}
