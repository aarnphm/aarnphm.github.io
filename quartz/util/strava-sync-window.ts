const DAY_SECONDS = 86_400

export const DEFAULT_STRAVA_REFRESH_WINDOW_DAYS = 7

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
