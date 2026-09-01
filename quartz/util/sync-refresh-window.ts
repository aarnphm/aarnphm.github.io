import { localIsoDayOffset } from './local-date'

export const DEFAULT_SYNC_REFRESH_WINDOW_DAYS = 30

type SyncRefreshEnvironment = Readonly<Record<string, string | undefined>>

function refreshDays(environment: SyncRefreshEnvironment, names: readonly string[]): number {
  for (const name of names) {
    const value = environment[name]?.trim()
    if (!value) continue
    const parsed = Number(value)
    if (!Number.isInteger(parsed) || parsed < 0)
      throw new Error(`${name} must be a nonnegative integer`)
    return parsed
  }
  return DEFAULT_SYNC_REFRESH_WINDOW_DAYS
}

export function syncRefreshDays(environment: SyncRefreshEnvironment = process.env): number {
  return refreshDays(environment, ['SYNC_REFRESH_DAYS', 'STRAVA_SYNC_REFRESH_DAYS'])
}

export function stravaSyncRefreshDays(environment: SyncRefreshEnvironment = process.env): number {
  return refreshDays(environment, ['STRAVA_SYNC_REFRESH_DAYS', 'SYNC_REFRESH_DAYS'])
}

export function calendarRefreshStart(refreshWindowDays: number, now = Date.now()): string {
  if (!Number.isInteger(refreshWindowDays) || refreshWindowDays < 0)
    throw new Error('refreshWindowDays must be a nonnegative integer')
  return localIsoDayOffset(-refreshWindowDays, now)
}
