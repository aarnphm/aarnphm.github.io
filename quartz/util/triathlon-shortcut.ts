import { triathlonDaySlug, triathlonOnSlugFromShortcutPath } from './triathlon-date-route'
import { isRecord } from './type-guards'

const TRIATHLON_PREFIX = '/triathlon'
const SHORTCUT_HOST_PREFIX = 't.'
const STRAVA_ACTIVITY_PATH = /^\/activities\/([1-9]\d*)\/?$/

export const STRAVA_ACTIVITY_INDEX_KIND = 'strava-activity-index-v1'
export const STRAVA_ACTIVITY_INDEX_PATH = '/static/strava-activity-index.json'

export interface StravaActivityIndex {
  kind: typeof STRAVA_ACTIVITY_INDEX_KIND
  activities: Record<string, string>
}

function canonicalShortcutBase(baseUrl: string, source: URL): URL {
  const target = new URL(baseUrl)
  if (target.hostname === source.hostname && target.hostname.startsWith(SHORTCUT_HOST_PREFIX)) {
    target.hostname = target.hostname.slice(SHORTCUT_HOST_PREFIX.length)
  }
  return target
}

function triathlonShortcutPathname(pathname: string): string {
  if (pathname === '/' || pathname === TRIATHLON_PREFIX || pathname === `${TRIATHLON_PREFIX}/`) {
    return TRIATHLON_PREFIX
  }
  if (pathname.startsWith(`${TRIATHLON_PREFIX}/`)) return pathname
  const temporalSlug = triathlonOnSlugFromShortcutPath(pathname)
  if (temporalSlug) return `/${temporalSlug}`
  return `${TRIATHLON_PREFIX}${pathname.startsWith('/') ? pathname : `/${pathname}`}`
}

export function buildStravaActivityIndex(
  details: Readonly<Record<string, { id: number; date: string }>>,
): StravaActivityIndex {
  const activities: Record<string, string> = {}
  for (const [id, detail] of Object.entries(details)) {
    if (!/^[1-9]\d*$/.test(id) || String(detail.id) !== id)
      throw new Error(`Strava activity index has invalid ID ${id}`)
    if (!triathlonDaySlug(detail.date))
      throw new Error(`Strava activity ${id} has invalid date ${detail.date}`)
    activities[id] = detail.date
  }
  return { kind: STRAVA_ACTIVITY_INDEX_KIND, activities }
}

export function isStravaActivityIndex(value: unknown): value is StravaActivityIndex {
  return (
    isRecord(value) &&
    value.kind === STRAVA_ACTIVITY_INDEX_KIND &&
    isRecord(value.activities) &&
    Object.entries(value.activities).every(
      ([id, date]) => /^[1-9]\d*$/.test(id) && typeof date === 'string' && !!triathlonDaySlug(date),
    )
  )
}

export function stravaActivityIdFromShortcutPath(pathname: string): string | null {
  return STRAVA_ACTIVITY_PATH.exec(pathname)?.[1] ?? null
}

export function triathlonActivityShortcutRedirectUrl(
  baseUrl: string,
  requestUrl: string | URL,
  activityDates: Readonly<Record<string, string>>,
): string | null {
  const source = requestUrl instanceof URL ? requestUrl : new URL(requestUrl)
  const activityId = stravaActivityIdFromShortcutPath(source.pathname)
  if (!activityId) return null
  const slug = triathlonDaySlug(activityDates[activityId] ?? '')
  if (!slug) return null
  const target = canonicalShortcutBase(baseUrl, source)
  target.pathname = `/${slug}`
  target.search = source.search
  target.hash = source.hash
  return target.toString()
}

export function triathlonShortcutRedirectUrl(
  baseUrl: string,
  requestUrl: string | URL,
  isDocument: boolean,
): string {
  const source = requestUrl instanceof URL ? requestUrl : new URL(requestUrl)
  const target = canonicalShortcutBase(baseUrl, source)
  target.pathname = isDocument ? triathlonShortcutPathname(source.pathname) : source.pathname
  target.search = source.search
  target.hash = source.hash
  return target.toString()
}
