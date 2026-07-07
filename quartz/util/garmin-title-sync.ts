import {
  matchGarminActivity,
  type GarminActivityMatch,
  type GarminCache,
} from '../plugins/stores/garmin'
import {
  normalizeKind,
  type ActivityKind,
  type RawStravaActivity,
  type StravaRawCache,
} from '../plugins/stores/strava'

export interface GarminTitleSyncOptions {
  kind?: ActivityKind
  since?: string | null
  limit?: number
  ids?: ReadonlySet<string>
}

export interface GarminTitleUpdate {
  stravaId: number
  garminId: string
  garminActivityId: string
  from: string
  to: string
  startDate: string
  startDateLocal: string
  score: number
  startDiffS: number
  distanceDiffM: number | null
  durationDiffS: number | null
}

function cleanTitle(value: string | null | undefined): string {
  return (value ?? '').trim().replace(/\s+/g, ' ')
}

export function garminConnectNumericActivityId(id: string): string | null {
  const value = id.startsWith('connect:') ? id.slice('connect:'.length) : id
  return /^\d+$/.test(value) ? value : null
}

function startDay(activity: RawStravaActivity): string {
  return (activity.startDateLocal || activity.startDate).slice(0, 10)
}

function startValue(activity: RawStravaActivity): string {
  return activity.startDateLocal || activity.startDate
}

function updateFor(
  activity: RawStravaActivity,
  match: GarminActivityMatch,
): GarminTitleUpdate | null {
  const garminActivityId = garminConnectNumericActivityId(match.activity.id)
  if (!garminActivityId) return null
  const to = cleanTitle(activity.name)
  const from = cleanTitle(match.activity.name)
  if (!to || from === to) return null
  return {
    stravaId: activity.id,
    garminId: match.activity.id,
    garminActivityId,
    from,
    to,
    startDate: activity.startDate,
    startDateLocal: activity.startDateLocal,
    score: match.score,
    startDiffS: Math.round(match.startDiffMs / 1000),
    distanceDiffM: match.distanceDiffM,
    durationDiffS: match.durationDiffS,
  }
}

export function selectGarminTitleUpdates(
  strava: StravaRawCache,
  garmin: GarminCache,
  options: GarminTitleSyncOptions = {},
): GarminTitleUpdate[] {
  const kind = options.kind ?? 'bike'
  const updates: GarminTitleUpdate[] = []
  const activities = Object.values(strava.activities)
    .filter(activity => normalizeKind(activity.sportType) === kind)
    .filter(activity => !options.ids?.size || options.ids.has(String(activity.id)))
    .filter(activity => !options.since || startDay(activity) >= options.since)
    .sort((left, right) => startValue(left).localeCompare(startValue(right)))

  for (const activity of activities) {
    const match = matchGarminActivity(activity, kind, garmin)
    if (!match) continue
    const update = updateFor(activity, match)
    if (update) updates.push(update)
  }

  return options.limit && options.limit > 0 ? updates.slice(0, options.limit) : updates
}
