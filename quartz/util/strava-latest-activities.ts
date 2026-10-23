import { statSync } from 'node:fs'
import { isTreatment, normalizeKind, type ActivityKind } from '../plugins/stores/strava'
import { readStravaCacheFileSync, stravaCachePath } from './strava-cache-file'

export const STRAVA_LATEST_ACTIVITIES_LIMIT = 5

// The card lists the three triathlon disciplines only.
export type StravaCardKind = Extract<ActivityKind, 'swim' | 'bike' | 'run'>

export interface StravaCardActivity {
  id: number
  kind: StravaCardKind
  name: string
  // Local calendar day of the start, YYYY-MM-DD.
  date: string
  distanceM: number
  movingTimeS: number
  elevationM: number
}

export interface StravaKindTotals {
  count: number
  distanceM: number
  movingTimeS: number
}

export interface StravaLatestActivities {
  athleteId: number
  syncedAt: number
  activities: StravaCardActivity[]
  year: number
  totals: Partial<Record<StravaCardKind, StravaKindTotals>>
}

type Memo = { stamp: string; value: StravaLatestActivities | null }

declare global {
  var __quartzStravaLatestActivitiesMemo: Memo | undefined
}

// The memo outlives bundle generations, so the key names the summary shape as well as the file.
// Bump it when the filter or fields change.
const MEMO_SCHEMA = 'swim-bike-run:1'

const stampOf = (path: string): string => {
  try {
    const info = statSync(path)
    return `${MEMO_SCHEMA}:${path}:${info.mtimeMs}:${info.size}`
  } catch {
    return `${MEMO_SCHEMA}:${path}:missing`
  }
}

const cardKind = (sportType: string, name: string): StravaCardKind | null => {
  if (isTreatment(sportType, name)) return null
  const kind = normalizeKind(sportType)
  return kind === 'swim' || kind === 'bike' || kind === 'run' ? kind : null
}

// The raw cache is ~150 MB and takes ~1.5 s to parse. Keep only the summary, keyed by the file
// stamp, and keep it across watch-mode bundle generations.
export function latestStravaActivities(path = stravaCachePath): StravaLatestActivities | null {
  const stamp = stampOf(path)
  const memo = globalThis.__quartzStravaLatestActivitiesMemo
  if (memo?.stamp === stamp) return memo.value

  const cache = readStravaCacheFileSync(path)
  let value: StravaLatestActivities | null = null
  if (cache) {
    const activities = Object.values(cache.activities)
      .sort((a, b) => b.startDate.localeCompare(a.startDate))
      .flatMap((activity): StravaCardActivity[] => {
        const kind = cardKind(activity.sportType, activity.name)
        if (!kind) return []
        return [
          {
            id: activity.id,
            kind,
            name: activity.name,
            date: activity.startDateLocal.slice(0, 10),
            distanceM: activity.distance,
            movingTimeS: activity.movingTime,
            elevationM: activity.totalElevationGain,
          },
        ]
      })
    const year = Number((activities[0]?.date ?? new Date(cache.lastSync).toISOString()).slice(0, 4))
    const totals: StravaLatestActivities['totals'] = {}
    for (const activity of activities) {
      if (!activity.date.startsWith(`${year}-`)) continue
      const sum = (totals[activity.kind] ??= { count: 0, distanceM: 0, movingTimeS: 0 })
      sum.count += 1
      sum.distanceM += activity.distanceM
      sum.movingTimeS += activity.movingTimeS
    }
    value = {
      athleteId: cache.athleteId,
      syncedAt: cache.lastSync,
      activities: activities.slice(0, STRAVA_LATEST_ACTIVITIES_LIMIT),
      year,
      totals,
    }
  }
  globalThis.__quartzStravaLatestActivitiesMemo = { stamp, value }
  return value
}
