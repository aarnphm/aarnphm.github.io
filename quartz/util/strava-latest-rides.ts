import { statSync } from 'node:fs'
import { normalizeSport } from '../plugins/stores/strava'
import { readStravaCacheFileSync } from './strava-cache-file'
import { stravaCachePath } from './strava-payload'

export const STRAVA_LATEST_RIDES_LIMIT = 5

export interface StravaRide {
  id: number
  name: string
  // Local calendar day of the start, YYYY-MM-DD.
  date: string
  distanceM: number
  movingTimeS: number
  elevationM: number
}

export interface StravaRideTotals {
  year: number
  count: number
  distanceM: number
  movingTimeS: number
  elevationM: number
}

export interface StravaLatestRides {
  athleteId: number
  syncedAt: number
  rides: StravaRide[]
  totals: StravaRideTotals
}

type Memo = { stamp: string; value: StravaLatestRides | null }

declare global {
  var __quartzStravaLatestRidesMemo: Memo | undefined
}

const stampOf = (path: string): string => {
  try {
    const info = statSync(path)
    return `${path}:${info.mtimeMs}:${info.size}`
  } catch {
    return `${path}:missing`
  }
}

// The raw cache is ~150 MB and takes ~1.5 s to parse. Keep only the summary, keyed by the file
// stamp, and keep it across watch-mode bundle generations.
export function latestStravaRides(path = stravaCachePath): StravaLatestRides | null {
  const stamp = stampOf(path)
  const memo = globalThis.__quartzStravaLatestRidesMemo
  if (memo?.stamp === stamp) return memo.value

  const cache = readStravaCacheFileSync(path)
  let value: StravaLatestRides | null = null
  if (cache) {
    const rides = Object.values(cache.activities)
      .filter(activity => normalizeSport(activity.sportType) === 'bike')
      .sort((a, b) => b.startDate.localeCompare(a.startDate))
      .map(activity => ({
        id: activity.id,
        name: activity.name,
        date: activity.startDateLocal.slice(0, 10),
        distanceM: activity.distance,
        movingTimeS: activity.movingTime,
        elevationM: activity.totalElevationGain,
      }))
    const year = Number((rides[0]?.date ?? new Date(cache.lastSync).toISOString()).slice(0, 4))
    const totals: StravaRideTotals = { year, count: 0, distanceM: 0, movingTimeS: 0, elevationM: 0 }
    for (const ride of rides) {
      if (!ride.date.startsWith(`${year}-`)) continue
      totals.count += 1
      totals.distanceM += ride.distanceM
      totals.movingTimeS += ride.movingTimeS
      totals.elevationM += ride.elevationM
    }
    value = {
      athleteId: cache.athleteId,
      syncedAt: cache.lastSync,
      rides: rides.slice(0, STRAVA_LATEST_RIDES_LIMIT),
      totals,
    }
  }
  globalThis.__quartzStravaLatestRidesMemo = { stamp, value }
  return value
}
