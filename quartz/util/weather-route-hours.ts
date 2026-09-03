import { createHash } from 'node:crypto'
import type {
  WeatherActivity,
  WeatherActivityCandidate,
  WeatherHour,
} from '../plugins/stores/weather'
import { weatherActivityHasCompleteRouteHours } from '../plugins/stores/weather'

const HOUR_MS = 3_600_000

export interface RouteWeatherStream {
  timeS: number[]
  latlng: [number, number][]
  distanceM: number[]
}

export interface RouteHourQuery {
  forecastStart: string
  hourlyStart: string
  hourlyEnd: string
  overlapStart: string
  overlapEnd: string
  elapsedMidpointS: number
  latitude: number
  longitude: number
  trailingCoordinate: boolean
}

const round = (value: number, digits: number): number => {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

const floorHourMs = (value: number): number => Math.floor(value / HOUR_MS) * HOUR_MS

export function routeWeatherFingerprint(
  activityId: number,
  start: string,
  end: string,
  stream: Pick<RouteWeatherStream, 'timeS' | 'latlng'>,
): string {
  const length = Math.min(stream.timeS.length, stream.latlng.length)
  const normalized: [number, number, number][] = []
  for (let index = 0; index < length; index += 1)
    normalized.push([
      round(stream.timeS[index], 1),
      round(stream.latlng[index][0], 5),
      round(stream.latlng[index][1], 5),
    ])
  return createHash('sha256')
    .update(JSON.stringify({ activityId, start, end, route: normalized }))
    .digest('hex')
}

function nearestCoordinate(
  stream: Pick<RouteWeatherStream, 'timeS' | 'latlng'>,
  elapsedS: number,
): { latitude: number; longitude: number; trailingCoordinate: boolean } | null {
  const length = Math.min(stream.timeS.length, stream.latlng.length)
  if (length === 0) return null
  let nearest = 0
  let nearestDifference = Number.POSITIVE_INFINITY
  for (let index = 0; index < length; index += 1) {
    const difference = Math.abs(stream.timeS[index] - elapsedS)
    if (difference >= nearestDifference) continue
    nearest = index
    nearestDifference = difference
  }
  return {
    latitude: stream.latlng[nearest][0],
    longitude: stream.latlng[nearest][1],
    trailingCoordinate: elapsedS > stream.timeS[length - 1],
  }
}

export function routeHourQueries(
  candidate: WeatherActivityCandidate,
  stream: Pick<RouteWeatherStream, 'timeS' | 'latlng'>,
): RouteHourQuery[] {
  const startMs = Date.parse(candidate.start)
  const endMs = Date.parse(candidate.end)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return []
  const queries: RouteHourQuery[] = []
  for (let hourStartMs = floorHourMs(startMs); hourStartMs < endMs; hourStartMs += HOUR_MS) {
    const overlapStartMs = Math.max(startMs, hourStartMs)
    const overlapEndMs = Math.min(endMs, hourStartMs + HOUR_MS)
    if (overlapEndMs <= overlapStartMs) continue
    const midpointS = ((overlapStartMs + overlapEndMs) / 2 - startMs) / 1_000
    const coordinate = nearestCoordinate(stream, midpointS)
    if (!coordinate) continue
    queries.push({
      forecastStart: new Date(hourStartMs).toISOString(),
      hourlyStart: new Date(hourStartMs).toISOString(),
      hourlyEnd: new Date(hourStartMs + HOUR_MS).toISOString(),
      overlapStart: new Date(overlapStartMs).toISOString(),
      overlapEnd: new Date(overlapEndMs).toISOString(),
      elapsedMidpointS: midpointS,
      ...coordinate,
    })
  }
  return queries
}

export function selectRouteHour(
  hours: readonly WeatherHour[],
  forecastStart: string,
): WeatherHour | null {
  const targetMs = Date.parse(forecastStart)
  if (!Number.isFinite(targetMs)) return null
  return hours.find(hour => Date.parse(hour.forecastStart) === targetMs) ?? null
}

export function routeWeatherNeedsRefresh(
  activity: WeatherActivity | undefined,
  routeFingerprint: string,
  cacheVersion: number | undefined,
  expectedCacheVersion: number,
  activityStart: string,
  refreshWindowStartMs: number,
  force: boolean,
): boolean {
  if (force || cacheVersion !== expectedCacheVersion) return true
  if (Date.parse(activityStart) >= refreshWindowStartMs) return true
  return !weatherActivityHasCompleteRouteHours(activity, routeFingerprint)
}
