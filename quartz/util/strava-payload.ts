import { readFileSync, statSync } from 'node:fs'
import type { GarminCache } from '../plugins/stores/garmin'
import type { OuraCache } from '../plugins/stores/oura'
import type { WeatherCache } from '../plugins/stores/weather'
import { ATHLETE } from '../plugins/stores/analytics'
import { buildPayload, type StravaPayload, type StravaRawCache } from '../plugins/stores/strava'
import { joinSegments, QUARTZ } from './path'

export const stravaCachePath = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
export const ouraCachePath = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
export const garminCachePath = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
export const appleCachePath = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')
export const weatherCachePath = joinSegments(QUARTZ, '.quartz-cache', 'weather.json')

const readJson = <T>(path: string): T | null => {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as T
  } catch {
    return null
  }
}

const stamp = (path: string): number => {
  try {
    return statSync(path).mtimeMs
  } catch {
    return 0
  }
}

let memo: { key: string; payload: StravaPayload } | null = null

export function loadStravaPayloadSync(since?: string): StravaPayload {
  const key = `${since ?? ''}:${stamp(stravaCachePath)}:${stamp(ouraCachePath)}:${stamp(garminCachePath)}:${stamp(weatherCachePath)}`
  if (memo?.key !== key) {
    memo = {
      key,
      payload: buildPayload(
        readJson<StravaRawCache>(stravaCachePath),
        readJson<OuraCache>(ouraCachePath),
        readJson<GarminCache>(garminCachePath),
        since,
        readJson<WeatherCache>(weatherCachePath),
        ATHLETE.ftp,
      ),
    }
  }
  return memo.payload
}
