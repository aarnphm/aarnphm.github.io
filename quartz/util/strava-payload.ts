import { readFileSync, statSync } from 'node:fs'
import type { AppleCache } from '../plugins/stores/apple'
import type { GarminCache } from '../plugins/stores/garmin'
import type { OuraCache } from '../plugins/stores/oura'
import type { WeatherCache } from '../plugins/stores/weather'
import { ATHLETE } from '../plugins/stores/analytics'
import {
  buildPayload,
  type StravaActivityDetail,
  type StravaPayload,
  type StravaRawCache,
} from '../plugins/stores/strava'
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

export function enrichSwimStrokes(payload: StravaPayload, apple: AppleCache | null): void {
  if (!apple?.swims) return
  const mainSwim = new Map<string, StravaActivityDetail>()
  for (const d of Object.values(payload.details)) {
    if (d.sport !== 'swim') continue
    const cur = mainSwim.get(d.date)
    if (!cur || d.distanceKm > cur.distanceKm) mainSwim.set(d.date, d)
  }
  for (const [date, d] of mainSwim) {
    const sw = apple.swims[date]
    if (sw && Object.keys(sw.strokes).length > 0) d.strokes = sw.strokes
  }
}

let memo: { key: string; payload: StravaPayload } | null = null

export function loadStravaPayloadSync(since?: string): StravaPayload {
  const key = `${since ?? ''}:${stamp(stravaCachePath)}:${stamp(ouraCachePath)}:${stamp(garminCachePath)}:${stamp(weatherCachePath)}:${stamp(appleCachePath)}`
  if (memo?.key !== key) {
    const payload = buildPayload(
      readJson<StravaRawCache>(stravaCachePath),
      readJson<OuraCache>(ouraCachePath),
      readJson<GarminCache>(garminCachePath),
      since,
      readJson<WeatherCache>(weatherCachePath),
      ATHLETE.ftp,
    )
    enrichSwimStrokes(payload, readJson<AppleCache>(appleCachePath))
    memo = { key, payload }
  }
  return memo.payload
}
