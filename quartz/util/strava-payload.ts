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
import { matchAppleSwims } from './apple-swim-match'
import { joinSegments, QUARTZ } from './path'
import { swimPaceSeconds, swimStrokeRate } from './swim-metrics'

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

export function enrichSwimMetrics(payload: StravaPayload, apple: AppleCache | null): void {
  const details = Object.values(payload.details).filter(
    (detail): detail is StravaActivityDetail => detail.sport === 'swim',
  )
  const matches = matchAppleSwims(
    Object.values(apple?.swims ?? {}),
    details.map(detail => ({
      id: detail.id,
      date: detail.date,
      start: detail.start,
      distanceM: detail.distanceKm * 1_000,
    })),
  )

  payload.swimTrend = []
  for (const detail of details) {
    const swim = matches.get(detail.id)
    if (swim && Object.keys(swim.strokes).length > 0) detail.strokes = swim.strokes
    const applePace = swim ? swimPaceSeconds(swim.totalM, swim.activeTimeS ?? 0) : null
    detail.swimPaceSPer100m =
      applePace ?? swimPaceSeconds(detail.distanceKm * 1_000, detail.movingTimeS)
    detail.strokeCount = swim?.strokeCount ?? null
    detail.strokeRateSpm = swim
      ? swimStrokeRate(swim.strokeCount ?? 0, swim.strokeTimeS ?? 0)
      : null
    if (detail.swimPaceSPer100m == null && detail.strokeRateSpm == null) continue
    payload.swimTrend.push({
      id: detail.id,
      date: detail.date,
      start: detail.start,
      paceSPer100m: detail.swimPaceSPer100m,
      strokeRateSpm: detail.strokeRateSpm,
    })
  }
  payload.swimTrend.sort((a, b) => a.start.localeCompare(b.start) || a.id - b.id)
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
    enrichSwimMetrics(payload, readJson<AppleCache>(appleCachePath))
    memo = { key, payload }
  }
  return memo.payload
}
