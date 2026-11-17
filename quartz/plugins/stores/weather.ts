import type {
  WeatherActivity,
  WeatherActivityCandidate,
  WeatherCache,
  WeatherDay,
  WeatherHour,
  WeatherRouteHour,
  WeatherSnapshot,
  WeatherTemperatureSample,
} from './weather/types'
import { parseGardenUvCalibrationArtifact } from '../../util/activity-uv-score'
import { type SurfaceCurrentEstimate } from '../../util/surface-current'
import { isRecord, readNumber } from '../../util/type-guards'
import {
  matchingSurfaceCurrent,
  readWeatherActivity,
  readWeatherAttribution,
  readWeatherSnapshot,
} from './weather/read'

const COMPASS = [
  'N',
  'NNE',
  'NE',
  'ENE',
  'E',
  'ESE',
  'SE',
  'SSE',
  'S',
  'SSW',
  'SW',
  'WSW',
  'W',
  'WNW',
  'NW',
  'NNW',
]

export function compassFromDegrees(degrees: number | null): string | null {
  if (degrees == null || !Number.isFinite(degrees)) return null
  const normalized = ((degrees % 360) + 360) % 360
  return COMPASS[Math.round(normalized / 22.5) % COMPASS.length]
}

function round(value: number, dp = 0): number {
  const f = 10 ** dp
  return Math.round(value * f) / f
}

export function weatherSnapshotFromHours(
  location: { latitude: number; longitude: number },
  hours: readonly WeatherHour[],
  atMs: number,
): WeatherSnapshot | null {
  if (!Number.isFinite(atMs)) return null
  const nearest = hours
    .map(hour => ({ hour, distance: Math.abs(Date.parse(hour.forecastStart) - atMs) }))
    .filter(candidate => Number.isFinite(candidate.distance))
    .sort((left, right) => left.distance - right.distance)[0]?.hour
  if (!nearest) return null
  return {
    forecastStart: nearest.forecastStart,
    latitude: round(location.latitude, 5),
    longitude: round(location.longitude, 5),
    temperatureC: nearest.temperature == null ? null : round(nearest.temperature, 1),
    conditionCode: nearest.conditionCode,
    precipitationChance:
      nearest.precipitationChance == null
        ? null
        : round(Math.min(1, Math.max(0, nearest.precipitationChance)), 2),
    precipitationType: nearest.precipitationType,
    source: 'weatherkit',
  }
}

function circularMeanDeg(values: { degrees: number; weight: number }[]): number | null {
  let x = 0
  let y = 0
  for (const value of values) {
    if (!Number.isFinite(value.degrees) || value.weight <= 0) continue
    const radians = (value.degrees * Math.PI) / 180
    x += Math.cos(radians) * value.weight
    y += Math.sin(radians) * value.weight
  }
  if (x === 0 && y === 0) return null
  return round(((Math.atan2(y, x) * 180) / Math.PI + 360) % 360)
}

export function weatherActivityFromHours(
  candidate: WeatherActivityCandidate,
  hours: WeatherHour[],
): WeatherActivity | null {
  const routeHours = hours.flatMap(hour => {
    const routeHour = weatherRouteHourFromForecast(candidate, hour, {
      latitude: candidate.latitude,
      longitude: candidate.longitude,
    })
    return routeHour ? [routeHour] : []
  })
  return weatherActivityFromRouteHours(candidate, routeHours)
}

export function weatherRouteHourFromForecast(
  candidate: WeatherActivityCandidate,
  hour: WeatherHour,
  location: { latitude: number; longitude: number },
): WeatherRouteHour | null {
  const startMs = Date.parse(candidate.start)
  const endMs = Date.parse(candidate.end)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return null
  const hourStartMs = Date.parse(hour.forecastStart)
  if (!Number.isFinite(hourStartMs)) return null
  const overlapStartMs = Math.max(startMs, hourStartMs)
  const overlapEndMs = Math.min(endMs, hourStartMs + 3_600_000)
  if (overlapEndMs <= overlapStartMs) return null
  return {
    forecastStart: hour.forecastStart,
    overlapStart: new Date(overlapStartMs).toISOString(),
    overlapEnd: new Date(overlapEndMs).toISOString(),
    elapsedStartS: (overlapStartMs - startMs) / 1_000,
    elapsedEndS: (overlapEndMs - startMs) / 1_000,
    latitude: location.latitude,
    longitude: location.longitude,
    uvIndex: hour.uvIndex,
    cloudCover: hour.cloudCover,
    temperatureC: hour.temperature,
    windSpeedKph: hour.windSpeed,
    windDirectionDeg: hour.windDirection,
    windGustKph: hour.windGust,
    relativeHumidity: hour.relativeHumidity,
    pressureHpa: hour.pressure,
    daylight: hour.daylight,
  }
}

export function weatherActivityFromRouteHours(
  candidate: WeatherActivityCandidate,
  routeHours: readonly WeatherRouteHour[],
  fetchedAt?: number,
  previousSurfaceCurrent?: SurfaceCurrentEstimate | null,
): WeatherActivity | null {
  const startMs = Date.parse(candidate.start)
  const endMs = Date.parse(candidate.end)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return null

  let windTotal = 0
  let windWeight = 0
  let tempTotal = 0
  let tempWeight = 0
  let relativeHumidityTotal = 0
  let relativeHumidityWeight = 0
  let overlapWeight = 0
  let gust: number | null = null
  const directions: { degrees: number; weight: number }[] = []
  const temperatureSeries: WeatherTemperatureSample[] = []

  const sortedRouteHours = routeHours
    .filter(
      hour =>
        Number.isFinite(hour.elapsedStartS) &&
        Number.isFinite(hour.elapsedEndS) &&
        hour.elapsedStartS >= 0 &&
        hour.elapsedEndS > hour.elapsedStartS &&
        hour.elapsedEndS <= candidate.durationS + 1,
    )
    .slice()
    .sort((left, right) => left.elapsedStartS - right.elapsedStartS)

  for (const hour of sortedRouteHours) {
    const overlap = (hour.elapsedEndS - hour.elapsedStartS) * 1_000
    overlapWeight += overlap
    if (hour.windSpeedKph != null && Number.isFinite(hour.windSpeedKph)) {
      windTotal += hour.windSpeedKph * overlap
      windWeight += overlap
    }
    if (hour.temperatureC != null) {
      tempTotal += hour.temperatureC * overlap
      tempWeight += overlap
      const sample = {
        elapsedS: round(hour.elapsedStartS),
        temperatureC: round(hour.temperatureC, 1),
      }
      const previous = temperatureSeries[temperatureSeries.length - 1]
      if (previous?.elapsedS === sample.elapsedS)
        temperatureSeries[temperatureSeries.length - 1] = sample
      else temperatureSeries.push(sample)
    }
    if (
      hour.relativeHumidity != null &&
      Number.isFinite(hour.relativeHumidity) &&
      hour.relativeHumidity >= 0 &&
      hour.relativeHumidity <= 1
    ) {
      relativeHumidityTotal += hour.relativeHumidity * overlap
      relativeHumidityWeight += overlap
    }
    if (hour.windGustKph != null) gust = Math.max(gust ?? 0, hour.windGustKph)
    if (hour.windDirectionDeg != null && hour.windSpeedKph != null)
      directions.push({
        degrees: hour.windDirectionDeg,
        weight: overlap * Math.max(hour.windSpeedKph, 1),
      })
  }

  if (overlapWeight <= 0) return null
  const finalTemperature = temperatureSeries[temperatureSeries.length - 1]
  if (finalTemperature && finalTemperature.elapsedS < candidate.durationS)
    temperatureSeries.push({
      elapsedS: candidate.durationS,
      temperatureC: finalTemperature.temperatureC,
    })
  const windKph = windWeight > 0 ? round(windTotal / windWeight) : null
  const windDirDeg = circularMeanDeg(directions)
  const relativeHumidityCoveragePct = round(
    Math.min(1, relativeHumidityWeight / (endMs - startMs)) * 100,
  )
  const activity: WeatherActivity = {
    activityId: candidate.activityId,
    date: candidate.date,
    start: candidate.start,
    end: candidate.end,
    latitude: round(candidate.latitude, 5),
    longitude: round(candidate.longitude, 5),
    durationS: candidate.durationS,
    windKph,
    windDir: compassFromDegrees(windDirDeg),
    windDirDeg,
    windGustKph: gust == null ? null : round(gust),
    averageRelativeHumidityPct:
      relativeHumidityWeight > 0
        ? round((relativeHumidityTotal / relativeHumidityWeight) * 100)
        : null,
    relativeHumidityProvenance: {
      source: 'weatherkit',
      sourceKind: 'modeled',
      samplingMethod: 'route-hour',
      inputTimestamp: candidate.start,
      coveragePct: relativeHumidityCoveragePct,
    },
    temperatureC: tempWeight > 0 ? round(tempTotal / tempWeight) : null,
    temperatureSeries,
    ...(candidate.routeFingerprint ? { routeFingerprint: candidate.routeFingerprint } : {}),
    ...(fetchedAt == null ? {} : { fetchedAt }),
    routeHours: sortedRouteHours.map(hour => ({ ...hour })),
    source: 'weatherkit',
  }
  const surfaceCurrent = matchingSurfaceCurrent(previousSurfaceCurrent, activity)
  if (surfaceCurrent) activity.surfaceCurrent = surfaceCurrent
  return activity
}

export function summarizeWeatherDays(
  activities: Record<string, WeatherActivity>,
): Record<string, WeatherDay> {
  const groups = new Map<string, WeatherActivity[]>()
  for (const activity of Object.values(activities)) {
    const group = groups.get(activity.date) ?? []
    group.push(activity)
    groups.set(activity.date, group)
  }

  const days: Record<string, WeatherDay> = {}
  for (const [date, group] of [...groups].sort((a, b) => a[0].localeCompare(b[0]))) {
    let windTotal = 0
    let windWeight = 0
    let durationS = 0
    let gust: number | null = null
    const directions: { degrees: number; weight: number }[] = []
    for (const activity of group) {
      const weight = Math.max(1, activity.durationS)
      durationS += activity.durationS
      if (activity.windKph != null) {
        windTotal += activity.windKph * weight
        windWeight += weight
      }
      if (activity.windGustKph != null) gust = Math.max(gust ?? 0, activity.windGustKph)
      if (activity.windDirDeg != null && activity.windKph != null)
        directions.push({
          degrees: activity.windDirDeg,
          weight: weight * Math.max(activity.windKph, 1),
        })
    }
    const windDirDeg = circularMeanDeg(directions)
    days[date] = {
      date,
      activityCount: group.length,
      durationS,
      windKph: windWeight > 0 ? round(windTotal / windWeight) : null,
      windDir: compassFromDegrees(windDirDeg),
      windDirDeg,
      windGustKph: gust,
    }
  }
  return days
}

export function parseWeatherCache(raw: unknown): WeatherCache | null {
  if (!isRecord(raw) || !isRecord(raw.activities)) return null
  const activities: Record<string, WeatherActivity> = {}
  for (const [id, value] of Object.entries(raw.activities)) {
    const activity = readWeatherActivity(value)
    if (activity && activity.activityId === id) activities[id] = activity
  }
  return {
    version: readNumber(raw, 'version'),
    lastSync: readNumber(raw, 'lastSync') ?? 0,
    current: readWeatherSnapshot(raw.current),
    attribution: readWeatherAttribution(raw.attribution),
    uvCalibration: parseGardenUvCalibrationArtifact(raw.uvCalibration),
    activities,
    days: summarizeWeatherDays(activities),
  }
}
export { weatherActivityHasCompleteRouteHours } from './weather/read'
export type {
  WeatherActivity,
  WeatherActivityCandidate,
  WeatherAttribution,
  WeatherCache,
  WeatherDay,
  WeatherHour,
  WeatherRelativeHumidityProvenance,
  WeatherRouteHour,
  WeatherSnapshot,
  WeatherTemperatureSample,
} from './weather/types'
