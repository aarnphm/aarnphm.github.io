import { stravaId } from '../../../util/strava-id'
import {
  parseSurfaceCurrentEstimate,
  type SurfaceCurrentEstimate,
} from '../../../util/surface-current'
import { isRecord, readNumber, readString } from '../../../util/type-guards'
const HOUR_MS = 3_600_000
import type {
  WeatherActivity,
  WeatherAttribution,
  WeatherRelativeHumidityProvenance,
  WeatherRouteHour,
  WeatherSnapshot,
  WeatherTemperatureSample,
} from './types'

export const matchingSurfaceCurrent = (
  value: unknown,
  activity: WeatherActivity,
): SurfaceCurrentEstimate | null => {
  const current = parseSurfaceCurrentEstimate(value)
  return current &&
    current.activityId === activity.activityId &&
    current.routeFingerprint === activity.routeFingerprint &&
    Date.parse(current.start) === Date.parse(activity.start) &&
    Date.parse(current.end) === Date.parse(activity.end) &&
    current.summary.elapsedDurationS === activity.durationS
    ? current
    : null
}

function readRelativeHumidityProvenance(value: unknown): WeatherRelativeHumidityProvenance | null {
  if (!isRecord(value)) return null
  const inputTimestamp = readString(value, 'inputTimestamp')
  const coveragePct = readNumber(value, 'coveragePct')
  if (
    value.source !== 'weatherkit' ||
    value.sourceKind !== 'modeled' ||
    value.samplingMethod !== 'route-hour' ||
    !inputTimestamp ||
    !Number.isFinite(Date.parse(inputTimestamp)) ||
    coveragePct == null ||
    coveragePct < 0 ||
    coveragePct > 100
  )
    return null
  return {
    source: 'weatherkit',
    sourceKind: 'modeled',
    samplingMethod: 'route-hour',
    inputTimestamp,
    coveragePct,
  }
}

const nullableNumber = (
  value: unknown,
  minimum: number,
  maximum: number,
): number | null | undefined => {
  if (value === null) return null
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined
  return value >= minimum && value <= maximum ? value : undefined
}

function readWeatherRouteHour(
  value: unknown,
  durationS: number,
  activityStartMs: number,
): WeatherRouteHour | null {
  if (!isRecord(value)) return null
  const forecastStart = readString(value, 'forecastStart')
  const overlapStart = readString(value, 'overlapStart')
  const overlapEnd = readString(value, 'overlapEnd')
  const elapsedStartS = readNumber(value, 'elapsedStartS')
  const elapsedEndS = readNumber(value, 'elapsedEndS')
  const latitude = readNumber(value, 'latitude')
  const longitude = readNumber(value, 'longitude')
  const uvIndex = nullableNumber(value.uvIndex, 0, 30)
  const cloudCover = nullableNumber(value.cloudCover, 0, 1)
  const temperatureC = nullableNumber(value.temperatureC, -100, 100)
  const windSpeedKph = nullableNumber(value.windSpeedKph, 0, 500)
  const windDirectionDeg = nullableNumber(value.windDirectionDeg, 0, 360)
  const windGustKph = nullableNumber(value.windGustKph, 0, 500)
  const relativeHumidity = nullableNumber(value.relativeHumidity, 0, 1)
  const pressureHpa = nullableNumber(value.pressureHpa, 0, 2_000)
  const daylight =
    value.daylight === null || typeof value.daylight === 'boolean' ? value.daylight : undefined
  const forecastStartMs = forecastStart ? Date.parse(forecastStart) : Number.NaN
  const overlapStartMs = overlapStart ? Date.parse(overlapStart) : Number.NaN
  const overlapEndMs = overlapEnd ? Date.parse(overlapEnd) : Number.NaN
  if (
    !forecastStart ||
    !overlapStart ||
    !overlapEnd ||
    !Number.isFinite(forecastStartMs) ||
    !Number.isFinite(overlapStartMs) ||
    !Number.isFinite(overlapEndMs) ||
    forecastStartMs % HOUR_MS !== 0 ||
    elapsedStartS == null ||
    elapsedEndS == null ||
    elapsedStartS < 0 ||
    elapsedEndS <= elapsedStartS ||
    elapsedEndS > durationS + 1 ||
    Math.abs(overlapStartMs - (activityStartMs + elapsedStartS * 1_000)) > 1_000 ||
    Math.abs(overlapEndMs - (activityStartMs + elapsedEndS * 1_000)) > 1_000 ||
    overlapStartMs < forecastStartMs ||
    overlapEndMs > forecastStartMs + HOUR_MS ||
    latitude == null ||
    latitude < -90 ||
    latitude > 90 ||
    longitude == null ||
    longitude < -180 ||
    longitude > 180 ||
    uvIndex === undefined ||
    cloudCover === undefined ||
    temperatureC === undefined ||
    windSpeedKph === undefined ||
    windDirectionDeg === undefined ||
    windGustKph === undefined ||
    relativeHumidity === undefined ||
    pressureHpa === undefined ||
    daylight === undefined
  )
    return null
  return {
    forecastStart,
    overlapStart,
    overlapEnd,
    elapsedStartS,
    elapsedEndS,
    latitude,
    longitude,
    uvIndex,
    cloudCover,
    temperatureC,
    windSpeedKph,
    windDirectionDeg,
    windGustKph,
    relativeHumidity,
    pressureHpa,
    daylight,
  }
}

export function weatherActivityHasCompleteRouteHours(
  activity: WeatherActivity | undefined,
  routeFingerprint: string,
): boolean {
  if (
    !activity ||
    activity.routeFingerprint !== routeFingerprint ||
    typeof activity.fetchedAt !== 'number' ||
    !Number.isFinite(activity.fetchedAt) ||
    !activity.routeHours?.length ||
    !/^[a-f0-9]{64}$/.test(routeFingerprint)
  )
    return false
  const startMs = Date.parse(activity.start)
  if (!Number.isFinite(startMs) || activity.durationS <= 0) return false
  const endMs = startMs + activity.durationS * 1_000
  const hours = activity.routeHours
    .slice()
    .sort((left, right) => left.elapsedStartS - right.elapsedStartS)
  let hourStartMs = Math.floor(startMs / HOUR_MS) * HOUR_MS
  let coveredUntilS = 0
  let index = 0
  while (hourStartMs < endMs) {
    const hour = hours[index]
    if (!hour || Date.parse(hour.forecastStart) !== hourStartMs) return false
    const expectedStartS = (Math.max(startMs, hourStartMs) - startMs) / 1_000
    const expectedEndS = (Math.min(endMs, hourStartMs + HOUR_MS) - startMs) / 1_000
    if (
      Math.abs(hour.elapsedStartS - expectedStartS) > 1 ||
      Math.abs(hour.elapsedEndS - expectedEndS) > 1 ||
      hour.elapsedStartS - coveredUntilS > 1
    )
      return false
    coveredUntilS = Math.max(coveredUntilS, hour.elapsedEndS)
    hourStartMs += HOUR_MS
    index += 1
  }
  return index === hours.length && activity.durationS - coveredUntilS <= 1
}

export function readWeatherActivity(value: unknown): WeatherActivity | null {
  if (!isRecord(value)) return null
  const activityId = stravaId(value.activityId)
  const date = readString(value, 'date')
  const start = readString(value, 'start')
  const end = readString(value, 'end')
  const latitude = readNumber(value, 'latitude')
  const longitude = readNumber(value, 'longitude')
  const durationS = readNumber(value, 'durationS')
  const startMs = start ? Date.parse(start) : Number.NaN
  const endMs = end ? Date.parse(end) : Number.NaN
  if (
    activityId == null ||
    !date ||
    !/^\d{4}-\d{2}-\d{2}$/.test(date) ||
    !start ||
    !end ||
    latitude == null ||
    latitude < -90 ||
    latitude > 90 ||
    longitude == null ||
    longitude < -180 ||
    longitude > 180 ||
    durationS == null ||
    durationS <= 0 ||
    !Number.isFinite(startMs) ||
    !Number.isFinite(endMs) ||
    Math.abs(endMs - startMs - durationS * 1_000) > 1_000 ||
    value.source !== 'weatherkit'
  )
    return null
  const temperatureSeries: WeatherTemperatureSample[] = []
  if (Array.isArray(value.temperatureSeries))
    for (const sample of value.temperatureSeries) {
      if (!isRecord(sample)) continue
      const elapsedS = readNumber(sample, 'elapsedS')
      const temperatureC = readNumber(sample, 'temperatureC')
      if (
        elapsedS == null ||
        elapsedS < 0 ||
        elapsedS > durationS ||
        temperatureC == null ||
        !Number.isFinite(temperatureC) ||
        temperatureC < -100 ||
        temperatureC > 100
      )
        continue
      temperatureSeries.push({ elapsedS, temperatureC })
    }
  temperatureSeries.sort((a, b) => a.elapsedS - b.elapsedS)
  const relativeHumidityProvenance = readRelativeHumidityProvenance(
    value.relativeHumidityProvenance,
  )
  const averageRelativeHumidityPct = readNumber(value, 'averageRelativeHumidityPct')
  const routeHours = Array.isArray(value.routeHours)
    ? value.routeHours
        .map(hour => readWeatherRouteHour(hour, durationS, startMs))
        .filter(hour => hour !== null)
        .sort((left, right) => left.elapsedStartS - right.elapsedStartS)
    : []
  const routeFingerprint = readString(value, 'routeFingerprint')
  const fetchedAt = readNumber(value, 'fetchedAt')
  const windKph = nullableNumber(value.windKph, 0, 500)
  const windDirDeg = nullableNumber(value.windDirDeg, 0, 360)
  const windGustKph = nullableNumber(value.windGustKph, 0, 500)
  const temperatureC = nullableNumber(value.temperatureC, -100, 100)
  if (
    windKph === undefined ||
    windDirDeg === undefined ||
    windGustKph === undefined ||
    temperatureC === undefined
  )
    return null
  const activity: WeatherActivity = {
    activityId,
    date,
    start,
    end,
    latitude,
    longitude,
    durationS,
    windKph,
    windDir: readString(value, 'windDir') ?? null,
    windDirDeg,
    windGustKph,
    averageRelativeHumidityPct:
      relativeHumidityProvenance != null &&
      averageRelativeHumidityPct != null &&
      averageRelativeHumidityPct >= 0 &&
      averageRelativeHumidityPct <= 100
        ? averageRelativeHumidityPct
        : null,
    relativeHumidityProvenance,
    temperatureC,
    temperatureSeries,
    ...(routeFingerprint ? { routeFingerprint } : {}),
    ...(fetchedAt != null && Number.isFinite(fetchedAt) ? { fetchedAt } : {}),
    ...(routeHours.length > 0 ? { routeHours } : {}),
    source: 'weatherkit',
  }
  const surfaceCurrent = matchingSurfaceCurrent(value.surfaceCurrent, activity)
  if (surfaceCurrent) activity.surfaceCurrent = surfaceCurrent
  return activity
}

export function readWeatherAttribution(value: unknown): WeatherAttribution | null {
  if (!isRecord(value)) return null
  const serviceName = readString(value, 'serviceName')
  const logoLightUrl = readString(value, 'logoLightUrl')
  const logoDarkUrl = readString(value, 'logoDarkUrl')
  const legalPageUrl = readString(value, 'legalPageUrl')
  if (!serviceName || !logoLightUrl || !logoDarkUrl || !legalPageUrl) return null
  return { serviceName, logoLightUrl, logoDarkUrl, legalPageUrl }
}

export function readWeatherSnapshot(value: unknown): WeatherSnapshot | null {
  if (!isRecord(value)) return null
  const forecastStart = readString(value, 'forecastStart')
  const latitude = readNumber(value, 'latitude')
  const longitude = readNumber(value, 'longitude')
  if (!forecastStart || latitude == null || longitude == null) return null
  const precipitationChance = readNumber(value, 'precipitationChance')
  return {
    forecastStart,
    latitude,
    longitude,
    temperatureC: readNumber(value, 'temperatureC') ?? null,
    conditionCode: readString(value, 'conditionCode') ?? null,
    precipitationChance:
      precipitationChance == null ? null : Math.min(1, Math.max(0, precipitationChance)),
    precipitationType: readString(value, 'precipitationType') ?? null,
    source: 'weatherkit',
  }
}
