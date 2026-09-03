import {
  parseGardenUvCalibrationArtifact,
  type GardenUvCalibrationArtifact,
} from '../../util/activity-uv-score'
import { isRecord, readNumber, readString } from '../../util/type-guards'

const HOUR_MS = 3_600_000

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

export interface WeatherActivity {
  activityId: number
  date: string
  start: string
  end: string
  latitude: number
  longitude: number
  durationS: number
  windKph: number | null
  windDir: string | null
  windDirDeg: number | null
  windGustKph: number | null
  averageRelativeHumidityPct: number | null
  relativeHumidityProvenance: WeatherRelativeHumidityProvenance | null
  temperatureC: number | null
  temperatureSeries?: WeatherTemperatureSample[]
  routeFingerprint?: string
  fetchedAt?: number
  routeHours?: WeatherRouteHour[]
  source: 'weatherkit'
}

export interface WeatherRouteHour {
  forecastStart: string
  overlapStart: string
  overlapEnd: string
  elapsedStartS: number
  elapsedEndS: number
  latitude: number
  longitude: number
  uvIndex: number | null
  cloudCover: number | null
  temperatureC: number | null
  windSpeedKph: number | null
  windDirectionDeg: number | null
  windGustKph: number | null
  relativeHumidity: number | null
  pressureHpa: number | null
  daylight: boolean | null
}

export interface WeatherAttribution {
  serviceName: string
  logoLightUrl: string
  logoDarkUrl: string
  legalPageUrl: string
}

export interface WeatherRelativeHumidityProvenance {
  source: 'weatherkit'
  sourceKind: 'modeled'
  samplingMethod: 'route-hour'
  inputTimestamp: string
  coveragePct: number
}

export interface WeatherTemperatureSample {
  elapsedS: number
  temperatureC: number
}

export interface WeatherDay {
  date: string
  activityCount: number
  durationS: number
  windKph: number | null
  windDir: string | null
  windDirDeg: number | null
  windGustKph: number | null
}

export interface WeatherCache {
  version?: number
  lastSync: number
  current: WeatherSnapshot | null
  attribution: WeatherAttribution | null
  uvCalibration: GardenUvCalibrationArtifact | null
  activities: Record<string, WeatherActivity>
  days: Record<string, WeatherDay>
}

export interface WeatherHour {
  forecastStart: string
  windSpeed: number | null
  windDirection: number | null
  windGust: number | null
  relativeHumidity: number | null
  temperature: number | null
  uvIndex: number | null
  cloudCover: number | null
  pressure: number | null
  daylight: boolean | null
  conditionCode: string | null
  precipitationChance: number | null
  precipitationType: string | null
}

export interface WeatherSnapshot {
  forecastStart: string
  latitude: number
  longitude: number
  temperatureC: number | null
  conditionCode: string | null
  precipitationChance: number | null
  precipitationType: string | null
  source: 'weatherkit'
}

export interface WeatherActivityCandidate {
  activityId: number
  date: string
  start: string
  end: string
  latitude: number
  longitude: number
  durationS: number
  routeFingerprint?: string
}

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
  return {
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

function readWeatherActivity(value: unknown): WeatherActivity | null {
  if (!isRecord(value)) return null
  const activityId = readNumber(value, 'activityId')
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
    !Number.isInteger(activityId) ||
    activityId <= 0 ||
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
  return {
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
}

function readWeatherAttribution(value: unknown): WeatherAttribution | null {
  if (!isRecord(value)) return null
  const serviceName = readString(value, 'serviceName')
  const logoLightUrl = readString(value, 'logoLightUrl')
  const logoDarkUrl = readString(value, 'logoDarkUrl')
  const legalPageUrl = readString(value, 'legalPageUrl')
  if (!serviceName || !logoLightUrl || !logoDarkUrl || !legalPageUrl) return null
  return { serviceName, logoLightUrl, logoDarkUrl, legalPageUrl }
}

function readWeatherSnapshot(value: unknown): WeatherSnapshot | null {
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

export function parseWeatherCache(raw: unknown): WeatherCache | null {
  if (!isRecord(raw) || !isRecord(raw.activities)) return null
  const activities: Record<string, WeatherActivity> = {}
  for (const [id, value] of Object.entries(raw.activities)) {
    const activity = readWeatherActivity(value)
    if (activity) activities[id] = activity
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
