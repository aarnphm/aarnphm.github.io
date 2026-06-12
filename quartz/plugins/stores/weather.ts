import { isRecord, readNumber, readString } from '../../util/type-guards'

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
  temperatureC: number | null
  source: 'weatherkit'
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
  activities: Record<string, WeatherActivity>
  days: Record<string, WeatherDay>
}

export interface WeatherHour {
  forecastStart: string
  windSpeed: number
  windDirection: number | null
  windGust: number | null
  temperature: number | null
}

export interface WeatherActivityCandidate {
  activityId: number
  date: string
  start: string
  end: string
  latitude: number
  longitude: number
  durationS: number
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
  const startMs = Date.parse(candidate.start)
  const endMs = Date.parse(candidate.end)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return null

  let windTotal = 0
  let windWeight = 0
  let tempTotal = 0
  let tempWeight = 0
  let gust: number | null = null
  const directions: { degrees: number; weight: number }[] = []

  for (const hour of hours) {
    const hourStart = Date.parse(hour.forecastStart)
    if (!Number.isFinite(hourStart)) continue
    const hourEnd = hourStart + 3_600_000
    const overlap = Math.max(0, Math.min(endMs, hourEnd) - Math.max(startMs, hourStart))
    if (overlap <= 0) continue
    windTotal += hour.windSpeed * overlap
    windWeight += overlap
    if (hour.temperature != null) {
      tempTotal += hour.temperature * overlap
      tempWeight += overlap
    }
    if (hour.windGust != null) gust = Math.max(gust ?? 0, hour.windGust)
    if (hour.windDirection != null)
      directions.push({
        degrees: hour.windDirection,
        weight: overlap * Math.max(hour.windSpeed, 1),
      })
  }

  if (windWeight <= 0) return null
  const windKph = round(windTotal / windWeight)
  const windDirDeg = circularMeanDeg(directions)
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
    temperatureC: tempWeight > 0 ? round(tempTotal / tempWeight) : null,
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

function readWeatherActivity(value: unknown): WeatherActivity | null {
  if (!isRecord(value)) return null
  const activityId = readNumber(value, 'activityId')
  const date = readString(value, 'date')
  const start = readString(value, 'start')
  const end = readString(value, 'end')
  const latitude = readNumber(value, 'latitude')
  const longitude = readNumber(value, 'longitude')
  const durationS = readNumber(value, 'durationS')
  if (
    activityId == null ||
    !date ||
    !start ||
    !end ||
    latitude == null ||
    longitude == null ||
    durationS == null
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
    windKph: readNumber(value, 'windKph') ?? null,
    windDir: readString(value, 'windDir') ?? null,
    windDirDeg: readNumber(value, 'windDirDeg') ?? null,
    windGustKph: readNumber(value, 'windGustKph') ?? null,
    temperatureC: readNumber(value, 'temperatureC') ?? null,
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
    activities,
    days: summarizeWeatherDays(activities),
  }
}
