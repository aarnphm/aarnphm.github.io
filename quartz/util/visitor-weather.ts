import { isRecord } from './type-guards'

export type VisitorConditions = {
  code: number
  cloud: number
  wind: number
  temp: number
  day: boolean | null
  // Snow that fell over the twelve hours to now, cm.
  snow: number | null
}

export type VisitorSolarTimes = { timezone: string; sunrise: number[]; sunset: number[] }

export type VisitorForecast = {
  lat: number | null
  lon: number | null
  place: string | null
  country: string | null
  weather: VisitorConditions | null
  solar: VisitorSolarTimes | null
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const text = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null
const timestamps = (value: unknown): number[] =>
  Array.isArray(value) ? value.filter((item): item is number => finite(item) && item > 0) : []

function readConditions(value: unknown): VisitorConditions | null {
  if (
    !isRecord(value) ||
    !finite(value.code) ||
    !finite(value.cloud) ||
    !finite(value.wind) ||
    !finite(value.temp)
  )
    return null
  if (value.cloud < 0 || value.cloud > 1 || value.wind < 0) return null
  return {
    code: value.code,
    cloud: value.cloud,
    wind: value.wind,
    temp: value.temp,
    day: typeof value.day === 'boolean' ? value.day : null,
    snow: finite(value.snow) && value.snow >= 0 ? value.snow : null,
  }
}

// Each hourly snowfall is the hour ending at its time stamp.
function snowSince(value: Record<string, unknown>, now: unknown): number | null {
  if (!isRecord(value.hourly) || !finite(now)) return null
  const { time, snowfall } = value.hourly
  if (!Array.isArray(time) || !Array.isArray(snowfall)) return null
  let cm = 0
  for (let n = 0; n < time.length; n++) {
    const t = time[n]
    const fell = snowfall[n]
    if (finite(t) && finite(fell) && t > now - 12 * 3600 && t <= now) cm += fell
  }
  return Math.round(cm * 10) / 10
}

function readSolar(value: unknown): VisitorSolarTimes | null {
  if (
    !isRecord(value) ||
    !text(value.timezone) ||
    !Array.isArray(value.sunrise) ||
    !Array.isArray(value.sunset)
  )
    return null
  return {
    timezone: String(value.timezone),
    sunrise: timestamps(value.sunrise),
    sunset: timestamps(value.sunset),
  }
}

export function readVisitorForecast(value: unknown): VisitorForecast | null {
  if (!isRecord(value)) return null
  return {
    lat: finite(value.lat) && Math.abs(value.lat) <= 90 ? value.lat : null,
    lon: finite(value.lon) && Math.abs(value.lon) <= 180 ? value.lon : null,
    place: text(value.place),
    country: text(value.country),
    weather: readConditions(value.weather),
    solar: readSolar(value.solar),
  }
}

export function readOpenMeteo(value: unknown): Pick<VisitorForecast, 'weather' | 'solar'> {
  if (!isRecord(value) || !isRecord(value.current)) return { weather: null, solar: null }
  const current = value.current
  const weather = readConditions({
    code: current.weather_code,
    cloud: finite(current.cloud_cover) ? current.cloud_cover / 100 : null,
    wind: current.wind_speed_10m,
    temp: current.temperature_2m,
    day: current.is_day === 1 ? true : current.is_day === 0 ? false : null,
    snow: snowSince(value, current.time),
  })
  const solar = isRecord(value.daily)
    ? readSolar({
        timezone: value.timezone,
        sunrise: value.daily.sunrise,
        sunset: value.daily.sunset,
      })
    : null
  return { weather, solar }
}

export function nextSunEvent(
  solar: VisitorSolarTimes | null,
  now: number,
): { kind: 'sunrise' | 'sunset'; at: number } | null {
  if (!solar) return null
  const sunrise = solar.sunrise.find(time => time * 1000 > now)
  const sunset = solar.sunset.find(time => time * 1000 > now)
  if (sunrise === undefined && sunset === undefined) return null
  if (sunrise !== undefined && (sunset === undefined || sunrise < sunset))
    return { kind: 'sunrise', at: sunrise * 1000 }
  return sunset === undefined ? null : { kind: 'sunset', at: sunset * 1000 }
}

export function visitorConditionsLabel(code: number): string {
  if (code === 0) return 'clear'
  if (code === 1) return 'mostly clear'
  if (code === 2) return 'partly cloudy'
  if (code === 3) return 'overcast'
  if (code === 45 || code === 48) return 'fog'
  if (code >= 51 && code <= 57) return 'drizzle'
  if ((code >= 71 && code <= 77) || code === 85 || code === 86) return 'snow'
  if (code >= 95) return 'thunderstorm'
  if ((code >= 61 && code <= 67) || (code >= 80 && code <= 82)) return 'rain'
  return 'weather'
}
