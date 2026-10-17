import { almanac, guessSite, readSky, type Site, type Sky } from '../components/scripts/404-almanac'
import { isRecord } from './type-guards'
import { readVisitorDaylightCell } from './visitor-daylight'
import { nextSunEvent, type VisitorForecast } from './visitor-weather'
import { seasonBucket } from './visitor-world'

export type TrailSeason = 'spring' | 'summer' | 'autumn' | 'winter'

export type TrailWeather = {
  code: number
  cloud: number
  windKmh: number
  gustKmh: number | null
  windFromDegrees: number | null
  temperatureC: number | null
  snowCm: number | null
  season: TrailSeason | null
  day: boolean | null
  // The visitor's forecast cell, which places the sun and moon over the trail.
  lat: number | null
  lon: number | null
}

type TrailWeatherEvents = {
  'visitor-trail-weather:request': CustomEvent<void>
  'visitor-trail-weather:change': CustomEvent<TrailWeather | null>
}

declare global {
  interface CustomEventMap extends TrailWeatherEvents {}
  interface DocumentEventMap extends TrailWeatherEvents {}
}

// Where the trail stands: the forecast's place, else the worker's cell on <html>, else a guess
// from the time zone.
export function trailSite(weather: TrailWeather | null, document: Document): Site {
  if (weather && weather.lat !== null && weather.lon !== null)
    return { lat: weather.lat, lon: weather.lon, place: null, weather: null }
  const cell = readVisitorDaylightCell(document)
  return cell ? { lat: cell.lat, lon: cell.lon, place: null, weather: null } : guessSite()
}

// The shared sky reading of a trail forecast; an unknown temperature stays unknown.
export function trailSky(weather: TrailWeather | null): Sky {
  if (!weather) return readSky(null)
  const { code, cloud, windKmh: wind, temperatureC, snowCm: snow } = weather
  return { ...readSky({ code, cloud, wind, temp: temperatureC ?? 0, snow }), temp: temperatureC }
}

const requestEvent = 'visitor-trail-weather:request'
const changeEvent = 'visitor-trail-weather:change'
const storageKey = 'visitor-weather:last'
const maxAgeMs = 15 * 60_000
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

function calendarSeason(latitude: number | null, now: number, longitude = 0): TrailSeason | null {
  if (latitude === null || Math.abs(latitude) < 23.5) return null
  const sky = almanac(new Date(now), { lat: latitude, lon: longitude, place: null, weather: null })
  return seasonBucket(sky.winterDay).season
}

function readTrailWeather(value: unknown): TrailWeather | null {
  if (
    !isRecord(value) ||
    !finite(value.code) ||
    !finite(value.cloud) ||
    value.cloud < 0 ||
    value.cloud > 1 ||
    !finite(value.windKmh) ||
    value.windKmh < 0
  )
    return null
  return {
    code: value.code,
    cloud: value.cloud,
    windKmh: value.windKmh,
    gustKmh: finite(value.gustKmh) && value.gustKmh >= 0 ? value.gustKmh : null,
    windFromDegrees:
      finite(value.windFromDegrees) && value.windFromDegrees >= 0 && value.windFromDegrees <= 360
        ? value.windFromDegrees
        : null,
    temperatureC: finite(value.temperatureC) ? value.temperatureC : null,
    snowCm: finite(value.snowCm) && value.snowCm >= 0 ? value.snowCm : null,
    season:
      value.season === 'spring' ||
      value.season === 'summer' ||
      value.season === 'autumn' ||
      value.season === 'winter'
        ? value.season
        : null,
    day: typeof value.day === 'boolean' ? value.day : null,
    lat: finite(value.lat) && Math.abs(value.lat) <= 90 ? value.lat : null,
    lon: finite(value.lon) && Math.abs(value.lon) <= 180 ? value.lon : null,
  }
}

export function readStoredTrailWeather(
  view: Window | null | undefined,
  now = Date.now(),
): TrailWeather | null {
  try {
    const raw = view?.sessionStorage.getItem(storageKey)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    if (!isRecord(parsed)) return null
    const at = finite(parsed.at) ? parsed.at : null
    if (at === null || now - at > maxAgeMs || now < at - maxAgeMs) return null
    return readTrailWeather(parsed.weather)
  } catch {
    return null
  }
}

export function createVisitorTrailWeatherSource(target: Document, signal: AbortSignal) {
  let latest: TrailWeather | null = null
  let signature = 'null'
  const dispatch = () => target.dispatchEvent(new CustomEvent(changeEvent, { detail: latest }))
  target.addEventListener(requestEvent, dispatch, { signal })
  signal.addEventListener(
    'abort',
    () => {
      latest = null
      dispatch()
    },
    { once: true },
  )
  return {
    publish(forecast: VisitorForecast | null, now = Date.now()) {
      if (signal.aborted) return
      const weather = forecast?.weather
      const next = nextSunEvent(forecast?.solar ?? null, now)
      const current: TrailWeather | null = weather
        ? {
            code: weather.code,
            cloud: weather.cloud,
            windKmh: weather.wind,
            gustKmh: weather.gust,
            windFromDegrees: weather.windDirection,
            temperatureC: weather.temp,
            snowCm: weather.snow,
            season: calendarSeason(forecast?.lat ?? null, now, forecast?.lon ?? 0),
            day: next ? next.kind === 'sunset' : weather.day,
            lat: forecast?.lat ?? null,
            lon: forecast?.lon ?? null,
          }
        : null
      if (current) {
        try {
          target.defaultView?.sessionStorage.setItem(
            storageKey,
            JSON.stringify({ at: now, weather: current }),
          )
        } catch {}
      }
      const value = JSON.stringify(current)
      if (value === signature) return
      latest = current
      signature = value
      dispatch()
    },
  }
}

export function subscribeVisitorTrailWeather(
  target: Document,
  receive: (weather: TrailWeather | null) => void,
  signal: AbortSignal,
) {
  target.addEventListener(
    changeEvent,
    event => {
      if ('detail' in event) receive(readTrailWeather(event.detail))
    },
    { signal },
  )
  // The iframe and page have separate bundles. Request the page's latest local conditions.
  target.dispatchEvent(new CustomEvent<void>(requestEvent))
}
