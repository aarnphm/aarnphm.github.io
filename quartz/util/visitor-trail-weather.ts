import { isRecord } from './type-guards'
import { nextSunEvent, type VisitorForecast } from './visitor-weather'

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
}

type TrailWeatherEvents = {
  'visitor-trail-weather:request': CustomEvent<void>
  'visitor-trail-weather:change': CustomEvent<TrailWeather | null>
}

declare global {
  interface CustomEventMap extends TrailWeatherEvents {}
  interface DocumentEventMap extends TrailWeatherEvents {}
}

const requestEvent = 'visitor-trail-weather:request'
const changeEvent = 'visitor-trail-weather:change'
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

function calendarSeason(latitude: number | null, now: number): TrailSeason | null {
  if (latitude === null || Math.abs(latitude) < 23.5) return null
  const month = new Date(now).getMonth()
  const seasons: readonly TrailSeason[] = ['winter', 'spring', 'summer', 'autumn']
  return seasons[Math.floor(((month + 1 + (latitude < 0 ? 6 : 0)) % 12) / 3)] ?? null
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
            season: calendarSeason(forecast?.lat ?? null, now),
            day: next ? next.kind === 'sunset' : weather.day,
          }
        : null
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
