import { isRecord } from './type-guards'
import { nextSunEvent, type VisitorForecast } from './visitor-weather'

export type TrailWeather = {
  code: number
  cloud: number
  windKmh: number
  windFromDegrees: number | null
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
    windFromDegrees:
      finite(value.windFromDegrees) && value.windFromDegrees >= 0 && value.windFromDegrees <= 360
        ? value.windFromDegrees
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
            windFromDegrees: weather.windDirection,
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
