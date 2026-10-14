import type { VisitorForecast } from './visitor-weather'
import { almanac, guessSite } from '../components/scripts/404-almanac'
import { isRecord } from './type-guards'
import { visitorPrecipitation } from './visitor-precipitation'

export type VisitorDaylight = {
  at: number
  phase: 'dawn' | 'day' | 'dusk' | 'night'
  altitude: number
  bearing: number
  daylight: number
  twilight: number
  sunlight: number
  source: 'forecast-cell' | 'clock-estimate'
  dark: boolean
}

type DaylightForecast = Pick<VisitorForecast, 'lat' | 'lon' | 'weather'>
type DaylightEvents = {
  'visitor-daylight:request': CustomEvent<void>
  'visitor-daylight:change': CustomEvent<VisitorDaylight | null>
}

declare global {
  interface CustomEventMap extends DaylightEvents {}
  interface DocumentEventMap extends DaylightEvents {}
}

const requestEvent = 'visitor-daylight:request'
const changeEvent = 'visitor-daylight:change'
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const fraction = (value: unknown): value is number => finite(value) && value >= 0 && value <= 1

export function sampleVisitorDaylight(
  forecast: DaylightForecast | null,
  now = Date.now(),
): VisitorDaylight {
  const lat = forecast?.lat
  const lon = forecast?.lon
  const located = finite(lat) && Math.abs(lat) <= 90 && finite(lon) && Math.abs(lon) <= 180
  const site = located ? { lat, lon, place: null, weather: null } : guessSite()
  const sky = almanac(new Date(now), site)
  const rising = almanac(new Date(now + 5 * 60_000), site).sun.alt >= sky.sun.alt
  const phase = sky.sun.alt <= -6 ? 'night' : sky.sun.alt >= 6 ? 'day' : rising ? 'dawn' : 'dusk'
  const cloud = Math.max(0, Math.min(1, forecast?.weather?.cloud ?? 0.3))
  const precipitation = visitorPrecipitation(forecast?.weather?.code)
  return {
    at: now,
    phase,
    altitude: sky.sun.alt,
    bearing: sky.sun.bearing,
    daylight: sky.day,
    twilight: sky.dusk,
    sunlight: sky.day * (1 - cloud) * (1 - precipitation.strength * 0.85),
    source: located ? 'forecast-cell' : 'clock-estimate',
    dark: sky.sun.alt < -4,
  }
}

export function readVisitorDaylight(value: unknown): VisitorDaylight | null {
  if (
    !isRecord(value) ||
    !finite(value.at) ||
    !finite(value.altitude) ||
    Math.abs(value.altitude) > 90 ||
    !finite(value.bearing) ||
    Math.abs(value.bearing) > 180 ||
    !fraction(value.daylight) ||
    !fraction(value.twilight) ||
    !fraction(value.sunlight) ||
    typeof value.dark !== 'boolean' ||
    (value.phase !== 'dawn' &&
      value.phase !== 'day' &&
      value.phase !== 'dusk' &&
      value.phase !== 'night') ||
    (value.source !== 'forecast-cell' && value.source !== 'clock-estimate')
  )
    return null
  return {
    at: value.at,
    phase: value.phase,
    altitude: value.altitude,
    bearing: value.bearing,
    daylight: value.daylight,
    twilight: value.twilight,
    sunlight: value.sunlight,
    source: value.source,
    dark: value.dark,
  }
}

export function createVisitorDaylightSource(target: Document, signal: AbortSignal) {
  let latest: VisitorDaylight | null = sampleVisitorDaylight(null)
  let signature = JSON.stringify(latest)
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
    publish(forecast: DaylightForecast | null, now = Date.now()) {
      if (signal.aborted) return
      const current = sampleVisitorDaylight(forecast, now)
      const value = JSON.stringify(current)
      if (value === signature) return
      latest = current
      signature = value
      dispatch()
    },
  }
}

export function subscribeVisitorDaylight(
  target: Document,
  receive: (daylight: VisitorDaylight | null) => void,
  signal: AbortSignal,
) {
  target.addEventListener(
    changeEvent,
    event => {
      if ('detail' in event) receive(readVisitorDaylight(event.detail))
    },
    { signal },
  )
  target.dispatchEvent(new CustomEvent<void>(requestEvent))
}
