import type { TrailSeason, TrailWeather } from '../../../util/visitor-trail-weather'
import type { Almanac } from '../../scripts/404-almanac'
import { visitorPrecipitation, wetFall } from '../../../util/visitor-precipitation'
import { trailWinter } from '../winter'

export function decoration(
  weather: TrailWeather | null,
  al?: Almanac,
  seasonOverride?: TrailSeason | null,
) {
  const precipitation = visitorPrecipitation(weather?.code)
  const rain = wetFall(precipitation.fall) ? precipitation.heavy : 0
  const snow = precipitation.fall === 'snow' ? 1 : 0
  const { fog, storm } = precipitation
  const snowCover = trailWinter(weather, false, al, seasonOverride).cover
  const sun =
    weather?.day && weather.code >= 0 && weather.code <= 2 && !rain && !snow
      ? Math.max(0, (0.85 - weather.cloud) / 0.85)
      : 0
  // A valley always has a breeze. Without a reading, or with a calm or variable one, gusts still
  // drift through from the west at a light strength.
  const angle = ((weather?.windFromDegrees ?? 270) * Math.PI) / 180
  const wind = weather ? Math.min(1, weather.windKmh / 55) : 0
  const gust = Math.max(
    0.22,
    weather ? Math.min(1, Math.max(weather.windKmh, weather.gustKmh ?? weather.windKmh) / 55) : 0,
  )
  // Meteorological directions describe where the wind comes from; SVG y grows southward.
  const dx = -Math.sin(angle)
  const dy = Math.cos(angle)
  const mode = !weather
    ? 'quiet'
    : rain
      ? 'rain'
      : snow
        ? 'snow'
        : weather?.day === false
          ? 'night'
          : sun
            ? 'sun'
            : 'cloud'
  return { rain, snow, fog, storm, snowCover, sun, wind, gust, dx, dy, mode }
}
