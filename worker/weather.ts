// Weather for the homepage and 404 landscape. Cloudflare geolocates the request; the position is
// rounded to a tenth-degree cell (about 11 km north to south) before it goes anywhere, so Open-Meteo
// only learns the district, sees the worker's address instead of the visitor's, and everyone in the
// cell shares one forecast from the cache. The cell comes back with the conditions so the page can
// place the sun and moon; without a position or a forecast the page falls back on its own guess.
// A tenth of a degree is about one cell of ECMWF's 9 km global model, so the rounding stays inside
// the forecast's own blur. Half a degree put Toronto's cell centre 12 km out on Lake Ontario, whose
// water held the forecast near 16 °C from dawn to afternoon while the city ran 11 to 24 °C.

import { isRecord } from '../quartz/util/type-guards'
import { readOpenMeteo, readVisitorForecast } from '../quartz/util/visitor-weather'

const FRESH = 900
const cell = (v: number) => Math.round(v * 10) / 10
const coordinate = (value: unknown, limit: number): number | null => {
  if (typeof value !== 'string' || !value.trim()) return null
  const number = Number(value)
  return Number.isFinite(number) && Math.abs(number) <= limit ? cell(number) : null
}

export async function handleWeather(
  request: Request,
  ctx: { waitUntil(promise: Promise<unknown>): void },
): Promise<Response> {
  if (request.method !== 'GET')
    return new Response('method not allowed', { status: 405, headers: { Allow: 'GET' } })
  const geo = 'cf' in request && isRecord(request.cf) ? request.cf : null
  const place =
    [geo?.city, geo?.region, geo?.country]
      .map(value => (typeof value === 'string' ? value.trim() : ''))
      .find(Boolean) ?? null
  const country = typeof geo?.country === 'string' ? geo.country : null
  const lat = coordinate(geo?.latitude, 90)
  const lon = coordinate(geo?.longitude, 180)
  const headers = { 'Cache-Control': `private, max-age=${FRESH}` }
  if (lat === null || lon === null)
    return Response.json({ place, country, weather: null, solar: null }, { headers })

  const key = new Request(`https://weather.cache/v3/${lat},${lon}`)
  const cache = await caches.open('weather')
  const cached = await cache.match(key)
  const stored = cached ? readVisitorForecast(await cached.json().catch(() => null)) : null
  if (stored?.weather)
    return Response.json(
      { lat, lon, place, country, weather: stored.weather, solar: stored.solar },
      { headers },
    )

  const target = new URL('https://api.open-meteo.com/v1/forecast')
  target.searchParams.set('latitude', String(lat))
  target.searchParams.set('longitude', String(lon))
  target.searchParams.set(
    'current',
    'weather_code,cloud_cover,wind_speed_10m,temperature_2m,is_day',
  )
  target.searchParams.set('hourly', 'snowfall')
  target.searchParams.set('daily', 'sunrise,sunset')
  target.searchParams.set('timezone', 'auto')
  target.searchParams.set('timeformat', 'unixtime')
  target.searchParams.set('past_days', '1')
  target.searchParams.set('forecast_days', '2')
  const upstream = await fetch(target, { signal: AbortSignal.timeout(4000) }).catch(() => null)
  const { weather, solar } = readOpenMeteo(
    upstream?.ok ? await upstream.json().catch(() => null) : null,
  )
  if (!weather)
    return Response.json({ lat, lon, place, country, weather: null, solar: null }, { headers })
  ctx.waitUntil(
    cache.put(
      key,
      Response.json(
        { weather, solar },
        { headers: { 'Cache-Control': `public, max-age=${FRESH}` } },
      ),
    ),
  )
  return Response.json({ lat, lon, place, country, weather, solar }, { headers })
}
