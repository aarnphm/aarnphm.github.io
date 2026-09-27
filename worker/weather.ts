// The weather over the 404 page's visitor. Cloudflare geolocates the request; the position is
// rounded to a tenth-degree cell (about 11 km north to south) before it goes anywhere, so Open-Meteo
// only learns the district, sees the worker's address instead of the visitor's, and everyone in the
// cell shares one forecast from the cache. The cell comes back with the conditions so the page can
// place the sun and moon; without a position or a forecast the page falls back on its own guess.
// A tenth of a degree is about one cell of ECMWF's 9 km global model, so the rounding stays inside
// the forecast's own blur. Half a degree put Toronto's cell centre 12 km out on Lake Ontario, whose
// water held the forecast near 16 °C from dawn to afternoon while the city ran 11 to 24 °C.

type Geo = {
  city?: string | null
  region?: string | null
  country?: string | null
  latitude?: string | null
  longitude?: string | null
}
type Current = {
  weather_code?: number
  cloud_cover?: number
  wind_speed_10m?: number
  temperature_2m?: number
}

const FRESH = 900
const cell = (v: number) => Math.round(v * 10) / 10

export async function handleWeather(
  request: Request,
  ctx: { waitUntil(promise: Promise<unknown>): void },
): Promise<Response> {
  if (request.method !== 'GET')
    return new Response('method not allowed', { status: 405, headers: { Allow: 'GET' } })
  const geo = (request as Request & { cf?: Geo }).cf
  const place =
    [geo?.city, geo?.region, geo?.country]
      .map(value => (typeof value === 'string' ? value.trim() : ''))
      .find(Boolean) ?? null
  const lat = geo?.latitude ? cell(Number(geo.latitude)) : NaN
  const lon = geo?.longitude ? cell(Number(geo.longitude)) : NaN
  const headers = { 'Cache-Control': `private, max-age=${FRESH}` }
  if (!Number.isFinite(lat) || !Number.isFinite(lon))
    return Response.json({ place, weather: null }, { headers })

  const key = new Request(`https://weather.cache/${lat},${lon}`)
  const cache = await caches.open('weather')
  const cached = await cache.match(key)
  if (cached) return Response.json({ lat, lon, place, weather: await cached.json() }, { headers })

  const target = new URL('https://api.open-meteo.com/v1/forecast')
  target.searchParams.set('latitude', String(lat))
  target.searchParams.set('longitude', String(lon))
  target.searchParams.set('current', 'weather_code,cloud_cover,wind_speed_10m,temperature_2m')
  const upstream = await fetch(target, { signal: AbortSignal.timeout(4000) }).catch(() => null)
  const current = upstream?.ok
    ? ((await upstream.json().catch(() => null)) as { current?: Current } | null)?.current
    : undefined
  const { weather_code, cloud_cover, wind_speed_10m, temperature_2m } = current ?? {}
  if (
    ![weather_code, cloud_cover, wind_speed_10m, temperature_2m].every(
      v => typeof v === 'number' && Number.isFinite(v),
    )
  )
    return Response.json({ lat, lon, place, weather: null }, { headers })
  const weather = {
    code: weather_code!,
    cloud: cloud_cover! / 100,
    wind: wind_speed_10m!,
    temp: temperature_2m!,
  }
  ctx.waitUntil(
    cache.put(
      key,
      Response.json(weather, { headers: { 'Cache-Control': `public, max-age=${FRESH}` } }),
    ),
  )
  return Response.json({ lat, lon, place, weather }, { headers })
}
