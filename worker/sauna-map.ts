import { SAUNA_LOCATIONS } from '../quartz/plugins/stores/tracking'
import { mapboxStyleUrl } from '../quartz/util/mapbox-style'

export const saunaMapTarget = (params: URLSearchParams): URL | null => {
  const location = SAUNA_LOCATIONS.find(candidate => candidate.name === params.get('location'))
  const theme = params.get('theme') ?? 'light'
  if (!location || (theme !== 'light' && theme !== 'dark')) return null
  const style = mapboxStyleUrl('mono', theme).replace('mapbox://styles/', '')
  const target = new URL(
    `https://api.mapbox.com/styles/v1/${style}/static/${location.longitude},${location.latitude},15.5/240x240@2x`,
  )
  target.searchParams.set('attribution', 'false')
  return target
}

export async function handleSaunaMap(
  request: Request,
  env: { MAPBOX_API_KEY: string },
  ctx: { waitUntil(promise: Promise<unknown>): void },
): Promise<Response> {
  if (request.method !== 'GET')
    return new Response('method not allowed', { status: 405, headers: { Allow: 'GET' } })
  const url = new URL(request.url)
  const target = saunaMapTarget(url.searchParams)
  if (!target) return new Response('unknown sauna map', { status: 404 })
  if (!env.MAPBOX_API_KEY) return new Response('map unavailable', { status: 503 })
  const key = new Request(
    new URL(
      `/api/sauna-map?location=${encodeURIComponent(url.searchParams.get('location') ?? '')}&theme=${url.searchParams.get('theme') ?? 'light'}`,
      url.origin,
    ),
  )
  const cache = await caches.open('sauna-maps')
  const cached = await cache.match(key)
  if (cached) return cached
  target.searchParams.set('access_token', env.MAPBOX_API_KEY)
  const upstream = await fetch(target, {
    headers: { Referer: url.origin },
    signal: AbortSignal.timeout(15_000),
  }).catch(() => null)
  if (!upstream?.ok || !upstream.headers.get('Content-Type')?.startsWith('image/'))
    return new Response('map unavailable', { status: 502 })
  const response = new Response(upstream.body, {
    headers: {
      'Content-Type': upstream.headers.get('Content-Type') ?? 'image/png',
      'Cache-Control': 'public, max-age=86400, s-maxage=604800',
      'X-Content-Type-Options': 'nosniff',
    },
  })
  ctx.waitUntil(cache.put(key, response.clone()))
  return response
}
