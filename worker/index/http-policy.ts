import type { Env } from './environment'

export function withHeaders(
  response: Response,
  newHeaders: Record<string, string | null>,
): Response {
  if (Object.keys(newHeaders).length === 0) return response
  const headers = new Headers(response.headers)
  for (const [key, val] of Object.entries(newHeaders)) {
    if (val === null) headers.delete(key)
    else headers.set(key, val)
  }
  return new Response(response.body, {
    headers,
    status: response.status,
    statusText: response.statusText,
  })
}

export function mergeVary(response: Response, ...values: string[]): string {
  const current = response.headers.get('Vary')
  if (current?.trim() === '*') return '*'
  const vary = new Map<string, string>()
  for (const value of current?.split(',') ?? []) {
    const trimmed = value.trim()
    if (trimmed) vary.set(trimmed.toLowerCase(), trimmed)
  }
  for (const value of values) vary.set(value.toLowerCase(), value)
  return [...vary.values()].join(', ')
}

export function withHeadersFromSource(
  response: Response,
  source: Response,
  headers: string[],
): Response {
  const map: Record<string, string | null> = {}
  for (const h of headers) map[h] = source.headers.get(h)
  return withHeaders(response, map)
}

function getAllowedOrigin(env: Env, request: Request): string | null {
  const origin = request.headers.get('Origin')
  if (!origin) return null
  try {
    const o = new URL(origin)
    const base = env.PUBLIC_BASE_URL ? new URL(env.PUBLIC_BASE_URL) : null
    if (base && o.origin === base.origin) return origin
    if (o.hostname === 'localhost' || o.hostname === '127.0.0.1') return origin
    if (o.hostname.endsWith('.workers.dev')) return origin
  } catch {}
  return null
}

export function buildCorsHeaders(env: Env, request: Request): Record<string, string> {
  const origin = getAllowedOrigin(env, request)
  const headers: Record<string, string> = {
    'Access-Control-Allow-Methods': 'GET,OPTIONS,PATCH,DELETE,POST,PUT',
    'Access-Control-Allow-Headers':
      'Authorization, X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version',
  }
  if (origin) {
    headers['Access-Control-Allow-Origin'] = origin
    headers['Access-Control-Allow-Credentials'] = 'true'
  }
  return headers
}
