type LeanEnv = { LEAN_VERIFY_ORIGIN?: string; LEAN_VERIFY_TOKEN?: string }
type CfCacheStorage = CacheStorage & { readonly default?: Cache }

const MAX_CODE_LENGTH = 100_000
const MAX_IMPORTS_LENGTH = 256
const UPSTREAM_TIMEOUT_MS = 45_000
const CACHE_CONTROL_VALUE = 'public, max-age=86400, s-maxage=604800'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

function jsonError(message: string, status: number, extra?: Record<string, string>): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { ...JSON_HEADERS, ...extra },
  })
}

function defaultCache(): Cache | undefined {
  if (typeof caches === 'undefined' || caches === null) return undefined
  const storage = caches as CfCacheStorage
  return storage.default && typeof storage.default.match === 'function'
    ? storage.default
    : undefined
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}

function isValidJson(text: string): boolean {
  if (!text.trim()) return false
  try {
    JSON.parse(text)
    return true
  } catch {
    return false
  }
}

function localDevFallbackOrigin(request: Request): string | undefined {
  try {
    const url = new URL(request.url)
    if (url.hostname !== 'localhost' && url.hostname !== '127.0.0.1') return undefined
    const port = Number(url.port)
    if (!Number.isInteger(port) || port <= 0 || port > 65533) return undefined
    return `http://127.0.0.1:${port + 2}/api/lean/verify`
  } catch {
    return undefined
  }
}

export async function handleLeanVerify(request: Request, env: LeanEnv): Promise<Response> {
  if (request.method !== 'POST') {
    return jsonError('method not allowed', 405, { Allow: 'POST, OPTIONS' })
  }

  const configuredOrigin = env.LEAN_VERIFY_ORIGIN?.trim()
  const fallbackOrigin = configuredOrigin ? undefined : localDevFallbackOrigin(request)
  const origin = configuredOrigin || fallbackOrigin
  if (!origin) return jsonError('lean verification is not configured', 503)

  let body: { code?: unknown; imports?: unknown }
  try {
    body = (await request.json()) as typeof body
  } catch {
    return jsonError('invalid json', 400)
  }

  const code = typeof body.code === 'string' ? body.code : null
  if (!code) return jsonError('code required', 400)
  if (code.length > MAX_CODE_LENGTH) return jsonError('code too large', 413)
  const imports =
    typeof body.imports === 'string' && body.imports.length <= MAX_IMPORTS_LENGTH
      ? body.imports
      : 'mathlib'

  const hash = await sha256Hex(`${imports}\0${code}`)
  const cacheKey = new Request(`https://lean-verify.internal/${hash}`, { method: 'GET' })
  const cache = defaultCache()
  if (cache) {
    try {
      const cached = await cache.match(cacheKey)
      if (cached) {
        const headers = new Headers(cached.headers)
        headers.set('Content-Type', 'application/json')
        headers.set('Cache-Control', cached.headers.get('Cache-Control') ?? CACHE_CONTROL_VALUE)
        headers.set('X-Lean-Verify-Cache', 'HIT')
        return new Response(cached.body, { status: 200, headers })
      }
    } catch {
      // Proceed to upstream if cache read fails
    }
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS)
  try {
    const headers: Record<string, string> = { ...JSON_HEADERS, Accept: 'application/json' }
    if (env.LEAN_VERIFY_TOKEN) headers.Authorization = `Bearer ${env.LEAN_VERIFY_TOKEN}`
    const upstream = await fetch(origin, {
      method: 'POST',
      headers,
      body: JSON.stringify({ code, imports }),
      signal: controller.signal,
    })
    const text = await upstream.text()
    if (!upstream.ok) {
      if (!configuredOrigin && fallbackOrigin) {
        return jsonError('lean verification is not configured', 503)
      }
      return new Response(text, { status: 502, headers: JSON_HEADERS })
    }
    const cacheable = isValidJson(text)
    const responseHeaders: Record<string, string> = {
      ...JSON_HEADERS,
      ...(cacheable ? { 'Cache-Control': CACHE_CONTROL_VALUE, 'X-Lean-Verify-Cache': 'MISS' } : {}),
    }
    const response = new Response(text, { status: 200, headers: responseHeaders })
    if (cache && cacheable) {
      try {
        await cache.put(cacheKey, response.clone())
      } catch {
        // Ignore cache write errors
      }
    }
    return response
  } catch (error) {
    if (!configuredOrigin && fallbackOrigin) {
      return jsonError('lean verification is not configured', 503)
    }
    const aborted = error instanceof Error && error.name === 'AbortError'
    return jsonError(aborted ? 'verification timed out' : 'verification upstream error', 504)
  } finally {
    clearTimeout(timeout)
  }
}
