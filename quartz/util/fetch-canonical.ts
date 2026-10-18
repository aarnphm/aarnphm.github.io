const canonicalPattern = /<link rel="canonical" href="([^"]*)">/

// Callers want the rendered page. Without an explicit preference the worker's Accept
// negotiation treats agent user agents (incl. the Claude app browser) as markdown clients.
const htmlAccept = 'text/html, */*;q=0.8'

const HTML_CACHE_TTL_MS = 15_000
const MAX_CACHE_ENTRIES = 32

interface CachedHtmlEntry {
  html: string
  url: string
  status: number
  statusText: string
  headers: [string, string][]
  redirected: boolean
  expiresAt: number
}

interface InFlightHtmlRequest {
  controller: AbortController
  subscribers: Set<symbol>
  promise: Promise<Response>
}

const htmlCache = new Map<string, CachedHtmlEntry>()
const inFlight = new Map<string, InFlightHtmlRequest>()

function normalizeCacheKey(url: URL): string {
  const clean = new URL(url.toString())
  clean.hash = ''
  return clean.toString()
}

function isCacheableGet(init?: RequestInit): boolean {
  if (!init) return true
  if (init.method && init.method.toUpperCase() !== 'GET') return false
  if (init.body != null) return false
  if (init.headers) {
    const h = new Headers(init.headers)
    for (const key of h.keys()) {
      if (key.toLowerCase() !== 'accept') return false
    }
  }
  return true
}

function toHtmlResponse(entry: CachedHtmlEntry): Response {
  const res = new Response(entry.html, {
    status: entry.status,
    statusText: entry.statusText,
    headers: entry.headers,
  })
  Object.defineProperty(res, 'url', { value: entry.url, configurable: true })
  Object.defineProperty(res, 'redirected', { value: entry.redirected, configurable: true })
  res.text = () => Promise.resolve(entry.html)
  res.clone = () => toHtmlResponse(entry)
  return res
}

function storeCachedHtml(key: string, entry: CachedHtmlEntry): void {
  if (htmlCache.has(key)) htmlCache.delete(key)
  else if (htmlCache.size >= MAX_CACHE_ENTRIES) {
    const oldest = htmlCache.keys().next().value
    if (oldest !== undefined) htmlCache.delete(oldest)
  }
  htmlCache.set(key, entry)
}

function createAbortError(): DOMException {
  return new DOMException('The operation was aborted.', 'AbortError')
}

async function performFetchCanonical(
  url: URL,
  headers: Headers,
  init: RequestInit | undefined,
  signal: AbortSignal | undefined,
  cacheKey: string | null,
): Promise<Response> {
  const response = await fetch(url, { ...init, headers, ...(signal ? { signal } : {}) })
  if (!response.headers.get('content-type')?.startsWith('text/html')) return response

  const html = await response.text()
  const canonical = html.match(canonicalPattern)?.[1]
  if (!canonical) {
    const entry: CachedHtmlEntry = {
      html,
      url: response.url || url.toString(),
      status: response.status,
      statusText: response.statusText,
      headers: Array.from(response.headers.entries()),
      redirected: response.redirected,
      expiresAt: Date.now() + HTML_CACHE_TTL_MS,
    }
    if (cacheKey && response.ok) storeCachedHtml(cacheKey, entry)
    return toHtmlResponse(entry)
  }

  const targetUrl = new URL(canonical, url)
  const followed = await fetch(targetUrl, { headers, ...(signal ? { signal } : {}) })
  if (!followed.headers.get('content-type')?.startsWith('text/html')) return followed

  const followedHtml = await followed.text()
  const entry: CachedHtmlEntry = {
    html: followedHtml,
    url: followed.url || targetUrl.toString(),
    status: followed.status,
    statusText: followed.statusText,
    headers: Array.from(followed.headers.entries()),
    redirected: followed.redirected,
    expiresAt: Date.now() + HTML_CACHE_TTL_MS,
  }
  if (cacheKey && followed.ok) {
    storeCachedHtml(cacheKey, entry)
    storeCachedHtml(normalizeCacheKey(targetUrl), entry)
  }
  return toHtmlResponse(entry)
}

export async function fetchCanonical(url: URL, init?: RequestInit): Promise<Response> {
  const callerSignal = init?.signal ?? undefined
  if (callerSignal?.aborted) {
    throw callerSignal.reason instanceof Error ? callerSignal.reason : createAbortError()
  }

  const headers = new Headers(init?.headers)
  if (!headers.has('Accept')) headers.set('Accept', htmlAccept)

  if (!isCacheableGet(init)) {
    return performFetchCanonical(url, headers, init, callerSignal, null)
  }

  const key = normalizeCacheKey(url)
  const now = Date.now()
  const cached = htmlCache.get(key)
  if (cached) {
    if (cached.expiresAt > now) return toHtmlResponse(cached)
    htmlCache.delete(key)
  }

  let shared = inFlight.get(key)
  if (!shared) {
    const controller = new AbortController()
    const subscribers = new Set<symbol>()
    const promise = performFetchCanonical(url, headers, undefined, controller.signal, key).finally(
      () => {
        if (inFlight.get(key)?.controller === controller) {
          inFlight.delete(key)
        }
      },
    )
    shared = { controller, subscribers, promise }
    inFlight.set(key, shared)
  }

  const token = Symbol()
  shared.subscribers.add(token)

  if (!callerSignal) {
    const res = await shared.promise
    return res.clone()
  }

  return new Promise<Response>((resolve, reject) => {
    const onAbort = () => {
      shared!.subscribers.delete(token)
      if (shared!.subscribers.size === 0) {
        shared!.controller.abort()
      }
      reject(createAbortError())
    }
    callerSignal.addEventListener('abort', onAbort, { once: true })
    shared!.promise.then(
      res => {
        callerSignal.removeEventListener('abort', onAbort)
        if (callerSignal.aborted) {
          reject(createAbortError())
          return
        }
        resolve(res.clone())
      },
      err => {
        callerSignal.removeEventListener('abort', onAbort)
        reject(err)
      },
    )
  })
}

export function prefetchCanonical(url: URL): void {
  const key = normalizeCacheKey(url)
  const cached = htmlCache.get(key)
  if (cached && cached.expiresAt > Date.now()) return
  if (inFlight.has(key)) return
  void fetchCanonical(url).catch(() => {})
}

export function invalidateCanonicalCache(url?: URL | string): void {
  if (!url) {
    htmlCache.clear()
    return
  }
  const parsed = typeof url === 'string' ? new URL(url, 'http://localhost') : url
  htmlCache.delete(normalizeCacheKey(parsed))
}
