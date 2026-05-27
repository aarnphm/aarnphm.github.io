const ARENA_EMBED_USER_AGENT =
  'Mozilla/5.0 (compatible; AarnphmGardenArena/1.0; +https://aarnphm.xyz/arena)'
const ARENA_EMBED_HTML_CSP = [
  "default-src 'none'",
  'img-src https: data:',
  "style-src 'unsafe-inline' https:",
  'font-src https: data:',
  'media-src https:',
  "frame-ancestors 'self'",
  "base-uri 'none'",
  "form-action 'none'",
].join('; ')

type ArenaEmbedCapabilityMode = 'iframe' | 'fetch' | 'disabled'
type ArenaEmbedUrlUse = 'document' | 'resource'
type ArenaEmbedBlockReason =
  | 'ok'
  | 'invalid-url'
  | 'unsupported-protocol'
  | 'blocked-host'
  | 'blocked-port'
  | 'userinfo'
  | 'upstream-error'
  | 'fetch-error'
  | 'non-html'
  | 'frame-ancestors-none'
  | 'frame-ancestors-empty'
  | 'frame-ancestors-mismatch'
  | 'x-frame-options-deny'
  | 'x-frame-options-sameorigin'

interface ArenaEmbedCapability {
  mode: ArenaEmbedCapabilityMode
  finalUrl?: string
  reason: ArenaEmbedBlockReason
}

type ArenaEmbedTarget =
  | { ok: true; url: URL }
  | { ok: false; status: number; reason: ArenaEmbedBlockReason; message: string }

interface ArenaHtmlRewriterElement {
  readonly tagName: string
  readonly attributes: IterableIterator<string[]>
  getAttribute(name: string): string | null
  setAttribute(name: string, value: string): ArenaHtmlRewriterElement
  removeAttribute(name: string): ArenaHtmlRewriterElement
  remove(): ArenaHtmlRewriterElement
}

interface ArenaHtmlRewriterHandlers {
  element(element: ArenaHtmlRewriterElement): void
}

interface ArenaHtmlRewriter {
  on(selector: string, handlers: ArenaHtmlRewriterHandlers): ArenaHtmlRewriter
  transform(response: Response): Response
}

declare const HTMLRewriter: { new (): ArenaHtmlRewriter }

export function validateArenaEmbedTarget(rawUrl: string | null): ArenaEmbedTarget {
  if (!rawUrl) {
    return { ok: false, status: 400, reason: 'invalid-url', message: 'missing url parameter' }
  }

  let url: URL
  try {
    url = new URL(rawUrl)
  } catch {
    return { ok: false, status: 400, reason: 'invalid-url', message: 'invalid url' }
  }

  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    return {
      ok: false,
      status: 400,
      reason: 'unsupported-protocol',
      message: 'unsupported protocol',
    }
  }

  if (url.username || url.password) {
    return {
      ok: false,
      status: 400,
      reason: 'userinfo',
      message: 'url credentials are unsupported',
    }
  }

  if (url.port && url.port !== '80' && url.port !== '443') {
    return { ok: false, status: 400, reason: 'blocked-port', message: 'unsupported port' }
  }

  if (isBlockedArenaEmbedHostname(url.hostname)) {
    return { ok: false, status: 400, reason: 'blocked-host', message: 'unsupported host' }
  }

  return { ok: true, url }
}

function isBlockedArenaEmbedHostname(rawHostname: string): boolean {
  const hostname = rawHostname
    .trim()
    .toLowerCase()
    .replace(/^\[|\]$/g, '')
    .replace(/\.$/, '')
  if (hostname.length === 0) return true
  if (
    hostname === 'localhost' ||
    hostname.endsWith('.localhost') ||
    hostname.endsWith('.local') ||
    hostname.endsWith('.internal') ||
    hostname.endsWith('.home.arpa') ||
    hostname === 'metadata.google.internal'
  ) {
    return true
  }

  if (hostname === '::1' || hostname === '0:0:0:0:0:0:0:1' || hostname === '::') return true
  if (hostname.startsWith('fc') || hostname.startsWith('fd') || hostname.startsWith('fe80:')) {
    return true
  }

  const parts = hostname.split('.')
  if (parts.length !== 4) return false
  const octets = parts.map(part => Number.parseInt(part, 10))
  if (octets.some(octet => !Number.isInteger(octet) || octet < 0 || octet > 255)) return false

  const [a, b] = octets
  if (a === 0 || a === 10 || a === 127) return true
  if (a === 100 && b >= 64 && b <= 127) return true
  if (a === 169 && b === 254) return true
  if (a === 172 && b >= 16 && b <= 31) return true
  if (a === 192 && b === 168) return true
  if (a === 198 && (b === 18 || b === 19)) return true
  if (a >= 224) return true

  return false
}

function arenaJsonResponse(body: ArenaEmbedCapability, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': status === 200 ? 'public, s-maxage=3600, max-age=600' : 'no-store',
    },
  })
}

function arenaTextResponse(message: string, status: number): Response {
  return new Response(message, {
    status,
    headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' },
  })
}

function isHtmlResponse(headers: Headers): boolean {
  return (headers.get('Content-Type') ?? '').split(';')[0].trim().toLowerCase() === 'text/html'
}

function arenaEmbedFetchInit(method: 'GET' | 'HEAD'): RequestInit {
  return {
    method,
    headers: {
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'User-Agent': ARENA_EMBED_USER_AGENT,
    },
    redirect: 'follow',
    cf: { cacheTtl: 3600, cacheEverything: true },
  }
}

async function fetchArenaEmbedHeaders(targetUrl: URL): Promise<Response> {
  const head = await fetch(targetUrl.toString(), arenaEmbedFetchInit('HEAD'))
  if (head.status !== 405 && head.status !== 403) return head

  const fallback = await fetch(targetUrl.toString(), arenaEmbedFetchInit('GET'))
  if (fallback.body) await fallback.body.cancel()
  return fallback
}

function splitCspPolicies(value: string): string[] {
  return value
    .split(',')
    .map(policy => policy.trim())
    .filter(policy => policy.length > 0)
}

function frameAncestorSources(headers: Headers): string[][] {
  const csp = headers.get('Content-Security-Policy')
  if (!csp) return []

  const directives: string[][] = []
  for (const policy of splitCspPolicies(csp)) {
    for (const rawDirective of policy.split(';')) {
      const tokens = rawDirective
        .trim()
        .split(/\s+/)
        .filter(token => token.length > 0)
      if (tokens[0]?.toLowerCase() === 'frame-ancestors') {
        directives.push(tokens.slice(1))
      }
    }
  }

  return directives
}

function hostMatchesFrameAncestorSource(pattern: string, hostname: string): boolean {
  const normalizedPattern = pattern.toLowerCase().replace(/\.$/, '')
  const normalizedHostname = hostname.toLowerCase().replace(/\.$/, '')
  if (normalizedPattern === '*') return true
  if (normalizedPattern.startsWith('*.')) {
    const suffix = normalizedPattern.slice(2)
    return normalizedHostname.endsWith(`.${suffix}`)
  }
  return normalizedPattern === normalizedHostname
}

function sourceMatchesFrameAncestor(source: string, embedder: URL, target: URL): boolean {
  const normalized = source.trim()
  if (normalized === '*') return true
  if (normalized === "'self'") return embedder.origin === target.origin
  if (normalized === "'none'") return false
  if (/^[a-z][a-z0-9+.-]*:$/i.test(normalized)) {
    return embedder.protocol === normalized.toLowerCase()
  }

  let sourceUrl: URL
  try {
    if (normalized.startsWith('//')) {
      sourceUrl = new URL(`${target.protocol}${normalized}`)
    } else if (/^[a-z][a-z0-9+.-]*:\/\//i.test(normalized)) {
      sourceUrl = new URL(normalized)
    } else {
      sourceUrl = new URL(`${target.protocol}//${normalized}`)
    }
  } catch {
    return false
  }

  if (sourceUrl.protocol !== embedder.protocol) return false
  if (sourceUrl.port && sourceUrl.port !== embedder.port) return false
  return hostMatchesFrameAncestorSource(sourceUrl.hostname, embedder.hostname)
}

export function classifyArenaFrameHeaders(
  headers: Headers,
  targetUrl: URL,
  embedderOrigin: string,
): ArenaEmbedCapability {
  let embedder: URL
  try {
    embedder = new URL(embedderOrigin)
  } catch {
    return { mode: 'fetch', reason: 'frame-ancestors-mismatch' }
  }

  for (const sources of frameAncestorSources(headers)) {
    if (sources.length === 0) return { mode: 'fetch', reason: 'frame-ancestors-empty' }
    if (sources.some(source => source === "'none'")) {
      return { mode: 'fetch', reason: 'frame-ancestors-none' }
    }
    if (!sources.some(source => sourceMatchesFrameAncestor(source, embedder, targetUrl))) {
      return { mode: 'fetch', reason: 'frame-ancestors-mismatch' }
    }
  }

  const xFrameOptions = headers.get('X-Frame-Options')
  if (xFrameOptions) {
    const values = xFrameOptions
      .split(',')
      .map(value => value.trim().toUpperCase())
      .filter(value => value.length > 0)

    if (values.includes('DENY')) return { mode: 'fetch', reason: 'x-frame-options-deny' }
    if (values.includes('SAMEORIGIN') && targetUrl.origin !== embedder.origin) {
      return { mode: 'fetch', reason: 'x-frame-options-sameorigin' }
    }
  }

  return { mode: 'iframe', reason: 'ok' }
}

function responseFinalUrl(response: Response, fallback: URL): URL {
  if (!response.url) return fallback
  try {
    return new URL(response.url)
  } catch {
    return fallback
  }
}

export async function handleArenaEmbedCapability(request: Request): Promise<Response> {
  if (request.method === 'OPTIONS') return new Response(null, { status: 204 })
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return arenaTextResponse('method not allowed', 405)
  }

  const requestUrl = new URL(request.url)
  const target = validateArenaEmbedTarget(requestUrl.searchParams.get('url'))
  if (!target.ok) {
    return arenaJsonResponse({ mode: 'disabled', reason: target.reason }, target.status)
  }

  try {
    const upstream = await fetchArenaEmbedHeaders(target.url)
    const finalUrl = responseFinalUrl(upstream, target.url)
    if (!upstream.ok) {
      return arenaJsonResponse(
        { mode: 'disabled', finalUrl: finalUrl.toString(), reason: 'upstream-error' },
        502,
      )
    }

    const decision = classifyArenaFrameHeaders(upstream.headers, finalUrl, requestUrl.origin)
    if (decision.mode === 'fetch' && !isHtmlResponse(upstream.headers)) {
      return arenaJsonResponse(
        { mode: 'disabled', finalUrl: finalUrl.toString(), reason: 'non-html' },
        200,
      )
    }

    return arenaJsonResponse({ ...decision, finalUrl: finalUrl.toString() })
  } catch {
    return arenaJsonResponse({ mode: 'disabled', reason: 'fetch-error' }, 502)
  }
}

function isAllowedArenaEmbedProtocol(protocol: string, use: ArenaEmbedUrlUse): boolean {
  if (use === 'resource') return protocol === 'https:' || protocol === 'data:'
  return (
    protocol === 'http:' || protocol === 'https:' || protocol === 'mailto:' || protocol === 'tel:'
  )
}

export function rebaseArenaEmbedUrl(
  rawUrl: string,
  baseUrl: URL,
  use: ArenaEmbedUrlUse = 'document',
): string {
  const trimmed = rawUrl.trim()
  if (trimmed.length === 0 || trimmed.startsWith('#')) {
    return rawUrl
  }

  try {
    const rebased = new URL(trimmed, baseUrl)
    return isAllowedArenaEmbedProtocol(rebased.protocol, use) ? rebased.toString() : 'about:blank'
  } catch {
    return rawUrl
  }
}

export function rebaseArenaEmbedSrcset(rawValue: string, baseUrl: URL): string {
  if (rawValue.includes('data:')) return rawValue

  return rawValue
    .split(',')
    .map(candidate => {
      const trimmed = candidate.trim()
      if (!trimmed) return trimmed
      const parts = trimmed.split(/\s+/)
      const [url, ...descriptors] = parts
      if (!url) return trimmed
      return [rebaseArenaEmbedUrl(url, baseUrl, 'resource'), ...descriptors].join(' ')
    })
    .join(', ')
}

export function arenaEmbedAttributeNamesToRemove(attributes: Iterable<string[]>): string[] {
  const names: string[] = []
  for (const attribute of attributes) {
    const name = attribute[0]
    if (!name) continue
    const normalized = name.toLowerCase()
    if (normalized.startsWith('on') || normalized === 'srcdoc' || normalized === 'nonce') {
      names.push(name)
    }
  }
  return names
}

function rewriteArenaEmbedElement(element: ArenaHtmlRewriterElement, baseUrl: URL) {
  const namesToRemove = arenaEmbedAttributeNamesToRemove(element.attributes)
  for (const name of namesToRemove) {
    element.removeAttribute(name)
  }

  const tagName = element.tagName.toLowerCase()
  const href = element.getAttribute('href')
  if (tagName === 'a' && href) {
    element.setAttribute('href', rebaseArenaEmbedUrl(href, baseUrl))
  }
  if (tagName === 'link' && href) {
    element.setAttribute('href', rebaseArenaEmbedUrl(href, baseUrl, 'resource'))
  }

  const src = element.getAttribute('src')
  if (src) element.setAttribute('src', rebaseArenaEmbedUrl(src, baseUrl, 'resource'))

  const poster = element.getAttribute('poster')
  if (poster) element.setAttribute('poster', rebaseArenaEmbedUrl(poster, baseUrl, 'resource'))

  const srcset = element.getAttribute('srcset')
  if (srcset) element.setAttribute('srcset', rebaseArenaEmbedSrcset(srcset, baseUrl))

  if (tagName === 'a') {
    element.setAttribute('target', '_blank')
    element.setAttribute('rel', 'noopener noreferrer')
  }
}

function sanitizeArenaEmbedHtml(upstream: Response, finalUrl: URL): Response {
  const removeElement = {
    element(element: ArenaHtmlRewriterElement) {
      element.remove()
    },
  }
  const rewriteElement = {
    element(element: ArenaHtmlRewriterElement) {
      rewriteArenaEmbedElement(element, finalUrl)
    },
  }
  const source = new Response(upstream.body, {
    status: 200,
    headers: { 'Content-Type': 'text/html; charset=utf-8' },
  })

  const rewritten = new HTMLRewriter()
    .on('*', rewriteElement)
    .on('script', removeElement)
    .on('iframe', removeElement)
    .on('object', removeElement)
    .on('embed', removeElement)
    .on('form', removeElement)
    .on('base', removeElement)
    .on('meta[http-equiv]', removeElement)
    .transform(source)

  return new Response(rewritten.body, {
    status: rewritten.status,
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Content-Security-Policy': ARENA_EMBED_HTML_CSP,
      'Referrer-Policy': 'no-referrer',
      'X-Content-Type-Options': 'nosniff',
      'Cache-Control': 'public, s-maxage=3600, max-age=600',
    },
  })
}

export async function handleArenaEmbedHtml(request: Request): Promise<Response> {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return arenaTextResponse('method not allowed', 405)
  }

  const requestUrl = new URL(request.url)
  const target = validateArenaEmbedTarget(requestUrl.searchParams.get('url'))
  if (!target.ok) return arenaTextResponse(target.message, target.status)

  try {
    const upstream = await fetch(target.url.toString(), arenaEmbedFetchInit('GET'))
    const finalUrl = responseFinalUrl(upstream, target.url)
    if (!upstream.ok) return arenaTextResponse(`upstream error: ${upstream.status}`, 502)
    if (!isHtmlResponse(upstream.headers)) return arenaTextResponse('upstream is not html', 415)
    if (request.method === 'HEAD') {
      if (upstream.body) await upstream.body.cancel()
      return new Response(null, {
        headers: {
          'Content-Type': 'text/html; charset=utf-8',
          'Content-Security-Policy': ARENA_EMBED_HTML_CSP,
          'Cache-Control': 'public, s-maxage=3600, max-age=600',
        },
      })
    }
    return sanitizeArenaEmbedHtml(upstream, finalUrl)
  } catch {
    return arenaTextResponse('proxy error', 502)
  }
}
