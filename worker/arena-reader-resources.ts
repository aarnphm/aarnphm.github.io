const IMAGE_LIMIT = 8 * 1024 * 1024
const PDF_LIMIT = 32 * 1024 * 1024
const SOURCE_TIMEOUT_MS = 30_000
const DNS_TIMEOUT_MS = 5_000
const IMAGE_TYPES = new Set([
  'image/avif',
  'image/bmp',
  'image/gif',
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/x-icon',
  'image/vnd.microsoft.icon',
])

export interface ArenaReaderResource {
  articleId: string
  snapshotId: string
  resourceId: string
  sourceUrl: string
  purpose: 'image' | 'pdf'
}

class ResourceError extends Error {
  constructor(
    readonly code: string,
    readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ArenaReaderResourceError'
  }
}

function normalizedHostname(hostname: string): string {
  return hostname
    .toLowerCase()
    .replace(/^\[([^\]]+)\]$/, '$1')
    .replace(/\.$/, '')
}

function isPublicIpv4(address: string): boolean {
  const parts = address.split('.')
  if (parts.length !== 4 || parts.some(part => !/^\d{1,3}$/.test(part))) return false
  const [a, b, c, d] = parts.map(Number)
  if ([a, b, c, d].some(part => part > 255)) return false
  if (a === 0 || a === 10 || a === 127 || a >= 224) return false
  if (a === 100 && b >= 64 && b <= 127) return false
  if (a === 169 && b === 254) return false
  if (a === 172 && b >= 16 && b <= 31) return false
  if (a === 192 && b === 168) return false
  if (a === 192 && b === 0 && (c === 0 || c === 2)) return false
  if (a === 192 && b === 31 && c === 196) return false
  if (a === 192 && b === 52 && c === 193) return false
  if (a === 192 && b === 88 && c === 99) return false
  if (a === 192 && b === 175 && c === 48) return false
  if (a === 198 && (b === 18 || b === 19)) return false
  if (a === 198 && b === 51 && c === 100) return false
  if (a === 203 && b === 0 && c === 113) return false
  return true
}

export function isArenaPublicAddress(rawAddress: string): boolean {
  const address = normalizedHostname(rawAddress)
  if (!address.includes(':')) return isPublicIpv4(address)
  if (!/^[a-f\d:.]+$/.test(address)) return false
  const parsed = URL.parse(`http://[${address}]/`)
  if (!parsed) return false
  const normalized = normalizedHostname(parsed.hostname)
  const halves = normalized.split('::')
  const left = halves[0] ? halves[0].split(':') : []
  const right = halves[1] ? halves[1].split(':') : []
  const words = [
    ...left,
    ...Array.from({ length: 8 - left.length - right.length }, () => '0'),
    ...right,
  ].map(word => Number.parseInt(word, 16))
  const [first, second, third] = words
  // Restrict IPv6 to global unicast, excluding IANA special-use and transition prefixes.
  if (first < 0x2000 || first > 0x3fff) return false
  if (first === 0x2001 && (second < 0x200 || second === 0xdb8)) return false
  if (first === 0x2002 || first === 0x3ffe || first === 0x3fff) return false
  if (first === 0x2620 && second === 0x4f && third === 0x8000) return false
  return true
}

function isPublicHostnameSyntax(hostname: string): boolean {
  const host = normalizedHostname(hostname)
  if (host.includes(':') || /^[\d.]+$/.test(host)) return isArenaPublicAddress(host)
  if (host.length > 253 || !host.includes('.')) return false
  if (
    ['localhost', 'local', 'internal', 'home.arpa', 'test', 'invalid', 'onion'].some(
      suffix => host === suffix || host.endsWith(`.${suffix}`),
    )
  )
    return false
  return host.split('.').every(label => /^[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?$/.test(label))
}

export function validateArenaReaderTarget(rawUrl: string): URL | null {
  if (rawUrl.length > 8192) return null
  const url = URL.parse(rawUrl)
  if (!url || (url.protocol !== 'https:' && url.protocol !== 'http:')) return null
  if (url.username || url.password || url.port) return null
  if (!isPublicHostnameSyntax(url.hostname)) return null
  return url
}

function dnsAddresses(value: unknown): string[] | null {
  if (!value || typeof value !== 'object' || !('Status' in value) || value.Status !== 0) return null
  if ('TC' in value && value.TC === true) return null
  if (!('Answer' in value)) return []
  if (!Array.isArray(value.Answer)) return null
  const addresses: string[] = []
  const answers: unknown[] = value.Answer
  for (const answer of answers) {
    if (!answer || typeof answer !== 'object' || !('type' in answer) || !('data' in answer))
      return null
    if (typeof answer.data !== 'string') return null
    if (answer.type === 1 || answer.type === 28) addresses.push(answer.data)
    if (answer.type === 5 && !isPublicHostnameSyntax(answer.data)) return null
  }
  return addresses
}

async function queryAddresses(hostname: string, type: 'A' | 'AAAA'): Promise<string[] | null> {
  const url = new URL('https://cloudflare-dns.com/dns-query')
  url.searchParams.set('name', normalizedHostname(hostname))
  url.searchParams.set('type', type)
  const response = await fetch(url, {
    headers: { Accept: 'application/dns-json' },
    redirect: 'manual',
    credentials: 'omit',
    signal: AbortSignal.timeout(DNS_TIMEOUT_MS),
  })
  if (!response.ok) {
    await response.body?.cancel()
    return null
  }
  const bytes = await readBoundedBody(response, 64 * 1024)
  const value: unknown = JSON.parse(new TextDecoder().decode(bytes))
  return dnsAddresses(value)
}

export async function isPublicArenaHostname(hostname: string): Promise<boolean> {
  if (!isPublicHostnameSyntax(hostname)) return false
  const host = normalizedHostname(hostname)
  if (host.includes(':') || /^[\d.]+$/.test(host)) return isArenaPublicAddress(host)
  const results = await Promise.allSettled([
    queryAddresses(host, 'A'),
    queryAddresses(host, 'AAAA'),
  ])
  const addresses: string[] = []
  for (const result of results) {
    if (result.status === 'rejected' || result.value === null) return false
    addresses.push(...result.value)
  }
  return addresses.length > 0 && addresses.every(isArenaPublicAddress)
}

export function arenaReaderSourceHeaders(input?: HeadersInit): Headers {
  const provided = new Headers(input)
  // Wikimedia requires an identifiable client with contact information.
  const headers = new Headers({
    'Accept-Encoding': 'identity',
    'User-Agent': 'GardenArenaReader/1.0 (https://aarnphm.xyz/arena)',
  })
  for (const name of ['Accept', 'Accept-Language']) {
    const value = provided.get(name)
    if (value && value.length <= 1024) headers.set(name, value)
  }
  const range = provided.get('Range')
  if (range && /^bytes=\d{0,16}-\d{0,16}$/.test(range)) headers.set('Range', range)
  return headers
}

export async function fetchArenaReaderSource(
  rawUrl: string,
  options: { signal?: AbortSignal; method?: 'GET' | 'HEAD'; headers?: HeadersInit } = {},
): Promise<{ response: Response; finalUrl: string }> {
  const headers = arenaReaderSourceHeaders(options.headers)
  const timeout = AbortSignal.timeout(SOURCE_TIMEOUT_MS)
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout
  let target = rawUrl
  for (let redirects = 0; redirects <= 5; redirects++) {
    signal.throwIfAborted()
    const url = validateArenaReaderTarget(target)
    if (!url || !(await isPublicArenaHostname(url.hostname)))
      throw new ResourceError('blocked-source', 400, 'The source must resolve to public addresses.')
    // DNS preflight is defense in depth; global_fetch_strictly_public is the egress boundary.
    const response = await fetch(url, {
      method: options.method ?? 'GET',
      headers,
      redirect: 'manual',
      credentials: 'omit',
      signal,
    })
    if (![301, 302, 303, 307, 308].includes(response.status))
      return { response, finalUrl: url.href }
    const location = response.headers.get('Location')
    await response.body?.cancel()
    const next = location ? URL.parse(location, url) : null
    if (!next)
      throw new ResourceError('invalid-redirect', 502, 'The source returned an invalid redirect.')
    target = next.href
  }
  throw new ResourceError('redirect-limit', 502, 'The source redirected too many times.')
}

function resourceKey(resource: ArenaReaderResource): string {
  for (const id of [resource.articleId, resource.snapshotId, resource.resourceId]) {
    if (!/^[a-z\d_-]{1,128}$/i.test(id))
      throw new ResourceError('invalid-resource', 400, 'Invalid saved resource identifier.')
  }
  return `arena-reader/v1/${resource.articleId}/resources/${resource.snapshotId}/${resource.resourceId}`
}

function contentTypeForResource(contentType: string | undefined | null, purpose: 'image' | 'pdf') {
  const mime = contentType?.split(';')[0].trim().toLowerCase()
  if (purpose === 'pdf') return mime === 'application/pdf' ? mime : null
  return mime && IMAGE_TYPES.has(mime) ? mime : null
}

async function readBoundedBody(
  response: Response,
  limit: number,
): Promise<Uint8Array<ArrayBuffer>> {
  const declaredLength = response.headers.get('Content-Length')
  if (declaredLength && Number(declaredLength) > limit) {
    await response.body?.cancel()
    throw new ResourceError('resource-too-large', 413, 'Open the source to view this larger file.')
  }
  const reader = response.body?.getReader()
  if (!reader) return new Uint8Array()
  const chunks: Uint8Array[] = []
  let length = 0
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      length += value.byteLength
      if (length > limit)
        throw new ResourceError(
          'resource-too-large',
          413,
          'Open the source to view this larger file.',
        )
      chunks.push(value)
    }
  } finally {
    await reader.cancel().catch(() => {})
  }
  const body = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return body
}

export async function saveArenaReaderResource(
  bucket: R2Bucket,
  resource: ArenaReaderResource,
  response: Response,
): Promise<void> {
  const key = resourceKey(resource)
  const contentType = contentTypeForResource(response.headers.get('Content-Type'), resource.purpose)
  if (response.status !== 200 || !contentType) {
    await response.body?.cancel()
    throw new ResourceError(
      'unsupported-resource',
      415,
      'The source did not return a supported file.',
    )
  }
  const body = await readBoundedBody(response, resource.purpose === 'pdf' ? PDF_LIMIT : IMAGE_LIMIT)
  if (body.length === 0)
    throw new ResourceError('empty-resource', 502, 'The source returned an empty file.')
  if (resource.purpose === 'pdf' && new TextDecoder().decode(body.subarray(0, 5)) !== '%PDF-')
    throw new ResourceError('unsupported-resource', 415, 'The source did not return a PDF file.')
  await bucket.put(key, body, { httpMetadata: { contentType }, onlyIf: { etagDoesNotMatch: '*' } })
}

function privateHeaders(): Headers {
  return new Headers({
    'Cache-Control': 'private, no-store',
    'X-Content-Type-Options': 'nosniff',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'Referrer-Policy': 'no-referrer',
    'Content-Security-Policy': "default-src 'none'; sandbox; frame-ancestors 'self'",
  })
}

function byteRange(
  header: string | null,
  size: number,
): { offset: number; length: number } | 'unsatisfiable' | undefined {
  if (!header) return undefined
  const match = header.match(/^bytes=(\d*)-(\d*)$/)
  if (!match || (!match[1] && !match[2])) return undefined
  const start = match[1] ? Number(match[1]) : undefined
  const end = match[2] ? Number(match[2]) : undefined
  if (
    (start !== undefined && !Number.isSafeInteger(start)) ||
    (end !== undefined && !Number.isSafeInteger(end))
  )
    return undefined
  if (start === undefined) {
    if (!end || !size) return 'unsatisfiable'
    const length = Math.min(end, size)
    return { offset: size - length, length }
  }
  if (end !== undefined && end < start) return undefined
  if (start >= size) return 'unsatisfiable'
  return { offset: start, length: Math.min(end ?? size - 1, size - 1) - start + 1 }
}

async function serveSavedResource(
  request: Request,
  bucket: R2Bucket,
  key: string,
  metadata: R2Object,
  contentType: string,
): Promise<Response> {
  const headers = privateHeaders()
  headers.set('Content-Type', contentType)
  headers.set('Content-Length', String(metadata.size))
  headers.set('Accept-Ranges', 'bytes')
  headers.set('ETag', metadata.httpEtag)
  headers.set('Last-Modified', metadata.uploaded.toUTCString())
  const etags = request.headers.get('If-None-Match')
  const modifiedSince = request.headers.get('If-Modified-Since')
  const uploaded = Math.floor(metadata.uploaded.getTime() / 1000) * 1000
  if (
    etags
      ? etags
          .split(',')
          .some(
            value => value.trim() === '*' || value.trim().replace(/^W\//, '') === metadata.httpEtag,
          )
      : modifiedSince && uploaded <= Date.parse(modifiedSince)
  )
    return new Response(null, { status: 304, headers })
  if (request.method === 'HEAD') return new Response(null, { headers })
  const ifRange = request.headers.get('If-Range')
  const matchesRange =
    !ifRange ||
    ifRange === metadata.httpEtag ||
    (!ifRange.startsWith('W/') && uploaded <= Date.parse(ifRange))
  const range = byteRange(matchesRange ? request.headers.get('Range') : null, metadata.size)
  if (range === 'unsatisfiable') {
    headers.set('Content-Range', `bytes */${metadata.size}`)
    headers.set('Content-Length', '0')
    return new Response(null, { status: 416, headers })
  }
  const object = await bucket.get(key, { range })
  if (!object) return new Response(null, { status: 404, headers: privateHeaders() })
  if (range) {
    headers.set(
      'Content-Range',
      `bytes ${range.offset}-${range.offset + range.length - 1}/${metadata.size}`,
    )
    headers.set('Content-Length', String(range.length))
  }
  return new Response(object.body, { status: range ? 206 : 200, headers })
}

export async function serveArenaReaderResource(
  request: Request,
  bucket: R2Bucket,
  resource: ArenaReaderResource,
): Promise<Response> {
  const headers = privateHeaders()
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    headers.set('Allow', 'GET, HEAD')
    return new Response(null, { status: 405, headers })
  }
  try {
    const key = resourceKey(resource)
    const saved = await bucket.head(key)
    const savedType = contentTypeForResource(saved?.httpMetadata?.contentType, resource.purpose)
    if (saved && savedType) return serveSavedResource(request, bucket, key, saved, savedType)
    if (request.method === 'HEAD') return new Response(null, { status: 404, headers })
    const { response } = await fetchArenaReaderSource(resource.sourceUrl, {
      signal: request.signal,
      headers: {
        Accept: resource.purpose === 'pdf' ? 'application/pdf' : [...IMAGE_TYPES].join(', '),
      },
    })
    if (!response.ok) {
      await response.body?.cancel()
      throw new ResourceError(
        'source-unavailable',
        response.status === 404 ? 404 : 502,
        'The source file is unavailable.',
      )
    }
    await saveArenaReaderResource(bucket, resource, response)
    const metadata = await bucket.head(key)
    const contentType = contentTypeForResource(
      metadata?.httpMetadata?.contentType,
      resource.purpose,
    )
    if (!metadata || !contentType)
      throw new ResourceError('cache-write-failed', 502, 'The file could not be saved.')
    return serveSavedResource(request, bucket, key, metadata, contentType)
  } catch (error) {
    const failure =
      error instanceof ResourceError
        ? error
        : new ResourceError('source-unavailable', 502, 'The source file could not be loaded.')
    headers.set('Content-Type', 'application/json; charset=utf-8')
    return new Response(
      JSON.stringify({
        error: failure.code,
        message: failure.message,
        sourceUrl: resource.sourceUrl,
      }),
      { status: failure.status, headers },
    )
  }
}
