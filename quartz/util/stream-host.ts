export const STREAM_HOSTNAME = 'stream.aarnphm.xyz'
export const STREAM_PREFIX = '/stream'

export function isStreamHostname(hostname: string): boolean {
  return hostname === STREAM_HOSTNAME
}

function leadingSlash(pathname: string): string {
  return pathname.startsWith('/') ? pathname : `/${pathname}`
}

export function isStreamRoutePathname(pathname: string): boolean {
  const normalized = leadingSlash(pathname)
  return (
    normalized === STREAM_PREFIX ||
    normalized.startsWith(`${STREAM_PREFIX}/`) ||
    normalized === '/on' ||
    normalized.startsWith('/on/')
  )
}

export function streamHostPathname(pathname: string): string {
  const normalized = leadingSlash(pathname)
  if (normalized === STREAM_PREFIX || normalized === `${STREAM_PREFIX}/`) return '/'
  if (normalized.startsWith(`${STREAM_PREFIX}/`)) return normalized.slice(STREAM_PREFIX.length)
  return normalized
}

export function streamAssetPathname(pathname: string, isDocument: boolean): string {
  const canonical = streamHostPathname(pathname)
  if (canonical === '/index.xml') return `${STREAM_PREFIX}/index.xml`
  if (!isDocument) return canonical
  return canonical === '/' ? STREAM_PREFIX : `${STREAM_PREFIX}${canonical}`
}

export function streamDocumentRedirectUrl(
  baseUrl: string,
  requestUrl: string | URL,
): string | null {
  const source = requestUrl instanceof URL ? requestUrl : new URL(requestUrl)
  const pathname = streamHostPathname(source.pathname)
  if (pathname === '/' || isStreamRoutePathname(pathname)) return null

  const target = new URL(baseUrl)
  if (target.hostname === source.hostname && target.hostname.startsWith('stream.')) {
    target.hostname = target.hostname.replace(/^stream\./, '')
  }
  target.pathname = pathname
  target.search = source.search
  target.hash = source.hash
  return target.toString()
}

export function streamHostUrl(href: string): string {
  const parsed = new URL(href, `https://${STREAM_HOSTNAME}`)
  parsed.pathname = streamHostPathname(parsed.pathname)
  return parsed.toString()
}
