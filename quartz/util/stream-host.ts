export const STREAM_HOSTNAME = 'stream.aarnphm.xyz'
export const STREAM_PREFIX = '/stream'

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
  if (!isDocument) return canonical
  return canonical === '/' ? STREAM_PREFIX : `${STREAM_PREFIX}${canonical}`
}

export function streamHostUrl(href: string): string {
  const parsed = new URL(href, `https://${STREAM_HOSTNAME}`)
  parsed.pathname = streamHostPathname(parsed.pathname)
  return parsed.toString()
}
