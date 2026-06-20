const TRIATHLON_PREFIX = '/triathlon'
const SHORTCUT_HOST_PREFIX = 't.'

function canonicalShortcutBase(baseUrl: string, source: URL): URL {
  const target = new URL(baseUrl)
  if (target.hostname === source.hostname && target.hostname.startsWith(SHORTCUT_HOST_PREFIX)) {
    target.hostname = target.hostname.slice(SHORTCUT_HOST_PREFIX.length)
  }
  return target
}

function triathlonShortcutPathname(pathname: string): string {
  if (pathname === '/' || pathname === TRIATHLON_PREFIX || pathname === `${TRIATHLON_PREFIX}/`) {
    return TRIATHLON_PREFIX
  }
  if (pathname.startsWith(`${TRIATHLON_PREFIX}/`)) return pathname
  return `${TRIATHLON_PREFIX}${pathname.startsWith('/') ? pathname : `/${pathname}`}`
}

export function triathlonShortcutRedirectUrl(
  baseUrl: string,
  requestUrl: string | URL,
  isDocument: boolean,
): string {
  const source = requestUrl instanceof URL ? requestUrl : new URL(requestUrl)
  const target = canonicalShortcutBase(baseUrl, source)
  target.pathname = isDocument ? triathlonShortcutPathname(source.pathname) : source.pathname
  target.search = source.search
  target.hash = source.hash
  return target.toString()
}
