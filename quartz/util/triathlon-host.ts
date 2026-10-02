import { triathlonOnSlugFromShortcutPath } from './triathlon-date-route'

export const TRIATHLON_HOSTNAME = 't.aarnphm.xyz'
export const TRIATHLON_PREFIX = '/triathlon'
const APEX_ORIGIN = 'https://aarnphm.xyz'
const SHARED_PATH =
  /^\/(?:api|comments|fonts|models|_plausible|mcp|sse|authorize|register|token|\.well-known)(?:\/|$)/
const SECTION_PATH =
  /^\/(?:tools|calc|analytics|maps|training|calendar|feed|data|on)(?:\/|\.(?:html?|md|ics)$|$)/

export function triathlonHostPathname(pathname: string): string {
  const normalized = pathname.startsWith('/') ? pathname : `/${pathname}`
  if ([TRIATHLON_PREFIX, `${TRIATHLON_PREFIX}/`, `${TRIATHLON_PREFIX}.html`].includes(normalized))
    return '/'
  if (normalized === `${TRIATHLON_PREFIX}.md`) return '/index.md'
  const path = normalized.startsWith(`${TRIATHLON_PREFIX}/`)
    ? normalized.slice(TRIATHLON_PREFIX.length)
    : normalized
  const dateSlug = triathlonOnSlugFromShortcutPath(path)
  if (dateSlug) return `/${dateSlug.slice('triathlon/'.length)}`
  if (path === '/index.html') return '/'
  return SECTION_PATH.test(path) ? path.replace(/\.html?$/, '').replace(/\/$/, '') : path
}

export function isTriathlonRoutePathname(pathname: string): boolean {
  const canonical = triathlonHostPathname(pathname)
  return canonical === '/' || canonical === '/index.md' || SECTION_PATH.test(canonical)
}

export function triathlonAssetPathname(pathname: string): string {
  const canonical = triathlonHostPathname(pathname)
  if (canonical === '/index.md') return `${TRIATHLON_PREFIX}.md`
  if (!isTriathlonRoutePathname(canonical)) return pathname
  return canonical === '/' ? TRIATHLON_PREFIX : `${TRIATHLON_PREFIX}${canonical}`
}

export function triathlonHostUrl(href: string | URL): string {
  const target = new URL(href, `https://${TRIATHLON_HOSTNAME}`)
  target.protocol = 'https:'
  target.host = TRIATHLON_HOSTNAME
  target.pathname = triathlonHostPathname(target.pathname)
  return target.toString()
}

export function triathlonApexRedirectUrl(source: URL): string | null {
  if (
    source.hostname !== 'aarnphm.xyz' ||
    ![TRIATHLON_PREFIX, `${TRIATHLON_PREFIX}/`, `${TRIATHLON_PREFIX}.html`].includes(
      source.pathname,
    )
  )
    return null
  return triathlonHostUrl(source)
}

export function triathlonDocumentRedirectUrl(baseUrl: string, source: URL): string | null {
  if (isTriathlonRoutePathname(source.pathname) || SHARED_PATH.test(source.pathname)) return null
  const target = new URL(baseUrl)
  if (target.hostname === TRIATHLON_HOSTNAME) target.hostname = 'aarnphm.xyz'
  target.pathname = source.pathname
  target.search = source.search
  target.hash = source.hash
  return target.toString()
}

export function triathlonHostLinkUrl(href: string, reference: string | URL): string {
  if (href.startsWith('#')) return href
  const target = new URL(href, reference)
  const apexTriathlon =
    target.origin === APEX_ORIGIN &&
    (target.pathname === TRIATHLON_PREFIX ||
      target.pathname.startsWith(`${TRIATHLON_PREFIX}/`) ||
      target.pathname === `${TRIATHLON_PREFIX}.md` ||
      target.pathname === `${TRIATHLON_PREFIX}.html`)
  if (apexTriathlon) target.hostname = TRIATHLON_HOSTNAME
  if (target.hostname !== TRIATHLON_HOSTNAME) return target.toString()
  if (isTriathlonRoutePathname(target.pathname)) return triathlonHostUrl(target)
  if (SHARED_PATH.test(target.pathname)) return target.toString()
  if (!/\.[^/]+$/.test(target.pathname) || /\.(?:html?|md)$/.test(target.pathname))
    target.hostname = 'aarnphm.xyz'
  return target.toString()
}
