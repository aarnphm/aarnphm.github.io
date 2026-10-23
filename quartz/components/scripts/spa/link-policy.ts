import { getFileExtension, sluggify } from '../../../util/path'
import { NOTES_HOSTNAME, stackedNoteRootSlug, stackedNotesUrl } from '../../../util/stacked-notes'
import {
  isStreamRoutePathname,
  STREAM_HOSTNAME,
  streamHostPathname,
  streamHostUrl,
} from '../../../util/stream-host'
import { TRIATHLON_HOSTNAME, triathlonHostLinkUrl } from '../../../util/triathlon-host'
import { triathlonPreviewTarget } from '../../../util/triathlon-preview'

// adapted from `micromorph`
// https://github.com/natemoo-re/micromorph
const NODE_TYPE_ELEMENT = 1

export const isElement = (target: EventTarget | null): target is Element =>
  (target as Node)?.nodeType === NODE_TYPE_ELEMENT

export const linkUrl = (link: HTMLAnchorElement): URL | null => {
  const href = link.getAttribute('href')
  if (!href) return null
  try {
    return new URL(href, window.location.toString())
  } catch {
    return null
  }
}

const isHttpUrl = (url: URL): boolean => url.protocol === 'http:' || url.protocol === 'https:'

export const isLocalHttpUrl = (url: URL): boolean =>
  isHttpUrl(url) && url.origin === window.location.origin

export const ensureExternalLinkTarget = (link: HTMLAnchorElement, url: URL): boolean => {
  if (!isHttpUrl(url) || url.origin === window.location.origin) return false
  link.target = '_blank'
  const rel = link.rel.split(/\s+/).filter(Boolean)
  for (const value of ['noopener', 'noreferrer']) {
    if (!rel.includes(value)) rel.push(value)
  }
  link.rel = rel.join(' ')
  return true
}

export const isSamePage = (url: URL): boolean => {
  const sameOrigin = url.origin === window.location.origin
  const samePath = url.pathname === window.location.pathname
  return sameOrigin && samePath && url.search === window.location.search
}

export const APEX_ORIGIN = 'https://aarnphm.xyz'

function normalizeStreamPath(raw: string): { path: string; streamRoute: boolean } | null {
  if (!raw) return null
  const trimmed = raw.trim()
  if (trimmed === '') return { path: '/', streamRoute: true }
  if (trimmed.startsWith('#')) return null

  try {
    const parsed = new URL(trimmed, `https://${STREAM_HOSTNAME}`)
    const hash = parsed.hash ?? ''
    const search = parsed.search ?? ''
    const streamRoute = isStreamRoutePathname(parsed.pathname)
    const canonicalPathname = streamRoute ? streamHostPathname(parsed.pathname) : parsed.pathname
    let pathname = canonicalPathname.replace(/^\/+/, '')
    if (pathname === '') return { path: `/${search}${hash}`, streamRoute }
    pathname = sluggify(pathname)
    const normalizedPath = `/${pathname}`
    return { path: `${normalizedPath}${search}${hash}`, streamRoute }
  } catch {
    return null
  }
}

export function transformHostInternalLinks(root: Document | Element) {
  if (typeof window === 'undefined') return
  const host = window.location.hostname
  if (host !== STREAM_HOSTNAME && host !== TRIATHLON_HOSTNAME && host !== NOTES_HOSTNAME) return

  const anchors = root.querySelectorAll<HTMLAnchorElement>('a.internal')
  anchors.forEach(anchor => {
    if (host === STREAM_HOSTNAME) {
      const preferred = anchor.dataset.slug ?? anchor.getAttribute('href') ?? ''
      const target = normalizeStreamPath(preferred)
      if (!target) return

      const absoluteHref = target.streamRoute
        ? streamHostUrl(target.path)
        : `${APEX_ORIGIN}${target.path.startsWith('/') ? target.path : `/${target.path}`}`
      if (anchor.getAttribute('href') !== absoluteHref) {
        anchor.setAttribute('href', absoluteHref)
      }
      anchor.dataset.routerIgnore = 'true'
      if (!triathlonPreviewTarget(anchor.href)) anchor.dataset.noPopover = 'true'
    } else if (host === TRIATHLON_HOSTNAME) {
      anchor.href = triathlonHostLinkUrl(anchor.href, window.location.href)
      anchor.dataset.noPopover = 'true'
    } else if (host === NOTES_HOSTNAME) {
      if (anchor.getAttribute('href')?.startsWith('#')) return
      const url = linkUrl(anchor)
      if (url && isLocalHttpUrl(url) && !getFileExtension(url.pathname)) {
        const slug = stackedNoteRootSlug(url)
        if (slug) anchor.href = stackedNotesUrl(url, [slug]).href
      }
    }
  })

  if (host === STREAM_HOSTNAME) {
    const transcludeRefs = root.querySelectorAll<HTMLElement>('.transclude-ref[data-href]')
    transcludeRefs.forEach(ref => {
      const preferred = ref.dataset.slug ?? ref.dataset.href ?? ''
      const target = normalizeStreamPath(preferred)
      if (!target) return
      const absoluteHref = target.streamRoute
        ? streamHostUrl(target.path)
        : `${APEX_ORIGIN}${target.path.startsWith('/') ? target.path : `/${target.path}`}`
      ref.dataset.href = absoluteHref
      ref.dataset.routerIgnore = 'true'
    })
  }
}
