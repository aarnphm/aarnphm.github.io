import { readTrailPreference } from '../../util/trail-preference'
import { loadPageStyles } from './page-styles'

export function preloadHomepageTrail(page: Document, url: URL) {
  if (!readTrailPreference()) return
  const source = page.querySelector<HTMLIFrameElement>('iframe[data-trail-frame]')?.dataset.trailSrc
  if (!source) return
  const href = new URL(source, url)
  if (href.origin !== window.location.origin) return
  void loadPageStyles(page.body)
  const previous = document.head.querySelector<HTMLLinkElement>('link[data-homepage-trail-preload]')
  if (previous?.href === href.href) return
  previous?.remove()
  const link = document.createElement('link')
  link.rel = 'preload'
  link.as = 'script'
  link.href = href.href
  link.fetchPriority = 'low'
  link.dataset.homepageTrailPreload = ''
  link.dataset.persist = ''
  document.head.append(link)
}
