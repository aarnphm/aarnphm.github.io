import { fetchCanonical, prefetchCanonical } from '../../util/fetch-canonical'
import {
  FullSlug,
  getFileExtension,
  getFullSlug,
  normalizeRelativeURLs,
  RelativeURL,
} from '../../util/path'
import { STREAM_HOSTNAME, streamDocumentRedirectUrl } from '../../util/stream-host'
import {
  TRIATHLON_HOSTNAME,
  triathlonApexRedirectUrl,
  triathlonHostLinkUrl,
} from '../../util/triathlon-host'
import { loadPageStyles } from './page-styles'
import { parseHtml } from './spa/html-parser'
import {
  APEX_ORIGIN,
  ensureExternalLinkTarget,
  isElement,
  isLocalHttpUrl,
  isSamePage,
  linkUrl,
  transformHostInternalLinks,
} from './spa/link-policy'
import { replacePageBody } from './spa/page-body'
import { stacked } from './spa/stacked-manager'
import { announcer } from './spa/stacked-view'
import { Toast } from './toast'

async function startViewTransition(callback: () => void): Promise<void> {
  if (
    document.visibilityState === 'hidden' ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches ||
    typeof document.startViewTransition !== 'function'
  ) {
    callback()
    return
  }

  const transition = document.startViewTransition(callback)
  void transition.finished.catch(() => undefined)
  await transition.updateCallbackDone
}

const getOpts = ({ target }: Event): { url: URL; scroll?: boolean } | undefined => {
  if (!isElement(target)) return
  const a = target.closest('a')
  if (!(a instanceof HTMLAnchorElement)) return
  const url = linkUrl(a)
  if (!url) return
  if (ensureExternalLinkTarget(a, url)) return
  if ((a.target && a.target !== '_self') || a.hasAttribute('download')) return
  if ('routerIgnore' in a.dataset) return
  if (!isLocalHttpUrl(url)) return
  // Hosted PDFs are documents: the worker redirects their navigations to the reader.
  if (/\.pdf$/i.test(url.pathname)) return
  return { url, scroll: 'routerNoscroll' in a.dataset ? false : undefined }
}

function notifyNav(url: FullSlug) {
  // Standalone viewers own their initialization and have no Quartz page structure.
  if (document.body.dataset.page === 'code-viewer') return
  const event: CustomEventMap['nav'] = new CustomEvent('nav', { detail: { url } })
  document.dispatchEvent(event)
}

const cleanupFns = new Set<() => void>()
window.addCleanup = fn => cleanupFns.add(fn)

if (!window.quartzToast) {
  const toast = new Toast()
  window.quartzToast = toast

  const handleToast = (event: CustomEventMap['toast']) => {
    const detail = event.detail
    if (!detail?.message) return
    toast.show(detail.message, {
      durationMs: detail.durationMs,
      styles: detail.styles,
      containerId: detail.containerId,
      containerStyles: detail.containerStyles,
      containerHost: detail.containerHost,
      action: detail.action,
    })
  }

  document.addEventListener('toast', handleToast)
  document.addEventListener('prenav', () => toast.destroy())
}

let navProgressBar: HTMLDivElement | null = null
let navProgressStart: number | null = null
let navProgressReset: number | null = null

function getNavProgressBar() {
  if (navProgressBar && document.body.contains(navProgressBar)) {
    return navProgressBar
  }

  const existing = document.querySelector<HTMLDivElement>('.navigation-progress')
  if (existing) {
    navProgressBar = existing
    return navProgressBar
  }

  navProgressBar = document.createElement('div')
  navProgressBar.className = 'navigation-progress'
  document.body.appendChild(navProgressBar)
  return navProgressBar
}

function startLoading() {
  const loadingBar = getNavProgressBar()
  if (navProgressStart) {
    window.clearTimeout(navProgressStart)
  }
  if (navProgressReset) {
    window.clearTimeout(navProgressReset)
  }

  loadingBar.style.opacity = '1'
  loadingBar.style.width = '0'
  loadingBar.style.animation = 'none'
  loadingBar.style.backgroundPosition = '-100% 0'

  navProgressStart = window.setTimeout(() => {
    loadingBar.style.width = '100%'
    loadingBar.style.animation = 'navigation-progress-sweep 2.5s linear infinite'
  }, 10)
}

function stopLoading() {
  const loadingBar = getNavProgressBar()
  if (navProgressStart) {
    window.clearTimeout(navProgressStart)
  }
  loadingBar.style.animation = 'none'
  loadingBar.style.opacity = '0'
  navProgressReset = window.setTimeout(() => {
    loadingBar.style.width = '0'
    loadingBar.style.backgroundPosition = '-100% 0'
  }, 300)
}

window.stacked = stacked

let activeNavigation: AbortController | undefined

async function navigate(
  url: URL,
  isBack: boolean = false,
  options: { transition?: boolean; signal?: AbortSignal } = {},
) {
  if (options.signal?.aborted) return
  activeNavigation?.abort()
  activeNavigation = new AbortController()
  const signal = options.signal
    ? AbortSignal.any([options.signal, activeNavigation.signal])
    : activeNavigation.signal
  if (
    window.location.hostname === STREAM_HOSTNAME &&
    isLocalHttpUrl(url) &&
    (!getFileExtension(url.pathname) || /\.html?$/i.test(url.pathname))
  ) {
    const redirect = streamDocumentRedirectUrl(APEX_ORIGIN, url)
    if (redirect) url = new URL(redirect)
  }
  const apexTriathlonRedirect = triathlonApexRedirectUrl(url)
  if (apexTriathlonRedirect) {
    window.location.assign(apexTriathlonRedirect)
    return
  }
  if (window.location.hostname === TRIATHLON_HOSTNAME) {
    url = new URL(triathlonHostLinkUrl(url.toString(), window.location.href))
  }
  if (url.origin !== window.location.origin) {
    window.location.assign(url)
    return
  }
  if (isBack && document.body.dataset.page !== 'code-viewer' && stacked.restore(url)) return
  if (isBack && stacked.active) stacked.destroy()
  if (!isBack && stacked.active) return await stacked.navigate(url)

  startLoading()

  const contents = await fetchCanonical(new URL(`${url}`), { signal })
    .then(res => {
      const contentType = res.headers.get('content-type')
      if (contentType?.startsWith('text/html')) {
        return res.text()
      } else {
        if (!signal.aborted) window.location.assign(url)
      }
    })
    .catch(() => {
      if (!signal.aborted) window.location.assign(url)
    })

  if (!contents || signal.aborted) return

  const html = parseHtml(contents)
  const isCodeViewer = html.body.dataset.page === 'code-viewer'
  normalizeRelativeURLs(html, url)
  transformHostInternalLinks(html)
  // Link the page-scoped sheets the new body uses while the old page still shows; a stalled
  // sheet only delays navigation.
  await Promise.race([loadPageStyles(html.body), new Promise(resolve => setTimeout(resolve, 2000))])
  if (signal.aborted) return

  const update = () => {
    if (signal.aborted) return
    // notify about to nav
    const event: CustomEventMap['prenav'] = new CustomEvent('prenav', { detail: {} })
    document.dispatchEvent(event)

    // cleanup old
    cleanupFns.forEach(fn => fn())
    cleanupFns.clear()

    if (isCodeViewer) {
      // Keep the sheets loaded for Back, without letting garden rules style the viewer.
      const sheets = Array.from(document.styleSheets).filter(sheet => !sheet.disabled)
      sheets.forEach(sheet => (sheet.disabled = true))
      window.addCleanup(() => sheets.forEach(sheet => (sheet.disabled = false)))
    }

    let title = html.querySelector('title')?.textContent
    if (title) {
      document.title = title
    } else {
      const h1 = document.querySelector('h1')
      title = h1?.innerText ?? h1?.textContent ?? url.pathname
    }
    if (announcer.textContent !== title) {
      announcer.textContent = title
    }
    announcer.dataset.persist = ''
    html.body.appendChild(announcer)

    replacePageBody(html.body)

    // scroll into place and add history
    if (!isBack) {
      if (url.hash) {
        const el = document.getElementById(decodeURIComponent(url.hash.substring(1)))
        el?.scrollIntoView()
      } else {
        window.scrollTo({ top: 0 })
      }
    }

    // now, patch head, re-executing scripts
    const elementsToRemove = document.head.querySelectorAll(':not([data-persist])')
    elementsToRemove.forEach(el => el.remove())
    const elementsToAdd = html.head.querySelectorAll(':not([data-persist])')
    elementsToAdd.forEach(el => {
      if (el instanceof HTMLScriptElement && el.hasAttribute('data-spa-init')) {
        const script = document.createElement('script')
        script.textContent = el.textContent
        document.head.appendChild(script)
      } else {
        document.head.appendChild(el)
      }
    })

    // delay setting the url until now
    // at this point everything is loaded so changing the url should resolve to the correct addresses
    if (!isBack) {
      history.pushState({}, '', url)
    }
    notifyNav(getFullSlug(window))
    delete announcer.dataset.persist
  }
  if (options.transition === true && !isCodeViewer && document.body.dataset.page !== 'code-viewer')
    await startViewTransition(update)
  else update()
  stopLoading()
}

window.spaNavigate = navigate
window.notifyNav = notifyNav

let activeRouterNavCleanup: (() => void) | null = null

document.addEventListener('nav', () => {
  activeRouterNavCleanup?.()
  activeRouterNavCleanup = null
  transformHostInternalLinks(document)
  const controller = new AbortController()
  let hoverPrefetchTimer = 0
  const clearHoverPrefetch = () => {
    if (hoverPrefetchTimer !== 0) {
      window.clearTimeout(hoverPrefetchTimer)
      hoverPrefetchTimer = 0
    }
  }
  const warmRoute = (event: Event) => {
    const options = getOpts(event)
    if (!options || isSamePage(options.url)) return
    const url = options.url
    clearHoverPrefetch()
    hoverPrefetchTimer = window.setTimeout(() => {
      hoverPrefetchTimer = 0
      prefetchCanonical(url)
    }, 45)
  }
  const onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
      return
    }
    const options = getOpts(event)
    if (!options || isSamePage(options.url)) return
    clearHoverPrefetch()
    prefetchCanonical(options.url)
  }
  const onContentMounted = (event: CustomEventMap['contentdecrypted']) => {
    transformHostInternalLinks(event.detail.content)
  }
  document.addEventListener('pointerover', warmRoute, { signal: controller.signal, passive: true })
  document.addEventListener('pointerout', clearHoverPrefetch, {
    signal: controller.signal,
    passive: true,
  })
  document.addEventListener('focusin', warmRoute, { signal: controller.signal })
  document.addEventListener('pointerdown', onPointerDown as EventListener, {
    signal: controller.signal,
    passive: true,
  })
  document.addEventListener('contentdecrypted', onContentMounted, { signal: controller.signal })
  const cleanupRouterNav = () => {
    if (activeRouterNavCleanup === cleanupRouterNav) activeRouterNavCleanup = null
    clearHoverPrefetch()
    controller.abort()
  }
  activeRouterNavCleanup = cleanupRouterNav
  window.addCleanup(cleanupRouterNav)
  window.addCleanup(() => stacked.release())
  queueMicrotask(() => stacked.restore(new URL(window.location.href)))
})

function createRouter() {
  if (typeof window !== 'undefined') {
    window.addEventListener('click', async event => {
      if (event.defaultPrevented) return
      const { url } = getOpts(event) ?? {}
      // dont hijack behaviour, just let browser act normally
      if (
        !url ||
        event.button !== 0 ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.shiftKey
      )
        return
      event.preventDefault()

      if (isSamePage(url) && url.hash) {
        const el = document.getElementById(decodeURIComponent(url.hash.substring(1)))
        el?.scrollIntoView()
        const oldURL = window.location.href
        history.pushState({}, '', url)
        window.dispatchEvent(new HashChangeEvent('hashchange', { oldURL, newURL: url.href }))
        return
      }

      try {
        await navigate(url, false)
      } catch (e) {
        console.error(e)
        window.location.assign(url)
      }
    })

    window.addEventListener('popstate', async event => {
      const local: CustomEventMap['beforepopstate'] = new CustomEvent('beforepopstate', {
        cancelable: true,
      })
      if (!document.dispatchEvent(local)) return
      const { url } = getOpts(event) ?? {}
      if (window.location.hash && window.location.pathname === url?.pathname) return
      try {
        await navigate(new URL(window.location.toString()), true)
      } catch (e) {
        window.location.reload()
        console.error(e)
      }
      return
    })
  }

  return new (class Router {
    go(pathname: RelativeURL) {
      const url = new URL(pathname, window.location.toString())
      return navigate(url, false)
    }

    back() {
      return window.history.back()
    }

    forward() {
      return window.history.forward()
    }
  })()
}

createRouter()
// We will have to trigger first call in spa.inline.ts here
notifyNav(getFullSlug(window))

if (!customElements.get('route-announcer')) {
  const attrs = {
    'aria-live': 'assertive',
    'aria-atomic': 'true',
    style:
      'position: absolute; left: 0; top: 0; clip: rect(0 0 0 0); clip-path: inset(50%); overflow: hidden; white-space: nowrap; width: 1px; height: 1px',
  }

  customElements.define(
    'route-announcer',
    class RouteAnnouncer extends HTMLElement {
      constructor() {
        super()
      }
      connectedCallback() {
        for (const [key, value] of Object.entries(attrs)) {
          this.setAttribute(key, value)
        }
      }
    },
  )
}
