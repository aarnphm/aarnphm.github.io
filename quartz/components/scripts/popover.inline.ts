import { fetchCanonical } from '../../util/fetch-canonical'
import { getContentType } from '../../util/mime'
import { getFullSlug, isFullSlug, normalizeRelativeURLs } from '../../util/path'
import { currentNavSignal } from './nav-lifecycle'
import { p } from './popover/content-render'
import { clearAllPopovers } from './popover/dom'
import {
  containsRelatedTarget,
  dismissPopover,
  mouseEnterHandler,
  mouseLeaveHandler,
} from './popover/hover-session'
import {
  arxivPdfUrl,
  cleanArxivIdentifier,
  createPdfEmbed,
  PDF_SIDE_PANEL_OPTIONS,
  pdfTitleFromUrl,
  isPdfUrl,
  mountPdfPreviews,
  preloadPopoverLinks,
} from './popover/pdf-preview'
import { clearActivePopover, compareUrls, findHashTarget, popoverState } from './popover/placement'
import { beginSidePanelRequest, disposeSidePanel, getOrCreateSidePanel } from './side-panel'

let stackedPopoverEvents: AbortController | null = null

let delegatedPopoverSignal: AbortSignal | undefined

function closestStackedPopoverLink(
  target: EventTarget | null,
  stacked: HTMLElement,
): HTMLAnchorElement | null {
  if (!target || !(target instanceof Element)) return null
  if (stackedPopoverForTarget(target)) return null
  const link = target.closest('a.internal')
  if (!(link instanceof HTMLAnchorElement)) return null
  if (!stacked.contains(link)) return null
  return link
}

function stackedPopoverForTarget(target: EventTarget | null): HTMLElement | null {
  if (!target || !(target instanceof Element)) return null
  const popover = target.closest('.stacked-popover')
  return popover instanceof HTMLElement ? popover : null
}

function setupStackedPopoverLinks() {
  stackedPopoverEvents?.abort()
  stackedPopoverEvents = null

  const stacked = document.getElementById('stacked-notes-container')
  if (!(stacked instanceof HTMLElement)) return

  const events = new AbortController()
  stackedPopoverEvents = events
  // Let the router read preview links before removing their DOM.
  stacked.addEventListener(
    'click',
    event => {
      if (event.target instanceof Element && event.target.closest('a')) {
        queueMicrotask(dismissPopover)
      }
    },
    { capture: true, signal: events.signal },
  )

  stacked.addEventListener(
    'mouseover',
    event => {
      const link = closestStackedPopoverLink(event.target, stacked)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      if (popoverState.anchor === link && stackedPopoverForTarget(event.relatedTarget)) return
      void mouseEnterHandler.call(link, event)
    },
    { signal: events.signal },
  )

  stacked.addEventListener(
    'mouseout',
    event => {
      const link = closestStackedPopoverLink(event.target, stacked)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      mouseLeaveHandler.call(link, event)
    },
    { signal: events.signal },
  )

  stacked.addEventListener(
    'focusin',
    event => {
      const link = closestStackedPopoverLink(event.target, stacked)
      if (!link?.matches(':focus-visible')) return
      const rect = link.getBoundingClientRect()
      void mouseEnterHandler.call(link, {
        clientX: rect.left + rect.width / 2,
        clientY: rect.top + rect.height / 2,
      })
    },
    { signal: events.signal },
  )
  stacked.addEventListener(
    'focusout',
    event => {
      const link = closestStackedPopoverLink(event.target, stacked)
      if (link) mouseLeaveHandler.call(link, event)
    },
    { signal: events.signal },
  )
}

async function mouseClickHandler(evt: MouseEvent, link: HTMLAnchorElement) {
  if (evt.defaultPrevented || evt.button !== 0 || evt.metaKey || evt.ctrlKey || evt.shiftKey) return
  clearActivePopover()
  if (link.dataset.curiusPopover) return
  if (getFullSlug(window) === 'index') return

  const thisUrl = new URL(document.location.href)
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)
  targetUrl.hash = ''
  targetUrl.search = ''

  const container = document.getElementById('stacked-notes-container')

  if (link.dataset.wikipediaLang && link.dataset.wikipediaTitle) {
    return
  }

  if (link.dataset.lesswrongPostId) {
    return
  }

  if (link.dataset.sepEntry) {
    return
  }

  if (evt.altKey && !container?.classList.contains('active')) {
    evt.preventDefault()
    evt.stopPropagation()

    // derive slug from href if data-slug not set (e.g. generated pages like /tags/*)
    const slug = (link.dataset.slug || targetUrl.pathname).replace(/^\/+|\/+$/g, '') || 'index'
    if (!isFullSlug(slug)) return

    let request: ReturnType<typeof beginSidePanelRequest> | undefined
    try {
      const asidePanel = getOrCreateSidePanel()
      const activeRequest = beginSidePanelRequest(asidePanel)
      request = activeRequest

      if (link.dataset.arxivId) {
        const arxivUrl = arxivPdfUrl(link.dataset.arxivId)
        const pdf = createPdfEmbed(
          arxivUrl.toString(),
          `${cleanArxivIdentifier(link.dataset.arxivId)}.pdf`,
          PDF_SIDE_PANEL_OPTIONS,
        )
        const sideInner = activeRequest.mount(slug, pdf)
        if (!sideInner) return
        mountPdfPreviews(sideInner)
        window.notifyNav(slug)
        return
      }

      let response: Response | void
      const fetchUrl = new URL(link.href)
      fetchUrl.hash = ''
      fetchUrl.search = ''

      if (isPdfUrl(fetchUrl)) {
        const pdf = createPdfEmbed(
          fetchUrl.toString(),
          pdfTitleFromUrl(fetchUrl.toString()),
          PDF_SIDE_PANEL_OPTIONS,
        )
        const sideInner = activeRequest.mount(slug, pdf)
        if (!sideInner) return
        mountPdfPreviews(sideInner)
        window.notifyNav(slug)
        return
      }

      response = await fetchCanonical(fetchUrl, { signal: activeRequest.signal }).catch(error => {
        if (!activeRequest.signal.aborted) console.error(error)
      })

      if (!response) {
        activeRequest.cancel()
        return
      }

      const headerContentType = response.headers.get('Content-Type')
      const contentType = headerContentType
        ? headerContentType.split(';')[0]
        : getContentType(targetUrl)

      if (contentType === 'application/pdf') {
        const src = response.url || fetchUrl.toString()
        const pdf = createPdfEmbed(
          src,
          pdfTitleFromUrl(fetchUrl.toString()),
          PDF_SIDE_PANEL_OPTIONS,
        )
        const sideInner = activeRequest.mount(slug, pdf)
        if (!sideInner) return
        mountPdfPreviews(sideInner)
      } else {
        const contents = await response.text()
        const html = p.parseFromString(contents, 'text/html')
        normalizeRelativeURLs(html, targetUrl)
        html.querySelectorAll('[id]').forEach(el => {
          const targetID = `popover-${el.id}`
          el.id = targetID
        })
        const elts = [
          ...(html.getElementsByClassName('popover-hint') as HTMLCollectionOf<HTMLElement>),
        ]
        if (elts.length === 0) {
          activeRequest.cancel()
          return
        }

        if (!activeRequest.mount(slug, ...elts)) return
      }

      window.notifyNav(slug)
    } catch (error) {
      const wasAborted = request?.signal.aborted ?? false
      request?.cancel()
      if (!wasAborted) console.error('Failed to create side panel:', error)
    }
    return
  }

  if (compareUrls(thisUrl, targetUrl) && hash !== '') {
    const mainContent = document.querySelector('article')
    const heading = mainContent ? findHashTarget(mainContent, hash) : null
    if (!heading) return
    evt.preventDefault()
    heading.scrollIntoView({ behavior: 'smooth' })
    history.pushState(null, '', hash)
  }
}

function closestPopoverLink(target: EventTarget | null): HTMLAnchorElement | null {
  const element =
    target instanceof Element ? target : target instanceof Node ? target.parentElement : null
  const link =
    element?.closest<HTMLAnchorElement>(
      'a.internal, a[data-power-activity-id], a[data-curius-popover]',
    ) ?? null
  return link?.closest('#stacked-notes-container') ? null : link
}

function setupPopoverDelegation(): void {
  const signal = currentNavSignal()
  if (delegatedPopoverSignal === signal) return
  delegatedPopoverSignal = signal

  document.addEventListener(
    'focusin',
    event => {
      const link = closestPopoverLink(event.target)
      if (!link || !link.matches(':focus-visible')) return
      const rect = link.getBoundingClientRect()
      void mouseEnterHandler.call(link, {
        clientX: rect.left + rect.width / 2,
        clientY: rect.top + rect.height / 2,
      })
    },
    { signal },
  )
  document.addEventListener(
    'focusout',
    event => {
      const link = closestPopoverLink(event.target)
      if (link) mouseLeaveHandler.call(link, event)
    },
    { signal },
  )
  document.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape') return
      dismissPopover()
    },
    { signal },
  )
  document.addEventListener(
    'mouseover',
    event => {
      const link = closestPopoverLink(event.target)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      void mouseEnterHandler.call(link, event)
    },
    { signal },
  )
  document.addEventListener(
    'mouseout',
    event => {
      const link = closestPopoverLink(event.target)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      mouseLeaveHandler.call(link, event)
    },
    { signal },
  )
  document.addEventListener(
    'click',
    event => {
      const link = closestPopoverLink(event.target)
      if (link) void mouseClickHandler(event, link)
    },
    { signal },
  )
  signal.addEventListener(
    'abort',
    () => {
      if (delegatedPopoverSignal === signal) delegatedPopoverSignal = undefined
    },
    { once: true },
  )
}

function cleanupPopoverRuntime(): void {
  popoverState.request?.abort()
  popoverState.request = null
  stackedPopoverEvents?.abort()
  stackedPopoverEvents = null
  clearActivePopover()
  clearAllPopovers()
}

document.addEventListener('nav', () => {
  window.addCleanup(disposeSidePanel)
  window.addCleanup(cleanupPopoverRuntime)
  document.addEventListener('curius:preview', dismissPopover, { signal: currentNavSignal() })
  setupPopoverDelegation()
  preloadPopoverLinks()
  setupStackedPopoverLinks()
})

document.addEventListener('contentdecrypted', e => {
  const { content } = e.detail
  if (content) preloadPopoverLinks(content)
})
