type ViewTransitionDocument = Document & {
  startViewTransition?: (cb: () => void | Promise<void>) => {
    finished: Promise<void>
    ready: Promise<void>
    updateCallbackDone: Promise<void>
    skipTransition: () => void
  }
}

const VT_NAME = 'zoomable-active'
const BACKDROP_CLASS = 'zoomable-backdrop'
const PLACEHOLDER_CLASS = 'zoomable-placeholder'

type Anchor = { parent: Node; nextSibling: Node | null; placeholder: HTMLElement }
type ZoomableRoot = Document | DocumentFragment | Element

const doc = document as ViewTransitionDocument
let anchors = new WeakMap<HTMLElement, Anchor>()
let activeZoomable: HTMLElement | null = null
let activeBackdrop: HTMLElement | null = null
let lastFocus: HTMLElement | null = null
let controller: AbortController | null = null
let boundZoomables = new Set<HTMLElement>()

function runWithTransition(target: HTMLElement, mutate: () => void) {
  if (typeof doc.startViewTransition === 'function') {
    target.style.setProperty('view-transition-name', VT_NAME)
    const transition = doc.startViewTransition(mutate)
    const clearTransitionName = () => {
      target.style.removeProperty('view-transition-name')
    }
    void transition.finished.then(clearTransitionName, clearTransitionName)
  } else {
    mutate()
  }
}

function restoreAnchor(target: HTMLElement) {
  const anchor = anchors.get(target)
  if (!anchor) return
  if (anchor.placeholder.parentNode) {
    anchor.placeholder.parentNode.insertBefore(target, anchor.placeholder)
    anchor.placeholder.remove()
  } else {
    anchor.parent.insertBefore(target, anchor.nextSibling)
  }
  anchors.delete(target)
}

function close(target: HTMLElement) {
  if (!target.classList.contains('is-zoomed')) return
  const backdrop = activeBackdrop
  activeZoomable = null
  activeBackdrop = null
  runWithTransition(target, () => {
    target.classList.remove('is-zoomed')
    restoreAnchor(target)
    backdrop?.remove()
  })
  target.removeAttribute('aria-modal')
  target.removeAttribute('role')
  if (lastFocus && typeof lastFocus.focus === 'function') {
    lastFocus.focus({ preventScroll: true })
  }
  lastFocus = null
}

function open(target: HTMLElement) {
  if (activeZoomable) close(activeZoomable)
  lastFocus = (document.activeElement as HTMLElement | null) ?? null
  const backdrop = document.createElement('div')
  backdrop.className = BACKDROP_CLASS
  backdrop.setAttribute('aria-hidden', 'true')
  backdrop.addEventListener('click', () => close(target), { signal: ensureController().signal })

  const placeholder = document.createElement('div')
  placeholder.className = PLACEHOLDER_CLASS
  placeholder.setAttribute('aria-hidden', 'true')
  const rect = target.getBoundingClientRect()
  placeholder.style.height = `${rect.height}px`

  const parent = target.parentNode as Node | null
  if (!parent) return
  anchors.set(target, { parent, nextSibling: target.nextSibling, placeholder })
  activeZoomable = target
  activeBackdrop = backdrop

  runWithTransition(target, () => {
    parent.insertBefore(placeholder, target)
    document.body.appendChild(backdrop)
    document.body.appendChild(target)
    target.classList.add('is-zoomed')
  })
  target.setAttribute('role', 'dialog')
  target.setAttribute('aria-modal', 'true')
  target.focus({ preventScroll: true })
}

function toggle(target: HTMLElement) {
  if (target.classList.contains('is-zoomed')) close(target)
  else open(target)
}

function onKeydown(event: KeyboardEvent) {
  if (event.key !== 'Escape' || !activeZoomable) return
  event.preventDefault()
  close(activeZoomable)
}

function cleanupZoomables() {
  controller?.abort()
  controller = null
  if (activeBackdrop) activeBackdrop.remove()
  if (activeZoomable) {
    activeZoomable.classList.remove('is-zoomed')
    activeZoomable.style.removeProperty('view-transition-name')
    restoreAnchor(activeZoomable)
  }
  for (const zoomable of boundZoomables) zoomable.removeAttribute('data-zoomable-bound')
  boundZoomables.clear()
  anchors = new WeakMap()
  activeZoomable = null
  activeBackdrop = null
  lastFocus = null
}

function ensureController(): AbortController {
  if (controller) return controller
  controller = new AbortController()
  document.addEventListener('keydown', onKeydown, { signal: controller.signal })
  return controller
}

function zoomableNodes(root: ZoomableRoot): HTMLElement[] {
  const nodes = Array.from(
    root.querySelectorAll<HTMLElement>('[data-zoomable]:not([data-zoomable-bound])'),
  )
  if (root instanceof HTMLElement && root.matches('[data-zoomable]:not([data-zoomable-bound])')) {
    nodes.unshift(root)
  }
  return nodes
}

function setupZoomables(root: ZoomableRoot = document) {
  const zoomables = zoomableNodes(root)
  if (zoomables.length === 0) return
  const signal = ensureController().signal

  for (const zoomable of zoomables) {
    zoomable.setAttribute('data-zoomable-bound', '')
    zoomable.setAttribute('tabindex', '-1')
    boundZoomables.add(zoomable)
    const triggers = zoomable.querySelectorAll<HTMLElement>('[data-zoomable-trigger]')
    for (const trigger of triggers) {
      trigger.addEventListener(
        'click',
        event => {
          event.preventDefault()
          event.stopPropagation()
          toggle(zoomable)
        },
        { signal },
      )
    }
  }
}

document.addEventListener('nav', () => {
  ensureController()
  setupZoomables()
  window.addCleanup(cleanupZoomables)
})

document.addEventListener('contentdecrypted', event => {
  setupZoomables(event.detail.content)
})

document.addEventListener('protectedcontentloaded', event => {
  const detail = event instanceof CustomEvent ? event.detail : null
  const container =
    detail && typeof detail === 'object' && 'container' in detail ? detail.container : null
  if (
    container instanceof Document ||
    container instanceof DocumentFragment ||
    container instanceof Element
  ) {
    setupZoomables(container)
  }
})
