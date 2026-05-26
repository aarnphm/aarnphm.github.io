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

document.addEventListener('nav', () => {
  const zoomables = Array.from(
    document.querySelectorAll<HTMLElement>('[data-zoomable]:not([data-zoomable-bound])'),
  )
  if (zoomables.length === 0) return

  const doc = document as ViewTransitionDocument
  const anchors = new WeakMap<HTMLElement, Anchor>()
  let activeZoomable: HTMLElement | null = null
  let activeBackdrop: HTMLElement | null = null
  let lastFocus: HTMLElement | null = null

  function runWithTransition(target: HTMLElement, mutate: () => void) {
    if (typeof doc.startViewTransition === 'function') {
      target.style.setProperty('view-transition-name', VT_NAME)
      const transition = doc.startViewTransition!(mutate)
      transition.finished.finally(() => {
        target.style.removeProperty('view-transition-name')
      })
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
    const onBackdrop = () => close(target)
    backdrop.addEventListener('click', onBackdrop)

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

  const handlers: Array<{ el: HTMLElement; type: string; fn: EventListener }> = []

  for (const zoomable of zoomables) {
    zoomable.setAttribute('data-zoomable-bound', '')
    zoomable.setAttribute('tabindex', '-1')
    const triggers = zoomable.querySelectorAll<HTMLElement>('[data-zoomable-trigger]')
    for (const trigger of triggers) {
      const onClick = (event: Event) => {
        event.preventDefault()
        event.stopPropagation()
        toggle(zoomable)
      }
      trigger.addEventListener('click', onClick)
      handlers.push({ el: trigger, type: 'click', fn: onClick })
    }
  }

  const onKeydown = (event: KeyboardEvent) => {
    if (event.key !== 'Escape' || !activeZoomable) return
    event.preventDefault()
    close(activeZoomable)
  }
  document.addEventListener('keydown', onKeydown)

  window.addCleanup(() => {
    document.removeEventListener('keydown', onKeydown)
    for (const { el, type, fn } of handlers) el.removeEventListener(type, fn)
    if (activeBackdrop) activeBackdrop.remove()
    if (activeZoomable) {
      activeZoomable.classList.remove('is-zoomed')
      activeZoomable.style.removeProperty('view-transition-name')
      restoreAnchor(activeZoomable)
    }
    for (const zoomable of zoomables) zoomable.removeAttribute('data-zoomable-bound')
    activeZoomable = null
    activeBackdrop = null
    lastFocus = null
  })
})
