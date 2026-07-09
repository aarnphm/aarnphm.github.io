import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom'
import { rootNavSignal } from './root-lifecycle'

const TRIGGER_CLASS = 'breadcrumb-overflow-trigger'
const TEMPLATE_SELECTOR = '[data-overflow-menu]'
const FLOATING_CLASS = 'breadcrumb-overflow-floating'
const BOUND_ATTRIBUTE = 'data-overflow-bound'

function mountFloating(
  trigger: HTMLElement,
  template: HTMLElement,
  onClose: () => void,
  signal: AbortSignal,
) {
  const clone = template.cloneNode(true)
  if (!(clone instanceof HTMLElement)) throw new Error('Breadcrumb menu clone is not an element')
  const floating = clone
  floating.removeAttribute('hidden')
  floating.removeAttribute('data-overflow-menu')
  floating.removeAttribute('aria-hidden')
  floating.classList.add(FLOATING_CLASS)
  floating.setAttribute('role', 'menu')
  floating.setAttribute('tabindex', '-1')
  floating.style.position = 'fixed'
  floating.style.visibility = 'hidden'

  floating.querySelectorAll<HTMLAnchorElement>('a').forEach(anchor => {
    anchor.setAttribute('role', 'menuitem')
  })

  document.body.appendChild(floating)
  trigger.setAttribute('aria-expanded', 'true')

  let mounted = true

  const update = async () => {
    const { x, y } = await computePosition(trigger, floating, {
      placement: 'bottom-start',
      strategy: 'fixed',
      middleware: [offset(8), flip(), shift({ padding: 8 })],
    })

    if (!mounted) return
    Object.assign(floating.style, { left: `${x}px`, top: `${y}px`, visibility: 'visible' })
  }

  const cleanupAutoUpdate = autoUpdate(trigger, floating, update)
  void update()

  function removeFloating() {
    if (!mounted) return
    mounted = false
    cleanupAutoUpdate()
    floating.remove()
    trigger.setAttribute('aria-expanded', 'false')
    document.removeEventListener('pointerdown', onPointerDown, true)
    floating.removeEventListener('click', onItemClick)
    floating.removeEventListener('keydown', onFloatingKeydown)
    signal.removeEventListener('abort', removeFloating)
    onClose()
  }

  const onPointerDown = (event: PointerEvent) => {
    const target = event.target
    if (!(target instanceof Node)) return
    if (floating.contains(target) || trigger.contains(target)) return
    removeFloating()
  }

  const onItemClick = () => removeFloating()
  const onFloatingKeydown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      removeFloating()
      trigger.focus()
    }
  }

  document.addEventListener('pointerdown', onPointerDown, { capture: true, signal })
  floating.addEventListener('click', onItemClick, { signal })
  floating.addEventListener('keydown', onFloatingKeydown, { signal })
  signal.addEventListener('abort', removeFloating, { once: true })

  const firstLink = floating.querySelector<HTMLAnchorElement>('a')
  if (firstLink) {
    requestAnimationFrame(() => firstLink.focus())
  }

  return removeFloating
}

function setupOverflow(container: HTMLElement, signal: AbortSignal) {
  if (container.getAttribute(BOUND_ATTRIBUTE) === 'true') return

  const trigger = container.querySelector<HTMLButtonElement>(`.${TRIGGER_CLASS}`)
  const template = container.querySelector<HTMLElement>(TEMPLATE_SELECTOR)
  if (!trigger || !template) return

  container.setAttribute(BOUND_ATTRIBUTE, 'true')
  template.setAttribute('aria-hidden', 'true')

  let closeFloating: (() => void) | null = null

  const toggle = () => {
    if (closeFloating) {
      closeFloating()
      return
    }

    closeFloating = mountFloating(
      trigger,
      template,
      () => {
        closeFloating = null
      },
      signal,
    )
  }

  trigger.addEventListener(
    'click',
    event => {
      event.preventDefault()
      event.stopPropagation()
      toggle()
    },
    { signal },
  )

  trigger.addEventListener(
    'keydown',
    event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        toggle()
      } else if (event.key === 'Escape' && closeFloating) {
        event.preventDefault()
        closeFloating()
      }
    },
    { signal },
  )

  signal.addEventListener('abort', () => {
    closeFloating?.()
    closeFloating = null
    container.removeAttribute(BOUND_ATTRIBUTE)
    trigger.setAttribute('aria-expanded', 'false')
  })
}

function ensureBreadcrumbContainer(): HTMLElement | null {
  const existing = document.querySelector<HTMLElement>('.breadcrumb-container')
  if (existing) {
    return existing
  }

  const header = document.querySelector<HTMLElement>('.header-content')
  if (!header) {
    return null
  }

  const container = document.createElement('nav')
  container.classList.add('breadcrumb-container')
  container.setAttribute('aria-label', 'breadcrumbs')
  header.insertAdjacentElement('afterbegin', container)
  return container
}

function hydrateBreadcrumbs() {
  ensureBreadcrumbContainer()
  document
    .querySelectorAll<HTMLElement>('.breadcrumb-overflow')
    .forEach(container => setupOverflow(container, rootNavSignal(container)))
}

document.addEventListener('nav', hydrateBreadcrumbs)
document.addEventListener('contentdecrypted', hydrateBreadcrumbs)
