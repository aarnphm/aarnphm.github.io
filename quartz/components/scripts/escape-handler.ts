import { currentNavSignal } from './nav-lifecycle'

export interface EscapeHandlerRegistration {
  root: HTMLElement
  callback: () => void
  isActive: () => boolean
  dispose: () => void
}

export interface EscapeHandlerRegistry {
  navSignal?: AbortSignal
  registrations: Map<HTMLElement, EscapeHandlerRegistration>
}

const registry = window.quartzEscapeHandlers ?? {
  registrations: new Map<HTMLElement, EscapeHandlerRegistration>(),
}
window.quartzEscapeHandlers = registry

function defaultIsActive(root: HTMLElement): boolean {
  const active = document.activeElement
  return (
    (active instanceof Node && root.contains(active)) ||
    (root instanceof HTMLDialogElement && root.open) ||
    root.classList.contains('active') ||
    root.getAttribute('aria-expanded') === 'true' ||
    root.getAttribute('aria-hidden') === 'false' ||
    root.querySelector('.active, [aria-expanded="true"]') !== null
  )
}

function ensureEscapeDispatcher(navSignal: AbortSignal): void {
  if (registry.navSignal === navSignal) return
  for (const registration of registry.registrations.values()) registration.dispose()
  registry.navSignal = navSignal

  document.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape') return
      const active = document.activeElement
      let selected: EscapeHandlerRegistration | undefined
      let selectedPriority = 0
      for (const registration of registry.registrations.values()) {
        if (!registration.root.isConnected) {
          registration.dispose()
          continue
        }
        if (!registration.isActive()) continue
        const priority = active instanceof Node && registration.root.contains(active) ? 2 : 1
        if (priority >= selectedPriority) {
          selected = registration
          selectedPriority = priority
        }
      }
      if (!selected) return
      event.preventDefault()
      selected.callback()
    },
    { signal: navSignal },
  )

  navSignal.addEventListener(
    'abort',
    () => {
      if (registry.navSignal === navSignal) registry.navSignal = undefined
    },
    { once: true },
  )
}

export function registerEscapeHandler(
  root: HTMLElement | null,
  callback: () => void,
  isActive?: () => boolean,
): () => void {
  if (!root) return () => {}
  const navSignal = currentNavSignal()
  ensureEscapeDispatcher(navSignal)
  for (const registration of registry.registrations.values()) {
    if (!registration.root.isConnected) registration.dispose()
  }
  registry.registrations.get(root)?.dispose()

  const onOutsideClick = (event: MouseEvent): void => {
    if (event.target !== event.currentTarget) return
    event.preventDefault()
    event.stopPropagation()
    callback()
  }
  root.addEventListener('click', onOutsideClick, { signal: navSignal })

  let disposed = false
  let registration: EscapeHandlerRegistration
  const dispose = (): void => {
    if (disposed) return
    disposed = true
    root.removeEventListener('click', onOutsideClick)
    navSignal.removeEventListener('abort', dispose)
    if (registry.registrations.get(root) === registration) registry.registrations.delete(root)
  }
  registration = { root, callback, isActive: isActive ?? (() => defaultIsActive(root)), dispose }
  registry.registrations.set(root, registration)
  navSignal.addEventListener('abort', dispose, { once: true })
  return dispose
}

export function cleanupEscapeHandlers(root?: Node): void {
  for (const registration of registry.registrations.values()) {
    if (root && registration.root !== root && !root.contains(registration.root)) continue
    registration.dispose()
  }
}
