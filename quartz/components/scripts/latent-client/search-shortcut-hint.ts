import type { Runtime } from './runtime'

/** Reveal search keys while Control is held. */
export function bindSearchShortcutHint(rt: Runtime): void {
  const form = rt.links.querySelector<HTMLElement>('.lat-search')
  const input = form?.querySelector<HTMLInputElement>('input')
  const search = document.createElement('kbd')
  search.className = 'lat-search-key'
  search.setAttribute('aria-hidden', 'true')
  form?.append(search)
  input?.setAttribute('aria-keyshortcuts', 'Control+k')
  let controlHeld = false
  const refresh = () => {
    rt.root.toggleAttribute('data-lat-control-hints', controlHeld)
    search.textContent = document.activeElement === input ? 'n / p' : 'k'
  }
  const paint = (event: KeyboardEvent) => {
    controlHeld = event.ctrlKey && !event.altKey && !event.metaKey && !event.shiftKey
    refresh()
  }
  const clear = () => {
    controlHeld = false
    rt.root.removeAttribute('data-lat-control-hints')
  }
  document.addEventListener('keydown', paint, { capture: true, signal: rt.signal })
  document.addEventListener('keyup', paint, { capture: true, signal: rt.signal })
  document.addEventListener('focusin', refresh, { signal: rt.signal })
  window.addEventListener('blur', clear, { signal: rt.signal })
  document.addEventListener('visibilitychange', clear, { signal: rt.signal })
  rt.signal.addEventListener(
    'abort',
    () => {
      clear()
      search.remove()
    },
    { once: true },
  )
}
