import type { Runtime } from './runtime'
import { selectBenchLevel } from './bench'

const CHORD_MS = 1000

export function bindProbeShortcuts(rt: Runtime): void {
  let prefix = false
  let optionHeld = false
  let timer: number | undefined
  const controls = [
    { selector: '.lat-tab[data-tab="bench"]', key: '3', level: false },
    { selector: '.lat-tab[data-tab="features"]', key: 'f', level: false },
    { selector: '.lat-bench-level[data-level="docs"]', key: 'd', level: true },
    { selector: '.lat-bench-level[data-level="rows"]', key: 'c', level: true },
    { selector: '.lat-bench-level[data-level="feats"]', key: 'f', level: true },
  ]
  const editing = () =>
    document.activeElement?.closest(
      'input, textarea, select, [contenteditable]:not([contenteditable="false"])',
    )
  const paint = () => {
    rt.probe.toggleAttribute('data-shortcut-hints', (optionHeld || prefix) && !editing())
    for (const { selector, key, level } of controls) {
      const control = rt.probe.querySelector<HTMLElement>(selector)
      if (!control) continue
      let badge = control.querySelector<HTMLElement>('.lat-probe-key')
      if (!badge) {
        badge = document.createElement('kbd')
        badge.className = 'lat-probe-key'
        badge.setAttribute('aria-hidden', 'true')
        control.append(badge)
      }
      const text = level && !prefix ? `3${key}` : key
      if (badge.textContent !== text) badge.textContent = text
      badge.hidden = prefix && !level
    }
  }
  const cancelPrefix = () => {
    window.clearTimeout(timer)
    timer = undefined
    prefix = false
  }
  const clear = () => {
    cancelPrefix()
    optionHeld = false
    paint()
  }
  const tab = (name: string) =>
    rt.probe.querySelector<HTMLButtonElement>(`.lat-tab[data-tab="${name}"]`)?.click()
  const keydown = (event: KeyboardEvent) => {
    if (
      event.defaultPrevented ||
      event.isComposing ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      editing()
    ) {
      clear()
      return
    }
    optionHeld = event.altKey
    if (event.key === 'Alt') {
      paint()
      return
    }
    const key = event.code || (event.key === '3' ? 'Digit3' : `Key${event.key.toUpperCase()}`)
    const level =
      key === 'KeyD' ? 'docs' : key === 'KeyC' ? 'rows' : key === 'KeyF' ? 'feats' : null
    if (prefix && level) {
      cancelPrefix()
      event.preventDefault()
      event.stopImmediatePropagation()
      if (!event.repeat)
        void selectBenchLevel(rt, level).catch(error => {
          if (!rt.signal.aborted) console.warn('[latent] bench shortcut', error)
        })
    } else {
      cancelPrefix()
      if (event.altKey && ['Digit3', 'KeyF'].includes(key)) {
        event.preventDefault()
        event.stopImmediatePropagation()
        if (!event.repeat) {
          tab(key === 'Digit3' ? 'bench' : 'features')
          if (key === 'Digit3') {
            prefix = true
            timer = window.setTimeout(() => {
              cancelPrefix()
              paint()
            }, CHORD_MS)
          }
        }
      }
    }
    paint()
  }
  paint()
  const observer = new MutationObserver(paint)
  observer.observe(rt.probe, { childList: true, subtree: true })
  document.addEventListener('keydown', keydown, { capture: true, signal: rt.signal })
  document.addEventListener(
    'keyup',
    event => {
      optionHeld = event.altKey
      paint()
    },
    { capture: true, signal: rt.signal },
  )
  document.addEventListener(
    'focusin',
    () => {
      if (editing()) cancelPrefix()
      paint()
    },
    { signal: rt.signal },
  )
  window.addEventListener('blur', clear, { signal: rt.signal })
  document.addEventListener('visibilitychange', clear, { signal: rt.signal })
  rt.signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      window.clearTimeout(timer)
      rt.probe.removeAttribute('data-shortcut-hints')
      rt.probe.querySelectorAll('.lat-probe-key').forEach(badge => badge.remove())
    },
    { once: true },
  )
}
