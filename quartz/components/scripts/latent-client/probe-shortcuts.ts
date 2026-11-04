import type { Runtime } from './runtime'
import { selectBenchLevel } from './bench'

const FEATURE_CHORD_MS = 1000

export function bindProbeShortcuts(rt: Runtime): void {
  let featurePrefixUntil = 0
  const tab = (name: string) =>
    rt.probe.querySelector<HTMLButtonElement>(`.lat-tab[data-tab="${name}"]`)
  const level = (name: 'docs' | 'rows' | 'feats') => {
    tab('bench')?.click()
    void selectBenchLevel(rt, name).catch(error => {
      if (!rt.signal.aborted) console.warn('[latent] bench shortcut', error)
    })
  }
  document.addEventListener(
    'keydown',
    event => {
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        (event.target instanceof Element &&
          event.target.closest(
            'input, textarea, select, [contenteditable]:not([contenteditable="false"])',
          ))
      ) {
        featurePrefixUntil = 0
        return
      }
      // Option changes event.key on macOS (f becomes ƒ); code keeps the physical shortcut.
      const key = event.code || (event.key === '3' ? 'Digit3' : `Key${event.key.toUpperCase()}`)
      const secondF = key === 'KeyF' && performance.now() < featurePrefixUntil
      if (!event.altKey && !secondF) {
        if (event.key !== 'Alt') featurePrefixUntil = 0
        return
      }
      let action: (() => void) | undefined
      if (secondF) action = () => level('feats')
      else if (key === 'KeyA') action = () => tab('axes')?.click()
      else if (key === 'Digit3') action = () => tab('bench')?.click()
      else if (key === 'KeyF') action = () => tab('features')?.click()
      else if (key === 'KeyD') action = () => level('docs')
      else if (key === 'KeyC') action = () => level('rows')
      if (!action) {
        featurePrefixUntil = 0
        return
      }
      event.preventDefault()
      event.stopImmediatePropagation()
      if (event.repeat) return
      featurePrefixUntil = key === 'KeyF' && !secondF ? performance.now() + FEATURE_CHORD_MS : 0
      action()
    },
    { capture: true, signal: rt.signal },
  )
}
