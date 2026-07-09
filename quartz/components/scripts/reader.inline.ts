import { currentNavSignal } from './nav-lifecycle'

let isReaderMode = false
let activeSignal: AbortSignal | undefined

const emitReaderModeChangeEvent = (mode: 'on' | 'off') => {
  const event: CustomEventMap['readermodechange'] = new CustomEvent('readermodechange', {
    detail: { mode },
  })
  document.dispatchEvent(event)
}

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (activeSignal === signal) return
  activeSignal = signal

  signal.addEventListener(
    'abort',
    () => {
      if (activeSignal === signal) activeSignal = undefined
    },
    { once: true },
  )

  const switchReaderMode = () => {
    isReaderMode = !isReaderMode
    const newMode = isReaderMode ? 'on' : 'off'
    document.documentElement.setAttribute('reader-mode', newMode)
    emitReaderModeChangeEvent(newMode)
  }

  const shortcutHandler = (e: HTMLElementEventMap['keydown']) => {
    if (e.key === 'b' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      switchReaderMode()
    }
  }

  document.addEventListener('keydown', shortcutHandler, { signal })

  document.documentElement.setAttribute('reader-mode', isReaderMode ? 'on' : 'off')
})
