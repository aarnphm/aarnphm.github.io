import { trailSupported } from '../../../util/trail-preference'
import { friendState, petState } from './actor-model'
import { showBubble } from './appearance'

const hintKey = 'garden:trail-hint-at'
const cooldown = 30 * 60 * 1000

export function mountTrailHint(scene: HTMLElement | null, signal: AbortSignal) {
  if (!scene || !trailSupported() || !matchMedia('(pointer: fine)').matches) return
  try {
    if (Date.now() - Number(sessionStorage.getItem(hintKey)) < cooldown) return
  } catch {}
  let timer = 0
  let hideTimer = 0
  const hide = () => {
    if (!petState.trailHint) return
    petState.trailHint = false
    showBubble()
  }
  const cancel = () => {
    window.clearTimeout(timer)
    window.clearTimeout(hideTimer)
    hide()
  }
  const suggest = () => {
    if (signal.aborted) return
    if (
      document.hidden ||
      !scene.hidden ||
      !petState.root ||
      petState.root.hidden ||
      petState.speaking ||
      petState.reminding ||
      friendState.chat ||
      petState.hop ||
      petState.carry ||
      document.querySelector('dialog[open], .search-container.active, #palette-container.active')
    ) {
      timer = window.setTimeout(suggest, 30_000)
      return
    }
    try {
      sessionStorage.setItem(hintKey, String(Date.now()))
    } catch {}
    petState.trailHint = true
    showBubble()
    hideTimer = window.setTimeout(hide, 10_000)
  }
  scene.addEventListener('trail:open', cancel, { signal })
  document.addEventListener('petstoggle', cancel, { signal })
  document.addEventListener(
    'click',
    event => {
      if (event.target instanceof Node && petState.root?.contains(event.target)) cancel()
    },
    { signal, passive: true },
  )
  document.addEventListener('keydown', event => event.key === 'Escape' && cancel(), { signal })
  document.addEventListener('visibilitychange', () => document.hidden && hide(), { signal })
  signal.addEventListener('abort', cancel, { once: true })
  timer = window.setTimeout(suggest, 18_000)
}
