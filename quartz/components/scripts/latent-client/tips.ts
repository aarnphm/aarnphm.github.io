import type { Runtime } from './runtime'

const GAP = 5
const EDGE = 8

/**
 * The explorer's hover notes (data-tip) drawn in one fixed box on the root, so a note on a term
 * near a pane's edge draws over the border instead of being cut by the pane's scroll clip. The
 * box opens below the term (above for data-tip-at="up"), flips when that side has no room, and
 * is clamped to the viewport. The CSS ::after notes stay as the fallback until this binds.
 */
export function bindTips(rt: Runtime): void {
  const root = rt.root
  const tip = document.createElement('div')
  tip.className = 'lat-tip'
  tip.setAttribute('role', 'tooltip')
  tip.hidden = true
  root.append(tip)
  root.dataset.tipsLive = ''
  let shown: HTMLElement | null = null

  const hide = () => {
    shown = null
    tip.hidden = true
    delete tip.dataset.open
  }

  const show = (el: HTMLElement) => {
    const text = el.dataset.tip
    if (!text) return
    shown = el
    tip.textContent = text
    tip.hidden = false
    delete tip.dataset.open
    // measure at the top left so the width is the note's own, not what is left of the viewport
    tip.style.left = '0px'
    tip.style.top = '0px'
    const at = el.getBoundingClientRect()
    const box = tip.getBoundingClientRect()
    const below = at.bottom + GAP
    const above = at.top - GAP - box.height
    const wantUp = el.dataset.tipAt === 'up'
    let top = wantUp ? above : below
    if (wantUp && top < EDGE && below + box.height <= innerHeight - EDGE) top = below
    if (!wantUp && top + box.height > innerHeight - EDGE && above >= EDGE) top = above
    let left = el.dataset.tipAt === 'end' ? at.right - box.width : at.left
    left = Math.max(EDGE, Math.min(left, innerWidth - EDGE - box.width))
    tip.style.left = `${Math.round(left)}px`
    tip.style.top = `${Math.round(Math.max(EDGE, top))}px`
    tip.toggleAttribute('data-up', top < at.top)
    requestAnimationFrame(() => {
      if (shown === el) tip.dataset.open = ''
    })
  }

  const termOf = (target: EventTarget | null) =>
    (target as Element | null)?.closest?.<HTMLElement>('[data-tip]') ?? null
  const opts = { signal: rt.signal }

  root.addEventListener(
    'pointerover',
    event => {
      // touch shows a note on focus (a tap), never on the pointer passing over
      if ((event as PointerEvent).pointerType === 'touch') return
      const el = termOf(event.target)
      if (el && el !== shown) show(el)
    },
    { ...opts, passive: true },
  )
  root.addEventListener(
    'pointerout',
    event => {
      if (!shown || (event as PointerEvent).pointerType === 'touch') return
      const next = event.relatedTarget as Node | null
      if (next && shown.contains(next)) return
      if (document.activeElement !== shown) hide()
    },
    { ...opts, passive: true },
  )
  root.addEventListener(
    'focusin',
    event => {
      const el = termOf(event.target)
      if (el) show(el)
    },
    opts,
  )
  root.addEventListener('focusout', () => hide(), opts)
  // a note follows nothing: a scroll that moves its term, or a resize, closes it
  window.addEventListener(
    'scroll',
    event => {
      const moved = event.target
      if (shown && (moved === document || (moved instanceof Node && moved.contains(shown)))) hide()
    },
    { ...opts, capture: true, passive: true },
  )
  window.addEventListener('resize', () => shown && hide(), opts)
  rt.signal.addEventListener('abort', () => {
    tip.remove()
    delete root.dataset.tipsLive
  })
}
