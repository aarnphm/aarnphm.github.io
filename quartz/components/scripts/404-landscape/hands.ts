import type { Pt } from '../404-pixel'
import type { LandscapeScope } from './scope'
import { alongCat, partUnder, patCat } from '../404-cat'
import { say } from '../404-crew'
import { paintShore } from './layers'

// The keyboard's hand takes STROKE seconds from the cat's face to its tail, and its ring is PAW scene
// units wide. What the button's status line says the cat does.
export const STROKE = 1.25
const PAW = 60

export const SAYS = {
  purr: 'Le chat ronronne.',
  warn: 'Le chat couche les oreilles.',
  swat: "Le chat donne un coup de patte et s'en va.",
  busy: "Le chat ne se laisse pas caresser pour l'instant.",
}

export function tell(st: LandscapeScope, text: string) {
  if (!st.says || !st.paw.on || text === st.said) return
  st.says.textContent = st.said = text
}

// The button sits in its own layer at the near layer's depth, so it is placed in shares of the
// near layer's grid and parallax carries it with the cat.
export function placePaw(st: LandscapeScope) {
  if (!st.petter || !st.glint) return
  const { g } = st.glint
  const [x, y] = alongCat(st.cat, st.paw.along)
  const share = (v: number, from: number, cells: number) =>
    ((((v - from) / g.px + 0.5) / cells) * 100).toFixed(2)
  const next = `left:${share(x, g.sx[0], g.bw)}%;top:${share(y, g.sy[0], g.bh)}%;width:${((PAW / g.px / g.bw) * 100).toFixed(2)}%`
  if (next !== st.placed) st.petter.style.cssText = st.placed = next
}

// A pat at a scene point. Without motion nothing wears its patience or ages the heart: a pat
// shows it content with a heart over it, taken down on a timer.
export function patAt(st: LandscapeScope, spot: Pt) {
  if (!st.reduce) return patCat(st.cat, ...spot, what => told(st, what))
  if (!partUnder(st.cat, ...spot)) return false
  st.cat.petting.joy = 1
  say(st.purr, 'cat', '♥')
  tell(st, SAYS.purr)
  paintShore(st)
  window.clearTimeout(st.purring)
  st.purring = window.setTimeout(() => {
    st.cat.petting.joy = 0
    st.purr.bubbles = []
    paintShore(st)
  }, 1800)
  return true
}

export function told(st: LandscapeScope, what: 'purr' | 'warn' | 'swat') {
  tell(st, SAYS[what])
  // The warning shows on its face.
  if (what === 'warn') return
  if (what === 'swat') return say(st.crew, 'friend', '!')
  say(st.purr, 'cat', '♥')
  if (!st.purred) say(st.crew, 'friend', '?')
  st.purred = true
}

export function handOver(st: LandscapeScope, over: boolean) {
  if (over !== st.stroking) st.scene.classList.toggle('is-stroking', (st.stroking = over))
}

// Client pixels to the scene as the near layer shows it, which drifts under parallax.
export function onNear(st: LandscapeScope, x: number, y: number): Pt | null {
  if (!st.glint || !st.near) return null
  const r = st.near.getBoundingClientRect()
  const { g } = st.glint
  return [
    g.sx[0] + (((x - r.left) / r.width) * g.bw - 0.5) * g.px,
    g.sy[0] + (((y - r.top) / r.height) * g.bh - 0.5) * g.px,
  ]
}
