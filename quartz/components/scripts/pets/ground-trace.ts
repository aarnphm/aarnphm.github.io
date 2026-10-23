import { hand, pressure, stroke } from '../../../util/mascot'
import { petState, viewTop, type Point } from './actor-model'

// Moves the pet onto `next`: a run or a stroll along the same edge, an arcing hop to a nearby one, a
// jetpack flight on a gentle curve after scrolling, or a drop from above when the old spot is far
// away or out of view. One Web Animation on the root each time.
// Walking the floor, the pet inks its own ground as it goes: a pen line along its path, laid in
// short pieces that appear as its feet reach them, hold while it passes, and fade out slowly behind
// it. Pieces it never reached (the walk was cut short) are dropped.
type Ground = { svg: SVGSVGElement; pieces: { anim: Animation; delay: number }[] }
const grounds = new Set<Ground>()
const groundPiece = 24
const groundFadeMs = 2800

export function layGround(from: Point, to: Point, duration: number) {
  if (!petState.root) return
  // The pet's own ground line sits 1.08 units down; centres, not corners, walk the floor.
  const y = to.y + (1.08 - viewTop) * petState.unit
  const x0 = from.x + petState.width / 2
  const x1 = to.x + petState.width / 2
  const span = Math.abs(x1 - x0)
  if (span < groundPiece) return
  const pad = 6
  const left = Math.min(x0, x1) - pad
  const ns = 'http://www.w3.org/2000/svg'
  const el = document.createElementNS(ns, 'svg')
  el.setAttribute('class', 'pet-ground')
  el.setAttribute('aria-hidden', 'true')
  el.setAttribute('viewBox', `${left} ${y - pad} ${span + 2 * pad} ${2 * pad}`)
  Object.assign(el.style, {
    position: 'absolute',
    left: `${left}px`,
    top: `${y - pad}px`,
    width: `${span + 2 * pad}px`,
    height: `${2 * pad}px`,
    pointerEvents: 'none',
    zIndex: 'var(--z-popover)',
    overflow: 'visible',
  })
  // One hand-drawn line, cut into pieces in walking order; each piece keeps the whole line's pen
  // pressure, so the joins don't bead.
  const line = hand(
    [
      [x0, y],
      [x1, y],
    ],
    4,
    0.6,
    Math.floor(Math.random() * 1e4),
  )
  const pen = pressure(Math.floor(Math.random() * 1e4), 0.064 * petState.unit, 0.3)
  const n = line.length - 1
  const per = Math.max(2, Math.round((groundPiece / span) * n))
  const speed = span / duration
  // Held until the pet's own ground line (2.4 units wide) has passed it.
  const hold = 500 + (2.4 * petState.unit) / speed
  const ground: Ground = { svg: el, pieces: [] }
  for (let i = 0; i < n; i += per) {
    const j = Math.min(n, i + per)
    const g = document.createElementNS(ns, 'g')
    g.setAttribute('style', 'opacity:0')
    g.innerHTML = stroke(line.slice(i, j + 1), t => pen((i + t * (j - i)) / n), 'var(--dark)')
    el.append(g)
    const delay = (duration * i) / n
    const total = hold + groundFadeMs
    const anim = g.animate(
      [
        { opacity: 0 },
        { opacity: 1, offset: 0.01 },
        { opacity: 1, offset: hold / total },
        { opacity: 0 },
      ],
      { duration: total, delay, fill: 'both', easing: 'linear' },
    )
    ground.pieces.push({ anim, delay })
  }
  petState.root.before(el)
  grounds.add(ground)
  void Promise.all(ground.pieces.map(p => p.anim.finished))
    .catch(() => {})
    .finally(() => {
      el.remove()
      grounds.delete(ground)
    })
}

// The walk was cut short: pieces the pet never reached never appear; the rest fade as planned.
export function stopGround() {
  for (const ground of grounds)
    for (const { anim, delay } of ground.pieces)
      if ((Number(anim.currentTime) || 0) < delay) anim.cancel()
}

export function clearGround() {
  for (const ground of grounds) {
    for (const { anim } of ground.pieces) anim.cancel()
    ground.svg.remove()
  }
  grounds.clear()
}
