import { writePetSpot } from '../../../util/pet-spot'
import {
  center,
  clamp,
  edgeY,
  friendState,
  height,
  margin,
  onClock,
  petState,
  reducedMotion,
  scheduleWander,
  transformFor,
  viewTop,
  type Perch,
  type Point,
} from './actor-model'
import { play, render } from './appearance'
import { friendFollow, friendPage, friendSpot, hideFriend } from './friend-motion'
import { layGround, stopGround } from './ground-trace'
import {
  coversText,
  floorNear,
  petObstacles,
  pickEdge,
  placeOn,
  type PlacementMeasurements,
} from './perch-placement'
import { hopArc } from './trail-clearance'
import { petTrail } from './trail-geometry'
import { clearTrailActor, leaveTrail, pauseTrail } from './trail-pose'

export const savePosition = () => writePetSpot(center())

export const observePerch = () => {
  petState.resizeObserver?.disconnect()
  if (!petState.perch?.el) return
  petState.resizeObserver ??= new ResizeObserver(() => {
    cancelAnimationFrame(petState.anchorFrame)
    petState.anchorFrame = requestAnimationFrame(reanchor)
  })
  petState.resizeObserver.observe(petState.perch.el)
  petState.resizeObserver.observe(document.body)
}

// Keeps the pet on its edge when the page reflows under it.
export function reanchor() {
  if (!petState.root || !petState.perch || petState.hop || petState.carry) return
  if (petState.perch.el && !petState.perch.el.isConnected) {
    relocate(center(), null)
    return
  }
  const next = placeOn(petState.perch)
  if (petState.doc && Math.hypot(next.x - petState.doc.x, next.y - petState.doc.y) < 1) return
  petState.doc = next
  petState.root.style.transform = transformFor(petState.doc)
  friendFollow(petState.perch, petState.doc, 'place')
  render()
}

export function moveTo(next: Perch, how: 'chute' | 'stroll' | 'fade' | 'leap' | null = null) {
  const after = petState.afterMove
  petState.afterMove = null
  if (!petState.root) return
  // Off the route the move starts where the twin stood, at the size it had there, and grows on
  // the way; leaving first would put the full-size pet at the foot for the first frame.
  let from = petState.doc
  let fromScale = petState.perch?.scale ?? 1
  if (!petTrail(next.el)) {
    if (petState.trailWalk) {
      pauseTrail()
      from = petState.doc
      fromScale = petState.perch?.scale ?? 1
    }
    leaveTrail()
  } else clearTrailActor()
  petState.root.hidden = false
  stopGround()
  petState.gaze = ''
  const target = placeOn(next)
  const same =
    !!from &&
    petState.perch?.el === next.el &&
    petState.perch?.kind === next.kind &&
    next.kind !== 'hang'
  petState.perch = next
  petState.doc = target
  observePerch()
  petState.hop?.cancel()
  petState.hop = null
  const end = transformFor(target)
  if (!from || reducedMotion()) {
    petState.root.style.transform = end
    friendFollow(next, target, from ? 'place' : 'appear')
    render()
    after?.()
    return
  }
  const d = Math.hypot(target.x - from.x, target.y - from.y)
  if (d < 2) {
    petState.root.style.transform = end
    friendFollow(next, target, 'place')
    render()
    after?.()
    return
  }
  // Coming off the rail, it floats down under a parachute to anything below it, and flies its
  // jetpack up or across to anything else.
  const chute = how === 'chute' && target.y - from.y > 40
  const jet = how === 'chute' && !chute
  const fromView = from.y - window.scrollY
  const offscreen = fromView + height() < 0 || fromView > window.innerHeight
  if (Math.abs(target.x - from.x) > 2) petState.face = target.x > from.x ? -1 : 1
  let frames: Keyframe[]
  let duration: number
  if (how === 'fade') {
    frames = [
      { transform: end, opacity: 0 },
      { transform: end, opacity: 1 },
    ]
    duration = 180
  } else if (chute) {
    // Eased down and across, drifting side to side as it swings, the drift dying out as it lands.
    const drift = Math.min(26, d * 0.08)
    const ease = (u: number) => u * u * (3 - 2 * u)
    frames = Array.from({ length: 25 }, (_, k) => {
      const u = k / 24
      const e = ease(u)
      return {
        transform: transformFor({
          x: from.x + (target.x - from.x) * e + Math.sin(u * Math.PI * 2.5) * drift * (1 - u),
          y: from.y + (target.y - from.y) * (0.15 * u + 0.85 * e),
        }),
      }
    })
    duration = clamp((d / 110) * 1000, 1400, 3400)
  } else if (jet) {
    const bulge = Math.min(28, d * 0.12)
    const mid = {
      x: (from.x + target.x) / 2 - ((target.y - from.y) / d) * bulge,
      y: (from.y + target.y) / 2 + ((target.x - from.x) / d) * bulge,
    }
    frames = [
      { transform: transformFor(from) },
      { transform: transformFor(mid), offset: 0.5 },
      { transform: end },
    ]
    duration = clamp((d / 380) * 1000, 320, 1400)
  } else if (offscreen || (d > 900 && how !== 'leap')) {
    frames = [
      { transform: transformFor({ x: target.x, y: target.y - 160 }), opacity: 0 },
      { transform: end, opacity: 1 },
    ]
    duration = 420
  } else if (same && d < 360) {
    frames = [{ transform: transformFor(from) }, { transform: end }]
    duration = clamp((d / (how === 'stroll' ? 60 : 160)) * 1000, 300, 4800)
  } else {
    frames = hopArc(from, target, fromScale, next.scale ?? 1).map((point, index) => ({
      transform: transformFor(point, point.scale),
      offset: index / 2,
      easing: index === 0 ? 'cubic-bezier(0.3, 0.6, 0.5, 1)' : 'cubic-bezier(0.5, 0, 0.7, 0.4)',
    }))
    duration = clamp(320 + d * 0.45, 380, 900)
  }
  const walking = how !== 'fade' && !jet && !chute && frames.length === 2 && !offscreen && d <= 900
  const pace = how === 'stroll' ? 'walk' : 'run'
  if (walking && next.kind === 'floor') layGround(from, target, duration)
  petState.override = {
    state: how === 'fade' ? 'idle' : chute ? 'chute' : jet ? 'jet' : walking ? pace : 'fly',
    until: Infinity,
  }
  render()
  const anim = petState.root.animate(frames, {
    duration,
    easing: jet ? 'ease-in-out' : walking || chute ? 'linear' : undefined,
  })
  petState.hop = anim
  friendFollow(
    next,
    target,
    how === 'fade'
      ? 'place'
      : walking
        ? 'walk'
        : jet || chute || offscreen || d > 900
          ? 'appear'
          : 'hop',
    duration,
  )
  anim.onfinish = () => {
    if (petState.hop !== anim || !petState.root) return
    petState.root.style.transform = end
    anim.cancel()
    petState.hop = null
    petState.override = null
    savePosition()
    if (walking || how === 'fade') render()
    else play('land', 380)
    scheduleWander()
    after?.()
  }
}

// On a slide page the pet keeps to the top-right corner of the slides, sitting on the frame. While
// presenting, the frame fills the screen and leaves only 2rem above the deck, so the pet keeps its
// full size and sits over the deck's top-right corner, legs over the side, its seat dropped as far
// as its head needs. When a slide puts text under it there, it steps out of sight.
const slidesPerch = (seen: PlacementMeasurements): Perch | 'covered' | null => {
  if (document.body.dataset.slides !== 'true') return null
  const frame = document.querySelector<HTMLElement>('.slides-root')
  if (!frame) return null
  const deck = frame.querySelector<HTMLElement>('.slides-deck')
  if (!frame.classList.contains('is-presenting') || !deck) {
    const rect = frame.getBoundingClientRect()
    const scale = clamp((rect.top - 4) / ((edgeY.sit - viewTop) * petState.unit), 0.4, 1)
    const t = clamp(1 - (petState.width * scale * 0.8) / rect.width, 0, 1)
    return { el: frame, kind: 'sit', t, scale }
  }
  const rect = deck.getBoundingClientRect()
  const drop = Math.max(0, margin + (edgeY.sit - viewTop) * petState.unit - rect.top)
  const p: Perch = { el: deck, kind: 'sit', t: 1 + (0.36 * petState.unit) / rect.width, drop }
  const at = { x: placeOn(p).x - window.scrollX + petState.width / 2, y: rect.top + drop }
  return coversText({ ...p, at, score: 0 }, (seen.obstacles ??= petObstacles())) ? 'covered' : p
}

// The page changed under a perched pet: if its edge went away or text now sits under it, it finds
// another edge.
export function recheckPerch() {
  if (
    !petState.root ||
    !petState.perch ||
    !petState.doc ||
    petState.hop ||
    petState.carry ||
    petTrail(petState.perch.el) ||
    onClock()
  )
    return
  const scale = petState.perch.scale ?? 1
  const at = {
    x: petState.doc.x - window.scrollX + (petState.width * scale) / 2,
    y:
      petState.doc.y -
      window.scrollY +
      (edgeY[petState.perch.kind] - viewTop) * petState.unit * scale,
  }
  const gone = !!petState.perch.el && !petState.perch.el.getClientRects().length
  if (gone || coversText({ ...petState.perch!, at, score: 0 }))
    relocate(center(), gone ? null : petState.perch.el)
}

export function relocate(anchor: Point, avoid: Element | null, how: 'chute' | null = null) {
  if (!petState.root) return
  // On the friendship page it lands where the friend fits too, beside it, on its head or along its
  // edge; only when no perch near has that room does it take one the friend has to sit out.
  const before = friendState.friend?.style.visibility ?? ''
  if (friendState.friend) friendState.friend.style.visibility = 'hidden'
  let next: Perch | null
  try {
    // No page writes occur until a perch is chosen. Every fallback shares this one snapshot. The
    // pet stays shown while it measures: every probe already skips the pet's own boxes, and hiding
    // it toggled inherited `visibility` across ~3k SVG nodes, ~20 ms of style per toggle.
    const seen: PlacementMeasurements = {}
    const slides = slidesPerch(seen)
    const fits =
      friendState.friend && friendPage()
        ? (p: Perch) => !!friendSpot(p, placeOn(p), false, seen)
        : undefined
    next =
      slides === 'covered'
        ? null
        : (slides ??
          (fits && (pickEdge(anchor, avoid, fits, seen) ?? floorNear(anchor.x, fits, seen))) ??
          pickEdge(anchor, avoid, undefined, seen) ??
          floorNear(anchor.x, undefined, seen))
  } finally {
    if (friendState.friend) friendState.friend.style.visibility = before
  }
  if (next) moveTo(next, how)
  else {
    petState.root.hidden = true
    hideFriend()
  }
}
