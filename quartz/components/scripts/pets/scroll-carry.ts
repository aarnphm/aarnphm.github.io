import { trailClockRegion } from '../../visitor-trail/clock-clearance'
import {
  center,
  clamp,
  friendState,
  height,
  margin,
  petState,
  railY,
  transformFor,
  type Rail,
} from './actor-model'
import { render } from './appearance'
import { friendFollow, hideFriend } from './friend-motion'
import { clearGround, stopGround } from './ground-trace'
import { reanchor, relocate } from './perch-motion'
import { dropHop } from './trail-drive'
import { clearTrailActor, pauseTrail, rememberTrail } from './trail-pose'

// The rail mirrors a scroll thumb: the pole (x = 1.32 in pet units) sits on the viewport's right
// edge, and the hands (y = -0.8) follow where the thumb's middle would be, kept on screen.
export const railFor = (): Rail => {
  const view = window.innerHeight
  const bottom = trailClockRegion(document)?.top ?? view
  const span = Math.max(document.documentElement.scrollHeight - view, 1)
  const thumb = Math.max((view * view) / (span + view), 32)
  const hands = 1.2 * petState.unit
  return {
    x: window.innerWidth - 6 - 3.12 * petState.unit,
    from: Math.max(thumb / 2 - hands, margin),
    to: Math.min(view - thumb / 2 - hands, bottom - height() - margin),
    span,
  }
}

// The page started scrolling: let go of the edge and leap onto the rail.
export function startCarry() {
  if (!petState.root || !petState.doc) return
  rememberTrail()
  clearTrailActor()
  const rect = petState.root.getBoundingClientRect()
  petState.hop?.cancel()
  petState.hop = null
  petState.override = null
  petState.trailWalk = null
  petState.perch = null
  petState.resizeObserver?.disconnect()
  stopGround()
  // The pole holds one; the friend waits out the ride and turns up again where the pet lands.
  hideFriend()
  const rail = railFor()
  petState.carry = rail
  petState.root.style.setProperty('--pet-rail-from', `${rail.from.toFixed(1)}px`)
  petState.root.style.setProperty('--pet-rail-to', `${rail.to.toFixed(1)}px`)
  petState.root.classList.add('is-carried')
  const end = `translate3d(${rail.x.toFixed(1)}px, 0, 0)`
  petState.root.style.transform = end
  // A new scroll-driven animation applies only from the next frame; until then the inline offset
  // holds the pet on the rail, or it flashes at the top of the view for a frame.
  petState.root.style.translate = `0 ${railY(rail).toFixed(1)}px`
  render()
  // The rail's own offset keeps moving with the scroll during the leap, so the leap animates only
  // the gap between where the pet was and the rail.
  // The scroll that started this may already have carried it out of view; it sets off from the
  // nearest edge of the view instead, so the flight up starts where it can be seen.
  const dx = rect.left - rail.x
  const dy = clamp(rect.top, 0, window.innerHeight - rect.height) - railY(rail)
  const d = Math.hypot(dx, dy)
  if (d < 2) return
  if (dy > 60) return jetTo(dx, dy, end)
  const anim = petState.root.animate(
    [
      { transform: `translate3d(${dx.toFixed(1)}px, ${dy.toFixed(1)}px, 0) ${end}` },
      { transform: end },
    ],
    { duration: clamp(140 + d * 0.3, 180, 420), easing: 'cubic-bezier(0.2, 0, 0, 1)' },
  )
  petState.hop = anim
  anim.onfinish = () => {
    if (petState.hop !== anim) return
    anim.cancel()
    petState.hop = null
  }
}

// Well below the rail, it straps on a jetpack and flies up to it: rising first, then veering over
// onto the pole, wobbling a little on the way. (dx, dy) is where it starts, relative to the rail.
function jetTo(dx: number, dy: number, end: string) {
  if (!petState.root) return
  const steps = 16
  const frames = Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps
    const u = 1 - t
    // A quadratic bow through (dx, 0): straight up to the rail's height, then across onto it.
    const x = u * u * dx + 2 * u * t * dx + Math.sin(t * Math.PI * 5) * 3 * u
    const y = u * u * dy
    return { transform: `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) ${end}` }
  })
  petState.override = { state: 'jet', until: Infinity }
  render()
  const anim = petState.root.animate(frames, {
    duration: clamp(380 + Math.hypot(dx, dy) * 0.9, 600, 1400),
    easing: 'cubic-bezier(0.45, 0, 0.25, 1)',
  })
  petState.hop = anim
  anim.onfinish = () => {
    if (petState.hop !== anim) return
    anim.cancel()
    petState.hop = null
    petState.override = null
    render()
    // Scrolling may have stopped mid-flight; let go only once it has landed.
    if (petState.landAfterJet) {
      petState.landAfterJet = false
      endCarry()
    }
  }
}

// Scrolling stopped: back into the page's coordinates, then down or over to the nearest edge.
export function endCarry() {
  if (!petState.root || !petState.carry) return
  if (petState.hop && petState.override?.state === 'jet') {
    petState.landAfterJet = true
    return
  }
  petState.hop?.cancel()
  petState.hop = null
  const at = { x: petState.carry.x, y: railY(petState.carry) }
  petState.carry = null
  petState.doc = { x: at.x + window.scrollX, y: at.y + window.scrollY }
  petState.root.classList.remove('is-carried')
  petState.root.style.translate = ''
  petState.root.style.transform = transformFor(petState.doc)
  relocate(center(), null, 'chute')
}

// Old document coordinates can enlarge the page before the delayed edge checks run.
export const onViewportWidthChange = () => {
  if (!petState.root) return
  if (petState.trailWalk) pauseTrail()
  dropHop()
  clearGround()
  for (const heart of document.querySelectorAll('.garden-pet-heart')) heart.remove()
  if (petState.carry) {
    petState.carry = railFor()
    petState.root.style.setProperty('--pet-rail-from', `${petState.carry.from.toFixed(1)}px`)
    petState.root.style.setProperty('--pet-rail-to', `${petState.carry.to.toFixed(1)}px`)
    petState.root.style.transform = `translate3d(${petState.carry.x.toFixed(1)}px, 0, 0)`
    petState.root.style.translate = `0 ${railY(petState.carry).toFixed(1)}px`
  } else {
    if (petState.doc) {
      petState.doc = {
        ...petState.doc,
        x: clamp(
          petState.doc.x,
          window.scrollX + margin,
          window.scrollX +
            window.innerWidth -
            margin -
            petState.width * (petState.perch?.scale ?? 1),
        ),
      }
      petState.root.style.transform = transformFor(petState.doc)
    }
    reanchor()
    if (friendState.friend && petState.perch && petState.doc)
      friendFollow(petState.perch, petState.doc, 'place')
  }
  render()
}
