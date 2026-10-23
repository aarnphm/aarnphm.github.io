import {
  center,
  clamp,
  edgeY,
  height,
  margin,
  onClock,
  petState,
  reducedMotion,
  viewTop,
  type Perch,
} from './actor-model'
import {
  armQuietTimer,
  play,
  render,
  showBubble,
  state,
  syncTrailArt,
  waitAfterMs,
  weatherState,
  workAfterMs,
} from './appearance'
import { friendWatch } from './friend-motion'
import { moveTo, reanchor, recheckPerch, relocate } from './perch-motion'
import { boxOf, coversText, floorClear, floorReach, inFlow, placeOn } from './perch-placement'
import { petTrail } from './trail-geometry'
import { reserveTrailFoot } from './trail-pose'
import { walkTrail } from './trail-walk'

const hydrationReminderIntervalMs = 45 * 60 * 1000
const hydrationReminderVisibleMs = 22 * 1000

const dwellMs = 1400
const dwellReach = 420
const relocateCooldownMs = 6000

const startleSpeed = 2600
const startleRadius = 150

// Left alone, the pet gets up now and then, laptop or not: mostly a stroll along its own edge, a
// sitter stands up first, and sometimes it hops off to another edge nearby.
const wander = () => {
  const now = performance.now()
  const s = state(now)
  const free = s === 'idle' || s === 'wait' || s === 'work'
  if (
    !free ||
    now < petState.scrollQuietUntil ||
    now - petState.lastMove < 3000 ||
    petState.hop ||
    petState.carry ||
    reducedMotion() ||
    document.hidden
  )
    return
  // The modal makes the page inert; scanning its geometry can stall the active dialog.
  if (document.querySelector('dialog:modal')) return
  if (document.body.dataset.slides === 'true') return
  // A sunny hot day makes it lazy: it mostly stays put, so it gets to lie out on its lounger.
  if (weatherState.beach && Math.random() < 0.75) return
  petState.settled = now
  armQuietTimer()
  if (petState.trailWalk) {
    walkTrail()
    return
  }
  const note = Math.random() < 0.4 ? sidenoteInView() : null
  if (note) return readSidenote(note)
  if (Math.random() < 0.6 && walkTrail()) return
  if (
    !petState.perch ||
    onClock() ||
    petState.perch.kind === 'hang' ||
    (petState.perch.kind === 'floor' && petState.perch.foot) ||
    // It stays on a triathlon window until the window closes, strolling along its top.
    (Math.random() < 0.25 && !petState.perch.el?.matches(triWindowSelector))
  )
    return relocate(center(), petState.perch?.el ?? null)
  const wanderTo = (next: Perch, how: 'stroll' | null = null) => {
    if (next.el && !petTrail(next.el)) {
      const target = placeOn(next)
      const scale = next.scale ?? 1
      const at = {
        x: target.x - window.scrollX + (petState.width * scale) / 2,
        y: target.y - window.scrollY + (edgeY[next.kind] - viewTop) * petState.unit * scale,
      }
      if (coversText({ ...next, at, score: 0 })) return
    }
    moveTo(next, how)
  }
  if (petState.perch.kind === 'sit') return wanderTo({ ...petState.perch, kind: 'stand' })
  const span = petState.perch.el ? boxOf(petState.perch.el).width : window.innerWidth
  let dir = Math.random() < 0.5 ? -1 : 1
  const step = (70 + Math.random() * 170) / Math.max(span, 1)
  if (petState.perch.t + dir * step > 0.96 || petState.perch.t + dir * step < 0.04) dir = -dir
  let t = clamp(petState.perch.t + dir * step, 0.04, 0.96)
  // On the floor it stops short of the first stretch that would put it over text.
  if (!petState.perch.el) {
    const here = center().x
    if (!floorClear(here)) return relocate(center(), null)
    t = floorReach(here, t * span) / span
  }
  if (Math.abs(t - petState.perch.t) * span < 30) return
  wanderTo({ ...petState.perch, t }, 'stroll')
}
petState.wander = wander

// A sidenote out in the margin and well inside the view, for the pet to go and read.
const sidenoteInView = () =>
  Array.from(
    document.querySelectorAll('.sidenote-content:is(.sidenote-left, .sidenote-right)'),
  ).find(el => {
    const r = el.getBoundingClientRect()
    return r.width > 60 && r.top > 80 && r.top < window.innerHeight - 120 && inFlow(el)
  }) ?? null

// Climbs onto the sidenote, reads it with a thought going, then gets straight back to the laptop.
const readSidenote = (note: Element) => {
  petState.afterMove = () => {
    const readMs = 4200 + Math.random() * 2400
    // Its quiet is set to have run out just as it finishes reading, so the laptop comes back out.
    petState.settled = performance.now() + readMs - workAfterMs - 200
    play('think', readMs)
    armQuietTimer()
  }
  moveTo({ el: note, kind: 'stand', t: 0.3 + Math.random() * 0.4 })
}

// The eyes follow the pointer round: they slide toward it along its direction,
// easing back to centre as it comes close, and the body tips a few degrees its way. The pet turns
// to face it once it is clearly to one side.
export const watch = () => {
  petState.watchFrame = 0
  friendWatch()
  const target = petState.lookAt ?? petState.pointer
  if (!petState.root || !target || petState.hop) return
  const c = center()
  const dx = target.x - c.x
  const dy = target.y - c.y
  if (Math.abs(dx) > 40 && !petState.carry) petState.face = dx > 0 ? -1 : 1
  const r = Math.hypot(dx, dy)
  const k = Math.min(1, r / (3 * petState.unit)) / Math.max(r, 1)
  // In the pet's own frame, which is mirrored when it faces right.
  const ux = petState.face * dx * k
  const uy = dy * k
  // Hundredths of a body radius (about a fifth of a pixel), so a still pointer repaints nothing.
  // The eyes sit high and to one side, so they travel less upward, where the outline is nearest.
  const look = `${petState.face} ${(ux * 0.18).toFixed(2)} ${(uy * (uy < 0 ? 0.11 : 0.16)).toFixed(2)} ${petState.carry ? 0 : (ux * 7).toFixed(1)}`
  if (look === petState.gaze) return
  petState.gaze = look
  const [f, lx, ly, tilt] = look.split(' ')
  petState.gazeArt?.face(Number(f))
  petState.gazeArt?.look(lx, ly, tilt)
  syncTrailArt()
  if (petState.trailWalk && petState.perch?.foot && !petState.hop)
    reserveTrailFoot(petState.trailWalk.path, {
      ...petState.perch.foot,
      scale: petState.perch.scale ?? 1,
      progress: petState.perch.progress ?? 0,
    })
}

// Triathlon panels open as windows over the page, unless set to full screen. The pet looks up at
// one as it opens, hops onto its top edge, and jumps back down when it closes.
const triWindows = [
  ['tri-analytics-open', '.tri-analytics'],
  ['tri-map-open', '.tri-map'],
  ['tri-calendar-open', '.tri-calendar-panel'],
  ['tri-calc-open', '.tri-calc'],
] as const
const triWindowSelector = triWindows.map(([, selector]) => selector).join(', ')

const openTriWindow = (): HTMLElement | null => {
  for (const tri of document.querySelectorAll<HTMLElement>(
    '.triathlon:not([data-tri-view]):not(.tri-panels-fullscreen)',
  )) {
    if (tri.closest('.sidepanel-container')) continue
    for (const [open, selector] of triWindows)
      if (tri.classList.contains(open)) return tri.querySelector<HTMLElement>(selector)
  }
  return null
}

// The windows start about 7vh from the top of the view, so the pet shrinks to fit above one.
const windowPerch = (win: HTMLElement): Perch => {
  const rect = win.getBoundingClientRect()
  const scale = clamp((rect.top - 4) / ((edgeY.stand - viewTop) * petState.unit), 0.55, 1)
  const t = clamp((center().x - rect.left) / rect.width, 0.1, 0.9)
  return { el: win, kind: 'stand', t, scale }
}

export const onTriWindows = () => {
  if (!petState.root || !petState.doc || petState.carry || reducedMotion()) return
  window.clearTimeout(petState.windowTimer)
  const win = openTriWindow()
  const on = petState.perch?.el?.matches(triWindowSelector) ? petState.perch.el : null
  if (win === on) return
  if (on) {
    petState.closing = on
    if (!win) relocate(center(), on)
    petState.closing = null
  }
  if (!win) {
    petState.lookAt = null
    return
  }
  // It opens with a short scale-up; the pet stares at its top edge meanwhile, then hops on.
  const rect = win.getBoundingClientRect()
  petState.lookAt = { x: rect.left + rect.width / 2, y: rect.top }
  play('notice', 560)
  watch()
  petState.windowTimer = window.setTimeout(() => {
    petState.lookAt = null
    if (!petState.root || petState.carry || openTriWindow() !== win) return
    // Once up there, its eyes go back to the pointer.
    petState.afterMove = () => {
      petState.gaze = ''
      watch()
    }
    moveTo(windowPerch(win))
  }, 520)
}

const onDwell = () => {
  if (
    !petState.root ||
    !petState.pointer ||
    petState.hop ||
    petState.carry ||
    petState.hovering ||
    reducedMotion()
  )
    return
  if (
    performance.now() < petState.relocateReadyAt ||
    petState.perch?.el?.matches(triWindowSelector)
  )
    return
  if (document.body.dataset.slides === 'true') return
  const c = center()
  if (Math.hypot(petState.pointer.x - c.x, petState.pointer.y - c.y) <= dwellReach) return
  petState.relocateReadyAt = performance.now() + relocateCooldownMs
  relocate(petState.pointer, null)
}

export const onPointerMove = (event: PointerEvent) => {
  if (event.pointerType === 'touch' || !petState.root) return
  const now = performance.now()
  petState.pointer = { x: event.clientX, y: event.clientY }
  if (petState.carry) {
    petState.watchFrame ||= requestAnimationFrame(watch)
    return
  }
  if (petState.lastPointer && now > petState.lastPointer.t && !petState.hop) {
    const speed =
      (Math.hypot(
        petState.pointer.x - petState.lastPointer.x,
        petState.pointer.y - petState.lastPointer.y,
      ) /
        (now - petState.lastPointer.t)) *
      1000
    if (speed > startleSpeed && now > petState.startleReadyAt) {
      // The scroll offsets in center() can flush pending page layout.
      const c = center()
      if (Math.hypot(petState.pointer.x - c.x, petState.pointer.y - c.y) < startleRadius) {
        petState.startleReadyAt = now + 5000
        play('failed', 1600)
      }
    }
  }
  petState.lastPointer = { ...petState.pointer, t: now }
  const wasQuiet = now - petState.lastMove > waitAfterMs
  petState.lastMove = now
  armQuietTimer()
  if (wasQuiet) render()
  petState.watchFrame ||= requestAnimationFrame(watch)
  window.clearTimeout(petState.dwellTimer)
  petState.dwellTimer = window.setTimeout(onDwell, dwellMs)
}

// Touch has no resting pointer, so the pet heads for the edge nearest wherever the reader taps.
export const onPointerDown = (event: PointerEvent) => {
  if (event.pointerType !== 'touch' || !petState.root || petState.hop) return
  if (petState.root.contains(event.target as Node)) return
  petState.pointer = { x: event.clientX, y: event.clientY }
  petState.lastMove = performance.now()
  armQuietTimer()
  if (!reducedMotion()) relocate(petState.pointer, null)
}

// Once a resize (or, with reduced motion, a scroll) settles, an edge that left the view sends the
// pet to one in view.
export const onSettle = () => {
  if (!petState.root || petState.hop) return
  const y = petState.doc ? petState.doc.y - window.scrollY : Infinity
  if (y >= -margin && y + height(petState.perch?.scale) <= window.innerHeight + margin) {
    reanchor()
    recheckPerch()
    return
  }
  const anchor = petState.pointer ?? { x: window.innerWidth * 0.75, y: window.innerHeight * 0.6 }
  relocate(anchor, null)
}

export const hideReminder = () => {
  window.clearTimeout(petState.reminderHideTimer)
  petState.reminding = false
  showBubble()
}

const showReminder = () => {
  if (!petState.root) return
  petState.reminding = true
  showBubble()
  window.clearTimeout(petState.reminderHideTimer)
  petState.reminderHideTimer = window.setTimeout(hideReminder, hydrationReminderVisibleMs)
}

export const scheduleReminder = () => {
  if (petState.reminderTimer !== undefined) return
  petState.reminderTimer = window.setTimeout(() => {
    petState.reminderTimer = undefined
    showReminder()
    scheduleReminder()
  }, hydrationReminderIntervalMs)
}
