import { petSvg } from '../../util/mascot'

// The garden's pet: the pen-drawn mascot, living on the page's edges. It perches on the nearest box
// it can find (stands or sits with legs dangling on a top edge, hangs from a bottom edge) and stays
// there, watching the pointer. It hops to a new edge only when the reader's pointer rests far away or
// when it gets bored. While the page scrolls it clings to a pole on the right edge and slides along
// it with the scroll thumb; when scrolling stops it swims to the nearest edge. A pointer move costs no layout and
// every move is one compositor animation. Looks and motion live in components/styles/pet.scss; this
// script picks edges and writes the SVG's data-state and data-kind.

type State =
  | 'idle'
  | 'run'
  | 'fly'
  | 'land'
  | 'wave'
  | 'failed'
  | 'wait'
  | 'work'
  | 'happy'
  | 'cheer'
  | 'swim'
  | 'slide'
type Kind = 'stand' | 'sit' | 'hang' | 'floor'
type Point = { x: number; y: number }
// An edge the pet holds: `t` is where its centre sits along the box's width, 0 to 1. The floor kind
// has no box: the pet stands on its own pen line at the bottom of the viewport, the fallback when no
// box nearby has an edge to hold.
type Perch = { el: Element | null; kind: Kind; t: number }
type Edge = Perch & { at: Point; score: number }

const petsEnabledKey = 'garden:pets-enabled:v1'
const positionKey = 'garden:pet-position:v2'
const hydrationReminderIntervalMs = 45 * 60 * 1000
const hydrationReminderVisibleMs = 22 * 1000

// Where the held edge crosses the pet, in mascot units (the viewBox is 3.6 wide, top at -2).
const edgeY: Record<Kind, number> = { stand: 1.42, sit: 0.94, hang: -1.45, floor: 1.42 }
const viewWidth = 3.6
const viewTop = -2
const viewBottom = 1.8

const margin = 8
const dwellMs = 1400
const dwellReach = 420
const relocateCooldownMs = 6000
const waitAfterMs = 7000
const workAfterMs = 20000
const startleSpeed = 2600
const startleRadius = 150
const replaced = new Set(['IMG', 'VIDEO', 'CANVAS', 'IFRAME', 'PRE', 'TABLE', 'HR'])
const headings = new Set(['H1', 'H2', 'H3', 'H4'])
const perchTags =
  '#quartz-root :is(h1, h2, h3, h4, hr, pre, figure, img, video, table, blockquote, .callout), [data-pet-perch]'

let root: HTMLElement | null = null
let svg: SVGSVGElement | null = null
let markup = ''
let width = 72
let unit = width / viewWidth
let origin: Point = { x: 0, y: 0 }
let doc: Point | null = null
// While the page scrolls the pet rides the scroll rail: position: fixed at x, its top running from
// `from` to `to` as the page scrolls from 0 to `span`. pet.scss drives that with a scroll timeline.
type Rail = { x: number; from: number; to: number; span: number }
let carry: Rail | null = null
const scrollTimeline = typeof CSS !== 'undefined' && CSS.supports('animation-timeline', 'scroll()')
let perch: Perch | null = null
let hop: Animation | null = null
let pointer: Point | null = null
let lastPointer: { x: number; y: number; t: number } | null = null
let lastMove = 0
let face = 1
let hovering = false
let reminding = false
let override: { state: State; until: number } | null = null
let startleReadyAt = 0
let relocateReadyAt = 0
let watchFrame = 0
let gaze = ''
let anchorFrame = 0
let stateTimer: number | undefined
let dwellTimer: number | undefined
let scrollTimer: number | undefined
let wanderTimer: number | undefined
let reminderTimer: number | undefined
let reminderHideTimer: number | undefined
let resizeObserver: ResizeObserver | null = null

const clamp = (v: number, min: number, max: number) => Math.min(Math.max(v, min), max)
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
const height = () => (viewBottom - viewTop) * unit

// On unless the reader turned pets off from the palette.
const readPetsEnabled = () => localStorage.getItem(petsEnabledKey) !== 'false'

// The pet's centre in viewport coordinates, from its cached document position: no layout read.
const railY = (rail: Rail) =>
  rail.from + (rail.to - rail.from) * clamp(window.scrollY / rail.span, 0, 1)

const center = (): Point =>
  carry
    ? { x: carry.x + width / 2, y: railY(carry) + height() / 2 }
    : doc
      ? { x: doc.x - window.scrollX + width / 2, y: doc.y - window.scrollY + height() / 2 }
      : { x: window.innerWidth - width, y: window.innerHeight - height() }

const transformFor = (p: Point) =>
  `translate3d(${(p.x - origin.x).toFixed(1)}px, ${(p.y - origin.y).toFixed(1)}px, 0)`

const state = (now: number): State => {
  if (override && now < override.until) return override.state
  if (carry) return 'slide'
  if (hovering) return 'happy'
  if (reminding) return 'wave'
  if (perch?.kind === 'hang') return 'idle'
  const quiet = now - lastMove
  if (quiet > workAfterMs) return 'work'
  if (quiet > waitAfterMs) return 'wait'
  return 'idle'
}

const render = () => {
  if (!root || !svg || (!doc && !carry)) return
  const next = state(performance.now())
  if (svg.dataset.state !== next) svg.dataset.state = next
  svg.dataset.kind = perch?.kind ?? 'stand'
  // On the rail it faces the page, its back to the pole.
  root.style.setProperty('--pet-face', String(carry ? 1 : face))
  root.dataset.bubble = center().y - height() / 2 < 64 ? 'below' : 'above'
}

// Shows `next` for `ms`, then falls back to whatever the pet would otherwise be doing.
const play = (next: State, ms: number) => {
  override = { state: next, until: performance.now() + ms }
  render()
  window.setTimeout(render, ms + 20)
}

// Re-renders when the pointer has been quiet long enough for the pet to fidget, then to work.
const armQuietTimer = () => {
  window.clearTimeout(stateTimer)
  const quiet = performance.now() - lastMove
  const due = quiet < waitAfterMs ? waitAfterMs : quiet < workAfterMs ? workAfterMs : null
  if (due === null) return
  stateTimer = window.setTimeout(
    () => {
      render()
      armQuietTimer()
    },
    due - quiet + 20,
  )
}

// Is `el` part of the document's flow, scrolled by the window? Boxes inside fixed or sticky
// furniture, or inside their own scroller, move without the pet, so it never holds them.
const inFlow = (el: Element) => {
  for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
    const style = getComputedStyle(node)
    if (style.position === 'fixed' || style.position === 'sticky') return false
    if (/(auto|scroll|hidden)/.test(style.overflowY + style.overflowX)) {
      if (node.scrollHeight > node.clientHeight + 1 || node.scrollWidth > node.clientWidth + 1)
        return false
    }
  }
  return true
}

const visibleLine = (style: CSSStyleDeclaration, side: 'Top' | 'Bottom') =>
  parseFloat(style[`border${side}Width`]) > 0 &&
  style[`border${side}Style`] !== 'none' &&
  !/rgba\(.*,\s*0\)$|transparent/.test(style[`border${side}Color`])

const filled = (style: CSSStyleDeclaration) =>
  !/rgba\(.*,\s*0\)$|transparent/.test(style.backgroundColor)

// The box the pet holds on `el`. A heading's box is its first line of text, so the pet stands on
// the letters rather than on the empty end of a full-width block.
function boxOf(el: Element): DOMRect {
  if (!headings.has(el.tagName)) return el.getBoundingClientRect()
  const range = document.createRange()
  range.selectNodeContents(el)
  for (const rect of range.getClientRects()) if (rect.width > 4 && rect.height > 4) return rect
  return el.getBoundingClientRect()
}

// Edges in view near `anchor`. Boxes come from the element stacks under a grid of points across the
// viewport plus a short list of tags worth standing on, so the cost follows the view rather than
// the size of the page.
function edgesNear(anchor: Point, avoid: Element | null): Edge[] {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const seen = new Set<Element>()
  for (let i = 0; i < 7; i++)
    for (let j = 0; j < 6; j++)
      for (const el of document.elementsFromPoint(vw * (0.06 + i * 0.147), vh * (0.08 + j * 0.168)))
        seen.add(el)
  for (const el of document.querySelectorAll(perchTags)) seen.add(el)

  const edges: Edge[] = []
  const minX = margin + width / 2
  const maxX = vw - margin - width / 2
  const h = height()
  for (const el of seen) {
    if (el === document.documentElement || el === document.body || root?.contains(el)) continue
    const rect = boxOf(el)
    if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) continue
    const isRule = el.tagName === 'HR'
    const isText = headings.has(el.tagName)
    if (rect.width < width * 0.8 || (!isRule && !isText && rect.height < 28)) continue
    if (el.closest('header, nav, dialog, .popover, section.header, [aria-hidden="true"]')) continue
    const style = getComputedStyle(el)
    if (style.visibility === 'hidden' || Number(style.opacity) === 0) continue
    const solid = replaced.has(el.tagName) || filled(style)
    const lineTop = visibleLine(style, 'Top')
    const lineBottom = visibleLine(style, 'Bottom')
    // Page-wide containers only count where they draw a line.
    if (rect.width > vw * 0.92 && !lineTop && !lineBottom) continue
    const top = isText || isRule || solid || lineTop
    const bottom = !isRule && !isText && (solid || lineBottom)
    const lo = clamp(rect.left + width * 0.3, minX, maxX)
    const hi = clamp(rect.right - width * 0.3, minX, maxX)
    if (hi < lo) continue
    const x = clamp(anchor.x, lo, hi)
    const t = (x - rect.left) / rect.width
    const push = (kind: Kind, y: number, weight: number) => {
      const y0 = y - (edgeY[kind] - viewTop) * unit
      if (y0 < margin || y0 + h > vh - margin) return
      const score =
        Math.hypot(anchor.x - x, anchor.y - y) * weight +
        (el === avoid ? 400 : 0) +
        Math.random() * 60
      edges.push({ el, kind, t, at: { x, y }, score })
    }
    if (top) {
      const sit = !isRule && !isText && rect.height > 0.6 * unit && Math.random() < 0.5
      push(sit ? 'sit' : 'stand', rect.top, 1)
    }
    if (bottom && rect.height > 40) push('hang', rect.bottom, 1.2)
  }
  return edges.sort((a, b) => a.score - b.score)
}

// The first edge that is actually in view at the point the pet would hold and moves with the page.
function pickEdge(anchor: Point, avoid: Element | null): Edge | null {
  for (const edge of edgesNear(anchor, avoid).slice(0, 12)) {
    const probe = edge.kind === 'hang' ? edge.at.y - 3 : edge.at.y + 3
    const hit = document.elementsFromPoint(edge.at.x, probe).find(e => !root?.contains(e))
    if (!hit || !(edge.el === hit || edge.el!.contains(hit))) continue
    if (!inFlow(edge.el!)) continue
    return edge
  }
  return null
}

// The document position (top-left of the pet) that puts its held edge on `perch`.
function placeOn(p: Perch): Point {
  let x: number
  let y: number
  if (p.el) {
    const rect = boxOf(p.el)
    x = rect.left + p.t * rect.width
    y = p.kind === 'hang' ? rect.bottom : rect.top
  } else {
    x = p.t * window.innerWidth
    y = window.innerHeight - margin - (viewBottom - edgeY.floor) * unit
  }
  x = clamp(x, margin + width / 2, window.innerWidth - margin - width / 2)
  return {
    x: x - width / 2 + window.scrollX,
    y: y - (edgeY[p.kind] - viewTop) * unit + window.scrollY,
  }
}

const savePosition = () => {
  const c = center()
  sessionStorage.setItem(positionKey, `${Math.round(c.x)},${Math.round(c.y)}`)
}

const observePerch = () => {
  resizeObserver?.disconnect()
  if (!perch?.el) return
  resizeObserver ??= new ResizeObserver(() => {
    cancelAnimationFrame(anchorFrame)
    anchorFrame = requestAnimationFrame(reanchor)
  })
  resizeObserver.observe(perch.el)
  resizeObserver.observe(document.body)
}

// Keeps the pet on its edge when the page reflows under it.
function reanchor() {
  if (!root || !perch || hop || carry) return
  if (perch.el && !perch.el.isConnected) {
    relocate(center(), null)
    return
  }
  const next = placeOn(perch)
  if (doc && Math.hypot(next.x - doc.x, next.y - doc.y) < 1) return
  doc = next
  root.style.transform = transformFor(doc)
  render()
}

const scheduleWander = () => {
  window.clearTimeout(wanderTimer)
  wanderTimer = window.setTimeout(
    () => {
      const now = performance.now()
      const s = state(now)
      if ((s === 'idle' || s === 'wait') && now - lastMove > 3000 && !hop && !carry)
        relocate(center(), perch?.el ?? null)
      scheduleWander()
    },
    30000 + Math.random() * 30000,
  )
}

// Moves the pet onto `next`: a walk along the same edge, an arcing hop to a nearby one, a swim on a
// gentle curve after scrolling, or a drop from above when the old spot is far away or out of view.
// One Web Animation on the root each time.
function moveTo(next: Perch, swim = false) {
  if (!root) return
  gaze = ''
  const target = placeOn(next)
  const from = doc
  const same = !!from && perch?.el === next.el && perch?.kind === next.kind && next.kind !== 'hang'
  perch = next
  doc = target
  observePerch()
  hop?.cancel()
  hop = null
  const end = transformFor(target)
  if (!from || reducedMotion()) {
    root.style.transform = end
    render()
    return
  }
  const d = Math.hypot(target.x - from.x, target.y - from.y)
  if (d < 2) {
    root.style.transform = end
    render()
    return
  }
  const fromView = from.y - window.scrollY
  const offscreen = fromView + height() < 0 || fromView > window.innerHeight
  if (Math.abs(target.x - from.x) > 2) face = target.x > from.x ? -1 : 1
  let frames: Keyframe[]
  let duration: number
  if (swim) {
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
  } else if (offscreen || d > 900) {
    frames = [
      { transform: transformFor({ x: target.x, y: target.y - 160 }), opacity: 0 },
      { transform: end, opacity: 1 },
    ]
    duration = 420
  } else if (same && d < 360) {
    frames = [{ transform: transformFor(from) }, { transform: end }]
    duration = clamp((d / 160) * 1000, 300, 2400)
  } else {
    const peak = Math.min(180, 50 + d * 0.3)
    const mid = { x: (from.x + target.x) / 2, y: Math.min(from.y, target.y) - peak }
    frames = [
      { transform: transformFor(from), easing: 'cubic-bezier(0.3, 0.6, 0.5, 1)' },
      { transform: transformFor(mid), offset: 0.5, easing: 'cubic-bezier(0.5, 0, 0.7, 0.4)' },
      { transform: end },
    ]
    duration = clamp(320 + d * 0.45, 380, 900)
  }
  const walking = !swim && frames.length === 2 && !offscreen && d <= 900
  override = { state: swim ? 'swim' : walking ? 'run' : 'fly', until: Infinity }
  render()
  const anim = root.animate(frames, {
    duration,
    easing: swim ? 'ease-in-out' : walking ? 'linear' : undefined,
  })
  hop = anim
  anim.onfinish = () => {
    if (hop !== anim || !root) return
    root.style.transform = end
    anim.cancel()
    hop = null
    override = null
    savePosition()
    if (walking) render()
    else play('land', 380)
    scheduleWander()
  }
}

function relocate(anchor: Point, avoid: Element | null, swim = false) {
  if (!root) return
  const edge = pickEdge(anchor, avoid)
  moveTo(edge ?? { el: null, kind: 'floor', t: clamp(anchor.x / window.innerWidth, 0, 1) }, swim)
}

// The rail mirrors a scroll thumb: the pole (x = 1.32 in pet units) sits on the viewport's right
// edge, and the hands (y = -0.8) follow where the thumb's middle would be, kept on screen.
const railFor = (): Rail => {
  const view = window.innerHeight
  const span = Math.max(document.documentElement.scrollHeight - view, 1)
  const thumb = Math.max((view * view) / (span + view), 32)
  const hands = 1.2 * unit
  return {
    x: window.innerWidth - 6 - 3.12 * unit,
    from: Math.max(thumb / 2 - hands, margin),
    to: Math.min(view - thumb / 2 - hands, view - height() - margin),
    span,
  }
}

// The page started scrolling: let go of the edge and leap onto the rail.
function startCarry() {
  if (!root || !doc) return
  const rect = root.getBoundingClientRect()
  hop?.cancel()
  hop = null
  override = null
  perch = null
  resizeObserver?.disconnect()
  const rail = railFor()
  carry = rail
  root.style.setProperty('--pet-rail-from', `${rail.from.toFixed(1)}px`)
  root.style.setProperty('--pet-rail-to', `${rail.to.toFixed(1)}px`)
  root.classList.add('is-carried')
  const end = `translate3d(${rail.x.toFixed(1)}px, 0, 0)`
  root.style.transform = end
  if (!scrollTimeline) root.style.translate = `0 ${railY(rail).toFixed(1)}px`
  render()
  // The rail's own offset keeps moving with the scroll during the leap, so the leap animates only
  // the gap between where the pet was and the rail.
  const dx = rect.left - rail.x
  const dy = rect.top - railY(rail)
  const d = Math.hypot(dx, dy)
  if (d < 2) return
  const anim = root.animate(
    [
      { transform: `translate3d(${dx.toFixed(1)}px, ${dy.toFixed(1)}px, 0) ${end}` },
      { transform: end },
    ],
    { duration: clamp(140 + d * 0.3, 180, 420), easing: 'cubic-bezier(0.2, 0, 0, 1)' },
  )
  hop = anim
  anim.onfinish = () => {
    if (hop !== anim) return
    anim.cancel()
    hop = null
  }
}

// Scrolling stopped: back into the page's coordinates, then swim to the nearest edge.
function endCarry() {
  if (!root || !carry) return
  hop?.cancel()
  hop = null
  const at = { x: carry.x, y: railY(carry) }
  carry = null
  doc = { x: at.x + window.scrollX, y: at.y + window.scrollY }
  root.classList.remove('is-carried')
  root.style.translate = ''
  root.style.transform = transformFor(doc)
  relocate(center(), null, true)
}

// Eyes drift toward the pointer, and the pet turns to face it once it is clearly to one side.
const watch = () => {
  watchFrame = 0
  if (!root || !pointer || hop) return
  const c = center()
  const dx = pointer.x - c.x
  const dy = pointer.y - c.y
  if (Math.abs(dx) > 40 && !carry) face = dx > 0 ? -1 : 1
  const d = Math.max(Math.hypot(dx, dy), 80)
  // Gaze snaps to a few positions, so most pointer moves change nothing and repaint nothing.
  const look = `${face} ${(Math.round((face * 4 * dx) / d) * 0.03).toFixed(2)} ${(Math.round((3 * dy) / d) * 0.027).toFixed(3)}`
  if (look === gaze) return
  gaze = look
  const [f, lx, ly] = look.split(' ')
  root.style.setProperty('--pet-face', f)
  root.style.setProperty('--pet-lx', lx)
  root.style.setProperty('--pet-ly', ly)
}

const onDwell = () => {
  if (!root || !pointer || hop || carry || hovering || reducedMotion()) return
  if (performance.now() < relocateReadyAt) return
  const c = center()
  if (Math.hypot(pointer.x - c.x, pointer.y - c.y) <= dwellReach) return
  relocateReadyAt = performance.now() + relocateCooldownMs
  relocate(pointer, null)
}

const onPointerMove = (event: PointerEvent) => {
  if (event.pointerType === 'touch' || !root) return
  const now = performance.now()
  pointer = { x: event.clientX, y: event.clientY }
  if (carry) {
    watchFrame ||= requestAnimationFrame(watch)
    return
  }
  if (lastPointer && now > lastPointer.t && !hop) {
    const speed =
      (Math.hypot(pointer.x - lastPointer.x, pointer.y - lastPointer.y) / (now - lastPointer.t)) *
      1000
    const c = center()
    if (
      speed > startleSpeed &&
      now > startleReadyAt &&
      Math.hypot(pointer.x - c.x, pointer.y - c.y) < startleRadius
    ) {
      startleReadyAt = now + 5000
      play('failed', 1600)
    }
  }
  lastPointer = { ...pointer, t: now }
  const wasQuiet = now - lastMove > waitAfterMs
  lastMove = now
  armQuietTimer()
  if (wasQuiet) render()
  watchFrame ||= requestAnimationFrame(watch)
  window.clearTimeout(dwellTimer)
  dwellTimer = window.setTimeout(onDwell, dwellMs)
}

// Touch has no resting pointer, so the pet heads for the edge nearest wherever the reader taps.
const onPointerDown = (event: PointerEvent) => {
  if (event.pointerType !== 'touch' || !root || hop) return
  if (root.contains(event.target as Node)) return
  pointer = { x: event.clientX, y: event.clientY }
  lastMove = performance.now()
  armQuietTimer()
  if (!reducedMotion()) relocate(pointer, null)
}

// Once a resize (or, with reduced motion, a scroll) settles, an edge that left the view sends the
// pet to one in view.
const onSettle = () => {
  if (!root || !doc || hop) return
  const y = doc.y - window.scrollY
  if (y >= -margin && y + height() <= window.innerHeight + margin) {
    reanchor()
    return
  }
  const anchor = pointer ?? { x: window.innerWidth * 0.75, y: window.innerHeight * 0.6 }
  relocate(anchor, null)
}

const hideReminder = () => {
  window.clearTimeout(reminderHideTimer)
  reminding = false
  root?.classList.remove('is-reminding')
  const text = root?.querySelector('.garden-pet-bubble-text')
  if (text) text.textContent = ''
  render()
}

const showReminder = () => {
  if (!root) return
  reminding = true
  root.classList.add('is-reminding')
  const text = root.querySelector('.garden-pet-bubble-text')
  if (text) text.textContent = 'drink water'
  render()
  window.clearTimeout(reminderHideTimer)
  reminderHideTimer = window.setTimeout(hideReminder, hydrationReminderVisibleMs)
}

const scheduleReminder = () => {
  if (reminderTimer !== undefined) return
  reminderTimer = window.setTimeout(() => {
    reminderTimer = undefined
    showReminder()
    scheduleReminder()
  }, hydrationReminderIntervalMs)
}

const spawn = () => {
  markup ||= petSvg()
  const el = document.createElement('div')
  el.className = 'garden-pet'
  el.dataset.petWidget = ''
  el.innerHTML = `<button type="button" class="garden-pet-button" aria-label="garden pet">${markup}</button><div class="garden-pet-bubble" role="status"><span class="garden-pet-bubble-text"></span><button type="button" class="garden-pet-close" aria-label="dismiss reminder">×</button></div>`
  document.body.appendChild(el)
  root = el
  svg = el.querySelector('svg')
  // The pet is absolutely positioned in document coordinates, so the window's scroll carries it
  // along its edge for free. The containing block may sit off the document origin; measure it once.
  const rect = el.getBoundingClientRect()
  width = rect.width || 72
  unit = width / viewWidth
  origin = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
  if (reminding) {
    el.classList.add('is-reminding')
    el.querySelector('.garden-pet-bubble-text')!.textContent = 'drink water'
  }
  requestAnimationFrame(() => el.classList.add('is-visible'))
}

const despawn = () => {
  hop?.cancel()
  hop = null
  if (doc || carry) savePosition()
  carry = null
  for (const timer of [stateTimer, dwellTimer, scrollTimer, wanderTimer, reminderTimer])
    window.clearTimeout(timer)
  reminderTimer = undefined
  resizeObserver?.disconnect()
  hideReminder()
  root?.remove()
  root = null
  svg = null
  doc = null
  perch = null
}

const syncPets = () => {
  const enabled = readPetsEnabled()
  document.body.dataset.pets = enabled ? 'on' : 'off'
  if (!enabled) {
    despawn()
    return
  }
  // The SPA morphs <body> on navigation, which drops the pet and the box it held. It comes back
  // near where it was on screen and drops onto the nearest edge of the new page.
  if (!root || !root.isConnected) {
    const [x, y] = (sessionStorage.getItem(positionKey) ?? '').split(',').map(Number)
    const before =
      (doc || carry) && !hop
        ? center()
        : Number.isFinite(x) && Number.isFinite(y) && sessionStorage.getItem(positionKey)
          ? { x, y }
          : { x: window.innerWidth * 0.8, y: window.innerHeight * 0.7 }
    hop?.cancel()
    hop = null
    doc = null
    carry = null
    perch = null
    spawn()
    relocate(before, null)
  } else if (perch?.el && !perch.el.isConnected) relocate(center(), null)
  if (lastMove === 0) lastMove = performance.now()
  armQuietTimer()
  scheduleWander()
  scheduleReminder()
}

document.addEventListener('petstoggle', event => {
  const enabled = event.detail.enabled ?? !readPetsEnabled()
  localStorage.setItem(petsEnabledKey, enabled ? 'true' : 'false')
  syncPets()
  const toast: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: { message: `pets ${enabled ? 'on' : 'off'}` },
  })
  document.dispatchEvent(toast)
})

document.addEventListener('nav', () => {
  syncPets()

  const controller = new AbortController()
  const { signal } = controller
  document.addEventListener('pointermove', onPointerMove, { passive: true, signal })
  document.addEventListener('pointerdown', onPointerDown, { passive: true, signal })
  document.addEventListener(
    'click',
    event => {
      const hit = event.target as Element | null
      if (!root || !hit || !root.contains(hit)) return
      if (hit.closest('.garden-pet-close')) hideReminder()
      else if (hit.closest('.garden-pet-button') && !hop) play('cheer', 1400)
    },
    { signal },
  )
  document.addEventListener(
    'pointerover',
    event => {
      if (event.pointerType === 'touch') return
      const over = !!root?.querySelector('.garden-pet-button')?.contains(event.target as Node)
      if (over === hovering) return
      hovering = over
      render()
    },
    { passive: true, signal },
  )
  window.addEventListener(
    'scroll',
    () => {
      window.clearTimeout(scrollTimer)
      if (reducedMotion()) {
        scrollTimer = window.setTimeout(onSettle, 220)
        return
      }
      if (!carry) startCarry()
      else if (root && !scrollTimeline) root.style.translate = `0 ${railY(carry).toFixed(1)}px`
      scrollTimer = window.setTimeout(endCarry, 200)
    },
    { passive: true, signal },
  )
  window.addEventListener(
    'resize',
    () => {
      window.clearTimeout(scrollTimer)
      scrollTimer = window.setTimeout(carry ? endCarry : onSettle, 220)
    },
    { passive: true, signal },
  )
  window.addEventListener('pagehide', () => (doc || carry) && savePosition(), { signal })
  window.addCleanup(() => controller.abort())
})
