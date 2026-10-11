import { hand, petSvg, pressure, stroke } from '../../util/mascot'
import { readPetSpot, writePetSpot } from '../../util/pet-spot'

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
  | 'walk'
  | 'chute'
  | 'think'
type Kind = 'stand' | 'sit' | 'hang' | 'floor'
type Point = { x: number; y: number }
// An edge the pet holds: `t` is where its centre sits along the box's width, 0 to 1. The floor kind
// has no box: the pet stands on its own pen line at the bottom of the viewport, the fallback when no
// box nearby has an edge to hold.
// `foot`, in document coordinates, pins a perch that is a point rather than a box edge: a spot on
// the home page's walking trail.
type Perch = { el: Element | null; kind: Kind; t: number; foot?: Point }
type Edge = Perch & { at: Point; score: number }

const petsEnabledKey = 'garden:pets-enabled:v1'
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
// What to do once the next move lands, if anything; moveTo takes it for that move alone.
let afterMove: (() => void) | null = null
// When the pet last got up and wandered off on its own: it starts its quiet over from there, so a
// stroll closes the laptop until it settles again.
let settled = 0
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
let trailWalk: { path: SVGPathElement; geometry: string; feet: Point[] } | null = null

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
  const quiet = now - Math.max(lastMove, settled)
  if (quiet > workAfterMs) return 'work'
  if (quiet > waitAfterMs) return 'wait'
  return 'idle'
}

const render = () => {
  if (!root || !svg || (!doc && !carry)) return
  const next = state(performance.now())
  if (svg.dataset.state !== next) svg.dataset.state = next
  svg.dataset.kind = perch?.kind ?? 'stand'
  // On the home page's trail it wears its walking hat, from the hop on until it leaves the path.
  svg.toggleAttribute('data-hat', !carry && !!perch?.el?.classList.contains('trail-path'))
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
  const quiet = performance.now() - Math.max(lastMove, settled)
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
    // A sitter takes the corner nearer the anchor with its legs over the box's side, and a hanger
    // hangs off the corner by one hand, so neither dangles over what the box holds.
    const side = anchor.x > (rect.left + rect.right) / 2 ? 1 : -1
    const corner = (off: number) => {
      const out = side > 0 ? rect.right + off * unit : rect.left - off * unit
      if (out >= minX && out <= maxX) return out
      const other = side > 0 ? rect.left - off * unit : rect.right + off * unit
      return other >= minX && other <= maxX ? other : null
    }
    const push = (kind: Kind, y: number, weight: number) => {
      const x =
        kind === 'sit' ? corner(0.36) : kind === 'hang' ? corner(0.1) : clamp(anchor.x, lo, hi)
      if (x === null) return
      const t = (x - rect.left) / rect.width
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

// Keep the pet's whole SVG frame clear of text, including space for its tilt and idle poses.
const coversText = (edge: Edge) => {
  const left = edge.at.x - width / 2 - 2
  const right = edge.at.x + width / 2 + 2
  const top = edge.at.y - (edgeY[edge.kind] - viewTop) * unit - 2
  const bottom = top + height() + 4
  const intersects = (rect: DOMRect) =>
    rect.left < right && rect.right > left && rect.top < bottom && rect.bottom > top
  const nodes = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
    {
      acceptNode: node => {
        if (root?.contains(node)) return NodeFilter.FILTER_REJECT
        if (node instanceof Element) {
          if (node.matches('script, style')) return NodeFilter.FILTER_REJECT
          const style = getComputedStyle(node)
          if (style.display === 'none' || style.visibility === 'hidden')
            return NodeFilter.FILTER_REJECT
          const rect = node.getBoundingClientRect()
          if (rect.width && rect.height && !intersects(rect)) return NodeFilter.FILTER_REJECT
          return NodeFilter.FILTER_SKIP
        }
        return node.textContent?.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT
      },
    },
  )
  const range = document.createRange()
  for (let node = nodes.nextNode(); node; node = nodes.nextNode()) {
    range.selectNodeContents(node)
    for (const rect of range.getClientRects()) if (intersects(rect)) return true
  }
  return false
}

// Probes run with the pet hidden, so its own drawing never counts as the page's.
function unseen<T>(probe: () => T): T {
  if (root) root.style.visibility = 'hidden'
  try {
    return probe()
  } finally {
    if (root) root.style.visibility = ''
  }
}

// The floor, too, keeps off text: a client point where the pet could stand on its own line.
const floorStep = 16
const floorY = () => window.innerHeight - margin - (viewBottom - edgeY.floor) * unit
const floorClear = (x: number, y = floorY()) =>
  !coversText({ el: null, kind: 'floor', t: 0, at: { x, y }, score: 0 })

// Scan out both ways, then move up when a line of text blocks the whole floor.
const floorNear = (x: number): Perch | null =>
  unseen<Perch | null>(() => {
    const lo = margin + width / 2
    const hi = window.innerWidth - margin - width / 2
    const start = clamp(x, lo, hi)
    const bottom = floorY()
    const top = margin + (edgeY.floor - viewTop) * unit
    for (let y = bottom; y >= top; y -= floorStep)
      for (let k = 0; k * floorStep <= hi - lo; k++)
        for (const c of k ? [start + k * floorStep, start - k * floorStep] : [start])
          if (c >= lo && c <= hi && floorClear(c, y))
            return {
              el: null,
              kind: 'floor',
              t: c / window.innerWidth,
              foot: y === bottom ? undefined : { x: c + window.scrollX, y: y + window.scrollY },
            }
    return null
  })

// How far along the floor from `from` toward `to` the pet can walk with text clear underfoot.
const floorReach = (from: number, to: number) =>
  unseen(() => {
    const dir = Math.sign(to - from)
    let x = from
    while (Math.abs(to - x) > floorStep && floorClear(x + dir * floorStep)) x += dir * floorStep
    return Math.abs(to - x) <= floorStep && floorClear(to) ? to : x
  })

// The first edge that is actually in view at the point the pet would hold, moves with the page,
// and leaves the text around it clear.
function pickEdge(anchor: Point, avoid: Element | null): Edge | null {
  return unseen(() => {
    for (const edge of edgesNear(anchor, avoid).slice(0, 20)) {
      const probe = edge.kind === 'hang' ? edge.at.y - 3 : edge.at.y + 3
      const hit = document.elementsFromPoint(edge.at.x, probe).find(e => !root?.contains(e))
      if (edge.kind === 'stand' && (!hit || !(edge.el === hit || edge.el!.contains(hit)))) continue
      if (!inFlow(edge.el!) || coversText(edge)) continue
      return edge
    }
    return null
  })
}

// The document position (top-left of the pet) that puts its held edge on `perch`.
function placeOn(p: Perch): Point {
  if (p.foot) return { x: p.foot.x - width / 2, y: p.foot.y - (edgeY[p.kind] - viewTop) * unit }
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

const savePosition = () => writePetSpot(center())

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

// Left alone, the pet gets up now and then, laptop or not: mostly a stroll along its own edge, a
// sitter stands up first, and sometimes it hops off to another edge nearby.
const wander = () => {
  const now = performance.now()
  const s = state(now)
  const free = s === 'idle' || s === 'wait' || s === 'work'
  if (!free || now - lastMove < 3000 || hop || carry || reducedMotion() || document.hidden) return
  settled = now
  armQuietTimer()
  const note = Math.random() < 0.4 ? sidenoteInView() : null
  if (note) return readSidenote(note)
  if (Math.random() < 0.6 && walkTrail()) return
  if (
    !perch ||
    perch.kind === 'hang' ||
    (perch.kind === 'floor' && perch.foot) ||
    Math.random() < 0.25
  )
    return relocate(center(), perch?.el ?? null)
  if (perch.kind === 'sit') return moveTo({ ...perch, kind: 'stand' })
  const span = perch.el ? boxOf(perch.el).width : window.innerWidth
  let dir = Math.random() < 0.5 ? -1 : 1
  const step = (70 + Math.random() * 170) / Math.max(span, 1)
  if (perch.t + dir * step > 0.96 || perch.t + dir * step < 0.04) dir = -dir
  let t = clamp(perch.t + dir * step, 0.04, 0.96)
  // On the floor it stops short of the first stretch that would put it over text.
  if (!perch.el) {
    const here = center().x
    if (!unseen(() => floorClear(here))) return relocate(center(), null)
    t = floorReach(here, t * span) / span
  }
  if (Math.abs(t - perch.t) * span < 30) return
  moveTo({ ...perch, t }, 'stroll')
}

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
  afterMove = () => {
    const readMs = 4200 + Math.random() * 2400
    // Its quiet is set to have run out just as it finishes reading, so the laptop comes back out.
    settled = performance.now() + readMs - workAfterMs - 200
    play('think', readMs)
    armQuietTimer()
  }
  moveTo({ el: note, kind: 'stand', t: 0.3 + Math.random() * 0.4 })
}

// The upper route belongs to the pet. Meadow and Alpine routes belong to the travelers.
const trailPoints = new WeakMap<SVGPathElement, { d: string; step: number; pts: DOMPoint[] }>()
const trailRoutes = () =>
  Array.from(
    document.querySelectorAll<SVGPathElement>(
      '.homepage-trail-scene:not([hidden]) [data-pet-walk-path]',
    ),
  ).flatMap(path => {
    const d = path.getAttribute('d')
    const m = path.getScreenCTM()
    if (!d || !m) return []
    const step = 4 / Math.max(Math.hypot(m.a, m.b), Math.hypot(m.c, m.d), 0.001)
    let hit = trailPoints.get(path)
    if (hit?.d !== d || hit.step !== step) {
      const length = path.getTotalLength()
      const segments = Math.max(1, Math.ceil(length / step))
      const pts = Array.from({ length: segments + 1 }, (_, k) =>
        path.getPointAtLength((k * length) / segments),
      )
      hit = { d, step, pts }
      trailPoints.set(path, hit)
    }
    // Feet in document coordinates.
    const feet = hit.pts.map(p => {
      const c = p.matrixTransform(m)
      return { x: c.x + window.scrollX, y: c.y + window.scrollY }
    })
    return [{ path, feet }]
  })

const trailGeometry = (path: SVGPathElement) => {
  const m = path.getScreenCTM()
  if (!m) return ''
  return `${path.getAttribute('d')}|${m.a},${m.b},${m.c},${m.d},${m.e},${m.f}|${window.innerWidth},${window.innerHeight}`
}

const trailPointClear = (p: Point, path: SVGPathElement) => {
  const at = { x: p.x - window.scrollX, y: p.y - window.scrollY }
  const top = at.y - (edgeY.stand - viewTop) * unit
  return (
    at.x >= margin + width / 2 &&
    at.x <= window.innerWidth - margin - width / 2 &&
    top >= margin &&
    top + height() <= window.innerHeight - margin &&
    !coversText({ el: path, kind: 'stand', t: 0, at, score: 0 })
  )
}

type TrailRun = { path: SVGPathElement; feet: Point[]; length: number }
const longestTrailRun = (): TrailRun | null => {
  let best: TrailRun | null = null
  for (const { path, feet } of trailRoutes()) {
    let run: Point[] = []
    let length = 0
    const finish = () => {
      if (length >= 50 && (!best || length > best.length)) best = { path, feet: run, length }
      run = []
      length = 0
    }
    for (const foot of feet) {
      if (!trailPointClear(foot, path)) {
        finish()
        continue
      }
      const previous = run[run.length - 1]
      if (previous) length += Math.hypot(foot.x - previous.x, foot.y - previous.y)
      run.push(foot)
    }
    finish()
  }
  return best
}

const leaveTrail = () => {
  if (!trailWalk) return
  trailWalk = null
  const rect = root?.isConnected ? root.getBoundingClientRect() : null
  hop?.cancel()
  hop = null
  afterMove = null
  override = null
  perch = null
  resizeObserver?.disconnect()
  cancelAnimationFrame(anchorFrame)
  if (root && rect) {
    doc = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
    root.style.transform = transformFor(doc)
  }
  settled = performance.now()
  render()
  armQuietTimer()
  scheduleWander()
}

// Follow increasing path length, using the longest visible run that clears the whole pet frame.
const walkTrail = () => {
  if (!root || hop || carry || reducedMotion() || document.hidden) return false
  const best = longestTrailRun()
  if (!best) return false
  const { path, feet } = best
  let route = feet
  if (perch?.el === path && perch.foot) {
    const foot = perch.foot
    const k = feet.findIndex(p => Math.hypot(p.x - foot.x, p.y - foot.y) < 8)
    if (k >= 0) route = feet.slice(k)
  }
  if (route.length < 11) return false
  trailWalk = { path, geometry: trailGeometry(path), feet: route }
  const walk = () => {
    if (
      !path.isConnected ||
      path.closest<HTMLElement>('.homepage-trail-scene')?.hidden !== false ||
      trailWalk?.geometry !== trailGeometry(path) ||
      reducedMotion() ||
      document.hidden
    ) {
      leaveTrail()
      return
    }
    follow(route, path)
  }
  const start = route[0]
  if (perch?.foot && Math.hypot(perch.foot.x - start.x, perch.foot.y - start.y) < 12) walk()
  else {
    afterMove = walk
    // An entry hop could cross prose. Fade onto the checked start before the walking animation.
    moveTo({ el: path, kind: 'stand', t: 0, foot: start }, 'fade')
  }
  return true
}

// Walks a run of trail points at a stroll, ending on the last of them.
function follow(feet: Point[], el: Element) {
  if (!root) return
  hop?.cancel()
  hop = null
  const from = feet[0]
  const to = feet[feet.length - 1]
  perch = { el, kind: 'stand', t: 0, foot: to }
  doc = placeOn(perch)
  observePerch()
  if (Math.abs(to.x - from.x) > 2) face = to.x > from.x ? -1 : 1
  let length = 0
  for (let k = 1; k < feet.length; k++)
    length += Math.hypot(feet[k].x - feet[k - 1].x, feet[k].y - feet[k - 1].y)
  override = { state: 'walk', until: Infinity }
  render()
  const anim = root.animate(
    feet.map(foot => ({ transform: transformFor(placeOn({ el, kind: 'stand', t: 0, foot })) })),
    { duration: (length / 55) * 1000, easing: 'linear' },
  )
  hop = anim
  anim.onfinish = () => {
    if (hop !== anim || !root || !doc) return
    root.style.transform = transformFor(doc)
    anim.cancel()
    hop = null
    override = null
    settled = performance.now()
    savePosition()
    render()
    armQuietTimer()
    scheduleWander()
  }
}

const scheduleWander = () => {
  window.clearTimeout(wanderTimer)
  wanderTimer = window.setTimeout(
    () => {
      wander()
      scheduleWander()
    },
    12000 + Math.random() * 20000,
  )
}

// Moves the pet onto `next`: a run or a stroll along the same edge, an arcing hop to a nearby one, a
// swim on a gentle curve after scrolling, or a drop from above when the old spot is far away or out
// of view. One Web Animation on the root each time.
// Walking the floor, the pet inks its own ground as it goes: a pen line along its path, laid in
// short pieces that appear as its feet reach them, hold while it passes, and fade out slowly behind
// it. Pieces it never reached (the walk was cut short) are dropped.
type Ground = { svg: SVGSVGElement; pieces: { anim: Animation; delay: number }[] }
const grounds = new Set<Ground>()
const groundPiece = 24
const groundFadeMs = 2800

function layGround(from: Point, to: Point, duration: number) {
  if (!root) return
  // The pet's own ground line sits 1.44 units down; centres, not corners, walk the floor.
  const y = to.y + (1.44 - viewTop) * unit
  const x0 = from.x + width / 2
  const x1 = to.x + width / 2
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
  const pen = pressure(Math.floor(Math.random() * 1e4), 0.064 * unit, 0.3)
  const n = line.length - 1
  const per = Math.max(2, Math.round((groundPiece / span) * n))
  const speed = span / duration
  // Held until the pet's own ground line (2.4 units wide) has passed it.
  const hold = 500 + (2.4 * unit) / speed
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
  root.before(el)
  grounds.add(ground)
  void Promise.all(ground.pieces.map(p => p.anim.finished))
    .catch(() => {})
    .finally(() => {
      el.remove()
      grounds.delete(ground)
    })
}

// The walk was cut short: pieces the pet never reached never appear; the rest fade as planned.
function stopGround() {
  for (const ground of grounds)
    for (const { anim, delay } of ground.pieces)
      if ((Number(anim.currentTime) || 0) < delay) anim.cancel()
}

function moveTo(next: Perch, how: 'swim' | 'chute' | 'stroll' | 'fade' | null = null) {
  const after = afterMove
  afterMove = null
  if (!root) return
  if (!(next.el instanceof SVGPathElement) || !next.el.hasAttribute('data-pet-walk-path'))
    trailWalk = null
  root.hidden = false
  stopGround()
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
    after?.()
    return
  }
  const d = Math.hypot(target.x - from.x, target.y - from.y)
  if (d < 2) {
    root.style.transform = end
    render()
    after?.()
    return
  }
  // Coming off the rail, it floats down under a parachute to anything below it, and swims up to
  // anything above.
  const chute = how === 'chute' && target.y - from.y > 40
  const swim = how === 'swim' || (how === 'chute' && !chute)
  const fromView = from.y - window.scrollY
  const offscreen = fromView + height() < 0 || fromView > window.innerHeight
  if (Math.abs(target.x - from.x) > 2) face = target.x > from.x ? -1 : 1
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
  } else if (swim) {
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
    duration = clamp((d / (how === 'stroll' ? 60 : 160)) * 1000, 300, 4800)
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
  const walking = how !== 'fade' && !swim && !chute && frames.length === 2 && !offscreen && d <= 900
  const pace = how === 'stroll' ? 'walk' : 'run'
  if (walking && next.kind === 'floor') layGround(from, target, duration)
  override = {
    state: how === 'fade' ? 'idle' : chute ? 'chute' : swim ? 'swim' : walking ? pace : 'fly',
    until: Infinity,
  }
  render()
  const anim = root.animate(frames, {
    duration,
    easing: swim ? 'ease-in-out' : walking || chute ? 'linear' : undefined,
  })
  hop = anim
  anim.onfinish = () => {
    if (hop !== anim || !root) return
    root.style.transform = end
    anim.cancel()
    hop = null
    override = null
    savePosition()
    if (walking || how === 'fade') render()
    else play('land', 380)
    scheduleWander()
    after?.()
  }
}

function relocate(anchor: Point, avoid: Element | null, how: 'chute' | null = null) {
  if (!root) return
  const next = pickEdge(anchor, avoid) ?? floorNear(anchor.x)
  if (next) moveTo(next, how)
  else root.hidden = true
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
  trailWalk = null
  perch = null
  resizeObserver?.disconnect()
  stopGround()
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
  relocate(center(), null, 'chute')
}

// The eyes follow the pointer round, as lydia's bot does: they slide toward it along its direction,
// easing back to centre as it comes close, and the body tips a few degrees its way. The pet turns
// to face it once it is clearly to one side.
const watch = () => {
  watchFrame = 0
  if (!root || !pointer || hop) return
  const c = center()
  const dx = pointer.x - c.x
  const dy = pointer.y - c.y
  if (Math.abs(dx) > 40 && !carry) face = dx > 0 ? -1 : 1
  const r = Math.hypot(dx, dy)
  const k = Math.min(1, r / (3 * unit)) / Math.max(r, 1)
  // In the pet's own frame, which is mirrored when it faces right.
  const ux = face * dx * k
  const uy = dy * k
  // Hundredths of a body radius (about a fifth of a pixel), so a still pointer repaints nothing.
  // The eyes sit high and to one side, so they travel less upward, where the outline is nearest.
  const look = `${face} ${(ux * 0.18).toFixed(2)} ${(uy * (uy < 0 ? 0.11 : 0.16)).toFixed(2)} ${carry ? 0 : (ux * 7).toFixed(1)}`
  if (look === gaze) return
  gaze = look
  const [f, lx, ly, tilt] = look.split(' ')
  root.style.setProperty('--pet-face', f)
  root.style.setProperty('--pet-lx', lx)
  root.style.setProperty('--pet-ly', ly)
  root.style.setProperty('--pet-tilt', tilt)
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
  if (!root || hop) return
  const y = doc ? doc.y - window.scrollY : Infinity
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
  trailWalk = null
  afterMove = null
  if (doc || carry) savePosition()
  carry = null
  for (const timer of [stateTimer, dwellTimer, scrollTimer, wanderTimer, reminderTimer])
    window.clearTimeout(timer)
  reminderTimer = undefined
  resizeObserver?.disconnect()
  hideReminder()
  for (const ground of grounds) ground.svg.remove()
  grounds.clear()
  root?.remove()
  root = null
  svg = null
  doc = null
  perch = null
}

const syncPets = () => {
  const enabled = readPetsEnabled()
  document.body.dataset.pets = enabled ? 'on' : 'off'
  // The 404 scene paints its own pet into the ruin, starting from where this one stood.
  if (!enabled || document.querySelector('.nf-scene')) {
    despawn()
    return
  }
  // The SPA morphs <body> on navigation, which drops the pet and the box it held. It comes back
  // near where it was on screen and drops onto the nearest edge of the new page.
  if (!root || !root.isConnected) {
    const before = ((doc || carry) && !hop ? center() : readPetSpot()) ?? {
      x: window.innerWidth * 0.8,
      y: window.innerHeight * 0.7,
    }
    hop?.cancel()
    hop = null
    doc = null
    carry = null
    perch = null
    trailWalk = null
    afterMove = null
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
  const scene = document.querySelector<HTMLElement>('.homepage-trail-scene')
  let trailFrame = 0
  let trailWanted = false
  let waiting: Animation | null = null
  const refreshTrail = () => {
    trailFrame = 0
    if (signal.aborted || !scene || scene.hidden || document.hidden || reducedMotion()) return
    const currentTrail = trailWalk
    if (
      currentTrail &&
      (currentTrail.geometry !== trailGeometry(currentTrail.path) ||
        currentTrail.feet.some(foot => !trailPointClear(foot, currentTrail.path)))
    ) {
      leaveTrail()
      trailWanted = true
    }
    if (!trailWanted || carry) return
    const current = hop
    if (current) {
      if (waiting === current) return
      waiting = current
      const ready = () => {
        if (waiting !== current || signal.aborted) return
        waiting = null
        queueTrail()
      }
      void current.finished.then(ready, ready)
      return
    }
    if (walkTrail()) trailWanted = false
  }
  // The frame can redraw in its own animation callback after the host's reflow event.
  const queueTrail = () => {
    if (trailFrame || signal.aborted) return
    trailFrame = requestAnimationFrame(() => {
      trailFrame = requestAnimationFrame(refreshTrail)
    })
  }
  const closeTrail = () => {
    cancelAnimationFrame(trailFrame)
    trailFrame = 0
    waiting = null
    trailWanted = false
    leaveTrail()
  }
  if (scene) {
    const openTrail = () => {
      if (scene.hidden) return
      trailWanted = true
      queueTrail()
    }
    scene.addEventListener('trail:show', openTrail, { signal })
    scene.addEventListener('trail:hide', closeTrail, { signal })
    scene.addEventListener('trail:dispose', closeTrail, { signal })
    scene.addEventListener('trail:reflow', queueTrail, { signal })
    scene.addEventListener('load', openTrail, { capture: true, signal })
    window.addEventListener('resize', queueTrail, { passive: true, signal })
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    motion.addEventListener('change', () => (motion.matches ? closeTrail() : openTrail()), {
      signal,
    })
    document.addEventListener(
      'visibilitychange',
      () => {
        if (document.hidden) {
          const resume = trailWanted || !!trailWalk
          closeTrail()
          trailWanted = resume
        } else queueTrail()
      },
      { signal },
    )
    if (!scene.hidden) openTrail()
  }
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
  window.addCleanup(() => {
    closeTrail()
    controller.abort()
  })
})
