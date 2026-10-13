import { hand, heart, petSvg, pressure, sprout, stroke } from '../../util/mascot'
import { getFullSlug } from '../../util/path'
import { readPetSpot, writePetSpot } from '../../util/pet-spot'
import { sectionOf } from '../../util/section-icon'
import { createSvgSpace, svgPathGeometry } from '../../util/svg-geometry'
import { easeTrailDepth } from '../../util/trail-depth'
import { trailHost, trailScene } from '../../util/trail-scene'
import {
  animalInWay,
  canMove,
  circleClearance,
  type CollisionActor,
} from '../visitor-trail/actor-spacing'

// The garden's pet: the pen-drawn mascot, living on the page's edges. It perches on the nearest box
// it can find (stands or sits with legs dangling on a top edge, hangs from a bottom edge) and stays
// there, watching the pointer. It hops to a new edge only when the reader's pointer rests far away or
// when it gets bored. While the page scrolls it clings to a pole on the right edge and slides along
// it with the scroll thumb; when scrolling stops it parachutes down, or jetpacks up, to the nearest
// edge. A pointer move costs no layout and every move is one compositor animation. Looks and motion live in components/styles/pet.scss; this
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
  | 'slide'
  | 'walk'
  | 'chute'
  | 'think'
  | 'talk'
  | 'jet'
  | 'notice'
type Kind = 'stand' | 'sit' | 'hang' | 'floor'
type Point = { x: number; y: number }
// An edge the pet holds: `t` is where its centre sits along the box's width, 0 to 1. The floor kind
// has no box: the pet stands on its own pen line at the bottom of the viewport, the fallback when no
// box nearby has an edge to hold.
// `foot`, in document coordinates, pins a perch that is a point rather than a box edge: a spot on
// the home page's walking trail.
type Perch = {
  el: Element | null
  kind: Kind
  t: number
  foot?: Point
  scale?: number
  progress?: number
  // How far below the element's edge the held edge sits, in client px.
  drop?: number
}
type Edge = Perch & { at: Point; score: number }
type TrailFoot = Point & { scale: number; progress: number; hidden?: boolean; handoff?: boolean }
type TrailWalk = {
  path: SVGPathElement
  geometry: string
  feet: TrailFoot[]
  offsets: number[]
  duration: number
}

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
let trailSpaceSignal: AbortSignal | undefined
let trailSpaces = new WeakMap<SVGSVGElement, ReturnType<typeof createSvgSpace>>()
const trailScreenMatrix = (element: SVGGraphicsElement) => {
  const canvas = element instanceof SVGSVGElement ? element : element.ownerSVGElement
  if (!canvas || !trailSpaceSignal || trailSpaceSignal.aborted) return element.getScreenCTM()
  let space = trailSpaces.get(canvas)
  if (!space) {
    space = createSvgSpace(canvas, trailSpaceSignal)
    trailSpaces.set(canvas, space)
  }
  return space.matrix(element)
}
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
let landAfterJet = false
// Where the eyes look instead of the pointer: a triathlon window as it opens.
let lookAt: Point | null = null
// A window closing under the pet: no edge on it or inside it counts while the pet jumps off.
let closing: Element | null = null
let windowTimer: number | undefined
// The phrase the French voice is reading (speech.inline.ts): thinking while it loads, then saying it.
let speaking: { text: string; lang: string; playing: boolean } | null = null
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
let trailWalk: TrailWalk | null = null
let trailJourney: { key: string; progress: number } | null = null

const clamp = (v: number, min: number, max: number) => Math.min(Math.max(v, min), max)
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
const height = (scale = 1) => (viewBottom - viewTop) * unit * scale

// On unless the reader turned pets off from the palette.
const readPetsEnabled = () => localStorage.getItem(petsEnabledKey) !== 'false'

// The pet's centre in viewport coordinates, from its cached document position: no layout read.
const railY = (rail: Rail) =>
  rail.from + (rail.to - rail.from) * clamp(window.scrollY / rail.span, 0, 1)

const center = (): Point =>
  carry
    ? { x: carry.x + width / 2, y: railY(carry) + height() / 2 }
    : doc
      ? {
          x: doc.x - window.scrollX + (width * (perch?.scale ?? 1)) / 2,
          y: doc.y - window.scrollY + height(perch?.scale) / 2,
        }
      : { x: window.innerWidth - width, y: window.innerHeight - height() }

const transformFor = (p: Point, scale = perch?.scale ?? 1) =>
  `translate3d(${(p.x - origin.x).toFixed(1)}px, ${(p.y - origin.y).toFixed(1)}px, 0) scale(${scale.toFixed(4)})`

const state = (now: number): State => {
  if (override && now < override.until) return override.state
  if (carry) return 'slide'
  if (speaking) return speaking.playing ? 'talk' : 'think'
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
  const outfit =
    sectionOf({
      slug: getFullSlug(window),
      base: document.body.dataset.isBase === 'true',
      slides: document.body.dataset.slides === 'true',
    }) ?? ''
  if (svg.dataset.outfit !== outfit) svg.dataset.outfit = outfit
  // Away from the sections, a page's tags show on it: glasses on technical pages, wild hair on
  // philosophy ones.
  const tags = new Set((document.body.dataset.tags ?? '').split(' '))
  const traits = outfit ? '' : ['technical', 'philosophy'].filter(t => tags.has(t)).join(' ')
  if (svg.dataset.traits !== traits) svg.dataset.traits = traits
  svg.toggleAttribute('data-reading', document.documentElement.getAttribute('reader-mode') === 'on')
  // On the rail it faces the page, its back to the pole.
  root.style.setProperty('--pet-face', String(carry ? 1 : face))
  root.dataset.bubble = center().y - height(perch?.scale) / 2 < 64 ? 'below' : 'above'
  placeBubble()
  renderFriend()
}

// What the bubble says: the phrase being read aloud, else the hydration reminder.
const showBubble = () => {
  if (!root) return
  const text = root.querySelector<HTMLElement>('.garden-pet-bubble-text')
  const said = speaking?.playing ? speaking : chat ? { text: chat, lang: 'fr' } : null
  root.classList.toggle('is-speaking', !!said)
  root.classList.toggle('is-reminding', reminding && !said)
  if (text) {
    text.textContent = said ? said.text : reminding ? 'drink water' : ''
    text.lang = said ? said.lang : ''
  }
  render()
}

// Keeps a shown bubble inside the view: it centres on the pet, so a long phrase near an edge
// shifts sideways instead of running off the page.
const placeBubble = () => {
  const bubble = root?.querySelector<HTMLElement>('.garden-pet-bubble')
  if (!root || !bubble || !(speaking?.playing || chat || reminding)) return
  root.style.removeProperty('--pet-bubble-x')
  const c = center().x
  const half = bubble.offsetWidth / 2
  const gutter = 8
  const shift =
    Math.max(gutter - (c - half), 0) - Math.max(c + half - (window.innerWidth - gutter), 0)
  if (shift) root.style.setProperty('--pet-bubble-x', `${shift.toFixed(0)}px`)
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
    if (closing?.contains(el)) continue
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

const campSolids = (): CollisionActor[] => {
  const obstacles: CollisionActor[] = []
  const scene = trailScene(document)
  if (!scene || trailHost(scene)?.hidden) return obstacles
  for (const element of scene.querySelectorAll<SVGGElement>('[data-camp-solid]')) {
    const matrix = trailScreenMatrix(element)
    const radius = Number(element.dataset.solidRadius)
    const id = element.dataset.campSolid
    if (!matrix || !id || !Number.isFinite(radius) || radius <= 0) continue
    obstacles.push({
      id,
      kind: 'trunk',
      x: matrix.e + window.scrollX,
      y: matrix.f + window.scrollY,
      radius: radius * Math.hypot(matrix.a, matrix.b),
    })
  }
  return obstacles
}
const trailSurfaces =
  '.homepage-board, .trail-tree-panel:not([hidden]), .trail-camp-tools:not([hidden])'

// Keep the pet's whole SVG frame clear of text, including space for its tilt and idle poses.
const coversText = (edge: Edge) => {
  const scale = edge.scale ?? 1
  const left = edge.at.x - (width * scale) / 2 - 2
  const right = edge.at.x + (width * scale) / 2 + 2
  const top = edge.at.y - (edgeY[edge.kind] - viewTop) * unit * scale - 2
  const bottom = top + height(scale) + 4
  const intersects = (rect: DOMRect) =>
    rect.left < right && rect.right > left && rect.top < bottom && rect.bottom > top
  for (const panel of [
    ...document.querySelectorAll<HTMLElement>(trailSurfaces),
    ...(trailScene(document)?.querySelectorAll<HTMLElement>(trailSurfaces) ?? []),
  ])
    if (intersects(panel.getBoundingClientRect())) return true
  const foot = { x: edge.at.x + window.scrollX, y: edge.at.y + window.scrollY }
  if (campSolids().some(other => circleClearance(foot, width * 0.35 * scale, other) < 0))
    return true
  const nodes = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
    {
      acceptNode: node => {
        if (root?.contains(node)) return NodeFilter.FILTER_REJECT
        if (node instanceof Element) {
          if (node.matches('script, style, .trail-canvas')) return NodeFilter.FILTER_REJECT
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
  if (p.foot) {
    const scale = p.scale ?? 1
    return {
      x: p.foot.x - (width * scale) / 2,
      y: p.foot.y - (edgeY[p.kind] - viewTop) * unit * scale,
    }
  }
  let x: number
  let y: number
  if (p.el) {
    const rect = boxOf(p.el)
    x = rect.left + p.t * rect.width
    y = (p.kind === 'hang' ? rect.bottom : rect.top) + (p.drop ?? 0)
  } else {
    x = p.t * window.innerWidth
    y = window.innerHeight - margin - (viewBottom - edgeY.floor) * unit
  }
  const scale = p.scale ?? 1
  x = clamp(x, margin + (width * scale) / 2, window.innerWidth - margin - (width * scale) / 2)
  return {
    x: x - (width * scale) / 2 + window.scrollX,
    y: y - (edgeY[p.kind] - viewTop) * unit * scale + window.scrollY,
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
  friendFollow(perch, doc, 'place')
  render()
}

// Left alone, the pet gets up now and then, laptop or not: mostly a stroll along its own edge, a
// sitter stands up first, and sometimes it hops off to another edge nearby.
const wander = () => {
  const now = performance.now()
  const s = state(now)
  const free = s === 'idle' || s === 'wait' || s === 'work'
  if (!free || now - lastMove < 3000 || hop || carry || reducedMotion() || document.hidden) return
  if (document.body.dataset.slides === 'true') return
  settled = now
  armQuietTimer()
  if (trailWalk?.path.hasAttribute('data-pet-expedition')) {
    walkTrail()
    return
  }
  const note = Math.random() < 0.4 ? sidenoteInView() : null
  if (note) return readSidenote(note)
  if (Math.random() < 0.6 && walkTrail()) return
  if (
    !perch ||
    perch.kind === 'hang' ||
    (perch.kind === 'floor' && perch.foot) ||
    // It stays on a triathlon window until the window closes, strolling along its top.
    (Math.random() < 0.25 && !perch.el?.matches(triWindowSelector))
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

type ScaleStop = readonly [number, number]
type PetTerrain = { top: number; base: number; treeline: number; meadowTop: number }
const trailData = (value: string | undefined): unknown => {
  if (!value) return null
  try {
    return JSON.parse(value)
  } catch {
    return null
  }
}
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const pair = (value: unknown): value is ScaleStop =>
  Array.isArray(value) && value.length === 2 && finite(value[0]) && finite(value[1])
const petHandoff = (path: SVGPathElement) => {
  const value = trailData(path.dataset.petHandoff)
  if (
    !value ||
    typeof value !== 'object' ||
    !('at' in value) ||
    !finite(value.at) ||
    value.at <= 0 ||
    value.at >= 1 ||
    !('exit' in value) ||
    !pair(value.exit) ||
    !('entry' in value) ||
    !pair(value.entry)
  )
    return null
  return { at: value.at, exit: value.exit, entry: value.entry }
}
const petTerrain = (path: SVGPathElement): PetTerrain | null => {
  const raw =
    path.ownerSVGElement?.querySelector<SVGElement>('[data-trail-terrain]')?.dataset.trailTerrain
  const value = trailData(raw)
  if (
    !value ||
    typeof value !== 'object' ||
    !('alpine' in value) ||
    !('treeline' in value) ||
    !('meadowTop' in value) ||
    !finite(value.treeline) ||
    !finite(value.meadowTop)
  )
    return null
  const alpine = value.alpine
  if (
    !alpine ||
    typeof alpine !== 'object' ||
    !('top' in alpine) ||
    !('base' in alpine) ||
    !finite(alpine.top) ||
    !finite(alpine.base)
  )
    return null
  return {
    top: alpine.top,
    base: alpine.base,
    treeline: value.treeline,
    meadowTop: value.meadowTop,
  }
}
const trailScaleStops = (path: SVGPathElement) => {
  const stops: ScaleStop[] = []
  const values = trailData(path.dataset.petScaleStops)
  if (Array.isArray(values))
    for (const value of values)
      if (pair(value) && value[0] >= 0 && value[0] <= 1 && value[1] > 0 && value[1] <= 1)
        stops.push(value)
  return stops.sort((left, right) => left[0] - right[0])
}
const trailScale = (
  position: number,
  point: Point,
  stops: ScaleStop[],
  terrain: PetTerrain | null,
) => {
  if (stops.length) {
    const first = stops[0]
    if (position <= first[0]) return first[1]
    for (let index = 1; index < stops.length; index++) {
      const from = stops[index - 1]
      const to = stops[index]
      if (position <= to[0]) {
        const fraction = clamp((position - from[0]) / Math.max(0.000001, to[0] - from[0]), 0, 1)
        return from[1] + (to[1] - from[1]) * (index === 2 ? easeTrailDepth(fraction) : fraction)
      }
    }
    return stops[stops.length - 1][1]
  }
  if (!terrain || point.y < terrain.top) return 1
  if (point.y < terrain.base)
    return (
      0.18 + 0.17 * clamp((point.y - terrain.top) / Math.max(1, terrain.base - terrain.top), 0, 1)
    )
  const depth =
    (point.y - terrain.treeline) / Math.max(1, terrain.meadowTop + 160 - terrain.treeline)
  return 0.35 + 0.65 * clamp(depth, 0, 1)
}
const trailKey = (path: SVGPathElement) => {
  const opening =
    path.ownerSVGElement?.querySelector<SVGPathElement>('[data-pet-expedition]')?.dataset
      .petExpedition
  return path.hasAttribute('data-pet-expedition')
    ? `expedition:${path.dataset.petExpedition ?? ''}`
    : `upper:${opening ?? ''}:${path.dataset.petWalkPath ?? ''}`
}
const petTrail = (element: Element | null): element is SVGPathElement =>
  element instanceof SVGPathElement &&
  element.matches('[data-pet-walk-path], [data-pet-expedition]')
const currentTrailPath = (path: SVGPathElement) =>
  path.isConnected && !!path.getAttribute('d') && trailScene(document)?.contains(path) === true

// The expedition's separate subpaths meet through a hidden screen-edge transfer.
type TrailSample = { point: DOMPoint; progress: number; transfer?: 'exit' | 'enter' | 'resume' }
const trailPoints = new WeakMap<
  SVGPathElement,
  { d: string; step: number; handoff: string | undefined; pts: TrailSample[] }
>()
const trailRoutes = () => {
  const scene = trailScene(document)
  const host = scene && trailHost(scene)
  if (
    !scene ||
    scene.dataset.trailRendered !== 'true' ||
    host?.hidden !== false ||
    host.dataset.trailResizing === 'true'
  )
    return []
  return Array.from(
    scene.querySelectorAll<SVGPathElement>('[data-pet-walk-path], [data-pet-expedition]'),
  ).flatMap(path => {
    const d = path.getAttribute('d')
    const m = trailScreenMatrix(path)
    if (!d || !m) return []
    const step = 4 / Math.max(Math.hypot(m.a, m.b), Math.hypot(m.c, m.d), 0.001)
    const handoff = petHandoff(path)
    let hit = trailPoints.get(path)
    if (hit?.d !== d || hit.step !== step || hit.handoff !== path.dataset.petHandoff) {
      const geometry = svgPathGeometry(path)
      const length = geometry.getTotalLength()
      const segments = Math.max(1, Math.ceil(length / step))
      let pts = Array.from({ length: segments + 1 }, (_, k): TrailSample => {
        const point = geometry.getPointAtLength((k * length) / segments)
        return { point: new DOMPoint(point.x, point.y), progress: k / segments }
      })
      if (handoff) {
        pts = [
          ...pts.filter(sample => sample.progress < handoff.at),
          { point: new DOMPoint(...handoff.exit), progress: handoff.at, transfer: 'exit' },
          { point: new DOMPoint(...handoff.entry), progress: handoff.at, transfer: 'enter' },
          { point: new DOMPoint(...handoff.entry), progress: handoff.at, transfer: 'resume' },
          ...pts.filter(sample => sample.progress > handoff.at),
        ]
      }
      hit = { d, step, handoff: path.dataset.petHandoff, pts }
      trailPoints.set(path, hit)
    }
    const expedition = path.hasAttribute('data-pet-expedition')
    const stops = expedition ? trailScaleStops(path) : []
    const terrain = expedition ? petTerrain(path) : null
    const span = trailData(path.dataset.petAlpineSpan)
    const base = path.ownerSVGElement ? trailScreenMatrix(path.ownerSVGElement) : null
    const foothill = path.ownerSVGElement?.querySelector<SVGGElement>(
      '[data-trail-depth="foothill"]',
    )
    const depth = foothill ? trailScreenMatrix(foothill) : null
    const dx = base && depth ? depth.e - base.e : 0
    const dy = base && depth ? depth.f - base.f : 0
    // Capture parallax once for a compositor walk, blending beyond the mountain's boundaries.
    const points = hit.pts
    const feet = points.map(({ point: p, progress, transfer }): TrailFoot => {
      const c = p.matrixTransform(m)
      let blend = 0
      if (
        expedition &&
        pair(span) &&
        span[1] > span[0] &&
        progress >= span[0] &&
        transfer !== 'exit'
      ) {
        const edge = Math.min(0.015, (span[1] - span[0]) / 4)
        blend = easeTrailDepth((span[1] + edge - progress) / edge)
      }
      return {
        x: c.x + window.scrollX + dx * blend,
        y: c.y + window.scrollY + dy * blend,
        progress,
        scale:
          transfer === 'enter' || transfer === 'resume'
            ? 0.18
            : expedition
              ? trailScale(progress, p, stops, terrain)
              : 1,
        hidden: transfer === 'exit' || transfer === 'enter',
        handoff: transfer === 'enter',
      }
    })
    return [{ path, feet, expedition, key: trailKey(path) }]
  })
}

const trailGeometry = (path: SVGPathElement) => {
  if (!currentTrailPath(path)) return ''
  const m = trailScreenMatrix(path)
  if (!m) return ''
  return `${path.getAttribute('d')}|${path.dataset.petScaleStops ?? ''}|${path.dataset.petAlpineSpan ?? ''}|${path.dataset.petHandoff ?? ''}|${m.a},${m.b},${m.c},${m.d},${m.e + window.scrollX},${m.f + window.scrollY}|${window.innerWidth},${window.innerHeight}`
}

const trailPointClear = (p: TrailFoot, path: SVGPathElement) => {
  const at = { x: p.x - window.scrollX, y: p.y - window.scrollY }
  const top = at.y - (edgeY.stand - viewTop) * unit * p.scale
  const departure = petHandoff(path)
  const right =
    departure && p.progress <= departure.at
      ? window.innerWidth + width
      : window.innerWidth - margin - (width * p.scale) / 2
  return (
    at.x >= margin + (width * p.scale) / 2 &&
    at.x <= right &&
    top >= margin &&
    top + height(p.scale) <= window.innerHeight - margin &&
    !coversText({ el: path, kind: 'stand', t: 0, at, score: 0, scale: p.scale })
  )
}

type TrailRun = { path: SVGPathElement; feet: TrailFoot[]; length: number }
const trailActors = (scene: SVGSVGElement | null): CollisionActor[] => {
  const actors: CollisionActor[] = campSolids()
  for (const element of scene?.querySelectorAll<SVGGElement>('[data-trail-collision]') ?? []) {
    const id = element.dataset.collisionId
    const kind = element.dataset.trailCollision
    if (
      !id ||
      id === 'pet-blob' ||
      element.dataset.collisionActive === 'false' ||
      (kind !== 'animal' && kind !== 'traveler')
    )
      continue
    const matrix = trailScreenMatrix(element)
    const parent = element.parentElement
    const ground = parent instanceof SVGGraphicsElement ? trailScreenMatrix(parent) : null
    const radius = Number(element.dataset.collisionRadius)
    if (!matrix || !ground || !Number.isFinite(radius)) continue
    actors.push({
      id,
      kind,
      x: matrix.e + window.scrollX,
      y: matrix.f + window.scrollY,
      radius: radius * Math.hypot(ground.a, ground.b),
    })
  }
  return actors
}
const visibleTrailRun = (): TrailRun | null => {
  let best: TrailRun | null = null
  const routes = trailRoutes().sort(
    (left, right) => Number(right.expedition) - Number(left.expedition),
  )
  for (const { path, feet, expedition, key } of routes) {
    const actors = trailActors(path.ownerSVGElement)
    const cursor = trailJourney?.key === key ? trailJourney.progress : 0
    const runs: TrailRun[] = []
    let run: TrailFoot[] = []
    let length = 0
    const finish = () => {
      if (length >= 50) runs.push({ path, feet: run, length })
      run = []
      length = 0
    }
    for (const foot of feet) {
      if (
        (expedition && foot.progress < cursor) ||
        !trailPointClear(foot, path) ||
        (!run.length &&
          actors.some(other => circleClearance(foot, width * 0.35 * foot.scale, other) < 0))
      ) {
        finish()
        continue
      }
      const previous = run[run.length - 1]
      if (previous && !foot.handoff) length += Math.hypot(foot.x - previous.x, foot.y - previous.y)
      run.push(foot)
    }
    finish()
    if (expedition && runs.length) return runs[0]
    if (expedition && trailJourney?.key === key && cursor > 0) return null
    for (const candidate of runs) if (!best || candidate.length > best.length) best = candidate
  }
  return best
}

const currentTrailProgress = () => {
  const current = trailWalk
  if (!current) return null
  if (!hop || !current.duration) return perch?.progress ?? current.feet[0].progress
  return trailFootAt(current, Number(hop.currentTime)).progress
}
const trailFootAt = (current: TrailWalk, time: number): TrailFoot => {
  const elapsed = clamp(time / Math.max(1, current.duration), 0, 1)
  const index = current.offsets.findIndex(offset => offset >= elapsed)
  if (index <= 0) return current.feet[index === 0 ? 0 : current.feet.length - 1]
  const from = current.feet[index - 1]
  const to = current.feet[index]
  if (to.handoff) return from
  const fraction =
    (elapsed - current.offsets[index - 1]) /
    Math.max(0.000001, current.offsets[index] - current.offsets[index - 1])
  return {
    x: from.x + (to.x - from.x) * fraction,
    y: from.y + (to.y - from.y) * fraction,
    scale: from.scale + (to.scale - from.scale) * fraction,
    progress: from.progress + (to.progress - from.progress) * fraction,
    hidden: from.hidden || to.hidden,
  }
}
let trailGuardTimer = 0
let trailActor: SVGGElement | null = null
let trailYieldEpisode = 0
const stopTrailGuard = () => {
  window.clearTimeout(trailGuardTimer)
  trailGuardTimer = 0
  if (root) {
    root.dataset.trailGuard = 'paused'
    root.dataset.yieldingTo = ''
  }
  if (trailActor) {
    trailActor.dataset.yieldingTo = ''
    trailActor.dataset.motion = 'still'
  }
}
const clearTrailActor = () => {
  stopTrailGuard()
  trailActor?.remove()
  trailActor = null
}
const reserveTrailFoot = (path: SVGPathElement, foot: TrailFoot) => {
  if (!currentTrailPath(path)) return clearTrailActor()
  const scene = path.ownerSVGElement
  const matrix = scene ? trailScreenMatrix(scene) : null
  if (!scene || !matrix) return
  if (trailActor?.ownerSVGElement !== scene) {
    trailActor?.remove()
    trailActor = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    trailActor.dataset.trailCollision = 'pet'
    trailActor.dataset.collisionId = 'pet-blob'
    trailActor.setAttribute('aria-hidden', 'true')
    scene.append(trailActor)
  }
  const point = new DOMPoint(foot.x - window.scrollX, foot.y - window.scrollY).matrixTransform(
    matrix.inverse(),
  )
  trailActor.setAttribute('transform', `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)})`)
  trailActor.dataset.collisionRadius = String(
    (width * 0.35 * foot.scale) / Math.hypot(matrix.a, matrix.b),
  )
  trailActor.dataset.collisionActive = 'true'
  trailActor.dataset.groundX = String(point.x)
  trailActor.dataset.groundY = String(point.y)
  trailActor.dataset.progress = foot.progress.toFixed(6)
  trailActor.dataset.scale = String(foot.scale)
}
const guardTrail = (animation: Animation, path: SVGPathElement) => {
  stopTrailGuard()
  const check = () => {
    trailGuardTimer = 0
    if (!root || hop !== animation || !trailWalk) return stopTrailGuard()
    if (
      !currentTrailPath(path) ||
      trailWalk.geometry !== trailGeometry(path) ||
      document.hidden ||
      reducedMotion() ||
      path.closest<HTMLElement>('.trail-scene')?.dataset.trailRendered !== 'true' ||
      trailHost(path)?.hidden !== false
    )
      return pauseTrail()
    const time = Number(animation.currentTime)
    const foot = trailFootAt(trailWalk, time)
    const next = trailFootAt(trailWalk, time + 150)
    if (foot.hidden || next.hidden) {
      reserveTrailFoot(path, foot)
      if (trailActor) trailActor.dataset.collisionActive = 'false'
      root.dataset.trailGuard = 'transfer'
      root.dataset.yieldingTo = ''
      if (animation.playState === 'paused') {
        override = { state: 'walk', until: Infinity }
        animation.play()
        render()
      }
      trailGuardTimer = window.setTimeout(check, 50)
      return
    }
    const actors = trailActors(path.ownerSVGElement)
    const animal = animalInWay(
      foot,
      next,
      width * 0.35 * foot.scale,
      actors,
      root.dataset.yieldingTo,
    )
    if (animal && root.dataset.yieldingTo !== animal.id) trailYieldEpisode++
    const radius = width * 0.35 * Math.max(foot.scale, next.scale)
    const distance = Math.hypot(next.x - foot.x, next.y - foot.y)
    const end = trailWalk.feet[trailWalk.feet.length - 1]
    const reach = Math.min(radius + 6, Math.hypot(end.x - next.x, end.y - next.y))
    const probe = distance
      ? {
          x: next.x + ((next.x - foot.x) * reach) / distance,
          y: next.y + ((next.y - foot.y) * reach) / distance,
        }
      : next
    if (
      !canMove(
        foot,
        probe,
        radius,
        actors.filter(other => other.kind === 'trunk'),
      )
    ) {
      pauseTrail()
      return
    }
    const blocked =
      animal ||
      !canMove(
        foot,
        probe,
        radius,
        actors.filter(other => other.kind === 'traveler'),
      )
    reserveTrailFoot(path, foot)
    root.dataset.trailGuard = 'active'
    root.dataset.yieldingTo = animal?.id ?? ''
    if (trailActor) {
      trailActor.dataset.motion = blocked ? 'yielding' : 'active'
      trailActor.dataset.yieldingTo = animal?.id ?? ''
      trailActor.dataset.yieldEpisode = String(trailYieldEpisode)
    }
    if (blocked && animation.playState !== 'paused') {
      animation.pause()
      override = { state: 'wait', until: Infinity }
      render()
    } else if (!blocked && animation.playState === 'paused') {
      override = { state: 'walk', until: Infinity }
      animation.play()
      render()
    }
    trailGuardTimer = window.setTimeout(check, 50)
  }
  check()
}
const rememberTrail = () => {
  const progress = currentTrailProgress()
  if (trailWalk && progress !== null) trailJourney = { key: trailKey(trailWalk.path), progress }
}

const pauseTrail = () => {
  if (!trailWalk) return
  rememberTrail()
  stopTrailGuard()
  const path = trailWalk.path
  const rect = root?.isConnected ? root.getBoundingClientRect() : null
  hop?.cancel()
  hop = null
  afterMove = null
  override = null
  resizeObserver?.disconnect()
  cancelAnimationFrame(anchorFrame)
  if (root && rect) {
    const scale = rect.width / width
    const foot = {
      x: rect.left + rect.width / 2 + window.scrollX,
      y: rect.top + (edgeY.stand - viewTop) * unit * scale + window.scrollY,
      scale,
      progress: trailJourney?.progress ?? 0,
    }
    perch = { el: path, kind: 'stand', t: 0, foot, scale, progress: foot.progress }
    trailWalk = { path, geometry: trailGeometry(path), feet: [foot], duration: 0, offsets: [0] }
    doc = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
    root.style.transform = transformFor(doc)
    reserveTrailFoot(path, foot)
  }
  settled = performance.now()
  render()
  armQuietTimer()
  scheduleWander()
}
const leaveTrail = () => {
  if (!trailWalk) return
  pauseTrail()
  clearTrailActor()
  const foot = perch?.foot
  trailWalk = null
  perch = null
  if (root && foot) {
    doc = placeOn({ el: null, kind: 'stand', t: 0, foot })
    root.style.transform = transformFor(doc)
  }
  render()
}

// Resume the connected expedition ahead of its cursor, or use a clear upper loop.
const walkTrail = () => {
  if (!root || hop || carry || reducedMotion() || document.hidden) return false
  const best = visibleTrailRun()
  if (!best) return false
  const { path, feet } = best
  let route = feet
  if (!path.hasAttribute('data-pet-expedition') && perch?.el === path && perch.foot) {
    const foot = perch.foot
    const k = feet.findIndex(p => Math.hypot(p.x - foot.x, p.y - foot.y) < 8)
    if (k >= 0) route = feet.slice(k)
  }
  if (route.length < 11) return false
  trailWalk = { path, geometry: trailGeometry(path), feet: route, duration: 0, offsets: [] }
  const walk = () => {
    if (
      !currentTrailPath(path) ||
      path.closest<HTMLElement>('.trail-scene')?.dataset.trailRendered !== 'true' ||
      trailHost(path)?.hidden !== false ||
      trailWalk?.geometry !== trailGeometry(path) ||
      reducedMotion() ||
      document.hidden
    ) {
      pauseTrail()
      return
    }
    follow(route, path)
  }
  const start = route[0]
  if (perch?.foot && Math.hypot(perch.foot.x - start.x, perch.foot.y - start.y) < 12) walk()
  else {
    afterMove = walk
    // An entry hop could cross prose. Fade onto the checked start before the walking animation.
    moveTo(
      { el: path, kind: 'stand', t: 0, foot: start, scale: start.scale, progress: start.progress },
      'fade',
    )
  }
  return true
}

// Walks a run of trail points at a stroll, ending on the last of them.
function follow(feet: TrailFoot[], el: SVGPathElement) {
  if (!root) return
  stopTrailGuard()
  hop?.cancel()
  hop = null
  const from = feet[0]
  const to = feet[feet.length - 1]
  perch = { el, kind: 'stand', t: 0, foot: to, scale: to.scale, progress: to.progress }
  doc = placeOn(perch)
  observePerch()
  if (Math.abs(to.x - from.x) > 2) face = to.x > from.x ? -1 : 1
  let duration = 0
  const times = [0]
  for (let k = 1; k < feet.length; k++) {
    const previous = feet[k - 1]
    const point = feet[k]
    const scale = (point.scale + previous.scale) / 2
    duration += point.handoff
      ? 450
      : previous.hidden && !point.hidden
        ? 180
        : (Math.hypot(point.x - previous.x, point.y - previous.y) / (55 * scale)) * 1000
    times.push(duration)
  }
  const offsets = times.map(time => time / Math.max(1, duration))
  trailWalk = { path: el, geometry: trailGeometry(el), feet, duration, offsets }
  override = { state: 'walk', until: Infinity }
  render()
  const anim = root.animate(
    feet.map((foot, index) => ({
      transform: transformFor(
        placeOn({ el, kind: 'stand', t: 0, foot, scale: foot.scale }),
        foot.scale,
      ),
      opacity: foot.hidden ? 0 : 1,
      easing: feet[index + 1]?.handoff ? 'steps(1, end)' : 'linear',
      offset: offsets[index],
    })),
    { duration, easing: 'linear' },
  )
  hop = anim
  guardTrail(anim, el)
  anim.onfinish = () => {
    if (hop !== anim || !root || !doc) return
    stopTrailGuard()
    reserveTrailFoot(el, to)
    root.style.transform = transformFor(doc)
    anim.cancel()
    hop = null
    override = null
    trailJourney = { key: trailKey(el), progress: to.progress }
    trailWalk = { path: el, geometry: trailGeometry(el), feet: [to], duration: 0, offsets: [0] }
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
// jetpack flight on a gentle curve after scrolling, or a drop from above when the old spot is far
// away or out of view. One Web Animation on the root each time.
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

function moveTo(next: Perch, how: 'chute' | 'stroll' | 'fade' | null = null) {
  const after = afterMove
  afterMove = null
  if (!root) return
  if (!petTrail(next.el)) leaveTrail()
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
    friendFollow(next, target, from ? 'place' : 'appear')
    render()
    after?.()
    return
  }
  const d = Math.hypot(target.x - from.x, target.y - from.y)
  if (d < 2) {
    root.style.transform = end
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
    // The arc stays in view, so a hop up to an edge near the top does not leave the screen.
    const peak = Math.min(
      180,
      50 + d * 0.3,
      Math.max(0, Math.min(from.y, target.y) - window.scrollY),
    )
    const mid = { x: (from.x + target.x) / 2, y: Math.min(from.y, target.y) - peak }
    frames = [
      { transform: transformFor(from), easing: 'cubic-bezier(0.3, 0.6, 0.5, 1)' },
      { transform: transformFor(mid), offset: 0.5, easing: 'cubic-bezier(0.5, 0, 0.7, 0.4)' },
      { transform: end },
    ]
    duration = clamp(320 + d * 0.45, 380, 900)
  }
  const walking = how !== 'fade' && !jet && !chute && frames.length === 2 && !offscreen && d <= 900
  const pace = how === 'stroll' ? 'walk' : 'run'
  if (walking && next.kind === 'floor') layGround(from, target, duration)
  override = {
    state: how === 'fade' ? 'idle' : chute ? 'chute' : jet ? 'jet' : walking ? pace : 'fly',
    until: Infinity,
  }
  render()
  const anim = root.animate(frames, {
    duration,
    easing: jet ? 'ease-in-out' : walking || chute ? 'linear' : undefined,
  })
  hop = anim
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

// On a slide page the pet keeps to the top-right corner of the slides, sitting on the frame. While
// presenting, the frame fills the screen and leaves only 2rem above the deck, so the pet keeps its
// full size and sits over the deck's top-right corner, legs over the side, its seat dropped as far
// as its head needs. When a slide puts text under it there, it steps out of sight.
const slidesPerch = (): Perch | 'covered' | null => {
  if (document.body.dataset.slides !== 'true') return null
  const frame = document.querySelector<HTMLElement>('.slides-root')
  if (!frame) return null
  const deck = frame.querySelector<HTMLElement>('.slides-deck')
  if (!frame.classList.contains('is-presenting') || !deck) {
    const rect = frame.getBoundingClientRect()
    const scale = clamp((rect.top - 4) / ((edgeY.sit - viewTop) * unit), 0.4, 1)
    const t = clamp(1 - (width * scale * 0.8) / rect.width, 0, 1)
    return { el: frame, kind: 'sit', t, scale }
  }
  const rect = deck.getBoundingClientRect()
  const drop = Math.max(0, margin + (edgeY.sit - viewTop) * unit - rect.top)
  const p: Perch = { el: deck, kind: 'sit', t: 1 + (0.36 * unit) / rect.width, drop }
  const at = { x: placeOn(p).x - window.scrollX + width / 2, y: rect.top + drop }
  return unseen(() => coversText({ ...p, at, score: 0 })) ? 'covered' : p
}

function relocate(anchor: Point, avoid: Element | null, how: 'chute' | null = null) {
  if (!root) return
  const slides = slidesPerch()
  const next =
    slides === 'covered' ? null : (slides ?? pickEdge(anchor, avoid) ?? floorNear(anchor.x))
  if (next) moveTo(next, how)
  else {
    root.hidden = true
    hideFriend()
  }
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
  rememberTrail()
  const rect = root.getBoundingClientRect()
  hop?.cancel()
  hop = null
  override = null
  trailWalk = null
  perch = null
  resizeObserver?.disconnect()
  stopGround()
  // The pole holds one; the friend waits out the ride and turns up again where the pet lands.
  hideFriend()
  const rail = railFor()
  carry = rail
  root.style.setProperty('--pet-rail-from', `${rail.from.toFixed(1)}px`)
  root.style.setProperty('--pet-rail-to', `${rail.to.toFixed(1)}px`)
  root.classList.add('is-carried')
  const end = `translate3d(${rail.x.toFixed(1)}px, 0, 0)`
  root.style.transform = end
  // A new scroll-driven animation applies only from the next frame; until then the inline offset
  // holds the pet on the rail, or it flashes at the top of the view for a frame.
  root.style.translate = `0 ${railY(rail).toFixed(1)}px`
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

// Well below the rail, it straps on a jetpack and flies up to it: rising first, then veering over
// onto the pole, wobbling a little on the way. (dx, dy) is where it starts, relative to the rail.
function jetTo(dx: number, dy: number, end: string) {
  if (!root) return
  const steps = 16
  const frames = Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps
    const u = 1 - t
    // A quadratic bow through (dx, 0): straight up to the rail's height, then across onto it.
    const x = u * u * dx + 2 * u * t * dx + Math.sin(t * Math.PI * 5) * 3 * u
    const y = u * u * dy
    return { transform: `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) ${end}` }
  })
  override = { state: 'jet', until: Infinity }
  render()
  const anim = root.animate(frames, {
    duration: clamp(380 + Math.hypot(dx, dy) * 0.9, 600, 1400),
    easing: 'cubic-bezier(0.45, 0, 0.25, 1)',
  })
  hop = anim
  anim.onfinish = () => {
    if (hop !== anim) return
    anim.cancel()
    hop = null
    override = null
    render()
    // Scrolling may have stopped mid-flight; let go only once it has landed.
    if (landAfterJet) {
      landAfterJet = false
      endCarry()
    }
  }
}

// Scrolling stopped: back into the page's coordinates, then down or over to the nearest edge.
function endCarry() {
  if (!root || !carry) return
  if (hop && override?.state === 'jet') {
    landAfterJet = true
    return
  }
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
  friendWatch()
  const target = lookAt ?? pointer
  if (!root || !target || hop) return
  const c = center()
  const dx = target.x - c.x
  const dy = target.y - c.y
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

// Triathlon panels open as windows over the page, unless set to full screen. The pet looks up at
// one as it opens, hops onto its top edge, and jumps back down when it closes.
const triWindows = [
  ['tri-analytics-open', '.tri-analytics'],
  ['tri-map-open', '.tri-map'],
  ['tri-training-open', '.tri-training'],
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
  const scale = clamp((rect.top - 4) / ((edgeY.stand - viewTop) * unit), 0.55, 1)
  const t = clamp((center().x - rect.left) / rect.width, 0.1, 0.9)
  return { el: win, kind: 'stand', t, scale }
}

const onTriWindows = () => {
  if (!root || !doc || carry || reducedMotion()) return
  window.clearTimeout(windowTimer)
  const win = openTriWindow()
  const on = perch?.el?.matches(triWindowSelector) ? perch.el : null
  if (win === on) return
  if (on) {
    closing = on
    if (!win) relocate(center(), on)
    closing = null
  }
  if (!win) {
    lookAt = null
    return
  }
  // It opens with a short scale-up; the pet stares at its top edge meanwhile, then hops on.
  const rect = win.getBoundingClientRect()
  lookAt = { x: rect.left + rect.width / 2, y: rect.top }
  play('notice', 560)
  watch()
  windowTimer = window.setTimeout(() => {
    lookAt = null
    if (!root || carry || openTriWindow() !== win) return
    // Once up there, its eyes go back to the pointer.
    afterMove = () => {
      gaze = ''
      watch()
    }
    moveTo(windowPerch(win))
  }, 520)
}

const onDwell = () => {
  if (!root || !pointer || hop || carry || hovering || reducedMotion()) return
  if (performance.now() < relocateReadyAt || perch?.el?.matches(triWindowSelector)) return
  if (document.body.dataset.slides === 'true') return
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
  if (y >= -margin && y + height(perch?.scale) <= window.innerHeight + margin) {
    reanchor()
    return
  }
  const anchor = pointer ?? { x: window.innerWidth * 0.75, y: window.innerHeight * 0.6 }
  relocate(anchor, null)
}

const hideReminder = () => {
  window.clearTimeout(reminderHideTimer)
  reminding = false
  showBubble()
}

const showReminder = () => {
  if (!root) return
  reminding = true
  showBubble()
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

// The friend, on the friendship page: a second blob, a size smaller, filled sage with a sprout on
// its head. It keeps to the pet's side on whatever edge the pet holds and hops after it. When both
// rest, they now and then hug (a heart rises between them), high-five, or trade a line in French.
// Clicking it makes it cheer, and the pet turns to wave back.
let friend: HTMLElement | null = null
let friendSvg: SVGSVGElement | null = null
let friendMarkup = ''
let friendPerch: Perch | null = null
let friendDoc: Point | null = null
let friendHop: Animation | null = null
// Which side of the pet it keeps (screen x), until that side has no room.
let friendSide = 1
let friendFace = 1
let friendHovering = false
let friendOverride: { state: State; until: number } | null = null
const friendTimers = new Set<number>()
let meetTimer: number | undefined
// The pet's half of a chat with its friend, shown in its own bubble.
let chat: string | null = null
const friendSize = 0.82
const friendLines = [
  ['salut !', 'salut, toi.'],
  ['ça va ?', 'ça va bien.'],
  ['on lit ?', 'toujours.'],
  ['tu restes ?', 'je reste.'],
  ['merci d’être là.', 'avec plaisir.'],
] as const
const friendPage = () => getFullSlug(window) === 'friends'
const later = (fn: () => void, ms: number) => {
  const id = window.setTimeout(() => {
    friendTimers.delete(id)
    fn()
  }, ms)
  friendTimers.add(id)
}

const friendScale = () => friendPerch?.scale ?? friendSize
const friendCenter = (): Point | null =>
  friendDoc && {
    x: friendDoc.x - window.scrollX + (width * friendScale()) / 2,
    y: friendDoc.y - window.scrollY + height(friendScale()) / 2,
  }

function renderFriend() {
  if (!friend || !friendSvg) return
  const now = performance.now()
  const main = svg?.dataset.state
  const next: State =
    friendOverride && now < friendOverride.until
      ? friendOverride.state
      : friendHovering
        ? 'happy'
        : main === 'work' || main === 'wait'
          ? main
          : 'idle'
  if (friendSvg.dataset.state !== next) friendSvg.dataset.state = next
  friendSvg.dataset.kind = friendPerch?.kind ?? 'stand'
  friendSvg.toggleAttribute(
    'data-reading',
    document.documentElement.getAttribute('reader-mode') === 'on',
  )
  friend.style.setProperty('--pet-face', String(friendFace))
  const c = friendCenter()
  if (c) friend.dataset.bubble = c.y - height(friendScale()) / 2 < 64 ? 'below' : 'above'
}

const friendPlay = (next: State, ms: number) => {
  friendOverride = { state: next, until: performance.now() + ms }
  renderFriend()
  later(renderFriend, ms + 20)
}

// Its eyes go to the pointer when it comes close, else to the pet.
function friendWatch() {
  const c = friendCenter()
  if (!friend || !c || friendHop) return
  const near = pointer && Math.hypot(pointer.x - c.x, pointer.y - c.y) < 140 ? pointer : null
  const target = near ?? (doc || carry ? center() : null)
  if (!target) return
  const dx = target.x - c.x
  const dy = target.y - c.y
  if (Math.abs(dx) > 12) friendFace = dx > 0 ? -1 : 1
  const r = Math.hypot(dx, dy)
  const k = Math.min(1, r / (3 * unit)) / Math.max(r, 1)
  const ux = friendFace * dx * k
  const uy = dy * k
  friend.style.setProperty('--pet-face', String(friendFace))
  friend.style.setProperty('--pet-lx', (ux * 0.18).toFixed(2))
  friend.style.setProperty('--pet-ly', (uy * (uy < 0 ? 0.11 : 0.16)).toFixed(2))
  friend.style.setProperty('--pet-tilt', (ux * 7).toFixed(1))
}

// Where the friend goes when the pet holds `p` from `m` (its top-left in document coordinates):
// the same edge, shoulder to shoulder, on its usual side if that side has room and no text.
function friendSpot(p: Perch, m: Point): { perch: Perch; doc: Point } | null {
  const s = p.scale ?? 1
  const fs = s * friendSize
  const mainX = m.x - window.scrollX + (width * s) / 2
  const gap = 0.38 * width * (s + fs)
  const beside = (side: number, off: number) => {
    const x = mainX + side * (gap + off)
    let perch: Perch
    if (p.foot) perch = { ...p, foot: { x: p.foot.x + side * (gap + off), y: p.foot.y }, scale: fs }
    else if (p.el) {
      // On a box it keeps over the box: a sitter holds a corner, which leaves only the inside.
      const rect = boxOf(p.el)
      if (x < rect.left + width * fs * 0.2 || x > rect.right - width * fs * 0.2) return null
      perch = { ...p, t: (x - rect.left) / rect.width, scale: fs }
    } else perch = { ...p, t: x / window.innerWidth, scale: fs }
    const at = placeOn(perch)
    const cx = at.x - window.scrollX + (width * fs) / 2
    // Held back into the view, it would crowd the pet.
    if (Math.abs(cx - x) > 2) return null
    const foot = at.y - window.scrollY + (edgeY[perch.kind] - viewTop) * unit * fs
    if (coversText({ ...perch, at: { x: cx, y: foot }, score: 0 })) return null
    friendSide = side
    return { perch, doc: at }
  }
  // Up on the pet's head.
  const onHead = () => {
    const head = { x: mainX, y: m.y - window.scrollY + (-0.97 - viewTop) * unit * s }
    const perch: Perch = {
      el: null,
      kind: 'stand',
      t: 0,
      foot: { x: head.x + window.scrollX, y: head.y + window.scrollY },
      scale: fs,
    }
    const top = head.y - (edgeY.stand - viewTop) * unit * fs
    if (top < margin || coversText({ ...perch, at: head, score: 0 })) return null
    return { perch, doc: placeOn(perch) }
  }
  // Boxed in by text, it takes the nearest clear edge a short way off, out of the pet's way.
  const nearby = () => {
    const pet = { x: mainX, y: m.y - window.scrollY, w: width * s, h: height(s) }
    const center = { x: pet.x, y: pet.y + pet.h / 2 }
    for (const edge of edgesNear(center, null).slice(0, 20)) {
      if (!inFlow(edge.el!)) continue
      const rect = boxOf(edge.el!)
      const offs =
        edge.kind === 'stand'
          ? [0, ...Array.from({ length: 24 }, (_, k) => (k + 1) * floorStep)]
          : [0]
      for (const off of offs)
        for (const side of off ? [friendSide, -friendSide] : [1]) {
          const x = edge.at.x + side * off
          if (x < rect.left + width * fs * 0.2 || x > rect.right - width * fs * 0.2) continue
          if (Math.hypot(x - center.x, edge.at.y - center.y) > 400) continue
          const top = edge.at.y - (edgeY[edge.kind] - viewTop) * unit * fs
          const overlaps =
            Math.abs(x - pet.x) < (pet.w + width * fs) / 2 &&
            top < pet.y + pet.h &&
            top + height(fs) > pet.y
          if (overlaps) continue
          if (edge.kind === 'stand') {
            const hit = document
              .elementsFromPoint(x, edge.at.y + 3)
              .find(e => !root?.contains(e) && !friend?.contains(e))
            if (!hit || !(edge.el === hit || edge.el!.contains(hit))) continue
          }
          const perch: Perch = {
            el: edge.el,
            kind: edge.kind,
            t: (x - rect.left) / rect.width,
            scale: fs,
          }
          if (coversText({ ...perch, at: { x, y: edge.at.y }, score: 0 })) continue
          return { perch, doc: placeOn(perch) }
        }
    }
    return null
  }
  // Shoulder to shoulder, else on its head, else a few steps along the same edge, else nearby.
  const probe = () => {
    for (const side of [friendSide, -friendSide]) {
      const spot = beside(side, 0)
      if (spot) return spot
    }
    const head = onHead()
    if (head) return head
    for (let off = floorStep; off <= 12 * floorStep; off += floorStep)
      for (const side of [friendSide, -friendSide]) {
        const spot = beside(side, off)
        if (spot) return spot
      }
    return nearby()
  }
  if (friend) friend.style.visibility = 'hidden'
  try {
    return unseen(probe)
  } finally {
    if (friend) friend.style.visibility = ''
  }
}

function hideFriend() {
  if (!friend) return
  friendHop?.cancel()
  friendHop = null
  friend.hidden = true
  friendDoc = null
  friendPerch = null
}

// Follows the pet onto `p`: placed at once, walking beside it, hopping after it, or appearing once
// it has landed (after a far jump or a ride on the rail).
function friendFollow(p: Perch, m: Point, how: 'place' | 'walk' | 'hop' | 'appear', duration = 0) {
  if (!friend) return
  const spot = friendSpot(p, m)
  if (!spot) return hideFriend()
  const from = friend.hidden ? null : friendDoc
  friendHop?.cancel()
  friendHop = null
  friendPerch = spot.perch
  friendDoc = spot.doc
  const fs = friendScale()
  const end = transformFor(spot.doc, fs)
  const settle = () => {
    if (!friend) return
    friend.style.transform = end
    friendWatch()
    renderFriend()
  }
  if (how === 'place' || reducedMotion()) {
    friend.hidden = false
    return settle()
  }
  if (!from || how === 'appear') {
    friend.hidden = true
    later(() => {
      if (!friend || friendDoc !== spot.doc) return
      friend.hidden = false
      friend.style.transform = end
      friend.animate(
        [
          { transform: transformFor({ x: spot.doc.x, y: spot.doc.y - 40 }, fs), opacity: 0 },
          { transform: end, opacity: 1 },
        ],
        { duration: 320, easing: 'cubic-bezier(0.2, 0, 0, 1)' },
      )
      friendPlay('land', 380)
      settle()
    }, duration)
    return
  }
  const d = Math.hypot(spot.doc.x - from.x, spot.doc.y - from.y)
  if (d < 2) return settle()
  if (Math.abs(spot.doc.x - from.x) > 2) friendFace = spot.doc.x > from.x ? -1 : 1
  const walking = how === 'walk'
  const peak = Math.min(
    140,
    40 + d * 0.3,
    Math.max(0, Math.min(from.y, spot.doc.y) - window.scrollY),
  )
  const frames: Keyframe[] = walking
    ? [{ transform: transformFor(from, fs) }, { transform: end }]
    : [
        { transform: transformFor(from, fs), easing: 'cubic-bezier(0.3, 0.6, 0.5, 1)' },
        {
          transform: transformFor(
            { x: (from.x + spot.doc.x) / 2, y: Math.min(from.y, spot.doc.y) - peak },
            fs,
          ),
          offset: 0.5,
          easing: 'cubic-bezier(0.5, 0, 0.7, 0.4)',
        },
        { transform: end },
      ]
  const anim = friend.animate(frames, {
    duration: walking ? duration : clamp(320 + d * 0.45, 380, 900),
    // A beat behind the pet, as a friend tagging along would be.
    delay: walking ? 120 : 160,
    easing: walking ? 'linear' : undefined,
    fill: 'backwards',
  })
  friendHop = anim
  friendOverride = {
    state: walking ? (svg?.dataset.state === 'walk' ? 'walk' : 'run') : 'fly',
    until: Infinity,
  }
  renderFriend()
  anim.onfinish = () => {
    if (friendHop !== anim || !friend) return
    friend.style.transform = end
    anim.cancel()
    friendHop = null
    friendOverride = null
    if (walking) settle()
    else {
      settle()
      friendPlay('land', 380)
    }
  }
}

const showFriendBubble = (text: string, ms: number) => {
  if (!friend) return
  const bubble = friend.querySelector<HTMLElement>('.garden-pet-bubble-text')
  if (bubble) {
    bubble.textContent = text
    bubble.lang = 'fr'
  }
  friend.classList.add('is-speaking')
  later(() => friend?.classList.remove('is-speaking'), ms)
}

const say = (text: string, ms: number) => {
  chat = text
  showBubble()
  later(() => {
    chat = null
    showBubble()
  }, ms)
}

// A heart in pen, rising from between the two and fading.
const heartUp = (at: Point) => {
  if (reducedMotion()) return
  const el = document.createElement('div')
  el.className = 'garden-pet-heart'
  el.setAttribute('aria-hidden', 'true')
  el.innerHTML = `<svg viewBox="-0.7 -0.7 1.4 1.4">${heart(0.07, 'var(--dark)')}</svg>`
  el.style.left = `${(at.x + window.scrollX - 11).toFixed(1)}px`
  el.style.top = `${(at.y + window.scrollY - 11).toFixed(1)}px`
  document.body.append(el)
  el.addEventListener('animationend', () => el.remove(), { once: true })
}

// Lean in toward each other and back.
const nudge = (el: Element | null | undefined, dx: number, ms: number) =>
  el?.animate(
    [
      { translate: '0 0' },
      { translate: `${dx}px 0`, offset: 0.3 },
      { translate: `${dx}px 0`, offset: 0.7 },
      { translate: '0 0' },
    ],
    { duration: ms, easing: 'ease-in-out' },
  )

const bounce = (el: Element | null | undefined, delay = 0) =>
  el?.animate([{ translate: '0 0' }, { translate: '0 -8px', offset: 0.4 }, { translate: '0 0' }], {
    duration: 360,
    delay,
    easing: 'cubic-bezier(0.3, 0.6, 0.5, 1)',
  })

// One of their moments together, when both rest and the reader is not busy with either.
function meet() {
  meetTimer = undefined
  const fc = friendCenter()
  const busy =
    !friend ||
    friend.hidden ||
    !fc ||
    !root ||
    root.hidden ||
    !doc ||
    hop ||
    friendHop ||
    carry ||
    speaking ||
    reminding ||
    hovering ||
    friendHovering ||
    document.hidden ||
    reducedMotion()
  if (busy) return scheduleMeet()
  const mc = center()
  // Face each other: -1 faces right, as drawn mirrored.
  face = fc.x > mc.x ? -1 : 1
  friendFace = -face
  lookAt = fc
  gaze = ''
  watch()
  friendWatch()
  const toward = fc.x > mc.x ? 1 : -1
  const pick = Math.random()
  // Apart, on different edges, they can only call over to each other.
  const close = Math.hypot(fc.x - mc.x, fc.y - mc.y) < 2.5 * width
  const meeting = !close || pick >= 0.7 ? 'chat' : pick < 0.4 ? 'hug' : 'five'
  if (root) root.dataset.meeting = meeting
  let ms = 1900
  if (meeting === 'hug') {
    play('happy', 1800)
    friendPlay('happy', 1800)
    nudge(svg, toward * 5, 1600)
    nudge(friendSvg, -toward * 5, 1600)
    later(
      () =>
        heartUp({ x: (mc.x + fc.x) / 2, y: Math.min(mc.y, fc.y) - height(friendScale()) * 0.45 }),
      380,
    )
  } else if (meeting === 'five') {
    play('cheer', 1100)
    friendPlay('cheer', 1100)
    bounce(svg)
    bounce(friendSvg, 120)
    ms = 1300
  } else {
    const [first, reply] = friendLines[Math.floor(Math.random() * friendLines.length)]
    const friendFirst = Math.random() < 0.5
    if (friendFirst) showFriendBubble(first, 1500)
    else say(first, 1500)
    later(() => (friendFirst ? say(reply, 1500) : showFriendBubble(reply, 1500)), 1300)
    ms = 2900
  }
  later(() => {
    if (root) delete root.dataset.meeting
    lookAt = null
    gaze = ''
    watch()
    scheduleMeet()
  }, ms)
}

function scheduleMeet() {
  window.clearTimeout(meetTimer)
  if (!friend) return
  meetTimer = window.setTimeout(meet, 6000 + Math.random() * 8000)
}

function spawnFriend() {
  if (friend || !friendPage()) return
  friendMarkup ||= petSvg('var(--pet-friend-fill)', 'var(--dark)', undefined, 19).replace(
    '<g class="pet-body">',
    `<g class="pet-body"><g data-part="sprout">${sprout((0.08 * 0.042) / 0.055, 'var(--dark)')}</g>`,
  )
  const el = document.createElement('div')
  el.className = 'garden-pet garden-pet-friend'
  el.style.transformOrigin = '0 0'
  el.dataset.petWidget = ''
  el.hidden = true
  el.innerHTML = `<button type="button" class="garden-pet-button" aria-label="the pet's friend">${friendMarkup}</button><div class="garden-pet-bubble" role="status"><span class="garden-pet-bubble-text"></span></div>`
  document.body.appendChild(el)
  friend = el
  friendSvg = el.querySelector('svg')
  requestAnimationFrame(() => el.classList.add('is-visible'))
  scheduleMeet()
}

function removeFriend() {
  friendHop?.cancel()
  friendHop = null
  for (const timer of friendTimers) window.clearTimeout(timer)
  friendTimers.clear()
  window.clearTimeout(meetTimer)
  meetTimer = undefined
  chat = null
  friend?.remove()
  friend = null
  friendSvg = null
  friendDoc = null
  friendPerch = null
  friendHovering = false
  friendOverride = null
}

const spawn = () => {
  markup ||= petSvg()
  const el = document.createElement('div')
  el.className = 'garden-pet'
  el.style.transformOrigin = '0 0'
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
  showBubble()
  requestAnimationFrame(() => el.classList.add('is-visible'))
}

const despawn = () => {
  clearTrailActor()
  hop?.cancel()
  hop = null
  landAfterJet = false
  trailWalk = null
  trailJourney = null
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
  removeFriend()
}

const syncPets = () => {
  const enabled = readPetsEnabled()
  document.body.dataset.pets = enabled ? 'on' : 'off'
  // The 404 scene paints its own pet into the ruin, starting from where this one stood.
  if (!enabled || document.querySelector('.nf-scene')) {
    despawn()
    return
  }
  // The friend lives on the friendship page alone; the morph drops it with the pet.
  if (friend && (!friend.isConnected || !friendPage())) removeFriend()
  // The SPA morphs <body> on navigation, which drops the pet and the box it held. It comes back
  // near where it was on screen and drops onto the nearest edge of the new page.
  if (!root || !root.isConnected) {
    clearTrailActor()
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
    trailJourney = null
    afterMove = null
    spawn()
    spawnFriend()
    relocate(before, null)
  } else if (perch?.el && !perch.el.isConnected) relocate(center(), null)
  if (friendPage() && !friend) {
    spawnFriend()
    if (perch && doc && !hop) friendFollow(perch, doc, 'appear')
  }
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
  const controller = new AbortController()
  const { signal } = controller
  trailSpaceSignal = signal
  trailSpaces = new WeakMap()
  syncPets()
  const scene = document.querySelector<HTMLElement>('.homepage-trail-scene')
  let trailFrame = 0
  let trailWanted = false
  let waiting: Animation | null = null
  const refreshTrail = () => {
    trailFrame = 0
    if (signal.aborted || !scene || scene.hidden || document.hidden || reducedMotion()) return
    if (scene.dataset.trailResizing === 'true') {
      trailWanted ||= !!trailWalk
      if (trailWalk?.duration) pauseTrail()
      return
    }
    // The frame replaces its routes over several frames of a redraw and reports rendered only at
    // the end. A walk whose route has moved or gone stops at once, before the rendered check, so
    // the pet does not keep hopping along a path that is no longer there; it resumes on the new
    // route once the scene is ready.
    const currentTrail = trailWalk
    if (
      currentTrail &&
      (!currentTrailPath(currentTrail.path) ||
        currentTrail.geometry !== trailGeometry(currentTrail.path) ||
        currentTrail.feet.some(foot => !trailPointClear(foot, currentTrail.path)))
    ) {
      pauseTrail()
      trailWanted = true
    }
    if (trailScene(document)?.dataset.trailRendered !== 'true') return
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
    if (walkTrail()) {
      trailWanted = false
      return
    }
    const resting = trailWalk
    if (
      resting &&
      (!currentTrailPath(resting.path) || !trailPointClear(resting.feet[0], resting.path))
    ) {
      const anchor = center()
      const next = pickEdge(anchor, resting.path) ?? floorNear(anchor.x)
      if (next) moveTo(next, 'fade')
      else if (root) root.hidden = true
    }
  }
  // The frame can redraw in its own animation callback after the host's reflow event.
  const queueTrail = () => {
    if (trailFrame || signal.aborted) return
    trailFrame = requestAnimationFrame(() => {
      trailFrame = requestAnimationFrame(refreshTrail)
    })
  }
  const pausePageTrail = () => {
    cancelAnimationFrame(trailFrame)
    trailFrame = 0
    waiting = null
    trailWanted = false
    pauseTrail()
  }
  const closeTrail = () => {
    pausePageTrail()
    leaveTrail()
    trailJourney = null
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
    scene.addEventListener('trail:ready', openTrail, { signal })
    scene.addEventListener('load', openTrail, { capture: true, signal })
    // The host's events bracket a redraw; the route paths change in between, so they are watched.
    let paths: MutationObserver | null = null
    const watchPaths = () => {
      paths?.disconnect()
      const trail = trailScene(document)
      if (!trail) return
      const canvas = trail.querySelector('svg.trail-canvas')
      paths = new MutationObserver(records => {
        if (
          records.some(
            record =>
              record.target !== canvas ||
              [...record.addedNodes, ...record.removedNodes].some(
                node => node instanceof Element && petTrail(node),
              ),
          )
        )
          queueTrail()
      })
      paths.observe(trail, { attributes: true, attributeFilter: ['data-trail-rendered'] })
      if (canvas) paths.observe(canvas, { childList: true })
      const walks = trail.querySelector('[data-pet-walk-paths]')
      if (walks)
        paths.observe(walks, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ['d'],
        })
      const expedition = trail.querySelector('[data-pet-expedition]')
      if (expedition)
        paths.observe(expedition, {
          attributes: true,
          attributeFilter: [
            'd',
            'data-pet-handoff',
            'data-pet-scale-stops',
            'data-pet-alpine-span',
          ],
        })
    }
    scene.addEventListener('trail:mounted', watchPaths, { signal })
    scene.addEventListener('trail:ready', watchPaths, { signal })
    watchPaths()
    signal.addEventListener('abort', () => paths?.disconnect(), { once: true })
    const controlsChanged = () => {
      if (root && !carry) {
        const pet = root.getBoundingClientRect()
        const covered = [...document.querySelectorAll<HTMLElement>(trailSurfaces)].some(panel => {
          const box = panel.getBoundingClientRect()
          return (
            box.left < pet.right &&
            box.right > pet.left &&
            box.top < pet.bottom &&
            box.bottom > pet.top
          )
        })
        if (covered) {
          pauseTrail()
          const anchor = center()
          const next = pickEdge(anchor, perch?.el ?? null) ?? floorNear(anchor.x)
          if (next) moveTo(next, 'fade')
          else {
            clearTrailActor()
            root.hidden = true
          }
        }
      }
      trailWanted = true
      queueTrail()
    }
    const controlsObserver = new MutationObserver(controlsChanged)
    controlsObserver.observe(scene, {
      subtree: true,
      attributes: true,
      attributeFilter: ['hidden'],
    })
    const observeControls = () => {
      const drawing = trailScene(document)
      if (drawing)
        controlsObserver.observe(drawing, {
          subtree: true,
          attributes: true,
          attributeFilter: ['hidden'],
        })
    }
    observeControls()
    scene.addEventListener('trail:mounted', observeControls, { signal })
    signal.addEventListener('abort', () => controlsObserver.disconnect(), { once: true })
    window.addEventListener('resize', queueTrail, { passive: true, signal })
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    motion.addEventListener('change', () => (motion.matches ? pausePageTrail() : openTrail()), {
      signal,
    })
    document.addEventListener(
      'visibilitychange',
      () => {
        if (document.hidden) {
          const resume = trailWanted || !!trailWalk
          pausePageTrail()
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
      // The friend cheers; the pet turns to it and waves back.
      if (hit && friend?.contains(hit) && hit.closest('.garden-pet-button') && !friendHop) {
        const fc = friendCenter()
        friendPlay('cheer', 1400)
        bounce(friendSvg)
        if (fc && root && doc && !hop && !carry) {
          lookAt = fc
          gaze = ''
          watch()
          play('wave', 1400)
          heartUp({ x: fc.x, y: fc.y - height(friendScale()) * 0.5 })
          later(() => {
            lookAt = null
            gaze = ''
            watch()
          }, 1400)
        }
        return
      }
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
      const overFriend = !!friend
        ?.querySelector('.garden-pet-button')
        ?.contains(event.target as Node)
      if (overFriend !== friendHovering) {
        friendHovering = overFriend
        renderFriend()
      }
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
      if (
        trailWalk?.path.hasAttribute('data-pet-expedition') ||
        trailJourney?.key.startsWith('expedition:')
      )
        trailWanted = true
      landAfterJet = false
      if (!carry) startCarry()
      else if (root && !scrollTimeline) root.style.translate = `0 ${railY(carry).toFixed(1)}px`
      scrollTimer = window.setTimeout(() => {
        endCarry()
        if (trailWanted) queueTrail()
      }, 200)
    },
    { passive: true, signal },
  )
  window.addEventListener(
    'resize',
    () => {
      window.clearTimeout(scrollTimer)
      scrollTimer = window.setTimeout(() => {
        if (carry) endCarry()
        else onSettle()
        if (trailWanted) queueTrail()
      }, 220)
    },
    { passive: true, signal },
  )
  window.addEventListener('pagehide', () => (doc || carry) && savePosition(), { signal })
  const onSpeech = (event: CustomEventMap['speech']) => {
    const { state, text, lang } = event.detail
    speaking = state === 'idle' ? null : { text, lang, playing: state === 'playing' }
    showBubble()
  }
  document.addEventListener('speech', onSpeech)
  // Presenting puts the slides in full screen, where only that element's subtree shows, so the
  // pet moves in with it and back out after.
  const onFullscreen = () => {
    if (!root || document.body.dataset.slides !== 'true') return
    const host =
      document.fullscreenElement instanceof HTMLElement ? document.fullscreenElement : document.body
    if (root.parentElement === host) return
    hop?.cancel()
    hop = null
    host.append(root)
    root.style.transform = 'none'
    const rect = root.getBoundingClientRect()
    origin = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
    doc = null
    perch = null
    // The window keeps its old size for a frame or two after the event (27 ms leaving full screen
    // in Chromium), so place the pet once the size has held for a few frames.
    let size = ''
    let still = 0
    const start = performance.now()
    const place = () => {
      const now = `${window.innerWidth}x${window.innerHeight}`
      still = now === size ? still + 1 : 0
      size = now
      if (still < 4 && performance.now() - start < 600) requestAnimationFrame(place)
      else relocate(center(), null)
    }
    requestAnimationFrame(place)
  }
  document.addEventListener('fullscreenchange', onFullscreen, { signal })
  // Reader mode puts its glasses on (render reads the mode).
  let readingTimer: number | undefined
  let reflowTimer: number | undefined
  const onReaderMode = (event: CustomEventMap['readermodechange']) => {
    if (!svg) return
    clearTimeout(readingTimer)
    clearTimeout(reflowTimer)
    // Only the toggle animates; a pet spawned on a new page is already wearing them.
    const change = event.detail.mode === 'on' ? 'data-reading-on' : 'data-reading-off'
    const animate = event.detail.mode === 'on' || svg.hasAttribute('data-reading')
    svg.removeAttribute('data-reading-on')
    svg.removeAttribute('data-reading-off')
    if (animate) {
      svg.setAttribute(change, '')
      readingTimer = window.setTimeout(() => svg?.removeAttribute(change), 420)
    }
    render()
    // The text reflows into the freed width, under the pet as often as not; once it has settled,
    // a pet whose edge went away or now covers text finds another.
    reflowTimer = window.setTimeout(() => {
      if (!root || !perch || !doc || hop || carry || petTrail(perch.el)) return
      const scale = perch.scale ?? 1
      const at = {
        x: doc.x - window.scrollX + (width * scale) / 2,
        y: doc.y - window.scrollY + (edgeY[perch.kind] - viewTop) * unit * scale,
      }
      const gone = !!perch.el && !perch.el.getClientRects().length
      if (gone || unseen(() => coversText({ ...perch!, at, score: 0 })))
        relocate(center(), gone ? null : perch.el)
    }, 300)
  }
  document.addEventListener('readermodechange', onReaderMode)
  // Each slide brings its own text, which may now sit under the presenting pet (or clear it).
  let slideTimer: number | undefined
  const onSlide = () => {
    if (!root?.parentElement?.classList.contains('is-presenting')) return
    clearTimeout(slideTimer)
    // Past the slide's enter transition, so the text is measured where it rests.
    slideTimer = window.setTimeout(() => relocate(center(), null), 450)
  }
  document.addEventListener('slidechange', onSlide)
  document.addEventListener('contentdecrypted', onSlide)
  const windows = new MutationObserver(onTriWindows)
  for (const tri of document.querySelectorAll('.triathlon'))
    windows.observe(tri, { attributes: true, attributeFilter: ['class'] })
  window.addCleanup(() => {
    closeTrail()
    controller.abort()
    if (trailSpaceSignal === signal) {
      trailSpaceSignal = undefined
      trailSpaces = new WeakMap()
    }
    document.removeEventListener('speech', onSpeech)
    document.removeEventListener('readermodechange', onReaderMode)
    clearTimeout(readingTimer)
    clearTimeout(reflowTimer)
    document.removeEventListener('slidechange', onSlide)
    document.removeEventListener('contentdecrypted', onSlide)
    clearTimeout(slideTimer)
    speaking = null
    windows.disconnect()
    window.clearTimeout(windowTimer)
    lookAt = null
  })
})
