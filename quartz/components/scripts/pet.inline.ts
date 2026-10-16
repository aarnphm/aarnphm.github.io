import { hand, heart, petSvg, pressure, sprout, stroke } from '../../util/mascot'
import { getFullSlug } from '../../util/path'
import { readPetSpot, writePetSpot } from '../../util/pet-spot'
import { sectionOf } from '../../util/section-icon'
import { createSvgSpace, svgPathGeometry } from '../../util/svg-geometry'
import { easeTrailDepth } from '../../util/trail-depth'
import { trailHost, trailScene } from '../../util/trail-scene'
import { STORM_PERIOD, visitorPrecipitation, wetFall } from '../../util/visitor-precipitation'
import { nextSunEvent, readVisitorForecast } from '../../util/visitor-weather'
import {
  animalInWay,
  canMove,
  circleClearance,
  queuedAnimal,
  readTrunks,
  yieldPatience,
  type CollisionActor,
  type SpacingBounds,
} from '../visitor-trail/actor-spacing'
import { clockObstacles, trailClockRegion } from '../visitor-trail/clock-clearance'
import { footprintTransform, impression } from '../visitor-trail/footprints'
import { pondDistance, type Pond } from '../visitor-trail/landscape'
import {
  detourPetTrailSteps,
  type PetTrailFoot,
  type PetTrailSearch,
} from '../visitor-trail/pet-trail'
import { TRAIL_PET_RADIUS, trailPetSprite, trailPetTop } from '../visitor-trail/trail-pet'
import { visibleTextBoxes } from './pet-text-boxes'

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
type HopPoint = Point & { scale: number }
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
type TrailFoot = PetTrailFoot
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

// The weather where the visitor is, from the worker's /api/weather (cached there and in the browser
// for 15 minutes). How cold: the wind chill, as Environment Canada works it out for 10 °C and under
// with 5 km/h of wind or more; 10 °C and under is 'cool', 0 °C and under 'cold'. Rain: drizzle, rain
// or a thunderstorm by its weather code, and snow. A lantern in fog, at dusk and at night; sunglasses in the heat; a hand on its hat in the wind; the umbrella blown inside out in a storm (pet.scss).
let chill: '' | 'cool' | 'cold' = ''
let rain = false
let snow = false
let lantern = false
let heat = false
let windy = false
let storm = false
let beach = false
// How tanned it is, 0 to 1: it deepens while it lies on its lounger, full after tanMs of sun, and
// lasts as long as the page is open.
const tanMs = 120_000
let tan = 0
let tanFrom: number | null = null
let weatherLoad: Promise<void> | null = null
const loadWeather = () =>
  (weatherLoad ??= fetch('/api/weather')
    .then(res => (res.ok ? res.json() : null))
    .then(body => {
      const forecast = readVisitorForecast(body)
      const w = forecast?.weather
      if (!w) return
      const v = w.wind ** 0.16
      const feels =
        w.temp <= 10 && w.wind >= 4.8
          ? 13.12 + 0.6215 * w.temp - 11.37 * v + 0.3965 * w.temp * v
          : w.temp
      chill = feels <= 0 ? 'cold' : feels <= 10 ? 'cool' : ''
      const { fall, fog, storm: thunder } = visitorPrecipitation(w.code)
      rain = wetFall(fall)
      snow = fall === 'snow'
      // Dusk is the 45 minutes before sunset; after it, night until sunrise.
      const now = Date.now()
      const sun = nextSunEvent(forecast.solar, now)
      const dark = sun ? sun.kind === 'sunrise' || sun.at - now < 45 * 60_000 : w.day === false
      lantern = fog || dark
      // 27 °C and over is heat; a clear or partly cloudy one by day is a day for the beach.
      heat = w.temp >= 27
      // Wind that takes hats: a fresh breeze (30 km/h) or gusts of 50.
      windy = w.wind >= 30 || (w.gust ?? 0) >= 50
      // A storm: a thunderstorm, or rain in that wind.
      storm = rain && (thunder || windy)
      beach = heat && !lantern && fall === 'none' && w.code <= 2
      render()
    })
    .catch(() => {}))

// The storm's gust flips the umbrella on the wall clock the lightning uses, so on the home page the
// flash and the flip land together. A CSS animation restarts each time its part shows again, so the
// clock is set on every animation, not once as a delay.
const syncStorm = (el: Element) => {
  const at = Date.now() % (STORM_PERIOD * 1000)
  for (const a of el.getAnimations({ subtree: true }))
    if (a instanceof CSSAnimation && a.animationName.startsWith('pet-storm')) {
      const t = Number(a.currentTime ?? 0) % (STORM_PERIOD * 1000)
      if (Math.abs(t - at) > 120) a.currentTime = at
    }
}

const render = () => {
  if (!root || !svg || (!doc && !carry)) return
  const next = state(performance.now())
  if (svg.dataset.state !== next) svg.dataset.state = next
  svg.dataset.kind = perch?.kind ?? 'stand'
  const now = performance.now()
  const lying =
    beach && next === 'work' && (svg.dataset.kind === 'stand' || svg.dataset.kind === 'floor')
  if (lying && tanFrom === null) tanFrom = now
  if (!lying && tanFrom !== null) {
    tan = Math.min(1, tan + (now - tanFrom) / tanMs)
    tanFrom = null
  }
  svg.style.setProperty('--pet-tan', tan.toFixed(3))
  svg.style.setProperty('--pet-tan-left', `${Math.round((1 - tan) * tanMs)}ms`)
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
  if (svg.dataset.cold !== chill) svg.dataset.cold = chill
  svg.toggleAttribute('data-rain', rain)
  svg.toggleAttribute('data-snow', snow)
  svg.toggleAttribute('data-lantern', lantern)
  svg.toggleAttribute('data-heat', heat)
  svg.toggleAttribute('data-windy', windy)
  svg.toggleAttribute('data-storm', storm)
  if (storm) {
    syncStorm(svg)
    if (trailArt) syncStorm(trailArt)
  }
  svg.toggleAttribute('data-beach', beach)
  svg.toggleAttribute('data-reading', document.documentElement.getAttribute('reader-mode') === 'on')
  // On the rail it faces the page, its back to the pole.
  root.style.setProperty('--pet-face', String(carry ? 1 : face))
  syncTrailArt()
  if (speaking?.playing || chat || reminding) {
    // On the trail the bubble sits on the pixel pet's head (reserveTrailFoot), not the hidden vector's.
    const top =
      trailHeadY === null ? center().y - height(perch?.scale) / 2 : trailHeadY - window.scrollY
    root.dataset.bubble = top < 64 ? 'below' : 'above'
    placeBubble()
  }
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
    const matrix = element.getScreenCTM()
    const parent = element.parentElement
    const ground = parent instanceof SVGGraphicsElement ? parent.getScreenCTM() : null
    const radius = Number(element.dataset.solidRadius)
    const id = element.dataset.campSolid
    if (!matrix || !ground || !id || !Number.isFinite(radius) || radius <= 0) continue
    obstacles.push({
      id,
      kind: 'trunk',
      x: matrix.e + window.scrollX,
      y: matrix.f + window.scrollY,
      radius: radius * Math.hypot(ground.a, ground.b),
    })
  }
  return obstacles
}
const petObstacles = () => ({
  text: visibleTextBoxes(document, root),
  panels: [
    ...Array.from(document.querySelectorAll<HTMLElement>('.homepage-board'), panel =>
      panel.getBoundingClientRect(),
    ),
    ...[trailClockRegion(document)].flatMap(region => (region ? [region] : [])),
  ],
  camps: campSolids(),
})
type PetObstacles = ReturnType<typeof petObstacles>
type PlacementMeasurements = { obstacles?: PetObstacles }

// A snapshot lasts only for the synchronous search, so scrolling and DOM changes get fresh boxes.
const coversText = (edge: Edge, obstacles = petObstacles()) => {
  const scale = edge.scale ?? 1
  const left = edge.at.x - (width * scale) / 2 - 2
  const right = edge.at.x + (width * scale) / 2 + 2
  const top = edge.at.y - (edgeY[edge.kind] - viewTop) * unit * scale - 2
  const bottom = top + height(scale) + 4
  const intersects = (rect: { left: number; right: number; top: number; bottom: number }) =>
    rect.left < right && rect.right > left && rect.top < bottom && rect.bottom > top
  const foot = { x: edge.at.x + window.scrollX, y: edge.at.y + window.scrollY }
  return (
    obstacles.panels.some(intersects) ||
    obstacles.camps.some(other => circleClearance(foot, width * 0.35 * scale, other) < 0) ||
    obstacles.text.some(intersects)
  )
}

// Probes run with the pet hidden, so its own drawing never counts as the page's.
function unseen<T>(probe: () => T): T {
  // Probes nest (a landing asks where the friend would go), so each restores what it found.
  const before = root?.style.visibility ?? ''
  if (root) root.style.visibility = 'hidden'
  try {
    return probe()
  } finally {
    if (root) root.style.visibility = before
  }
}

// The floor, too, keeps off text: a client point where the pet could stand on its own line.
const floorStep = 16
const floorY = () => window.innerHeight - margin - (viewBottom - edgeY.floor) * unit
const floorClear = (x: number, y = floorY(), obstacles = petObstacles()) =>
  !coversText({ el: null, kind: 'floor', t: 0, at: { x, y }, score: 0 }, obstacles)

// Scan out both ways, then move up when a line of text blocks the whole floor.
const floorNear = (
  x: number,
  fits?: (p: Perch) => boolean,
  seen: PlacementMeasurements = {},
): Perch | null =>
  unseen<Perch | null>(() => {
    const obstacles = (seen.obstacles ??= petObstacles())
    const lo = margin + width / 2
    const hi = window.innerWidth - margin - width / 2
    const start = clamp(x, lo, hi)
    const bottom = floorY()
    const top = margin + (edgeY.floor - viewTop) * unit
    for (let y = bottom; y >= top; y -= floorStep)
      for (let k = 0; k * floorStep <= hi - lo; k++)
        for (const c of k ? [start + k * floorStep, start - k * floorStep] : [start]) {
          if (c < lo || c > hi || !floorClear(c, y, obstacles)) continue
          const p: Perch = {
            el: null,
            kind: 'floor',
            t: c / window.innerWidth,
            foot: y === bottom ? undefined : { x: c + window.scrollX, y: y + window.scrollY },
          }
          if (!fits || fits(p)) return p
        }
    return null
  })

// How far along the floor from `from` toward `to` the pet can walk with text clear underfoot.
const floorReach = (from: number, to: number) =>
  unseen(() => {
    const obstacles = petObstacles()
    const dir = Math.sign(to - from)
    let x = from
    while (Math.abs(to - x) > floorStep && floorClear(x + dir * floorStep, floorY(), obstacles))
      x += dir * floorStep
    return Math.abs(to - x) <= floorStep && floorClear(to, floorY(), obstacles) ? to : x
  })

// The first edge that is actually in view at the point the pet would hold, moves with the page,
// and leaves the text around it clear.
function pickEdge(
  anchor: Point,
  avoid: Element | null,
  fits?: (p: Perch) => boolean,
  seen: PlacementMeasurements = {},
): Edge | null {
  return unseen(() => {
    const obstacles = (seen.obstacles ??= petObstacles())
    for (const edge of edgesNear(anchor, avoid).slice(0, 20)) {
      const probe = edge.kind === 'hang' ? edge.at.y - 3 : edge.at.y + 3
      const hit = document.elementsFromPoint(edge.at.x, probe).find(e => !root?.contains(e))
      if (edge.kind === 'stand' && (!hit || !(edge.el === hit || edge.el!.contains(hit)))) continue
      if (!inFlow(edge.el!) || coversText(edge, obstacles)) continue
      if (fits && !fits(edge)) continue
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
  // A sunny hot day makes it lazy: it mostly stays put, so it gets to lie out on its lounger.
  if (beach && Math.random() < 0.75) return
  settled = now
  armQuietTimer()
  if (trailWalk) {
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
type PetTerrain = {
  top: number
  base: number
  treeline: number
  meadowTop: number
  pond: Pond | null
}
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
  const pond = 'pond' in value ? value.pond : null
  return {
    top: alpine.top,
    base: alpine.base,
    treeline: value.treeline,
    meadowTop: value.meadowTop,
    pond:
      pond &&
      typeof pond === 'object' &&
      'x' in pond &&
      'y' in pond &&
      'rx' in pond &&
      'ry' in pond &&
      finite(pond.x) &&
      finite(pond.y) &&
      finite(pond.rx) &&
      finite(pond.ry)
        ? { x: pond.x, y: pond.y, rx: pond.rx, ry: pond.ry }
        : null,
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
        return from[1] + (to[1] - from[1]) * fraction
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
const trailKey = (path: SVGPathElement) => `expedition:${path.dataset.petExpedition ?? ''}`
const petTrail = (element: Element | null): element is SVGPathElement =>
  element instanceof SVGPathElement && element.hasAttribute('data-pet-expedition')
const currentTrailPath = (path: SVGPathElement) =>
  path.isConnected && !!path.getAttribute('d') && trailScene(document)?.contains(path) === true

type TrailSample = { point: DOMPoint; progress: number }
const trailPoints = new WeakMap<SVGPathElement, { d: string; step: number; pts: TrailSample[] }>()
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
  return Array.from(scene.querySelectorAll<SVGPathElement>('[data-pet-expedition]')).flatMap(
    path => {
      const d = path.getAttribute('d')
      const m = trailScreenMatrix(path)
      if (!d || !m) return []
      const step = 4 / Math.max(Math.hypot(m.a, m.b), Math.hypot(m.c, m.d), 0.001)
      let hit = trailPoints.get(path)
      if (hit?.d !== d || hit.step !== step) {
        const geometry = svgPathGeometry(path)
        const length = geometry.getTotalLength()
        const segments = Math.max(1, Math.ceil(length / step))
        const pts = Array.from({ length: segments + 1 }, (_, k): TrailSample => {
          const point = geometry.getPointAtLength((k * length) / segments)
          return { point: new DOMPoint(point.x, point.y), progress: k / segments }
        })
        hit = { d, step, pts }
        trailPoints.set(path, hit)
      }
      const stops = trailScaleStops(path)
      const terrain = petTerrain(path)
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
      const feet = points.map(({ point: p, progress }): TrailFoot => {
        const c = p.matrixTransform(m)
        let blend = 0
        if (pair(span) && span[1] > span[0] && progress >= span[0]) {
          const edge = Math.min(0.015, (span[1] - span[0]) / 4)
          blend = easeTrailDepth((span[1] + edge - progress) / edge)
        }
        return {
          x: c.x + window.scrollX + dx * blend,
          y: c.y + window.scrollY + dy * blend,
          progress,
          scale: trailScale(progress, p, stops, terrain),
        }
      })
      return [{ path, feet, key: trailKey(path) }]
    },
  )
}

const trailGeometry = (path: SVGPathElement) => {
  if (!currentTrailPath(path)) return ''
  const m = trailScreenMatrix(path)
  if (!m) return ''
  return `${path.getAttribute('d')}|${path.dataset.petScaleStops ?? ''}|${path.dataset.petAlpineSpan ?? ''}|${m.a},${m.b},${m.c},${m.d},${m.e + window.scrollX},${m.f + window.scrollY}|${window.innerWidth},${window.innerHeight}`
}

const trailPointClear = (p: TrailFoot, path: SVGPathElement, obstacles: PetObstacles) => {
  const at = { x: p.x - window.scrollX, y: p.y - window.scrollY }
  const top = at.y - (edgeY.stand - viewTop) * unit * p.scale
  const right = window.innerWidth - margin - (width * p.scale) / 2
  return (
    at.x >= margin + (width * p.scale) / 2 &&
    at.x <= right &&
    top >= margin &&
    top + height(p.scale) <= window.innerHeight - margin &&
    !coversText({ el: path, kind: 'stand', t: 0, at, score: 0, scale: p.scale }, obstacles)
  )
}

type TrailRun = { path: SVGPathElement; feet: TrailFoot[]; length: number }
const trailActors = (scene: SVGSVGElement | null): CollisionActor[] => {
  const actors: CollisionActor[] = clockObstacles(document, {
    x: window.scrollX,
    y: window.scrollY,
  })
  const drawing = scene?.closest<HTMLElement>('.trail-scene') ?? scene
  // These bodies use the current screen matrix; the guard checks the live camera, so their
  // radii need no reservation for future camera positions in the scene's coordinate space.
  for (const other of drawing ? readTrunks(drawing, undefined, false) : []) {
    const element = other.element
    const parent = element?.parentElement
    const cached = element?.ownerSVGElement === scene
    const matrix = element && (cached ? trailScreenMatrix(element) : element.getScreenCTM())
    const ground =
      parent instanceof SVGGraphicsElement
        ? cached
          ? trailScreenMatrix(parent)
          : parent.getScreenCTM()
        : null
    if (!matrix || !ground) continue
    actors.push({
      ...other,
      x: matrix.e + window.scrollX,
      y: matrix.f + window.scrollY,
      radius: other.radius * Math.hypot(ground.a, ground.b),
    })
  }
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
      resting: kind === 'animal' && element.dataset.phase === 'resting',
      element,
    })
  }
  return actors
}
const visibleTrailRun = (obstacles: PetObstacles): TrailRun | null => {
  const text = { ...obstacles, camps: [] }
  for (const { path, feet, key } of trailRoutes()) {
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
        foot.progress < cursor ||
        !trailPointClear(foot, path, text) ||
        (!run.length &&
          actors.some(
            other =>
              circleClearance(foot, TRAIL_PET_RADIUS * trailPx(path) * foot.scale, other) < 0,
          ))
      ) {
        finish()
        continue
      }
      const previous = run[run.length - 1]
      if (previous) length += Math.hypot(foot.x - previous.x, foot.y - previous.y)
      run.push(foot)
    }
    finish()
    if (runs.length) return runs[0]
  }
  return null
}

const planTrailWalk = (
  feet: TrailFoot[],
  path: SVGPathElement,
  bodies: readonly CollisionActor[],
  obstacles: PetObstacles,
) => {
  const search = planTrailWalkSteps(feet, path, bodies, obstacles)
  let step = search.next()
  while (!step.done) step = search.next()
  return step.value
}
function* planTrailWalkSteps(
  feet: TrailFoot[],
  path: SVGPathElement,
  bodies: readonly CollisionActor[],
  obstacles: PetObstacles,
): PetTrailSearch {
  const scale = Math.max(...feet.map(foot => foot.scale))
  const bounds: SpacingBounds = {
    left: window.scrollX + margin + (width * scale) / 2,
    right: window.scrollX + window.innerWidth - margin - (width * scale) / 2,
    top: window.scrollY + margin + (edgeY.stand - viewTop) * unit * scale,
    bottom:
      window.scrollY + window.innerHeight - margin - (viewBottom - edgeY.stand) * unit * scale,
  }
  const planned = yield* detourPetTrailSteps(feet, TRAIL_PET_RADIUS * trailPx(path), bodies, bounds)
  if (!planned) return null
  for (const foot of planned) {
    yield
    if (!trailPointClear(foot, path, obstacles)) return null
  }
  return planned
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
  const fraction =
    (elapsed - current.offsets[index - 1]) /
    Math.max(0.000001, current.offsets[index] - current.offsets[index - 1])
  return {
    x: from.x + (to.x - from.x) * fraction,
    y: from.y + (to.y - from.y) * fraction,
    scale: from.scale + (to.scale - from.scale) * fraction,
    progress: from.progress + (to.progress - from.progress) * fraction,
  }
}
let trailGuardTimer = 0
let trailPaintFrame = 0
let trailPlanFrame = 0
let trailActor: SVGGElement | null = null
let trailArt: SVGGElement | null = null
// The top of the trail pet's drawing in document pixels, while it is drawn there.
let trailHeadY: number | null = null
// The trail pet's prints go in the visitors' track layer, so moss under them is trampled and snow
// takes their colour. The last print's place is in that layer's units.
let trailTracks: SVGGElement | null = null
let trailPrint: { x: number; y: number; side: number } | null = null
let trailYieldEpisode = 0
// Screen pixels per scene unit of a trail's drawing: the trail pet is sized in scene units like the
// folk there, and the walk planner works in screen pixels.
const trailPx = (path: SVGPathElement) => {
  const scene = path.ownerSVGElement
  const m = scene ? trailScreenMatrix(scene) : null
  return m ? Math.hypot(m.a, m.b) : 1
}
const syncTrailArt = () => {
  if (!trailArt || !svg) return
  for (const name of trailArt.getAttributeNames())
    if (name.startsWith('data-') && !svg.hasAttribute(name)) trailArt.removeAttribute(name)
  for (const attribute of svg.attributes)
    if (
      attribute.name.startsWith('data-') &&
      trailArt.getAttribute(attribute.name) !== attribute.value
    )
      trailArt.setAttribute(attribute.name, attribute.value)
  for (const property of ['--pet-lx', '--pet-ly', '--pet-tilt']) {
    const value = root?.style.getPropertyValue(property) ?? ''
    if (trailArt.style.getPropertyValue(property) !== value)
      trailArt.style.setProperty(property, value)
  }
}
const stopTrailGuard = () => {
  window.clearTimeout(trailGuardTimer)
  trailGuardTimer = 0
  cancelAnimationFrame(trailPaintFrame)
  trailPaintFrame = 0
  cancelAnimationFrame(trailPlanFrame)
  trailPlanFrame = 0
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
  trailArt = null
  trailHeadY = null
  trailPrint = null
  root?.removeAttribute('data-trail-paint')
  for (const property of ['--pet-trail-top', '--pet-trail-foot', '--pet-trail-unscale'])
    root?.style.removeProperty(property)
}
const reserveTrailFoot = (path: SVGPathElement, foot: TrailFoot) => {
  if (!currentTrailPath(path)) return clearTrailActor()
  const scene = path.ownerSVGElement
  const matrix = scene ? trailScreenMatrix(scene) : null
  if (!scene || !matrix || !svg || !root) return
  const drawing = scene.closest<HTMLElement>('.trail-scene')
  const span = trailData(path.dataset.petAlpineSpan)
  const alpine = pair(span) && foot.progress <= span[1]
  let changed = false
  if (trailActor?.ownerSVGElement !== scene || !trailActor.isConnected) {
    trailActor?.remove()
    trailActor = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    trailActor.dataset.trailCollision = 'pet'
    trailActor.dataset.collisionId = 'pet-blob'
    trailActor.dataset.trailPet = ''
    trailActor.setAttribute('aria-hidden', 'true')
    trailArt = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    trailArt.setAttribute('class', 'trail-pet-art')
    trailArt.innerHTML = trailPetSprite()
    trailActor.append(trailArt)
    scene.append(trailActor)
    root.dataset.trailPaint = 'scene'
    syncTrailArt()
    changed = true
  }
  if (trailActor.dataset.petAlpine !== String(alpine)) {
    trailActor.dataset.petAlpine = String(alpine)
    changed = true
  }
  if (changed) drawing?.dispatchEvent(new Event('trail:depth-change'))
  const parent = trailActor.parentElement
  const ground = parent instanceof SVGGraphicsElement ? trailScreenMatrix(parent) : matrix
  if (!ground) return
  const point = new DOMPoint(foot.x - window.scrollX, foot.y - window.scrollY).matrixTransform(
    ground.inverse(),
  )
  // Scene units at the ground scale, as the folk on the trail are drawn, carried into the parent
  // the depth order put it in.
  const scale = (foot.scale * Math.hypot(matrix.a, matrix.b)) / Math.hypot(ground.a, ground.b)
  const transform = `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)})`
  if (trailActor.getAttribute('transform') !== transform)
    trailActor.setAttribute('transform', transform)
  const pose = `scale(${(scale * face).toFixed(4)} ${scale.toFixed(4)})`
  if (trailArt?.getAttribute('transform') !== pose) trailArt?.setAttribute('transform', pose)
  trailActor.dataset.collisionRadius = String(TRAIL_PET_RADIUS * scale)
  trailActor.dataset.collisionActive = 'true'
  trailActor.dataset.groundX = String(point.x)
  trailActor.dataset.groundY = String(point.y)
  trailActor.dataset.progress = foot.progress.toFixed(6)
  trailActor.dataset.scale = String(foot.scale)
  // The speech bubble, in the root's own pixels: the root is scaled by the foot's ground scale with
  // the hidden vector pet's feet on the trail, so the pixel pet's head is its height in scene units
  // above them. The bubble undoes the scale to read at the page's size.
  const head = trailPetTop(rain) * Math.hypot(matrix.a, matrix.b)
  const feet = (edgeY.stand - viewTop) * unit
  const gap = 4 / foot.scale
  trailHeadY = foot.y - head * foot.scale
  const bubble: [string, string][] = [
    ['--pet-trail-top', `${((viewBottom - edgeY.stand) * unit + head + gap).toFixed(1)}px`],
    ['--pet-trail-foot', `${(feet + gap).toFixed(1)}px`],
    ['--pet-trail-unscale', (1 / foot.scale).toFixed(4)],
  ]
  for (const [property, value] of bubble)
    if (root.style.getPropertyValue(property) !== value) root.style.setProperty(property, value)
  leaveTrailPrint(scene, foot)
}
// One blob print a stride along the walk, on alternating sides; muddy in the rain. A jump longer
// than a few strides (a hop on, a new perch) starts the line again without a print.
const leaveTrailPrint = (scene: SVGSVGElement, foot: TrailFoot) => {
  const layer = scene.querySelector<SVGGElement>('.trail-visitor-tracks')
  const plane = layer ? trailScreenMatrix(layer) : null
  const matrix = trailScreenMatrix(scene)
  if (!layer || !plane || !matrix) return
  if (trailTracks?.parentNode !== layer) {
    trailTracks = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    trailTracks.setAttribute('class', 'trail-pet-tracks')
    layer.append(trailTracks)
    trailPrint = null
  }
  const at = new DOMPoint(foot.x - window.scrollX, foot.y - window.scrollY).matrixTransform(
    plane.inverse(),
  )
  const size = (foot.scale * Math.hypot(matrix.a, matrix.b)) / Math.hypot(plane.a, plane.b)
  const stride = 9 * size
  if (!trailPrint) {
    trailPrint = { x: at.x, y: at.y, side: 1 }
    return
  }
  const step = Math.hypot(at.x - trailPrint.x, at.y - trailPrint.y)
  if (step > stride * 4) {
    trailPrint = { x: at.x, y: at.y, side: trailPrint.side }
    return
  }
  if (step < stride) return
  const angle = Math.atan2(at.y - trailPrint.y, at.x - trailPrint.x)
  const mark = impression(document, 'blob')
  mark.dataset.footprint = 'blob'
  mark.dataset.footprintOwner = 'pet'
  if (rain) mark.dataset.wet = ''
  mark.setAttribute(
    'transform',
    `${footprintTransform(at.x, at.y, angle, trailPrint.side, 3 * size)} scale(${(0.9 * size).toFixed(4)})`,
  )
  trailTracks.append(mark)
  while (trailTracks.childElementCount > 48) trailTracks.firstElementChild?.remove()
  trailPrint = { x: at.x, y: at.y, side: -trailPrint.side }
}
const paintTrail = (animation: Animation, path: SVGPathElement) => {
  const paint = () => {
    trailPaintFrame = 0
    if (hop !== animation || !trailWalk || !root || document.hidden) return
    if (!currentTrailPath(path) || trailHost(path)?.hidden !== false) return pauseTrail()
    const time = Number(animation.currentTime)
    const foot = trailFootAt(trailWalk, time)
    const ahead = trailFootAt(trailWalk, time + 120)
    const dx = ahead.x - foot.x
    if (Math.abs(dx) > 0.25 * foot.scale) {
      face = dx > 0 ? -1 : 1
      root.style.setProperty('--pet-face', String(face))
    }
    reserveTrailFoot(path, foot)
    trailPaintFrame = requestAnimationFrame(paint)
  }
  paint()
}
const guardTrail = (animation: Animation, path: SVGPathElement) => {
  stopTrailGuard()
  let blockedSince = 0
  let lastPlan = -Infinity
  let waited: { id: string; since: number } | null = null
  let passing: string | null = null
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
    const current = trailWalk
    const time = Number(animation.currentTime)
    const foot = trailFootAt(current, time)
    const next = trailFootAt(current, time + 150)
    const actors = trailActors(path.ownerSVGElement)
    const body = TRAIL_PET_RADIUS * trailPx(path)
    const radius = body * Math.max(foot.scale, next.scale)
    const now = performance.now()
    // As with the trail's walkers, an animal that holds the way past yieldPatience is passed
    // like a trunk. A grazing animal would otherwise keep the pet waiting indefinitely.
    if (
      passing &&
      !actors.some(
        other => other.id === passing && animalInWay(foot, next, radius, [other], passing),
      )
    )
      passing = null
    if (!passing && waited && now - waited.since > yieldPatience) passing = waited.id
    let animal = animalInWay(
      foot,
      next,
      body * foot.scale,
      actors.filter(other => other.id !== passing),
      root.dataset.yieldingTo,
    )
    const blocker = actors.find(
      other =>
        (other.kind !== 'animal' || other.id === passing) && !canMove(foot, next, radius, [other]),
    )
    if (!animal && blocker?.kind === 'traveler') {
      const waitingFor = queuedAnimal('pet-blob', blocker, actors)
      animal = actors.find(other => other.id === waitingFor && other.id !== passing)
    }
    if (!animal) waited = null
    else if (waited?.id !== animal.id) waited = { id: animal.id, since: now }
    if (animal && root.dataset.yieldingTo !== animal.id) trailYieldEpisode++
    const blocked = !!animal || !!blocker
    if (blocked) blockedSince ||= now
    else blockedSince = 0
    reserveTrailFoot(path, foot)
    // Keep the current foot when replacing the remaining native animation. Fixed obstructions
    // replan at once; a passing person gets time to clear before choosing another road.
    const fixed = blocker?.kind === 'trunk' || (!!passing && blocker?.id === passing)
    if (
      blocked &&
      !trailPlanFrame &&
      !animal &&
      now - lastPlan >= 750 &&
      (fixed || now - blockedSince >= 500)
    ) {
      lastPlan = now
      const remaining = current.feet.filter(
        (_, index) => current.offsets[index] * current.duration > time,
      )
      const bodies = actors.flatMap(other =>
        other.kind !== 'animal'
          ? [other]
          : other.id === passing
            ? [{ ...other, radius: other.radius + radius * 1.5 + 14 }]
            : [],
      )
      const search = planTrailWalkSteps([foot, ...remaining], path, bodies, petObstacles())
      const advance = () => {
        trailPlanFrame = 0
        if (
          !root ||
          hop !== animation ||
          animation.playState !== 'paused' ||
          Math.abs(Number(animation.currentTime) - time) > 1
        )
          return
        const deadline = performance.now() + 3
        let result = search.next()
        while (!result.done && performance.now() < deadline) result = search.next()
        if (!result.done) {
          trailPlanFrame = requestAnimationFrame(advance)
          return
        }
        const planned = result.value
        if (planned && planned.length > 1) {
          root.dataset.trailDetours = String(Number(root.dataset.trailDetours ?? 0) + 1)
          follow(planned, path)
        }
      }
      trailPlanFrame = requestAnimationFrame(advance)
    }
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

function hopArc(
  from: Point,
  to: Point,
  fromScale: number,
  toScale: number,
): [HopPoint, HopPoint, HopPoint] {
  const distance = Math.hypot(to.x - from.x, to.y - from.y)
  const peak = Math.min(
    180,
    50 + distance * 0.3,
    Math.max(0, Math.min(from.y, to.y) - window.scrollY),
  )
  return [
    { ...from, scale: fromScale },
    { x: (from.x + to.x) / 2, y: Math.min(from.y, to.y) - peak, scale: (fromScale + toScale) / 2 },
    { ...to, scale: toScale },
  ]
}

function trailHopClear(next: Perch, obstacles: PetObstacles) {
  if (!doc || doc.y < window.scrollY || doc.y > window.scrollY + window.innerHeight) return false
  const target = placeOn(next)
  if (Math.hypot(target.x - doc.x, target.y - doc.y) > 900) return false
  const arc = hopArc(doc, target, perch?.scale ?? 1, next.scale ?? 1)
  for (let segment = 1; segment < arc.length; segment++) {
    const from = arc[segment - 1]
    const to = arc[segment]
    const count = Math.max(1, Math.ceil(Math.hypot(to.x - from.x, to.y - from.y) / 6))
    for (let index = 0; index <= count; index++) {
      const fraction = index / count
      const scale = from.scale + (to.scale - from.scale) * fraction
      const at = {
        x: from.x + (to.x - from.x) * fraction + (width * scale) / 2 - window.scrollX,
        y:
          from.y +
          (to.y - from.y) * fraction +
          (edgeY.stand - viewTop) * unit * scale -
          window.scrollY,
      }
      if (coversText({ ...next, kind: 'stand', scale, at, score: 0 }, obstacles)) return false
    }
  }
  return true
}

// Resume the current alpine descent ahead of its cursor.
// Entering from the page it appears on the route (a fade); moves within the trail hop.
const walkTrail = (obstacles = petObstacles(), arrive: 'hop' | 'fade' = 'hop') => {
  if (!root || hop || carry || reducedMotion() || document.hidden) return false
  const best = visibleTrailRun(obstacles)
  if (!best) return false
  const { path, feet } = best
  const route = planTrailWalk(
    feet,
    path,
    trailActors(path.ownerSVGElement).filter(other => other.kind === 'trunk'),
    obstacles,
  )
  if (!route || route.length < 11) return false
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
    const next: Perch = {
      el: path,
      kind: 'stand',
      t: 0,
      foot: start,
      scale: start.scale,
      progress: start.progress,
    }
    moveTo(next, arrive === 'fade' || !trailHopClear(next, obstacles) ? 'fade' : null)
  }
  return true
}

// Walks a run of trail points at a stroll, ending on the last of them. A driven step (the arrow
// keys) goes at its own pace and hands back to the driver instead of the wander timer.
type Drive = { speed: number; done: () => void }
function follow(feet: TrailFoot[], el: SVGPathElement, drive?: Drive) {
  if (!root) return
  stopTrailGuard()
  hop?.cancel()
  hop = null
  const to = feet[feet.length - 1]
  perch = { el, kind: 'stand', t: 0, foot: to, scale: to.scale, progress: to.progress }
  doc = placeOn(perch)
  observePerch()
  let duration = 0
  const times = [0]
  for (let k = 1; k < feet.length; k++) {
    const previous = feet[k - 1]
    const point = feet[k]
    const scale = (point.scale + previous.scale) / 2
    duration +=
      (Math.hypot(point.x - previous.x, point.y - previous.y) / ((drive?.speed ?? 55) * scale)) *
      1000
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
      offset: offsets[index],
    })),
    { duration, easing: 'linear' },
  )
  hop = anim
  guardTrail(anim, el)
  if (hop === anim) paintTrail(anim, el)
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
    if (drive) drive.done()
    else scheduleWander()
  }
}

// Arrow keys walk the pet about the trail: a press hops it onto the walking path if it is not on
// it, then each step goes a short way in the keys' direction, at the folk's ground scale for that
// depth, round trunks, tents, animals and the pond, and stops short of the page's text. The keys
// are taken while the trail is open and nothing else holds the focus, so the page does not scroll.
const driveKeys = new Set<string>()
const arrowKeys: Record<string, Point> = {
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
}
const driveStride = 18
const driveSpeed = 150
// The step under way, so a held key's repeats leave it to finish; the obstacle scan is shared
// by the steps of one burst (a fresh scan each step made the page stutter).
let driving: Animation | null = null
// A tap that lands while a step is under way still gets its own step after it.
let driveQueued: Point | null = null
let driveScan: { at: number; text: PetObstacles; actors: CollisionActor[] } | null = null
const driveObstacles = (svg: SVGSVGElement) => {
  const now = performance.now()
  if (!driveScan || now - driveScan.at > 400) {
    driveScan = {
      at: now,
      text: { ...petObstacles(), camps: [] },
      actors: [...trailActors(svg), ...campSolids()],
    }
  }
  return driveScan
}
const expeditionPath = () => {
  const scene = trailScene(document)
  if (!scene || trailHost(scene)?.hidden !== false || scene.dataset.trailRendered !== 'true')
    return null
  const path = scene.querySelector<SVGPathElement>('[data-pet-expedition]')
  return path && petTrail(path) && currentTrailPath(path) ? path : null
}
const driveTakesKeys = (event: KeyboardEvent) => {
  if (!(event.key in arrowKeys) || event.altKey || event.ctrlKey || event.metaKey) return false
  if (!root || carry || reducedMotion() || !expeditionPath()) return false
  const active = document.activeElement
  return (
    !active ||
    active === document.body ||
    !!active.closest('.garden-pet, .homepage-trail-toggle, .homepage-trail-scene')
  )
}
// The walking path's progress nearest a spot, so the wander resumes the expedition from there.
const nearestTrailProgress = (path: SVGPathElement, point: Point) => {
  let best = perch?.progress ?? 0
  let near = Infinity
  for (const route of trailRoutes()) {
    if (route.path !== path) continue
    for (const foot of route.feet) {
      const d = Math.hypot(foot.x - point.x, foot.y - point.y)
      if (d < near) {
        near = d
        best = foot.progress
      }
    }
  }
  return best
}
// Hops onto the nearest clear spot of the walking path in view, down in the valley. A stroll or
// hop under way is cut where the pet is, so the hop starts from where it stands.
// Puts the pet on the nearest clear foot of the route and hands over once it stands there.
// Stops the move under way where it is, so the next move starts from there.
const dropHop = () => {
  if (!root || !hop) return
  const rect = root.getBoundingClientRect()
  hop.cancel()
  hop = null
  override = null
  afterMove = null
  doc = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
  root.style.transform = transformFor(doc, rect.width / width)
}
const landOnTrail = (path: SVGPathElement, then: () => void) => {
  if (!root) return false
  dropHop()
  const obstacles = petObstacles()
  const text = { ...obstacles, camps: [] }
  const actors = [...trailActors(path.ownerSVGElement), ...campSolids()]
  const here = center()
  const from = { x: here.x + window.scrollX, y: here.y + window.scrollY }
  let start: TrailFoot | null = null
  let near = Infinity
  for (const route of trailRoutes()) {
    if (route.path !== path) continue
    for (const foot of route.feet) {
      if (foot.scale < 0.35 || !trailPointClear(foot, path, text)) continue
      const radius = TRAIL_PET_RADIUS * trailPx(path) * foot.scale
      if (actors.some(other => circleClearance(foot, radius, other) < 0)) continue
      const d = Math.hypot(foot.x - from.x, foot.y - from.y)
      if (d < near) {
        near = d
        start = foot
      }
    }
  }
  if (!start) return false
  const foot = start
  afterMove = () => {
    trailWalk = { path, geometry: trailGeometry(path), feet: [foot], duration: 0, offsets: [0] }
    trailJourney = { key: trailKey(path), progress: foot.progress }
    reserveTrailFoot(path, foot)
    settled = performance.now()
    render()
    then()
  }
  const next: Perch = {
    el: path,
    kind: 'stand',
    t: 0,
    foot,
    scale: foot.scale,
    progress: foot.progress,
  }
  moveTo(next, trailHopClear(next, obstacles) ? null : 'fade')
  return true
}
const boardTrail = (path: SVGPathElement) => landOnTrail(path, driveStep)
// The scene has just opened: the pet appears on the route at once and sets off along it, or, with
// no walkable run in view, lands on the nearest clear foot and leaves the rest to the wander.
const enterTrail = (obstacles: PetObstacles) => {
  dropHop()
  if (walkTrail(obstacles, 'fade')) return true
  const path = expeditionPath()
  if (!path) return false
  return landOnTrail(path, () => {
    armQuietTimer()
    scheduleWander()
  })
}
const driveStep = () => {
  if (!root || carry || reducedMotion() || document.hidden) return
  const path = expeditionPath()
  if (!path) return
  let dx = 0
  let dy = 0
  for (const key of driveKeys) {
    dx += arrowKeys[key].x
    dy += arrowKeys[key].y
  }
  if (!dx && !dy && driveQueued) {
    dx = driveQueued.x
    dy = driveQueued.y
  }
  driveQueued = null
  if (!dx && !dy) {
    driveScan = null
    if (trailWalk && perch?.foot && !hop) {
      // Back on the expedition's clock from the nearest point of its path.
      const progress = nearestTrailProgress(path, perch.foot)
      perch.progress = progress
      trailJourney = { key: trailKey(path), progress }
      reserveTrailFoot(path, { ...perch.foot, scale: perch.scale ?? 1, progress })
      settled = performance.now()
      armQuietTimer()
      scheduleWander()
      savePosition()
    }
    return
  }
  if (hop && hop === driving) {
    driveQueued = { x: dx, y: dy }
    return
  }
  // A key mid-stroll takes the walk over from where the pet stands.
  if (hop && trailWalk?.duration) pauseTrail()
  if (hop) return
  if (!trailWalk || !perch?.foot) return boardTrail(path)
  const svg = path.ownerSVGElement
  const matrix = svg && trailScreenMatrix(svg)
  const terrain = petTerrain(path)
  if (!svg || !matrix || !terrain) return
  const px = Math.hypot(matrix.a, matrix.b)
  const from: TrailFoot = { ...perch.foot, scale: perch.scale ?? 1, progress: perch.progress ?? 0 }
  const length = Math.hypot(dx, dy)
  const { text, actors } = driveObstacles(svg)
  // The scene's own box, in scene units: the drawing may bleed past it for parallax.
  const drawing = path.closest<HTMLElement>('.trail-scene')
  const width = drawing?.clientWidth || svg.viewBox.baseVal.width
  const height = drawing?.clientHeight || svg.viewBox.baseVal.height
  const inverse = matrix.inverse()
  // The walk keeps below the treeline; where the route itself put the pet higher (its alpine
  // stretch) it may go level or down from there.
  const here = new DOMPoint(from.x - window.scrollX, from.y - window.scrollY).matrixTransform(
    inverse,
  )
  const top = Math.min(terrain.treeline + 6, here.y)
  // The full direction first, then along one axis only, so it slides along what blocks it.
  const tries = [
    { x: dx / length, y: dy / length },
    ...(dx && dy
      ? [
          { x: Math.sign(dx), y: 0 },
          { x: 0, y: Math.sign(dy) },
        ]
      : []),
  ]
  for (const dir of tries) {
    const stride = driveStride * from.scale * px
    const to = { x: from.x + dir.x * stride, y: from.y + dir.y * stride }
    const ground = new DOMPoint(to.x - window.scrollX, to.y - window.scrollY).matrixTransform(
      inverse,
    )
    if (ground.x < 14 || ground.x > width - 14) continue
    if (ground.y < top || ground.y > height - 12) continue
    if (terrain.pond && pondDistance(ground, terrain.pond) < 1.15) continue
    const scale = trailScale(0, ground, [], terrain)
    const foot: TrailFoot = { ...to, scale, progress: from.progress }
    if (!trailPointClear(foot, path, text)) continue
    const radius = TRAIL_PET_RADIUS * px * Math.max(from.scale, scale)
    if (!canMove(from, foot, radius, actors)) continue
    lastMove = performance.now()
    follow([from, foot], path, { speed: driveSpeed, done: driveStep })
    driving = hop
    return
  }
  // Blocked all round: it still turns to face the way it was asked.
  if (dx) {
    face = dx > 0 ? -1 : 1
    root.style.setProperty('--pet-face', String(face))
    syncTrailArt()
  }
}
const onDriveKey = (event: KeyboardEvent) => {
  if (!driveTakesKeys(event)) return
  event.preventDefault()
  driveKeys.add(event.key)
  driveStep()
}
const onDriveKeyUp = (event: KeyboardEvent) => {
  if (!driveKeys.delete(event.key)) return
  if (!driveKeys.size && !hop) driveStep()
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
  else clearTrailActor()
  root.hidden = false
  stopGround()
  gaze = ''
  const target = placeOn(next)
  const from = doc
  const fromScale = perch?.scale ?? 1
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
const slidesPerch = (seen: PlacementMeasurements): Perch | 'covered' | null => {
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
  return unseen(() => coversText({ ...p, at, score: 0 }, (seen.obstacles ??= petObstacles())))
    ? 'covered'
    : p
}

// The page changed under a perched pet: if its edge went away or text now sits under it, it finds
// another edge.
function recheckPerch() {
  if (!root || !perch || !doc || hop || carry || petTrail(perch.el)) return
  const scale = perch.scale ?? 1
  const at = {
    x: doc.x - window.scrollX + (width * scale) / 2,
    y: doc.y - window.scrollY + (edgeY[perch.kind] - viewTop) * unit * scale,
  }
  const gone = !!perch.el && !perch.el.getClientRects().length
  if (gone || unseen(() => coversText({ ...perch!, at, score: 0 })))
    relocate(center(), gone ? null : perch.el)
}

function relocate(anchor: Point, avoid: Element | null, how: 'chute' | null = null) {
  if (!root) return
  // On the friendship page it lands where the friend fits too, beside it, on its head or along its
  // edge; only when no perch near has that room does it take one the friend has to sit out.
  const before = friend?.style.visibility ?? ''
  if (friend) friend.style.visibility = 'hidden'
  let next: Perch | null
  try {
    next = unseen(() => {
      // No page writes occur until a perch is chosen. Every fallback shares this one snapshot.
      const seen: PlacementMeasurements = {}
      const slides = slidesPerch(seen)
      const fits =
        friend && friendPage() ? (p: Perch) => !!friendSpot(p, placeOn(p), false, seen) : undefined
      return slides === 'covered'
        ? null
        : (slides ??
            (fits && (pickEdge(anchor, avoid, fits, seen) ?? floorNear(anchor.x, fits, seen))) ??
            pickEdge(anchor, avoid, undefined, seen) ??
            floorNear(anchor.x, undefined, seen))
    })
  } finally {
    if (friend) friend.style.visibility = before
  }
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
  const bottom = trailClockRegion(document)?.top ?? view
  const span = Math.max(document.documentElement.scrollHeight - view, 1)
  const thumb = Math.max((view * view) / (span + view), 32)
  const hands = 1.2 * unit
  return {
    x: window.innerWidth - 6 - 3.12 * unit,
    from: Math.max(thumb / 2 - hands, margin),
    to: Math.min(view - thumb / 2 - hands, bottom - height() - margin),
    span,
  }
}

// The page started scrolling: let go of the edge and leap onto the rail.
function startCarry() {
  if (!root || !doc) return
  rememberTrail()
  clearTrailActor()
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
  syncTrailArt()
  if (trailWalk && perch?.foot && !hop)
    reserveTrailFoot(trailWalk.path, {
      ...perch.foot,
      scale: perch.scale ?? 1,
      progress: perch.progress ?? 0,
    })
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
    recheckPerch()
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
  if (friendSvg.dataset.cold !== chill) friendSvg.dataset.cold = chill
  friendSvg.toggleAttribute('data-rain', rain)
  friendSvg.toggleAttribute('data-snow', snow)
  friendSvg.toggleAttribute('data-lantern', lantern)
  friendSvg.toggleAttribute('data-heat', heat)
  friendSvg.toggleAttribute('data-storm', storm)
  if (storm) syncStorm(friendSvg)
  friendSvg.toggleAttribute('data-beach', beach)
  friendSvg.toggleAttribute(
    'data-reading',
    document.documentElement.getAttribute('reader-mode') === 'on',
  )
  friend.style.setProperty('--pet-face', String(friendFace))
  if (friend.classList.contains('is-speaking')) {
    const c = friendCenter()
    if (c) friend.dataset.bubble = c.y - height(friendScale()) / 2 < 64 ? 'below' : 'above'
  }
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
// the same edge, shoulder to shoulder, on its usual side if that side has room and no text. With
// `far` off it stays on the pet's own edge or head, which is what a landing asks for. A search over
// many perches shares `seen`, so the text is read once, with both blobs hidden.
function friendSpot(
  p: Perch,
  m: Point,
  far = true,
  seen: PlacementMeasurements = {},
): { perch: Perch; doc: Point } | null {
  const covered = (edge: Edge) => coversText(edge, (seen.obstacles ??= petObstacles()))
  const s = p.scale ?? 1
  const fs = s * friendSize
  const mainX = m.x - window.scrollX + (width * s) / 2
  const gap = 0.38 * width * (s + fs)
  const beside = (side: number, off: number) => {
    const x = mainX + side * (gap + off)
    // placeOn holds a box or floor spot inside the view but takes a foot as given, so a foot past
    // the view's side is caught here.
    const half = (width * fs) / 2
    if (x < margin + half - 2 || x > window.innerWidth - margin - half + 2) return null
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
    if (covered({ ...perch, at: { x: cx, y: foot }, score: 0 })) return null
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
    if (top < margin || covered({ ...perch, at: head, score: 0 })) return null
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
          if (covered({ ...perch, at: { x, y: edge.at.y }, score: 0 })) continue
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
    return far ? nearby() : null
  }
  const before = friend?.style.visibility ?? ''
  if (friend) friend.style.visibility = 'hidden'
  try {
    return unseen(probe)
  } finally {
    if (friend) friend.style.visibility = before
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
  renderFriend()
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
  document.body.appendChild(el)
  friend = el
  friendSvg = null
  return () => {
    el.innerHTML = `<button type="button" class="garden-pet-button" aria-label="the pet's friend">${friendMarkup}</button><div class="garden-pet-bubble" role="status"><span class="garden-pet-bubble-text"></span></div>`
    friendSvg = el.querySelector('svg')
    renderFriend()
    requestAnimationFrame(() => el.classList.add('is-visible'))
    scheduleMeet()
  }
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
  document.body.appendChild(el)
  root = el
  svg = null
  // The pet is absolutely positioned in document coordinates, so the window's scroll carries it
  // along its edge for free. The containing block may sit off the document origin; measure it once.
  const rect = el.getBoundingClientRect()
  width = rect.width || 72
  unit = width / viewWidth
  origin = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
  // Placement reads the page before the SVG's thousands of nodes need style and layout.
  return () => {
    el.innerHTML = `<button type="button" class="garden-pet-button" aria-label="garden pet">${markup}</button><div class="garden-pet-bubble" role="status"><span class="garden-pet-bubble-text"></span><button type="button" class="garden-pet-close" aria-label="dismiss reminder">×</button></div>`
    svg = el.querySelector('svg')
    showBubble()
    requestAnimationFrame(() => el.classList.add('is-visible'))
  }
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
  let placed = false
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
    const mount = spawn()
    const mountFriend = spawnFriend()
    relocate(before, null)
    placed = true
    mount()
    mountFriend?.()
  } else if (perch?.el && !perch.el.isConnected) {
    relocate(center(), null)
    placed = true
  }
  if (friendPage() && !friend) {
    const mountFriend = spawnFriend()
    if (perch && doc && !hop) friendFollow(perch, doc, 'appear')
    mountFriend?.()
  }
  if (lastMove === 0) lastMove = performance.now()
  armQuietTimer()
  scheduleWander()
  scheduleReminder()
  return placed
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
  const placed = syncPets()
  document.addEventListener('keydown', onDriveKey, { signal })
  document.addEventListener('keyup', onDriveKeyUp, { signal })
  window.addEventListener('blur', () => driveKeys.clear(), { signal })
  void loadWeather()
  const scene = document.querySelector<HTMLElement>('.homepage-trail-scene')
  let trailFrame = 0
  let trailWanted = false
  let waiting: Animation | null = null
  const refreshTrail = () => {
    trailFrame = 0
    if (signal.aborted || !scene || scene.hidden || document.hidden || reducedMotion()) return
    const drawing = trailScene(document)
    if (drawing?.dataset.trailOpening !== 'complete' || drawing.dataset.trailReveal !== undefined)
      return
    if (scene.dataset.trailResizing === 'true') {
      trailWanted ||= !!trailWalk
      if (trailWalk?.duration) pauseTrail()
      return
    }
    // The frame replaces its routes over several frames of a redraw and reports rendered only at
    // the end. A walk whose route has moved or gone stops at once, before the rendered check, so
    // the pet does not keep hopping along a path that is no longer there; it resumes on the new
    // route once the scene is ready.
    const obstacles = petObstacles()
    const text = { ...obstacles, camps: [] }
    const currentTrail = trailWalk
    if (
      currentTrail &&
      (!currentTrailPath(currentTrail.path) ||
        currentTrail.geometry !== trailGeometry(currentTrail.path) ||
        currentTrail.feet.some(foot => !trailPointClear(foot, currentTrail.path, text)))
    ) {
      pauseTrail()
      trailWanted = true
    }
    if (trailScene(document)?.dataset.trailRendered !== 'true') return
    if (!trailWanted || carry) return
    // On the trail a move finishes first; from the page the pet comes in at once.
    const current = trailWalk ? hop : null
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
    if (trailWalk ? walkTrail(obstacles) : enterTrail(obstacles)) {
      trailWanted = false
      return
    }
    const resting = trailWalk
    if (
      resting &&
      (!currentTrailPath(resting.path) ||
        !trailPointClear(resting.feet[0], resting.path, obstacles))
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
  const clearClock = () => {
    const region = trailClockRegion(document)
    if (!root || !region) return
    if (carry) {
      hop?.cancel()
      hop = null
      override = null
      landAfterJet = false
      carry = railFor()
      root.style.setProperty('--pet-rail-from', `${carry.from.toFixed(1)}px`)
      root.style.setProperty('--pet-rail-to', `${carry.to.toFixed(1)}px`)
      root.style.transform = `translate3d(${carry.x.toFixed(1)}px, 0, 0)`
      root.style.translate = `0 ${railY(carry).toFixed(1)}px`
      render()
      return
    }
    const box = root.getBoundingClientRect()
    if (box.right <= region.left || box.bottom <= region.top || box.top >= region.bottom) return
    trailWanted ||= !!trailWalk
    rememberTrail()
    const next = floorNear(region.left - width) ?? pickEdge(center(), null)
    if (next) moveTo(next, 'fade')
    else root.hidden = true
  }
  // Fresh page placement already checked the clock in its obstacle snapshot. Slides have their
  // own perch policy; retained pets still need the live check after the page changes.
  if (!placed || document.body.dataset.slides === 'true') clearClock()
  if (scene) {
    const openTrail = () => {
      if (scene.hidden) return
      trailWanted = true
      queueTrail()
    }
    scene.addEventListener('trail:show', openTrail, { signal })
    scene.addEventListener('trail:close', pausePageTrail, { signal })
    scene.addEventListener('trail:hide', closeTrail, { signal })
    scene.addEventListener('trail:dispose', closeTrail, { signal })
    scene.addEventListener('trail:reflow', queueTrail, { signal })
    scene.addEventListener('trail:ready', openTrail, { signal })
    scene.addEventListener('trail:revealed', openTrail, { signal })
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
      const expedition = trail.querySelector('[data-pet-expedition]')
      if (expedition)
        paths.observe(expedition, {
          attributes: true,
          attributeFilter: ['d', 'data-pet-scale-stops', 'data-pet-alpine-span'],
        })
    }
    scene.addEventListener('trail:mounted', watchPaths, { signal })
    scene.addEventListener('trail:ready', watchPaths, { signal })
    watchPaths()
    signal.addEventListener('abort', () => paths?.disconnect(), { once: true })
    const controlsChanged = () => {
      if (root && !carry) {
        const pet = root.getBoundingClientRect()
        const covered = [...document.querySelectorAll<HTMLElement>('.homepage-board')].some(
          panel => {
            const box = panel.getBoundingClientRect()
            return (
              box.left < pet.right &&
              box.right > pet.left &&
              box.top < pet.bottom &&
              box.bottom > pet.top
            )
          },
        )
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
        clearClock()
        scrollTimer = window.setTimeout(onSettle, 220)
        return
      }
      if (trailWalk || trailJourney) trailWanted = true
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
      clearClock()
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
    reflowTimer = window.setTimeout(recheckPerch, 300)
  }
  document.addEventListener('readermodechange', onReaderMode)
  // Content that lands after the pet has perched (a course page's hydrated rows, the lecture
  // workspace bar) or text that reflows once the web fonts load can end up under it. For the page's
  // first seconds, the perch is re-checked once the page goes quiet; the pet's own nodes don't count.
  let lateTimer: number | undefined
  // A pet still hopping onto its first edge is checked once it lands.
  const recheckSoon = (ms = 400) => {
    clearTimeout(lateTimer)
    lateTimer = window.setTimeout(() => (hop ? recheckSoon(200) : recheckPerch()), ms)
  }
  const late = new MutationObserver(records => {
    const outside = records.some(r => {
      const el = r.target instanceof Element ? r.target : r.target.parentElement
      return !el?.closest('.garden-pet, .pet-ground, .garden-pet-heart')
    })
    if (outside) recheckSoon()
  })
  late.observe(document.body, { childList: true, subtree: true })
  const lateStop = window.setTimeout(() => late.disconnect(), 5000)
  void document.fonts?.ready.then(() => recheckSoon())
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
    late.disconnect()
    clearTimeout(lateStop)
    clearTimeout(lateTimer)
    document.removeEventListener('slidechange', onSlide)
    document.removeEventListener('contentdecrypted', onSlide)
    clearTimeout(slideTimer)
    speaking = null
    windows.disconnect()
    window.clearTimeout(windowTimer)
    lookAt = null
  })
})
