import type { PetTrailFoot } from '../../visitor-trail/pet-trail'
import { createSvgSpace } from '../../../util/svg-geometry'
import { createPetGaze } from '../pet-gaze'

// The garden's pet: the pen-drawn mascot, living on the page's edges. It perches on the nearest box
// it can find (stands or sits with legs dangling on a top edge, hangs from a bottom edge) and stays
// there, watching the pointer. It hops to a new edge only when the reader's pointer rests far away or
// when it gets bored. While the page scrolls it clings to a pole on the right edge and slides along
// it with the scroll thumb; when scrolling stops it parachutes down, or jetpacks up, to the nearest
// edge. A pointer move costs no layout and every move is one compositor animation. Looks and motion live in components/styles/pet.scss; this
// script picks edges and writes the SVG's data-state and data-kind.

export type State =
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
  | 'thumb'
  | 'slide'
  | 'walk'
  | 'chute'
  | 'think'
  | 'talk'
  | 'jet'
  | 'notice'
export type Kind = 'stand' | 'sit' | 'hang' | 'floor'
export type Point = { x: number; y: number }
export type HopPoint = Point & { scale: number }
// An edge the pet holds: `t` is where its centre sits along the box's width, 0 to 1. The floor kind
// has no box: the pet stands on its own pen line at the bottom of the viewport, the fallback when no
// box nearby has an edge to hold.
// `foot`, in document coordinates, pins a perch that is a point rather than a box edge: a spot on
// the home page's walking trail.
export type Perch = {
  el: Element | null
  kind: Kind
  t: number
  foot?: Point
  scale?: number
  progress?: number
  // How far below the element's edge the held edge sits, in client px.
  drop?: number
}
export type Edge = Perch & { at: Point; score: number }
export type TrailFoot = PetTrailFoot
export type TrailWalk = {
  path: SVGPathElement
  geometry: string
  feet: TrailFoot[]
  offsets: number[]
  duration: number
}

// Where the held edge crosses the pet, in mascot units (the viewBox is 3.6 wide, top at -2).
export const edgeY: Record<Kind, number> = { stand: 1.42, sit: 0.94, hang: -1.45, floor: 1.42 }
export const viewWidth = 3.6
export const viewTop = -2
export const viewBottom = 1.8

export const margin = 8

export const petState = {
  root: null as HTMLElement | null,
  trailSpaceSignal: undefined as AbortSignal | undefined,
  trailSpaces: new WeakMap<SVGSVGElement, ReturnType<typeof createSvgSpace>>(),
  svg: null as SVGSVGElement | null,
  gazeArt: null as ReturnType<typeof createPetGaze> | null,
  markup: '',
  width: 72,
  unit: 0,
  origin: { x: 0, y: 0 } as Point,
  doc: null as Point | null,
  carry: null as Rail | null,
  perch: null as Perch | null,
  hop: null as Animation | null,
  pointer: null as Point | null,
  lastPointer: null as { x: number; y: number; t: number } | null,
  lastMove: 0,
  // What to do once the next move lands, if anything; moveTo takes it for that move alone.
  afterMove: null as (() => void) | null,
  // When the pet last got up and wandered off on its own: it starts its quiet over from there, so a
  // stroll closes the laptop until it settles again.
  settled: 0,
  face: 1,
  hovering: false,
  reminding: false,
  trailHint: false,
  landAfterJet: false,
  // Where the eyes look instead of the pointer: a triathlon window as it opens.
  lookAt: null as Point | null,
  clockLanded: 0,
  // A window closing under the pet: no edge on it or inside it counts while the pet jumps off.
  closing: null as Element | null,
  windowTimer: undefined as number | undefined,
  // The phrase the French voice is reading (speech.inline.ts): thinking while it loads, then saying it.
  speaking: null as { text: string; lang: string; playing: boolean } | null,
  override: null as { state: State; until: number } | null,
  startleReadyAt: 0,
  relocateReadyAt: 0,
  watchFrame: 0,
  gaze: '',
  anchorFrame: 0,
  stateTimer: undefined as number | undefined,
  dwellTimer: undefined as number | undefined,
  scrollTimer: undefined as number | undefined,
  scrollQuietUntil: 0,
  wanderTimer: undefined as number | undefined,
  // What the wander timer does, set where it is defined: wandering moves the pet, and each move
  // rearms the timer when it lands, so the timer cannot call the moves' caller directly.
  wander: (() => {}) as () => void,
  reminderTimer: undefined as number | undefined,
  reminderHideTimer: undefined as number | undefined,
  resizeObserver: null as ResizeObserver | null,
  trailWalk: null as TrailWalk | null,
  trailJourney: null as { key: string; progress: number } | null,
}
petState.unit = petState.width / viewWidth
export const trailScreenMatrix = (element: SVGGraphicsElement) => {
  const canvas = element instanceof SVGSVGElement ? element : element.ownerSVGElement
  if (!canvas || !petState.trailSpaceSignal || petState.trailSpaceSignal.aborted)
    return element.getScreenCTM()
  let space = petState.trailSpaces.get(canvas)
  if (!space) {
    space = createSvgSpace(canvas, petState.trailSpaceSignal)
    petState.trailSpaces.set(canvas, space)
  }
  return space.matrix(element)
}
// While the page scrolls the pet rides the scroll rail: position: fixed at x, its top running from
// `from` to `to` as the page scrolls from 0 to `span`. pet.scss drives that with a scroll timeline.
export type Rail = { x: number; from: number; to: number; span: number }

export const clockEl = () => document.querySelector<HTMLElement>('.trail-local-time')
export const onClock = () => !!petState.perch?.el && petState.perch.el === clockEl()

export const clamp = (v: number, min: number, max: number) => Math.min(Math.max(v, min), max)
export const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
export const readerMode = () => document.documentElement.getAttribute('reader-mode') === 'on'
export const height = (scale = 1) => (viewBottom - viewTop) * petState.unit * scale

// The pet's centre in viewport coordinates, from its cached document position.
export const railY = (rail: Rail) =>
  rail.from + (rail.to - rail.from) * clamp(window.scrollY / rail.span, 0, 1)

export const center = (): Point =>
  petState.carry
    ? { x: petState.carry.x + petState.width / 2, y: railY(petState.carry) + height() / 2 }
    : petState.doc
      ? {
          x: petState.doc.x - window.scrollX + (petState.width * (petState.perch?.scale ?? 1)) / 2,
          y: petState.doc.y - window.scrollY + height(petState.perch?.scale) / 2,
        }
      : { x: window.innerWidth - petState.width, y: window.innerHeight - height() }

const transformWithin = (p: Point, scale: number, left: number, width: number) => {
  // Rail exits and jet control points can extend past the viewport even when the perch fits.
  const right = Math.max(left, left + width - petState.width * scale)
  const x = clamp(p.x, left, right)
  return `translate3d(${(x - petState.origin.x).toFixed(1)}px, ${(p.y - petState.origin.y).toFixed(1)}px, 0) scale(${scale.toFixed(4)})`
}

export const transformFor = (p: Point, scale = petState.perch?.scale ?? 1) =>
  transformWithin(p, scale, window.scrollX, window.innerWidth)

export const capturePetTransform = () => {
  const left = window.scrollX
  const width = window.innerWidth
  return (p: Point, scale = petState.perch?.scale ?? 1) => transformWithin(p, scale, left, width)
}

export const trailState = {
  trailGuardTimer: 0,
  trailPaintFrame: 0,
  trailPlanFrame: 0,
  trailActor: null as SVGGElement | null,
  trailArt: null as SVGGElement | null,
  // The top of the trail pet's drawing in document pixels, while it is drawn there.
  trailHeadY: null as number | null,
  // The trail pet's prints go in the visitors' track layer, so moss under them is trampled and snow
  // takes their colour. The last print's place is in that layer's units.
  trailTracks: null as SVGGElement | null,
  trailPrint: null as { x: number; y: number; side: number } | null,
  trailYieldEpisode: 0,
}

export const scheduleWander = () => {
  window.clearTimeout(petState.wanderTimer)
  petState.wanderTimer = window.setTimeout(
    () => {
      petState.wander()
      scheduleWander()
    },
    12000 + Math.random() * 20000,
  )
}

export const friendState = {
  // The friend, on the friendship page: a second blob, a size smaller, filled sage with a sprout on
  // its head. It keeps to the pet's side on whatever edge the pet holds and hops after it. When both
  // rest, they now and then hug (a heart rises between them), high-five, or trade a line in French.
  // Clicking it makes it cheer, and the pet turns to wave back.
  friend: null as HTMLElement | null,
  friendSvg: null as SVGSVGElement | null,
  friendGaze: null as ReturnType<typeof createPetGaze> | null,
  friendMarkup: '',
  friendPerch: null as Perch | null,
  friendDoc: null as Point | null,
  friendHop: null as Animation | null,
  // Which side of the pet it keeps (screen x), until that side has no room.
  friendSide: 1,
  friendFace: 1,
  friendHovering: false,
  friendOverride: null as { state: State; until: number } | null,
  meetTimer: undefined as number | undefined,
  // The pet's half of a chat with its friend, shown in its own bubble.
  chat: null as string | null,
}
export const friendTimers = new Set<number>()
export const friendSize = 0.82

export const later = (fn: () => void, ms: number) => {
  const id = window.setTimeout(() => {
    friendTimers.delete(id)
    fn()
  }, ms)
  friendTimers.add(id)
}

export const friendScale = () => friendState.friendPerch?.scale ?? friendSize
export const friendCenter = (): Point | null =>
  friendState.friendDoc && {
    x: friendState.friendDoc.x - window.scrollX + (petState.width * friendScale()) / 2,
    y: friendState.friendDoc.y - window.scrollY + height(friendScale()) / 2,
  }
