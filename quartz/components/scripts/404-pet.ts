// The garden's pet (quartz/util/mascot.ts) as the 404 scene prints it: a blob in the ruin's
// register, sitting on the keystone just inside its right-hand corner with its legs over the
// front. It hops in from wherever the site pet last stood on screen, watches the pointer, calls out
// when the portal opens, and jumps into the rift with the reader when they take the door. Now and
// then it goes off on its own: off into the distance, up the ridge trail at the back as a speck,
// a look about by the flag on top, and back down and home.
import { GATE, hourAngle, onStone } from './404-gate'
import {
  type Grid,
  type Pt,
  type RGB,
  type Tones,
  col,
  line,
  noise,
  paintBlob,
  row,
  stamp,
} from './404-pixel'

type Mode = 'away' | 'arrive' | 'sit' | 'leave' | 'ridge' | 'home' | 'dive' | 'gone'

// A point `u` of the way up the ridge trail, and which way that stretch runs.
export type Trail = (u: number) => { x: number; y: number; dir: number }

export type Pet = {
  mode: Mode
  // Seconds into the current mode.
  t: number
  from: Pt
  x: number
  y: number
  // Size against the seated pet; it shrinks as the rift takes it.
  scale: number
  // Radians, as it tumbles in.
  spin: number
  // A frame flattened on landing or crouching to jump.
  squash: boolean
  // 1 faces left as drawn, -1 mirrored.
  face: number
  look: Pt
  hot: boolean
  // Seconds of sitting left before it next goes off up the ridge.
  restless: number
}

export type PetInk = { body: Tones; ink: RGB }

// Body radius in scene units, and its parts in body radii as the mascot draws them.
const R = 26
// Pulled in from the mascot's -0.6 and 0.24, so the near eye clears the keyline on the pixel grid.
const EYES = [-0.46, 0.2]
const ARRIVE = 0.9
const CROUCH = 0.12
const DIVE = 0.66
// The excursion, in seconds: off into the distance, up the trail, about on top, down, and back.
const TRIP = { leave: 0.8, up: 24, top: 6, down: 14, home: 1.1 }
const restless = () => 40 + Math.random() * 50
// The seat: on the keystone's outer face, 10° short of its right-hand corner. The body rests with
// the edge through it at 0.94 radii below its centre, as the site pet sits on a ledge.
function seat(): Pt {
  const [x, y] = onStone(0, GATE.r + GATE.ring + 38, hourAngle(0) + (10 * Math.PI) / 180)
  return [x, y - 0.94 * R]
}

export function makePet(reduce: boolean): Pet {
  const [x, y] = seat()
  return {
    mode: reduce ? 'sit' : 'away',
    t: 0,
    from: [x, y],
    x,
    y,
    scale: 1,
    spin: 0,
    squash: false,
    face: 1,
    look: [0, 0],
    hot: false,
    restless: restless(),
  }
}

// Hops onto the seat from a scene point, or drops from the sky above it.
export function arrive(p: Pet, from: Pt | null) {
  const [x, y] = seat()
  p.mode = 'arrive'
  p.t = 0
  p.from = from ?? [x + 30, y - 360]
}

// Into the rift after the reader, from wherever it is; from the ridge it is still a speck.
export function dive(p: Pet) {
  if (p.mode === 'away' || p.mode === 'gone') return
  if (p.mode !== 'ridge') p.scale = 1
  p.mode = 'dive'
  p.t = 0
  p.from = [p.x, p.y]
}

const arc = (p: Pet, from: Pt, to: Pt, u: number, lift: number) => {
  p.x = from[0] + (to[0] - from[0]) * u
  p.y = from[1] + (to[1] - from[1]) * u - lift * 4 * u * (1 - u)
}

export function stepPet(p: Pet, dt: number, pointer: Pt | null, trail: Trail) {
  p.t += dt
  const [sx, sy] = seat()
  const foot = trail(0)
  const away: Pt = [foot.x - 60, foot.y - 140]
  if (p.mode === 'sit' && !p.hot && (p.restless -= dt) < 0) {
    p.mode = 'leave'
    p.t = 0
  }
  if (p.mode === 'leave') {
    // A leap off the back of the keystone toward the trail, shrinking into the distance.
    const u = Math.min(1, p.t / TRIP.leave)
    arc(p, [sx, sy], away, u, 90)
    p.scale = 1 - 0.8 * u
    p.face = -1
    if (u === 1) {
      p.mode = 'ridge'
      p.t = 0
    }
  } else if (p.mode === 'ridge') {
    const { up, top, down } = TRIP
    const s = p.t < up ? p.t / up : p.t < up + top ? 1 : 1 - (p.t - up - top) / down
    const at = trail(Math.max(0, s))
    p.x = at.x
    p.y = at.y
    p.face = p.t < up ? -at.dir : p.t < up + top ? (Math.floor(p.t / 1.3) % 2 ? 1 : -1) : at.dir
    if (p.t > up + top + down) {
      p.mode = 'home'
      p.t = 0
    }
  } else if (p.mode === 'home') {
    const u = Math.min(1, p.t / TRIP.home)
    arc(p, away, [sx, sy], u, 80)
    p.scale = 0.2 + 0.8 * u
    if (u === 1) {
      p.mode = 'sit'
      p.t = 0
      p.restless = restless()
    }
  } else if (p.mode === 'arrive') {
    const u = Math.min(1, p.t / ARRIVE)
    const lift = 60 + 0.2 * Math.hypot(sx - p.from[0], sy - p.from[1])
    p.x = p.from[0] + (sx - p.from[0]) * u
    p.y = p.from[1] + (sy - p.from[1]) * u - lift * 4 * u * (1 - u)
    if (u === 1) {
      p.mode = 'sit'
      p.t = 0
    }
  } else if (p.mode === 'dive') {
    // A crouch, a spring up off the stone, then a tumbling fall into the middle of the rift.
    const u = Math.max(0, Math.min(1, (p.t - CROUCH) / (DIVE - CROUCH)))
    p.x = p.from[0] + (GATE.x - p.from[0]) * u
    p.y = p.from[1] + (GATE.y - p.from[1]) * u * u - 70 * 4 * u * (1 - u)
    p.scale = Math.min(p.scale, 1 - 0.85 * u * u)
    p.spin = u * Math.PI * 2.5
    if (u === 1) p.mode = 'gone'
  } else if (p.mode === 'sit') {
    p.x = sx
    p.y = sy
  }
  p.squash = (p.mode === 'sit' && p.t < 0.2) || (p.mode === 'dive' && p.t < CROUCH)
  if (p.mode !== 'sit' || !pointer || pointer[0] < -1e3) {
    p.look = [0, 0]
    return
  }
  const dx = pointer[0] - p.x
  const dy = pointer[1] - p.y
  if (Math.abs(dx) > 60) p.face = dx > 0 ? -1 : 1
  p.look = [Math.abs(dx) > 30 ? Math.sign(dx) : 0, Math.abs(dy) > 50 ? Math.sign(dy) : 0]
}

// Paints the pet and returns its head pixel, for speech and for being hailed.
export function paintPet(
  d: Uint8ClampedArray,
  g: Grid,
  p: Pet,
  ink: PetInk,
  clock: number,
): Pt | null {
  if (p.mode === 'away' || p.mode === 'gone' || p.mode === 'ridge') return null
  const r = R * p.scale
  const kx = p.squash ? 1.14 : 1
  const ky = p.squash ? 0.84 : 1
  // Squashed, it keeps its seat: the bottom stays where it was.
  const cx = p.x
  const cy = p.y + r * (1 - ky)
  const cos = Math.cos(p.spin)
  const sin = Math.sin(p.spin)
  // A point in the pet's own frame (body radii, facing left as drawn) to buffer pixels.
  const at = (u: number, v: number): Pt => {
    const x = u * p.face * kx * r
    const y = v * ky * r
    return [col(g, cx + x * cos - y * sin), row(g, cy + x * sin + y * cos)]
  }

  // Seated, the legs hang over the front of the stone and swing in turn; in the air they tuck up.
  const seated = p.mode === 'sit'
  const pace = p.hot ? 7 : 2.4
  for (const [k, hip] of [-0.34, 0.3].entries()) {
    const swing = seated ? Math.round(Math.sin(clock * pace + k * Math.PI) * 1.2) : 0
    const [i0, j0] = at(hip, 0.8)
    const [i1, j1] = at(hip, seated ? 1.42 : 1.1)
    line(d, g, i0, j0, i1 + swing, j1, ink.ink)
    if (seated) line(d, g, i1 + swing, j1, i1 + swing - p.face, j1, ink.ink)
  }

  const field = (x: number, y: number) => {
    const u = (x - cx) * cos + (y - cy) * sin
    const v = -(x - cx) * sin + (y - cy) * cos
    return Math.hypot(u / kx, v / ky) - r + (noise(u * 0.09 + 3, v * 0.09, 7) - 0.5) * 3 * p.scale
  }
  const reach = r * 1.3 + 4
  paintBlob(d, g, field, [cx - reach, cy - reach, cx + reach, cy + reach], ink.body)

  // The dash eyes, set left of centre as the mascot's are. Every few seconds they blink for a frame;
  // with the portal open they stretch wide; seated they follow the pointer by a pixel.
  if (p.scale > 0.35) {
    const blink = seated && !p.hot && clock % 4.2 < 0.12
    const top = p.hot ? -0.5 : -0.38
    for (const ex of EYES) {
      const [i0, j0] = at(ex, blink ? -0.2 : top)
      const [, j1] = at(ex, blink ? -0.2 : 0)
      // The outer eye holds still when the look would push it into the keyline.
      const li = Math.abs(ex) > 0.3 && Math.sign(ex * p.face) === p.look[0] ? 0 : p.look[0]
      line(d, g, i0 + li, j0 + p.look[1], i0 + li, j1 + p.look[1], ink.ink)
    }
  }
  const [hi] = at(0, 0)
  const [, hj] = at(0, -1)
  return [hi, hj - 1]
}

// Up on the ridge it is a speck like the folk there: a cream body in an ink keyline, one eye on
// the side it faces, feet that patter as it runs and a hop now and then while it looks about on
// top. Painted into the ranges layer; returns its head pixel.
const SPECK = ['.KKK.', 'KKCCK', 'KCCCK', '.KKK.']

export function paintSpeck(
  d: Uint8ClampedArray,
  g: Grid,
  p: Pet,
  ink: PetInk,
  clock: number,
): Pt | null {
  if (p.mode !== 'ridge') return null
  const i = col(g, p.x)
  const j = row(g, p.y)
  const top = p.t > TRIP.up && p.t < TRIP.up + TRIP.top
  const hop = top && Math.floor(clock * 2.5) % 4 === 0 ? 1 : 0
  const step = Math.floor(clock * 6) & 1
  const j0 = j - SPECK.length - 1 - hop
  stamp(d, g, SPECK, i - 2, j0, p.face < 0, ch => (ch === 'K' ? ink.ink : ink.body.light))
  for (const k of top || step ? [-1, 1] : [0])
    line(d, g, i + k, j - 1 - hop, i + k, j - 1 - hop, ink.ink)
  return [i, j0 - 1]
}
