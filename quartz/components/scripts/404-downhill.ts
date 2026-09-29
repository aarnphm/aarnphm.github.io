// Downhill on the left-hand hill: a bike-park line cut down the face under the crest, a drag lift up
// its left side, and a start hut on the crest where the two people by the cypress stand watching.
// The line is built as a surface to ride: the ground, plus what the builders put on it (three
// berms, a wooden drop with a chicken line round it, and a tabletop). Who rides it, and how, is
// 404-riders.ts; this file owns the shapes, their physics tables and their paint.
//
// The face is seen from level, so a unit down the screen is a unit of height, and it falls at
// thirty degrees, so each of those units also runs √3 units back into the view. Grades, speeds and
// flights are worked on that ground, and only the drawing goes back onto the screen.

import { type Grid, type Pt, type RGB, col, dot, row, smooth } from './404-pixel'

// People here stand four pixels, some eighteen units, to their metre seventy, so a metre is ten
// units.
export const UNIT = 10
export const G = 9.81 * UNIT
const DEPTH = Math.sqrt(3)
// Samples along the line, in screen units.
const STEP = 2

// The line through its turns, in scene units, from the start hut on the crest to the finish above
// the island's flank. The corners are rounded into berms before anything is measured.
const LINE: Pt[] = [
  [318, 484],
  [352, 524],
  [338, 560],
  [262, 598],
  [236, 634],
  [352, 716],
  [292, 750],
  [240, 764],
  [205, 790],
]
// The chicken line: it leaves the main line after the first berm, runs below the drop, and comes
// back in above the second berm.
const FORK: Pt = [326, 566]
const DETOUR: Pt[] = [
  [314, 580],
  [294, 592],
]
const JOIN: Pt = [266, 598]
// The drop's deck from where it leaves the ground to its lip, and the foot of the tabletop's
// takeoff ramp.
const DROP: [Pt, Pt] = [
  [316, 570],
  [296, 581],
]
const TABLE: Pt = [258, 650]
// The drop stands this proud of the ground at its lip; the table's deck this proud all along it.
const DROP_H = 9
const TABLE_H = 11
// The table's takeoff ramp, deck and landing, over the ground.
const RAMP = 24
const DECK = 18
const LAND = 34
// Turns sharper than this, over the ground, were given a berm.
const BERMED: [number, number] = [0.006, 0.014]

// The drag lift, bottom station to top, on the ground; the path from its top along the crest to
// the start hut, where the queue for the gate stands; the run-out from the finish to the lift; and
// the way in from the left edge, which is how everyone arrives and leaves.
export const LIFT: [Pt, Pt] = [
  [174, 808],
  [226, 466],
]
const LINK: Pt[] = [LIFT[1], [258, 473], [298, 481], LINE[0]]
const RUNOUT: Pt[] = [LINE[LINE.length - 1], [190, 800], LIFT[0]]
const BASE: Pt[] = [[36, 815], [110, 813], LIFT[0]]
// Where along the lift its two towers stand.
export const LIFT_TOWERS = [0.36, 0.7]
// Where the photographer crouches, beside the table's landing.
export const PHOTO: Pt = [300, 700]

export type Track = {
  n: number
  x: Float32Array
  y: Float32Array
  // Distance from the gate over the ground as seen from above, the height of the surface that is
  // ridden (whatever is built included), its grade dz/dq arriving at each sample, and how far what
  // is built stands proud of the ground.
  q: Float32Array
  z: Float32Array
  grade: Float32Array
  built: Float32Array
  // Curvature over the ground, signed, and how much of a berm the turn was given, 0..1.
  bend: Float32Array
  bank: Float32Array
  // Where each sample lies along the main line, so riders on either line keep their distance.
  main: Float32Array
  // Samples where the surface runs out from under a rider: the drop's lip, the table's.
  lips: number[]
  length: number
}

function chaikin(p: Pt[], rounds: number): Pt[] {
  for (let n = 0; n < rounds; n++) {
    const q: Pt[] = [p[0]]
    for (let k = 0; k < p.length - 1; k++) {
      const [ax, ay] = p[k]
      const [bx, by] = p[k + 1]
      q.push([ax * 0.75 + bx * 0.25, ay * 0.75 + by * 0.25])
      q.push([ax * 0.25 + bx * 0.75, ay * 0.25 + by * 0.75])
    }
    q.push(p[p.length - 1])
    p = q
  }
  return p
}

function resample(p: Pt[], step: number): Pt[] {
  const out: Pt[] = [p[0]]
  let need = step
  for (let k = 1; k < p.length; k++) {
    let [ax, ay] = p[k - 1]
    const [bx, by] = p[k]
    let len = Math.hypot(bx - ax, by - ay)
    while (len >= need) {
      const u = need / len
      ax += (bx - ax) * u
      ay += (by - ay) * u
      out.push([ax, ay])
      len -= need
      need = step
    }
    need -= len
  }
  return out
}

const nearest = (pts: Pt[], [x, y]: Pt) => {
  let best = 0
  for (let k = 1; k < pts.length; k++)
    if (Math.hypot(pts[k][0] - x, pts[k][1] - y) < Math.hypot(pts[best][0] - x, pts[best][1] - y))
      best = k
  return best
}

// Over the ground, as seen from above, and along it.
const plan = (dx: number, dy: number) => Math.hypot(dx, dy * DEPTH)
export const slant = (dx: number, dy: number) => Math.hypot(dx, dy * 2)

function survey(pts: Pt[], built: Float32Array): Track {
  const n = pts.length
  const x = Float32Array.from(pts, p => p[0])
  const y = Float32Array.from(pts, p => p[1])
  const q = new Float32Array(n)
  for (let k = 1; k < n; k++) q[k] = q[k - 1] + plan(x[k] - x[k - 1], y[k] - y[k - 1])
  const heading = (k: number) => {
    const a = Math.max(0, k - 1)
    const b = Math.min(n - 1, k + 1)
    return Math.atan2((y[b] - y[a]) * DEPTH, x[b] - x[a])
  }
  const bend = new Float32Array(n)
  const bank = new Float32Array(n)
  for (let k = 3; k < n - 3; k++) {
    const turn = heading(k + 3) - heading(k - 3)
    bend[k] = Math.atan2(Math.sin(turn), Math.cos(turn)) / (q[k + 3] - q[k - 3])
    bank[k] = smooth(BERMED[0], BERMED[1], Math.abs(bend[k]))
  }
  const z = Float32Array.from(y, (v, k) => -v + built[k])
  const grade = new Float32Array(n)
  for (let k = 1; k < n; k++) grade[k] = (z[k] - z[k - 1]) / (q[k] - q[k - 1])
  grade[0] = grade[1]
  return { n, x, y, q, z, grade, built, bend, bank, main: q, lips: [], length: q[n - 1] }
}

const MAIN_PTS = resample(chaikin(LINE, 3), STEP)
const KD0 = nearest(MAIN_PTS, DROP[0])
const KLIP = nearest(MAIN_PTS, DROP[1])
const KF = nearest(MAIN_PTS, FORK)
const KJ = nearest(MAIN_PTS, JOIN)
const KT = nearest(MAIN_PTS, TABLE)

// What the builders raised on the main line: the drop's deck climbs off the ground and levels out
// toward its lip, then stops dead; the table rises on a ramp that steepens into its lip, runs level
// with the ground along its deck, and comes down a landing that is steepest in its middle.
const MAIN_BUILT = (() => {
  const n = MAIN_PTS.length
  const b = new Float32Array(n)
  const q = new Float32Array(n)
  for (let k = 1; k < n; k++)
    q[k] = q[k - 1] + plan(MAIN_PTS[k][0] - MAIN_PTS[k - 1][0], MAIN_PTS[k][1] - MAIN_PTS[k - 1][1])
  for (let k = KD0; k <= KLIP; k++) b[k] = DROP_H * Math.sqrt((q[k] - q[KD0]) / (q[KLIP] - q[KD0]))
  const q0 = q[KT]
  for (let k = KT; k < n; k++) {
    const u = q[k] - q0
    if (u < RAMP) b[k] = TABLE_H * (u / RAMP) ** 1.5
    else if (u < RAMP + DECK) b[k] = TABLE_H
    else if (u < RAMP + DECK + LAND) b[k] = TABLE_H * (1 - smooth(0, 1, (u - RAMP - DECK) / LAND))
    else break
  }
  return b
})()

export const MAIN: Track = survey(MAIN_PTS, MAIN_BUILT)
export const KTL = (() => {
  let k = KT
  while (MAIN.q[k + 1] - MAIN.q[KT] <= RAMP) k++
  return k
})()
MAIN.lips = [KLIP, KTL]

// The chicken line shares the main line's samples up to the fork and after the join, so the table
// and the berms below are the same ground either way.
export const CHICKEN: Track = (() => {
  const detour = resample(chaikin([MAIN_PTS[KF], ...DETOUR, MAIN_PTS[KJ]], 3), STEP)
  const pts = [...MAIN_PTS.slice(0, KF), ...detour, ...MAIN_PTS.slice(KJ + 1)]
  const built = new Float32Array(pts.length)
  const off = KF + detour.length - (KJ + 1)
  for (let k = KJ + 1; k < MAIN_PTS.length; k++) built[k + off] = MAIN_BUILT[k]
  const t = survey(pts, built)
  // The detour's length is spread over the main line's between the fork and the join.
  const main = new Float32Array(t.n)
  const a = t.q[KF]
  const b = t.q[KF + detour.length - 1]
  for (let k = 0; k < t.n; k++)
    main[k] =
      k < KF
        ? t.q[k]
        : k < KF + detour.length
          ? MAIN.q[KF] + ((t.q[k] - a) / (b - a)) * (MAIN.q[KJ] - MAIN.q[KF])
          : t.q[k] - b + MAIN.q[KJ]
  t.main = main
  t.lips = [KTL + off]
  return t
})()
export const TRACKS = [MAIN, CHICKEN] as const

// Height of the ridden surface, and the sample at or before `q`, starting the search from `k`.
export function seek(t: Track, q: number, k: number) {
  while (k < t.n - 1 && t.q[k + 1] <= q) k++
  while (k > 0 && t.q[k] > q) k--
  return k
}
export function surface(t: Track, q: number, k: number) {
  if (k >= t.n - 1) return t.z[t.n - 1]
  const u = Math.max(0, Math.min(1, (q - t.q[k]) / (t.q[k + 1] - t.q[k])))
  // The drop's deck ends in a wall at its lip, so past the lip there is only the ground below it.
  const z0 = t.built[k + 1] < t.built[k] - 3 ? -t.y[k] : t.z[k]
  return z0 + (t.z[k + 1] - z0) * u
}

// A flight from a lip at a given speed along the lip's own grade: where over the ground it comes
// down, and after how long.
export function fly(t: Track, lip: number, v: number, pop = 0) {
  const a = Math.atan(t.grade[lip])
  let vh = v * Math.cos(a)
  let vz = v * Math.sin(a) + pop
  let q = t.q[lip]
  let z = t.z[lip]
  let k = lip
  let time = 0
  const dt = 1 / 240
  while (time < 3) {
    q += vh * dt
    vz -= G * dt
    z += vz * dt
    time += dt
    k = seek(t, q, k)
    if (k >= t.n - 1 || z <= surface(t, q, k)) break
  }
  return { q, time, vh, vz, k }
}

// The table's sweet spot, a third of the way down its landing, and the speed off the lip that
// finds it, by bisection: flights lengthen with speed.
export const SWEET = MAIN.q[KT] + RAMP + DECK + LAND * 0.4
export const TABLE_FLIGHT = (() => {
  let lo = 20
  let hi = 200
  for (let n = 0; n < 40; n++) {
    const mid = (lo + hi) / 2
    if (fly(MAIN, KTL, mid).q < SWEET) lo = mid
    else hi = mid
  }
  // How much further the flight goes for each unit of speed off the lip, around that speed: the
  // deck sits level with the lip and the landing drops away under it, so this grows much faster
  // than the square of the speed would say.
  const reach = (fly(MAIN, KTL, lo * 1.05).q - fly(MAIN, KTL, lo * 0.95).q) / (lo * 0.1)
  return { v: lo, time: fly(MAIN, KTL, lo).time, reach }
})()
// Off the drop, what speed at the lip comes down where, and how fast along the ground below: the
// landing turns some of the fall into speed down the hill, and the second berm is not far below
// it, so riders check their speed on the deck.
export const DROP_OFF = (() => {
  const lip = MAIN.lips[0]
  const out: { v: number; k: number; glide: number }[] = []
  for (let v = 10; v <= 150; v += 5) {
    const r = fly(MAIN, lip, v)
    const gl = MAIN.grade[Math.min(MAIN.n - 1, r.k + 1)]
    out.push({ v, k: r.k, glide: (r.vh + r.vz * gl) / Math.hypot(1, gl) })
  }
  return out
})()
// Where the table's landing starts and ends, over the ground: before it is the deck, after it the
// flat of the hill.
export const LANDING: [number, number] = [MAIN.q[KT] + RAMP + DECK, MAIN.q[KT] + RAMP + DECK + LAND]

// A path walked or ridden at a steady pace: the points, and their distance along the slope.
export type Path = { pts: Pt[]; at: Float32Array; length: number }
function path(p: Pt[]): Path {
  const pts = resample(p, STEP)
  const at = new Float32Array(pts.length)
  for (let k = 1; k < pts.length; k++)
    at[k] = at[k - 1] + slant(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])
  return { pts, at, length: at[pts.length - 1] }
}
export const PATHS = { lift: path(LIFT), link: path(LINK), runout: path(RUNOUT), base: path(BASE) }
// The point `s` along a path, and which way the path heads there.
export function along(p: Path, s: number): { x: number; y: number; dir: number } {
  const s1 = Math.max(0, Math.min(p.length, s))
  let k = 0
  while (k < p.pts.length - 2 && p.at[k + 1] < s1) k++
  const [ax, ay] = p.pts[k]
  const [bx, by] = p.pts[k + 1]
  const u = (s1 - p.at[k]) / Math.max(1e-6, p.at[k + 1] - p.at[k])
  return { x: ax + (bx - ax) * u, y: ay + (by - ay) * u, dir: Math.sign(bx - ax) || 1 }
}

// Whether an outcrop at this spot, this wide, would sit on the line, the chicken line, the lift or
// the paths.
export function nearCourse(x: number, y: number, clear: number) {
  for (const t of TRACKS)
    for (let k = 0; k < t.n; k += 2) if (Math.hypot(t.x[k] - x, t.y[k] - y) < clear) return true
  return Object.values(PATHS).some(p =>
    p.pts.some(([px, py]) => Math.hypot(px - x, py - y) < clear),
  )
}

export type CourseInk = {
  tread: RGB
  rut: RGB
  berm: RGB
  packed: RGB
  path: RGB
  wood: RGB
  post: RGB
  steel: RGB
  flag: [RGB, RGB]
}

// The line, into the ranges' base: worn dirt a pixel or two wide with a darker pixel under it where
// it cuts the slope, packed snow where the snow lies, a built-up lip round the outside of each
// berm, the table's mound with its lit deck, the drop's deck on its post, the chicken line trodden
// into the grass, the start hut, the finish banner, and the drag lift's track, cable and towers.
export function paintCourse(d: Uint8ClampedArray, g: Grid, ink: CourseInk, snowline: number) {
  const trodden = (p: Pt[]) =>
    p.forEach(([px, py], k) => {
      if (k % 2 === 0) dot(d, g, col(g, px), row(g, py), ink.path)
    })
  trodden(PATHS.link.pts)
  trodden(PATHS.runout.pts)
  trodden(PATHS.base.pts)
  PATHS.lift.pts.forEach(([px, py]) => dot(d, g, col(g, px), row(g, py), ink.path))
  const detour = CHICKEN.n - MAIN.n + (KJ - KF)
  for (let k = KF; k <= KF + detour; k++)
    dot(d, g, col(g, CHICKEN.x[k]), row(g, CHICKEN.y[k]), ink.path)

  const { x, y, bend, bank, built } = MAIN
  for (let k = 0; k < MAIN.n; k++) dot(d, g, col(g, x[k]), row(g, y[k]) + 1, ink.rut)
  for (let k = 0; k < MAIN.n; k++) {
    const i = col(g, x[k])
    const j = row(g, y[k] - built[k])
    const tread = y[k] < snowline ? ink.packed : ink.tread
    // The table's mound, filled in under its surface; the drop's deck is timber on a post instead.
    if (k >= KT && built[k] > 0)
      for (let jj = j + 1; jj <= row(g, y[k]); jj++) dot(d, g, i, jj, ink.rut)
    if (k >= KD0 && k <= KLIP) {
      dot(d, g, i, j, ink.wood)
      continue
    }
    dot(d, g, i, j, tread)
    // Across the fall line the tread shows a pixel wider.
    const run = Math.abs(x[Math.min(MAIN.n - 1, k + 1)] - x[Math.max(0, k - 1)]) / (2 * STEP)
    if (run > 0.6) dot(d, g, i, j - 1, tread)
    // The berm's lip, built up on the outside of the turn.
    if (bank[k] > 0.6) {
      const a = Math.max(0, k - 1)
      const b = Math.min(MAIN.n - 1, k + 1)
      const len = Math.hypot(x[b] - x[a], y[b] - y[a]) || 1
      const side = Math.sign(bend[k])
      const ox = ((y[b] - y[a]) / len) * side
      const oy = (-(x[b] - x[a]) / len) * side
      dot(d, g, col(g, x[k] + ox * 6), row(g, y[k] + oy * 6), ink.berm)
    }
  }
  const pi = col(g, x[KLIP])
  for (let j = row(g, y[KLIP] - DROP_H); j <= row(g, y[KLIP]); j++) dot(d, g, pi, j, ink.post)

  // The start hut: two posts under a roof, astride the gate. The beacon on its roof is the riders'.
  const si = col(g, x[0])
  const sj = row(g, y[0])
  for (let k = 1; k <= 4; k++) {
    dot(d, g, si - 2, sj - k, ink.post)
    dot(d, g, si + 2, sj - k, ink.post)
  }
  for (let di = -3; di <= 3; di++) dot(d, g, si + di, sj - 5, ink.wood)
  // The finish: a chequered banner over the line on two poles.
  const last = MAIN.n - 1
  const fi = col(g, x[last])
  const fj = row(g, y[last])
  for (let k = 1; k <= 5; k++) {
    dot(d, g, fi - 3, fj - k, ink.post)
    dot(d, g, fi + 3, fj - k, ink.post)
  }
  for (let di = -2; di <= 2; di++) {
    dot(d, g, fi + di, fj - 6, ink.flag[(di + 2) & 1])
    dot(d, g, fi + di, fj - 5, ink.flag[(di + 3) & 1])
  }

  // The lift runs straight up the fall line, so from across the valley its track, its up and down
  // cables and its towers stack into one steel thread: the towers show as crossarms on it, and
  // there is a bullwheel hut at either end.
  for (const [px, py] of PATHS.lift.pts) dot(d, g, col(g, px), row(g, py), ink.steel)
  const [b0, b1] = LIFT
  for (const u of LIFT_TOWERS) {
    const i = col(g, b0[0] + (b1[0] - b0[0]) * u)
    const j = row(g, b0[1] + (b1[1] - b0[1]) * u)
    for (let di = -1; di <= 1; di++) dot(d, g, i + di, j - 1, ink.steel)
  }
  for (const [bx, by] of LIFT) {
    const i = col(g, bx)
    const j = row(g, by)
    for (let k = 1; k <= 2; k++) {
      dot(d, g, i - 2, j - k, ink.post)
      dot(d, g, i + 2, j - k, ink.post)
    }
    for (let di = -2; di <= 2; di++) dot(d, g, i + di, j - 3, ink.wood)
  }
}
