// Downhill on the left-hand hill: a bike-park line cut down the face under the crest, where the two
// people by the cypress stand watching the start. Four riders take it in turn. They drop in from
// the start hut, carry speed into each berm and brake for it, roll off a wooden drop, fly the
// tabletop (every other lap with a whip), cross under the finish banner, and push their bikes back
// up the walkers' path to go again. As everyone on the ranges, they are a pixel wide.

import type { Voices } from './404-crew'
import { type Grid, type Pt, type RGB, BAYER, col, dot, hash, row } from './404-pixel'

// The line through its turns, in scene units, from the start hut between two pines on the crest to
// the finish above the island's flank. The corners are rounded into berms before anything is
// measured.
const LINE: Pt[] = [
  [318, 484],
  [352, 524],
  [338, 560],
  [262, 598],
  [236, 634],
  [330, 700],
  [300, 736],
  [240, 764],
  [205, 790],
]
// The walkers' path back up, left of the line, so nobody pushes up into a rider.
const PUSH: Pt[] = [
  [205, 790],
  [196, 720],
  [214, 650],
  [228, 580],
  [262, 520],
  [318, 484],
]
// The wooden drop's deck from where it leaves the ground to its lip, and the tabletop's takeoff and
// landing, as points on the line.
const DROP: [Pt, Pt] = [
  [314, 571],
  [296, 581],
]
const TABLE: [Pt, Pt] = [
  [250, 644],
  [312, 687],
]

// People here stand four pixels, some eighteen units, to their metre seventy, so a metre is about
// ten units and gravity a hundred units a second squared. A rider tops out near ten metres a
// second, holds a g and a half round a berm, brakes at 0.8 g, and gains half a g down the fall line.
const G = 100
const VMAX = 100
const GRIP = 150
const BRAKE = 80
const PULL = 50
// The drop's lip stands about a metre off the ground.
const DROP_H = 9
// Samples along the line, in scene units.
const STEP = 2

const RIDERS = 4
const FINISH = 1.6
// Pushing a downhill bike up the hill goes at a metre a second.
const PUSH_SPEED = 10
const WAIT = 2

type Course = {
  x: Float32Array
  y: Float32Array
  // Signed curvature, per unit of length: positive where the heading turns clockwise on screen.
  bend: Float32Array
  // Seconds from the gate, and height off the ground in scene units.
  t: Float32Array
  lift: Float32Array
  time: number
  // The samples at the drop's lip and the tabletop's takeoff and landing.
  lip: number
  deck: number
  takeoff: number
  landing: number
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

const nearest = (xs: Float32Array, ys: Float32Array, [x, y]: Pt) => {
  let best = 0
  for (let k = 1; k < xs.length; k++)
    if (Math.hypot(xs[k] - x, ys[k] - y) < Math.hypot(xs[best] - x, ys[best] - y)) best = k
  return best
}

// Where a rider can go how fast: the grip limit round each berm, then braking back from every
// slower stretch and pulling away from the gate, the way a lap-time simulation builds its speed
// trace. The airtime falls out of it: a jump's flight takes the time the trace gives it, and the
// height is whatever gravity makes of that time.
const COURSE: Course = (() => {
  const pts = resample(chaikin(LINE, 3), STEP)
  const n = pts.length
  const x = Float32Array.from(pts, p => p[0])
  const y = Float32Array.from(pts, p => p[1])
  const heading = (k: number) => {
    const a = Math.max(0, k - 1)
    const b = Math.min(n - 1, k + 1)
    return Math.atan2(y[b] - y[a], x[b] - x[a])
  }
  const bend = new Float32Array(n)
  for (let k = 3; k < n - 3; k++) {
    const turn = heading(k + 3) - heading(k - 3)
    bend[k] = Math.atan2(Math.sin(turn), Math.cos(turn)) / (6 * STEP)
  }
  const v = Float32Array.from(bend, b =>
    Math.min(VMAX, Math.sqrt(GRIP / Math.max(1e-4, Math.abs(b)))),
  )
  v[n - 1] = Math.min(v[n - 1], 30)
  for (let k = n - 2; k >= 0; k--)
    v[k] = Math.min(v[k], Math.sqrt(v[k + 1] ** 2 + 2 * BRAKE * STEP))
  v[0] = 0
  for (let k = 1; k < n; k++) v[k] = Math.min(v[k], Math.sqrt(v[k - 1] ** 2 + 2 * PULL * STEP))
  const t = new Float32Array(n)
  for (let k = 1; k < n; k++) t[k] = t[k - 1] + STEP / Math.max(1, (v[k] + v[k - 1]) / 2)

  const lift = new Float32Array(n)
  const deck = nearest(x, y, DROP[0])
  const lip = nearest(x, y, DROP[1])
  for (let k = deck; k < n; k++) {
    if (k <= lip) lift[k] = (DROP_H * (k - deck)) / Math.max(1, lip - deck)
    else {
      const fall = t[k] - t[lip]
      const h = DROP_H - (G * fall * fall) / 2
      if (h <= 0) break
      lift[k] = h
    }
  }
  const takeoff = nearest(x, y, TABLE[0])
  const landing = nearest(x, y, TABLE[1])
  const flight = t[landing] - t[takeoff]
  for (let k = takeoff; k <= landing; k++) {
    const u = (t[k] - t[takeoff]) / flight
    lift[k] = Math.max(lift[k], ((G * flight * flight) / 2) * u * (1 - u))
  }
  return { x, y, bend, t, lift, time: t[n - 1], lip, deck, takeoff, landing }
})()

const PUSH_PTS = resample(PUSH, STEP)

// Whether an outcrop at this spot, this wide, would sit on the line, its drop or the path up.
export function nearCourse(x: number, y: number, clear: number) {
  const { x: xs, y: ys } = COURSE
  for (let k = 0; k < xs.length; k += 2) if (Math.hypot(xs[k] - x, ys[k] - y) < clear) return true
  return PUSH_PTS.some(([px, py]) => Math.hypot(px - x, py - y) < clear)
}

export type CourseInk = {
  tread: RGB
  rut: RGB
  berm: RGB
  packed: RGB
  path: RGB
  wood: RGB
  post: RGB
  flag: [RGB, RGB]
  skin: RGB
  shorts: RGB
  tyre: RGB
  dust: RGB
  lamp: RGB
  glow: RGB
  kits: { jersey: RGB; helmet: RGB; frame: RGB }[]
}

// The line, into the ranges' base: worn dirt a pixel or two wide with a darker pixel under it where
// it cuts the slope, packed snow where the snow lies, a built-up lip round the outside of each
// berm, the tabletop's mound, the drop's deck on its post, the start hut, the finish banner, and
// the walkers' path trodden into the grass.
export function paintCourse(d: Uint8ClampedArray, g: Grid, ink: CourseInk, snowline: number) {
  const { x, y, bend } = COURSE
  PUSH_PTS.forEach(([px, py], k) => {
    if (k % 2 === 0) dot(d, g, col(g, px), row(g, py), ink.path)
  })
  for (let k = 0; k < x.length; k++) dot(d, g, col(g, x[k]), row(g, y[k]) + 1, ink.rut)
  for (let k = 0; k < x.length; k++) {
    const i = col(g, x[k])
    const j = row(g, y[k])
    const tread = y[k] < snowline ? ink.packed : ink.tread
    dot(d, g, i, j, tread)
    // Across the fall line the tread shows a pixel wider.
    const run = Math.abs(x[Math.min(x.length - 1, k + 1)] - x[Math.max(0, k - 1)]) / (2 * STEP)
    if (run > 0.6) dot(d, g, i, j - 1, tread)
    // The berm's lip, built up on the outside of the turn.
    if (Math.abs(bend[k]) > 0.014) {
      const a = Math.max(0, k - 1)
      const b = Math.min(x.length - 1, k + 1)
      const len = Math.hypot(x[b] - x[a], y[b] - y[a]) || 1
      const side = Math.sign(bend[k])
      const ox = ((y[b] - y[a]) / len) * side
      const oy = (-(x[b] - x[a]) / len) * side
      dot(d, g, col(g, x[k] + ox * 6), row(g, y[k] + oy * 6), ink.berm)
    }
  }
  for (let k = COURSE.takeoff + 2; k <= COURSE.landing - 2; k++) {
    const i = col(g, x[k])
    const j = row(g, y[k])
    dot(d, g, i, j - 1, y[k] < snowline ? ink.packed : ink.tread)
    dot(d, g, i, j, ink.rut)
  }
  for (let k = COURSE.deck; k <= COURSE.lip; k++)
    dot(d, g, col(g, x[k]), row(g, y[k] - COURSE.lift[k]) - 1, ink.wood)
  const pi = col(g, x[COURSE.lip])
  for (let j = row(g, y[COURSE.lip] - DROP_H); j <= row(g, y[COURSE.lip]); j++)
    dot(d, g, pi, j, ink.post)

  // The start hut: two posts under a roof, astride the gate.
  const si = col(g, x[0])
  const sj = row(g, y[0])
  for (let k = 1; k <= 4; k++) {
    dot(d, g, si - 2, sj - k, ink.post)
    dot(d, g, si + 2, sj - k, ink.post)
  }
  for (let di = -3; di <= 3; di++) dot(d, g, si + di, sj - 5, ink.wood)
  dot(d, g, si, sj - 6, ink.flag[0])
  // The finish: a chequered banner over the line on two poles.
  const last = x.length - 1
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
}

// Seconds into a rider's lap at which they are at the top of the tabletop, so a still frame
// (reduced motion) finds the first of them in the air.
const APEX = (COURSE.t[COURSE.takeoff] + COURSE.t[COURSE.landing]) / 2

const along = (pts: Pt[], u: number) => {
  const k = Math.min(pts.length - 1, Math.max(0, Math.floor(u * (pts.length - 1))))
  const next = pts[Math.min(pts.length - 1, k + 1)]
  return { x: pts[k][0], y: pts[k][1], dir: Math.sign(next[0] - pts[k][0]) || 1 }
}

export function paintRiders(
  d: Uint8ClampedArray,
  g: Grid,
  ink: CourseInk,
  clock: number,
  dark: boolean,
  heads: Voices['heads'],
) {
  const { x, y, t, lift, bend, time } = COURSE
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  // A helmet lamp after dark.
  const lamp = (i: number, j: number) => {
    for (let dj = -2; dj <= 2; dj++)
      for (let di = -2; di <= 2; di++) {
        const r = Math.hypot(di, dj)
        if (r > 0) plot(i + di, j + dj, ink.glow, (1 - r / 2.8) * 0.6)
      }
    dot(d, g, i, j, ink.lamp)
  }
  const pushTime = (PUSH_PTS.length * STEP) / PUSH_SPEED
  const period = time + FINISH + pushTime + WAIT
  const beat = Math.floor(clock * 8)
  for (let n = 0; n < RIDERS; n++) {
    const kit = ink.kits[n % ink.kits.length]
    let s = (((clock + APEX - (n * period) / RIDERS) % period) + period) % period
    let i: number
    let j: number
    let top: number
    if (s < time) {
      // Riding: the sample the trace puts them at, standing on the pedals with the weight back.
      let k = 0
      while (k < t.length - 1 && t[k + 1] <= s) k++
      const a = Math.max(0, k - 2)
      const b = Math.min(x.length - 1, k + 2)
      const face = Math.sign(x[b] - x[a]) || 1
      const air = Math.round(lift[k] / g.px)
      i = col(g, x[k])
      const ground = row(g, y[k])
      j = ground - air
      // The odd root jolts them a pixel off the saddle.
      const jolt = !air && hash(beat, n, 71) < 0.08 ? 1 : 0
      // Every other lap they throw the back end sideways at the top of the tabletop.
      const whip =
        air > 1 &&
        k > COURSE.takeoff &&
        k < COURSE.landing &&
        Math.floor(clock / period) % 2 === n % 2
      dot(d, g, i - face, j - 1 - (whip ? 1 : 0), ink.tyre)
      dot(d, g, i, j - 1, kit.frame)
      dot(d, g, i + face, j - 1, ink.tyre)
      dot(d, g, i + face, j - 2 - jolt, ink.tyre)
      dot(d, g, i, j - 2 - jolt, ink.shorts)
      dot(d, g, i - face, j - 3 - jolt, kit.jersey)
      dot(d, g, i, j - 3 - jolt, kit.jersey)
      dot(d, g, i - face, j - 4 - jolt, kit.helmet)
      top = j - 4 - jolt
      if (air) dot(d, g, i, ground, ink.rut)
      else if (Math.abs(bend[k]) > 0.02) {
        // Roost off the back wheel, thrown out of the berm.
        const out = Math.sign(bend[k]) * face
        dot(d, g, i - 2 * face, j - 1 - (beat & 1), ink.dust)
        dot(d, g, i - 2 * face, j - 2 + out, ink.dust)
        if (beat & 1) dot(d, g, i - 3 * face, j - 2, ink.dust)
      } else if (beat & 1) dot(d, g, i - 2 * face, j - 1, ink.dust)
      if (dark) lamp(i - face, top)
      heads[`rider${n}`] = [i - face, top]
      continue
    } else if ((s -= time) < FINISH) {
      // Astride the bike past the banner, a fist up for the time.
      const last = x.length - 1
      i = col(g, x[last]) - 5
      j = row(g, y[last])
      dot(d, g, i - 1, j - 1, ink.tyre)
      dot(d, g, i, j - 1, kit.frame)
      dot(d, g, i + 1, j - 1, ink.tyre)
      dot(d, g, i, j - 2, ink.shorts)
      dot(d, g, i, j - 3, kit.jersey)
      dot(d, g, i, j - 4, kit.helmet)
      if (s < 1 && beat & 2) dot(d, g, i + 1, j - 5, ink.skin)
      top = j - 4
    } else if ((s -= FINISH) < pushTime) {
      // Walking up with a hand on the bars, the bike rolling on ahead of them.
      const at = along(PUSH_PTS, s / pushTime)
      const f = at.dir
      i = col(g, at.x)
      j = row(g, at.y)
      if (Math.floor(clock * 2.5 + n) % 2) {
        dot(d, g, i - 1, j - 1, ink.shorts)
        dot(d, g, i + 1, j - 1, ink.shorts)
      } else dot(d, g, i, j - 1, ink.shorts)
      dot(d, g, i, j - 2, kit.jersey)
      dot(d, g, i, j - 3, kit.helmet)
      dot(d, g, i + f, j - 2, ink.skin)
      dot(d, g, i + f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 1, kit.frame)
      dot(d, g, i + 3 * f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 2, ink.tyre)
      top = j - 3
    } else {
      // In the hut, waiting for the beep.
      i = col(g, x[0])
      j = row(g, y[0])
      dot(d, g, i - 1, j - 1, ink.tyre)
      dot(d, g, i, j - 1, kit.frame)
      dot(d, g, i + 1, j - 1, ink.tyre)
      dot(d, g, i, j - 2, ink.shorts)
      dot(d, g, i, j - 3, kit.jersey)
      dot(d, g, i, j - 4, kit.helmet)
      top = j - 4
    }
    if (dark) lamp(i, top)
    heads[`rider${n}`] = [i, top]
  }
}
