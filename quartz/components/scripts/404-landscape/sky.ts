import type { Palette } from '../../../util/scene-palette'
import type { Body } from '../404-almanac'
import {
  BAYER,
  clampCol,
  col,
  dot,
  fbm,
  frac,
  hash,
  mix,
  mulberry,
  pick,
  put,
  putMix,
  ramp,
  row,
  smooth,
  type Grid,
  type Pt,
  type RGB,
} from '../404-pixel'
import { skyline } from './geography'
import { CREAM, CROSS, halo } from './light-marks'
import { brush, seeds, type Angle, type Keep } from './raster'

// The sun or the moon as drawn: its centre and radius in scene units, its phase (0.5 is the full
// disc, which is how the sun is always drawn), whether the lit limb is mirrored (south of the
// equator the moon waxes from the left), and how much cloud veils it.
export type Disc = { x: number; y: number; r: number; phase: number; mirror: boolean; veil: number }

// Where a body hangs over this grid: across the visible sky from its azimuth, and up from the
// skyline under it by its altitude, so it sets behind whatever ridge stands there.
export function hang(b: Body, g: Grid, r: number) {
  const left = g.sx[0] + 48
  const right = g.sx[g.bw - 1] - 48
  const half = Math.min(700, (right - left) / 2 - 60)
  const x = (left + right) / 2 + b.across * half * 0.85
  const floor = skyline(x)
  const y =
    b.alt >= 0
      ? floor - (floor - 70) * Math.sin((b.alt * Math.PI) / 180) ** 0.6
      : floor - b.alt * 8 + r * 0.4
  return { x, y }
}

// 1 on the lit part of the disc, 0 on its dark side, -1 off it.
function onDisc(b: Disc, x: number, y: number) {
  const u = (x - b.x) / b.r
  const v = (y - b.y) / b.r
  if (u * u + v * v >= 1) return -1
  const rim = Math.sqrt(1 - v * v)
  const k = Math.cos(2 * Math.PI * b.phase)
  const w = b.mirror ? -u : u
  return (b.phase <= 0.5 ? w > k * rim : w < -k * rim) ? 1 : 0
}

// Sky flow. A stream function of three drifting plane waves: its curl is divergence-free, so the
// strokes eddy without pooling. Lamb–Oseen vortices on top: the sun's halo, the eddies the
// cursor sheds, and the portal's pull (a vortex plus a sink, so the sky spirals into the gate).
type Vortex = { x: number; y: number; gamma: number; age: number }
export type Air = {
  drift: number
  time: number
  pull: number
  pole: Pt
  vortices: Vortex[]
  body: Disc | null
  cover: number
}

const WAVES = [
  [0.006, 0.014, 0.11, 0.4],
  [-0.013, 0.021, -0.17, 2.1],
  [0.024, -0.017, 0.23, 4.4],
].map(([kx, ky, w, phi]) => ({ kx, ky, w, phi, a: 6 / Math.hypot(kx, ky) }))

const V = { x: 0, y: 0 }

function vortex(dx: number, dy: number, gamma: number, core: number, sink: number) {
  const r2 = dx * dx + dy * dy + 1
  const k = (1 - Math.exp(-r2 / (core * core))) / (2 * Math.PI * r2)
  V.x += (-dy * gamma - dx * sink) * k
  V.y += (dx * gamma - dy * sink) * k
}

export function airAngle(air: Air, x: number, y: number) {
  V.x = 24
  V.y = 2
  const u = x - air.drift
  for (const w of WAVES) {
    const c = w.a * Math.cos(w.kx * u + w.ky * y + w.w * air.time + w.phi)
    V.x += w.ky * c
    V.y -= w.kx * c
  }
  if (air.body) vortex(x - air.body.x, y - air.body.y, 42000, 70, 0)
  for (const v of air.vortices) vortex(x - v.x, y - v.y, v.gamma, 40 + Math.sqrt(v.age) * 30, 0)
  if (air.pull > 0.01)
    vortex(x - air.pole[0], y - air.pole[1], air.pull * 160000, 240, air.pull * 70000)
  return Math.atan2(V.y, V.x)
}

// What a sky stroke carries: the sun or the lit part of the moon, its banded halo, drifting cloud,
// or the sky gradient. More cloud lowers the threshold and lets the bank reach further down.
// How much cloud stands at a point in the sky, and the field it is cut from.
export function cloudAt(air: Air, x: number, y: number) {
  const c = fbm((x - air.drift) * 0.0035, y * 0.012, 131, 3)
  const lo = 0.62 - 0.4 * air.cover
  const low = Math.max(0, air.cover - 0.5) * 2
  return [smooth(lo, lo + 0.18, c) * (1 - smooth(230 + 260 * low, 480 + 300 * low, y)), c] as const
}

function skyInk(pal: Palette, air: Air, x: number, y: number, thr: number, mask: number) {
  const b = air.body
  if (b && b.veil < 0.85) {
    const r = Math.hypot(x - b.x, y - b.y)
    if (r < b.r && onDisc(b, x, y) === 1) return pick(pal.sun, 0.7 + 0.3 * (1 - r / b.r), thr)
    if (r >= b.r && r < b.r + 70 && b.veil < 0.4) {
      const fall = 1 - (r - b.r) / 70
      const band = 0.5 + 0.5 * Math.cos((r - b.r) * 0.21)
      if (band * fall > 0.3 + mask * 0.25) return pick(pal.sun, fall * 0.9, thr)
    }
  }
  const [cloud, c] = cloudAt(air, x, y)
  if (cloud > mask) return pick(pal.cloud, (c - 0.5) * 3.5, thr)
  return pick(pal.sky, smooth(60, 520, y) + (c - 0.5) * 0.35, thr)
}

// Inside an eddy the strokes take alternating light and dark bands by distance from its centre,
// so the swirl reads the way the swirls do in the Saint-Rémy night sky.
function eddyInk(pal: Palette, air: Air, x: number, y: number, mask: number) {
  let swirl = 0
  let band = 0
  const feel = (dx: number, dy: number, strength: number, reach: number, wave: number) => {
    const r2 = dx * dx + dy * dy
    const e = strength * Math.exp(-r2 / (2 * reach * reach))
    if (e <= swirl) return
    swirl = e
    band = Math.cos(Math.sqrt(r2) * wave)
  }
  for (const v of air.vortices)
    feel(x - v.x, y - v.y, Math.min(1, Math.abs(v.gamma) / 30000), 70 + Math.sqrt(v.age) * 50, 0.16)
  if (air.pull > 0.01)
    feel(x - air.pole[0], y - air.pole[1], Math.min(1, air.pull) * 0.7, 420, 0.045)
  if (swirl <= mask) return null
  return band > 0.2 ? pal.swirl[1] : band < -0.2 ? pal.swirl[0] : null
}

// The dark side of the moon keeps a trace of earthshine.
export function paintSkyBase(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  floor: number,
  b: Disc | null,
) {
  const { bw, bh, sx, sy } = g
  const shown = b && b.veil < 0.85
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    if (y > floor) break
    for (let i = 0; i < bw; i++) {
      const k = (j * bw + i) * 4
      const sky = ramp(pal.sky, smooth(60, 520, y), i, j)
      const r = shown ? Math.hypot(sx[i] - b.x, y - b.y) : Infinity
      const on = shown && r < b.r ? onDisc(b, sx[i], y) : -1
      if (on === 1) put(d, k, pal.sun[2])
      else if (on === 0) put(d, k, mix(sky, pal.sun[0], 0.18))
      else if (shown && r < b.r + 18 && b.veil < 0.4 && b.phase > 0.3 && b.phase < 0.7)
        put(d, k, ramp(pal.sun, 1 - (r - b.r) / 18, i, j))
      else put(d, k, sky)
    }
  }
}

export type SkyStar = { x: number; y: number; mag: number; freq: number; phase: number }

// Fewer show through cloud, and none near the moon.
export function makeStars(b: Disc | null, cover: number): SkyStar[] {
  const r = mulberry(77)
  const stars: SkyStar[] = []
  const most = Math.round(18 * (1 - smooth(0.2, 0.9, cover)))
  for (let n = 0; n < 80 && stars.length < most; n++) {
    const x = 40 + r() * 1520
    const y = 30 + r() * 420
    const star = { x, y, mag: r() ** 1.6, freq: 0.6 + r() * 1.6, phase: r() * Math.PI * 2 }
    if (y < skyline(x) - 50 && (!b || Math.hypot(x - b.x, y - b.y) > 130)) stars.push(star)
  }
  return stars
}

export function paintSky(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  air: Air,
  base: Uint8ClampedArray,
  ridge: Float32Array,
  stars: SkyStar[],
  clock: number,
) {
  d.set(base)
  const x0 = g.sx[0]
  const y0 = g.sy[0]
  const angle: Angle = (bx, by) => airAngle(air, x0 + bx * g.px, y0 + by * g.px)
  let floor = 0
  for (const y of ridge) floor = Math.max(floor, y)
  // The disc stays whole: only its own strokes may cross into it.
  const b = air.body && air.body.veil < 0.85 ? air.body : null
  const clear: Keep = (i, j) => !b || Math.hypot(g.sx[i] - b.x, g.sy[j] - b.y) > b.r
  seeds(g, 3, air.drift / g.px, [0, row(g, floor)], 131, (bx, by, h, k) => {
    const x = x0 + bx * g.px
    const y = y0 + by * g.px
    if (y > ridge[clampCol(g, bx)] + g.px * 2) return
    const c = eddyInk(pal, air, x, y, k) ?? skyInk(pal, air, x, y, h, k)
    const len = 6 + Math.floor(frac(k * 7.13) * 5)
    const near = !!b && Math.hypot(x - b.x, y - b.y) < b.r + len * g.px
    const guard = near && !pal.sun.includes(c) ? clear : undefined
    brush(d, g, bx, by, len, frac(k * 3.7) < 0.25, c, angle, guard)
  })

  const plot = (x: number, y: number, c: RGB, alpha: number) => {
    const i = col(g, x)
    const j = row(g, y)
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  for (const s of stars) {
    const twinkle = 0.6 + 0.4 * Math.sin(clock * s.freq + s.phase)
    plot(s.x, s.y, pal.star, 1)
    if (s.mag > 0.4)
      for (const [ox, oy] of CROSS) plot(s.x + ox * g.px, s.y + oy * g.px, pal.star, twinkle * 0.8)
    if (s.mag > 0.65)
      halo(
        plot,
        s.x,
        s.y,
        (2.4 + s.mag * 1.4) * g.px,
        g.px * 0.8,
        pal.sun[1],
        twinkle * 0.7,
        clock * 0.2 + s.phase,
        5,
      )
  }
}

// A storm throws a bolt every few seconds: a jagged line from the cloud base down to the ridges,
// lit for two ticks with the sky blanched round it.
export function paintBolt(d: Uint8ClampedArray, g: Grid, clock: number, ridge: Float32Array) {
  const n = Math.floor(clock / 6.5)
  const t = clock - n * 6.5 - hash(n, 1, 89) * 4
  if (t < 0 || t > 0.22) return
  for (let k = 0; k < d.length; k += 4)
    if (d[k + 3]) putMix(d, k, d[k], d[k + 1], d[k + 2], CREAM, 0.22)
  const fork = (i: number, j: number, seed: number, reach: number) => {
    for (let s = 0; s < reach; s++) {
      const floor = row(g, ridge[clampCol(g, i)])
      if (j >= floor) return
      dot(d, g, i, j, CREAM)
      j++
      if (hash(i, j, seed) < 0.4) i += hash(j, i, seed) < 0.5 ? -1 : 1
      if (seed === n && s === 9) fork(i, j, n + 1, 14)
    }
  }
  fork(Math.floor(g.bw * (0.1 + 0.8 * hash(n, 2, 89))), row(g, 110 + 60 * hash(n, 3, 89)), n, 200)
}
