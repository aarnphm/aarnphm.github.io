import type { Palette } from '../../../util/scene-palette'
import type { Sky } from '../404-almanac'
import type { Disc } from './sky'
import {
  BAYER,
  col,
  dither,
  dot,
  hash,
  mix,
  mulberry,
  noise,
  put,
  putMix,
  ramp,
  row,
  type Grid,
  type RGB,
} from '../404-pixel'
import { FIRE, HOME } from './camp'
import { midRidge } from './geography'
import { FLAME, GOLDEN, ROSE } from './light-marks'

// Glitter on the water: horizontal dashes wherever a facet tilts the sun (or the open portal)
// toward us, widening with distance below the horizon.
export function paintGlitter(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  heat: number,
  clock: number,
  body: Disc | null,
) {
  // The moon lays a path only in proportion to how much of it is lit.
  const lit = body ? (1 - Math.abs(body.phase - 0.5) * 2) * (1 - body.veil) : 0
  const { bw, bh, sx, sy } = g
  for (let j = Math.max(0, row(g, water) + 1); j < bh; j += 2) {
    const depth = sy[j] - water
    const ws = 14 + depth * 0.9
    const wp = 20 + depth * 0.5
    for (let i = (j >> 1) % 3; i < bw; i += 3) {
      const ps = body ? 0.42 * lit * Math.exp(-(((sx[i] - body.x) / ws) ** 2)) : 0
      const pp = (0.05 + 0.45 * heat) * Math.exp(-(((sx[i] - HOME) / wp) ** 2))
      if (ps + pp < 0.01) continue
      // Each facet rerolls on its own beat, so the glitter shimmers instead of strobing.
      const beat = Math.floor(clock * 3 + hash(i, j, 7) * 3)
      const roll = hash(i, j, beat)
      if (roll >= ps + pp) continue
      const ink =
        roll < ps ? pal.glint[roll < ps * 0.4 ? 0 : 1] : roll < ps + pp * 0.5 ? ROSE : GOLDEN
      const len = 2 + Math.floor(hash(i, j, beat + 1) * 3)
      for (let s = 0; s < len; s++) dot(d, g, i + s, j, ink)
    }
  }
}

export type Ripple = {
  x: number
  y: number
  w: number
  bright: boolean
  half: number
  phase: number
}

export function makeRipples(water: number): Ripple[] {
  const r = mulberry(99)
  return Array.from({ length: 52 }, () => {
    const y = water + 12 + r() ** 0.8 * 110
    const x = r() * 1600
    const w = 20 + r() * (30 + (y - water) * 0.6)
    return { x, y, w, bright: r() < 0.35, half: 0.8 + r() * 1.2, phase: r() * 4 }
  })
}

// The river: a plate that darkens toward the viewer, with the scene above folded into it. Near the
// bank the fold keeps the old squash (0.62); further out it steepens, so the bottom of the view
// reaches up the gate however little water the viewport shows. The fold is cut into slices that
// shear on a stop-motion beat, and wherever the reflection crosses from one mass to another (a
// ridge into sky, the ruin into the hill) it draws a contour. Then the campfire's light in broken
// streaks, ripple dashes that hop back and forth, and the bank as one row of ink.
const LIFT = 480
// Reflected layers, front to back, and the ids the contours compare: the ranges split into the mid
// ridge and the far one, and NONE for open sky.
const FIGURES = 0
const WELL = 3
const RANGES = 4
const FAR = 5
const NONE = 6
// The figures and the rift carry their own keylines, and the rift's dithered bleed would speckle.
const contoured = (a: number, b: number) =>
  a !== b && a !== FIGURES && b !== FIGURES && a !== WELL && b !== WELL && (a > 1 || b > 1)

let waterMid = new Float32Array(0)
let waterMidSx: Float32Array | null = null
let waterAbove = new Uint8Array(0)
let waterHere = new Uint8Array(0)

export function paintWater(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  ink: RGB,
  layers: (Uint8ClampedArray | null)[],
  ripples: Ripple[],
  clock: number,
  dark: boolean,
  doused: boolean,
) {
  const { bw, bh, sx, sy } = g
  const shore = row(g, water)
  const slice = Math.floor(clock / 0.3)
  const reach = Math.max(8 * g.px, g.bottom - water)
  const knee = reach * 0.25
  const bend = Math.max(0, (LIFT - reach / 0.62) / (reach - knee) ** 2)
  const contour = mix(pal.water[0], ink, dark ? 0.5 : 0.45)
  if (waterMidSx !== sx || waterMid.length < bw) {
    if (waterMid.length < bw) waterMid = new Float32Array(bw)
    for (let i = 0; i < bw; i++) waterMid[i] = midRidge(sx[i])
    waterMidSx = sx
  }
  const mid = waterMid
  if (waterAbove.length < bw) {
    waterAbove = new Uint8Array(bw)
    waterHere = new Uint8Array(bw)
  } else {
    waterAbove.fill(0, 0, bw)
    waterHere.fill(0, 0, bw)
  }
  let above = waterAbove
  let here = waterHere
  for (let j = Math.max(0, shore + 1); j < bh; j++) {
    const depth = sy[j] - water
    const ys = water - depth / 0.62 - bend * Math.max(0, depth - knee) ** 2
    const src = row(g, ys)
    // Three flat bands, paler with distance from the bank.
    const pale = [0.35, 0.5, 0.65][Math.min(2, Math.floor((depth / reach) * 3))]
    const shear = Math.round((noise(Math.floor(j / 3) * 0.9, slice * 1.7, 91) - 0.5) * 6)
    for (let i = 0; i < bw; i++) {
      const k = (j * bw + i) * 4
      const plate = ramp(pal.water, 1 - depth / 180, i, j)
      put(d, k, plate)
      const si = i - shear
      let id = NONE
      if (src >= 0 && si >= 0 && si < bw) {
        const q = (src * bw + si) * 4
        for (let n = 0; n < layers.length; n++) {
          const s = layers[n]
          if (!s || s[q + 3] !== 255) continue
          id = n === RANGES && ys < mid[si] ? FAR : n
          putMix(d, k, s[q], s[q + 1], s[q + 2], plate, pale)
          break
        }
      }
      here[i] = id
      const up = j > shore + 1 ? above[i] : id
      const left = i > 0 ? here[i - 1] : id
      if (contoured(id, up) || contoured(id, left)) put(d, k, contour)
    }
    ;[above, here] = [here, above]
  }

  const fc = col(g, FIRE.x)
  const beat = Math.floor(clock * 6)
  for (let j = Math.max(0, shore + 1); j < bh && !doused; j++) {
    const depth = sy[j] - water
    const p = (1 - depth / 110) * (dark ? 0.85 : 0.45)
    if (p <= 0) break
    if (hash(j, beat, 29) > p) continue
    const len = 1 + Math.floor(hash(j, beat, 31) * 3)
    const off = Math.round((hash(j, beat, 37) - 0.5) * 4) - (len >> 1)
    const c = depth < 20 && hash(j, beat, 39) < 0.4 ? FLAME[0] : depth < 45 ? FLAME[1] : FLAME[2]
    for (let s = 0; s < len; s++) dot(d, g, fc + off + s, j, c)
  }

  const paper: RGB = dark ? [16, 15, 15] : [255, 252, 240]
  for (const r of ripples) {
    const j = row(g, r.y)
    if (j <= shore || j >= bh) continue
    const hop = Math.floor((clock + r.phase) / r.half) % 2 ? 10 : 0
    const c = r.bright ? paper : ink
    const a = r.bright ? 0.75 : 0.35
    const i0 = col(g, r.x + hop)
    for (let s = Math.max(1, Math.round(r.w / g.px)) - 1; s >= 0; s--) {
      const i = i0 + s
      if (i < 0 || i >= bw) continue
      const k = (j * bw + i) * 4
      for (let n = 0; n < 3; n++) d[k + n] += (c[n] - d[k + n]) * a
    }
  }

  if (shore >= 0 && shore < bh) for (let i = 0; i < bw; i++) put(d, (shore * bw + i) * 4, ink)
}

export type Splash = { x: number; y: number; size: number; age: number }

const splashSeen = new Set<number>()

// Where a letter lands: two flat rings that widen in five steps, darkening the water under them
// less at every step until they dither away.
export function paintSplashes(d: Uint8ClampedArray, g: Grid, splashes: Splash[], ink: RGB) {
  const { bw, bh } = g
  for (const s of splashes)
    for (let n = 0; n < 2; n++) {
      const t = (s.age - n * 0.28) / 1.2
      if (t < 0 || t >= 1) continue
      const f = Math.floor(t * 5) / 5
      const rx = (s.size * (0.15 + 1.15 * f)) / g.px
      const ry = Math.max(1, rx * 0.2)
      const cx = (s.x - g.sx[0]) / g.px
      const cy = (s.y - g.sy[0]) / g.px
      splashSeen.clear()
      const steps = Math.ceil(2 * Math.PI * rx)
      for (let k = 0; k < steps; k++) {
        const a = (k / steps) * 2 * Math.PI
        const i = Math.round(cx + Math.cos(a) * rx)
        const j = Math.round(cy + Math.sin(a) * ry)
        const p = j * bw + i
        if (i < 0 || j < 0 || i >= bw || j >= bh || splashSeen.has(p)) continue
        splashSeen.add(p)
        if (1 - f <= BAYER[(j & 3) * 4 + (i & 3)]) continue
        for (let c = 0; c < 3; c++) d[p * 4 + c] += (ink[c] - d[p * 4 + c]) * 0.5
      }
    }
}

// Rain in streaks slanting with the wind, a fresh set every tick, the way stop-motion rain is drawn;
// snow in flakes that keep their place from tick to tick and wander as they fall, the nearest ones
// two pixels across.
export function paintFall(
  d: Uint8ClampedArray,
  g: Grid,
  sky: Sky,
  clock: number,
  ink: RGB,
  flake: RGB,
) {
  if (sky.fall === 'none') return
  const { bw, bh } = g
  if (sky.fall === 'snow') {
    const n = Math.round(70 + 330 * sky.heavy)
    for (let k = 0; k < n; k++) {
      const speed = 6 + 10 * hash(k, 1, 61)
      const y = ((hash(k, 2, 61) * (bh + 8) + clock * speed) % (bh + 8)) - 4
      const x = hash(k, 3, 61) * bw + clock * sky.wind * 8 + Math.sin(clock * 0.8 + k) * 2
      const i = Math.round(((x % bw) + bw) % bw)
      const j = Math.round(y)
      dot(d, g, i, j, flake)
      if (hash(k, 5, 61) > 0.86) {
        dot(d, g, i + 1, j, flake)
        dot(d, g, i, j + 1, flake)
        dot(d, g, i + 1, j + 1, flake)
      }
    }
    return
  }
  const beat = Math.floor(clock * 10)
  const n = Math.round((sky.fall === 'drizzle' ? 40 : 60) + 300 * sky.heavy)
  const drizzleShare = sky.fall === 'drizzle' ? 1 : 0.7 - 0.48 * sky.heavy
  const streakLen = 3 + Math.round(sky.heavy * 2)
  const slant = 0.15 + 0.6 * sky.wind
  for (let k = 0; k < n; k++) {
    const len = hash(k, beat, 65) < drizzleShare ? 2 : streakLen
    const i0 = hash(k, beat, 67) * (bw + len) - len * slant
    const j0 = Math.floor(hash(k, beat, 71) * bh)
    for (let s = 0; s < len; s++) {
      const i = Math.round(i0 + s * slant)
      if (dither(0.8, i, j0 + s)) dot(d, g, i, j0 + s, ink)
    }
  }
}

// Where rain meets the water: a ring seen edge-on, a pale pixel either side of the drop, gone by
// the next tick.
export function paintRings(
  d: Uint8ClampedArray,
  g: Grid,
  water: number,
  sky: Sky,
  clock: number,
  ink: RGB,
) {
  if (sky.fall !== 'rain' && sky.fall !== 'drizzle') return
  const { bw, bh } = g
  const shore = row(g, water) + 2
  if (shore >= bh) return
  const beat = Math.floor(clock * 10)
  const n = Math.round(8 + 40 * sky.heavy)
  for (let k = 0; k < n; k++) {
    const i = Math.floor(hash(k, beat, 73) * bw)
    const j = shore + Math.floor(hash(k, beat, 79) * (bh - shore))
    const w = 1 + Math.floor(hash(k, beat, 83) * 2)
    dot(d, g, i - w, j, ink)
    dot(d, g, i + w, j, ink)
  }
}
