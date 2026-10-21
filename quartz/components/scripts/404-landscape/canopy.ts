import type { Palette } from '../../../util/scene-palette'
import type { Season } from '../../../util/visitor-world'
import type { Sky } from '../404-almanac'
import {
  BAYER,
  clamp01,
  fbm,
  hash,
  mix,
  noise,
  put,
  putRgb,
  ramp,
  smooth,
  type Grid,
} from '../404-pixel'
import { hillTop } from './geography'
import { segHit, type Canopy } from './trees'

// Gold only where the tone is mid-range and the gradient is sharp: fringes, never interiors.
const GOLD_START = 0.45
const GOLD_END = 0.7

// Octaves drift at their own speed, so coarse masses sway while fine leaves shimmer.
function windFbm(x: number, y: number, seed: number, octaves: number, wind: number) {
  let sum = 0
  let amp = 0.5
  let norm = 0
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise(x + wind * (1 + i * 0.7), y, seed + i * 17)
    norm += amp
    amp *= 0.5
    x *= 2.03
    y *= 2.03
  }
  return sum / norm
}

let canopyWood = new Uint8Array(0)
let canopyDensity = new Float32Array(0)
let canopyDepth = new Float32Array(0)
let canopyCover = new Float32Array(0)
let dappleFloor = new Float32Array(0)
let dappleFloorSx: Float32Array | null = null
let dappleFloorWater = NaN

// Leaves: each tip splats density and depth; noise dragged by per-octave wind thresholds it into
// masses. Depth follows the limbs: near leaves are denser, lighter and crisp-edged, far leaves
// darker with a wider penumbra, and that penumbra is what the ordered dither renders.
export function paintCanopy(
  d: Uint8ClampedArray,
  g: Grid,
  canopy: Canopy,
  pal: Palette,
  wind: number,
  drift: number,
  side: number,
  s: Season,
  sky: Sky,
) {
  const { bw, bh, sx, sy } = g
  const { bare, autumn, blossom } = s
  const snow = s.ground
  // A hard frost rimes the twigs in any weather (rime ice in freezing fog, glaze in freezing rain).
  // Hoarfrost adds to it only on clear, still, dry nights, since season() zeroes s.frost under
  // rain, snow or fog; twigs radiate their warmth away as the grass does and hold it until the
  // sun is on them.
  const rime = Math.max(
    sky.temp === null ? 0 : smooth(0, -8, sky.temp),
    smooth(0.4, 1, s.frost) * (1 - s.thaw),
  )
  const wet = sky.fall === 'rain' || sky.fall === 'drizzle' ? 0.4 + 0.6 * sky.heavy : 0
  const petal = mix(pal.snow[3], pal.poppy, 0.25)
  const total = bw * bh
  let wood: Uint8Array | null = null
  if (snow + rime > 0.05) {
    if (canopyWood.length < total) canopyWood = new Uint8Array(total)
    else canopyWood.fill(0, 0, total)
    wood = canopyWood
  }
  const radius = 40
  if (canopyDensity.length < total) {
    canopyDensity = new Float32Array(total)
    canopyDepth = new Float32Array(total)
    canopyCover = new Float32Array(total)
  } else {
    canopyDensity.fill(0, 0, total)
    canopyDepth.fill(0, 0, total)
    canopyCover.fill(0, 0, total)
  }
  const density = canopyDensity
  const depth = canopyDepth
  const toI = (x: number) => Math.floor((x - sx[0]) / g.px + 0.5)
  const toJ = (y: number) => Math.floor((y - sy[0]) / g.px + 0.5)

  for (const tree of canopy.trees) {
    for (const [tx, ty, tz] of tree.tips) {
      const i0 = Math.max(0, toI(tx - radius))
      const i1 = Math.min(bw - 1, toI(tx + radius))
      const j0 = Math.max(0, toJ(ty - radius))
      const j1 = Math.min(bh - 1, toJ(ty + radius))
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const q = 1 - Math.hypot(sx[i] - tx, sy[j] - ty) / radius
          if (q <= 0) continue
          const p = j * bw + i
          density[p] += q * q
          depth[p] += q * q * tz
        }
    }
    for (const s of tree.segs) {
      // Twigs thinner than a pixel stay hidden in the leaves instead of stair-stepping across it,
      // until the leaves are down and the twigs are all there is to see.
      if (s.w0 < g.px * (0.8 - 0.45 * bare)) continue
      const pad = Math.max(s.w0, s.w1) / 2 + g.px
      const i0 = Math.max(0, toI(Math.min(s.x0, s.x1) - pad))
      const i1 = Math.min(bw - 1, toI(Math.max(s.x0, s.x1) + pad))
      const j0 = Math.max(0, toJ(Math.min(s.y0, s.y1) - pad))
      const j1 = Math.min(bh - 1, toJ(Math.max(s.y0, s.y1) + pad))
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const [dist, w, side] = segHit(s, sx[i], sy[j])
          const half = Math.max(w / 2, g.px * 0.5)
          if (dist > half) continue
          put(d, (j * bw + i) * 4, ramp(pal.bark, 0.5 + (side / half) * 0.5, i, j))
          if (wood) wood[j * bw + i] = 1
        }
    }
  }
  // Snow lies along the tops of the limbs, a pixel deep and two in a heavy fall; in a hard frost
  // rime whitens their edges.
  if (wood)
    for (let j = 1; j < bh - 1; j++)
      for (let i = 1; i < bw - 1; i++) {
        const p = j * bw + i
        if (!wood[p]) continue
        const r = hash(i, j, 5)
        if (!wood[p - bw] && r < snow * 1.4) {
          put(d, p * 4, pal.snow[3])
          if (r < (snow - 0.6) * 2.5) put(d, (p - bw) * 4, pal.snow[3])
        } else if ((!wood[p - 1] || !wood[p + 1] || !wood[p + bw]) && r < rime * 0.7)
          put(d, p * 4, pal.snow[2])
      }

  // Continuous coverage T: 0.5 is the leaf edge, the slope is the penumbra width.
  const cover = canopyCover
  const scale = 1 / (radius * 1.6)
  for (let p = 0; p < total; p++) {
    if (density[p] < 0.05) continue
    const i = p % bw
    const j = (p - i) / bw
    const z = depth[p] / density[p]
    depth[p] = z
    const n = windFbm(sx[i] * scale, sy[j] * scale + drift, 71, 2, wind)
    // In winter most of the leaves are down and the limbs show.
    const v = density[p] * (0.35 + n) * (1 + 0.15 * z) * (1 - 0.92 * bare)
    const penumbra = 0.1 + 0.16 * (1 - z) * 0.5
    cover[p] = clamp01((v - 0.55) / penumbra + 0.5)
  }
  for (let j = 1; j < bh - 1; j++)
    for (let i = 1; i < bw - 1; i++) {
      const p = j * bw + i
      const t = cover[p]
      if (t <= BAYER[(j & 3) * 4 + (i & 3)]) continue
      const k = p * 4
      const grad = Math.max(
        Math.abs(cover[p + 1] - cover[p - 1]),
        Math.abs(cover[p + bw] - cover[p - bw]),
      )
      // Only the fringe facing the light, up and to its side, catches gold.
      const sunward = cover[p - bw + side] < t
      // A tree turns a limb at a time, so the autumn comes in patches the size of a limb.
      const leaf =
        autumn > 0 && fbm(sx[i] * 0.012, sy[j] * 0.012, 29, 2) < 0.15 + 0.55 * autumn
          ? pal.autumn
          : pal.crown
      // Snow settles on the tops of the masses, and blossom comes out in clusters.
      const cap = cover[p - bw] < 0.5 ? 1 : cover[p - 2 * bw] < 0.5 ? 2 : 0
      if (cap && hash(i, j, 9) < (cap === 1 ? snow * 1.3 : snow - 0.5))
        put(d, k, pal.snow[cap === 1 ? 3 : 2])
      else if (blossom > 0.05 && fbm(sx[i] * 0.07, sy[j] * 0.07, 37, 2) > 0.7 - 0.25 * blossom)
        put(d, k, petal)
      else if (t >= GOLD_START && t <= GOLD_END && grad > 0.3 && sunward && wet < 0.3)
        put(d, k, pal.gold)
      else if (t < 0.6 && !sunward) put(d, k, leaf[0])
      else {
        // Wet leaves go darker.
        const tone = 0.2 + fbm(sx[i] * 0.05, sy[j] * 0.05, 83, 2) * 0.5 + depth[p] * 0.18
        put(d, k, ramp(leaf, tone + (sunward ? 0.12 : 0) - 0.12 * wet, i, j))
      }
    }
}

// Cast shade from the canopy on the ground plane. Penumbra follows U = f·b/a: the farther the
// ground sits from the occluding limbs (b), the wider the soft edge, so shade is crisp at the
// roots and dissolves into dithered haze toward the temple. Fringes are gold only while sharp.
export function paintDapple(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  roots: number[],
  wind: number,
  clock: number,
  water: number,
) {
  const { bw, bh, sx, sy } = g
  const [sr, sg, sb, sa] = pal.shade
  const [fr, fg, fb, fa] = pal.fringe
  const focal = 0.09
  // Shade lands only on the near ground, the hill.
  if (dappleFloorSx !== sx || dappleFloorWater !== water || dappleFloor.length < bw) {
    if (dappleFloor.length < bw) dappleFloor = new Float32Array(bw)
    for (let i = 0; i < bw; i++) dappleFloor[i] = hillTop(sx[i], water) - 6
    dappleFloorSx = sx
    dappleFloorWater = water
  }
  const floor = dappleFloor
  const fa255 = fa * 255
  const sa255 = sa * 255
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    const ground = 1 - smooth(water - 20, water, y)
    if (ground <= 0 || y < 560) continue
    for (let i = 0; i < bw; i++) {
      const x = sx[i]
      if (y < floor[i]) continue
      let rMin = Infinity
      for (let r = 0; r < roots.length; r++) {
        const dist = Math.abs(x - roots[r])
        if (dist < rMin) rMin = dist
      }
      const b = Math.min(1, rMin / 640)
      const cover = ground * (1 - smooth(0.25, 0.85, b))
      if (cover < 0.06) continue
      const k = (j * bw + i) * 4
      if (d[k + 3] === 255) continue
      const n = windFbm(x * 0.011, y * 0.03 + Math.sin(clock * 0.4) * 0.05, 97, 3, wind)
      const cut = 1 - cover * 0.6
      const u = 0.012 + focal * b
      const shade = smooth(cut - u, cut + u, n)
      if (shade >= GOLD_START && shade <= GOLD_END && u < 0.035) putRgb(d, k, fr, fg, fb, fa255)
      else if (shade > BAYER[(j & 3) * 4 + (i & 3)]) putRgb(d, k, sr, sg, sb, sa255)
    }
  }
}
