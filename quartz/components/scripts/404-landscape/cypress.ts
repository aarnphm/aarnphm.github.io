import {
  clamp01,
  col,
  fbm,
  hash,
  noise,
  pick,
  put,
  ramp,
  row,
  type Grid,
  type RGB,
} from '../404-pixel'
import { brush, type Angle } from './raster'

export type Cypress = { x: number; base: number; h: number; w: number }

// Flame-shaped cypress: widest a third of the way up, pinched to a point, lit on the right flank,
// then painted over with upward strokes that fan toward the flanks and turn in an S as they climb.
export function paintCypress(
  d: Uint8ClampedArray,
  g: Grid,
  c: Cypress,
  stops: RGB[],
  seed: number,
  sway = 0,
) {
  const { bw, bh, sx, sy } = g
  const lean = (t: number) => Math.sin(t * 3 + seed) * c.w * 0.08 + sway * t * t * c.w
  const half = (y: number, t: number) => {
    const profile = t < 0.3 ? Math.sqrt(t / 0.3) : ((1 - t) / 0.7) ** 0.85
    return Math.max(g.px * 0.5, (c.w / 2) * profile * (0.82 + 0.36 * noise(y * 0.06, seed, seed)))
  }
  const i0 = Math.max(0, col(g, c.x - c.w))
  const i1 = Math.min(bw - 1, col(g, c.x + c.w))
  const j0 = Math.max(0, row(g, c.base - c.h))
  const j1 = Math.min(bh - 1, row(g, c.base))
  for (let j = j0; j <= j1; j++) {
    const y = sy[j]
    const t = (c.base - y) / c.h
    if (t < 0 || t > 1) continue
    const hw = half(y, t)
    const mid = c.x + lean(t)
    for (let i = i0; i <= i1; i++) {
      const u = (sx[i] - mid) / hw
      if (Math.abs(u) > 1) continue
      const tone = 0.3 + u * 0.4 + (fbm(sx[i] * 0.08, y * 0.05, seed, 2) - 0.5) * 0.5
      put(d, (j * bw + i) * 4, ramp(stops, tone, i, j))
    }
  }
  let fan = 0
  const angle: Angle = (_, by) => {
    const t = clamp01((c.base - (sy[0] + by * g.px)) / c.h)
    return -Math.PI / 2 + fan * 0.5 * (1 - t) + 0.45 * Math.sin(t * 9 + seed + sway * 4)
  }
  for (let j = j0; j <= j1; j += 2)
    for (let i = i0; i <= i1; i += 2) {
      const t = (c.base - sy[j]) / c.h
      if (t < 0.03 || t > 0.97) continue
      fan = (sx[i] - c.x - lean(t)) / half(sy[j], t)
      if (Math.abs(fan) > 0.9) continue
      const h = hash(i, j, seed)
      const ink = pick(stops, 0.32 + fan * 0.4 + (h - 0.5) * 0.3, hash(i, j, seed + 1))
      brush(d, g, i + h - 0.5, j, 3 + Math.floor(h * 3), false, ink, angle)
    }
}
