import { GATE, hourAngle, MISSING, TILT } from '../404-gate'
import {
  BAYER,
  col,
  frac,
  hash,
  noise,
  pick,
  put,
  ramp,
  row,
  type Grid,
  type RGB,
} from '../404-pixel'
import { CREAM, CROSS, GOLDEN, halo, RIFT, ROSE } from './light-marks'
import { LEAK, orbit, type Opening, type Rift } from './rift-motion'

const GLOW = 230

// Idle stroke pitch, and the winding that makes two arms match it: tan(pitch) = 2 / WIND.
const PITCH = 0.3
const WIND = 2 / Math.tan(PITCH)

// How much of the rift shows at each buffer pixel, out of 255: all of it inside the back edge of the
// opening, a pixel generous so it tucks under the soffit and the ring. Where the eight o'clock stone
// is missing nothing holds it in, so it bleeds out through the gap and thins into dither.
function opening(g: Grid): Opening {
  const mask = new Uint8Array(g.bw * g.bh)
  const reach = GATE.r + g.px
  const gap = hourAngle(MISSING) + (TILT * Math.PI) / 180
  let [i0, i1, j0, j1] = [g.bw, -1, g.bh, -1]
  for (let j = 0; j < g.bh; j++)
    for (let i = 0; i < g.bw; i++) {
      const dx = g.sx[i] - GATE.x
      const dy = g.sy[j] - GATE.y
      const r = Math.hypot(dx, dy)
      const off = Math.atan2(dy, dx) - gap
      const bleed = r > GATE.r && Math.abs(Math.atan2(Math.sin(off), Math.cos(off))) < Math.PI / 12
      const cover = bleed
        ? Math.round(255 * Math.max(0, 1 - (r - GATE.r) / LEAK) ** 1.5)
        : Math.hypot(dx, dy - GATE.depth) <= reach
          ? 255
          : 0
      if (!cover) continue
      mask[j * g.bw + i] = cover
      i0 = Math.min(i0, i)
      i1 = Math.max(i1, i)
      j0 = Math.min(j0, j)
      j1 = Math.max(j1, j)
    }
  return { grid: g, mask, i0, i1, j0, j1 }
}

export function paintRift(d: Uint8ClampedArray, g: Grid, rift: Rift, clock: number) {
  const { bw, sx, sy } = g
  const { pole } = rift
  if (rift.opening?.grid !== g) rift.opening = opening(g)
  const { mask, i0, i1, j0, j1 } = rift.opening
  const plot = (x: number, y: number, c: RGB, alpha: number) => {
    const i = col(g, x)
    const j = row(g, y)
    if (i < i0 || i > i1 || j < j0 || j > j1) return
    const k = j * bw + i
    if (alpha * mask[k] > BAYER[(j & 3) * 4 + (i & 3)] * 255) put(d, k * 4, c)
  }

  // Underpainting: a dark well that brightens toward the pole.
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      if (mask[j * bw + i] <= BAYER[(j & 3) * 4 + (i & 3)] * 255) continue
      const glow = 1 - Math.min(1, Math.hypot(sx[i] - pole[0], sy[j] - pole[1]) / GLOW)
      put(d, (j * bw + i) * 4, ramp(RIFT, 0.04 + glow * 0.34, i, j))
    }

  // Each stroke runs back along its mote's spiral, whose pitch opens as the inflow outruns the
  // spin. Colour mixes the nebula the mote was born in with a two-armed density wave (Lin–Shu):
  // trailing arms at the strokes' own idle pitch, turned against the spin so they stream inward
  // while the material circles through them, and nothing ever winds up.
  const spin = (rift.rate * Math.PI) / 180
  const trail = Math.min(1.1, Math.max(0, (rift.rate - 2) / 30))
  for (const m of rift.motes) {
    const x = pole[0] + Math.cos(m.a) * m.r
    const y = pole[1] + Math.sin(m.a) * m.r
    const pitch = Math.min(1.25, PITCH + Math.atan2(rift.inflow, orbit(m.r) * spin))
    const glow = 1 - Math.min(1, m.r / GLOW)
    const arm = Math.cos(2 * (m.a - rift.pattern) + WIND * Math.log(m.r / 25 + 0.05))
    const twinkle = m.star
      ? 0.75 * (0.35 + 0.65 * m.mag * (0.6 + 0.4 * Math.sin(clock * m.freq + m.phase)))
      : 1
    const crest = Math.max(0, arm) ** 4 * 0.22
    const ink = m.star ?? pick(RIFT, m.tone * 1.2 - 0.36 + glow * 0.3 + arm * 0.16 + crest, m.thr)
    const len = Math.max(m.len * g.px, m.r * trail * orbit(m.r) * 0.6)
    const n = Math.ceil(len / (g.px * 0.8))
    const cp = (Math.cos(pitch) * len) / n
    const sp = (Math.sin(pitch) * len) / n
    let r = m.r
    let a = m.a
    for (let s = 0; s < n; s++) {
      const fade = twinkle * (1 - (s / n) * trail * 0.85)
      plot(pole[0] + Math.cos(a) * r, pole[1] + Math.sin(a) * r, ink, fade)
      if (m.wide)
        plot(pole[0] + Math.cos(a) * (r + g.px), pole[1] + Math.sin(a) * (r + g.px), ink, fade)
      a -= cp / r
      r += sp
    }
    if (m.star && m.mag > 0.8)
      for (const [ox, oy] of CROSS) plot(x + ox * g.px, y + oy * g.px, m.star, twinkle * 0.8)
    if (m.star && m.mag > 0.93)
      halo(plot, x, y, g.px * 2.3, g.px * 0.8, m.star, twinkle * 0.45, clock * 0.3, 4)
  }

  // Polaris as the Saint-Rémy stars are painted: a hard core inside counter-turning broken rings.
  const breath = 0.5 + 0.5 * Math.sin(clock * 2.2)
  plot(pole[0], pole[1], CREAM, 1)
  for (const [ox, oy] of CROSS) plot(pole[0] + ox * g.px, pole[1] + oy * g.px, CREAM, 1)
  halo(
    plot,
    pole[0],
    pole[1],
    (3.2 + rift.flare * 2) * g.px,
    g.px * 0.7,
    GOLDEN,
    0.9,
    clock * 0.5,
    6,
  )
  halo(
    plot,
    pole[0],
    pole[1],
    (5.6 + breath * 0.8 + rift.flare * 4) * g.px,
    g.px * 0.7,
    ROSE,
    0.45 + 0.35 * breath,
    -clock * 0.35,
    8,
  )

  // The way back: echoes of the dial peel off the ring and sink into the pole, sixty ticks each with
  // the hours cut longer, turning with the whirlpool and wobbling as they go.
  if (rift.heat > 0.02) {
    for (let k = 0; k < 4; k++) {
      const cycle = clock / 1.7 + k / 4
      const p = frac(cycle)
      const size = (GATE.r - 6) * (1 - p) ** 1.3
      const cy = pole[1] + GATE.depth * (1 - p)
      const alpha = (1 - p) * Math.min(1, rift.heat)
      const ink = p < 0.45 ? CREAM : p < 0.75 ? GOLDEN : ROSE
      for (let t = 0; t < 60; t++) {
        // Broken ticks, recut every cycle.
        if (hash(t, k + Math.floor(cycle) * 7, 17) < 0.18) continue
        const th = (t * Math.PI) / 30 - Math.PI / 2 + p * 1.2
        const r =
          size *
          (1 +
            0.07 * Math.sin(3 * th + clock * 2.4 + k) +
            0.18 * (noise(th * 1.7 + k * 5, clock * 0.9, 311) - 0.5))
        const hour = t % 5 === 0
        const len = r * (hour ? 0.14 : 0.06)
        const n = Math.max(1, Math.ceil(len / (g.px * 0.8)))
        for (let s = 0; s <= n; s++) {
          // Each tick bends a little along the spiral as it runs inward.
          const rr = r - (len * s) / n
          const a = th + (s / n) * 0.05
          plot(pole[0] + Math.cos(a) * rr, cy + Math.sin(a) * rr, ink, alpha)
          if (hour)
            plot(
              pole[0] + Math.cos(a + g.px / rr) * rr,
              cy + Math.sin(a + g.px / rr) * rr,
              ink,
              alpha,
            )
        }
      }
    }
  }

  for (const c of rift.comets) {
    const fade = Math.min(1, (3.2 - c.age) / 0.6)
    const pts = c.trail
    const n = pts.length / 2
    for (let k = 1; k < n; k++) {
      const ax = pts[2 * k - 2]
      const ay = pts[2 * k - 1]
      const dx = pts[2 * k] - ax
      const dy = pts[2 * k + 1] - ay
      const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / (g.px * 0.8)))
      const w = k / n
      const ink = w > 0.7 ? CREAM : w > 0.35 ? GOLDEN : ROSE
      for (let s = 0; s < steps; s++)
        plot(ax + (dx * s) / steps, ay + (dy * s) / steps, ink, fade * w)
    }
    plot(c.x, c.y, CREAM, fade)
  }
}
