import type { Palette } from '../../../util/scene-palette'
import { BAYER, col, dot, hash, pick, row, type Grid } from '../404-pixel'
import { FIRE } from './camp'
import { FLAME } from './light-marks'
import { airAngle, type Air } from './sky'

type Puff = {
  x: number
  y: number
  vx: number
  vy: number
  age: number
  life: number
  seed: number
}
// Rain or snow puts the fire out: no flames, embers or firelight, and the wet logs send up a
// heavier, slower column of smoke from the hearth itself.
export type Hearth = {
  smoke: Puff[]
  embers: Puff[]
  spawn: number
  spark: number
  doused: boolean
}

// Smoke rises off the fire and rides the same air as the sky strokes, so it leans with the wind,
// curls through the cursor's eddies, and spirals into the gate while it is open. Each puff also
// keeps a sideways drift of its own, so the column fans out as it climbs. Embers shoot up, slow,
// and go out.
export function stepHearth(h: Hearth, air: Air, dt: number) {
  h.spawn -= dt
  while (h.spawn <= 0) {
    h.spawn += h.doused ? 0.06 : 0.08
    h.smoke.push({
      x: FIRE.x + (Math.random() - 0.5) * 6,
      y: FIRE.y - (h.doused ? 6 : 26),
      vx: 0,
      vy: h.doused ? -10 : -20,
      age: 0,
      life: (h.doused ? 6.5 : 5) + Math.random() * 2.5,
      seed: Math.random(),
    })
  }
  h.spark -= dt
  if (h.spark <= 0 && !h.doused) {
    h.spark = 0.15 + Math.random() * 0.35
    h.embers.push({
      x: FIRE.x + (Math.random() - 0.5) * 10,
      y: FIRE.y - 20,
      vx: (Math.random() - 0.5) * 24,
      vy: -50 - Math.random() * 40,
      age: 0,
      life: 0.7 + Math.random() * 0.9,
      seed: Math.random(),
    })
  }
  const ease = Math.min(1, dt * 1.5)
  let sw = 0
  for (let i = 0; i < h.smoke.length; i++) {
    const p = h.smoke[i]
    const a = airAngle(air, p.x, p.y)
    // A doused fire sends up a slow column of wet smoke that stands straight for a couple of
    // seconds before the wind takes it; drifting from the start, it reads as a smear on the grass.
    const drift = (10 + 30 * air.pull) * (h.doused ? Math.min(1, p.age / 2.5) : 1)
    const rise = (h.doused ? 18 : 24) * (1 - (0.5 * p.age) / p.life)
    p.vx += (Math.cos(a) * drift + (p.seed - 0.5) * 18 - p.vx) * ease
    p.vy += (Math.sin(a) * drift - rise - p.vy) * ease
    p.x += p.vx * dt
    p.y += p.vy * dt
    p.age += dt
    if (p.age < p.life) h.smoke[sw++] = p
  }
  h.smoke.length = sw
  let ew = 0
  for (let i = 0; i < h.embers.length; i++) {
    const p = h.embers[i]
    p.vy += 30 * dt
    p.vx += (Math.random() - 0.5) * 80 * dt
    p.x += p.vx * dt
    p.y += p.vy * dt
    p.age += dt
    if (p.age < p.life) h.embers[ew++] = p
  }
  h.embers.length = ew
}

// Smoke as short strokes along its own motion that lengthen, widen, pale and thin as they age.
export function paintHearth(d: Uint8ClampedArray, g: Grid, pal: Palette, h: Hearth, clock: number) {
  const beat = Math.floor(clock * 12)
  for (const p of h.embers) {
    if (hash(Math.floor(p.seed * 1e6), beat, 3) < 0.25) continue
    const t = p.age / p.life
    dot(d, g, col(g, p.x), row(g, p.y), t < 0.35 ? FLAME[1] : t < 0.75 ? FLAME[2] : FLAME[3])
  }
  for (const p of h.smoke) {
    const t = p.age / p.life
    const alpha = Math.min(1, p.age / 0.4) * (1 - t) * 0.9
    const c = pick(pal.smoke, 0.15 + t * 0.8, p.seed)
    const len = 1 + Math.floor(t * 4.5)
    const v = Math.hypot(p.vx, p.vy) || 1
    const ux = p.vx / v
    const uy = p.vy / v
    const bx = (p.x - g.sx[0]) / g.px
    const by = (p.y - g.sy[0]) / g.px
    for (let w = t > 0.3 ? 1 : 0; w >= 0; w--)
      for (let s = 0; s < len; s++) {
        const i = Math.round(bx + ux * (s - len / 2) - uy * w)
        const j = Math.round(by + uy * (s - len / 2) + ux * w)
        if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
      }
  }
}
