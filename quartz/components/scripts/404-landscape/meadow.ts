import type { Palette } from '../../../util/scene-palette'
import type { Season } from '../../../util/visitor-world'
import { GATE } from '../404-gate'
import {
  clampCol,
  dither,
  fbm,
  frac,
  hash,
  pick,
  put,
  ramp,
  row,
  smooth,
  type Grid,
} from '../404-pixel'
import { paintCypress } from './cypress'
import { facing, hillTop, lean, spur } from './geography'
import { brush, seeds, type Angle } from './raster'
import { paintRock, rimeAt, rimeOver, type Relief, type Rock } from './relief'

const HILL_CYPRESSES = [
  { x: 420, h: 250, w: 46 },
  { x: 1195, h: 215, w: 40 },
]

// The island's crest line, how each column of it faces the light (already signed and scaled for
// where the light is and how hard it falls), the flow of its strokes, how its spurs lean, the
// outcrops on them, and the season's flowers and leaves.
export type Meadow = {
  top: Float32Array
  lit: Float32Array
  ang: Float32Array
  tilt: Float32Array
  rocks: Rock[]
  crest: number
  relief: Relief
  bloom: number
  autumn: number
  snow: number
}

// The spur under a point of the island, as on the ranges but broader, being nearer: its phase, and
// how the plane there is lit, +1 square to the light and -1 turned away, softening toward the water.
function islandSpur(tilt: number, x: number, down: number) {
  return spur(x - tilt * 110 * Math.tanh(down / 100), down, 41, 0.0065)
}
// Under cloud the light comes from the whole sky, and the gullies, open to less of it than the
// spurs, stay the darker for it.
function islandFacet(m: Meadow, i: number, x: number, y: number) {
  const down = y - m.top[i]
  const u = islandSpur(m.tilt[i], x, down)
  const { side, contrast } = m.relief
  const open = 1 - 4 * Math.min(frac(u), 1 - frac(u))
  return (
    (facing(u) * side * contrast + open * 0.5 * (1 - contrast)) * (1 - 0.6 * smooth(0, 220, down))
  )
}

// Outcrops break out along the spur crests, clear of the gate and of everyone's footing: the
// tablet, the camp and the strand, the cypresses, the cat and the fallen stone.
function islandRocks(water: number): Rock[] {
  const top = (x: number) => hillTop(x, water)
  const clear = (x: number, y: number) =>
    Math.hypot(x - GATE.x, y - GATE.y) > 300 &&
    !(Math.abs(x - 805) < 160 && Math.abs(y - 786) < 60) &&
    y < 812 &&
    !(x > 480 && x < 720 && y > 780) &&
    Math.abs(x - 420) > 45 &&
    Math.abs(x - 1195) > 45 &&
    Math.abs(x - 1250) > 110 &&
    Math.abs(x - 1136) > 40 &&
    !(Math.abs(x - 366) < 80 && Math.abs(y - 772) < 50) &&
    !(Math.abs(x - 548) < 50 && Math.abs(y - 740) < 50)
  const rocks: Rock[] = []
  for (let x = 170; x <= 1440; x += 23)
    for (let down = 30; down <= 200; down += 17) {
      const y = top(x) + down
      const u = islandSpur(lean(top, x), x, down)
      if (Math.min(frac(u), 1 - frac(u)) > 0.05 || !clear(x, y)) continue
      if (rocks.some(r => Math.hypot(r.x - x, r.base - y) < 150)) continue
      const w = 32 + hash(x, down, 5) * 26
      const peak = (hash(x, down, 6) - 0.5) * 0.9
      rocks.push({
        x,
        base: y,
        w,
        h: w * (0.5 + hash(x, down, 7) * 0.2),
        peak,
        shoulder: peak + (peak < 0 ? 0.5 : -0.5),
        mid: true,
      })
    }
  return rocks.sort((a, b) => a.base - b.base)
}

export function meadow(g: Grid, water: number, relief: Relief, s: Season): Meadow {
  const top = Float32Array.from(g.sx, x => hillTop(x, water))
  let crest = Infinity
  for (const y of top) crest = Math.min(crest, y)
  const k = relief.side * relief.contrast
  return {
    top,
    lit: Float32Array.from(g.sx, x => ((hillTop(x + 8, water) - hillTop(x - 8, water)) / 16) * k),
    ang: Float32Array.from(g.sx, x =>
      Math.atan((hillTop(x + 24, water) - hillTop(x - 24, water)) / 48),
    ),
    tilt: Float32Array.from(g.sx, x => lean(xx => hillTop(xx, water), x)),
    rocks: islandRocks(water),
    crest,
    relief,
    bloom: s.bloom,
    autumn: s.autumn,
    snow: s.ground,
  }
}

// How much hoarfrost is left at a point of the island, read off its tone, which already carries the
// spur planes, the broad mottle of the turf and the ring's shadow: the sun clears the brightest
// ground first and the shade last. The lake gives up its warmth all night, so the bank stays green.
function rimed(r: Relief, tone: number, y: number, water: number) {
  return rimeAt(r, tone) * (1 - smooth(water - 44, water - 12, y))
}

function hillTone(m: Meadow, i: number, x: number, y: number, water: number) {
  // Weighted as on the ranges: the spur planes carry the form, the broad mottle stays under them.
  let t = 0.5 + Math.max(-1, Math.min(1, m.lit[i] * 2.2)) * 0.14 + islandFacet(m, i, x, y) * 0.44
  t += (fbm(x * 0.012, y * 0.004, 61) - 0.5) * 0.26 + (fbm(x * 0.018, y * 0.07, 62) - 0.5) * 0.14
  // The turf round the foot of the ring sits in its shadow, cast down and away from the light.
  const rim = Math.hypot(x - GATE.x + 10 * m.relief.side, y - GATE.y - 6) - GATE.r - GATE.ring
  if (y > GATE.y) t -= 0.4 * (1 - smooth(0, 30, rim))
  return t - smooth(water - 16, water, y) * 0.2
}

export function paintHillBase(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  m: Meadow,
) {
  const { bw, bh, sx, sy } = g
  const white = rimeOver(pal.hill, pal.snow)
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    if (y > water + g.px) break
    for (let i = 0; i < bw; i++) {
      if (y < m.top[i]) continue
      const k = (j * bw + i) * 4
      if (y - m.top[i] < g.px) put(d, k, m.lit[i] > 0.08 ? pal.gold : pal.contour[2])
      else {
        const t = hillTone(m, i, sx[i], y, water)
        put(d, k, ramp(dither(rimed(m.relief, t, y, water), i, j) ? white : pal.hill, t, i, j))
      }
    }
  }
}

// The field in short dashes that follow the slope and flatten toward the bank; a travelling
// wave leans them as the wind crosses, with the odd poppy, cornflower and ochre dab, and the fallen
// leaves in autumn. The outcrops stand up out of it.
export function paintField(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  m: Meadow,
  base: Uint8ClampedArray,
  clock: number,
  gust: number,
) {
  d.set(base)
  const x0 = g.sx[0]
  const y0 = g.sy[0]
  const white = rimeOver(pal.hill, pal.snow)
  const sway = 0.45 + 0.55 * Math.min(1, gust)
  let jit = 0
  const angle: Angle = (bx, by) => {
    const i = clampCol(g, bx)
    const x = x0 + bx * g.px
    const y = y0 + by * g.px
    const wave = 0.36 * Math.sin(0.012 * x - 2.1 * clock + 0.035 * y) * sway
    const contour = m.ang[i] * (1 - 0.5 * smooth(0, 120, y - m.top[i]))
    return (contour + wave + jit) * (1 - smooth(water - 34, water - 8, y))
  }
  seeds(g, 3, 0, [row(g, m.crest), row(g, water)], 151, (bx, by, h, k) => {
    const i = clampCol(g, bx)
    const x = x0 + bx * g.px
    const y = y0 + by * g.px
    if (y < m.top[i] + g.px || y > water - g.px * 0.5) return
    // Behind the ring: never seen.
    if (Math.hypot(x - GATE.x, y - GATE.y) < GATE.r + GATE.ring - 10) return
    const t = hillTone(m, i, x, y, water)
    // Flowers come and go with the season; the grass goes to seed in the autumn, and the leaves
    // come down on it. Under snow the wind scours the spur crests back to the dead grass, as it
    // bares the rock along the ranges' spurs.
    const roll = frac(k * 5.31)
    const u = m.snow > 0 ? islandSpur(m.tilt[i], x, y - m.top[i]) : 0.5
    const scoured = Math.min(frac(u), 1 - frac(u)) < 0.07 * (1.3 - m.snow) + 0.03 * frac(k * 2.9)
    // Rimed blades, and among them the odd crystal catching the light where the sun is at the frost.
    const rime = rimed(m.relief, t, y, water)
    const frosted = frac(k * 4.13) < rime
    const ink = frosted
      ? frac(k * 6.07) < 0.12 * smooth(0.2, 0.6, m.relief.thaw) * (1 - m.relief.thaw)
        ? pal.snow[3]
        : pick(white, t + 0.12, h)
      : scoured
        ? pick(pal.dun, t - 0.15, h)
        : t > 0.42 && roll < 0.016 * m.bloom
          ? pal.poppy
          : t > 0.42 && roll < 0.03 * m.bloom
            ? pal.cornflower
            : t > 0.5 && roll < 0.03 * m.bloom + 0.03 + 0.05 * m.autumn
              ? pal.ochre
              : roll > 1 - 0.08 * m.autumn
                ? pal.autumn[1 + Math.floor(frac(k * 7.7) * 3)]
                : pick(pal.hill, t, h)
    jit = (frac(k * 9.73) - 0.5) * 0.7
    brush(d, g, bx, by, 3 + Math.floor(frac(k * 3.37) * 3), frac(k * 11.3) < 0.65, ink, angle)
  })
  for (const rock of m.rocks) paintRock(d, g, pal, rock, m.relief)
  HILL_CYPRESSES.forEach((c, n) =>
    paintCypress(
      d,
      g,
      { ...c, base: hillTop(c.x, water) + 6 },
      pal.leaf.slice(0, 3),
      500 + n,
      Math.sin(clock * 1.1 + n * 2) * 0.12 * sway,
    ),
  )
}
