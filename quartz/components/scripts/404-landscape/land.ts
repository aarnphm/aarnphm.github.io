import type { Palette } from '../../../util/scene-palette'
import { nearCourse, paintCourse, type CourseInk } from '../404-downhill'
import { onTerrace, TERRACES } from '../404-herd'
import {
  BAYER,
  clampCol,
  col,
  fbm,
  frac,
  INK,
  line,
  mix,
  mulberry,
  pick,
  put,
  row,
  smooth,
  type Grid,
} from '../404-pixel'
import { paintCypress, type Cypress } from './cypress'
import { CAMP, facing, farRidge, FOLK, glacier, lean, midRidge, spurAt, TRAIL } from './geography'
import { CREAM } from './light-marks'
import { brush, seeds, type Angle, type Keep } from './raster'
import { paintRock, rimeAt, rimeOver, type Relief, type Rock } from './relief'

// The ranges, laid in strokes that follow each ridge's profile and relax toward level deeper in
// the face, the way the Alpilles are combed in the Saint-Rémy canvases, then modelled by the spurs
// and capped with snow down to the snowline. The sky stays clear.
export function paintLand(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  pines: Cypress[],
  relief: Relief,
) {
  const { bw, bh, sx, sy } = g
  const { side, contrast, snowline } = relief
  const far = Float32Array.from(sx, farRidge)
  const mid = Float32Array.from(sx, midRidge)
  const slope = (f: (x: number) => number, x: number, h: number) => (f(x + h) - f(x - h)) / (2 * h)
  const farLit = Float32Array.from(sx, x => slope(farRidge, x, 8) * side * contrast)
  const midLit = Float32Array.from(sx, x => slope(midRidge, x, 8) * side * contrast)
  const farAng = Float32Array.from(sx, x => Math.atan(slope(farRidge, x, 24)))
  const midAng = Float32Array.from(sx, x => Math.atan(slope(midRidge, x, 24)))
  // The plane under a point: +1 on a spur's flank square to the light, -1 on one turned away,
  // fading down the face as the spurs soften into the foothills; and how near a spur's crest it is,
  // in phase.
  const farTilt = Float32Array.from(sx, x => lean(farRidge, x))
  const midTilt = Float32Array.from(sx, x => lean(midRidge, x))
  const facet = (i: number, down: number, back: boolean) => {
    const u = spurAt(back, sx[i], down, back ? farTilt[i] : midTilt[i])
    const strength = 1 - 0.5 * smooth(0, 300, down)
    return [facing(u) * side * strength * contrast, Math.min(frac(u), 1 - frac(u))] as const
  }
  const farTone = (i: number, y: number, lit: number) =>
    0.45 +
    Math.max(-1, Math.min(1, farLit[i] * 1.6)) * 0.12 +
    lit * 0.34 +
    (fbm(sx[i] * 0.009, y * 0.003, 53) - 0.5) * 0.22 +
    smooth(30, 360, y - far[i]) * 0.35
  const midTone = (i: number, y: number, lit: number) =>
    0.5 +
    Math.max(-1, Math.min(1, midLit[i] * 1.6)) * 0.12 +
    lit * 0.4 +
    (fbm(sx[i] * 0.012, y * 0.0035, 41) - 0.5) * 0.26 +
    smooth(40, 320, y - mid[i]) * 0.22
  // Snow lies where it is cold enough, reaches lower in the shade, and slides off the spur crests.
  const snowy = (x: number, y: number, down: number, lit: number, crest: number) => {
    const edge = snowline + (fbm(x * 0.02, 5.5, 71, 2) - 0.5) * 80 - lit * 30
    if (y >= edge && !glacier(x, y)) return 0
    return crest < 0.045 && down > 6 && down < 170 ? 2 : 1
  }
  const snowTone = (lit: number, y: number, top: number) =>
    0.62 + lit * 0.36 - smooth(0, 200, y - top) * 0.12
  // Below the snow the near range's grass takes the night's frost as the island's does.
  const rimed = rimeOver(pal.mid, pal.snow)
  // What the face shows at a point: its ink for strokes and the flat underpainting alike.
  // `shift` lightens or darkens it, for the steps trodden into the grass.
  const surface = (i: number, y: number, h: number, back: boolean, shift = 0) => {
    const top = back ? far[i] : mid[i]
    const down = y - top
    const [lit, crest] = facet(i, down, back)
    const cover = snowy(sx[i], y, down, lit, crest)
    if (cover === 2) return pick(pal.rock, 0.35 + lit * 0.35 + shift, h)
    if (cover === 1) return pick(pal.snow, snowTone(lit, y, top) + shift, h)
    if (back) return pick(pal.far, farTone(i, y, lit) + shift, h)
    // Read off the face's tone as on the island, less the haze that lightens it toward the foot,
    // where the cold air pools.
    const frosted =
      rimeAt(relief, midTone(i, y, lit) - 0.35 * smooth(40, 320, down)) > frac(h * 7.31)
    return pick(frosted ? rimed : pal.mid, midTone(i, y, lit) + shift, h)
  }

  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    for (let i = 0; i < bw; i++) {
      const k = (j * bw + i) * 4
      const t = BAYER[(j & 3) * 4 + (i & 3)]
      // Each crest is drawn as a contour, so where the mid ridge passes in front of the far one the
      // overlap shows as a line. Slopes that fall away toward the light catch gold on the crest.
      if (y >= mid[i]) {
        if (y - mid[i] < g.px) put(d, k, midLit[i] > 0.15 ? pal.gold : pal.contour[1])
        else put(d, k, surface(i, y, t, false))
      } else if (y >= far[i]) put(d, k, y - far[i] < g.px ? pal.contour[0] : surface(i, y, t, true))
    }
  }

  const y0 = sy[0]
  let ridge = far
  let ang = farAng
  let jit = 0
  const angle: Angle = (bx, by) => {
    const i = clampCol(g, bx)
    return ang[i] * (1 - 0.6 * smooth(0, 260, y0 + by * g.px - ridge[i])) + jit
  }
  const inFar: Keep = (i, j) => sy[j] >= far[i] + g.px && sy[j] < mid[i]
  const inMid: Keep = (i, j) => sy[j] >= mid[i] + g.px
  let top = Infinity
  for (const y of far) top = Math.min(top, y)
  seeds(g, 3, 0, [row(g, top), bh], 141, (bx, by, h, k) => {
    const i = clampCol(g, bx)
    const y = y0 + by * g.px
    if (y < far[i] + g.px * 1.5 || y > mid[i] - g.px) return
    jit = (k - 0.5) * 0.4
    const ink = surface(i, y, h, true)
    brush(d, g, bx, by, 4 + Math.floor(frac(k * 9.1) * 4), false, ink, angle, inFar)
  })
  ridge = mid
  ang = midAng
  seeds(g, 3, 0, [row(g, top), bh], 143, (bx, by, h, k) => {
    const i = clampCol(g, bx)
    const y = y0 + by * g.px
    if (y < mid[i] + g.px * 1.5 || y > 900) return
    jit = (k - 0.5) * 0.5
    const ink = surface(i, y, h, false)
    brush(d, g, bx, by, 4 + Math.floor(frac(k * 9.1) * 5), frac(k * 4.3) < 0.4, ink, angle, inMid)
  })
  // The herd's terracettes (404-herd.ts), laid as broken strokes along each step: the riser one or
  // two stops darker than the slope and the tread over it lighter, each stroke choosing its stops
  // once, so a step reads as a run of short shadows and never as a dotted rule.
  TERRACES.forEach((t, n) => {
    const r = mulberry(t.seed)
    for (let x = t.x0; x < t.x1; x += 4 + r() * 8) {
      const h = r()
      for (const end = Math.min(t.x1, x + 16 + r() * 32); x <= end; x += g.px) {
        const y = onTerrace(n, x)
        if (y === null || nearCourse(x, y, 8)) continue
        const i = col(g, x)
        const j = row(g, y)
        if (i < 0 || i >= bw || j < 0 || j + 1 >= bh) continue
        const fade = smooth(0, 14, Math.min(x - t.x0, t.x1 - x))
        put(d, (j * bw + i) * 4, surface(i, y, h, false, 0.4 * fade))
        put(d, ((j + 1) * bw + i) * 4, surface(i, y + g.px, h, false, -0.75 * fade))
      }
    }
  })
  for (const rock of ROCKS) paintRock(d, g, pal, rock, relief)
  paintCourse(d, g, courseInk(pal), snowline)
  // The trail: a pale tread with the bench's lip a pixel under it.
  const tread = mix(pal.mid[2], CREAM, relief.dark ? 0.12 : 0.5)
  const lip = mix(pal.mid[0], INK, 0.25)
  for (let k = 1; k < TRAIL.length; k++) {
    const [i0, j0, i1, j1] = [
      col(g, TRAIL[k - 1][0]),
      row(g, TRAIL[k - 1][1]),
      col(g, TRAIL[k][0]),
      row(g, TRAIL[k][1]),
    ]
    line(d, g, i0, j0 + 1, i1, j1 + 1, lip)
    line(d, g, i0, j0, i1, j1, tread)
  }
  pines.forEach((c, n) => paintCypress(d, g, c, pal.pine, 300 + n))
}

// Boulders and outcrops on both ranges, clear of the trail, the people and the lift.
const ROCKS: Rock[] = (() => {
  const r = mulberry(909)
  const rocks: Rock[] = []
  const nearTrail = (x: number, y: number) =>
    TRAIL.some(([tx, ty], k) => {
      if (k === 0) return false
      const [ax, ay] = TRAIL[k - 1]
      const len2 = (tx - ax) ** 2 + (ty - ay) ** 2
      const u = Math.max(0, Math.min(1, ((x - ax) * (tx - ax) + (y - ay) * (ty - ay)) / len2))
      return Math.hypot(x - ax - (tx - ax) * u, y - ay - (ty - ay) * u) < 22
    })
  for (let n = 0; n < 1200 && rocks.length < 34; n++) {
    const mid = r() < 0.62
    const x = -40 + r() * 1680
    const down = mid ? 12 + r() * 170 : 8 + r() * 80
    // Outcrops break out along the spur crests, where the rock is nearest the surface.
    const ridge = mid ? midRidge : farRidge
    const u = spurAt(!mid, x, down, lean(ridge, x))
    if (Math.min(frac(u), 1 - frac(u)) > 0.06) continue
    const big = r() < 0.3
    const w = (mid ? 28 + r() * 22 : 16 + r() * 12) * (big ? 1.6 : 1)
    const base = ridge(x) + down
    const peak = (r() - 0.5) * 0.9
    const rock = {
      x,
      base,
      w,
      h: w * (0.45 + r() * 0.2),
      peak,
      shoulder: peak + (peak < 0 ? 0.5 : -0.5) + (r() - 0.5) * 0.3,
      mid,
    }
    if (!mid && base > midRidge(x) - 4) continue
    if (nearTrail(x, base) || FOLK.some(f => Math.abs(f.x - x) < 20) || Math.abs(x - CAMP) < 24)
      continue
    if (!mid && x > 1300 && x < 1500) continue
    rocks.push(rock)
  }
  // Nearer outcrops overlap the ones above them. None sits on the downhill line on the left hill.
  return rocks
    .filter(rock => !rock.mid || !nearCourse(rock.x, rock.base, rock.w / 2 + 10))
    .sort((a, b) => a.base - b.base)
})()

// The downhill line in the ranges' colours: dirt from the bark, timber, and the lift's steel.
function courseInk(pal: Palette): CourseInk {
  return {
    tread: mix(pal.bark[2], pal.ochre, 0.3),
    rut: pal.bark[1],
    berm: mix(pal.ochre, pal.bark[2], 0.5),
    packed: pal.snow[1],
    path: mix(pal.mid[1], pal.bark[2], 0.45),
    wood: pal.bark[2],
    post: pal.bark[0],
    steel: mix(pal.rock[0], INK, 0.4),
    flag: [CREAM, INK],
  }
}
