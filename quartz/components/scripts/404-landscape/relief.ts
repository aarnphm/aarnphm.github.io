import type { Palette } from '../../../util/scene-palette'
import { col, hash, mix, put, ramp, row, smooth, type Grid, type RGB } from '../404-pixel'

// How the land is lit: from which side (1 when the light comes from the right), how hard (cloud
// flattens it), and how far down the ranges the snow reaches, in scene units.
export type Relief = {
  side: number
  contrast: number
  snowline: number
  dark: boolean
  frost: number
  thaw: number
}

// Whether hoarfrost holds on ground of this tone, which runs from about 0 in shade to 1 square to
// the light (the brightest turf a little past it). A marginal night frosts only the coldest shade,
// and the sun clears the brightest ground first, so both move the frost's edge down the tones
// instead of thinning it everywhere.
export function rimeAt(r: Relief, tone: number) {
  if (r.frost <= 0) return 0
  const edge = -0.4 + 1.7 * r.frost * (1 - r.thaw)
  return smooth(edge + 0.08, edge - 0.08, tone)
}
// Frosted grass keeps the green under its rime; a three-stop ramp takes the upper snow stops.
export const rimeOver = (stops: RGB[], snow: RGB[]) =>
  stops.map((c, k) => mix(c, snow[k + snow.length - stops.length], 0.7))

// An outcrop, as a low block with a summit (`peak`, across it from -1 to 1) and a lower shoulder.
export type Rock = {
  x: number
  base: number
  w: number
  h: number
  peak: number
  shoulder: number
  mid: boolean
}

// An outcrop breaking the face: a flank turned toward the light and one turned away, split at the
// summit, with a lit edge along the sunny flank's top, a darker course where it beds into the
// slope, and snow on top above the snowline. Its top steps unevenly from column to column.
export function paintRock(d: Uint8ClampedArray, g: Grid, pal: Palette, rock: Rock, relief: Relief) {
  const { bw, bh, sx, sy } = g
  const { side, contrast } = relief
  const half = rock.w / 2
  const i0 = Math.max(0, col(g, rock.x - half))
  const i1 = Math.min(bw - 1, col(g, rock.x + half))
  const j0 = Math.max(0, row(g, rock.base - rock.h * 1.2))
  const j1 = Math.min(bh - 1, row(g, rock.base))
  const tent = (t: number, at: number, h: number) =>
    Math.max(0, t < at ? ((t + 1) / (at + 1)) * h : ((1 - t) / (1 - at)) * h)
  const snowy = rock.base - rock.h < relief.snowline
  const edge = g.px / rock.h
  const seed = Math.round(rock.x)
  for (let i = i0; i <= i1; i++) {
    const t = (sx[i] - rock.x) / half
    if (Math.abs(t) >= 1) continue
    const top =
      Math.max(tent(t, rock.peak, 1), tent(t, rock.shoulder, 0.6)) +
      (hash(i, seed, 7) - 0.5) * 2 * edge
    const lit = (t < rock.peak ? -side : side) * contrast
    for (let j = j0; j <= j1; j++) {
      const up = (rock.base - sy[j]) / rock.h
      if (up >= top || up < 0) continue
      const k = (j * bw + i) * 4
      if (snowy && up > top - 0.4) put(d, k, ramp(pal.snow, 0.6 + lit * 0.35, i, j))
      else if (lit > 0 && top - up < edge) put(d, k, pal.rock[3])
      else put(d, k, ramp(pal.rock, 0.4 + lit * 0.45 - (up < edge ? 0.2 : 0), i, j))
    }
  }
}
