import type { Element, ElementContent } from 'hast'
import { s } from 'hastscript'
import { LEVEL_RADIUS } from '../atlas'

/**
 * The bench as an SVG poster: the first canvas frame, drawn at build time with the canvas's own
 * banding (latent-client/bench-canvas-points.ts): 12 depth bands × 4 sources, far bands first,
 * pdf, base, image, then notes within a band. A band is one `<path>` of zero-length subpaths
 * whose round (docs, features) or square (chunks) caps are the dots, so the stroke width is the
 * dot and `vector-effect: non-scaling-stroke` keeps dots at css px as the canvas does. Colour
 * comes from a class per source, so the page's theme paints it and dark mode needs no second
 * poster. The square viewBox under `xMidYMid meet` puts every point where the canvas, which fits
 * the cloud to the shorter side, will draw it.
 */

export const POSTER = 420

const BANDS = 12
const KIND_ORDER = [1, 3, 2, 0]
const HEAT_DEPTHS = 6
const HEAT_EDGES = [64, 128, 192]
const HEAT_ALPHA = [0.55, 0.7, 0.85, 1]
/** at chunk level a poster keeps at most this many points, by a fixed stride */
export const POSTER_POINTS = 6000

export interface PosterView {
  yaw: number
  pitch: number
  zoom: number
  /** pan as a fraction of the box, as the share payload carries it */
  panX: number
  panY: number
}

export interface PosterInput {
  view: PosterView
  level: 'docs' | 'rows' | 'feats'
  coords: Float32Array
  extent: number
  kinds: Uint8Array
  /** Matryoshka level per point (features level) */
  lvl?: Uint8Array
  /** a bit per source kind left undrawn */
  hidden: number
  /** activation per point, 0..255, of the picked feature (docs and chunks) */
  heat?: Uint8Array | null
  /** the ringed point, or -1 */
  mark: number
  /** the marked feature's edges at the features level */
  edges?: { near: number[]; parent: number; children: number[] }
  /** axis names for the x, y, z letters' titles */
  axes: [string, string, string]
}

type Screen = { xs: Float32Array; ys: Float32Array; depth: Float32Array }

const round = (v: number) => Math.round(v * 10) / 10

/** The canvas's projectAll at a square box of POSTER units. */
function project(input: PosterInput): Screen {
  const { coords: c, extent, view } = input
  const n = c.length / 3
  const xs = new Float32Array(n)
  const ys = new Float32Array(n)
  const depth = new Float32Array(n)
  const w = POSTER
  const s = (w * 0.42 * view.zoom) / extent
  const cy = Math.cos(view.yaw)
  const sy = Math.sin(view.yaw)
  const cp = Math.cos(view.pitch)
  const sp = Math.sin(view.pitch)
  const ox = w / 2 + view.panX * w
  const oy = w / 2 + view.panY * w
  const inv = 1 / extent
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    const x = c[j]
    const y = c[j + 1]
    const z = c[j + 2]
    const rz = x * sy + z * cy
    xs[i] = ox + (x * cy - z * sy) * s
    ys[i] = oy - (y * cp - rz * sp) * s
    depth[i] = (y * sp + rz * cp) * inv
  }
  return { xs, ys, depth }
}

function axisPoint(view: PosterView, extent: number, x: number, y: number, z: number) {
  const w = POSTER
  const s = (w * 0.42 * view.zoom) / extent
  const cy = Math.cos(view.yaw)
  const sy = Math.sin(view.yaw)
  const cp = Math.cos(view.pitch)
  const sp = Math.sin(view.pitch)
  const rx = x * cy - z * sy
  const rz = x * sy + z * cy
  const ry = y * cp - rz * sp
  return [w / 2 + view.panX * w + rx * s, w / 2 + view.panY * w - ry * s]
}

class Bands {
  private d = new Map<string, string[]>()
  add(key: string, x: number, y: number) {
    let parts = this.d.get(key)
    if (!parts) this.d.set(key, (parts = []))
    parts.push(`M${round(x)} ${round(y)}h0`)
  }
  get(key: string): string | undefined {
    return this.d.get(key)?.join('')
  }
}

const inBox = (x: number, y: number) => x > -8 && y > -8 && x < POSTER + 8 && y < POSTER + 8

export function benchPoster(input: PosterInput): Element {
  const { level, kinds, hidden } = input
  const at = project(input)
  const n = Math.min(kinds.length, at.xs.length)
  const stride = level === 'rows' && n > POSTER_POINTS ? Math.ceil(n / POSTER_POINTS) : 1
  const shown = (i: number) => !((hidden >> kinds[i]) & 1) && inBox(at.xs[i], at.ys[i])
  const paths: ElementContent[] = []
  const heat = level === 'feats' ? null : input.heat
  if (heat) {
    // the floor in depth bands of ink, the firing points in activation × depth bands over it
    const docs = level === 'docs'
    const grow = docs ? [1.2, 1.6, 2.0, 2.5] : [1.1, 1.4, 1.8, 2.3]
    const floor = new Bands()
    const fire = new Bands()
    const floorR = new Map<string, number>()
    const fireR = new Map<string, number>()
    for (let i = 0; i < n; i++) {
      const a = heat[i] ?? 0
      if (!a && i % stride) continue
      if (!shown(i)) continue
      const t = Math.max(0, Math.min(0.999, (at.depth[i] + 1) / 2))
      const base = docs ? 1.2 + t * 2.2 : 0.6 + t * 1.4
      if (!a) {
        const b = (t * BANDS) | 0
        floor.add(String(b), at.xs[i], at.ys[i])
        floorR.set(String(b), docs ? base : base * 0.886)
        continue
      }
      let band = 0
      while (band < 3 && a > HEAT_EDGES[band]) band++
      const key = `${band}:${(t * HEAT_DEPTHS) | 0}`
      fire.add(key, at.xs[i], at.ys[i])
      fireR.set(key, base * grow[band])
    }
    for (let b = 0; b < BANDS; b++) {
      const d = floor.get(String(b))
      if (!d) continue
      paths.push(
        s('path', {
          class: `k0${docs ? '' : ' sq'}`,
          d,
          'stroke-width': round(2 * floorR.get(String(b))!),
          'stroke-opacity': (0.06 + ((b + 0.5) / BANDS) * 0.12).toFixed(3),
        }),
      )
    }
    for (let band = 0; band < 4; band++)
      for (let dz = 0; dz < HEAT_DEPTHS; dz++) {
        const key = `${band}:${dz}`
        const d = fire.get(key)
        if (!d) continue
        paths.push(
          s('path', {
            class: 'fire',
            d,
            'stroke-width': round(2 * fireR.get(key)!),
            'stroke-opacity': (
              (0.25 + ((dz + 0.5) / HEAT_DEPTHS) * 0.65) *
              HEAT_ALPHA[band]
            ).toFixed(3),
          }),
        )
      }
  } else {
    const docs = level !== 'rows'
    const feats = level === 'feats'
    const bands = new Bands()
    const widths = new Map<string, number>()
    for (let i = 0; i < n; i += stride) {
      if (!shown(i)) continue
      const t = (at.depth[i] + 1) / 2
      const b = t <= 0 ? 0 : t >= 1 ? BANDS - 1 : (t * BANDS) | 0
      // a feature's size is its Matryoshka level, eased by depth; its own sub-band per level
      const lvl = feats && input.lvl ? input.lvl[i] : 0
      const r = feats
        ? LEVEL_RADIUS[lvl] * (0.8 + 0.4 * Math.max(0, Math.min(1, (b + 0.5) / BANDS)))
        : docs
          ? 1.2 + ((b + 0.5) / BANDS) * 2.2
          : 0.6 + ((b + 0.5) / BANDS) * 1.4
      const key = `${b}:${kinds[i]}:${lvl}`
      bands.add(key, at.xs[i], at.ys[i])
      // chunks are equal-area squares: side r·√π
      widths.set(key, docs ? 2 * r : r * 1.772)
    }
    for (let b = 0; b < BANDS; b++)
      for (const k of KIND_ORDER)
        for (let lvl = 0; lvl < (feats ? 3 : 1); lvl++) {
          const key = `${b}:${k}:${lvl}`
          const d = bands.get(key)
          if (!d) continue
          paths.push(
            s('path', {
              class: `k${k}${docs ? '' : ' sq'}`,
              d,
              'stroke-width': round(widths.get(key)!),
              'stroke-opacity': (0.25 + ((b + 0.5) / BANDS) * 0.65).toFixed(3),
            }),
          )
        }
  }

  // the three axes through the origin, ticks at the half and full extent, the letter past the end
  const { view, extent } = input
  const axes: ElementContent[] = []
  const [ox, oy] = axisPoint(view, extent, 0, 0, 0)
  const ends: [number, number, number][] = [
    [extent, 0, 0],
    [0, extent, 0],
    [0, 0, extent],
  ]
  ends.forEach(([ex, ey, ez], slot) => {
    const pt = (t: number) => axisPoint(view, extent, ex * t, ey * t, ez * t)
    const [ax, ay] = pt(-1)
    const [bx, by] = pt(1)
    axes.push(s('line', { x1: round(ax), y1: round(ay), x2: round(bx), y2: round(by) }))
    const len = Math.hypot(bx - ax, by - ay)
    if (len > 8) {
      const nx = -(by - ay) / len
      const ny = (bx - ax) / len
      const ticks = [-1, -0.5, 0.5, 1].map(t => {
        const [tx, ty] = pt(t)
        return `M${round(tx - nx * 3)} ${round(ty - ny * 3)}L${round(tx + nx * 3)} ${round(ty + ny * 3)}`
      })
      axes.push(s('path', { d: ticks.join('') }))
    }
    const dx = bx - ox
    const dy = by - oy
    const dl = Math.hypot(dx, dy) || 1
    axes.push(
      s('text', { x: round(bx + (dx / dl) * 9), y: round(by + (dy / dl) * 9) }, [
        s('title', input.axes[slot]),
        'xyz'[slot],
      ]),
    )
  })

  const marks: ElementContent[] = []
  const { mark } = input
  if (mark >= 0 && mark < n && !((hidden >> kinds[mark]) & 1)) {
    const edges = input.edges
    if (level === 'feats' && edges) {
      const line = (b: number) =>
        b >= 0 && b < n && shown(b)
          ? `M${round(at.xs[mark])} ${round(at.ys[mark])}L${round(at.xs[b])} ${round(at.ys[b])}`
          : ''
      const near = edges.near.map(line).join('')
      if (near) marks.push(s('path', { class: 'edge near', d: near }))
      const parent = line(edges.parent)
      if (parent) marks.push(s('path', { class: 'edge parent', d: parent }))
      const kids = edges.children.map(line).join('')
      if (kids) marks.push(s('path', { class: 'edge child', d: kids }))
    }
    marks.push(s('circle', { class: 'mark', cx: round(at.xs[mark]), cy: round(at.ys[mark]), r: 6 }))
  }

  return s(
    'svg.lat-embed-poster',
    {
      viewBox: `0 0 ${POSTER} ${POSTER}`,
      preserveAspectRatio: 'xMidYMid meet',
      ariaHidden: 'true',
      focusable: 'false',
    },
    [s('g.axes', axes), s('g.pts', paths), s('g.marks', marks)],
  )
}

/** Points shown on the poster at this level (chunks are sampled by a stride). */
export function posterCount(level: PosterInput['level'], n: number): number {
  return level === 'rows' && n > POSTER_POINTS ? Math.ceil(n / Math.ceil(n / POSTER_POINTS)) : n
}
