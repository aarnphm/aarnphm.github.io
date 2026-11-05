import type { Bench } from './runtime'
import { LEVEL_RADIUS } from './bench-feats'

export type Screen = {
  coords: Float32Array
  xs: Float32Array
  ys: Float32Array
  depth: Float32Array
}

// depth bands for the points: each band × source is one path and one fill, far bands first, so
// a frame is at most 48 fills whatever the level (it was one fill per point, 53k at chunks)
const BANDS = 12
const KINDS = 4
// within a band the many go under the few: pdf (44k chunks), bases, images, then notes, so the
// notes read through the pdf cloud as they did when every point was sorted by depth
const KIND_ORDER = [1, 3, 2, 0]

export function drawBenchPoints(
  ctx: CanvasRenderingContext2D,
  bench: Bench,
  at: Screen,
  shown: (i: number) => boolean,
  colours: string[],
): void {
  const kinds = bench.kinds!
  const n = Math.min(kinds.length, at.xs.length)
  const docs = bench.level !== 'rows'
  const atlas = bench.level === 'feats' ? bench.atlas : null
  const heat = bench.level === 'feats' ? null : bench.heat
  if (heat) return drawBenchHeat(ctx, bench, at, shown, colours, heat, n)
  const radius = new Float32Array(BANDS)
  for (let b = 0; b < BANDS; b++) {
    const t = (b + 0.5) / BANDS
    radius[b] = docs ? 1.2 + t * 2.2 : 0.6 + t * 1.4
  }
  // a chunk dot is 1 to 4 device px across, where an antialiased square reads as the disc
  // does; squares raster in about a third of the time of 53k arcs
  const square = !docs
  const paths: (Path2D | undefined)[] = Array.from({ length: BANDS * KINDS })
  const TAU = Math.PI * 2
  for (let i = 0; i < n; i++) {
    if (!shown(i)) continue
    const t = (at.depth[i] + 1) / 2
    const b = t <= 0 ? 0 : t >= 1 ? BANDS - 1 : (t * BANDS) | 0
    const slot = b * KINDS + kinds[i]
    const path = (paths[slot] ??= new Path2D())
    // a feature's size is its Matryoshka level, eased by depth
    const r = atlas
      ? LEVEL_RADIUS[atlas.lvl[i]] * (0.8 + 0.4 * Math.max(0, Math.min(1, t)))
      : radius[b]
    const x = at.xs[i]
    const y = at.ys[i]
    // same area as the disc: side r·√π
    if (square) path.rect(x - r * 0.886, y - r * 0.886, r * 1.772, r * 1.772)
    else {
      path.moveTo(x + r, y)
      path.arc(x, y, r, 0, TAU)
    }
  }
  for (let b = 0; b < BANDS; b++) {
    ctx.globalAlpha = 0.25 + ((b + 0.5) / BANDS) * 0.65
    for (const k of KIND_ORDER) {
      const path = paths[b * KINDS + k]
      if (!path) continue
      ctx.fillStyle = colours[k] ?? colours[0]
      ctx.fill(path)
    }
  }
}

// heat: the floor in depth bands of ink, the firing points in activation × depth bands of
// the fire colour over it, weak first; 12 + 24 fills whatever the feature
const HEAT_DEPTHS = 6
const HEAT_EDGES = [64, 128, 192]
const HEAT_ALPHA = [0.55, 0.7, 0.85, 1]

function drawBenchHeat(
  ctx: CanvasRenderingContext2D,
  bench: Bench,
  at: Screen,
  shown: (i: number) => boolean,
  colours: string[],
  heat: Uint8Array,
  n: number,
): void {
  const docs = bench.level === 'docs'
  const grow = docs ? [1.2, 1.6, 2.0, 2.5] : [1.1, 1.4, 1.8, 2.3]
  const floor: (Path2D | undefined)[] = Array.from({ length: BANDS })
  const fire: (Path2D | undefined)[] = Array.from({ length: 4 * HEAT_DEPTHS })
  const TAU = Math.PI * 2
  for (let i = 0; i < n; i++) {
    if (!shown(i)) continue
    const t = Math.max(0, Math.min(0.999, (at.depth[i] + 1) / 2))
    const base = docs ? 1.2 + t * 2.2 : 0.6 + t * 1.4
    const x = at.xs[i]
    const y = at.ys[i]
    const a = heat[i] ?? 0
    if (!a) {
      const path = (floor[(t * BANDS) | 0] ??= new Path2D())
      if (docs) {
        path.moveTo(x + base, y)
        path.arc(x, y, base, 0, TAU)
      } else path.rect(x - base * 0.886, y - base * 0.886, base * 1.772, base * 1.772)
      continue
    }
    let band = 0
    while (band < 3 && a > HEAT_EDGES[band]) band++
    const path = (fire[band * HEAT_DEPTHS + ((t * HEAT_DEPTHS) | 0)] ??= new Path2D())
    const r = base * grow[band]
    path.moveTo(x + r, y)
    path.arc(x, y, r, 0, TAU)
  }
  ctx.fillStyle = colours[0]
  floor.forEach((path, b) => {
    if (!path) return
    ctx.globalAlpha = 0.06 + ((b + 0.5) / BANDS) * 0.12
    ctx.fill(path)
  })
  ctx.fillStyle = colours[4]
  fire.forEach((path, slot) => {
    if (!path) return
    const band = Math.floor(slot / HEAT_DEPTHS)
    const d = slot % HEAT_DEPTHS
    ctx.globalAlpha = (0.25 + ((d + 0.5) / HEAT_DEPTHS) * 0.65) * HEAT_ALPHA[band]
    ctx.fill(path)
  })
}
