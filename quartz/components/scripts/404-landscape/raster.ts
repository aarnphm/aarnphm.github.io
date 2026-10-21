import { frac, grid, hash, put, type Grid, type RGB } from '../404-pixel'

// An animated canvas keeps one ImageData and repaints into it every tick.
export type Layer = { g: Grid; ctx: CanvasRenderingContext2D; img: ImageData }

export function layer(canvas: HTMLCanvasElement, prev?: Layer | null): Layer | null {
  const g = grid(canvas, prev?.g)
  if (!g) return null
  if (prev && prev.g.bw === g.bw && prev.g.bh === g.bh) {
    prev.g = g
    prev.img.data.fill(0)
    return prev
  }
  const ctx = prev?.ctx ?? canvas.getContext('2d')!
  return { g, ctx, img: ctx.createImageData(g.bw, g.bh) }
}

export type Angle = (x: number, y: number) => number
export type Keep = (i: number, j: number) => boolean

// Curved brush stroke (Hertzmann 1998): march one buffer pixel at a time along the flow in both
// directions from the seed, two pixels wide through the body and one at the tips. Flow angles
// are orientations, so each step flips to stay continuous with the last.
export function brush(
  d: Uint8ClampedArray,
  g: Grid,
  x: number,
  y: number,
  len: number,
  wide: boolean,
  c: RGB,
  angle: Angle,
  keep?: Keep,
) {
  const { bw, bh } = g
  const dab = (px: number, py: number) => {
    const i = Math.round(px)
    const j = Math.round(py)
    if (i < 0 || j < 0 || i >= bw || j >= bh || (keep && !keep(i, j))) return
    put(d, (j * bw + i) * 4, c)
  }
  for (const dir of [1, -1]) {
    const n = dir > 0 ? Math.ceil(len / 2) : Math.floor(len / 2) + 1
    let cx = x
    let cy = y
    let lx = 0
    let ly = 0
    for (let s = 0; s < n; s++) {
      const a = angle(cx, cy)
      let dx = Math.cos(a) * dir
      let dy = Math.sin(a) * dir
      if (s > 0 && dx * lx + dy * ly < 0) {
        dx = -dx
        dy = -dy
      }
      if (dir > 0 || s > 0) {
        dab(cx, cy)
        if (wide && s < n - 1) dab(cx - dy, cy + dx)
      }
      lx = dx
      ly = dy
      cx += dx
      cy += dy
    }
  }
}

// Jittered lattice of stroke seeds riding a medium that has drifted `shift` buffer pixels, so a
// stroke keeps its place in the medium, and its dither threshold, while it travels. Two
// interleaved passes let later strokes overlap earlier ones without a raster grain.
export function seeds(
  g: Grid,
  cell: number,
  shift: number,
  rows: [number, number],
  seed: number,
  visit: (x: number, y: number, h: number, k: number) => void,
) {
  const c0 = Math.floor(-shift / cell) - 1
  const c1 = Math.ceil((g.bw - shift) / cell) + 1
  const r0 = Math.max(0, Math.floor(rows[0] / cell) - 1)
  const r1 = Math.min(Math.ceil(g.bh / cell), Math.ceil(rows[1] / cell) + 1)
  for (const pass of [0, 1])
    for (let cj = r0; cj <= r1; cj++)
      for (let ci = c0; ci <= c1; ci++) {
        const h = hash(ci, cj, seed)
        if ((h < 0.5 ? 0 : 1) !== pass) continue
        visit(
          ci * cell + hash(ci, cj, seed + 1) * cell + shift,
          cj * cell + hash(ci, cj, seed + 2) * cell,
          frac(h * 2),
          hash(ci, cj, seed + 3),
        )
      }
}
