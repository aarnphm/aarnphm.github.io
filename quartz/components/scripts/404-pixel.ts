// Pixel primitives shared by the 404 scene's canvases: the buffer grid, which maps pixels onto the
// 1600×1000 scene with the same slice fit as the SVG layers, ordered dither, value noise, and the
// painters the ruin's inhabitants are drawn with.

export type RGB = [number, number, number]
export type Pt = [number, number]

export const W = 1600
export const H = 1000
// Every layer overhangs the viewport by this many CSS pixels, so parallax never shows an edge.
const INSET = 48
export const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16)

export const hex = (h: string): RGB => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)) as RGB

export function mulberry(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function hash(x: number, y: number, seed: number) {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 144269504)) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

export function noise(x: number, y: number, seed: number) {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const xf = x - xi
  const yf = y - yi
  const u = xf * xf * (3 - 2 * xf)
  const v = yf * yf * (3 - 2 * yf)
  const a = hash(xi, yi, seed)
  const b = hash(xi + 1, yi, seed)
  const c = hash(xi, yi + 1, seed)
  const d = hash(xi + 1, yi + 1, seed)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

export function fbm(x: number, y: number, seed: number, octaves = 3) {
  let sum = 0
  let amp = 0.5
  let norm = 0
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise(x, y, seed + i * 17)
    norm += amp
    amp *= 0.5
    x *= 2.03
    y *= 2.03
  }
  return sum / norm
}

export const clamp01 = (t: number) => (t < 0 ? 0 : t > 0.9999 ? 0.9999 : t)
export const smooth = (a: number, b: number, t: number) => {
  const x = clamp01((t - a) / (b - a))
  return x * x * (3 - 2 * x)
}
export const frac = (t: number) => t - Math.floor(t)
export const mul = (c: RGB, k: number): RGB => [c[0] * k, c[1] * k, c[2] * k]
export const mix = (a: RGB, b: RGB, k: number): RGB =>
  [0, 1, 2].map(i => a[i] + (b[i] - a[i]) * k) as RGB
// Whether a coverage of `a` lights this pixel under the 4×4 ordered dither.
export const dither = (a: number, i: number, j: number) => a > BAYER[(j & 3) * 4 + (i & 3)]

export function step(n: number, t: number, bx: number, by: number) {
  const p = clamp01(t) * (n - 1)
  const i = Math.floor(p)
  return p - i > BAYER[(by & 3) * 4 + (bx & 3)] ? i + 1 : i
}

// Ordered dither between neighbouring ramp stops.
export function ramp(stops: RGB[], t: number, bx: number, by: number): RGB {
  return stops[step(stops.length, t, bx, by)]
}

// Stroke-level dither: the stroke, not the pixel, chooses between neighbouring stops.
export function pick(stops: RGB[], t: number, threshold: number): RGB {
  const p = clamp01(t) * (stops.length - 1)
  const i = Math.floor(p)
  return stops[p - i > threshold ? i + 1 : i]
}

// Polynomial smooth minimum (Quílez): two distance fields merge into one with a fillet of about k.
export function smin(a: number, b: number, k: number) {
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.min(a, b) - (h * h * k) / 4
}

export type Grid = {
  bw: number
  bh: number
  sx: Float32Array
  sy: Float32Array
  // scene units per buffer pixel
  px: number
  toBuffer: DOMMatrix
  // scene y of the viewport's bottom edge
  bottom: number
}

// Buffer pixel centres mapped into scene units with the same xMidYMid-slice fit as the SVG layers.
export function grid(canvas: HTMLCanvasElement): Grid | null {
  const cw = canvas.clientWidth
  const ch = canvas.clientHeight
  if (!cw || !ch) return null
  const px = Math.max(3, Math.min(6, Math.round(cw / 380)))
  const bw = Math.ceil(cw / px)
  const bh = Math.ceil(ch / px)
  const scale = Math.max(cw / W, ch / H)
  const ox = (cw - W * scale) / 2
  const oy = (ch - H * scale) / 2
  const sx = new Float32Array(bw)
  const sy = new Float32Array(bh)
  for (let i = 0; i < bw; i++) sx[i] = ((i + 0.5) * px - ox) / scale
  for (let j = 0; j < bh; j++) sy[j] = ((j + 0.5) * px - oy) / scale
  canvas.width = bw
  canvas.height = bh
  const toBuffer = new DOMMatrix([scale / px, 0, 0, scale / px, ox / px, oy / px])
  return { bw, bh, sx, sy, px: px / scale, toBuffer, bottom: (ch - INSET - oy) / scale }
}

export const col = (g: Grid, x: number) => Math.round((x - g.sx[0]) / g.px)
export const row = (g: Grid, y: number) => Math.round((y - g.sy[0]) / g.px)
export const clampCol = (g: Grid, i: number) => Math.max(0, Math.min(g.bw - 1, Math.round(i)))

export function put(data: Uint8ClampedArray, k: number, c: RGB | readonly number[], a = 255) {
  data[k] = c[0]
  data[k + 1] = c[1]
  data[k + 2] = c[2]
  data[k + 3] = a
}

export function dot(d: Uint8ClampedArray, g: Grid, i: number, j: number, c: RGB) {
  if (i >= 0 && j >= 0 && i < g.bw && j < g.bh) put(d, (j * g.bw + i) * 4, c)
}

// A one-pixel line between buffer pixels, eight-connected.
export function line(
  d: Uint8ClampedArray,
  g: Grid,
  i0: number,
  j0: number,
  i1: number,
  j1: number,
  c: RGB,
) {
  const n = Math.max(Math.abs(i1 - i0), Math.abs(j1 - j0), 1)
  for (let s = 0; s <= n; s++)
    dot(d, g, Math.round(i0 + ((i1 - i0) * s) / n), Math.round(j0 + ((j1 - j0) * s) / n), c)
}

// A sprite from rows of letters, `.` for clear, stamped with its top-left pixel at (i0, j0).
export function stamp(
  d: Uint8ClampedArray,
  g: Grid,
  rows: readonly string[],
  i0: number,
  j0: number,
  flip: boolean,
  ink: (ch: string) => RGB | null,
) {
  const w = rows[0].length
  rows.forEach((text, r) => {
    for (let c = 0; c < w; c++) {
      const ch = text[flip ? w - 1 - c : c]
      if (ch === '.') continue
      const color = ink(ch)
      if (color) dot(d, g, i0 + c, j0 + r, color)
    }
  })
}

export type Tones = { ink: RGB; light: RGB; plate: RGB; shade: RGB }
export type Blob = { i0: number; j0: number; w: number; h: number; mask: Uint8Array }

// A shape from a signed distance field (negative inside), printed the way the ruin is: an ink
// keyline wherever it meets anything that is not itself, a two-pixel rim of light on the edges
// facing the sun (up and to the right), a pixel of shade on the edges away from it, and a flat
// plate between. Painting blobs back to front outlines each one where it overlaps the last.
export function paintBlob(
  d: Uint8ClampedArray,
  g: Grid,
  field: (x: number, y: number) => number,
  box: readonly [number, number, number, number],
  tones: Tones,
  keep?: (x: number, y: number) => boolean,
): Blob | null {
  const i0 = Math.max(0, col(g, box[0]) - 1)
  const i1 = Math.min(g.bw - 1, col(g, box[2]) + 1)
  const j0 = Math.max(0, row(g, box[1]) - 1)
  const j1 = Math.min(g.bh - 1, row(g, box[3]) + 1)
  if (i1 < i0 || j1 < j0) return null
  const w = i1 - i0 + 1
  const h = j1 - j0 + 1
  const mask = new Uint8Array(w * h)
  let any = false
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      const x = g.sx[i]
      const y = g.sy[j]
      if ((keep && !keep(x, y)) || field(x, y) >= 0) continue
      mask[(j - j0) * w + i - i0] = 1
      any = true
    }
  if (!any) return null
  const at = (i: number, j: number) =>
    i >= i0 && i <= i1 && j >= j0 && j <= j1 && mask[(j - j0) * w + i - i0] === 1
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      if (!at(i, j)) continue
      const c =
        !at(i - 1, j) || !at(i + 1, j) || !at(i, j - 1) || !at(i, j + 1)
          ? tones.ink
          : !at(i + 1, j - 1) || !at(i + 2, j) || !at(i, j - 2) || !at(i + 2, j - 2)
            ? tones.light
            : !at(i - 1, j + 1)
              ? tones.shade
              : tones.plate
      put(d, (j * g.bw + i) * 4, c)
    }
  return { i0, j0, w, h, mask }
}

// Pixels of a painted blob that sit clear of its rims, for flowers, faces and the like.
export function interior(b: Blob, visit: (i: number, j: number) => void) {
  const { i0, j0, w, h, mask } = b
  for (let j = 2; j < h - 2; j++)
    for (let i = 2; i < w - 2; i++) {
      const p = j * w + i
      if (mask[p] && mask[p - 2] && mask[p + 2] && mask[p - 2 * w] && mask[p + 2 * w])
        visit(i0 + i, j0 + j)
    }
}

// Distance from p to the segment ab.
export function segment(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax
  const dy = by - ay
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)))
  return Math.hypot(px - ax - dx * t, py - ay - dy * t)
}

// Signed distance to the triangle abc (Quílez), negative inside.
export function triangle(px: number, py: number, a: Pt, b: Pt, c: Pt) {
  const d = Math.min(
    segment(px, py, a[0], a[1], b[0], b[1]),
    segment(px, py, b[0], b[1], c[0], c[1]),
    segment(px, py, c[0], c[1], a[0], a[1]),
  )
  const side = (p: Pt, q: Pt) => (q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])
  const s1 = side(a, b)
  const s2 = side(b, c)
  const s3 = side(c, a)
  const inside = (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)
  return inside ? -d : d
}
