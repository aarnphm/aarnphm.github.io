import type { Pt } from './glyph'
import { edt } from './vendor/initialis/util.js'

// The blocked mask the ornament field starts from: cells on the field's grid (node i at i * cell,
// as Field.sample reads it) that fall outside the frame, inside the letter, or within `gap` of it.

export interface Grid {
  nx: number
  ny: number
  cell: number
}

function rings(mask: Uint8Array, rings: Pt[][], { nx, ny, cell }: Grid) {
  const edges: [number, number, number, number][] = []
  for (const ring of rings)
    for (let k = 0; k < ring.length; k++) {
      const [x0, y0] = ring[k]
      const [x1, y1] = ring[(k + 1) % ring.length]
      if (y0 !== y1) edges.push([x0, y0, x1, y1])
    }
  for (let j = 0; j < ny; j++) {
    const y = j * cell
    const hits: [number, number][] = []
    for (const [x0, y0, x1, y1] of edges) {
      if (y < Math.min(y0, y1) || y >= Math.max(y0, y1)) continue
      hits.push([x0 + ((y - y0) * (x1 - x0)) / (y1 - y0), y1 > y0 ? 1 : -1])
    }
    hits.sort((a, b) => a[0] - b[0])
    let winding = 0
    for (let h = 0; h + 1 < hits.length; h++) {
      winding += hits[h][1]
      if (winding === 0) continue
      const from = Math.max(0, Math.ceil(hits[h][0] / cell))
      const to = Math.min(nx - 1, Math.floor(hits[h + 1][0] / cell))
      for (let i = from; i <= to; i++) mask[j * nx + i] = 1
    }
  }
}

/** Grows a mask outward by `gap` units (exact distance transform). */
function dilate(mask: Uint8Array, gap: number, { nx, ny, cell }: Grid) {
  const f = new Float64Array(nx * ny)
  for (let k = 0; k < f.length; k++) f[k] = mask[k] ? 0 : 1e20
  edt(f, nx, ny)
  const limit = (gap / cell) ** 2
  for (let k = 0; k < f.length; k++) if (f[k] <= limit) mask[k] = 1
}

export function blockedMask(
  width: number,
  height: number,
  grid: Grid,
  inset: number,
  letter: Pt[][],
  gap: number,
  region: { halo?: number; embolden?: number } = {},
): Uint8Array {
  const { nx, ny, cell } = grid
  const mask = new Uint8Array(nx * ny)
  if (letter.length) {
    const ink = new Uint8Array(nx * ny)
    rings(ink, letter, grid)
    const { halo = 0, embolden = 0 } = region
    if (embolden > 0) dilate(ink, embolden, grid)
    mask.set(ink)
    if (gap > 0) dilate(mask, gap, grid)
    if (halo > 0) {
      const f = new Float64Array(nx * ny)
      for (let k = 0; k < f.length; k++) f[k] = ink[k] ? 0 : 1e20
      edt(f, nx, ny)
      const limit = (halo / cell) ** 2
      for (let k = 0; k < f.length; k++) if (f[k] > limit) mask[k] = 1
    }
  }
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const x = i * cell
      const y = j * cell
      if (x < inset || y < inset || x > width - inset || y > height - inset) mask[j * nx + i] = 1
    }
  return mask
}
