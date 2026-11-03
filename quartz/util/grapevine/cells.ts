import type { Pt } from './glyph'
import { offsetPolygon } from './vendor/initialis/util.js'

export function cellPebbles(
  width: number,
  height: number,
  clear: (x: number, y: number) => number,
  rnd: () => number,
  scale: number,
  size: number,
  gap: number,
): Pt[][] {
  const pts: Pt[] = []
  let row = 0
  for (let y = 0; y < height; y += scale * 0.87, row++) {
    for (let x = (row % 2) * (scale / 2); x < width; x += scale) {
      const px = x + (rnd() - 0.5) * scale * 0.8
      const py = y + (rnd() - 0.5) * scale * 0.8
      if (clear(px, py) > gap + scale * 0.25) pts.push([px, py])
    }
  }
  const hc = scale * 1.2
  const grid = new Map<number, Pt[]>()
  for (const p of pts) {
    const k = ((p[0] / hc) | 0) * 100003 + ((p[1] / hc) | 0)
    const bucket = grid.get(k)
    if (bucket) bucket.push(p)
    else grid.set(k, [p])
  }
  const inset = Math.max(0.5, size * 0.5)
  const polys: Pt[][] = []
  for (const p of pts) {
    let poly: Pt[] = [
      [p[0] - scale * 1.6, p[1] - scale * 1.6],
      [p[0] + scale * 1.6, p[1] - scale * 1.6],
      [p[0] + scale * 1.6, p[1] + scale * 1.6],
      [p[0] - scale * 1.6, p[1] + scale * 1.6],
    ]
    const i0 = (p[0] / hc) | 0
    const j0 = (p[1] / hc) | 0
    let dead = false
    for (let i = i0 - 2; i <= i0 + 2 && !dead; i++) {
      for (let j = j0 - 2; j <= j0 + 2 && !dead; j++) {
        for (const q of grid.get(i * 100003 + j) ?? []) {
          if (q === p) continue
          const mx = (p[0] + q[0]) / 2
          const my = (p[1] + q[1]) / 2
          const dx = q[0] - p[0]
          const dy = q[1] - p[1]
          const side = (v: Pt) => (v[0] - mx) * dx + (v[1] - my) * dy
          const next: Pt[] = []
          for (let k = 0; k < poly.length; k++) {
            const A = poly[k]
            const B = poly[(k + 1) % poly.length]
            const sa = side(A)
            const sb = side(B)
            if (sa <= 0) next.push(A)
            if (sa <= 0 !== sb <= 0) {
              const t = sa / (sa - sb)
              next.push([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t])
            }
          }
          poly = next
          if (poly.length < 3) {
            dead = true
            break
          }
        }
      }
    }
    if (dead || poly.length < 3) continue
    poly = offsetPolygon(poly, inset).map((p): Pt => [p[0], p[1]])
    for (const v of poly) {
      for (let k = 0; k < 10 && clear(v[0], v[1]) < gap; k++) {
        v[0] += (p[0] - v[0]) * 0.25
        v[1] += (p[1] - v[1]) * 0.25
      }
    }
    for (let it = 0; it < 2; it++) {
      const sm: Pt[] = []
      for (let k = 0; k < poly.length; k++) {
        const A = poly[k]
        const B = poly[(k + 1) % poly.length]
        sm.push(
          [A[0] * 0.75 + B[0] * 0.25, A[1] * 0.75 + B[1] * 0.25],
          [A[0] * 0.25 + B[0] * 0.75, A[1] * 0.25 + B[1] * 0.75],
        )
      }
      poly = sm
    }
    polys.push(poly)
  }
  return polys
}
