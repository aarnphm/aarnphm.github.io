import { mulberry } from '../404-pixel'

// Leaf tip: position plus depth toward (+1) or away from (-1) the viewer.
type Tip = [number, number, number]

type Seg = { x0: number; y0: number; x1: number; y1: number; w0: number; w1: number }
type Tree = { segs: Seg[]; tips: Tip[] }

// Probabilistic L-system: X → FX | F[+X]X | F[-X]X | F[+X][-X]X, unrolled until every limb reaches
// the terminal width. The trunk stays bare until it thins, forks conserve cross-sectional area
// (da Vinci), and sides follow a running golden-angle phase so branching never settles into a beat.
export function growTree(
  seed: number,
  x: number,
  y: number,
  lean: number,
  width: number,
  len: number,
) {
  const r = mulberry(seed)
  const segs: Seg[] = []
  const tips: Tip[] = []
  const up = -Math.PI / 2
  const stack = [{ x, y, a: up + lean, w: width, l: len, phase: r() * Math.PI * 2 }]
  while (stack.length) {
    const s = stack.pop()!
    if (s.w < 1.6 || segs.length > 1400) {
      // The phyllotaxis phase that picks a side also says how far the limb reaches toward us.
      tips.push([s.x, s.y, Math.sin(s.phase)])
      continue
    }
    const w1 = s.w * 0.95
    const nx = s.x + Math.cos(s.a) * s.l
    const ny = s.y + Math.sin(s.a) * s.l
    segs.push({ x0: s.x, y0: s.y, x1: nx, y1: ny, w0: s.w, w1 })
    const curl = (a: number, k: number) => a + (up - a) * k + (r() - 0.5) * 0.3
    const roll = r()
    if (roll < 0.3 || s.w > width * 0.8) {
      stack.push({ ...s, x: nx, y: ny, a: curl(s.a, 0.1), w: w1, l: s.l * 0.94 })
      continue
    }
    const both = roll > 0.79
    const phase = s.phase + 2.39996
    const side = Math.cos(phase) >= 0 ? 1 : -1
    const keep = both ? 0.45 + r() * 0.15 : 0.6 + r() * 0.2
    stack.push({ x: nx, y: ny, a: curl(s.a, 0.1), w: w1 * Math.sqrt(keep), l: s.l * 0.9, phase })
    const kids = both ? [side, -side] : [side]
    const kw = w1 * Math.sqrt((1 - keep) / kids.length)
    for (const k of kids)
      stack.push({
        x: nx,
        y: ny,
        a: curl(s.a + k * (0.35 + r() * 0.4), 0.14),
        w: kw,
        l: s.l * (0.7 + r() * 0.2),
        phase: phase + k,
      })
  }
  return { segs, tips } satisfies Tree
}

// Distance from the scene point to the segment, with the signed side for bark lighting.
export function segHit(s: Seg, x: number, y: number): [number, number, number] {
  const dx = s.x1 - s.x0
  const dy = s.y1 - s.y0
  const len2 = dx * dx + dy * dy || 1
  const t = Math.max(0, Math.min(1, ((x - s.x0) * dx + (y - s.y0) * dy) / len2))
  const qx = x - (s.x0 + dx * t)
  const qy = y - (s.y0 + dy * t)
  const side = (dx * qy - dy * qx) / Math.sqrt(len2)
  return [Math.hypot(qx, qy), s.w0 + (s.w1 - s.w0) * t, side]
}

export type Canopy = { trees: Tree[]; roots: number[] }
