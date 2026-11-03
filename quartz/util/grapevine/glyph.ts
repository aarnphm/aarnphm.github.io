import ah from './glyphs/a-h.json'
import ip from './glyphs/i-p.json'
import qz from './glyphs/q-z.json'
import { resample } from './vendor/initialis/util.js'

// The capital of EB Garamond Initials' letter layer (Georg Duffner, OFL; the face the default
// drop cap already serves), A to Z as SVG path data in font units (1000/em, y up), dumped with
// fontTools' SVGPathPen. box = [xmin, ymin, xmax, ymax] of the ink.
const GLYPHS: Record<string, { d: string; box: number[] }> = { ...ah, ...ip, ...qz }

export type Pt = [number, number]

export interface Placed {
  /** Path data in font units; `transform` maps it onto the canvas. */
  d: string
  transform: string
  /** Ink outline as closed polygons in canvas units (y down), for the clearance field. */
  rings: Pt[][]
}

// Cap height shared by every letter, so the alphabet keeps one size on the tile (A's box).
const CAP = 832
const ARITY: Record<string, number> = { M: 2, L: 2, H: 1, V: 1, C: 6, Q: 4, Z: 0 }

function* commands(d: string): Generator<[string, number[]]> {
  const tokens = d.match(/[MLHVCQZ]|-?\d*\.?\d+/g) ?? []
  let cmd = ''
  let i = 0
  while (i < tokens.length) {
    if (/[A-Z]/.test(tokens[i])) cmd = tokens[i++]
    const n = ARITY[cmd]
    const args = tokens.slice(i, i + n).map(Number)
    i += n
    yield [cmd, args]
    if (n === 0) cmd = ''
  }
}

function bezier(from: Pt, controls: Pt[], to: Pt, into: Pt[]) {
  const pts = [from, ...controls, to]
  const span = Math.hypot(to[0] - from[0], to[1] - from[1])
  const steps = Math.max(3, Math.ceil(span / 10))
  for (let s = 1; s <= steps; s++) {
    const t = s / steps
    // De Casteljau on 3 or 4 points.
    let level = pts
    while (level.length > 1) {
      const next: Pt[] = []
      for (let k = 0; k + 1 < level.length; k++)
        next.push([
          level[k][0] + (level[k + 1][0] - level[k][0]) * t,
          level[k][1] + (level[k + 1][1] - level[k][1]) * t,
        ])
      level = next
    }
    into.push(level[0])
  }
}

/** Flattens the outline into polygons, in font units. */
function flatten(d: string): Pt[][] {
  const rings: Pt[][] = []
  let ring: Pt[] = []
  let cur: Pt = [0, 0]
  for (const [cmd, a] of commands(d)) {
    if (cmd === 'M') {
      if (ring.length > 2) rings.push(ring)
      ring = [[a[0], a[1]]]
      cur = [a[0], a[1]]
    } else if (cmd === 'L') ring.push((cur = [a[0], a[1]]))
    else if (cmd === 'H') ring.push((cur = [a[0], cur[1]]))
    else if (cmd === 'V') ring.push((cur = [cur[0], a[0]]))
    else if (cmd === 'C') {
      bezier(
        cur,
        [
          [a[0], a[1]],
          [a[2], a[3]],
        ],
        [a[4], a[5]],
        ring,
      )
      cur = [a[4], a[5]]
    } else if (cmd === 'Q') {
      bezier(cur, [[a[0], a[1]]], [a[2], a[3]], ring)
      cur = [a[2], a[3]]
    } else if (cmd === 'Z') {
      if (ring.length > 2) rings.push(ring)
      ring = []
    }
  }
  if (ring.length > 2) rings.push(ring)
  return rings
}

/**
 * Places a capital on a square canvas of `size` units: cap height `scale` of the canvas, the ink
 * box centred horizontally and the cap box centred vertically, so letters sit on one baseline.
 */
export function placeLetter(letter: string, size: number, scale: number): Placed | null {
  const glyph = GLYPHS[letter.toUpperCase()]
  if (!glyph) return null
  const s = (size * scale) / CAP
  const [xmin, , xmax] = glyph.box
  const tx = size / 2 - ((xmin + xmax) / 2) * s
  // Baseline: the cap box (0..CAP above it) is centred on the canvas.
  const baseline = size / 2 + (CAP * s) / 2
  const map = ([x, y]: Pt): Pt => [tx + x * s, baseline - y * s]
  return {
    d: glyph.d,
    transform: `matrix(${s.toFixed(4)} 0 0 ${(-s).toFixed(4)} ${tx.toFixed(1)} ${baseline.toFixed(1)})`,
    rings: flatten(glyph.d).map(ring => ring.map(map)),
  }
}

export interface Tip {
  x: number
  y: number
  nx: number
  ny: number
}

function inside(rings: Pt[][], x: number, y: number): boolean {
  let wind = 0
  for (const ring of rings) {
    for (let i = 0, n = ring.length; i < n; i++) {
      const [x0, y0] = ring[i]
      const [x1, y1] = ring[(i + 1) % n]
      if ((y0 <= y && y1 > y) || (y1 <= y && y0 > y)) {
        const t = (y - y0) / (y1 - y0)
        if (x < x0 + t * (x1 - x0)) wind += y1 > y0 ? 1 : -1
      }
    }
  }
  return wind !== 0
}

export function letterTips(rings: Pt[][], count: number, rnd: () => number): Tip[] {
  const cand: (Tip & { score: number })[] = []
  for (const ring of rings) {
    const pts = resample(ring, 6, true)
    const n = pts.length
    if (n < 8) continue
    for (let i = 0; i < n; i++) {
      const a = pts[(i - 3 + n) % n]
      const p = pts[i]
      const b = pts[(i + 3) % n]
      const t1 = Math.atan2(p[1] - a[1], p[0] - a[0])
      const t2 = Math.atan2(b[1] - p[1], b[0] - p[0])
      let turn = t2 - t1
      while (turn > Math.PI) turn -= Math.PI * 2
      while (turn < -Math.PI) turn += Math.PI * 2
      const tx = b[0] - a[0]
      const ty = b[1] - a[1]
      const tl = Math.hypot(tx, ty) || 1
      let nx = ty / tl
      let ny = -tx / tl
      if (inside(rings, p[0] + nx * 6, p[1] + ny * 6)) {
        nx = -nx
        ny = -ny
      }
      const mx = (a[0] + b[0]) / 2 - p[0]
      const my = (a[1] + b[1]) / 2 - p[1]
      if (mx * nx + my * ny > -0.5) continue
      cand.push({ x: p[0], y: p[1], nx, ny, score: Math.abs(turn) + rnd() * 0.05 })
    }
  }
  cand.sort((u, v) => v.score - u.score)
  const out: Tip[] = []
  for (const c of cand) {
    if (out.length >= count) break
    if (out.some(o => Math.hypot(o.x - c.x, o.y - c.y) < 90)) continue
    out.push(c)
  }
  return out
}
