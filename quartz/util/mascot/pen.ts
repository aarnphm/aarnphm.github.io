import { mulberry } from '../../components/scripts/404-pixel'

export type Pt = readonly [number, number]

export const PAPER = '#fffcf0'
export const INK = '#1c1b1a'
export const PINK = '#fdb2a2'
export const SAGE = '#cdd597'

export const poly = (pts: readonly Pt[]) =>
  pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(3)} ${y.toFixed(3)}`).join('') + 'Z'

// A few slow sines with seeded phases, summing to roughly [-1, 1].
export function drift(seed: number, ks: readonly number[]) {
  const rand = mulberry(seed)
  const waves = ks.map(k => ({ k, phase: rand() * 2 * Math.PI, a: 1 / Math.sqrt(k) }))
  const norm = waves.reduce((s, w) => s + w.a, 0)
  return (t: number) => waves.reduce((s, w) => s + w.a * Math.sin(w.k * t + w.phase), 0) / norm
}

// A pen stroke along `pts`: the centre line offset both ways by half of `width(t)`, closed with a
// round end at each tip so the pen's landing and lift read as dots of ink.
export function stroke(pts: readonly Pt[], width: (t: number) => number, ink = INK) {
  const n = pts.length - 1
  const left: Pt[] = []
  const right: Pt[] = []
  for (let i = 0; i <= n; i++) {
    const [ax, ay] = pts[Math.max(0, i - 1)]
    const [bx, by] = pts[Math.min(n, i + 1)]
    const len = Math.hypot(bx - ax, by - ay) || 1
    const h = width(i / n) / 2
    const [x, y] = pts[i]
    left.push([x - ((by - ay) / len) * h, y + ((bx - ax) / len) * h])
    right.push([x + ((by - ay) / len) * h, y - ((bx - ax) / len) * h])
  }
  const tip = ([x, y]: Pt, r: number) =>
    `<circle cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="${r.toFixed(3)}"/>`
  return `<g style="fill:${ink}"><path d="${poly([...left, ...right.reverse()])}"/>${tip(pts[0], width(0) / 2)}${tip(pts[n], width(1) / 2)}</g>`
}

// Pressure along a stroke: thin where the pen lands and lifts, swelling in a few places between.
export const pressure = (seed: number, base: number, swell = 0.45) => {
  const wave = drift(seed, [3, 5, 8])
  return (t: number) =>
    base * (0.45 + 0.55 * Math.sin(Math.PI * t) ** 0.35) * (1 + swell * wave(t * 6))
}

// A circle drawn by hand, radius 1 around the origin: the radius drifts, the circle leans a little
// egg-ward, and the pen runs past where it started so the line overlaps itself near the top.
export function loop(seed: number, wobble = 0.06, overrun = 0.03, n = 220): Pt[] {
  const r = drift(seed, [2, 3, 5, 7])
  const start = -2.1
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n
    const a = start + t * (2 * Math.PI + 0.3)
    const out = 1 + wobble * r(a) + overrun * Math.max(0, t - 0.9) * 10
    return [out * Math.cos(a) * 1.02, out * Math.sin(a) * (0.98 + 0.03 * Math.sin(a))] as const
  })
}

// Points along a polyline, about `step` apart, so a hand wobble has somewhere to live.
export function densify(pts: readonly Pt[], step: number): Pt[] {
  const out: Pt[] = []
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i]
    const [bx, by] = pts[i + 1]
    const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / step))
    for (let s = 0; s < n; s++) out.push([ax + ((bx - ax) * s) / n, ay + ((by - ay) * s) / n])
  }
  out.push(pts[pts.length - 1])
  return out
}

// A polyline as the hand draws it: densified, then nudged sideways by a slow seeded drift.
export function hand(pts: readonly Pt[], step: number, amp: number, seed: number): Pt[] {
  const line = densify(pts, step)
  const wave = drift(seed, [3, 5, 9])
  const n = line.length - 1
  return line.map(([x, y], i) => {
    const [ax, ay] = line[Math.max(0, i - 1)]
    const [bx, by] = line[Math.min(n, i + 1)]
    const len = Math.hypot(bx - ax, by - ay) || 1
    const o = amp * wave((i / n) * 2 * Math.PI)
    return [x - ((by - ay) / len) * o, y + ((bx - ax) / len) * o] as const
  })
}

// A rounded rectangle's outline, clockwise from the top left, closed back on itself.
export function rounded(cx: number, cy: number, w: number, h: number, r: number): Pt[] {
  const corners: [number, number, number][] = [
    [cx + w / 2 - r, cy - h / 2 + r, -Math.PI / 2],
    [cx + w / 2 - r, cy + h / 2 - r, 0],
    [cx - w / 2 + r, cy + h / 2 - r, Math.PI / 2],
    [cx - w / 2 + r, cy - h / 2 + r, Math.PI],
  ]
  const pts: Pt[] = []
  for (const [x, y, a0] of corners)
    for (let i = 0; i <= 6; i++) {
      const a = a0 + (i / 6) * (Math.PI / 2)
      pts.push([x + r * Math.cos(a), y + r * Math.sin(a)])
    }
  pts.push(pts[0])
  return pts
}

// An upright eye: a short stroke that bulges in the middle like a pressed pen.
export function eye(x: number, base: number, ink: string): string {
  const pts: Pt[] = Array.from({ length: 13 }, (_, i) => {
    const t = i / 12
    return [x + 0.025 * t + 0.012 * Math.sin(Math.PI * t), -0.36 + 0.32 * t] as const
  })
  return stroke(pts, t => base * (0.7 + 0.5 * Math.sin(Math.PI * t)), ink)
}

// Where the eyes rest, as in aarnphm's sketch: both shifted left of centre.
export const EYES = [-0.6, 0.24] as const

// A quadratic curve from `a` to `b` pulled toward `c`, as `n + 1` points.
export const curve = (a: Pt, c: Pt, b: Pt, n = 12): Pt[] =>
  Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n
    const u = 1 - t
    return [
      u * u * a[0] + 2 * u * t * c[0] + t * t * b[0],
      u * u * a[1] + 2 * u * t * c[1] + t * t * b[1],
    ] as const
  })

// Thin at both tips, full in the middle: for short marks where seeded pressure would be noise.
export const even = (base: number) => (t: number) => base * (0.7 + 0.4 * Math.sin(Math.PI * t))

// A small hand circle of radius `r` at (x, y).
export const ring = (seed: number, x: number, y: number, r: number): Pt[] =>
  loop(seed, 0.1, 0.1, 36).map(([px, py]) => [x + r * px, y + r * py] as const)

export const part = (name: string, body: string, cls = '') =>
  `<g data-part="${name}"${cls ? ` class="${cls}"` : ''}>${body}</g>`

// A closed band around a centre line, `half(t)` wide to each side, for a ribbon, a loaf or a quill.
export function ribbon(centre: readonly Pt[], half: (t: number) => number): Pt[] {
  const n = centre.length - 1
  const side = (sign: number) =>
    centre.map(([x, y], i) => {
      const [ax, ay] = centre[Math.max(0, i - 1)]
      const [bx, by] = centre[Math.min(n, i + 1)]
      const len = Math.hypot(bx - ax, by - ay) || 1
      const h = sign * half(i / n)
      return [x - ((by - ay) / len) * h, y + ((bx - ax) / len) * h] as const
    })
  return [...side(1), ...side(-1).reverse(), side(1)[0]]
}

// A closed shape in one colour inside a pen line.
export const filled = (
  pts: readonly Pt[],
  fill: string,
  width: (t: number) => number,
  ink: string,
) => `<path d="${poly(pts)}" style="fill:${fill}"/>${stroke(pts, width, ink)}`

export const CRUST = '#fcc192'

// What each drawing of the pet shares: its line width and outfit scale, colours, outline, its pen
// and the mitten every shown hand wears in the snow.
export type PetPen = {
  L: number
  w: number
  paper: string
  ink: string
  pink: string
  outline: Pt[]
  pen: (pts: readonly Pt[], w: (t: number) => number, color?: string) => string
  mitt: (pts: readonly Pt[], side: 1 | -1, seed: number) => string
}
