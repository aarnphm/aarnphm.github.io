import type { Sketch } from './sketch.js'
import { add, circle, polar, ribbon, type Poly, type Vec } from './geometry.js'
import { pow, sin } from './math.js'

/**
 * Woodcut motifs. Everything is a black mass with white cut into it — a
 * midrib, a vein, the slit between two petals — because that is what a
 * knife does to a block, and it's what makes a cut read as printed rather
 * than drawn. Motifs check their own footprint and return false when there
 * is no room, so callers can step down a size.
 */

export interface Spine {
  points: Vec[]
  headings: number[]
}

/** A centreline from `base` along `heading`, bending by `curve` radians in total. */
export function spine(
  base: Vec,
  heading: number,
  length: number,
  curve: number,
  steps = 28,
): Spine {
  const points: Vec[] = [base]
  const headings: number[] = [heading]
  let p = base
  for (let i = 1; i <= steps; i++) {
    const h = heading + curve * ((i - 0.5) / steps)
    p = add(p, polar(h, length / steps))
    points.push(p)
    headings.push(heading + curve * (i / steps))
  }
  return { points, headings }
}

function clear(sketch: Sketch, points: Vec[], radii: number[], skip: number): boolean {
  for (let i = skip; i < points.length; i++) {
    if (!sketch.free(points[i], Math.max(0.8, radii[i]))) return false
  }
  return true
}

/** A tapered white slit along a spine: the cut of the graver. */
function slit(sketch: Sketch, s: Spine, from: number, to: number, width: number): void {
  const pts = s.points.slice(from, to + 1)
  if (pts.length < 3 || width < 0.35) return
  sketch.hole(
    ribbon(
      pts,
      pts.map((_, i) => width * sin(Math.PI * (i / (pts.length - 1)))),
    ),
  )
}

/* ------------------------------------------------------------------ leaves */

export type LeafKind = 'lance' | 'acanthus'

export interface LeafSpec {
  base: Vec
  heading: number
  length: number
  /** Total bend of the midrib; the tip curls this way. */
  curve: number
  kind: LeafKind
}

export function leaf(sketch: Sketch, spec: LeafSpec, check = true): boolean {
  const s = spine(spec.base, spec.heading, spec.length, spec.curve)
  const n = s.points.length - 1
  const W = spec.length * (spec.kind === 'acanthus' ? 0.27 : 0.23)
  const left: number[] = []
  const right: number[] = []
  for (let i = 0; i <= n; i++) {
    const t = i / n
    // Full shoulder a third of the way up, a long drawn point.
    const body = W * pow(sin(Math.PI * pow(t, 0.62)), 0.9)
    if (spec.kind === 'acanthus') {
      // Lobes that sweep forward to points, offset on each side.
      const lobe = (u: number) => {
        const f = ((u % 1) + 1) % 1
        return f < 0.78 ? pow(f / 0.78, 0.55) : (1 - f) / 0.22
      }
      const fade = Math.min(1, t * 5) * Math.min(1, (1 - t) * 4)
      left.push(body * (1 - 0.42 * fade * (1 - lobe(t * 3.6 + 0.15))))
      right.push(body * (1 - 0.42 * fade * (1 - lobe(t * 3.6 + 0.6))))
    } else {
      left.push(body)
      right.push(body)
    }
  }
  if (
    check &&
    !clear(
      sketch,
      s.points,
      left.map((l, i) => Math.max(l, right[i])),
      4,
    )
  )
    return false
  sketch.fill(ribbon(s.points, left, right))

  // Midrib, then veins on the larger leaves, kept clear of the midrib so
  // the two cuts never overlap (overlapping holes would print black again).
  const rib = Math.min(W * 0.11, 1.6 + W * 0.04)
  slit(sketch, s, Math.round(n * 0.1), Math.round(n * 0.86), rib)
  if (W > 7) {
    for (const [t, side] of [
      [0.3, 1],
      [0.42, -1],
      [0.55, 1],
      [0.66, -1],
    ] as const) {
      const i = Math.round(t * n)
      const w = side > 0 ? left[i] : right[i]
      const h = s.headings[i] - side * 0.75
      const start = add(s.points[i], polar(s.headings[i] - (side * Math.PI) / 2, rib * 2.2))
      const vein = spine(start, h, w * 0.62, side * 0.25, 8)
      slit(sketch, vein, 0, 8, rib * 0.7)
    }
  }
  sketch.reservePath(
    s.points,
    left.map((l, i) => Math.max(l, right[i])),
  )
  return true
}

/* ----------------------------------------------------------------- flowers */

export type FlowerKind = 'rose' | 'daisy' | 'tulip'

/** A face-on flower: a ring of petals, an inner ring, a seeded heart. */
export function rosette(
  sketch: Sketch,
  center: Vec,
  r: number,
  petals: number,
  rotation: number,
  kind: 'rose' | 'daisy',
  check = true,
): boolean {
  if (check && !sketch.free(center, r)) return false
  // The cuts inside a flower scale with it; a stroke-sized gap would eat a small one.
  const g = Math.min(sketch.gap, Math.max(0.7, r * 0.09))
  const petal = (
    from: number,
    to: number,
    a: number,
    half: number,
    point: number,
    cut: boolean,
  ) => {
    const s = spine(add(center, polar(a, from)), a, to - from, 0, 18)
    const widths = s.points.map((_, i) => {
      const t = i / (s.points.length - 1)
      const round = Math.sqrt(Math.max(0, sin(Math.PI * pow(t, 0.8))))
      const pointed = sin(Math.PI * pow(t, 0.7))
      return half * (round * (1 - point) + pointed * point)
    })
    sketch.fill(ribbon(s.points, widths))
    if (cut && half > 3.6) slit(sketch, s, 3, s.points.length - 4, Math.min(half * 0.2, 1.2))
  }

  if (kind === 'rose') {
    const heart = r * 0.2
    const innerTo = r * 0.52
    const outerFrom = innerTo + g
    const outerHalf = Math.min(r * 0.34, (Math.PI * (outerFrom + r)) / 2 / petals - g / 2)
    const innerHalf = Math.min(r * 0.18, (Math.PI * (heart + innerTo)) / 2 / petals - g / 2)
    for (let k = 0; k < petals; k++) {
      const a = rotation + (k / petals) * Math.PI * 2
      petal(outerFrom - outerHalf * 0.3, r, a, outerHalf, 0.25, true)
      if (innerHalf > 0.8) petal(heart + g, innerTo, a + Math.PI / petals, innerHalf, 0.1, false)
    }
    sketch.fill(circle(center, heart))
    if (heart > 3.2) {
      sketch.hole(circle(center, heart * 0.6))
      sketch.fill(circle(center, heart * 0.3))
    }
  } else {
    const n = petals * 2
    const disc = r * 0.3
    const half = Math.min(r * 0.16, (Math.PI * (disc + r)) / 2 / n - g / 2)
    for (let k = 0; k < n; k++)
      petal(disc + g, r, rotation + (k / n) * Math.PI * 2, half, 0.7, false)
    sketch.fill(circle(center, disc))
    if (disc > 3) {
      sketch.hole(circle(center, disc * 0.68))
      sketch.fill(circle(center, disc * 0.4))
    }
  }
  sketch.reserve(center, r)
  return true
}

/**
 * A side-on tulip opening along `heading`: one solid cup, three points at
 * the lip, and two slits that separate the petals.
 */
export function tulip(
  sketch: Sketch,
  base: Vec,
  heading: number,
  size: number,
  check = true,
): boolean {
  const mid = add(base, polar(heading, size * 0.55))
  if (check && !sketch.free(mid, size * 0.5)) return false
  // Local frame: u along the heading, v across it.
  const at = (u: number, v: number): Vec =>
    add(base, add(polar(heading, u * size), polar(heading + Math.PI / 2, v * size)))
  const cup: Poly = []
  const side = (sgn: number) => {
    const pts: Vec[] = []
    for (let i = 0; i <= 14; i++) {
      const u = i / 14
      // Narrow foot, round belly, slight flare to the outer point.
      const v = 0.08 + 0.36 * sin(Math.PI * Math.min(1, u * 0.78)) + 0.06 * pow(u, 3)
      pts.push(at(u * 0.92, sgn * v))
    }
    return pts
  }
  const l = side(-1)
  const r = side(1)
  cup.push(...l)
  cup.push(at(0.74, -0.2), at(1.04, 0), at(0.74, 0.2))
  cup.push(...r.reverse())
  sketch.fill(cup)
  for (const sgn of [-1, 1]) {
    const s = spine(at(0.3, sgn * 0.13), heading + sgn * 0.05, size * 0.52, -sgn * 0.12, 10)
    slit(sketch, s, 0, 10, Math.max(0.6, size * 0.035))
  }
  // Sepals cupping the foot.
  for (const sgn of [-1, 1]) {
    leaf(
      sketch,
      {
        base: at(0.06, 0),
        heading: heading + sgn * 2.2,
        length: size * 0.34,
        curve: -sgn * 0.6,
        kind: 'lance',
      },
      false,
    )
  }
  sketch.reserve(mid, size * 0.5)
  return true
}

/** Closed bud with a slit and two sepals. */
export function bud(
  sketch: Sketch,
  base: Vec,
  heading: number,
  size: number,
  check = true,
): boolean {
  const s = spine(base, heading, size, 0, 14)
  const widths = s.points.map((_, i) => {
    const t = i / (s.points.length - 1)
    return size * 0.27 * pow(sin(Math.PI * pow(t, 0.6)), 0.85)
  })
  if (check && !clear(sketch, s.points, widths, 3)) return false
  sketch.fill(ribbon(s.points, widths))
  slit(sketch, s, 4, 12, Math.max(0.5, size * 0.035))
  for (const sgn of [-1, 1]) {
    leaf(
      sketch,
      {
        base,
        heading: heading + sgn * 0.75,
        length: size * 0.42,
        curve: -sgn * 0.6,
        kind: 'lance',
      },
      false,
    )
  }
  sketch.reservePath(s.points, widths)
  return true
}

/** Three berries in a triangle. */
export function berries(
  sketch: Sketch,
  at: Vec,
  heading: number,
  r: number,
  check = true,
): boolean {
  const cs = [0, 2.1, -2.1].map(a => add(at, polar(heading + a, r * 1.25)))
  if (check && !cs.every(c => sketch.free(c, r))) return false
  for (const c of cs) sketch.dot(c, r)
  return true
}
