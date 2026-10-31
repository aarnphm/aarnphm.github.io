import type { Sketch } from './sketch.js'
import { dist, ribbon, type Vec } from './geometry.js'
import { atan2, cos, pow, sin } from './math.js'

/**
 * Tendrils: curves defined by how their heading changes along their length.
 *
 *   heading(t) = h0 + bend·t + turn·t^(tight+1)
 *
 * `bend` sweeps evenly; `turn` piles up near the end, so curvature keeps
 * increasing and the line winds into a volute — the defining gesture of
 * scrollwork. Small turns give flowering stems, large ones give curls, and
 * a high `tight` with a modest turn gives the sudden hook of a thorn.
 */

export interface TendrilSpec {
  start: Vec
  heading: number
  length: number
  bend: number
  turn: number
  tight: number
  /** Half-width at t ∈ [0, 1]. */
  width: (t: number) => number
}

export interface Tendril {
  spec: TendrilSpec
  points: Vec[]
  headings: number[]
  widths: number[]
  owner: number
}

export function trace(spec: TendrilSpec, step = 1.2): Omit<Tendril, 'owner'> {
  const n = Math.max(8, Math.ceil(spec.length / step))
  const ds = spec.length / n
  const points: Vec[] = [spec.start]
  const headings: number[] = []
  const widths: number[] = []
  const heading = (t: number) => spec.heading + spec.bend * t + spec.turn * pow(t, spec.tight + 1)

  let p = spec.start
  for (let i = 0; i <= n; i++) {
    const t = i / n
    headings.push(heading(t))
    widths.push(Math.max(0, spec.width(t)))
    if (i === n) break
    // Midpoint rule keeps tight curls from drifting open.
    const h = heading((i + 0.5) / n)
    p = { x: p.x + cos(h) * ds, y: p.y + sin(h) * ds }
    points.push(p)
  }
  return { spec, points, headings, widths }
}

export interface FitOptions {
  /** Skip checks within this distance of the start (where it leaves its parent). */
  launch?: number
  /** Ink belonging to this owner never blocks (e.g. the parent stem). */
  ignoreOwner?: number
}

export function fits(sketch: Sketch, t: Omit<Tendril, 'owner'>, opts: FitOptions = {}): boolean {
  const launch = opts.launch ?? 0
  const start = t.points[0]
  let leftLaunch = launch <= 0
  for (let i = 0; i < t.points.length; i++) {
    const p = t.points[i]
    if (!leftLaunch) {
      if (dist(p, start) < launch) continue
      leftLaunch = true
    }
    if (!sketch.free(p, Math.max(0.6, t.widths[i]), opts.ignoreOwner)) return false
  }
  return true
}

export function commit(sketch: Sketch, t: Omit<Tendril, 'owner'>, roundStart = false): Tendril {
  const owner = sketch.newOwner()
  sketch.fill(ribbon(t.points, t.widths, t.widths, roundStart))
  sketch.reservePath(t.points, t.widths, owner)
  return { ...t, owner }
}

export interface GrowOptions extends FitOptions {
  roundStart?: boolean
  /** Length multipliers to try, longest first. */
  attempts?: number[]
  /** Also try with the curl reversed if nothing fits. */
  flip?: boolean
}

/**
 * Grows the longest version of `spec` that fits. Shrinking keeps the curl
 * intact (it's parametrised by t, not arc length), so a cramped spot gets a
 * small complete volute rather than a truncated stump.
 */
export function grow(sketch: Sketch, spec: TendrilSpec, opts: GrowOptions = {}): Tendril | null {
  const attempts = opts.attempts ?? [1, 0.8, 0.64, 0.5]
  const variants = opts.flip ? [spec, { ...spec, turn: -spec.turn, bend: -spec.bend }] : [spec]
  for (const variant of variants) {
    for (const k of attempts) {
      const traced = trace({ ...variant, length: variant.length * k })
      if (fits(sketch, traced, opts)) return commit(sketch, traced, opts.roundStart)
    }
  }
  return null
}

/** Treat an arbitrary polyline (a sine stem, a rule) as a tendril others can branch from. */
export function fromPolyline(
  sketch: Sketch,
  points: Vec[],
  widths: number[],
  draw = true,
): Tendril {
  const headings = points.map((_p, i) => {
    const a = points[Math.max(0, i - 1)]
    const b = points[Math.min(points.length - 1, i + 1)]
    return atan2(b.y - a.y, b.x - a.x)
  })
  const owner = sketch.newOwner()
  if (draw) sketch.fill(ribbon(points, widths))
  sketch.reservePath(points, widths, owner)
  const spec: TendrilSpec = {
    start: points[0],
    heading: headings[0],
    length: 0,
    bend: 0,
    turn: 0,
    tight: 1,
    width: () => widths[0],
  }
  return { spec, points, headings, widths, owner }
}

/** Point, heading and half-width at parameter t along a grown tendril. */
export function at(
  t: Tendril,
  u: number,
): { p: Vec; heading: number; width: number; index: number } {
  const index = Math.min(t.points.length - 1, Math.max(0, Math.round(u * (t.points.length - 1))))
  return { p: t.points[index], heading: t.headings[index], width: t.widths[index], index }
}

/** Arc length of a tendril's polyline. */
export function arcLength(t: { points: Vec[] }): number {
  let total = 0
  for (let i = 1; i < t.points.length; i++) total += dist(t.points[i - 1], t.points[i])
  return total
}
