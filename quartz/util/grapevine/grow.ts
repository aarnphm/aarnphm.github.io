import type { Disc, Field } from './field'
import type { Pt } from './glyph'

// Growth after azlen/initialis (MIT, Azlen Elza): stems step forward through the clearance
// field, curl more as they age, branch to the outer side, carry leaves, and end in a tendril
// or a bud. Everything draws from one seeded generator, so a page and a letter always grow the
// same vine.

export type Rng = () => number

export function rng(seed: string): Rng {
  let state = 2166136261
  for (const char of seed) state = Math.imul(state ^ char.charCodeAt(0), 16777619)
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 4294967296
  }
}

export interface Stem {
  id: number
  pts: Pt[]
  w0: number
  w1: number
  depth: number
}

export interface Leaf {
  x: number
  y: number
  /** Direction the blade points, radians. */
  angle: number
  size: number
  /** Where a filler leaf's stalk starts, when it has one. */
  stalk?: Pt
}

export interface Tendril {
  pts: Pt[]
  w: number
}

export interface Grapes {
  x: number
  y: number
  size: number
  stalk?: Pt
}

export interface Dot {
  x: number
  y: number
  r: number
}

export interface Items {
  stems: Stem[]
  leaves: Leaf[]
  tendrils: Tendril[]
  grapes: Grapes[]
  dots: Dot[]
}

export interface Options {
  step: number
  margin: number
  width: number
  taper: number
  curl: number
  turns: number
  wander: number
  length: number
  branch: number
  branchAngle: number
  thin: number
  depth: number
  leafEvery: number
  leafSize: number
  leafAngle: number
  tendril: number
}

export type Steer = (p: Pt, heading: number, travelled: number) => number

const TAU = Math.PI * 2
const FIT = [220, 160, 120, 90, 65, 48, 35, 25]
const MIN_LEAF = 26

export const empty = (): Items => ({ stems: [], leaves: [], tendrils: [], grapes: [], dots: [] })

export function noise(next: Rng): (t: number) => number {
  const f1 = 0.03 + next() * 0.03
  const f2 = 0.08 + next() * 0.07
  const p1 = next() * TAU
  const p2 = next() * TAU
  return t => 0.6 * Math.sin(t * f1 + p1) + 0.4 * Math.sin(t * f2 + p2)
}

export class Grower {
  private nextId = 1
  constructor(
    readonly field: Field,
    readonly next: Rng,
    readonly items: Items,
    readonly opts: Options,
  ) {}

  /** Grows a stem and, after it settles, its branches. Returns the stem or null if it was too short. */
  stem(start: Pt, heading: number, w0: number, length: number, depth: number, steer?: Steer) {
    const { field, next, opts } = this
    const id = this.nextId++
    const pts: Pt[] = [start]
    const wander = noise(next)
    const side = next() < 0.5 ? -1 : 1
    let p = start
    let travelled = 0
    let turned = 0
    let leafAt = opts.leafEvery * (0.45 + next() * 0.5)
    let leafSide = next() < 0.5 ? -1 : 1
    let branchAt = opts.branch * (0.6 + next() * 0.8)
    const branches: { p: Pt; heading: number; w: number }[] = []
    const slots: { p: Pt; angle: number; idx: number }[] = []
    const recent =
      (i: number): ((d: Disc) => boolean) =>
      d =>
        d.owner === id && d.idx > i - 12
    let w = w0

    for (let i = 0; travelled < length; i++) {
      const age = travelled / length
      w = w0 * (1 - opts.taper * age)
      let desired: number
      if (steer) desired = steer(p, heading, travelled)
      else {
        const fitR = this.fit(p, heading, side, recent(i))
        const k = (side * opts.curl * (0.12 + 0.88 * age * age)) / fitR
        desired = heading + k * opts.step + opts.wander * wander(travelled) * 0.04
      }
      let chosen: number | null = null
      let q: Pt = p
      for (let c = 0; c < 15; c++) {
        const dev = Math.ceil(c / 2) * 0.075 * (c % 2 ? 1 : -1)
        const h = desired + dev
        q = [p[0] + Math.cos(h) * opts.step, p[1] + Math.sin(h) * opts.step]
        if (field.clearance(q[0], q[1], recent(i), w / 2 + opts.margin) >= w / 2 + opts.margin) {
          chosen = h
          break
        }
      }
      if (chosen === null) break
      turned += Math.abs(chosen - heading)
      heading = chosen
      p = q
      pts.push(p)
      travelled += opts.step
      // Stamp two steps back so the stem never blocks its own next step.
      if (i % 2 === 1 && pts.length > 3) {
        const back = pts[pts.length - 3]
        field.stamp(back[0], back[1], w / 2 + opts.margin * 0.5 + opts.step * 0.5, id, i)
      }
      if (turned > opts.turns * TAU + 0.6) break
      if (travelled >= leafAt) {
        leafSide = -leafSide
        slots.push({
          p,
          angle: heading + leafSide * (opts.leafAngle + (next() - 0.5) * 0.5),
          idx: i,
        })
        leafAt += opts.leafEvery * (0.7 + next() * 0.6)
      }
      if (travelled >= branchAt && depth < opts.depth) {
        branches.push({ p, heading: heading - side * opts.branchAngle, w })
        branchAt += opts.branch * (0.7 + next() * 0.7)
      }
    }

    if (travelled < Math.max(w0 * 2.5, opts.margin * 3, 14)) {
      field.rollback(id)
      return null
    }
    const stem: Stem = { id, pts, w0, w1: w, depth }
    this.items.stems.push(stem)
    this.end(pts, heading, side, w, id)
    for (const b of branches) {
      this.stem(
        b.p,
        b.heading,
        Math.max(2.5, b.w * opts.thin),
        length * (0.4 + next() * 0.4),
        depth + 1,
      )
    }
    // Leaves go on once the stem and its branches have taken their room, so none blocks a stem.
    for (const slot of slots) this.leaf(slot.p, slot.angle, id, slot.idx)
    return stem
  }

  // The largest circle on the curl side that the stem could follow from here.
  private fit(p: Pt, heading: number, side: number, ignore: (d: Disc) => boolean): number {
    const n: Pt = [-Math.sin(heading) * side, Math.cos(heading) * side]
    for (const R of FIT) {
      if (this.field.clearance(p[0] + n[0] * R, p[1] + n[1] * R, ignore, R * 0.85) >= R * 0.85)
        return R
    }
    return FIT[FIT.length - 1]
  }

  /** A leaf on a stem at `p`, shrunk until it has room; stems it belongs to are ignored near it. */
  leaf(p: Pt, angle: number, owner: number, idx: number, stalk?: Pt): Leaf | null {
    const { field, next, opts } = this
    let size = opts.leafSize * (0.8 + next() * 0.4)
    const ignore = (d: Disc) => d.owner === owner && Math.abs(d.idx - idx) < 25
    while (size >= MIN_LEAF) {
      const cx = p[0] + Math.cos(angle) * size * 0.55
      const cy = p[1] + Math.sin(angle) * size * 0.55
      if (field.clearance(cx, cy, ignore, size * 0.5) >= size * 0.5) {
        const leaf: Leaf = { x: p[0], y: p[1], angle, size, stalk }
        this.items.leaves.push(leaf)
        field.stamp(cx, cy, size * 0.42, this.nextId++)
        return leaf
      }
      size *= 0.84
    }
    return null
  }

  // A stem ends in a tendril that curls into whatever room is left, or in a bud.
  private end(pts: Pt[], heading: number, side: number, w: number, owner: number) {
    const { field, next, opts } = this
    const e = pts[pts.length - 1]
    const n: Pt = [-Math.sin(heading) * side, Math.cos(heading) * side]
    const ignore = (d: Disc) => d.owner === owner && d.idx > pts.length - 16
    for (let R = opts.tendril; R >= opts.tendril * 0.4; R *= 0.8) {
      const c: Pt = [e[0] + n[0] * R, e[1] + n[1] * R]
      if (field.clearance(c[0], c[1], ignore, R + 1) < R + 1) continue
      const turns = 1.5 + next() * 0.8
      const t0 = Math.atan2(e[1] - c[1], e[0] - c[0])
      const spiral: Pt[] = []
      const steps = Math.ceil(turns * 14)
      for (let s = 0; s <= steps; s++) {
        const t = s / steps
        const r = R * (1 - 0.78 * t)
        const a = t0 + side * t * turns * TAU
        spiral.push([c[0] + Math.cos(a) * r, c[1] + Math.sin(a) * r])
      }
      this.items.tendrils.push({ pts: spiral, w: Math.min(w, 5) })
      field.stamp(c[0], c[1], R, this.nextId++)
      return
    }
    this.items.dots.push({ x: e[0], y: e[1], r: Math.max(3, w * 0.9) })
  }
}
