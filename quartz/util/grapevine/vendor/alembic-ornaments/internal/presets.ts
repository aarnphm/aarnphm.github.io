import type { Rng } from './rng.js'
import { add, polar, ribbon, translation, IDENTITY, type Matrix, type Vec } from './geometry.js'
import { fromPolyline } from './grow.js'
import { atan, atan2, cos, hypot, pow, sin } from './math.js'
import { bud, leaf, rosette, tulip, type LeafKind } from './motifs.js'
import { Sketch, type Shape } from './sketch.js'

/**
 * The presets. Each is an authored composition — where the stem runs, where
 * it branches, where the flower sits — and the seed only chooses within
 * that: which flower, which leaf, how the curl is proportioned. Nothing is
 * left to emergent growth, which is where the uncanny comes from.
 */

export type OrnamentPreset = 'vine' | 'frieze' | 'garland' | 'sprig'

export interface Part {
  shapes: Shape[]
}

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export interface Placement {
  part: number
  m: Matrix
  /** Drawn inside this group's transform and clip. */
  group?: number
}

/** A transformed, optionally clipped frame of reference — one edge of a border. */
export interface Group {
  m: Matrix
  clip?: Rect
}

export interface Composition {
  width: number
  height: number
  parts: Part[]
  placements: Placement[]
  groups?: Group[]
  /** Ungrouped ink is clipped to this rect (running bands). */
  clip?: Rect
  /**
   * Printed as solid ink with the drawing cut out of it: the outer rect,
   * less the inner one if given (a frame's ring).
   */
  block?: { outer: Rect; inner?: Rect }
}

/** Per-ornament character, rolled once so every flower and leaf agrees. */
export interface Character {
  leaf: LeafKind
  flower: 'rose' | 'daisy'
  petals: number
}

export function character(rng: Rng): Character {
  return {
    leaf: rng.pick(['lance', 'acanthus'] as const),
    flower: rng.pick(['rose', 'rose', 'daisy'] as const),
    petals: rng.int(5, 7),
  }
}

export class Builder {
  parts: Part[] = []
  placements: Placement[] = []
  groups: Group[] = []
  part(shapes: Shape[]): number {
    this.parts.push({ shapes })
    return this.parts.length - 1
  }
  place(part: number, m: Matrix = IDENTITY, group?: number) {
    if (this.parts[part].shapes.length) this.placements.push({ part, m, group })
  }
  group(m: Matrix, clip?: Rect): number {
    this.groups.push({ m, clip })
    return this.groups.length - 1
  }
}

/** Solid bars, as shapes. */
export const bars = (rects: Rect[]): Shape[] =>
  rects.map(({ x, y, w, h }) => ({
    poly: [
      { x, y },
      { x: x + w, y },
      { x: x + w, y: y + h },
      { x, y: y + h },
    ],
    hole: false,
    layer: 'field' as const,
  }))

/** A rectangle outlined by four bars of half-width `hw`, centred on its edges. */
export const outline = (r: Rect, hw: number): Rect[] => [
  { x: r.x - hw, y: r.y - hw, w: r.w + 2 * hw, h: 2 * hw },
  { x: r.x - hw, y: r.y + r.h - hw, w: r.w + 2 * hw, h: 2 * hw },
  { x: r.x - hw, y: r.y - hw, w: 2 * hw, h: r.h + 2 * hw },
  { x: r.x + r.w - hw, y: r.y - hw, w: 2 * hw, h: r.h + 2 * hw },
]

/** Radius of the largest empty disc centred on `p`. */
function clearance(sketch: Sketch, p: Vec, max: number): number {
  if (!sketch.free(p, 0.5)) return 0
  let lo = 0.5
  let hi = max
  if (sketch.free(p, hi)) return hi
  for (let i = 0; i < 8; i++) {
    const mid = (lo + hi) / 2
    if (sketch.free(p, mid)) lo = mid
    else hi = mid
  }
  return lo
}

/** The roomiest spot within `radius` of `around`. */
export function roomiest(
  sketch: Sketch,
  around: Vec,
  radius: number,
  max: number,
): { p: Vec; r: number } {
  let best = { p: around, r: clearance(sketch, around, max) }
  const steps = 6
  for (let i = -steps; i <= steps; i++) {
    for (let j = -steps; j <= steps; j++) {
      const p = { x: around.x + (i / steps) * radius, y: around.y + (j / steps) * radius }
      const r = clearance(sketch, p, max)
      if (r > best.r) best = { p, r }
    }
  }
  return best
}

/* ------------------------------------------------------------------ strokes */

export interface Stroke {
  points: Vec[]
  headings: number[]
  widths: number[]
  owner: number
}

/**
 * A volute, built rather than grown: a Bézier lead-in that leaves `start`
 * along `heading` and meets a logarithmic spiral tangentially at its outer
 * edge, which then winds clockwise `turns` times about `center`, closing from
 * radius `r0` to `r1`. Returns null if any of it would hit ink.
 */
export function volute(
  sketch: Sketch,
  start: Vec,
  heading: number,
  center: Vec,
  r0: number,
  r1: number,
  turns: number,
  width: (t: number) => number,
  launch: number,
  flip = false,
  /** Where on the spiral the lead-in joins it: π = its left side (arriving upward), 3π/2 = its top (arriving rightward). */
  phi0 = Math.PI,
): Stroke | null {
  // Flipped volutes are built upside down about the start and turned back,
  // so they arrive heading down and wind counter-clockwise.
  const m = (p: Vec): Vec => (flip ? { x: p.x, y: 2 * start.y - p.y } : p)
  if (flip) {
    center = m(center)
    heading = -heading
  }
  const pts: Vec[] = []
  const dir = 1
  const spiral: Vec[] = []
  const steps = Math.ceil(turns * 90)
  for (let i = 0; i <= steps; i++) {
    const u = i / steps
    const phi = phi0 + dir * u * turns * 2 * Math.PI
    const rho = r0 * pow(r1 / r0, u)
    spiral.push({ x: center.x + rho * cos(phi), y: center.y + rho * sin(phi) })
  }
  const p0 = spiral[0]
  const d = hypot(p0.x - start.x, p0.y - start.y)
  const c1 = add(start, polar(heading, d * 0.4))
  // Clockwise, the tangent at angle φ points along φ + π/2.
  const c2 = add(p0, polar(phi0 - Math.PI / 2, Math.min(d * 0.4, r0 * 1.3)))
  const leadSteps = Math.max(8, Math.ceil(d * 1.2))
  for (let i = 0; i < leadSteps; i++) {
    const t = i / leadSteps
    const a = pow(1 - t, 3),
      b = 3 * (1 - t) * (1 - t) * t,
      c = 3 * (1 - t) * t * t,
      e = pow(t, 3)
    pts.push({
      x: a * start.x + b * c1.x + c * c2.x + e * p0.x,
      y: a * start.y + b * c1.y + c * c2.y + e * p0.y,
    })
  }
  pts.push(...spiral)

  for (let i = 0; i < pts.length; i++) pts[i] = m(pts[i])

  // Resample to even spacing so widths and checks are uniform.
  const even: Vec[] = [pts[0]]
  let carry = 0
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1],
      b = pts[i]
    let seg = hypot(b.x - a.x, b.y - a.y)
    let from = a
    while (carry + seg >= 1) {
      const k = (1 - carry) / seg
      from = { x: from.x + (b.x - from.x) * k, y: from.y + (b.y - from.y) * k }
      even.push(from)
      seg = hypot(b.x - from.x, b.y - from.y)
      carry = 0
    }
    carry += seg
  }
  const n = even.length
  const widths = even.map((_, i) => width(i / (n - 1)))
  const headings = even.map((_p, i) => {
    const a = even[Math.max(0, i - 1)],
      b = even[Math.min(n - 1, i + 1)]
    return atan2(b.y - a.y, b.x - a.x)
  })
  for (let i = 0; i < n; i++) {
    if (hypot(even[i].x - start.x, even[i].y - start.y) < launch) continue
    if (!sketch.free(even[i], Math.max(0.6, widths[i]))) {
      ;(globalThis as any).DBG?.(i, n, even[i], sketch.inBounds(even[i], widths[i]))
      return null
    }
  }
  const owner = sketch.newOwner()
  sketch.fill(ribbon(even, widths))
  sketch.reservePath(even, widths, owner)
  return { points: even, headings, widths, owner }
}

/* -------------------------------------------------------------------- vine */

/**
 * One cell of a running scroll: the stretch of band above the stem from one
 * crest to the next. A branch peels off the crest and rolls into the hollow
 * with a flower in its eye; leaves take what room is left. The cell below
 * the stem is this one glide-reflected, which is how a rinceau repeats.
 */
export function vineCell(
  rng: Rng,
  ch: Character,
  T: number,
  cy: number,
  A: number,
  top: number,
  stemW: number,
  gap: number,
): Shape[] {
  const stemY = (x: number) => cy - A * cos((2 * Math.PI * x) / T)
  const slope = (x: number) => atan(A * ((2 * Math.PI) / T) * sin((2 * Math.PI * x) / T))
  const sketch = new Sketch({
    rng,
    gap,
    weight: 1,
    inBounds: (q, r) =>
      q.x - r - gap / 2 >= 0 &&
      q.x + r + gap / 2 <= T &&
      q.y - r >= top &&
      q.y + r <= stemY(q.x) + stemW,
  })
  const pts: Vec[] = []
  for (let x = -T * 0.3; x <= T * 1.3; x += 1) pts.push({ x, y: stemY(x) })
  const stem = fromPolyline(
    sketch,
    pts,
    pts.map(() => stemW),
    false,
  )

  // The branch: leaves just past the crest, rises, and rolls clockwise into
  // the bowl above the trough with a flower in its eye.
  const x0 = T * rng.range(0.03, 0.07)
  const start = { x: x0, y: stemY(x0) - stemW * 0.3 }
  const w = stemW * 0.85
  const floor = stemY(T / 2) - stemW - gap
  // Not quite a full turn, closing only a little: the stem cradles the
  // flower rather than coiling round a point.
  const turns = rng.range(0.82, 0.92)
  const close = rng.range(0.74, 0.82)
  const lean = rng.range(0.25, 0.4)
  const cx = T * rng.range(0.5, 0.55)
  let branch: Stroke | null = null
  let eye = { x: 0, y: 0 }
  let eyeR = 0
  for (const f of [1, 0.93, 0.86, 0.79, 0.72]) {
    const r0 = Math.min((floor - top) / 2 - w - gap / 2, T * 0.3) * f
    const center = { x: cx, y: floor - w - gap / 2 - r0 }
    branch = sketch.ignoring(stem.owner, () =>
      volute(
        sketch,
        start,
        slope(x0) - lean,
        center,
        r0,
        r0 * close,
        turns,
        t => w * (1 - 0.45 * pow(t, 1.5)),
        (stemW + w + gap) * 3,
      ),
    )
    if (branch) {
      eye = center
      eyeR = r0 * close - w * 0.55 - gap * 0.7
      break
    }
  }

  if (branch) {
    const n = branch.points.length
    const end = branch.points[n - 1]
    sketch.dot(end, branch.widths[n - 1] * 1.5, branch.owner)
    const spot = roomiest(sketch, eye, eyeR * 0.15, eyeR * 1.1)
    if (spot.r > gap * 1.8)
      rosette(sketch, spot.p, spot.r, ch.petals, rng.range(0, Math.PI), ch.flower, false)
  }

  // Leaves: on the outside of the branch, then on the stem under the curl.
  const big = (cy - top) * 0.75
  const sizes = [1, 0.86, 0.73, 0.62, 0.52, 0.44].map(f => big * f)
  if (branch) {
    for (const u of [0.2, 0.38, 0.58]) {
      const i = Math.round(u * (branch.points.length - 1))
      const side = -1
      const p = add(
        branch.points[i],
        polar(branch.headings[i] + (side * Math.PI) / 2, branch.widths[i] * 0.5),
      )
      sketch.ignoring(branch.owner, () =>
        sizes.some(len =>
          leaf(sketch, {
            base: p,
            heading: branch.headings[i] + side * 0.55,
            length: len,
            curve: -side * 0.3,
            kind: ch.leaf,
          }),
        ),
      )
    }
  }
  for (const x of [T * 0.62, T * 0.8, T * 0.42]) {
    const p = { x, y: stemY(x) - stemW * 0.5 }
    const h = slope(x) - rng.range(0.7, 1.0)
    sketch.ignoring(stem.owner, () =>
      sizes.some(len =>
        leaf(sketch, { base: p, heading: h, length: len, curve: 0.35, kind: ch.leaf }),
      ),
    )
  }
  return sketch.shapes
}

function vine(rng: Rng, ch: Character, reversed: boolean, W = 600): Composition {
  const H = 96
  const b = new Builder()
  const rule = reversed ? 0 : 1.6
  const pad = reversed ? 6 : 5
  const top = pad + rule
  const bottom = H - pad - rule
  const cy = H / 2
  const A = (bottom - top) * rng.range(0.16, 0.2)
  const stemW = (bottom - top) * 0.04
  const gap = (bottom - top) * 0.035
  // Whole repeats across the width, the band's centre on a crest.
  const n = Math.max(3, Math.round(W / ((bottom - top) * rng.range(2.2, 2.6))))
  const T = W / n
  const xc = W / 2 - T * Math.floor(n / 2) - T

  const cell = b.part(vineCell(rng.fork('cell'), ch, T, cy, A, top, stemW, gap))
  for (let i = 0; i <= n + 1; i++) {
    const x = xc + i * T
    b.place(cell, translation(x, 0))
    // Glide reflection: flip about the stem's axis, shift half a repeat.
    b.place(cell, { a: 1, b: 0, c: 0, d: -1, e: x + T / 2, f: 2 * cy })
  }
  const stem: Vec[] = []
  for (let x = -4; x <= W + 4; x += 1)
    stem.push({ x, y: cy - A * cos((2 * Math.PI * (x - xc)) / T) })
  const s = new Sketch({ rng, gap, weight: 1, inBounds: () => true })
  s.line(
    stem,
    stem.map(() => stemW),
  )
  b.place(b.part(s.shapes))

  const comp: Composition = { width: W, height: H, parts: b.parts, placements: b.placements }
  if (reversed) {
    // A paper line just inside the block's edge, the ornament kept within it.
    const inset = 3
    comp.block = { outer: { x: 0, y: 0, w: W, h: H } }
    comp.clip = { x: inset + gap, y: 0, w: W - 2 * (inset + gap), h: H }
    b.place(
      b.part(bars(outline({ x: inset, y: inset, w: W - 2 * inset, h: H - 2 * inset }, 0.45))),
      IDENTITY,
      b.group(IDENTITY),
    )
    comp.groups = b.groups
  } else {
    b.place(
      b.part(
        bars([
          { x: 0, y: 0, w: W, h: rule },
          { x: 0, y: H - rule, w: W, h: rule },
        ]),
      ),
    )
    comp.clip = { x: 0, y: 0, w: W, h: H }
  }
  return comp
}

/* ---------------------------------------------------------- symmetric pieces */

/** Leaves along strokes, alternating sides, each the largest that fits. */
export function foliate(
  sketch: Sketch,
  ch: Character,
  strokes: Stroke[],
  sizes: number[],
  step: number,
  from = 0.08,
  to = 0.8,
): void {
  for (const st of strokes) {
    const n = st.points.length
    let side: 1 | -1 = 1
    let travelled = 0
    let next = step * 0.5
    for (let i = 1; i < n; i++) {
      travelled += hypot(st.points[i].x - st.points[i - 1].x, st.points[i].y - st.points[i - 1].y)
      const u = i / (n - 1)
      if (u < from || u > to || travelled < next) continue
      next = travelled + step
      for (const sd of [side, -side as 1 | -1]) {
        const base = add(
          st.points[i],
          polar(st.headings[i] + (sd * Math.PI) / 2, st.widths[i] * 0.5),
        )
        const ok = sketch.ignoring(st.owner, () =>
          sizes.some(len =>
            leaf(sketch, {
              base,
              heading: st.headings[i] + sd * 0.6,
              length: len * (1 - 0.35 * u),
              curve: -sd * 0.3,
              kind: ch.leaf,
            }),
          ),
        )
        if (ok) break
      }
      side = -side as 1 | -1
    }
  }
}

/** A flower in a volute's eye, as large as the eye allows. */
export function cradle(sketch: Sketch, ch: Character, rng: Rng, eye: Vec, r: number): void {
  const spot = roomiest(sketch, eye, r * 0.2, r * 1.1)
  if (spot.r > sketch.gap * 2.2)
    rosette(sketch, spot.p, spot.r, ch.petals, rng.range(0, Math.PI), ch.flower, false)
}

interface Arm {
  /** Where the arm leaves the axis, and which way. */
  root: Vec
  heading: number
  /** The volute it ends in. */
  center: Vec
  r0: number
  /** Optional branch from partway along, curling the other way. */
  branch?: { u: number; peel: number; center: Vec; r0: number; flip: boolean }
}

/**
 * A mirrored piece: a hero on the axis, and one arm per side that sweeps out
 * and rolls up round a flower, with a branch curling the opposite way and
 * leaves on everything. Only the right half is drawn; the left is its mirror.
 */
function symmetric(
  rng: Rng,
  ch: Character,
  W: number,
  H: number,
  stemW: number,
  gap: number,
  hero: (sketch: Sketch) => void,
  arm: Arm,
  leafSize: number,
): Composition {
  const pad = gap
  const sketch = new Sketch({
    rng,
    gap,
    weight: 1,
    inBounds: (q, r) =>
      q.x - r >= gap / 2 && q.x + r <= W / 2 - pad && q.y - r >= pad && q.y + r <= H - pad,
  })
  sketch.onAxis(() => hero(sketch))

  const strokes: Stroke[] = []
  const eyes: { p: Vec; r: number }[] = []
  const w = stemW
  for (const f of [1, 0.92, 0.84, 0.76, 0.68]) {
    const r0 = arm.r0 * f
    const close = rng.range(0.72, 0.8)
    const main = volute(
      sketch,
      arm.root,
      arm.heading,
      arm.center,
      r0,
      r0 * close,
      rng.range(0.85, 0.95),
      t => w * (1.15 - 0.6 * pow(t, 1.3)),
      w * 6,
    )
    if (main) {
      strokes.push(main)
      eyes.push({ p: arm.center, r: r0 * close - w * 0.5 - gap * 0.7 })
      break
    }
  }
  if (arm.branch && strokes[0]) {
    const b = arm.branch
    const main = strokes[0]
    const i = Math.round(b.u * (main.points.length - 1))
    const bw = main.widths[i] * 0.8
    for (const f of [1, 0.9, 0.8, 0.7, 0.6]) {
      const r0 = b.r0 * f
      const close = rng.range(0.72, 0.8)
      const center = { x: b.center.x, y: b.center.y + (b.r0 - r0) * 0.5 }
      // Only the launch may touch the arm it springs from; past that, the
      // fit check sees the arm like any other ink.
      const br = volute(
        sketch,
        main.points[i],
        main.headings[i] + (b.flip ? b.peel : -b.peel),
        center,
        r0,
        r0 * close,
        rng.range(0.8, 0.9),
        t => bw * (1 - 0.45 * pow(t, 1.3)),
        (main.widths[i] + bw + gap) * 3,
        b.flip,
      )
      if (br) {
        strokes.push(br)
        eyes.push({ p: center, r: r0 * close - bw * 0.5 - gap * 0.7 })
        break
      }
    }
  }
  for (const st of strokes)
    sketch.dot(st.points[st.points.length - 1], st.widths[st.widths.length - 1] * 1.5, st.owner)
  for (const e of eyes) cradle(sketch, ch, rng, e.p, e.r)
  foliate(
    sketch,
    ch,
    strokes,
    [1, 0.85, 0.72, 0.6, 0.5, 0.42].map(k => leafSize * k),
    leafSize * 0.55,
  )

  const b = new Builder()
  const field = b.part(sketch.shapes.filter(x => x.layer === 'field'))
  const axis = b.part(sketch.shapes.filter(x => x.layer === 'axis'))
  b.place(field, translation(W / 2, 0))
  b.place(field, { a: -1, b: 0, c: 0, d: 1, e: W / 2, f: 0 })
  b.place(axis, translation(W / 2, 0))
  return { width: W, height: H, parts: b.parts, placements: b.placements }
}

/** Headpiece: a lily between two long sweeping arms. */
function garland(rng: Rng, ch: Character): Composition {
  const W = 600
  const H = 150
  const cy = H * rng.range(0.6, 0.64)
  const stemW = H * 0.024
  const gap = H * 0.024
  const hero = (sk: Sketch) => {
    const tu = H * rng.range(0.36, 0.4)
    const neck = cy - H * 0.11
    tulip(sk, { x: 0, y: neck }, -Math.PI / 2, tu, false)
    const knot = cy + H * 0.12
    sk.line(
      [
        { x: 0, y: knot },
        { x: 0, y: neck },
      ],
      [stemW * 0.9, stemW * 1.2],
    )
    bud(sk, { x: 0, y: knot }, Math.PI / 2, H * 0.18, false)
    sk.reserve({ x: 0, y: neck - tu * 0.5 }, tu * 0.52)
  }
  const r0 = H * rng.range(0.22, 0.26)
  return symmetric(
    rng,
    ch,
    W,
    H,
    stemW,
    gap,
    hero,
    {
      root: { x: 0, y: cy },
      heading: rng.range(0, 0.06),
      center: { x: W / 2 - gap * 2 - r0 * 1.15, y: cy - H * rng.range(0.02, 0.08) },
      r0,
      branch: {
        u: rng.range(0.22, 0.3),
        peel: rng.range(0.25, 0.4),
        center: { x: W * rng.range(0.2, 0.23), y: cy - H * 0.3 },
        r0: H * 0.18,
        flip: false,
      },
    },
    H * 0.3,
  )
}

/** Tailpiece: a flower on a stalk, two arms rolling outward like a lyre. */
function sprig(rng: Rng, ch: Character): Composition {
  const W = 240
  const H = 190
  const stemW = H * 0.022
  const gap = H * 0.022
  const root = { x: 0, y: H * 0.74 }
  const crown = H * rng.range(0.4, 0.44)
  const hero = (sk: Sketch) => {
    const pts: Vec[] = []
    for (let i = 0; i <= 20; i++) pts.push({ x: 0, y: root.y + (crown - root.y) * (i / 20) })
    sk.line(
      pts,
      pts.map((_, i) => stemW * (1.5 - 0.6 * (i / 20))),
    )
    const tu = H * rng.range(0.3, 0.34)
    tulip(sk, { x: 0, y: crown }, -Math.PI / 2, tu, false)
    const knot = root.y + H * 0.09
    sk.line(
      [
        { x: 0, y: knot },
        { x: 0, y: root.y },
      ],
      [stemW * 1.1, stemW * 1.5],
    )
    bud(sk, { x: 0, y: knot }, Math.PI / 2, H * 0.15, false)
    sk.reserve({ x: 0, y: crown - tu * 0.5 }, tu * 0.5)
  }
  const r0 = H * rng.range(0.15, 0.17)
  return symmetric(
    rng,
    ch,
    W,
    H,
    stemW,
    gap,
    hero,
    {
      root: { x: 0, y: root.y - stemW },
      heading: -rng.range(0.15, 0.35),
      center: { x: W / 2 - gap * 2 - r0 * 1.1, y: H * rng.range(0.4, 0.46) },
      r0,
    },
    H * 0.28,
  )
}

/* ----------------------------------------------------------------- exports */

export interface PresetInfo {
  id: OrnamentPreset
  label: string
  blurb: string
  /** Natural width as a fraction of the text column. */
  span: number
  /** Width ÷ height, fixed per preset so layout can reserve room before drawing. */
  aspect: number
}

export const PRESETS: PresetInfo[] = [
  { id: 'vine', label: 'vine', blurb: 'running scroll between rules', span: 1, aspect: 600 / 96 },
  {
    id: 'frieze',
    label: 'frieze',
    blurb: 'the vine cut white from a black block',
    span: 1,
    aspect: 600 / 96,
  },
  {
    id: 'garland',
    label: 'garland',
    blurb: 'a lily between two sweeping arms',
    span: 0.8,
    aspect: 600 / 150,
  },
  { id: 'sprig', label: 'sprig', blurb: 'a small flower to end on', span: 0.3, aspect: 240 / 190 },
]

/** `width` sets the running bands' length (vine, frieze) in their 96-unit-tall space; 600 upstream. */
export function composePreset(preset: OrnamentPreset, rng: Rng, width?: number): Composition {
  const ch = character(rng.fork('character'))
  switch (preset) {
    case 'frieze':
      return vine(rng.fork('vine'), ch, true, width)
    case 'garland':
      return garland(rng.fork('garland'), ch)
    case 'sprig':
      return sprig(rng.fork('sprig'), ch)
    default:
      return vine(rng.fork('vine'), ch, false, width)
  }
}
