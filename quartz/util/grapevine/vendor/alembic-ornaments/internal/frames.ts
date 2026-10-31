import type { Rng } from './rng.js'
import { add, translation, IDENTITY, type Matrix, type Vec } from './geometry.js'
import { cos, hypot, pow, sin } from './math.js'
import { leaf, rosette } from './motifs.js'
import {
  bars,
  Builder,
  character,
  cradle,
  foliate,
  outline,
  vineCell,
  volute,
  type Character,
  type Composition,
  type Rect,
  type Stroke,
} from './presets.js'
import { Sketch } from './sketch.js'

/**
 * Frames: the same woodcut vocabulary as the presets, laid round a page.
 * A frame is drawn to its box's proportions, so it's generated per aspect.
 */

export type FramePreset = 'border' | 'block' | 'corners'

export interface FrameInfo {
  id: FramePreset
  label: string
  blurb: string
}

export const FRAMES: FrameInfo[] = [
  { id: 'border', label: 'border', blurb: 'the vine round all four sides' },
  { id: 'block', label: 'block', blurb: 'the border cut from black' },
  { id: 'corners', label: 'corners', blurb: 'a spray in each corner' },
]

/** Frame units: at scale 1, the shorter side is this long. */
const M = 500

/** Longest side ÷ shortest a frame is drawn to; beyond it, it's drawn at this and letterboxed */
const ASPECT_LIMIT = 12

/** How heavy a frame can be set, relative to its natural size */
export const FRAME_SCALE_MIN = 0.6
export const FRAME_SCALE_MAX = 1.6

/**
 * The page a frame is drawn on. Its parts are sized in fixed units, so
 * scale works by changing the page instead: drawn on a smaller page and
 * stretched to the box, every band, flower and rule comes out heavier,
 * all in proportion.
 */
export function frameSize(aspect: number, scale = 1): { W: number; H: number } {
  const a = Math.min(ASPECT_LIMIT, Math.max(1 / ASPECT_LIMIT, aspect || 1))
  const short = M / scale
  return a >= 1 ? { W: short * a, H: short } : { W: short, H: short / a }
}

/**
 * How a frame's parts sit: outer margin, band depth, and how far content
 * must stay in: past the bands and arms along the edges (`edges`), or past
 * every bit of ink, corner sprays included (`ink`). Only corners differ.
 */
function metrics(preset: FramePreset) {
  const breath = M * 0.04
  if (preset === 'corners') {
    const ri = M * 0.045
    const D = M * 0.2
    return { o: ri, D, edges: ri + M * 0.075, ink: ri + D + breath }
  }
  const o = M * 0.035
  const D = M * 0.1
  return { o, D, edges: o + D + breath, ink: o + D + breath }
}

/** Whether content clears every bit of a frame, or only what runs along its edges */
export type FrameClear = 'ink' | 'edges'

/**
 * How far in from each edge content has to start to clear a frame, as
 * fractions of the box: x of its width (left and right), y of its height
 * (top and bottom). The distance is the same on all four sides, so for a
 * box w × h, x·w === y·h.
 *
 * `ink` (the default) clears all of it. `edges` lets content run in among
 * the corner sprays of a `corners` frame, which suits centred content;
 * for the other frames the two are the same.
 */
export function frameInset(
  preset: FramePreset,
  aspect: number,
  scale = 1,
  clear: FrameClear = 'ink',
): { x: number; y: number } {
  const { W, H } = frameSize(aspect, scale)
  const d = metrics(preset)[clear === 'edges' ? 'edges' : 'ink']
  return { x: d / W, y: d / H }
}

/* ------------------------------------------------------------------ bands */

/** Edges of a band of depth D inset by o, as transforms from band space (x along, y inward). */
function edges(W: number, H: number, o: number, D: number): { m: Matrix; L: number }[] {
  const Lh = W - 2 * o - 2 * D
  const Lv = H - 2 * o - 2 * D
  return [
    { L: Lh, m: { a: 1, b: 0, c: 0, d: 1, e: o + D, f: o } },
    { L: Lv, m: { a: 0, b: 1, c: -1, d: 0, e: W - o, f: o + D } },
    { L: Lh, m: { a: -1, b: 0, c: 0, d: -1, e: W - o - D, f: H - o } },
    { L: Lv, m: { a: 0, b: -1, c: 1, d: 0, e: o, f: H - o - D } },
  ]
}

/** Running vine along every edge, and a rosette tile in each corner box. */
function borderInk(
  b: Builder,
  rng: Rng,
  ch: Character,
  W: number,
  H: number,
  o: number,
  D: number,
  gap: number,
): void {
  const pad = gap * 1.2
  const top = pad
  const cy = D / 2
  const inner = D - 2 * pad
  const A = inner * 0.17
  const stemW = inner * 0.045
  const cells = new Map<number, number>()
  for (const { m, L } of edges(W, H, o, D)) {
    const n = Math.max(2, Math.round(L / (D * 2.4)))
    const T = L / n
    // Opposite edges share a cell; a cell is cut per length so repeats land whole.
    const key = Math.round(T * 100)
    if (!cells.has(key))
      cells.set(key, b.part(vineCell(rng.fork('cell'), ch, T, cy, A, top, stemW, gap)))
    const cell = cells.get(key)!
    const g = b.group(m, { x: 0, y: 0, w: L, h: D })
    for (let i = -1; i <= n; i++) {
      b.place(cell, translation(i * T, 0), g)
      b.place(cell, { a: 1, b: 0, c: 0, d: -1, e: i * T + T / 2, f: 2 * cy }, g)
    }
    const s = new Sketch({ rng, gap, weight: 1, inBounds: () => true })
    const stem: Vec[] = []
    for (let x = -2; x <= L + 2; x += 1) stem.push({ x, y: cy - A * cos((2 * Math.PI * x) / T) })
    s.line(
      stem,
      stem.map(() => stemW),
    )
    b.place(b.part(s.shapes), IDENTITY, g)
  }

  // Corner tile: a rosette with a leaf pointing into each corner of the box.
  const tile = new Sketch({
    rng,
    gap,
    weight: 1,
    inBounds: (q, r) =>
      q.x - r >= pad && q.y - r >= pad && q.x + r <= D - pad && q.y + r <= D - pad,
  })
  const c = { x: D / 2, y: D / 2 }
  const rr = D * 0.3
  rosette(tile, c, rr, ch.petals, rng.range(0, Math.PI), ch.flower, false)
  for (let k = 0; k < 4; k++) {
    const h = Math.PI / 4 + (k * Math.PI) / 2
    const base = add(c, { x: cos(h) * (rr + gap * 0.6), y: sin(h) * (rr + gap * 0.6) })
    const len = D * 0.707 - rr - gap * 2.2
    leaf(tile, { base, heading: h, length: len, curve: 0, kind: 'lance' }, false)
  }
  const t = b.part(tile.shapes)
  const g = b.group(IDENTITY)
  for (const [x, y] of [
    [o, o],
    [W - o - D, o],
    [o, H - o - D],
    [W - o - D, H - o - D],
  ])
    b.place(t, translation(x, y), g)
}

/** The lines that box the band: outer edge, inner edge, and the corner boxes' sides. */
function borderRules(W: number, H: number, o: number, D: number, hw: number): Rect[] {
  const rules = [
    ...outline({ x: o, y: o, w: W - 2 * o, h: H - 2 * o }, hw * 1.6),
    ...outline({ x: o + D, y: o + D, w: W - 2 * (o + D), h: H - 2 * (o + D) }, hw),
  ]
  for (const x of [o + D, W - o - D])
    for (const y of [o, H - o - D]) rules.push({ x: x - hw, y, w: 2 * hw, h: D })
  for (const y of [o + D, H - o - D])
    for (const x of [o, W - o - D]) rules.push({ x, y: y - hw, w: D, h: 2 * hw })
  return rules
}

function border(
  rng: Rng,
  ch: Character,
  aspect: number,
  scale: number,
  reversed: boolean,
): Composition {
  const { W, H } = frameSize(aspect, scale)
  const { o, D } = metrics('border')
  const b = new Builder()
  const gap = D * 0.045
  if (!reversed) {
    borderInk(b, rng, ch, W, H, o, D, gap)
    b.place(b.part(bars(borderRules(W, H, o, D, 0.55))), IDENTITY, b.group(IDENTITY))
    return { width: W, height: H, parts: b.parts, placements: b.placements, groups: b.groups }
  }
  // Cut from a black ring: the rules become paper lines just inside its edges.
  const inset = 2.6
  borderInk(b, rng, ch, W, H, o + inset, D - 2 * inset, gap)
  b.place(
    b.part(bars(borderRules(W, H, o + inset, D - 2 * inset, 0.45))),
    IDENTITY,
    b.group(IDENTITY),
  )
  return {
    width: W,
    height: H,
    parts: b.parts,
    placements: b.placements,
    groups: b.groups,
    block: {
      outer: { x: o, y: o, w: W - 2 * o, h: H - 2 * o },
      inner: { x: o + D, y: o + D, w: W - 2 * (o + D), h: H - 2 * (o + D) },
    },
  }
}

/* ---------------------------------------------------------------- corners */

/**
 * One corner, drawn on the half of it above the diagonal and mirrored
 * across it: a flower on the diagonal, an arm sweeping along the edge into
 * a volute round another flower, a branch curling in towards the page.
 */
function corners(rng: Rng, ch: Character, aspect: number, scale: number): Composition {
  const { W, H } = frameSize(aspect, scale)
  const { o: ri } = metrics('corners')
  const hw = 0.6
  const depth = M * 0.2
  // The arm follows the page, but its flowers and leaves don't: past the
  // lightest proportional weight it would only get longer and barer, so
  // lighter frames (a thin frame round a big box) keep their sprays tight.
  const reach = Math.min(W, H, M / FRAME_SCALE_MIN) * 0.42
  const w = M * 0.0075
  const gap = M * 0.0075
  const edge = ri + hw + gap
  const sketch = new Sketch({
    rng,
    gap,
    weight: 1,
    inBounds: (q, r) =>
      (q.x - q.y) / Math.SQRT2 >= r + gap / 2 &&
      q.y - r >= edge &&
      q.x + r <= ri + reach &&
      q.y + r <= ri + depth,
  })

  const rh = M * rng.range(0.045, 0.052)
  const hero = { x: ri + rh * 1.2 + gap, y: ri + rh * 1.2 + gap }
  sketch.onAxis(() => {
    rosette(sketch, hero, rh, ch.petals, Math.PI / 4, ch.flower, false)
  })

  const strokes: Stroke[] = []
  const eyes: { p: Vec; r: number }[] = []
  // The arm runs along the rule with a slight dip and rolls down into the
  // page at its end; a branch hangs from it, curling back under.
  const root = { x: hero.x + rh * 0.5, y: hero.y }
  for (const f of [1, 0.9, 0.8, 0.7]) {
    const r0 = M * 0.075 * f
    const center = { x: ri + reach - gap * 2 - r0, y: edge + w * 1.5 + r0 }
    const close = rng.range(0.72, 0.8)
    const arm = volute(
      sketch,
      root,
      rng.range(0.15, 0.3),
      center,
      r0,
      r0 * close,
      rng.range(0.82, 0.92),
      t => w * (1.2 - 0.6 * pow(t, 1.3)),
      rh + w * 4,
      false,
      Math.PI * 1.5,
    )
    if (arm) {
      strokes.push(arm)
      eyes.push({ p: center, r: r0 * close - w * 0.5 - gap * 0.7 })
      break
    }
  }
  if (strokes[0]) {
    const arm = strokes[0]
    const eye = eyes[0]
    for (const [u, f] of [
      [0.24, 1],
      [0.2, 0.9],
      [0.28, 0.85],
      [0.16, 0.75],
      [0.24, 0.65],
    ]) {
      const i = Math.round(u * (arm.points.length - 1))
      const p = arm.points[i]
      const bw = arm.widths[i] * 0.8
      const r0 = M * 0.055 * f
      const center = { x: p.x + r0 * 0.4, y: p.y + arm.widths[i] + gap * 2.5 + bw + r0 }
      // Never inside the arm's own curl: that's where its flower goes.
      if (hypot(center.x - eye.p.x, center.y - eye.p.y) < eye.r + r0 + gap * 6) continue
      const close = rng.range(0.72, 0.8)
      const br = volute(
        sketch,
        p,
        arm.headings[i] + 0.45,
        center,
        r0,
        r0 * close,
        rng.range(0.8, 0.9),
        t => bw * (1 - 0.45 * pow(t, 1.3)),
        (arm.widths[i] + bw + gap) * 3,
        false,
        0,
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
  const leafSize = M * 0.06
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
  const diagonal: Matrix = { a: 0, b: 1, c: 1, d: 0, e: 0, f: 0 }
  const at: Matrix[] = [
    IDENTITY,
    { a: -1, b: 0, c: 0, d: 1, e: W, f: 0 },
    { a: 1, b: 0, c: 0, d: -1, e: 0, f: H },
    { a: -1, b: 0, c: 0, d: -1, e: W, f: H },
  ]
  for (const m of at) {
    const g = b.group(m)
    b.place(field, IDENTITY, g)
    b.place(field, diagonal, g)
    b.place(axis, IDENTITY, g)
  }
  b.place(
    b.part(bars(outline({ x: ri, y: ri, w: W - 2 * ri, h: H - 2 * ri }, hw))),
    IDENTITY,
    b.group(IDENTITY),
  )
  return { width: W, height: H, parts: b.parts, placements: b.placements, groups: b.groups }
}

export function composeFrame(
  preset: FramePreset,
  rng: Rng,
  aspect: number,
  scale = 1,
): Composition {
  const ch = character(rng.fork('character'))
  switch (preset) {
    case 'block':
      return border(rng.fork('border'), ch, aspect, scale, true)
    case 'corners':
      return corners(rng.fork('corners'), ch, aspect, scale)
    default:
      return border(rng.fork('border'), ch, aspect, scale, false)
  }
}
