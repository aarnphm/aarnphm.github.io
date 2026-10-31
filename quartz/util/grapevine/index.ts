import type { Element } from 'hast'
import { s } from 'hastscript'
import { Field } from './field'
import { placeLetter, type Pt } from './glyph'
import { empty, Grower, rng, type Items, type Options, type Rng } from './grow'
import { defs, grapes, leaf, ribbon, tendril } from './parts'

// A grapevine grown around a capital at build time, and the same vine as a chapter rule. The
// seed is the page and the letter, so a page always prints the same ornament and nothing moves
// after first paint. Mechanism after azlen/initialis (MIT, Azlen Elza), redrawn as SVG.

export const TILE = 1000
const LETTER = 0.62
const FRAME = 14
const GAP = 22
const RULE_W = 1000
const RULE_H = 96

const INITIAL: Options = {
  step: 4,
  margin: 9,
  width: 13,
  taper: 0.55,
  curl: 0.75,
  turns: 1.1,
  wander: 0.5,
  length: 620,
  branch: 150,
  branchAngle: 0.7,
  thin: 0.55,
  depth: 2,
  leafEvery: 115,
  leafSize: 96,
  leafAngle: 0.8,
  tendril: 40,
}

const RULE: Options = {
  ...INITIAL,
  step: 3,
  margin: 4,
  width: 6,
  taper: 0.4,
  length: 990,
  branch: 125,
  branchAngle: 0.9,
  thin: 0.6,
  curl: 1.1,
  depth: 1,
  leafEvery: 90,
  leafSize: 40,
  leafAngle: 0.9,
  tendril: 17,
}

const SVG = { xmlns: 'http://www.w3.org/2000/svg', ariaHidden: 'true', focusable: 'false' }

function nearestStem(items: Items, x: number, y: number, within: number): Pt | null {
  let best: Pt | null = null
  let dmin = within
  for (const stem of items.stems)
    for (const p of stem.pts) {
      const d = Math.hypot(p[0] - x, p[1] - y)
      if (d < dmin) {
        dmin = d
        best = p
      }
    }
  return best
}

/** Fills what the stems left open, biggest gap first: a sprout, a cluster, a leaf, then buds. */
function fill(g: Grower, next: Rng, limit: { grapes: number; leaf: number; dot: number }) {
  const { field, items } = g
  for (let n = 0; n < 48; n++) {
    const gap = field.biggest()
    if (gap.r < limit.dot) break
    const from = nearestStem(items, gap.x, gap.y, gap.r * 2.5)
    // Big room gets a new stem first; a cluster hangs in what a stem would only cross.
    if (gap.r >= limit.leaf * 1.6 && from && (gap.r >= limit.grapes * 1.5 || next() < 0.5)) {
      const heading = Math.atan2(gap.y - from[1], gap.x - from[0]) + (next() - 0.5) * 1.2
      const stem = g.stem(from, heading, g.opts.width * 0.5, gap.r * 2.2, 1)
      if (stem) continue
    }
    if (gap.r >= limit.grapes && from && next() < 0.6) {
      const size = Math.min(gap.r * 1.7, limit.grapes * 2.4)
      items.grapes.push({ x: gap.x, y: gap.y, size, stalk: from })
      field.stamp(gap.x, gap.y, size * 0.55, 0)
      continue
    }
    if (gap.r >= limit.leaf) {
      const angle = from ? Math.atan2(gap.y - from[1], gap.x - from[0]) : next() * Math.PI * 2
      const size = Math.min(gap.r * 1.8, limit.leaf * 3)
      const base: Pt = [gap.x - Math.cos(angle) * size * 0.5, gap.y - Math.sin(angle) * size * 0.5]
      const stalk =
        from && Math.hypot(from[0] - base[0], from[1] - base[1]) < size ? from : undefined
      items.leaves.push({ x: base[0], y: base[1], angle, size, stalk })
      field.stamp(gap.x, gap.y, size * 0.5, 0)
      continue
    }
    items.dots.push({ x: gap.x, y: gap.y, r: Math.min(gap.r * 0.55, 8) })
    field.stamp(gap.x, gap.y, Math.min(gap.r, 14), 0)
  }
}

function draw(id: string, items: Items, extra: Element[]): Element[] {
  return [
    defs(id),
    s('g', { class: 'gv-stem' }, [
      ...items.stems.map(st => ribbon(st.pts, st.w0, st.w1)),
      ...items.tendrils.map(tendril),
      ...items.dots.map(d =>
        s('circle', { cx: Math.round(d.x), cy: Math.round(d.y), r: d.r.toFixed(1) }),
      ),
    ]),
    s(
      'g',
      { class: 'gv-leaf' },
      items.leaves.flatMap(l => leaf(id, l)),
    ),
    s(
      'g',
      { class: 'gv-grapes' },
      items.grapes.flatMap(gr => grapes(id, gr)),
    ),
    ...extra,
  ]
}

/**
 * The initial: a square tile, 1 em a side, with the capital in the middle and the vine grown
 * from the corners and the biggest gaps into the room around it.
 */
export function vineInitial(letter: string, seed: string): Element | null {
  const placed = placeLetter(letter, TILE, LETTER)
  if (!placed) return null
  const next = rng(`${seed}:${letter}`)
  const id = `gv${(next() * 1e6).toFixed(0)}`
  const field = new Field(TILE, TILE, 5, FRAME + 8, placed.rings, GAP)
  const g = new Grower(field, next, empty(), INITIAL)

  // Corners first, each creeping along a side, then whatever room is biggest.
  const inset = FRAME + 8 + GAP + INITIAL.width / 2 + INITIAL.margin + 3
  const corners: [Pt, number][] = [
    [[inset, inset], 0],
    [[TILE - inset, inset], Math.PI / 2],
    [[TILE - inset, TILE - inset], Math.PI],
    [[inset, TILE - inset], -Math.PI / 2],
  ]
  const start = Math.floor(next() * 4)
  for (let k = 0; k < 4; k++) {
    const [p, heading] = corners[(start + k) % 4]
    if (field.clearance(p[0], p[1]) < INITIAL.width) continue
    g.stem(p, heading + (next() - 0.5) * 0.6, INITIAL.width, INITIAL.length, 0)
  }
  fill(g, next, { grapes: 46, leaf: 24, dot: 12 })

  const glyph = s('path', { class: 'gv-letter', d: placed.d, transform: placed.transform })
  const frame = s('rect', {
    class: 'gv-frame',
    x: FRAME,
    y: FRAME,
    width: TILE - FRAME * 2,
    height: TILE - FRAME * 2,
  })
  return s('svg', { ...SVG, class: 'gv-initial', viewBox: `0 0 ${TILE} ${TILE}` }, [
    frame,
    ...draw(id, g.items, [glyph]),
  ])
}

/** The chapter rule: one vine running the measure, scrolling branches above and below. */
export function vineRule(seed: string): Element {
  const next = rng(`${seed}:rule`)
  const id = `gv${(next() * 1e6).toFixed(0)}`
  const field = new Field(RULE_W, RULE_H, 4, 2, [], 2)
  const g = new Grower(field, next, empty(), RULE)
  const amp = 0.2 + next() * 0.12
  const wave = 70 + next() * 40
  const phase = next() * Math.PI * 2
  g.stem(
    [4 + RULE.width / 2 + RULE.margin + 3, RULE_H / 2],
    0,
    RULE.width,
    RULE.length,
    0,
    (_p, _h, t) => amp * Math.sin(t / wave + phase),
  )
  fill(g, next, { grapes: 20, leaf: 11, dot: 6 })
  return s(
    'svg',
    { ...SVG, class: 'gv-rule', viewBox: `0 0 ${RULE_W} ${RULE_H}` },
    draw(id, g.items, []),
  )
}
