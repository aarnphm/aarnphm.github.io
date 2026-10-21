import type { Ink, Palette } from '../../../util/scene-palette'
import {
  clampCol,
  col,
  dot,
  frac,
  hash,
  INK,
  mix,
  mul,
  noise,
  put,
  row,
  step,
  type Grid,
  type RGB,
} from '../404-pixel'
import { hillTop } from './geography'
import { FLAME } from './light-marks'

// Staffage after the woman in blue crossing the field at Clichy: cap, face, dress, stride.
export type Walker = {
  x: number
  dir: number
  pause: number
  next: number
  stride: number
  walking: boolean
}

export const FIGURE = ['.CC.', '.CS.', '.BB.', 'DBBB', 'DBB.', 'DBBB', 'DDBB', 'DBBB']
const LEGS = ['.K.K', '..K.', '.KK.']
export const HOME = 800

// Wanders the strand right of the camp, stops now and then to look at the gate, and walks to it
// while the portal is open.
export function stepWalker(w: Walker, dt: number, drawn: boolean) {
  if (drawn) {
    const gap = HOME - w.x
    w.dir = Math.sign(gap) || w.dir
    w.walking = Math.abs(gap) > 12
    if (w.walking) w.x += w.dir * Math.min(Math.abs(gap), 40 * dt)
  } else if (w.pause > 0) {
    w.pause -= dt
    w.walking = false
    if (w.pause <= 0) w.dir = Math.random() < 0.5 ? -1 : 1
  } else {
    w.walking = true
    w.x += w.dir * 16 * dt
    if (w.x < 720) w.dir = 1
    if (w.x > 1350) w.dir = -1
    w.next -= dt
    if (w.next <= 0) {
      w.pause = 2.5 + Math.random() * 3
      w.next = 8 + Math.random() * 10
      w.dir = Math.sign(HOME - w.x) || 1
    }
  }
  if (w.walking) w.stride += dt
}

// A sprite standing on `foot`, and its reflection folded about the waterline with the ruin's
// squash, on alternate pixels, wobbling row by row.
function sprite(
  d: Uint8ClampedArray,
  g: Grid,
  rows: string[],
  ci: number,
  foot: number,
  flip: boolean,
  ink: (ch: string, i: number) => RGB,
  water: number,
  clock: number,
) {
  const w = rows[0].length
  rows.forEach((line, r) => {
    const j = foot - rows.length + 1 + r
    const mirror = water + 0.62 * (water - (g.sy[0] + j * g.px))
    const mj = row(g, mirror)
    const wobble = Math.round(Math.sin(mj * 1.7 + clock * 5) * 0.7)
    for (let c = 0; c < w; c++) {
      const ch = line[flip ? w - 1 - c : c]
      if (ch === '.') continue
      const i = ci - (w >> 1) + c
      const color = ink(ch, i)
      dot(d, g, i, j, color)
      if (mirror > water + g.px && (mj + i) % 2 === 0) dot(d, g, i + wobble, mj, color)
    }
  })
}

// The camp on the bank below the gate: a rose tent, a ring of hearth stones round two logs, a
// fire in stop-motion, two people sitting either side of it (one toasting something on a stick),
// and the walker. Everything within reach of the fire takes its colour on the side facing it.
export const FIRE = { x: 628, y: 848 }
const TENT_X = 556
const TENT = ['....S....', '...SSL...', '..SSSLL..', '.SSSLKLL.', 'SSSSLKKLL']
export const SITTER = ['.CC.', '.CS.', 'BBB.', 'BBBS', 'DDDD']
export const CAMPERS = [
  { x: 600, flip: false },
  { x: 658, flip: true },
]
const FLAME_ROWS = [2, 4, 7, 5, 2]
// The camp's reach along the strand at a given pixel size, from the tent's left edge to the right
// camper's: the triathlon keeps its transition out of it and rides round behind it.
export const campSpan = (px: number) =>
  [
    TENT_X - ((TENT[0].length >> 1) + 1) * px,
    CAMPERS[1].x + (SITTER[0].length - (SITTER[0].length >> 1)) * px,
  ] as const

// Firelight on the ground: an ellipse that reaches further up the bank than down it, stepped
// through three warm stops, and breathing with the flames. By day it barely leaves the hearth.
export function paintPool(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  clock: number,
  dark: boolean,
  doused: boolean,
) {
  if (doused) return
  const { bw, sx, sy } = g
  const strength = (dark ? 0.8 : 0.45) * (0.85 + 0.15 * noise(clock * 4, 0.5, 13))
  const [rx, up, down] = dark ? [130, 64, 34] : [60, 26, 16]
  const i0 = Math.max(0, col(g, FIRE.x - rx))
  const i1 = Math.min(bw - 1, col(g, FIRE.x + rx))
  const j0 = Math.max(0, row(g, FIRE.y - up))
  const j1 = Math.min(g.bh - 1, row(g, water) - 1)
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      const dy = (sy[j] - FIRE.y) / (sy[j] < FIRE.y ? up : down)
      const q = 1 - Math.hypot((sx[i] - FIRE.x) / rx, dy)
      if (q <= 0) continue
      const level = step(4, q ** 1.4 * strength, i, j)
      if (level) put(d, (j * bw + i) * 4, pal.glow[level - 1])
    }
}

export function paintCamp(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  w: Walker,
  clock: number,
  dark: boolean,
  doused: boolean,
) {
  const foot = row(g, FIRE.y)
  const fc = col(g, FIRE.x)
  const warmth = doused ? 0 : dark ? 0.5 : 0.2
  const lit = (c: RGB, i: number) => {
    const k = warmth * Math.max(0, 1 - Math.abs(g.sx[clampCol(g, i)] - FIRE.x) / 90)
    return k > 0 ? mix(c, FLAME[1], k) : dark ? mul(c, 0.8) : c
  }

  sprite(
    d,
    g,
    TENT,
    col(g, TENT_X),
    foot,
    false,
    (ch, i) => lit(ch === 'K' ? pal.figure.K : pal.tent[ch === 'L' ? 1 : 0], i),
    water,
    clock,
  )

  for (const o of [-4, -3, 3, 4]) dot(d, g, fc + o, foot, lit(pal.far[o & 1 ? 1 : 0], fc + o))
  for (let o = -2; o <= 2; o++) dot(d, g, fc + o, foot, pal.bark[Math.abs(o) === 2 ? 2 : 0])
  const beat = Math.floor(clock * 9)
  // Doused, the logs lie charred in the ring.
  if (doused) for (let o = -1; o <= 1; o++) dot(d, g, fc + o, foot - 1, mix(pal.bark[2], INK, 0.5))
  for (let o = -2; o <= 2 && !doused; o++) {
    const h = Math.max(1, Math.round(FLAME_ROWS[o + 2] * (0.55 + 0.6 * hash(o, beat, 5))))
    for (let r = 0; r < h; r++) {
      const t = (r + 0.5) / h
      const c =
        t < 0.34 && Math.abs(o) < 2 ? FLAME[0] : t < 0.6 ? FLAME[1] : t < 0.85 ? FLAME[2] : FLAME[3]
      dot(d, g, fc + o, foot - 1 - r, c)
    }
  }
  // Now and then a tongue tears off the tip.
  if (!doused && hash(beat, 1, 7) < 0.35)
    dot(d, g, fc - (hash(beat, 2, 7) < 0.5 ? 1 : 0), foot - 9 - (beat & 1), FLAME[2])

  CAMPERS.forEach((c, n) => {
    const ink = (ch: string, i: number) =>
      lit(
        ch === 'B' ? pal.coats[n] : ch === 'C' && n === 1 ? pal.tent[1] : pal.figure[ch as Ink],
        i,
      )
    sprite(d, g, SITTER, col(g, c.x), foot, c.flip, ink, water, clock)
  })
  // The stick runs from the left camper's hand to just over the flames, where the marshmallow
  // toasts from cream to gold to brown, and then there is a fresh one.
  const hand = col(g, CAMPERS[0].x) + 2
  const run = Math.max(1, fc - 2 - hand, 2)
  for (let s = 0; s <= run; s++)
    dot(
      d,
      g,
      hand + Math.round(((fc - 2 - hand) * s) / run),
      foot - 2 - Math.round((2 * s) / run),
      pal.bark[1],
    )
  const toast = doused ? 0.9 : frac(clock / 14)
  dot(d, g, fc - 1, foot - 5, toast < 0.5 ? FLAME[0] : toast < 0.8 ? FLAME[1] : pal.bark[2])

  const rows = [...FIGURE, LEGS[w.walking ? Math.floor(w.stride / 0.22) % 2 : 2]]
  const walkerFoot = row(g, Math.max(848, hillTop(w.x, water) + 8))
  sprite(
    d,
    g,
    rows,
    col(g, w.x),
    walkerFoot,
    w.dir < 0,
    (ch, i) => lit(pal.figure[ch as Ink], i),
    water,
    clock,
  )
}
