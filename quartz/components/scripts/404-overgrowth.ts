// Overgrowth on the 404 ruin. Moss has worked into the joints of the ring and pours down its right
// side onto the island, a sapling has rooted in the keystone (and leans with the ring), roots bridge the gap the eight o'clock stone
// left, and vines hang into the opening, where the rift pulls their leaves loose while it is open.
// Every clump is a handful of balls merged with a smooth minimum and wobbled by noise, printed by
// paintBlob with the ruin's keyline and light, so each one outlines itself where it overlaps the
// last.

import { GATE, hourAngle, onStone, slipPoint, stoneOuter, tilt } from './404-gate'
import {
  type Grid,
  type Pt,
  type RGB,
  type Tones,
  col,
  dot,
  hash,
  hex,
  interior,
  line,
  mulberry,
  noise,
  paintBlob,
  put,
  row,
  smin,
} from './404-pixel'

type Ball = { x: number; y: number; r: number; at: number }
// A tapered capsule from a (radius r0) to b (radius r1).
type Stem = { a: Pt; b: Pt; r0: number; r1: number; at: number }
type Kind = 'moss' | 'bark' | 'crown'
type Clump = { kind: Kind; balls: Ball[]; stems: Stem[]; flowers: number; seed: number }
// One-pixel roots hanging free.
type Hair = { pts: Pt[]; at: number }

export type Greens = { moss: Tones; bark: Tones; crown: Tones; flowers: RGB[]; hair: RGB }
export type Overgrowth = { clumps: Clump[]; hairs: Hair[] }

// Seconds from the reveal until the last clump is out.
export const GROWN = 6
const deg = (d: number) => (d * Math.PI) / 180

function clump(kind: Kind, flowers: number, seed: number): Clump {
  return { kind, balls: [], stems: [], flowers, seed }
}

// Balls strewn along a polyline, shrinking from r0 to r1 and appearing from t0 to t1.
function strew(
  c: Clump,
  pts: Pt[],
  gap: number,
  r0: number,
  r1: number,
  t0: number,
  t1: number,
  rand: () => number,
) {
  let total = 0
  for (let k = 1; k < pts.length; k++)
    total += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])
  let walked = 0
  for (let k = 1; k < pts.length; k++) {
    const [ax, ay] = pts[k - 1]
    const [bx, by] = pts[k]
    const len = Math.hypot(bx - ax, by - ay)
    for (let s = 0; s < len; s += gap) {
      const u = (walked + s) / total
      const nx = -(by - ay) / len
      const ny = (bx - ax) / len
      const off = (rand() - 0.5) * 18
      const x = ax + ((bx - ax) * s) / len + nx * off
      const y = ay + ((by - ay) * s) / len + ny * off
      const r = r0 + (r1 - r0) * u + (rand() - 0.3) * 7
      const at = t0 + (t1 - t0) * u
      c.balls.push({ x, y, r, at })
      // A lump budding off one side, so the run never settles into a tube.
      if (rand() < 0.4) {
        const side = rand() < 0.5 ? -1 : 1
        c.balls.push({ x: x + nx * side * r, y: y + ny * side * r, r: r * 0.5, at: at + 0.3 })
      }
    }
    walked += len
  }
}

export function makeOvergrowth(): Overgrowth {
  const rand = mulberry(808)
  const clumps: Clump[] = []
  const hairs: Hair[] = []

  // Roots across the gap, from the seven o'clock stone to the nine, sagging toward the opening, with
  // moss balled up where they grip.
  const roots = clump('bark', 0, 41)
  const gripMoss = clump('moss', 0.04, 43)
  for (const [ra, rb, sag, w, t0] of [
    [198, 202, 16, 4.5, 1.6],
    [182, 188, 26, 3, 1.9],
  ] as const) {
    const a = onStone(7, ra, deg(133))
    const b = onStone(9, rb, deg(167))
    const mid = deg(150)
    const c = tilt([
      GATE.x + Math.cos(mid) * ((ra + rb) / 2 - sag * 2),
      GATE.y + Math.sin(mid) * ((ra + rb) / 2 - sag * 2),
    ])
    const curve = (t: number): Pt => [
      (1 - t) ** 2 * a[0] + 2 * t * (1 - t) * c[0] + t * t * b[0],
      (1 - t) ** 2 * a[1] + 2 * t * (1 - t) * c[1] + t * t * b[1],
    ]
    // Thickest where they grip, thinnest where they sag.
    const swell = (t: number) => w * (0.75 + 0.35 * Math.abs(t - 0.5) * 2)
    for (let k = 0; k < 8; k++) {
      const u = k / 8
      const v = (k + 1) / 8
      roots.stems.push({ a: curve(u), b: curve(v), r0: swell(u), r1: swell(v), at: t0 + u * 0.8 })
    }
    for (const t of [0.28, 0.44, 0.61, 0.77]) {
      if (w < 4 && t > 0.6) continue
      const [x, y] = curve(t)
      const len = 16 + rand() * 26
      hairs.push({
        pts: [
          [x, y],
          [x + (rand() - 0.5) * 6, y + len * 0.5],
          [x + (rand() - 0.5) * 10, y + len],
        ],
        at: t0 + 0.8 + rand() * 0.6,
      })
    }
    for (const p of [a, b])
      for (let n = 0; n < 3; n++)
        gripMoss.balls.push({
          x: p[0] + (rand() - 0.5) * 14,
          y: p[1] + (rand() - 0.5) * 14,
          r: 5 + rand() * 4,
          at: t0 + rand() * 0.4,
        })
  }
  clumps.push(roots, gripMoss)

  // The beard: moss over the ring from twenty to ten to half past eleven, lobes hanging over the
  // inner edge and dripping into the opening.
  const beard = clump('moss', 0.035, 47)
  for (let s = 0; s <= 24; s++) {
    const hr = 9.55 + (2 * s) / 24
    const h = Math.round(hr) % 12
    const a = hourAngle(hr)
    const t = 0.3 + s * 0.08
    const ball = (r: number, size: number, at: number) => {
      const [x, y] = onStone(h, r, a + (rand() - 0.5) * 0.04)
      beard.balls.push({ x, y, r: size, at })
    }
    ball(GATE.r + 8 + rand() * 10, 8 + rand() * 6, t)
    if (s % 2 === 0) ball(GATE.r + 26 + rand() * 10, 7 + rand() * 6, t + 0.25)
    if (s % 4 === 1) ball(stoneOuter(h) + 1, 5 + rand() * 5, t + 0.5)
    if (s % 3 === 2) {
      const [x, y] = onStone(h, GATE.r - 5, a)
      beard.balls.push({ x, y, r: 7 + rand() * 3, at: t + 0.4 })
      const n = 1 + Math.floor(rand() * 3)
      for (let k = 1; k <= n; k++)
        beard.balls.push({
          x: x + (rand() - 0.5) * 4,
          y: y + k * 8,
          r: Math.max(2.8, 6 - k * 1.1),
          at: t + 0.5 + k * 0.2,
        })
    }
  }
  clumps.push(beard)

  // A cushion on the keystone, with a sapling rooted in its chipped corner.
  const cushion = clump('moss', 0.05, 53)
  for (let k = 0; k < 10; k++) {
    const [x, y] = onStone(0, 244 + rand() * 8, deg(-108 + k * 3.6))
    cushion.balls.push({ x, y, r: 6 + rand() * 6, at: 2 + k * 0.06 })
  }
  clumps.push(cushion)
  const trunk = clump('bark', 0, 59)
  const base = slipPoint(0, 798, 190)
  trunk.stems.push(
    { a: base, b: [base[0] + 6, base[1] - 26], r0: 4.2, r1: 3.2, at: 2.6 },
    { a: [base[0] + 6, base[1] - 26], b: tilt([812, 140]), r0: 3.2, r1: 2.2, at: 2.75 },
    { a: [base[0] + 7, base[1] - 30], b: tilt([790, 150]), r0: 1.9, r1: 1.3, at: 2.9 },
  )
  clumps.push(trunk)
  const crown = clump('crown', 0.06, 61)
  for (const [x, y, r, at] of [
    [790, 148, 6, 2.95],
    [798, 134, 10, 3],
    [812, 126, 15, 3.1],
    [826, 132, 11, 3.2],
    [806, 114, 9, 3.3],
    [820, 116, 8, 3.4],
  ]) {
    const [bx, by] = tilt([x, y])
    crown.balls.push({ x: bx, y: by, r, at })
  }
  clumps.push(crown)

  // The cascade: from the four o'clock stone down the five o'clock's face, over the shoulder of the
  // six o'clock stone where it comes out of the ground, and away down the island's slope.
  const cascade = clump('moss', 0.045, 67)
  strew(
    cascade,
    [
      slipPoint(4, 966, 560),
      slipPoint(4, 958, 598),
      onStone(5, 204, deg(56)),
      onStone(5, 204, deg(68)),
      [904, 648],
      [926, 666],
      [954, 680],
      [984, 694],
      [1014, 706],
      [1044, 720],
    ],
    11,
    15,
    7,
    1.2,
    4.2,
    rand,
  )
  clumps.push(cascade)

  // Tufts: on the three o'clock and two o'clock faces, in the dish of the fallen stone, on the
  // tablet's corner and where the ring goes into the ground on the left.
  ;(
    [
      [...tilt([1036, 418]), 3, 3.6],
      [...tilt([985, 330]), 3, 3.9],
      [372, 748, 4, 4.4],
      [690, 760, 3, 4.8],
      [744, 648, 3, 5.2],
    ] as const
  ).forEach(([x, y, n, at], k) => {
    const tuft = clump('moss', 0.08, 71 + k)
    for (let b = 0; b < n; b++)
      tuft.balls.push({
        x: x + (rand() - 0.5) * 16,
        y: y + (rand() - 0.5) * 8,
        r: 5 + rand() * 4,
        at: at + b * 0.12,
      })
    clumps.push(tuft)
  })

  return { clumps, hairs }
}

function taper(x: number, y: number, s: Stem) {
  const dx = s.b[0] - s.a[0]
  const dy = s.b[1] - s.a[1]
  const t = Math.max(
    0,
    Math.min(1, ((x - s.a[0]) * dx + (y - s.a[1]) * dy) / (dx * dx + dy * dy || 1)),
  )
  return Math.hypot(x - s.a[0] - dx * t, y - s.a[1] - dy * t) - (s.r0 + (s.r1 - s.r0) * t)
}

// A ball pops out at three fifths of its size and fills out a beat later.
const POP = 0.25

export function paintOvergrowth(
  d: Uint8ClampedArray,
  g: Grid,
  og: Overgrowth,
  t: number,
  greens: Greens,
) {
  for (const c of og.clumps) {
    const balls = c.balls
      .filter(b => b.at <= t)
      .map(b => ({ ...b, r: t - b.at < POP ? b.r * 0.6 : b.r }))
    const stems = c.stems.filter(s => s.at <= t)
    if (!balls.length && !stems.length) continue
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity]
    for (const b of balls) {
      x0 = Math.min(x0, b.x - b.r - 4)
      y0 = Math.min(y0, b.y - b.r - 4)
      x1 = Math.max(x1, b.x + b.r + 4)
      y1 = Math.max(y1, b.y + b.r + 4)
    }
    for (const s of stems) {
      const r = Math.max(s.r0, s.r1) + 2
      x0 = Math.min(x0, s.a[0] - r, s.b[0] - r)
      y0 = Math.min(y0, s.a[1] - r, s.b[1] - r)
      x1 = Math.max(x1, s.a[0] + r, s.b[0] + r)
      y1 = Math.max(y1, s.a[1] + r, s.b[1] + r)
    }
    const wobble = c.kind === 'moss' ? 5 : c.kind === 'crown' ? 4 : 1.5
    const field = (x: number, y: number) => {
      let f = 1e9
      for (const b of balls) f = smin(f, Math.hypot(x - b.x, y - b.y) - b.r, 7)
      for (const s of stems) f = smin(f, taper(x, y, s), 3)
      return f + (noise(x * 0.09, y * 0.09, c.seed) - 0.5) * wobble
    }
    const blob = paintBlob(d, g, field, [x0, y0, x1, y1], greens[c.kind])
    if (!blob || !c.flowers) continue
    interior(blob, (i, j) => {
      const h = hash(i, j, c.seed)
      if (h < c.flowers) put(d, (j * g.bw + i) * 4, greens.flowers[h < c.flowers / 2 ? 0 : 1])
    })
  }
  for (const h of og.hairs) {
    if (h.at > t) continue
    for (let k = 1; k < h.pts.length; k++)
      line(
        d,
        g,
        col(g, h.pts[k - 1][0]),
        row(g, h.pts[k - 1][1]),
        col(g, h.pts[k][0]),
        row(g, h.pts[k][1]),
        greens.hair,
      )
  }
}

// Vines lit by the rift, the same in either theme since the rift is always night.
export const VINE = {
  stem: hex('#4d6212'),
  leaf: ['#536b0e', '#879a39', '#b4bf6a'].map(hex),
  flower: hex('#e3a19a'),
}

type Knot = { x: number; y: number; px: number; py: number }
type Leaf = {
  knot: number
  side: number
  flower: boolean
  // 0 on the vine, 1 flying into the pole, 2 gone until it regrows
  state: 0 | 1 | 2
  r: number
  a: number
  back: number
}
type Vine = { knots: Knot[]; leaves: Leaf[]; at: number; seed: number }
type Drifter = { r: number; a: number; tone: number }
export type Curtain = { vines: Vine[]; drifters: Drifter[]; acc: number }

const KNOT = 7
// Anchors under the inner edge of the ring (upright frame), with how far each vine hangs.
const ANCHORS = (
  [
    [670, 340, 150],
    [712, 304, 120],
    [758, 283, 95],
    [812, 283, 70],
    [910, 318, 60],
  ] as const
).map(([x, y, len]) => [...tilt([x, y]), len] as const)

export function makeCurtain(): Curtain {
  const rand = mulberry(909)
  const vines = ANCHORS.map(([x, y, len], v) => {
    const n = Math.round(len / KNOT) + 1
    const knots = Array.from({ length: n }, (_, k) => {
      const kx = x + (rand() - 0.5) * 2
      const ky = y + k * KNOT
      return { x: kx, y: ky, px: kx, py: ky }
    })
    const leaves: Leaf[] = []
    for (let k = 2; k < n; k += 2)
      leaves.push({
        knot: k,
        side: (k / 2) % 2 ? 1 : -1,
        flower: rand() < 0.18,
        state: 0,
        r: 0,
        a: 0,
        back: 0,
      })
    return { knots, leaves, at: 2.4 + v * 0.35, seed: 900 + v }
  })
  const drifters = Array.from({ length: 28 }, () => ({
    r: 60 + rand() * (GATE.r - 70),
    a: rand() * Math.PI * 2,
    tone: rand(),
  }))
  return { vines, drifters, acc: 0 }
}

// Verlet ropes at a fixed 60 Hz substep. Gravity and a wandering breeze while the rift is idle; while
// it is open the pole pulls and the whirl drags them round with it, and their leaves come loose and
// spiral in. `turn(r)` is the rift's angular speed at radius r, in radians per second.
export function stepCurtain(
  c: Curtain,
  dt: number,
  clock: number,
  heat: number,
  inflow: number,
  turn: (r: number) => number,
) {
  const [px, py] = [GATE.x, GATE.y]
  c.acc = Math.min(c.acc + dt, 0.1)
  const h = 1 / 60
  while (c.acc >= h) {
    c.acc -= h
    for (const v of c.vines) {
      const ks = v.knots
      for (let k = 1; k < ks.length; k++) {
        const n = ks[k]
        const vx = (n.x - n.px) * 0.97
        const vy = (n.y - n.py) * 0.97
        const dx = px - n.x
        const dy = py - n.y
        const dist = Math.hypot(dx, dy) || 1
        const breeze = (noise(clock * 0.45 + v.seed, k * 0.18, 5) - 0.5) * 140
        const ax = breeze + heat * ((dx / dist) * 220 - (dy / dist) * 260)
        const ay = 320 + heat * ((dy / dist) * 220 + (dx / dist) * 260)
        n.px = n.x
        n.py = n.y
        n.x += vx + ax * h * h
        n.y += vy + ay * h * h
      }
      for (let it = 0; it < 4; it++)
        for (let k = 1; k < ks.length; k++) {
          const a = ks[k - 1]
          const b = ks[k]
          const dx = b.x - a.x
          const dy = b.y - a.y
          const len = Math.hypot(dx, dy) || 1
          const f = (len - KNOT) / len
          if (k === 1) {
            b.x -= dx * f
            b.y -= dy * f
          } else {
            a.x += dx * f * 0.5
            a.y += dy * f * 0.5
            b.x -= dx * f * 0.5
            b.y -= dy * f * 0.5
          }
        }
    }
  }

  const sink = (20 + 160 * inflow) * dt
  for (const v of c.vines)
    for (const l of v.leaves) {
      if (l.state === 0 && inflow > 0.1 && Math.random() < dt * inflow * 2.5) {
        const n = v.knots[l.knot]
        l.state = 1
        l.r = Math.hypot(n.x - px, n.y - py)
        l.a = Math.atan2(n.y - py, n.x - px)
      } else if (l.state === 1) {
        l.r -= sink
        l.a += turn(l.r) * dt
        if (l.r < 6) {
          l.state = 2
          l.back = 6 + Math.random() * 4
        }
      } else if (l.state === 2 && (l.back -= dt) <= 0 && inflow < 0.1) l.state = 0
    }
  for (const m of c.drifters) {
    m.a += turn(m.r) * dt
    m.r -= sink * 0.5
    if (m.r < 8) {
      m.r = GATE.r - 8 - Math.random() * 24
      m.a = Math.random() * Math.PI * 2
    }
  }
}

// A leaf four pixels big, slanted toward `side`: two plate, one lit, one shaded.
export function leaf(
  d: Uint8ClampedArray,
  g: Grid,
  i: number,
  j: number,
  side: number,
  flower: boolean,
  keep: (i: number, j: number) => boolean,
) {
  const cells: [number, number, RGB][] = flower
    ? [
        [0, 0, VINE.flower],
        [side, 0, VINE.flower],
        [0, -1, VINE.flower],
        [side, -1, VINE.leaf[2]],
      ]
    : [
        [0, 0, VINE.leaf[1]],
        [side, 0, VINE.leaf[1]],
        [0, 1, VINE.leaf[0]],
        [side, -1, VINE.leaf[2]],
      ]
  for (const [di, dj, c] of cells) if (keep(i + di, j + dj)) dot(d, g, i + di, j + dj, c)
}

// Vines unfurl a knot at a time once their anchor's moss is out; loose leaves and drifters ride
// the whirl. `keep` clips everything to the opening.
export function paintCurtain(
  d: Uint8ClampedArray,
  g: Grid,
  c: Curtain,
  grown: number,
  keep: (i: number, j: number) => boolean,
) {
  const px = GATE.x
  const py = GATE.y
  for (const v of c.vines) {
    const shown = Math.min(v.knots.length, Math.floor((grown - v.at) / 0.08) + 1)
    if (shown < 2) continue
    for (let k = 1; k < shown; k++) {
      const a = v.knots[k - 1]
      const b = v.knots[k]
      const n = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / (g.px * 0.7)))
      for (let s = 0; s <= n; s++) {
        const i = col(g, a.x + ((b.x - a.x) * s) / n)
        const j = row(g, a.y + ((b.y - a.y) * s) / n)
        if (keep(i, j)) dot(d, g, i, j, VINE.stem)
      }
    }
    for (const l of v.leaves) {
      if (l.state === 2) continue
      if (l.state === 1) {
        leaf(
          d,
          g,
          col(g, px + Math.cos(l.a) * l.r),
          row(g, py + Math.sin(l.a) * l.r),
          l.side,
          l.flower,
          keep,
        )
        continue
      }
      if (l.knot >= shown) continue
      const n = v.knots[l.knot]
      leaf(d, g, col(g, n.x) + l.side, row(g, n.y), l.side, l.flower, keep)
    }
  }
  if (grown < 3) return
  for (const m of c.drifters) {
    const i = col(g, px + Math.cos(m.a) * m.r)
    const j = row(g, py + Math.sin(m.a) * m.r)
    const ink = m.tone < 0.2 ? VINE.flower : VINE.leaf[m.tone < 0.6 ? 1 : 2]
    for (const [di, dj] of [
      [0, 0],
      [1, 0],
    ])
      if (keep(i + di, j + dj)) dot(d, g, i + di, j + dj, di ? VINE.leaf[0] : ink)
  }
}

// Ivy climbing the hour hand: a stem winding round the bar, hidden where it passes behind, with
// leaves at its outer swings and one on the tail. (ux, uy) is the hand's direction, `half` its
// half width, `skip` the hub's radius.
export function paintIvy(
  d: Uint8ClampedArray,
  g: Grid,
  ux: number,
  uy: number,
  half: number,
  keep: (i: number, j: number) => boolean,
) {
  const nx = -uy
  const ny = ux
  const at = (u: number, v: number): [number, number] => [
    col(g, GATE.x + ux * u + nx * v),
    row(g, GATE.y + uy * u + ny * v),
  ]
  for (let u = 22; u <= 96; u += g.px * 0.5) {
    const phase = u * 0.16
    const v = (half + g.px * 0.6) * Math.sin(phase)
    if (Math.cos(phase) < 0 && Math.abs(v) < half) continue
    const [i, j] = at(u, v)
    if (keep(i, j)) dot(d, g, i, j, VINE.stem)
  }
  ;[34, 50, 66, 84, -30].forEach((u, k) => {
    const side = k % 2 ? -1 : 1
    const [i, j] = at(u, side * (half + g.px * 1.5))
    leaf(d, g, i, j, side * (nx >= 0 ? 1 : -1), k === 2, keep)
  })
}
