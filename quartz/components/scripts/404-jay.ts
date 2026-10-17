// The homepage trail's blue jay (visitor-trail/jay.ts) lives here too, in the tree at the right
// edge of the view, at the size of the folk in the left-hand tree. It stands on the level limbs and
// moves between them, and now and then flies out of view on an errand that goes with the season:
// in autumn it picks an acorn and carries it off to cache, in winter and spring it comes back with
// one and eats it on the limb, and in summer it goes out and comes back with nothing to show. On a
// limb it faces into the wind, as birds do. While it is away a downy woodpecker comes to the trunk
// and drums in short rolls; it leaves when the jay comes back. Night and storms keep the jay on its
// limb and the woodpecker away.

import { type Grid, type Pt, type RGB, col, dot, row, stamp } from './404-pixel'

type Limb = { x0: number; y0: number; x1: number; y1: number; w0: number; w1: number }
type Crown = { segs: Limb[] }
type Errand = 'cache' | 'retrieve' | 'forage'

export type Jay = {
  // Where its feet go on each limb, and where the woodpecker clings, facing the trunk.
  perches: Pt[]
  trunk: { at: Pt; face: 1 | -1 } | null
  exit: Pt
  pose: 'perched' | 'flying' | 'away'
  at: Pt
  from: Pt
  to: Pt
  // Flight progress 0..1 and its length in seconds.
  t: number
  span: number
  perch: number
  wait: number
  face: 1 | -1
  // Something in its bill, and whether it is eating what it brought back.
  acorn: boolean
  eating: number
  out: boolean
  guest: boolean
  drum: number
}

export type JayEnv = {
  // False at night and in a storm.
  active: boolean
  errand: Errand
  // The wind across the view, -1 from the right .. 1 from the left.
  wind: number
}

export type JayInk = {
  blue: RGB
  bar: RGB
  ink: RGB
  white: RGB
  belly: RGB
  nape: RGB
  acorn: RGB
}

// Facing right: crest, white face and black bill, barred wing, grey belly, tail and foot.
const PERCHED = ['...B..', '..BBWK', '.BBDW.', 'BBDBG.', 'B..K..']
// Head down to the limb, pecking at what it holds under its foot.
const PECK = ['......', '...B..', '.BBBWK', 'BBDBG.', 'B..K..']
const UP = ['.BD...', '..BB..', 'BBBBWK', '...G..']
const DOWN = ['......', 'BBBBWK', '..BD..', '.BD...']
// Clinging upright with the trunk on its right, tail braced; the strike puts its head to the bark.
const CLING = ['R..', 'KK.', 'WK.', 'WK.', 'K..']
const STRIKE = ['.R.', '.KK', 'WK.', 'WK.', 'K..']

export function makeJay(): Jay {
  return {
    perches: [],
    trunk: null,
    exit: [0, 0],
    pose: 'perched',
    at: [0, 0],
    from: [0, 0],
    to: [0, 0],
    t: 0,
    span: 1,
    perch: 0,
    wait: 3,
    face: -1,
    acorn: false,
    eating: 0,
    out: false,
    guest: false,
    drum: 0,
  }
}

// Up to three level limbs that are seen, a jay's length apart, and the lowest upright limb or trunk
// in view for the woodpecker. Returns false when the tree shows no limb to stand on.
export function perchJay(
  jay: Jay,
  tree: Crown,
  g: Grid,
  view: [number, number, number, number],
  hidden: (x: number, y: number) => boolean,
) {
  const px = g.px
  const seen = (x: number, y: number) =>
    x > view[0] + 4 * px &&
    x < view[2] - 4 * px &&
    y > view[1] + 8 * px &&
    y < view[3] - 12 * px &&
    !hidden(x, y)
  const limbs: { at: Pt; score: number }[] = []
  for (const s of tree.segs) {
    const dx = s.x1 - s.x0
    const dy = s.y1 - s.y0
    const len = Math.hypot(dx, dy)
    if (s.w0 < px || s.w0 > 6 * px || Math.abs(dy) > Math.abs(dx) * 0.8 || len < 4 * px) continue
    const at: Pt = [(s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2 - (s.w0 + s.w1) / 4]
    if (!seen(at[0], at[1] - 6 * px)) continue
    limbs.push({ at, score: len / px - (Math.abs(dy) / Math.abs(dx)) * 6 })
  }
  limbs.sort((a, b) => b.score - a.score)
  jay.perches = []
  for (const { at } of limbs) {
    if (jay.perches.every(p => Math.hypot(p[0] - at[0], p[1] - at[1]) > 12 * px))
      jay.perches.push(at)
    if (jay.perches.length === 3) break
  }
  let trunk: Limb | null = null
  for (const s of tree.segs) {
    const mx = (s.x0 + s.x1) / 2
    const my = (s.y0 + s.y1) / 2
    if (s.w0 < 3 * px || Math.abs(s.y1 - s.y0) < Math.abs(s.x1 - s.x0) || !seen(mx, my)) continue
    if (!trunk || my > (trunk.y0 + trunk.y1) / 2) trunk = s
  }
  if (trunk) {
    // The side that faces into the view, so the bird is not behind the bark.
    const face: 1 | -1 = (trunk.x0 + trunk.x1) / 2 > (view[0] + view[2]) / 2 ? 1 : -1
    const x = (trunk.x0 + trunk.x1) / 2 - face * ((trunk.w0 + trunk.w1) / 4 + 2 * px)
    jay.trunk = { at: [x, (trunk.y0 + trunk.y1) / 2], face }
  } else jay.trunk = null
  if (!jay.perches.length) return false
  const outward = (tree.segs[0]?.x0 ?? view[2]) > (view[0] + view[2]) / 2 ? 1 : -1
  jay.exit = [outward > 0 ? view[2] + 12 * px : view[0] - 12 * px, jay.perches[0][1] + 30 * px]
  jay.perch = 0
  jay.pose = 'perched'
  jay.at = jay.perches[0]
  jay.guest = false
  return true
}

const fly = (jay: Jay, to: Pt, span: number) => {
  jay.from = jay.at
  jay.to = to
  jay.t = 0
  jay.span = span
  jay.pose = 'flying'
  jay.face = to[0] < jay.at[0] ? -1 : 1
}

export function stepJay(jay: Jay, dt: number, env: JayEnv) {
  if (!jay.perches.length) return
  jay.drum += dt
  if (jay.pose === 'flying') {
    jay.t = Math.min(1, jay.t + dt / jay.span)
    const u = jay.t
    const lift = Math.min(
      40,
      6 + Math.hypot(jay.to[0] - jay.from[0], jay.to[1] - jay.from[1]) * 0.2,
    )
    jay.at = [
      jay.from[0] + (jay.to[0] - jay.from[0]) * u,
      jay.from[1] + (jay.to[1] - jay.from[1]) * u - Math.sin(u * Math.PI) * lift,
    ]
    if (u < 1) return
    jay.at = jay.to
    if (jay.out) {
      jay.pose = 'away'
      jay.wait = 5 + Math.random() * 5
      // The acorn is cached somewhere out of view; the woodpecker takes its chance.
      jay.acorn = false
      jay.guest = env.active && !!jay.trunk && Math.random() < 0.7
      jay.drum = 0
    } else {
      jay.pose = 'perched'
      jay.wait = 4 + Math.random() * 7
      // What it brought back it eats on the limb.
      if (jay.acorn) jay.eating = 2.4
    }
    return
  }
  if (jay.pose === 'away') {
    if ((jay.wait -= dt) > 0) return
    jay.out = false
    jay.guest = false
    jay.acorn = env.errand === 'retrieve'
    jay.perch = Math.floor(Math.random() * jay.perches.length)
    fly(jay, jay.perches[jay.perch], 1.4)
    return
  }
  if (jay.eating > 0) {
    if ((jay.eating -= dt) <= 0) jay.acorn = false
    return
  }
  // On the limb it turns into any wind worth the name.
  if (Math.abs(env.wind) > 0.15) jay.face = env.wind > 0 ? -1 : 1
  if (!env.active || (jay.wait -= dt) > 0) return
  if (jay.perches.length > 1 && Math.random() < 0.45) {
    const next =
      (jay.perch + 1 + Math.floor(Math.random() * (jay.perches.length - 1))) % jay.perches.length
    jay.perch = next
    fly(jay, jay.perches[next], 0.7)
    return
  }
  jay.out = true
  // An acorn picked from the crown goes with it to be cached.
  jay.acorn = env.errand === 'cache'
  fly(jay, jay.exit, 1.4)
}

export function paintJay(
  d: Uint8ClampedArray,
  g: Grid,
  jay: Jay,
  ink: JayInk,
  clock: number,
  active: boolean,
) {
  if (!jay.perches.length) return
  const tone = (ch: string) =>
    ch === 'B'
      ? ink.blue
      : ch === 'D'
        ? ink.bar
        : ch === 'K'
          ? ink.ink
          : ch === 'W'
            ? ink.white
            : ch === 'G'
              ? ink.belly
              : ch === 'R'
                ? ink.nape
                : null
  if (jay.guest && active && jay.trunk) {
    // A roll of strikes about fifteen a second for a second, then a few seconds' quiet.
    const roll = jay.drum % 4 < 1 && Math.floor(clock * 15) % 2 === 0
    const [x, y] = jay.trunk.at
    const sprite = roll ? STRIKE : CLING
    stamp(d, g, sprite, col(g, x) - 1, row(g, y) - 2, jay.trunk.face < 0, tone)
  }
  if (jay.pose === 'away') return
  const i = col(g, jay.at[0])
  const j = row(g, jay.at[1])
  const flip = jay.face < 0
  let sprite = PERCHED
  if (jay.pose === 'flying') sprite = Math.floor(clock * 8) % 2 ? UP : DOWN
  else if (jay.eating > 0 && Math.floor(clock * 4) % 2) sprite = PECK
  stamp(d, g, sprite, i - 3, j - sprite.length + 1, flip, tone)
  if (jay.acorn) {
    // In the bill, one pixel past its tip.
    const bill = sprite === DOWN ? 1 : sprite === UP ? 2 : sprite === PECK ? 2 : 1
    dot(d, g, flip ? i - 4 : i + 3, j - sprite.length + 1 + bill, ink.acorn)
  }
}
