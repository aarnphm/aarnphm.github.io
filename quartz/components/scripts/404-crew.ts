// A survey party a few pixels tall, sent to find out what the clock is. One climbs a ladder into
// the gap to read the roots through a glass, one hangs off the keystone on a rope, one fishes the
// rift for stars from the one o'clock stone and keeps them in a jar, a child pokes the opening with
// a stick, a surveyor sights it through a theodolite, someone sits in the grass at the ring's foot
// taking notes, someone points, and someone down on the bank is trying to make friends with the
// cat. They say "?" to
// each other now and then, "!" when the clock jumps or the portal opens, and "?" to whoever clicks
// on them (as does everyone else on the bank, whose heads the landscape files here too). While it is open, it
// pulls: the rope swings wide, the fishing line spirals in, and the child's stick is taken.

import { GATE, slipPoint } from './404-gate'
import { type Grid, type Pt, type RGB, BAYER, col, dot, line, noise, row, stamp } from './404-pixel'

const WHO = ['ladder', 'rope', 'fisher', 'kid', 'surveyor', 'writer', 'pointer', 'friend'] as const
type Who = (typeof WHO)[number]
type Glyph = '!' | '?'
type Bubble = { who: string; glyph: Glyph; age: number }
// Who is talking, and the top-middle pixel of each figure that can talk, as last painted.
export type Voices = { bubbles: Bubble[]; heads: Partial<Record<string, Pt>> }

export type Crew = {
  // The rope: angle from plumb, its rate, and its length.
  swing: number
  spin: number
  len: number
  fishNext: number
  fishT: number
  stars: number
  // The child's stick: 0 in hand, 1 being pulled in, 2 gone until a new one is found.
  stick: { state: 0 | 1 | 2; r: number; a: number; back: number }
  peer: boolean
  peerNext: number
  bubbles: Bubble[]
  chatter: number
  hot: boolean
  heads: Partial<Record<string, Pt>>
}

export type CrewInk = {
  skin: RGB
  trousers: RGB
  boots: RGB
  hat: RGB
  coats: RGB[]
  wood: RGB
  paper: RGB
  gold: RGB
  rose: RGB
  fish: RGB
  flame: RGB
  glow: RGB
  ink: RGB
  dark: boolean
}

const LADDER: [Pt, Pt] = [
  [548, 740],
  [636, 560],
]
const ANCHOR = slipPoint(0, 800, 188)
const SEAT: Pt = [906, 254]
const TIP: Pt = [866, 214]
const BOB: Pt = [866, 330]
const JAR: Pt = [924, 254]
const CHILD: Pt = [772, 606]
const POKE: Pt = [800, 570]
// Where the ones on the island stand, along the crown the ring has sunk into.
const SURVEYOR = 610
const TRIPOD = 634
const WRITER = 912
const POINTER = 704
const FRIEND = 1136

// Sprites face right; letters are hat, skin, coat, trousers, boots, paper (W), rose (R), fish (F).
const STAND = ['.HH.', 'HHHH', '.SS.', 'BBBB', '.BB.', '.BB.', '.DD.', '.K.K']
const REACH = ['.HH..', 'HHHH.', '.SS.S', '.BBB.', '.BB..', '.BB..', '.DD..', '.K.K.']
const PEER = ['....', '..HH', '.HHH', '.BSS', 'BBB.', '.BB.', '.DD.', '.K.K']
const ALARM = ['S..S', 'BHHB', 'HHHH', '.SS.', '.BB.', '.BB.', '.DD.', 'K..K']
const SIT = ['.HH.', 'HHHH', '.SS.', '.BB.', '.BBS', '.DDD', '...K']
const WRITE = ['.HH.', 'HHHH', '.SS.', '.BB.', '.BWW', '.DDD', '...K']
const HANG = ['.SS.', '.HH.', 'HHHH', '.SS.', 'BBBB', '.BB.', '.DD.', '.DD.', '.KK.']
const KID = ['.R.', '.S.', 'BBB', '.B.', '.D.', 'D.D']
const KID_SIT = ['.R.', '.S.', 'BBB', 'DDD']
const FISH = ['F..FF.', '.FFFFF', 'F..FF.']
const GLYPHS: Record<Glyph, string[]> = {
  '!': ['.K.', '.K.', '.K.', '...', '.K.'],
  '?': ['KK.', '..K', '.K.', '...', '.K.'],
}

export function makeCrew(): Crew {
  return {
    swing: 0.12,
    spin: 0,
    len: 132,
    fishNext: 4,
    fishT: -1,
    stars: 1,
    stick: { state: 0, r: 0, a: 0, back: 0 },
    peer: false,
    peerNext: 3,
    bubbles: [],
    chatter: 2.5,
    hot: false,
    heads: {},
  }
}

function say(v: Voices, who: string, glyph: Glyph) {
  v.bubbles = v.bubbles.filter(b => b.who !== who)
  if (v.bubbles.length >= 3) v.bubbles.shift()
  v.bubbles.push({ who, glyph, age: 0 })
}

// Whoever has a head within reach of buffer pixel (i, j), nearest first: a box a few pixels either
// side of the head and down to the feet of a figure `tall` pixels high.
export function heard(v: Voices, i: number, j: number, reach: number, tall: number) {
  let best: string | null = null
  let near = Infinity
  for (const [who, head] of Object.entries(v.heads)) {
    if (!head) continue
    const di = Math.abs(i - head[0])
    const dj = j - head[1]
    if (di > reach || dj < -2 || dj > tall) continue
    const dist = di + Math.abs(dj - tall / 2) * 0.5
    if (dist < near) {
      near = dist
      best = who
    }
  }
  return best
}

// Clicked on: they look up and ask.
export const ask = (v: Voices, who: string) => say(v, who, '?')

export function ageBubbles(v: Voices, dt: number) {
  for (const b of v.bubbles) b.age += dt
  v.bubbles = v.bubbles.filter(b => b.age < 1.8)
}

// Without motion the party holds one pose per state of the portal: open, the child has lost the
// stick and three of them are calling out; closed, they are quietly at work.
export function poseCrew(c: Crew, open: boolean) {
  c.hot = open
  c.stick.state = open ? 2 : 0
  c.bubbles = open
    ? (['kid', 'pointer', 'friend'] as const).map(who => ({ who, glyph: '!' as const, age: 0 }))
    : []
}

const anyone = () => WHO[Math.floor(Math.random() * WHO.length)]

export function stepCrew(
  c: Crew,
  dt: number,
  env: {
    clock: number
    heat: number
    inflow: number
    impulse: boolean
    turn: (r: number) => number
  },
) {
  const hot = env.heat > 0.3
  // A pendulum on the rope, stirred by the breeze; the open portal pays out more rope and drives
  // the swing round.
  c.len += (132 + 56 * env.heat - c.len) * Math.min(1, dt * 1.5)
  const breeze = (noise(env.clock * 0.3, 1.7, 9) - 0.5) * 0.9
  const drive = env.heat * 5 * Math.sin(env.clock * 2.4)
  c.spin += (-(400 / c.len) * Math.sin(c.swing) - 0.35 * c.spin + breeze + drive) * dt
  c.swing = Math.max(-1.1, Math.min(1.1, c.swing + c.spin * dt))

  if (c.fishT >= 0) {
    c.fishT += dt
    if (c.fishT > 1.8) {
      c.stars = c.stars >= 5 ? 1 : c.stars + 1
      c.fishT = -1
    }
  } else if (!hot && (c.fishNext -= dt) <= 0) {
    c.fishT = 0
    c.fishNext = 10 + Math.random() * 6
  }

  const s = c.stick
  if (s.state === 0 && hot) {
    s.state = 1
    s.r = Math.hypot(POKE[0] - GATE.x, POKE[1] - GATE.y)
    s.a = Math.atan2(POKE[1] - GATE.y, POKE[0] - GATE.x)
    say(c, 'kid', '!')
  } else if (s.state === 1) {
    s.r -= (50 + 200 * env.inflow) * dt
    s.a += env.turn(s.r) * dt
    if (s.r < 6) {
      s.state = 2
      s.back = 4
    }
  } else if (s.state === 2 && !hot && (s.back -= dt) <= 0) s.state = 0

  if ((c.peerNext -= dt) <= 0) {
    c.peer = !c.peer
    c.peerNext = 2.5 + Math.random() * 3
  }

  ageBubbles(c, dt)
  if (hot && !c.hot) for (const who of ['rope', 'pointer', 'friend'] as const) say(c, who, '!')
  c.hot = hot
  if (env.impulse) say(c, anyone(), '!')
  if ((c.chatter -= dt) <= 0) {
    c.chatter = 4 + Math.random() * 3
    say(c, anyone(), '?')
  }
}

export function paintCrew(
  d: Uint8ClampedArray,
  g: Grid,
  c: Crew,
  ink: CrewInk,
  clock: number,
  heat: number,
  alert: number,
  ground: (x: number) => number,
) {
  const letter =
    (coat: RGB, hat = ink.hat) =>
    (ch: string) =>
      ch === 'H'
        ? hat
        : ch === 'S'
          ? ink.skin
          : ch === 'B'
            ? coat
            : ch === 'D'
              ? ink.trousers
              : ch === 'K'
                ? ink.boots
                : ch === 'W'
                  ? ink.paper
                  : ch === 'R'
                    ? ink.rose
                    : ch === 'F'
                      ? ink.fish
                      : null
  // Stands the sprite with its last row on the pixel above `feet`; returns its top-left pixel.
  const figure = (
    who: Who | null,
    rows: readonly string[],
    x: number,
    feet: number,
    flip: boolean,
    paint: (ch: string) => RGB | null,
    top = row(g, feet) - rows.length,
  ): Pt => {
    const w = rows[0].length
    const i0 = col(g, x) - (w >> 1)
    stamp(d, g, rows, i0, top, flip, paint)
    if (who) c.heads[who] = [i0 + (w >> 1), top]
    return [i0, top]
  }
  const seg = (a: Pt, b: Pt, color: RGB) =>
    line(d, g, col(g, a[0]), row(g, a[1]), col(g, b[0]), row(g, b[1]), color)
  const halo = (i: number, j: number, r: number, strength: number) => {
    for (let dj = -r; dj <= r; dj++)
      for (let di = -r; di <= r; di++) {
        const q = Math.hypot(di, dj)
        if (q > 0 && (1 - q / (r + 0.8)) * strength > BAYER[((j + dj) & 3) * 4 + ((i + di) & 3)])
          dot(d, g, i + di, j + dj, ink.glow)
      }
  }
  const beat = Math.floor(clock * 4)

  // The ladder, rattling while the portal is open, and the inspector a few rungs from the top.
  const [foot, top] = LADDER
  const wob = heat > 0.2 ? Math.sin(clock * 14) * heat * 5 : 0
  const tip: Pt = [top[0] + wob, top[1]]
  const len = Math.hypot(tip[0] - foot[0], tip[1] - foot[1])
  const ux = (tip[0] - foot[0]) / len
  const uy = (tip[1] - foot[1]) / len
  const rail = (s: number, side: number): Pt => [
    foot[0] + ux * s - uy * 6 * side,
    foot[1] + uy * s + ux * 6 * side,
  ]
  seg(rail(0, 1), rail(len, 1), ink.wood)
  seg(rail(0, -1), rail(len, -1), ink.wood)
  for (let s = 10; s < len; s += 16) seg(rail(s, 1), rail(s, -1), ink.wood)
  const rung = 10 + 16 * Math.round((len * (0.56 + 0.08 * Math.sin(clock * 0.3)) - 10) / 16)
  const [li, lj] = figure(
    'ladder',
    REACH,
    foot[0] + ux * rung,
    foot[1] + uy * rung,
    false,
    letter(ink.coats[0]),
  )
  // The glass: a ring on the raised hand that glints on the beat.
  for (const [di, dj] of [
    [4, -1],
    [5, -2],
    [6, -1],
    [5, 0],
  ])
    dot(d, g, li + di + 1, lj + dj + 1, ink.ink)
  dot(d, g, li + 5, lj + 1, ink.ink)
  dot(d, g, li + 6, lj, beat % 3 === 0 ? ink.paper : ink.gold)

  // The rope off the keystone, and whoever is on the end of it, with a lantern.
  const hands: Pt = [ANCHOR[0] + Math.sin(c.swing) * c.len, ANCHOR[1] + Math.cos(c.swing) * c.len]
  seg(ANCHOR, hands, ink.wood)
  const hi = col(g, hands[0])
  const hj = row(g, hands[1])
  line(d, g, hi, hj + 9, hi + (c.swing > 0 ? -1 : 1), hj + 12, ink.wood)
  figure('rope', HANG, hands[0], 0, false, letter(ink.coats[1]), hj)
  if (ink.dark) halo(hi + 2, hj + 6, 2, 0.7)
  dot(d, g, hi + 2, hj + 6, beat % 2 ? ink.flame : ink.gold)

  // The fisher on the one o'clock stone.
  const [fi, fj] = figure('fisher', SIT, SEAT[0], SEAT[1] + 1 * g.px, true, letter(ink.coats[2]))
  seg([g.sx[0] + fi * g.px, g.sy[0] + (fj + 4) * g.px], TIP, ink.wood)
  const bite = c.fishT >= 0 && c.fishT < 0.6 ? (beat % 2) * 2 : 0
  const idle: Pt = [BOB[0], BOB[1] + (Math.sin(clock * 1.7) > 0.4 ? g.px : 0) + bite * g.px]
  const swirl = clock * 2.4
  const bob: Pt = [
    idle[0] + (GATE.x + Math.cos(swirl) * (70 - 40 * heat) - idle[0]) * heat,
    idle[1] + (GATE.y + Math.sin(swirl) * (70 - 40 * heat) - idle[1]) * heat,
  ]
  seg(TIP, bob, ink.wood)
  dot(d, g, col(g, bob[0]), row(g, bob[1]), ink.rose)
  dot(d, g, col(g, bob[0]), row(g, bob[1]) + 1, ink.paper)
  if (c.fishT >= 0.6) {
    const u = (c.fishT - 0.6) / 1.2
    const [a, b, k] = u < 0.5 ? [bob, TIP, u / 0.5] : [TIP, JAR, (u - 0.5) / 0.5]
    const si = col(g, a[0] + (b[0] - a[0]) * k)
    const sj = row(g, a[1] + (b[1] - a[1]) * k)
    if (ink.dark) halo(si, sj, 2, 0.6)
    dot(d, g, si, sj, ink.gold)
  }
  const ji = col(g, JAR[0])
  const jj = row(g, JAR[1]) - 1
  const lit = Math.min(3, Math.ceil((c.stars * 3) / 5))
  if (ink.dark && lit) halo(ji, jj - 2, 2 + (lit > 2 ? 1 : 0), 0.35 + 0.15 * lit)
  for (let r = 0; r < 4; r++) {
    dot(d, g, ji - 1, jj - r, ink.paper)
    dot(d, g, ji + 1, jj - r, ink.paper)
  }
  dot(d, g, ji, jj, ink.paper)
  for (let r = 1; r <= 3; r++) if (r <= lit) dot(d, g, ji, jj - r, ink.gold)

  // The child on the sill, and the stick.
  const s = c.stick
  const [ki, kj] = figure(
    'kid',
    s.state === 0 ? KID : KID_SIT,
    CHILD[0],
    CHILD[1],
    false,
    letter(ink.coats[3], ink.rose),
  )
  if (s.state === 0) {
    const back = beat % 2 ? g.px : 0
    const dx = POKE[0] - CHILD[0]
    const dy = POKE[1] - CHILD[1]
    const n = Math.hypot(dx, dy)
    line(
      d,
      g,
      ki + 2,
      kj + 2,
      col(g, POKE[0] - (dx / n) * back),
      row(g, POKE[1] - (dy / n) * back),
      ink.wood,
    )
  } else if (s.state === 1) {
    const x = GATE.x + Math.cos(s.a) * s.r
    const y = GATE.y + Math.sin(s.a) * s.r
    const tx = -Math.sin(s.a) * 2 * g.px
    const ty = Math.cos(s.a) * 2 * g.px
    seg([x - tx, y - ty], [x + tx, y + ty], ink.wood)
  }

  // The surveyor at the theodolite on the slope below the ring.
  const head = ground(TRIPOD) - 20
  const ti = col(g, TRIPOD)
  const tj = row(g, head)
  for (const o of [-7, 0, 7]) seg([TRIPOD, head], [TRIPOD + o, ground(TRIPOD + o) + 2], ink.wood)
  dot(d, g, ti - 1, tj, ink.ink)
  dot(d, g, ti, tj, ink.ink)
  dot(d, g, ti, tj - 1, ink.ink)
  dot(d, g, ti + 1, tj, ink.gold)
  figure(
    'surveyor',
    c.peer ? PEER : STAND,
    SURVEYOR + (c.peer ? g.px : 0),
    ground(SURVEYOR) + 4,
    false,
    letter(ink.coats[4]),
  )

  // The note-taker in the grass, facing the gate, pencil going.
  const [wi, wj] = figure('writer', WRITE, WRITER, ground(WRITER) + 4, true, letter(ink.coats[5]))
  dot(d, g, wi + (beat % 2), wj + 4, ink.ink)

  // Someone pointing.
  const pointing = heat > 0.3 || Math.floor(clock / 0.8) % 3 !== 0
  figure(
    'pointer',
    pointing ? REACH : STAND,
    POINTER,
    ground(POINTER) + 4,
    false,
    letter(ink.coats[1], ink.rose),
  )

  // Someone on the bank offering the cat a fish: held out while it peeks, dropped when it wakes.
  const floor = ground(FRIEND) + 4
  const scared = alert === 2
  const fx = FRIEND - (scared ? 8 : 0)
  const [ri, rj] = figure(
    'friend',
    scared ? ALARM : alert === 1 ? REACH : STAND,
    fx,
    floor,
    false,
    letter(ink.coats[2]),
  )
  const paintFish = (i: number, j: number) =>
    stamp(d, g, FISH, i, j, false, ch => (ch === 'F' ? ink.fish : null))
  if (scared) paintFish(ri + 6, row(g, floor) - 3)
  else if (alert === 1) paintFish(ri + 5, rj)
  else paintFish(ri + 4, rj + 4)
}

// Speech bubbles over whoever is talking, drawn after the reflection so they stay out of the water.
// They blink out over their last third of a second.
export function paintBubbles(d: Uint8ClampedArray, g: Grid, v: Voices, paper: RGB, ink: RGB) {
  for (const b of v.bubbles) {
    const head = v.heads[b.who]
    if (!head || (b.age > 1.5 && Math.floor(b.age * 10) % 2)) continue
    const [hi, hj] = head
    const i0 = hi - 1
    const j0 = hj - 9
    // A paper glyph on an ink plate. An ink glyph inside an ink-edged paper box gives three
    // 1 px ink strokes with paper between them, and the paper reads as an "H". The paper halo,
    // chamfered at the corners, keeps the plate apart from the ruin's own keylines.
    for (let y = -1; y <= 7; y++)
      for (let x = -1; x <= 5; x++)
        if (Math.min(x + 1, 5 - x) + Math.min(y + 1, 7 - y) > 1) dot(d, g, i0 + x, j0 + y, paper)
    dot(d, g, i0, j0 + 7, paper)
    dot(d, g, i0 + 1, j0 + 8, paper)
    for (let y = 0; y < 7; y++)
      for (let x = 0; x < 5; x++)
        if (!((x === 0 || x === 4) && (y === 0 || y === 6))) dot(d, g, i0 + x, j0 + y, ink)
    dot(d, g, i0 + 1, j0 + 7, ink)
    stamp(d, g, GLYPHS[b.glyph], i0 + 1, j0 + 1, false, () => paper)
  }
}
