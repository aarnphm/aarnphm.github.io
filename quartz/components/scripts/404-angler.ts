// The farmer keeps a rowboat pulled up at the island's left foot, under the brow the house sits
// behind (404-herd.ts). On a fair day with the stock out, they come down over the brow, push off,
// row the length of the river to the deep water at the bottom right, and fish it from the boat: a
// cast, the float riding the swell, a nibble or two, the bite and the strike, and the rod bowed
// while the fish runs. Most come in on the line. A big one runs until it jumps clear of the water,
// throws the hook and is gone, and the farmer swears. What comes in goes back over the side or,
// when the stone cat is lying at home, is thrown up the bank to it (404-cat.ts). After a while they
// row back, pull the boat up and climb home over the brow, and come out of the house on the far
// side with feed for the cows.

import type { Glyph } from './404-crew'
import { type Grid, type Pt, type RGB, col, row } from './404-pixel'

type Doing = 'home' | 'down' | 'launch' | 'row' | 'turn' | 'fish' | 'beach' | 'up'
// At the rod: the cast, the float out, the strike, a fish on, a big one leaping, the line gone
// slack, a fish lifted out, held up, thrown to the cat, let go over the side, and a fresh bait.
type Rod =
  | 'cast'
  | 'wait'
  | 'strike'
  | 'fight'
  | 'jump'
  | 'snap'
  | 'lift'
  | 'hold'
  | 'toss'
  | 'release'
  | 'bait'

export type Angler = {
  doing: Doing
  // The boat's middle at its waterline, and the farmer's feet while ashore.
  x: number
  y: number
  walk: Pt
  face: number
  // The boat's bow. At the oars the farmer sits facing the stern; fishing, they face the island.
  bow: number
  t: number
  way: Pt[]
  // Seconds left at the house, or of fishing; bound for home; oars shipped for a swimmer.
  wait: number
  done: boolean
  rest: boolean
  rod: Rod
  rt: number
  float: Pt
  // Seconds into the wait when a fish takes, and the nibbles before it.
  bite: number
  nibbles: number[]
  // The fish on: its weight in pounds, where it is, whether it gets off, and the throw.
  size: number
  fish: Pt
  loose: boolean
  from: Pt
  to: Pt | null
  // The river in view: where the boat is pulled up and fished from, the lane it is rowed along,
  // and the waterline.
  spot: number
  lane: number
  water: number
}

export type AnglerEnv = {
  // A fair day with the stock out, and the farmer at the house rather than at the barn.
  fit: boolean
  home: boolean
  // Where the triathlete swims, if they are in the water.
  swimmer: number | null
  // Where a fish thrown to the cat should land, while it is lying at home with nothing to eat.
  cat: Pt | null
  say(glyph: Glyph): void
  splash(x: number, y: number, size: number): void
  toss(to: Pt): void
  landed(at: Pt): void
  // Home over the brow, with the catch put by.
  back(): void
}

export type AnglerInk = {
  hat: RGB
  skin: RGB
  shirt: RGB
  bib: RGB
  trousers: RGB
  boots: RGB
  hull: RGB
  keel: RGB
  wood: RGB
  rod: RGB
  line: RGB
  float: RGB
  cork: RGB
  fish: RGB
  deep: RGB
  foam: RGB
}

// The boat is pulled up at the island's left foot. The way up runs from beside it along the slope
// under the crest to the brow and over, where the crest hides them.
const LAND = 262
const UP: Pt[] = [
  [298, 866],
  [312, 803],
  [360, 767],
  [440, 737],
  [500, 701],
  [540, 662],
  [562, 704],
]
// Past this point of the way the farmer is behind the crest.
const BROW = UP.length - 2
// Scene units per second hiking and rowing, and seconds for a stroke of the oars.
const HIKE = 14
const ROW = 30
const STROKE = 1.1
// Pounds. Most are under one; past this the line will not hold one for long.
const LIMIT = 2.2
// Seconds the arm goes back before a throw.
const WIND = 0.4

const gauss = () =>
  Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())
const wait = (mean: number) => -Math.log(1 - Math.random()) * mean

// With motion they have been out a while by the time anyone looks, as often as not. Without it,
// they are fishing whenever they would be.
export function makeAngler(fit: boolean, reduce: boolean): Angler {
  const a: Angler = {
    doing: 'home',
    x: LAND,
    y: 875,
    walk: [UP[0][0], UP[0][1]],
    face: -1,
    bow: -1,
    t: 0,
    way: [],
    wait: 30 + Math.random() * 150,
    done: false,
    rest: false,
    rod: 'wait',
    rt: 0,
    float: [0, 0],
    bite: 0,
    nibbles: [],
    size: 1,
    fish: [0, 0],
    loose: false,
    from: [0, 0],
    to: null,
    spot: 1385,
    lane: 904,
    water: 870,
  }
  if (fit && (reduce || Math.random() < 0.7)) {
    fishing(a, reduce ? Infinity : 60 + Math.random() * 300)
    if (reduce) a.bite = Infinity
  }
  return a
}

function fishing(a: Angler, left: number) {
  a.doing = 'fish'
  a.x = a.spot
  a.y = a.lane
  a.bow = 1
  a.face = -1
  a.wait = left
  a.done = false
  a.float = [a.x - 62 - Math.random() * 34, a.lane - 3 + Math.random() * 8]
  arm(a)
}

// The float out and settled: a fish takes after a wait, with a nibble or two first.
function arm(a: Angler) {
  a.rod = 'wait'
  a.rt = 0
  a.bite = 5 + wait(16)
  a.nibbles = Array.from({ length: Math.floor(Math.random() * 3) }, () => Math.random() * a.bite)
}

// The stretch of river the viewport shows: the boat is fished from short of the right-hand tree,
// along a lane a little further out than the swimmer's.
export function layAngler(a: Angler, g: Grid, water: number) {
  const spot = Math.min(1385, g.sx[g.bw - 1] - 160)
  const lane = water + Math.min(34, (g.bottom - water) * 0.42)
  const shift = spot - a.spot
  const lift = lane - a.lane
  a.way = a.way.map(([x, y]): Pt => [x === a.spot ? spot : x, y === a.lane ? lane : y])
  if (a.doing === 'fish') {
    a.x += shift
    a.y = lane
    a.float = [a.float[0] + shift, a.float[1] + lift]
    a.fish = [a.fish[0] + shift, a.fish[1] + lift]
  }
  a.spot = spot
  a.lane = lane
  a.water = water
}

// On along the way at a speed, true on reaching its end.
function along(p: Pt, a: Angler, v: number, dt: number, boat: boolean) {
  let left = v * dt
  while (a.way.length && left > 0) {
    const [tx, ty] = a.way[0]
    const dx = tx - p[0]
    const dy = ty - p[1]
    const d = Math.hypot(dx, dy)
    if (Math.abs(dx) > 1) {
      if (boat) a.bow = Math.sign(dx)
      else a.face = Math.sign(dx)
    }
    if (d <= left) {
      p[0] = tx
      p[1] = ty
      left -= d
      a.way.shift()
    } else {
      p[0] += (dx / d) * left
      p[1] += (dy / d) * left
      left = 0
    }
  }
  return !a.way.length
}

export function stepAngler(a: Angler, dt: number, env: AnglerEnv) {
  a.t += dt
  switch (a.doing) {
    case 'home':
      if (!env.fit || !env.home || (a.wait -= dt) > 0) break
      a.doing = 'down'
      a.t = 0
      a.way = UP.slice(0, -1).reverse()
      a.walk = [UP[UP.length - 1][0], UP[UP.length - 1][1]]
      break
    case 'down':
    case 'up':
      if (!along(a.walk, a, HIKE, dt, false)) break
      a.t = 0
      if (a.doing === 'down') {
        a.doing = 'launch'
        a.face = -1
      } else {
        a.doing = 'home'
        a.wait = 150 + Math.random() * 240
        env.back()
      }
      break
    case 'launch':
      if (a.t < 1.6) break
      a.doing = 'row'
      a.t = 0
      a.bow = 1
      a.x = LAND
      a.y = a.water + 5
      a.way = [
        [LAND + 44, a.lane],
        [a.spot, a.lane],
      ]
      break
    case 'row': {
      // Oars shipped while the swimmer goes by.
      a.rest = env.swimmer !== null && Math.abs(env.swimmer - a.x) < 90 && a.y > a.water + 12
      if (a.rest) break
      // Quicker through the drive than the recovery.
      const surge = Math.floor((a.t / STROKE) * 4) % 4 < 2 ? 1.3 : 0.7
      const p: Pt = [a.x, a.y]
      const there = along(p, a, ROW * surge, dt, true)
      a.x = p[0]
      a.y = p[1]
      if (!there) break
      a.t = 0
      if (a.done) {
        a.doing = 'beach'
        a.done = false
        a.walk = [UP[0][0], UP[0][1]]
        a.face = -1
      } else fishing(a, 240 + Math.random() * 200)
      break
    }
    case 'turn':
      // Swung round on one oar.
      if (a.t > 0.7) a.bow = -1
      if (a.t < 1.4) break
      a.doing = 'row'
      a.t = 0
      a.bow = -1
      a.way = [
        [LAND + 44, a.lane],
        [LAND, a.water + 5],
      ]
      break
    case 'beach':
      if (a.t < 1.4) break
      a.doing = 'up'
      a.t = 0
      a.way = UP.slice(1)
      break
    case 'fish':
      stepRod(a, dt, env)
      break
  }
}

function stepRod(a: Angler, dt: number, env: AnglerEnv) {
  if (!env.fit) a.wait = Math.min(a.wait, 0)
  a.wait -= dt
  const was = a.rt
  a.rt += dt
  const go = (rod: Rod) => {
    a.rod = rod
    a.rt = 0
  }
  switch (a.rod) {
    case 'bait':
      if (a.rt < 2.4) break
      if (a.wait <= 0) {
        a.doing = 'turn'
        a.t = 0
        a.done = true
        break
      }
      go('cast')
      a.float = [a.x - 62 - Math.random() * 34, a.lane - 3 + Math.random() * 8]
      break
    case 'cast':
      if (a.rt < 1.2) break
      env.splash(a.float[0], a.float[1], 8)
      arm(a)
      break
    case 'wait':
      // Reeled in to go home, or a fish takes.
      if (a.wait <= 0) go('bait')
      else if (a.rt >= a.bite) go('strike')
      break
    case 'strike':
      if (a.rt < 0.5) break
      // Missed it: the float comes back up and they wait on.
      if (Math.random() < 0.25) {
        arm(a)
        break
      }
      a.size = Math.exp(gauss() * 0.9)
      a.loose = Math.random() < (a.size > LIMIT ? 0.8 : 0.08)
      a.fish = [a.float[0], a.float[1]]
      if (a.size > LIMIT) env.say('!')
      go('fight')
      break
    case 'fight': {
      // It runs across and back, and is worked in toward the boat.
      const length = Math.min(16, 2.5 + 2.2 * a.size)
      const u = Math.min(1, a.rt / length)
      const k = 1 - 0.8 * u
      const run = Math.sin(a.rt * (1.1 + 0.8 / a.size)) * 6 * Math.min(a.size, 3) * k
      const near = a.x - 22
      a.fish = [near - (near - a.float[0]) * k + run, a.float[1] + Math.cos(a.rt * 1.7) * 3 * k]
      if (a.loose && a.size > LIMIT && u > 0.6) {
        a.from = [a.fish[0], a.fish[1]]
        go('jump')
      } else if (a.loose && a.size <= LIMIT && u > 0.45) {
        env.say('?')
        go('snap')
      } else if (u >= 1) {
        env.splash(a.fish[0], a.fish[1], 10)
        go('lift')
      }
      break
    }
    case 'jump':
      if (a.rt < 0.8) break
      env.splash(a.from[0], a.from[1], 34)
      env.say('#')
      go('snap')
      break
    case 'snap':
      if (a.rt > 2.6) go('bait')
      break
    case 'lift':
      if (a.rt > 1.1) go('hold')
      break
    case 'hold':
      if (a.rt < 1.4) break
      // Back it goes, unless the cat is lying at home and it is a cat's dinner.
      if (env.cat && a.size < 1.8 && Math.random() < 0.6) {
        a.to = env.cat
        go('toss')
      } else go('release')
      break
    case 'toss':
      if (!a.to) {
        go('bait')
        break
      }
      if (was < WIND && a.rt >= WIND) {
        a.from = [a.x - 8, a.lane - 40]
        env.toss(a.to)
      } else if (a.rt >= WIND + flight(a)) {
        env.landed(a.to)
        a.to = null
        go('bait')
      }
      break
    case 'release':
      if (was < 0.7 && a.rt >= 0.7) env.splash(a.x + 16 * a.face, a.lane + 2, 10)
      if (a.rt > 1.6) go('bait')
      break
  }
}

const flight = (a: Angler) => (a.to ? 1.1 + Math.abs(a.to[0] - a.from[0]) / 500 : 0)

// Where a thrown fish is, part way through its flight, high over the bank.
function arc(a: Angler, u: number): Pt {
  const [x0, y0] = a.from
  const [x1, y1] = a.to!
  const h = 50 + 0.2 * Math.abs(x1 - x0)
  return [x0 + (x1 - x0) * u, y0 + (y1 - y0) * u - 4 * h * u * (1 - u)]
}

// Sprites face right: straw hat (H), skin (S), shirt (C), the bib of the overalls (B), their legs
// (D), boots (K). Sat in the boat, the gunwale hides the rest.
const STAND = ['.HH.', 'HHHH', '.SS.', 'CBBC', '.BB.', '.DD.', '.DD.', '.KK.']
const STRIDE = ['.HH.', 'HHHH', '.SS.', 'CBBC', '.BB.', '.DD.', 'D..D', 'K..K']
const SEAT = ['.HH.', 'HHHH', '.SS.', 'CBBC', '.BB.']
// The rowboat side on, bow to the right: the stem rising, the gunwale, and the planking down to
// the waterline, cut back under the bow.
const BOAT = ['...........W', 'WWWWWWWWWWWW', 'RRRRRRRRRRR.', '.rrrrrrrrr..']
// The catch, by size, side on with the head to the right; hung from the line, head up.
const FISH = [
  ['F.FF.', '.FFFF', 'F.FF.'],
  ['F..FF.', '.FFFFF', 'F..FF.'],
  ['F...FFF.', '.FFFFFFF', 'F..FFFF.'],
]
const HUNG = [
  ['.F.', 'FFF', '.F.', 'F.F'],
  ['.F.', 'FFF', 'FFF', '.F.', 'F.F'],
]
// The blade of the oar in the water through a stroke, from the catch to the recovery, relative to
// the rowlock with the bow to the right; the handle in the farmer's hands is opposite it.
const OAR: Pt[] = [
  [5, 3],
  [1, 4],
  [-4, 3],
  [2, -1],
]

// Returns the top-middle pixel of the farmer's head for a speech bubble, or null while they are out
// of sight.
export function paintAngler(
  d: Uint8ClampedArray,
  g: Grid,
  a: Angler,
  ink: AnglerInk,
  clock: number,
  crest: (x: number) => number,
): Pt | null {
  // Over the brow, the crest hides whatever of them is below it.
  const behind = (a.doing === 'up' || a.doing === 'down') && a.walk[0] > UP[BROW][0] - 2
  const hidden = (i: number, j: number) => behind && g.sy[j] > crest(g.sx[i]) + 1
  const dot = (i: number, j: number, c: RGB) => {
    if (i < 0 || j < 0 || i >= g.bw || j >= g.bh || hidden(i, j)) return
    const k = (j * g.bw + i) * 4
    d[k] = c[0]
    d[k + 1] = c[1]
    d[k + 2] = c[2]
    d[k + 3] = 255
  }
  const seg = (i0: number, j0: number, i1: number, j1: number, c: RGB) => {
    const n = Math.max(Math.abs(i1 - i0), Math.abs(j1 - j0), 1)
    for (let s = 0; s <= n; s++)
      dot(Math.round(i0 + ((i1 - i0) * s) / n), Math.round(j0 + ((j1 - j0) * s) / n), c)
  }
  const letter = (ch: string): RGB =>
    ch === 'H'
      ? ink.hat
      : ch === 'S'
        ? ink.skin
        : ch === 'C'
          ? ink.shirt
          : ch === 'B'
            ? ink.bib
            : ch === 'D'
              ? ink.trousers
              : ch === 'K'
                ? ink.boots
                : ch === 'W'
                  ? ink.wood
                  : ch === 'R'
                    ? ink.hull
                    : ch === 'r'
                      ? ink.keel
                      : ink.fish
  const sprite = (rows: readonly string[], i0: number, j0: number, flip: boolean, lean = 0) => {
    const w = rows[0].length
    rows.forEach((text, r) => {
      // Leaning shifts the head and shoulders a pixel.
      const s = r < 3 ? lean : 0
      for (let c = 0; c < w; c++) {
        const ch = text[flip ? w - 1 - c : c]
        if (ch !== '.') dot(i0 + c + s, j0 + r, letter(ch))
      }
    })
  }
  const fishAt = (i: number, j: number, size: number, flip: boolean, hung = false) => {
    const rows = hung ? HUNG[size < 1 ? 0 : 1] : FISH[size < 1 ? 0 : size < LIMIT ? 1 : 2]
    sprite(rows, i - (rows[0].length >> 1), j - (rows.length >> 1), flip)
  }
  const boat = (x: number, y: number, bow: number, afloat: boolean) => {
    const ic = col(g, x)
    const jw = row(g, y)
    sprite(BOAT, ic - 6, jw - 3, bow < 0)
    if (!afloat) return
    // Its shadow on the water, broken by the swell.
    const beat = Math.floor(clock * 3)
    for (let s = -4; s <= 4; s++) if ((s + beat) % 5) dot(ic + s, jw + 1, ink.deep)
  }

  const beat = Math.floor(clock * 6)
  switch (a.doing) {
    case 'home':
      boat(LAND, a.water + 5, -1, false)
      return null
    case 'down':
    case 'up':
    case 'launch':
    case 'beach': {
      boat(LAND, a.water + 5, -1, false)
      const i = col(g, a.walk[0])
      const j = row(g, a.walk[1])
      const walking = (a.doing === 'down' || a.doing === 'up') && Math.floor(a.t * 2.5) % 2 === 1
      sprite(walking ? STRIDE : STAND, i - 2, j - 8, a.face < 0)
      if (a.doing === 'launch' || a.doing === 'beach') {
        // A hand on the stern.
        dot(i - 3, j - 4, ink.skin)
        return [i, j - 8]
      }
      // The rod over the shoulder, butt forward.
      seg(i + a.face, j - 4, i - 5 * a.face, j - 13, ink.rod)
      return behind && g.sy[j - 8] > crest(a.walk[0]) ? null : [i, j - 8]
    }
    case 'row':
    case 'turn': {
      const ic = col(g, a.x)
      const jw = row(g, a.y)
      const face = -a.bow
      const phase = a.rest || a.doing === 'turn' ? 3 : Math.floor((a.t / STROKE) * 4) % 4
      // Reaching to the stern at the catch, lying back at the finish.
      const lean = phase === 0 ? face : phase === 2 ? -face : 0
      sprite(SEAT, ic - 2, jw - 7, face < 0, lean)
      boat(a.x, a.y, a.bow, true)
      const [bi, bj] = OAR[phase]
      const lock: Pt = [ic, jw - 2]
      const blade: Pt = [lock[0] + bi * a.bow, lock[1] + bj]
      const hands: Pt = [lock[0] - Math.round(bi * a.bow * 0.4), lock[1] - 2]
      seg(hands[0], hands[1], blade[0], blade[1], ink.wood)
      dot(hands[0], hands[1], ink.skin)
      if (phase === 2) dot(blade[0] - a.bow, blade[1], ink.foam)
      // The wake opening astern.
      if (!a.rest && a.doing === 'row' && a.y > a.water + 12)
        for (let s = 2; s <= 10; s++)
          if ((s + beat) % 3) dot(ic - a.bow * (6 + s), jw + 1 + (s >> 2), ink.foam)
      return [ic, jw - 7]
    }
    case 'fish':
      break
  }

  // Fishing over the stern, facing the island.
  const ic = col(g, a.x)
  const bob = Math.floor(clock * 0.9) % 2
  const jw = row(g, a.y) + bob
  const face = a.face
  const u = a.rt
  const lean = a.rod === 'release' && u > 0.3 && u < 1.2 ? face : 0
  sprite(SEAT, ic - 2, jw - 7 + (lean ? 1 : 0), face < 0, lean)
  boat(a.x, a.y + bob * g.px, a.bow, true)
  const hand: Pt = [ic + 3 * face, jw - 4]
  // Where the rod's tip is for each moment at the rod, relative to the hand.
  let tip: Pt = [9 * face, -7]
  if (a.rod === 'cast') tip = u < 0.4 ? [-4 * face, -10] : u < 0.55 ? [10 * face, -5] : tip
  else if (a.rod === 'strike') tip = [6 * face, -10]
  else if (a.rod === 'bait' || a.rod === 'hold' || a.rod === 'toss' || a.rod === 'release')
    tip = [7 * face, -2]
  else if (a.rod === 'lift') tip = [5 * face, -11]
  else if (a.rod === 'snap' && u < 0.6) tip = [9 * face, -7 + (beat % 2 ? -1 : 1)]
  // Bowed toward the fish while it runs.
  const on = a.rod === 'fight' || a.rod === 'jump'
  const fi = col(g, on && a.rod === 'jump' ? jumpAt(a)[0] : a.fish[0])
  const fj = row(g, on && a.rod === 'jump' ? jumpAt(a)[1] : a.fish[1])
  if (on) {
    const dx = fi - hand[0]
    const dy = fj - hand[1]
    const pull = Math.min(0.75, 0.35 + 0.12 * a.size) + (beat % 2) * 0.08
    const len = Math.hypot(tip[0], tip[1])
    const to = Math.hypot(dx, dy) || 1
    const bx = tip[0] + ((dx / to) * len - tip[0]) * pull
    const by = tip[1] + ((dy / to) * len - tip[1]) * pull
    const k = (len * (1 - 0.15 * pull)) / (Math.hypot(bx, by) || 1)
    tip = [Math.round(bx * k), Math.round(by * k)]
  }
  // The butt half keeps its line; the tip half takes the bend.
  const butt = on ? [9 * face, -7] : tip
  const mid: Pt = [hand[0] + Math.round(butt[0] / 2), hand[1] + Math.round(butt[1] / 2)]
  const end: Pt = [hand[0] + tip[0], hand[1] + tip[1]]
  seg(hand[0], hand[1], mid[0], mid[1], ink.rod)
  seg(mid[0], mid[1], end[0], end[1], ink.rod)
  dot(hand[0], hand[1], ink.skin)

  const flt: Pt = [col(g, a.float[0]), row(g, a.float[1])]
  switch (a.rod) {
    case 'cast': {
      if (u < 0.55) break
      // The float flies out on its line and drops.
      const v = Math.min(1, (u - 0.55) / 0.55)
      const fx = Math.round(end[0] + (flt[0] - end[0]) * v)
      const fy = Math.round(end[1] + (flt[1] - end[1]) * v - 10 * v * (1 - v))
      seg(end[0], end[1], fx, fy, ink.line)
      dot(fx, fy, ink.float)
      break
    }
    case 'wait': {
      const nib = a.nibbles.some(n => u > n && u < n + 0.3)
      const dip = nib ? 1 : 0
      seg(end[0], end[1], flt[0], flt[1] - 1, ink.line)
      dot(flt[0], flt[1] - 1 + dip, ink.float)
      if (!dip) dot(flt[0], flt[1], ink.cork)
      else dot(flt[0] + 1, flt[1] + 1, ink.foam)
      break
    }
    case 'strike':
      // Under, and the water closing over it.
      seg(end[0], end[1], flt[0], flt[1], ink.line)
      dot(flt[0] - 1, flt[1], ink.foam)
      dot(flt[0] + 1, flt[1], ink.foam)
      break
    case 'fight': {
      seg(end[0], end[1], fi, fj, ink.line)
      const w = Math.min(3, 1 + Math.round(a.size))
      for (let s = 0; s < w; s++) dot(fi - (w >> 1) + s, fj + 1, ink.deep)
      // Thrashing at the surface now and then.
      if (beat % 4 === 0) {
        dot(fi - 2, fj, ink.foam)
        dot(fi + 2, fj, ink.foam)
      }
      break
    }
    case 'jump':
      seg(end[0], end[1], fi, fj, ink.line)
      fishAt(fi, fj, a.size, beat % 2 === 0)
      if (u < 0.2)
        for (const s of [-2, 0, 2]) dot(col(g, a.from[0]) + s, row(g, a.from[1]), ink.foam)
      break
    case 'lift': {
      // Hung off the tip, kicking.
      const kick = beat % 2 ? 1 : -1
      seg(end[0], end[1], end[0] + kick, end[1] + 2, ink.line)
      fishAt(end[0] + kick, end[1] + 4, a.size, false, true)
      break
    }
    case 'hold':
      // Held up to look at.
      fishAt(ic + 5 * face, jw - 8, a.size, face < 0)
      break
    case 'toss':
      if (!a.to) break
      if (u < WIND) {
        // The arm back over the shoulder.
        fishAt(ic - face, jw - 10, a.size, face < 0, true)
        dot(ic - face, jw - 8, ink.skin)
      } else {
        const v = Math.min(1, (u - WIND) / flight(a))
        const [x, y] = arc(a, v)
        fishAt(col(g, x), row(g, y), a.size, beat % 2 === 0, beat % 4 < 2)
        dot(ic + 2 * face, jw - 7, ink.skin)
      }
      break
    case 'release':
      if (u < 0.7) fishAt(ic + 5 * face, jw - 1, a.size, face < 0)
      else if (u < 1.2) {
        // A dark shape going down and away.
        const k = Math.round((u - 0.7) * 16)
        dot(ic + (5 + k) * face, jw + 1 + (k >> 2), ink.deep)
        dot(ic + (6 + k) * face, jw + 1 + (k >> 2), ink.deep)
      }
      break
    case 'snap':
      // A big one gone: both fists up, then a while sat looking at where it went.
      if (u < 1.2 && a.size > LIMIT) {
        dot(ic - 2, jw - 9 - (beat % 2), ink.skin)
        dot(ic + 1, jw - 9 - ((beat + 1) % 2), ink.skin)
      }
      break
  }
  return [ic, jw - 7]
}

// The big one in the air, clear of the water and shaking.
function jumpAt(a: Angler): Pt {
  const u = Math.min(1, a.rt / 0.8)
  const h = 14 + 6 * Math.min(a.size, 4)
  return [a.from[0] + 8 * u * a.face, a.from[1] - 4 * h * u * (1 - u)]
}
