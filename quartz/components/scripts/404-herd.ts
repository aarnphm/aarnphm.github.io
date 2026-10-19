// The stock on the bike park's hill (404-riders.ts), out on the grass inside the line's middle
// turns from the thaw until the first snow: four cows, two of them with a calf at foot, and three
// black-faced ewes that keep to the upper end. It is the same herd every day. The cattle graze along
// the slope a mouthful at a time, walk off to fresh grass now and then, and lie down to chew the
// cud through the heat of the day, mostly together. They keep to their own side of the line, and
// the riders swerve past whoever grazes at its edge. A bike going by close makes a cow look up and
// the ewes bolt; someone lying hurt on the line brings the cows over to stand and stare until they
// are gone. At sundown they walk in file behind the tall cypress to the barn, with the farmer up
// from the house below the island's brow to see them in and shut the doors on them, and come out
// again after sunrise once the farmer has come back up to open them; through the winter they stay in.
// When the farmer is home from fishing (404-angler.ts) they make up feed and carry it out to the
// trough at the lane's mouth, and the cattle come over and take their turns at it.

import { type Almanac, type Sky, winter } from './404-almanac'
import { PHOTO, UNIT, nearCourse, slant } from './404-downhill'
import { type Grid, type Pt, type RGB, col, dot, mulberry, noise, row } from './404-pixel'

export type HerdEnv = {
  // Out on the pasture (not in winter, and only while the sun is up), the solar hour, and whether
  // it is hot enough to lie up through the day.
  out: boolean
  hour: number
  hot: boolean
}

export function herdEnv(al: Almanac, sky: Sky): HerdEnv {
  return {
    out: !winter(al) && al.sun.alt > 2,
    hour: al.hour,
    hot: sky.temp !== null && sky.temp > 26,
  }
}

type Kind = 'cow' | 'calf' | 'sheep'
type Doing = 'graze' | 'walk' | 'look' | 'lie' | 'run' | 'eat'

export type Beast = {
  kind: Kind
  // Coat and head as indices into the inks, and which pixels of the body carry the second colour.
  coat: number
  head: number
  spots: number
  // A calf's dam, by its place in the herd.
  dam: number
  x: number
  y: number
  face: number
  doing: Doing
  t: number
  until: number
  to: Pt | null
  // What it is looking at; someone hurt it has come over to stand and stare at; seconds before a
  // passing bike can startle it again.
  at: Pt | null
  watch: Pt | null
  calm: number
  // In the barn, and the rest of its way into it or out of it to the grass.
  in: boolean
  path: Pt[]
  // The place at the trough it is headed for or eating at, and the last feed it had there.
  spot: number
  fed: number
}

// The farmer: at the house, on the way to a point and what to do there, at the barn's doors swinging
// them a leaf at a time, stood clear of them while the stock go out or come in, making up feed in
// the house, or tipping it into the trough.
type Chore = 'home' | 'walk' | 'doors' | 'watch' | 'make' | 'pour'
export type Farmer = {
  chore: Chore
  after: Chore
  way: Pt[]
  x: number
  y: number
  face: number
  t: number
}

// Whether the herd has been called out or in, the stock's seconds since through open doors, how
// many leaves of the doors stand open, the farmer, and the feed in the trough (1 full) and how many
// times it has been filled.
export type Herd = {
  seed: number
  rng: () => number
  beasts: Beast[]
  out: boolean
  since: number
  doors: number
  farmer: Farmer
  trough: number
  feeds: number
}

// Someone on the hill as the herd sees them: where they are, whether they are going by fast on a
// bike, and whether they are lying hurt.
export type Passer = { x: number; y: number; fast: boolean; hurt: boolean }

// Who the herd is: two black-and-white cows, a red one with a white face, a black one, a calf each
// for the first and the red one, and the three ewes. Coats are white, black, red.
const STOCK: Pick<Beast, 'kind' | 'coat' | 'head' | 'spots' | 'dam'>[] = [
  { kind: 'cow', coat: 0, head: 1, spots: 0b10000110, dam: -1 },
  { kind: 'cow', coat: 0, head: 1, spots: 0b00111001, dam: -1 },
  { kind: 'cow', coat: 2, head: 0, spots: 0, dam: -1 },
  { kind: 'cow', coat: 1, head: 1, spots: 0, dam: -1 },
  { kind: 'calf', coat: 0, head: 1, spots: 0b010, dam: 0 },
  { kind: 'calf', coat: 2, head: 0, spots: 0, dam: 2 },
  { kind: 'sheep', coat: 0, head: 1, spots: 0, dam: -1 },
  { kind: 'sheep', coat: 0, head: 1, spots: 0, dam: -1 },
  { kind: 'sheep', coat: 0, head: 1, spots: 0, dam: -1 },
]

// The barn on the open slope right of the tall cypress (the middle of its front, at the foot of the
// wall), and where a beast stands at its doors to go in or come out.
export const BARN: Pt = [460, 674]
const DOOR: Pt = [460, 678]
// The farmer's way: from the house along the slope below the island's brow, where the island hides
// a standing figure at any parallax, to stand clear of the doors on their right, and in front of
// them. A leaf takes this many seconds to swing.
const HOUSE: Pt = [548, 686]
const ASIDE: Pt = [484, 680]
const FRONT: Pt = [460, 680]
const STRIDE = 1.1 * UNIT
const SWING = 0.8
// The trough by the mouth of the lane, left of the cypress, with a place for a beast at either
// end; the way out to it along the lane, where the farmer stands behind it to tip the feed in; the
// seconds it takes to make the feed up, and the beast-seconds of eating a trough holds.
export const TROUGH: Pt = [356, 686]
const PLACES: Pt[] = [
  [TROUGH[0] - 20, TROUGH[1]],
  [TROUGH[0] + 20, TROUGH[1]],
]
const MOUTH: Pt = [400, 682]
const BEHIND: Pt = [TROUGH[0], TROUGH[1] - 5]
const MAKE = 12
const HOLDS = 110

// The pasture on a grid of four units, about a pixel: grass clear of the line, the paths and the
// photographer, in one piece with the lane along the slope behind the cypress to the barn's doors.
// The lines cut off the grass across them, so the stock stay on the side of the barn. The cattle
// range over the middle turns; the ewes keep above them.
const CELL = 4
// A beast stands this far from the middle of a line or path, so its body stays off the tread.
const CLEAR = 16
const BOX = { x0: 244, y0: 548, x1: 468, y1: 744 }
const NX = Math.ceil((BOX.x1 - BOX.x0) / CELL)
const NY = Math.ceil((BOX.y1 - BOX.y0) / CELL)
const AREA: Record<Kind, [number, number, number, number]> = {
  cow: [256, 598, 376, 736],
  calf: [256, 598, 376, 736],
  sheep: [282, 556, 376, 628],
}
// The lane keeps in front of the barn's wall, so nobody is painted over the barn from behind it.
const LANE = { x0: 370, y0: 676, x1: 466, y1: 700 }
const STEPS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
]
// Where a beast may set foot, and how many cells it is from there to the barn's doors.
let walk: Uint8Array | null = null
let home: Int16Array | null = null
function survey() {
  const open = new Uint8Array(NX * NY)
  const inside = ([x0, y0, x1, y1]: number[], x: number, y: number) =>
    x >= x0 && x <= x1 && y >= y0 && y <= y1
  for (let j = 0; j < NY; j++)
    for (let i = 0; i < NX; i++) {
      const x = BOX.x0 + (i + 0.5) * CELL
      const y = BOX.y0 + (j + 0.5) * CELL
      const ground =
        inside(AREA.cow, x, y) ||
        inside(AREA.sheep, x, y) ||
        inside([LANE.x0, LANE.y0, LANE.x1, LANE.y1], x, y)
      open[j * NX + i] =
        ground && !nearCourse(x, y, CLEAR) && Math.hypot(x - PHOTO[0], y - PHOTO[1]) > 12 ? 1 : 0
    }
  home = new Int16Array(NX * NY).fill(-1)
  const start = cell(DOOR[0], DOOR[1])
  home[start] = 0
  const queue = [start]
  for (let n = 0; n < queue.length; n++) {
    const c = queue[n]
    const i = c % NX
    const j = (c - i) / NX
    for (const [di, dj] of STEPS) {
      const ni = i + di
      const nj = j + dj
      const nc = nj * NX + ni
      if (ni < 0 || nj < 0 || ni >= NX || nj >= NY || !open[nc] || home[nc] >= 0) continue
      home[nc] = home[c] + 1
      queue.push(nc)
    }
  }
  walk = Uint8Array.from(home, d => (d >= 0 ? 1 : 0))
}
const cell = (x: number, y: number) => {
  const i = Math.floor((x - BOX.x0) / CELL)
  const j = Math.floor((y - BOX.y0) / CELL)
  return i >= 0 && j >= 0 && i < NX && j < NY ? j * NX + i : -1
}
const centreOf = (c: number): Pt => [
  BOX.x0 + ((c % NX) + 0.5) * CELL,
  BOX.y0 + (Math.floor(c / NX) + 0.5) * CELL,
]
const footing = (x: number, y: number) => {
  const c = cell(x, y)
  return c >= 0 && walk![c] === 1
}
const onGrass = (k: Kind, x: number, y: number) => {
  const [x0, y0, x1, y1] = AREA[k]
  return x >= x0 && x <= x1 && y >= y0 && y <= y1 && footing(x, y)
}
// Whether a beast can walk straight from one point to another without leaving the grass.
function sight(a: Pt, b: Pt) {
  const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]))
  for (let k = 1; k <= n; k++)
    if (!footing(a[0] + ((b[0] - a[0]) * k) / n, a[1] + ((b[1] - a[1]) * k) / n)) return false
  return true
}
// The way from a point to the barn's doors: down the count of cells to them, then pulled tight so
// that it runs straight wherever the grass allows.
function route(x: number, y: number): Pt[] {
  const cells: Pt[] = []
  // Each step to the neighbour nearest the doors. A beast can stand a hair past the edge of the
  // grass where its way clipped the corner of a cell, and then the first step is back onto it.
  for (let c = cell(x, y); c >= 0 && home![c] !== 0; ) {
    const i = c % NX
    const j = (c - i) / NX
    let to = -1
    for (const [di, dj] of STEPS) {
      const nc = (j + dj) * NX + i + di
      if (i + di < 0 || i + di >= NX || j + dj < 0 || j + dj >= NY || home![nc] < 0) continue
      if (to < 0 || home![nc] < home![to]) to = nc
    }
    c = to
    if (to >= 0) cells.push(centreOf(to))
  }
  cells.push(DOOR)
  const out: Pt[] = []
  let from: Pt = [x, y]
  for (let k = 0; k < cells.length; ) {
    let far = k
    for (let m = cells.length - 1; m > k; m--)
      if (sight(from, cells[m])) {
        far = m
        break
      }
    out.push(cells[far])
    from = cells[far]
    k = far + 1
  }
  return out
}

// The terracettes: years of hooves working along the slope have trodden it into steps, a tread for
// every two metres or so of height, each wandering a little up and down and running out at
// the edge of the grazing. Where one wanders into the step below, the two run on as one. The stock
// graze along the treads.
type Terrace = { y: number; x0: number; x1: number; seed: number }
export const TERRACES: Terrace[] = (() => {
  const r = mulberry(0x7e44ac)
  const out: Terrace[] = []
  for (let y = 562 + r() * 6; y < 736; y += 14 + r() * 10) {
    const [x0, , x1] = y < AREA.cow[1] ? AREA.sheep : AREA.cow
    out.push({ y, x0: x0 + r() * 14, x1: x1 - r() * 14, seed: 90 + out.length })
  }
  return out
})()
// Where the nth terrace runs at a point across the slope, or null past its ends or where it has
// joined the one below.
export function onTerrace(n: number, x: number) {
  const at = (t: Terrace) =>
    x < t.x0 || x > t.x1 ? null : t.y + (noise(x * 0.02, 0, t.seed) - 0.5) * 16
  const y = at(TERRACES[n])
  const below = n + 1 < TERRACES.length ? at(TERRACES[n + 1]) : null
  return y === null || (below !== null && below - y < 6) ? null : y
}
// The tread within a couple of pixels of a point, if there is one.
function tread(x: number, y: number) {
  let best: number | null = null
  for (let n = 0; n < TERRACES.length; n++) {
    const ty = onTerrace(n, x)
    if (
      ty !== null &&
      Math.abs(ty - y) < 9 &&
      (best === null || Math.abs(ty - y) < Math.abs(best - y))
    )
      best = ty
  }
  return best
}

// Ground speeds: a walk, a run, and the creep of grazing.
const PACE: Record<Kind, { walk: number; run: number }> = {
  cow: { walk: 0.9 * UNIT, run: 2.5 * UNIT },
  calf: { walk: 1.1 * UNIT, run: 3 * UNIT },
  sheep: { walk: 0.9 * UNIT, run: 3 * UNIT },
}
const CREEP = 0.12 * UNIT
// How close a bike has to pass for a cow to look up or the ewes to bolt, and how far off the hurt
// the cows will come to stand.
const STARTLE = { cow: 32, calf: 32, sheep: 34 }
const STARE = 150
const STAND_OFF = 26
const HEEL = 28

// The share of the herd down chewing the cud at an hour: most of the night, some of the middle of
// the day (more in the heat), and few through the morning and evening grazing.
function lying(env: HerdEnv) {
  const h = env.hour
  if (h < 5 || h >= 21) return 0.75
  if (h >= 10.5 && h < 16) return env.hot ? 0.6 : 0.35
  return 0.12
}

// A beast lies down for five to twenty minutes at a time, and does anything else for about a
// minute before it chooses again. To keep the share of time down at the hour's, the odds of lying
// down at each choice are scaled by those lengths: at the morning's 12% they are under 1%.
const LIE = { min: 300, span: 900 }
const BOUT = 60
function liesDown(share: number) {
  const lie = LIE.min + LIE.span / 2
  return (share * BOUT) / (share * BOUT + (1 - share) * lie)
}

// Each keeps a little room from the others: about a body length apart along the slope, or three
// rows up or down it, so no two are painted over each other. Lengths are in units.
const LEN: Record<Kind, number> = { cow: 24, calf: 16, sheep: 16 }
function roomy(h: Herd, b: Beast | null, kind: Kind, x: number, y: number) {
  for (const o of h.beasts) {
    if (o === b || o.in) continue
    const gap = (LEN[kind] + LEN[o.kind]) / 2 + 4
    // Where it stands, where it is walking, and the end of its way out to the grass.
    const at: Pt[] = [[o.x, o.y]]
    if (o.to) at.push(o.to)
    if (o.path.length) at.push(o.path[o.path.length - 1])
    for (const [ox, oy] of at) if (Math.abs(ox - x) < gap && Math.abs(oy - y) < 12) return false
  }
  return true
}

// Grass with room for a beast within `spread` of a point, on a tread where one runs near and has
// room, sampled wider across the slope than up and down it, as cattle work along the contour; the
// search widens if the near grass is taken. From where a beast stands, only grass it can walk
// straight to.
function near(
  h: Herd,
  b: Beast | null,
  kind: Kind,
  x: number,
  y: number,
  spread: number,
  from: Pt | null = b && [b.x, b.y],
): Pt | null {
  for (let n = 0; n < 32; n++) {
    const s = spread * (1 + n / 8)
    const px = x + (h.rng() - 0.5) * 2 * s
    const raw = y + (h.rng() - 0.5) * s
    for (const py of [tread(px, raw) ?? raw, raw])
      if (onGrass(kind, px, py) && roomy(h, b, kind, px, py) && (!from || sight(from, [px, py])))
        return [px, py]
  }
  return null
}

const kinScratch: Beast[] = []
function kin(h: Herd, b: Beast): Beast[] {
  kinScratch.length = 0
  const isSheep = b.kind === 'sheep'
  for (const o of h.beasts) {
    if (!o.in && (o.kind === 'sheep') === isSheep) kinScratch.push(o)
  }
  return kinScratch
}
function centre(group: { x: number; y: number }[]): Pt {
  let x = 0
  let y = 0
  for (const o of group) {
    x += o.x
    y += o.y
  }
  return [x / group.length, y / group.length]
}

function go(b: Beast, to: Pt | null, run = false) {
  if (!to) return graze(b, 10)
  b.doing = run ? 'run' : 'walk'
  b.to = to
  b.path = []
  b.t = 0
  b.until = 40
}
function graze(b: Beast, until: number) {
  b.doing = 'graze'
  b.to = null
  b.path = []
  b.t = 0
  b.until = until
}
// Along a way in or out, which nothing turns it from.
function follow(b: Beast, path: Pt[]) {
  b.spot = -1
  b.doing = 'walk'
  b.path = path
  b.to = path[0]
  b.t = 0
  b.until = 120
  b.at = null
  b.watch = null
}
const homeward = (b: Beast) => b.path[b.path.length - 1] === DOOR
// The cattle go in and come out in their order: the lead cow with her calf at heel, the other two
// cows, the red cow's calf after her, the black cow, then the ewes, a few seconds apart.
const ORDER = [0, 4, 1, 2, 5, 3, 6, 7, 8]
const GAP = 6
const turn = (h: Herd, n: number) => h.since >= ORDER.indexOf(n) * GAP
// Out of the doors in the morning to where its own are headed: a calf to her dam's heel, the first
// of the cattle or the ewes to the middle of the pasture or the upper end.
const HERD: Pt = [318, 668]
const FLOCK: Pt = [336, 592]
function leave(h: Herd, b: Beast) {
  b.in = false
  b.x = DOOR[0]
  b.y = DOOR[1]
  const bound = (o: Beast): Pt => (o.path.length ? o.path[o.path.length - 1] : [o.x, o.y])
  const dam = b.kind === 'calf' && !h.beasts[b.dam].in ? h.beasts[b.dam] : null
  const own = kin(h, b).filter(o => o !== b && o.kind !== 'calf')
  const [x, y]: Pt = dam
    ? [bound(dam)[0] - HEEL * dam.face, bound(dam)[1]]
    : own.length
      ? centre(own.map(o => ({ x: bound(o)[0], y: bound(o)[1] })))
      : b.kind === 'sheep'
        ? FLOCK
        : HERD
  // Nobody stops in the mouth of the lane, where the rest are still coming out.
  const home = b.kind === 'sheep' ? FLOCK : HERD
  const clear = (p: Pt | null) => (p && slant(p[0] - LANE.x0, p[1] - DOOR[1]) > 40 ? p : null)
  const to =
    clear(near(h, b, b.kind, x, y, dam ? 6 : 30, null)) ??
    clear(near(h, b, b.kind, home[0], home[1], 30, null)) ??
    home
  const back = route(to[0], to[1])
  back.pop()
  follow(b, [...back.reverse(), to])
}

// What it does next: down to chew the cud, more readily when the others are down; back to its dam,
// the flock or the herd if it has strayed; otherwise on grazing where it is, off a little way to
// fresh grass, or a look about.
function next(h: Herd, b: Beast, env: HerdEnv) {
  const r = h.rng
  const group = kin(h, b)
  let downCount = 0
  for (const o of group) if (o.doing === 'lie') downCount++
  const down = downCount / group.length
  b.at = null
  b.watch = null
  b.path = []
  // Cattle lie down together, so the share pulls a third of the way toward how many are down now.
  const share = Math.min(0.9, lying(env) + (b.kind === 'calf' ? 0.2 : 0))
  if (r() < liesDown(share + (down - share) / 3)) {
    b.doing = 'lie'
    b.t = 0
    b.until = LIE.min + r() * LIE.span
    return
  }
  const dam = b.kind === 'calf' && !h.beasts[b.dam].in ? h.beasts[b.dam] : null
  // A calf follows at her heels, a body length behind her.
  const [cx, cy] = dam ? [dam.x - HEEL * dam.face, dam.y] : centre(group)
  const leash = dam ? 14 : b.kind === 'sheep' ? 16 : 44
  if (slant(cx - b.x, cy - b.y) > leash) return go(b, near(h, b, b.kind, cx, cy, dam ? 4 : 12))
  const roll = r()
  if (roll < 0.2)
    return go(b, near(h, b, b.kind, b.x + (r() - 0.5) * 80, b.y + (r() - 0.5) * 20, 8))
  if (roll < 0.28) {
    b.doing = 'look'
    b.t = 0
    b.until = 3 + r() * 6
    b.at = [b.x + b.face * 40, b.y - 8]
    return
  }
  if (r() < 0.25) b.face = -b.face
  graze(b, 20 + r() * 70)
}

// Out onto the pasture as found when anyone looks: the cattle about one spot, calves by their dams,
// the ewes together above, each at its own point of whatever it is doing; or all in the barn.
function scatter(h: Herd, env: HerdEnv) {
  const r = h.rng
  h.beasts = []
  h.out = env.out
  h.since = Infinity
  h.doors = env.out ? 2 : 0
  h.farmer = { chore: 'home', after: 'home', way: [], x: HOUSE[0], y: HOUSE[1], face: -1, t: 0 }
  h.trough = 0
  h.feeds = 0
  const herd = near(h, null, 'cow', 318, 668, 40) ?? [318, 668]
  const flock = near(h, null, 'sheep', 336, 592, 30) ?? [336, 592]
  for (const s of STOCK) {
    const dam = s.dam >= 0 ? h.beasts[s.dam] : null
    const [x, y] = dam
      ? (near(h, null, 'calf', dam.x - HEEL * dam.face, dam.y, 4) ??
        near(h, null, 'calf', dam.x, dam.y, 30) ??
        near(h, null, 'calf', herd[0], herd[1], 40) ?? [dam.x, dam.y])
      : s.kind === 'sheep'
        ? (near(h, null, 'sheep', flock[0], flock[1], 20) ?? flock)
        : (near(h, null, 'cow', herd[0], herd[1], 40) ?? herd)
    const b: Beast = {
      ...s,
      x,
      y,
      face: r() < 0.5 ? -1 : 1,
      doing: 'graze',
      t: 0,
      until: 0,
      to: null,
      at: null,
      watch: null,
      calm: 0,
      in: !env.out,
      path: [],
      spot: -1,
      fed: 0,
    }
    h.beasts.push(b)
    if (r() < Math.min(0.9, lying(env) + (s.kind === 'calf' ? 0.2 : 0))) {
      b.doing = 'lie'
      b.until = LIE.min + r() * LIE.span
    } else graze(b, 20 + r() * 70)
    b.t = r() * b.until
  }
}

export function makeHerd(seed: number, env: HerdEnv): Herd {
  if (!walk) survey()
  const h = { seed, rng: mulberry(seed ^ 0x5eed), beasts: [] } as unknown as Herd
  scatter(h, env)
  return h
}

// Without motion: the herd as it stands at this hour, the same on every repaint.
export function poseHerd(h: Herd, env: HerdEnv) {
  h.rng = mulberry(h.seed ^ 0x5eed)
  scatter(h, env)
}

// On a step toward a point at a ground speed, true on arriving. Every point it walks to is in
// sight over the grass, so the way there stays on it.
function toward(b: { x: number; y: number; face: number }, to: Pt, v: number, dt: number) {
  const dx = to[0] - b.x
  const dy = to[1] - b.y
  const s = Math.min(1, (v * dt) / Math.max(1e-6, slant(dx, dy)))
  b.x += dx * s
  b.y += dy * s
  if (Math.abs(dx) > 0.5) b.face = Math.sign(dx)
  return s >= 1
}

function send(f: Farmer, way: Pt[], after: Chore) {
  f.chore = 'walk'
  f.way = way
  f.after = after
  f.t = 0
}

// Up from the house to open the doors in the morning, or to stand by them while the stock come in
// at night and shut them once the last is in; back down to the house when done.
function stepFarmer(h: Herd, dt: number) {
  const f = h.farmer
  f.t += dt
  switch (f.chore) {
    case 'walk':
      if (!toward(f, f.way[0], STRIDE, dt)) break
      f.way.shift()
      if (!f.way.length) {
        f.chore = f.after
        f.t = 0
      }
      break
    case 'doors': {
      if (f.t < SWING) break
      f.t = 0
      const want = h.out ? 2 : 0
      if (h.doors !== want) h.doors += Math.sign(want - h.doors)
      else if (h.out) send(f, [ASIDE], 'watch')
      else send(f, [ASIDE, HOUSE], 'home')
      break
    }
    case 'make':
      if (f.t > MAKE) send(f, [ASIDE, FRONT, MOUTH, BEHIND], 'pour')
      break
    case 'pour':
      f.face = 1
      if (f.t < 2.5) break
      h.trough = 1
      h.feeds++
      send(f, [MOUTH, FRONT, ASIDE, HOUSE], 'home')
      break
    case 'watch': {
      // Facing the lane until the last is out and clear of its mouth, or in, and a moment after.
      f.face = -1
      const done = h.out
        ? h.beasts.every(b => !b.in && slant(b.x - DOOR[0], b.y - DOOR[1]) > 40)
        : h.beasts.every(b => b.in)
      if (!done) f.t = 0
      else if (f.t > 4) {
        if (h.out) send(f, [HOUSE], 'home')
        else send(f, [FRONT], 'doors')
      }
      break
    }
  }
}

// Home from the river with the stock out: the farmer goes in and makes up their feed.
export function feedHerd(h: Herd) {
  const f = h.farmer
  if (!h.out || f.chore !== 'home') return
  f.chore = 'make'
  f.t = 0
}

// At the trough: whichever of the cattle that have not eaten from this filling is nearest a free
// place walks over to it, until the trough is empty. A beast that has turned to something else on
// the way gives its place up.
function serve(h: Herd) {
  if (h.trough <= 0) return
  for (const b of h.beasts) if (b.spot >= 0 && b.doing !== 'eat' && b.doing !== 'walk') b.spot = -1
  for (const [k, at] of PLACES.entries()) {
    if (h.beasts.some(b => b.spot === k)) continue
    let best: Beast | null = null
    for (const b of h.beasts) {
      if (b.in || b.kind === 'sheep' || b.fed === h.feeds || b.spot >= 0) continue
      if (b.path.length || b.watch || b.doing === 'run') continue
      if (!best || slant(b.x - at[0], b.y - at[1]) < slant(best.x - at[0], best.y - at[1])) best = b
    }
    if (!best) return
    best.spot = k
    toPlace(h, best)
  }
}
function toPlace(h: Herd, b: Beast) {
  const at = PLACES[b.spot]
  const to = sight([b.x, b.y], at) ? at : near(h, b, b.kind, at[0], at[1], 12)
  // Nowhere to get at it from: it misses this filling.
  if (!to) {
    b.fed = h.feeds
    b.spot = -1
    return
  }
  go(b, to)
}

const hurtScratch: Passer[] = []
export function stepHerd(h: Herd, env: HerdEnv, dt: number, people: Passer[]) {
  if (env.out !== h.out) {
    h.out = env.out
    h.since = 0
    const f = h.farmer
    if (f.chore === 'home') {
      f.x = HOUSE[0]
      f.y = HOUSE[1]
    }
    if (h.out) send(f, [ASIDE, FRONT], 'doors')
    else send(f, [ASIDE], 'watch')
  }
  stepFarmer(h, dt)
  // The stock go through the doors, and their turns count, only while the doors stand open and the
  // farmer is clear of them.
  const open = h.doors === 2 && slant(h.farmer.x - FRONT[0], h.farmer.y - FRONT[1]) > 20
  if (open) h.since += dt
  serve(h)
  hurtScratch.length = 0
  for (const o of people) if (o.hurt) hurtScratch.push(o)
  const hurt = hurtScratch
  for (const [n, b] of h.beasts.entries()) {
    // Called in, each in its turn sets off for the barn from whatever it was doing; called out,
    // each in its turn comes out of the doors once the one before is clear of them.
    if (b.in) {
      if (
        h.out &&
        open &&
        turn(h, n) &&
        h.beasts.every(o => o.in || slant(o.x - DOOR[0], o.y - DOOR[1]) > 30)
      )
        leave(h, b)
      continue
    }
    if (!h.out && turn(h, n) && !homeward(b)) follow(b, route(b.x, b.y))
    b.t += dt
    b.calm = Math.max(0, b.calm - dt)
    const free = !b.path.length
    // A cow comes over to stand and stare at whoever lies hurt within sight, and goes back to the
    // grass once they are gone.
    if (free && b.kind === 'cow') {
      const sore = hurt.find(o => slant(o.x - b.x, o.y - b.y) < STARE)
      if (sore && !b.watch && b.doing !== 'lie') {
        const d = slant(b.x - sore.x, b.y - sore.y) || 1
        const u = STAND_OFF / d
        go(b, near(h, b, 'cow', sore.x + (b.x - sore.x) * u, sore.y + (b.y - sore.y) * u, 20))
        b.watch = [sore.x, sore.y]
      } else if (b.watch && !hurt.some(o => slant(o.x - b.watch![0], o.y - b.watch![1]) < 20))
        next(h, b, env)
    }
    // A bike going by close: a cow looks up, a calf as well; the ewes bolt together, away from it.
    const by =
      !free || b.calm > 0 || b.watch
        ? null
        : people.find(o => o.fast && slant(o.x - b.x, o.y - b.y) < STARTLE[b.kind])
    if (by) {
      b.calm = 8
      if (b.kind === 'sheep' && h.rng() < 0.7) {
        const d = slant(b.x - by.x, b.y - by.y) || 1
        const ux = (b.x - by.x) / d
        const uy = (b.y - by.y) / d
        for (const o of kin(h, b))
          if (!o.path.length && slant(o.x - b.x, o.y - b.y) < 24) {
            o.calm = 8
            o.watch = null
            go(o, near(h, o, 'sheep', o.x + ux * 30, o.y + uy * 15, 5), true)
          }
      } else if (b.kind !== 'sheep' && b.doing === 'graze' && h.rng() < 0.4) {
        b.doing = 'look'
        b.t = 0
        b.until = 3 + h.rng() * 4
        b.at = [by.x, by.y]
      }
    }
    switch (b.doing) {
      case 'graze': {
        // A mouthful, a step along the tread, another; turning back at the edge of the grass, and
        // grazing on the spot when another is in the way.
        const nx = b.x + b.face * CREEP * dt
        const ty = tread(nx, b.y)
        const ny = ty !== null && Math.abs(ty - b.y) < 3 ? ty : b.y
        if (!onGrass(b.kind, nx, ny)) b.face = -b.face
        else if (roomy(h, b, b.kind, nx, ny)) {
          b.x = nx
          b.y = ny
        }
        if (b.t > b.until) next(h, b, env)
        break
      }
      case 'walk':
      case 'run': {
        const v = b.doing === 'run' ? PACE[b.kind].run : PACE[b.kind].walk
        // In file along the lane, in or out: it waits while the one ahead of it, nearer the doors
        // on the way in or further from them on the way out, is within a length.
        const way = homeward(b) ? -1 : 1
        const ahead =
          b.path.length > 0 &&
          h.beasts.some(
            o =>
              o !== b &&
              !o.in &&
              o.path.length > 0 &&
              homeward(o) === homeward(b) &&
              way * (home![cell(o.x, o.y)] - home![cell(b.x, b.y)]) > 0 &&
              slant(o.x - b.x, o.y - b.y) < (LEN[o.kind] + LEN[b.kind]) / 2 + 2,
          )
        if (ahead) {
          if (b.t > b.until) next(h, b, env)
        } else if (toward(b, b.to!, v, dt)) {
          if (b.path.length > 1) {
            b.path.shift()
            b.to = b.path[0]
          } else if (homeward(b)) {
            b.in = true
            b.path = []
            b.to = null
          } else if (b.watch) {
            b.doing = 'look'
            b.at = b.watch
            b.t = 0
            b.until = Infinity
          } else if (b.doing === 'run') {
            b.doing = 'look'
            b.t = 0
            b.until = 3 + h.rng() * 4
            b.at = null
          } else if (b.spot >= 0) {
            const at = PLACES[b.spot]
            if (slant(b.x - at[0], b.y - at[1]) > 1) toPlace(h, b)
            else {
              // Head down in the trough, facing along it.
              b.doing = 'eat'
              b.to = null
              b.t = 0
              b.until = 22 + h.rng() * 14
              b.face = Math.sign(TROUGH[0] - b.x)
            }
          } else graze(b, 20 + h.rng() * 70)
        } else if (b.t > b.until) next(h, b, env)
        break
      }
      case 'look':
        if (b.at) b.face = Math.sign(b.at[0] - b.x) || b.face
        if (b.t > b.until) next(h, b, env)
        break
      case 'lie':
        if (b.t > b.until) next(h, b, env)
        break
      case 'eat':
        h.trough = Math.max(0, h.trough - dt / HOLDS)
        if (b.t < b.until && h.trough > 0) break
        b.fed = h.feeds
        b.spot = -1
        next(h, b, env)
        break
    }
  }
}

export type HerdInk = {
  // Coats (white, black, red), the muzzle, legs and hooves, and a ewe's fleece and face.
  coats: [RGB, RGB, RGB]
  muzzle: RGB
  leg: RGB
  fleece: RGB
  face: RGB
  // The barn's boards, its tin roof, its painted doors, and the dark inside it.
  boards: RGB
  roof: RGB
  trim: RGB
  dark: RGB
  // The farmer's straw hat, face and hands, and the bib and legs of the overalls.
  hat: RGB
  skin: RGB
  bib: RGB
  trousers: RGB
  // The pail and the feed in it.
  pail: RGB
  feed: RGB
}

// The barn, end on: boards under a tin roof pitched at forty-five degrees, a hay door up in the
// gable, and the big doors: shut, two painted leaves under a header; open, the dark inside with the
// leaves swung back flat against the wall either side, the left one first.
export function paintBarn(d: Uint8ClampedArray, g: Grid, ink: HerdInk, h: Herd) {
  const i = col(g, BARN[0])
  const j = row(g, BARN[1])
  for (let dj = 1; dj <= 9; dj++) {
    const eave = dj < 4 ? -1 : dj === 4 ? 5 : 9 - dj
    const wall = dj <= 4 ? 4 : eave - 1
    for (let di = -wall; di <= wall; di++) dot(d, g, i + di, j - dj, ink.boards)
    if (eave >= 0) {
      dot(d, g, i - eave, j - dj, ink.roof)
      dot(d, g, i + eave, j - dj, ink.roof)
    }
  }
  for (let dj = 1; dj <= 3; dj++) {
    for (let di = -1; di <= 1; di++) {
      const ajar = h.doors === 2 || (h.doors === 1 && di < 1)
      dot(d, g, i + di, j - dj, ajar ? ink.dark : di === 0 && dj < 3 ? ink.boards : ink.trim)
    }
    if (h.doors > 0) dot(d, g, i - 2, j - dj, ink.trim)
    if (h.doors > 1) dot(d, g, i + 2, j - dj, ink.trim)
  }
  dot(d, g, i, j - 6, ink.dark)
}

// The trough, galvanised like the roof, with the feed heaped in it while there is any.
export function paintTrough(d: Uint8ClampedArray, g: Grid, ink: HerdInk, h: Herd) {
  const i = col(g, TROUGH[0])
  const j = row(g, TROUGH[1])
  const heap = Math.ceil(h.trough * 3)
  for (let di = -2; di <= 2; di++) {
    const inside = Math.abs(di) < 2
    const fed = inside && Math.abs(di) * 2 < heap
    dot(d, g, i + di, j - 2, inside ? (fed ? ink.feed : ink.dark) : ink.roof)
    dot(d, g, i + di, j - 1, ink.roof)
  }
}

// The farmer in a straw hat and denim overalls, legs apart in a stride; at the doors, an arm out to
// the leaf in hand; carrying the pail of feed out to the trough, and tipping it in.
export function paintFarmer(d: Uint8ClampedArray, g: Grid, ink: HerdInk, h: Herd, time: number) {
  const f = h.farmer
  if (f.chore === 'home' || f.chore === 'make') return
  const i = col(g, f.x)
  const j = row(g, f.y)
  if (f.chore === 'walk' && Math.floor(time * 2.5) % 2) {
    dot(d, g, i - 1, j - 1, ink.trousers)
    dot(d, g, i + 1, j - 1, ink.trousers)
  } else dot(d, g, i, j - 1, ink.trousers)
  dot(d, g, i, j - 2, ink.bib)
  dot(d, g, i, j - 3, ink.skin)
  for (let di = -1; di <= 1; di++) dot(d, g, i + di, j - 4, ink.hat)
  if (f.chore === 'doors') {
    // Opening, the left leaf and then the right; shutting, the right and then the left.
    const left = h.out ? h.doors === 0 : h.doors === 1
    dot(d, g, i + (left ? -1 : 1), j - 2, ink.skin)
  } else if (f.chore === 'walk' && f.after === 'pour') dot(d, g, i + f.face, j - 1, ink.pail)
  else if (f.chore === 'pour') {
    dot(d, g, i + 1, j - 3, ink.pail)
    if (Math.floor(time * 6) % 2) dot(d, g, i + 1, j - 2, ink.feed)
  }
}

// A cow is six pixels long with her head, standing three and a half: two legs, two rows of body
// with her markings, the head carried level and the muzzle below it, down in the grass to graze or
// lifted to look, and the tail swinging at the flies; lying, the body sits on the grass. A calf is
// a row of body on two legs; a ewe a woolly row and a half on black legs with her black face.
export function paintBeast(
  d: Uint8ClampedArray,
  g: Grid,
  ink: HerdInk,
  b: Beast,
  time: number,
  n: number,
) {
  if (b.in) return
  const i = col(g, b.x)
  const j = row(g, b.y)
  const f = b.face
  const coat = ink.coats[b.coat]
  const head = ink.coats[b.head]
  const low = b.doing === 'lie' ? 1 : 0
  const stride =
    (b.doing === 'walk' || b.doing === 'run') &&
    Math.floor(time * (b.doing === 'run' ? 6 : 2.5) + n) % 2
  const up = b.doing === 'look' || b.doing === 'walk' || b.doing === 'run'
  if (b.kind === 'cow') {
    if (!low) for (const di of stride ? [-1, 0] : [-2, 1]) dot(d, g, i + f * di, j - 1, ink.leg)
    for (let k = 0; k < 8; k++) {
      const di = (k % 4) - 2
      const dj = k < 4 ? -3 : -2
      dot(d, g, i + f * di, j + dj + low, b.spots & (1 << k) ? ink.coats[1] : coat)
    }
    if (low) {
      dot(d, g, i + 2 * f, j - 2, head)
      dot(d, g, i + 3 * f, j - 2, ink.muzzle)
    } else if (b.doing === 'graze' || b.doing === 'eat') {
      dot(d, g, i + 2 * f, j - 2, head)
      dot(d, g, i + 3 * f, j - 1, ink.muzzle)
    } else if (b.doing === 'look') {
      dot(d, g, i + 2 * f, j - 3, coat)
      dot(d, g, i + 2 * f, j - 4, head)
      dot(d, g, i + 3 * f, j - 4, ink.muzzle)
    } else {
      dot(d, g, i + 2 * f, j - 3, head)
      dot(d, g, i + 3 * f, j - 2, ink.muzzle)
    }
    const swish = Math.sin(time * 1.7 + n * 2.3) > 0.85 ? 1 : 0
    dot(d, g, i - 3 * f, j - 2 + low - swish, b.coat === 0 ? ink.coats[1] : coat)
  } else if (b.kind === 'calf') {
    if (!low) for (const di of stride ? [0] : [-1, 1]) dot(d, g, i + f * di, j - 1, ink.leg)
    for (let di = -1; di <= 1; di++)
      dot(d, g, i + f * di, j - 2 + low, b.spots & (1 << (di + 1)) ? ink.coats[1] : coat)
    if (up && !low) {
      dot(d, g, i + 2 * f, j - 3, head)
      dot(d, g, i + 3 * f, j - 3, ink.muzzle)
    } else {
      dot(d, g, i + 2 * f, j - 2, head)
      if (!low) dot(d, g, i + 3 * f, j - 1, ink.muzzle)
    }
  } else {
    if (!low) for (const di of stride ? [0] : [-1, 1]) dot(d, g, i + f * di, j - 1, ink.face)
    for (let di = -1; di <= 1; di++) dot(d, g, i + f * di, j - 2 + low, ink.fleece)
    dot(d, g, i, j - 3 + low, ink.fleece)
    dot(d, g, i - f, j - 3 + low, ink.fleece)
    dot(d, g, i + 2 * f, j - (up && !low ? 3 : 2), ink.face)
  }
}
