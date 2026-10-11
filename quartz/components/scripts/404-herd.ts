// The stock on the bike park's hill (404-riders.ts), out on the grass inside the line's middle
// turns from the thaw until the first snow: four cows, two of them with a calf at foot, and three
// black-faced ewes that keep to the upper end. It is the same herd every day. The cattle graze along
// the slope a mouthful at a time, walk off to fresh grass now and then, cross the line when it
// suits them (after a look up it), and lie down to chew the cud through the night and the heat of
// the day, mostly together. A bike going by close makes a cow look up and the ewes bolt; someone
// lying hurt on the line brings the cows over to stand and stare until they are gone.

import { type Almanac, type Sky, winter } from './404-almanac'
import { PHOTO, TRACKS, UNIT, nearCourse, slant } from './404-downhill'
import { type Grid, type Pt, type RGB, col, dot, mulberry, noise, row } from './404-pixel'

export type HerdEnv = {
  // Out on the hill at all, the solar hour, and whether it is hot enough to lie up through the day.
  out: boolean
  hour: number
  hot: boolean
}

export function herdEnv(al: Almanac, sky: Sky): HerdEnv {
  return { out: !winter(al), hour: al.hour, hot: sky.temp !== null && sky.temp > 26 }
}

type Kind = 'cow' | 'calf' | 'sheep'
type Doing = 'graze' | 'walk' | 'look' | 'lie' | 'run'

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
  // Where along the main line it stands, while it stands on a line.
  line: number | null
}

export type Herd = { seed: number; rng: () => number; beasts: Beast[] }

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

// The pasture on a grid of four units, about a pixel: grass clear of the line, the paths and the
// photographer, and for each cell on a line where it lies along the main line. The cattle range over
// the middle turns and the strip under the table; the ewes keep above them.
const CELL = 4
// A beast stands this far from the middle of a line or path, so its body stays off the tread.
const CLEAR = 16
const BOX = { x0: 244, y0: 548, x1: 392, y1: 744 }
const NX = Math.ceil((BOX.x1 - BOX.x0) / CELL)
const NY = Math.ceil((BOX.y1 - BOX.y0) / CELL)
const AREA: Record<Kind, [number, number, number, number]> = {
  cow: [256, 598, 376, 736],
  calf: [256, 598, 376, 736],
  sheep: [282, 556, 376, 628],
}
let grass: Uint8Array | null = null
let lineQ: Float32Array | null = null
function survey() {
  grass = new Uint8Array(NX * NY)
  lineQ = new Float32Array(NX * NY).fill(NaN)
  for (let j = 0; j < NY; j++)
    for (let i = 0; i < NX; i++) {
      const x = BOX.x0 + (i + 0.5) * CELL
      const y = BOX.y0 + (j + 0.5) * CELL
      const c = j * NX + i
      grass[c] = !nearCourse(x, y, CLEAR) && Math.hypot(x - PHOTO[0], y - PHOTO[1]) > 12 ? 1 : 0
      let best = 7
      for (const t of TRACKS)
        for (let k = 0; k < t.n; k++) {
          const dd = Math.hypot(t.x[k] - x, t.y[k] - y)
          if (dd < best) {
            best = dd
            lineQ[c] = t.main[k]
          }
        }
    }
}
const cell = (x: number, y: number) => {
  const i = Math.floor((x - BOX.x0) / CELL)
  const j = Math.floor((y - BOX.y0) / CELL)
  return i >= 0 && j >= 0 && i < NX && j < NY ? j * NX + i : -1
}
const onGrass = (k: Kind, x: number, y: number) => {
  const [x0, y0, x1, y1] = AREA[k]
  const c = cell(x, y)
  return x >= x0 && x <= x1 && y >= y0 && y <= y1 && c >= 0 && grass![c] === 1
}
const lineAt = (x: number, y: number) => {
  const c = cell(x, y)
  return c < 0 || Number.isNaN(lineQ![c]) ? null : lineQ![c]
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
const STARTLE = { cow: 32, calf: 32, sheep: 30 }
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
    if (o === b) continue
    const gap = (LEN[kind] + LEN[o.kind]) / 2 + 4
    for (const [ox, oy] of o.to ? [[o.x, o.y], o.to] : [[o.x, o.y]])
      if (Math.abs(ox - x) < gap && Math.abs(oy - y) < 12) return false
  }
  return true
}

// Grass with room for a beast within `spread` of a point, on a tread where one runs near, sampled
// wider across the slope than up and down it, as cattle work along the contour; the search widens
// if the near grass is taken.
function near(
  h: Herd,
  b: Beast | null,
  kind: Kind,
  x: number,
  y: number,
  spread: number,
): Pt | null {
  for (let n = 0; n < 32; n++) {
    const s = spread * (1 + n / 8)
    const px = x + (h.rng() - 0.5) * 2 * s
    const raw = y + (h.rng() - 0.5) * s
    const py = tread(px, raw) ?? raw
    if (onGrass(kind, px, py) && roomy(h, b, kind, px, py)) return [px, py]
  }
  return null
}

const kin = (h: Herd, b: Beast) =>
  h.beasts.filter(o => (o.kind === 'sheep') === (b.kind === 'sheep'))
function centre(group: Beast[]): Pt {
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
  b.t = 0
  b.until = 40
}
function graze(b: Beast, until: number) {
  b.doing = 'graze'
  b.to = null
  b.t = 0
  b.until = until
}

// What it does next: down to chew the cud, more readily when the others are down; back to its dam,
// the flock or the herd if it has strayed; otherwise on grazing where it is, off a little way to
// fresh grass, or a look about.
function next(h: Herd, b: Beast, env: HerdEnv) {
  const r = h.rng
  const group = kin(h, b)
  const down = group.filter(o => o.doing === 'lie').length / group.length
  b.at = null
  b.watch = null
  // Cattle lie down together, so the share pulls a third of the way toward how many are down now.
  const share = Math.min(0.9, lying(env) + (b.kind === 'calf' ? 0.2 : 0))
  if (r() < liesDown(share + (down - share) / 3)) {
    b.doing = 'lie'
    b.t = 0
    b.until = LIE.min + r() * LIE.span
    return
  }
  const dam = b.kind === 'calf' ? h.beasts[b.dam] : null
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
// the ewes together above, each at its own point of whatever it is doing.
function scatter(h: Herd, env: HerdEnv) {
  const r = h.rng
  h.beasts = []
  const herd = near(h, null, 'cow', 318, 668, 40) ?? [318, 668]
  const flock = near(h, null, 'sheep', 336, 592, 30) ?? [336, 592]
  for (const s of STOCK) {
    const dam = s.dam >= 0 ? h.beasts[s.dam] : null
    const [x, y] = dam
      ? (near(h, null, 'calf', dam.x - HEEL * dam.face, dam.y, 4) ??
        near(h, null, 'calf', dam.x, dam.y, 30) ?? [dam.x, dam.y])
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
      line: null,
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
  if (!grass) survey()
  const h: Herd = { seed, rng: mulberry(seed ^ 0x5eed), beasts: [] }
  if (env.out) scatter(h, env)
  return h
}

// Without motion: the herd as it stands at this hour, the same on every repaint.
export function poseHerd(h: Herd, env: HerdEnv) {
  h.rng = mulberry(h.seed ^ 0x5eed)
  h.beasts = []
  if (env.out) scatter(h, env)
}

// On a step toward a point at a ground speed, true on arriving. A step onto a line waits while a
// bike is coming down it.
function toward(b: Beast, to: Pt, v: number, dt: number, clear: (q: number) => boolean) {
  const dx = to[0] - b.x
  const dy = to[1] - b.y
  const len = slant(dx, dy)
  const s = Math.min(1, (v * dt) / Math.max(1e-6, len))
  const nx = b.x + dx * s
  const ny = b.y + dy * s
  if (Math.abs(dx) > 0.5) b.face = Math.sign(dx)
  const q = lineAt(nx, ny)
  if (q !== null && b.line === null && b.doing !== 'run' && !clear(q)) return false
  b.x = nx
  b.y = ny
  b.line = q
  return s >= 1
}

export function stepHerd(
  h: Herd,
  env: HerdEnv,
  dt: number,
  people: Passer[],
  clear: (q: number) => boolean,
) {
  if (!env.out) {
    h.beasts = []
    return
  }
  if (!h.beasts.length) scatter(h, env)
  const hurt = people.filter(o => o.hurt)
  for (const b of h.beasts) {
    b.t += dt
    b.calm = Math.max(0, b.calm - dt)
    // A cow comes over to stand and stare at whoever lies hurt within sight, and goes back to the
    // grass once they are gone.
    if (b.kind === 'cow') {
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
      b.calm > 0 || b.watch
        ? null
        : people.find(o => o.fast && slant(o.x - b.x, o.y - b.y) < STARTLE[b.kind])
    if (by) {
      b.calm = 8
      if (b.kind === 'sheep' && h.rng() < 0.7) {
        const d = slant(b.x - by.x, b.y - by.y) || 1
        const ux = (b.x - by.x) / d
        const uy = (b.y - by.y) / d
        for (const o of kin(h, b))
          if (slant(o.x - b.x, o.y - b.y) < 24) {
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
        if (toward(b, b.to!, v, dt, clear)) {
          if (b.watch) {
            b.doing = 'look'
            b.at = b.watch
            b.t = 0
            b.until = Infinity
          } else if (b.doing === 'run') {
            b.doing = 'look'
            b.t = 0
            b.until = 3 + h.rng() * 4
            b.at = null
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
    } else if (b.doing === 'graze') {
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
