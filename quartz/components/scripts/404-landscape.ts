// Pixel landscape for the 404 ruin, painted the way Van Gogh painted the factories at Clichy:
// low-resolution canvases upscaled with `image-rendering: pixelated`, an ordered-dither
// underpainting, then short brush strokes laid along flow fields. Each stroke picks its ramp stop
// once, so the dither happens per stroke instead of per pixel. The ruin itself is printed into the
// same grid from the hidden SVG geometry: plates, one-pixel keylines, dial marks and chips, with
// the moss grown over it as blobs (404-overgrowth). The cat, the survey party and a passing
// triathlete (404-cat, 404-crew, 404-triathlete) stand in the same canvas as the water, which
// reflects them along with the ruin, the hill and the ranges. The ranges and the ruin paint once per size or theme; sky, field, water,
// camp, canopy and portal repaint in stop-motion ticks.

import { type Almanac, type Body, type Site, type Sky, almanac, readSky } from './404-almanac'
import {
  type Bough,
  type BoughInk,
  type Shed,
  findBough,
  paintBough,
  paintShed,
  stepShed,
  workFor,
} from './404-boughs'
import { type Cat, type CatInk, catHead, makeCat, paintCat, poseCat, stepCat } from './404-cat'
import {
  type CrewInk,
  type Voices,
  ageBubbles,
  ask,
  heard,
  makeCrew,
  paintBubbles,
  paintCrew,
  poseCrew,
  stepCrew,
} from './404-crew'
import { type CourseInk, nearCourse, paintCourse } from './404-downhill'
import { GATE, MISSING, TILT, hourAngle, stoneOuter } from './404-gate'
import {
  GROWN,
  type Greens,
  makeCurtain,
  makeOvergrowth,
  paintCurtain,
  paintIvy,
  paintOvergrowth,
  stepCurtain,
} from './404-overgrowth'
import {
  BAYER,
  type Grid,
  type Pt,
  type RGB,
  clamp01,
  clampCol,
  col,
  dither,
  dot,
  fbm,
  frac,
  grid,
  hash,
  hex,
  line,
  mix,
  mul,
  mulberry,
  noise,
  pick,
  put,
  ramp,
  row,
  smooth,
  step,
  type Tones,
} from './404-pixel'
import {
  type Park,
  type RiderInk,
  daySeed,
  makePark,
  paintPark,
  parkEnv,
  posePark,
  stepPark,
} from './404-riders'
import {
  type Hill,
  type SkiInk,
  hillEnv,
  makeHill,
  paintHill,
  poseHill,
  stepHill,
  warmHill,
} from './404-skiers'
import {
  type TriInk,
  headPixel,
  layCourse,
  makeTriathlete,
  paintAscent,
  paintTriathlete,
  sighting,
  stepTriathlete,
} from './404-triathlete'

// Leaf tip: position plus depth toward (+1) or away from (-1) the viewer.
type Tip = [number, number, number]
type Pointer = { x: number; y: number }
type Seg = { x0: number; y0: number; x1: number; y1: number; w0: number; w1: number }
type Tree = { segs: Seg[]; tips: Tip[] }
type Mode = 'idle' | 'hot' | 'enter'
type Cypress = { x: number; base: number; h: number; w: number }
type Ink = 'C' | 'S' | 'B' | 'D' | 'K'
type Palette = {
  sky: RGB[]
  cloud: RGB[]
  swirl: [RGB, RGB]
  sun: RGB[]
  far: RGB[]
  mid: RGB[]
  hill: RGB[]
  pine: RGB[]
  bark: RGB[]
  leaf: RGB[]
  // The broadleaf crowns, which turn with the season, and what the season turns things toward.
  crown: RGB[]
  autumn: RGB[]
  dun: RGB[]
  snow: RGB[]
  rock: RGB[]
  gold: RGB
  poppy: RGB
  cornflower: RGB
  ochre: RGB
  star: RGB
  glint: RGB[]
  figure: Record<Ink, RGB>
  shade: [...RGB, number]
  fringe: [...RGB, number]
  water: RGB[]
  smoke: RGB[]
  glow: RGB[]
  coats: [RGB, RGB]
  tent: [RGB, RGB]
  moss: RGB[]
  // Crest lines: far ridge, mid ridge, hill.
  contour: [RGB, RGB, RGB]
  crew: RGB[]
  rope: RGB
  fish: RGB
}
// Hand angles in degrees, clockwise from twelve.
type Dial = { hour: number; minute: number; second: number }

export type Landscape = {
  step(dt: number, clock: number, pointer: Pointer): void
  portal(mode: Mode): void
  splash(x: number, y: number, size: number): void
  // Repaints the layers that carry the clock, for when nothing else is animating them.
  refresh(): void
  // Where the visitor is and their weather, once the worker has said.
  place(site: Site): void
  // Reduced motion: poses the cat and the crew for where the pointer is, repainting only on a change.
  still(pointer: Pointer): void
  // Whoever is under the click (client pixels) asks "?"; true if someone was there.
  hail(x: number, y: number): boolean
  // Buffer pixel pitch and origin in scene units, for snapping vector sprites onto the grid.
  pixel(): { px: number; x0: number; y0: number } | null
  dispose(): void
}

const TICK = 1 / 10
// Gold only where the tone is mid-range and the gradient is sharp: fringes, never interiors.
const GOLD_START = 0.45
const GOLD_END = 0.7

const LIGHT: Palette = {
  sky: ['#b8d9d2', '#c9e1d9', '#dae9e0', '#ecf1e6', '#fffcf0'].map(hex),
  cloud: ['#a6abbd', '#bcbfcc', '#d2d3da'].map(hex),
  swirl: [hex('#9fcbc2'), hex('#fffcf0')],
  sun: ['#fbe3d6', '#fdc4b4', '#fdb2a2'].map(hex),
  far: ['#b3c6d2', '#c7d6de', '#dde6e6'].map(hex),
  mid: ['#879578', '#a2ad8c', '#c3c9a6'].map(hex),
  hill: ['#56652a', '#74853a', '#97a652', '#bcc47e'].map(hex),
  pine: ['#5d6b4c', '#6d7b5b', '#83906c'].map(hex),
  bark: ['#3e3128', '#5e4b3b', '#7e6a55'].map(hex),
  leaf: ['#2f3d0c', '#4d6212', '#7d902a', '#b4bf6a'].map(hex),
  crown: ['#2f3d0c', '#4d6212', '#7d902a', '#b4bf6a'].map(hex),
  autumn: ['#6b2c0b', '#bc5215', '#da702c', '#d0a215'].map(hex),
  dun: ['#6b6a45', '#8a875c', '#aaa57a', '#c9c39c'].map(hex),
  snow: ['#98a9bf', '#bfcad8', '#dfe5ea', '#fffcf0'].map(hex),
  rock: ['#57534b', '#7f796c', '#a39c8c', '#cdc6b2'].map(hex),
  gold: hex('#e0b43c'),
  poppy: hex('#e0604a'),
  cornflower: hex('#4f72b0'),
  ochre: hex('#d8bd55'),
  star: hex('#fffcf0'),
  glint: ['#fffcf0', '#fdd6c9'].map(hex),
  figure: {
    C: hex('#fffcf0'),
    S: hex('#e2ab91'),
    B: hex('#3f6db0'),
    D: hex('#28497e'),
    K: hex('#100f0f'),
  },
  shade: [...hex('#34435e'), 0.2],
  fringe: [...hex('#d9a93a'), 0.32],
  water: ['#8fc9be', '#a4d5ca', '#bbe0d6'].map(hex),
  smoke: ['#8f909c', '#a9aab4', '#c3c3ca'].map(hex),
  glow: ['#a8984a', '#c9a557', '#e8bf6a'].map(hex),
  coats: [hex('#bc5215'), hex('#24837b')],
  tent: [hex('#d98f7e'), hex('#fdb2a2')],
  moss: ['#4d6212', '#74853a', '#97a652', '#c6cc84'].map(hex),
  contour: [hex('#8199ab'), hex('#5f6c50'), hex('#56652a')],
  crew: ['#bc5215', '#24837b', '#205ea6', '#ad8301', '#5e409d', '#a02f6f'].map(hex),
  rope: hex('#7e6a55'),
  fish: hex('#6f8fa8'),
}

const DARK: Palette = {
  sky: ['#121726', '#141a28', '#161b25', '#141619', '#100f0f'].map(hex),
  cloud: ['#1b2233', '#212940', '#29324b'].map(hex),
  swirl: [hex('#0d1019'), hex('#33405e')],
  sun: ['#2a2622', '#8f8a7c', '#cecdc3'].map(hex),
  far: ['#1b1f25', '#20252b', '#262c33'].map(hex),
  mid: ['#191e14', '#20271a', '#293221'].map(hex),
  hill: ['#12160a', '#1a200e', '#232b13', '#2e3819'].map(hex),
  pine: ['#161b11', '#1e2517', '#262f1d'].map(hex),
  bark: ['#0d0b09', '#1c1814', '#2c251f'].map(hex),
  leaf: ['#0b0f04', '#151d07', '#24300c', '#3d4c10'].map(hex),
  crown: ['#0b0f04', '#151d07', '#24300c', '#3d4c10'].map(hex),
  autumn: ['#1e0d04', '#3d1a07', '#5c2a0c', '#6e5406'].map(hex),
  dun: ['#15150e', '#1e1d14', '#29281c', '#363426'].map(hex),
  snow: ['#1f2533', '#2b3342', '#3c4556', '#58627a'].map(hex),
  rock: ['#0e0d0b', '#171613', '#22201b', '#302d25'].map(hex),
  gold: hex('#5a4714'),
  poppy: hex('#8c3a2e'),
  cornflower: hex('#27365a'),
  ochre: hex('#4d4519'),
  star: hex('#e6e4d9'),
  glint: ['#cecdc3', '#6f6a5e'].map(hex),
  figure: {
    C: hex('#cecdc3'),
    S: hex('#8c6b5c'),
    B: hex('#2d4c7c'),
    D: hex('#1d3356'),
    K: hex('#050505'),
  },
  shade: [...hex('#fffcf0'), 0.05],
  fringe: [...hex('#d0a215'), 0.14],
  water: ['#132726', '#193230', '#203d3a'].map(hex),
  smoke: ['#2e323d', '#3b3f4b', '#4a4d58'].map(hex),
  glow: ['#2e2412', '#5c3514', '#9a5019'].map(hex),
  coats: [hex('#7a3a12'), hex('#1b5953')],
  tent: [hex('#4a302b'), hex('#7d4f45')],
  moss: ['#1a220a', '#2c3a10', '#3d4c10', '#5a6b22'].map(hex),
  contour: [hex('#39414b'), hex('#3a4530'), hex('#3d4a22')],
  crew: ['#7a3a12', '#1b5953', '#1a3f6e', '#6e5406', '#3c2a66', '#6a1f4a'].map(hex),
  rope: hex('#8f8a7c'),
  fish: hex('#4f6b82'),
}

// Twilight: the sky warms from the horizon up, the far range and the water go violet.
const DUSK = {
  sky: ['#4a5478', '#7d7a9c', '#c98f98', '#f0a58c', '#fdc9a0'].map(hex),
  cloud: ['#6d6285', '#9a7f98', '#d59a96'].map(hex),
  sun: ['#fdd2a0', '#f59a6e', '#e0604a'].map(hex),
  far: ['#7e7496', '#978aa8', '#b3a3b8'].map(hex),
  water: ['#8a7f9c', '#b095a6', '#d8aca4'].map(hex),
  glint: ['#fdd6a0', '#f0a58c'].map(hex),
} satisfies Partial<Palette>

// Colour by colour, part way from one palette (or any nest of colours) to another.
function blend<T>(a: T, b: T, t: number): T {
  if (t <= 0) return a
  if (t >= 1) return b
  if (typeof a === 'number') return (a + ((b as number) - a) * t) as T
  if (Array.isArray(a)) return a.map((v, k) => blend(v, (b as unknown[])[k], t)) as T
  return Object.fromEntries(
    Object.entries(a as object).map(([k, v]) => [
      k,
      blend(v, (b as Record<string, unknown>)[k], t),
    ]),
  ) as T
}

// The year where the visitor is, from days since midwinter: how far the leaves have turned or
// fallen, how thick the flowers stand in the meadow, how far down the ranges the snow lies (as a
// height in scene units), and how much of it lies on the island.
type Season = {
  autumn: number
  bare: number
  bloom: number
  // The trees in flower.
  blossom: number
  snowline: number
  ground: number
  // Hoarfrost on the island, and how far the sun has taken it back: at 0 it covers everything, at
  // 1 it holds only in the deepest shade.
  frost: number
  thaw: number
}

function season(al: Almanac, sky: Sky): Season {
  const { winterDay } = al
  const near = (peak: number, width: number) => {
    const off = ((((winterDay - peak + 182.6) % 365.24) + 365.24) % 365.24) - 182.6
    return Math.max(0, 1 - Math.abs(off) / width)
  }
  // Lowest six weeks after midwinter, highest in late summer, and down to the water while it snows.
  let snowline = 435 + 165 * Math.cos((2 * Math.PI * (winterDay - 40)) / 365.24)
  const cold = sky.temp !== null && sky.temp < 1
  if (sky.fall === 'snow') snowline = Math.max(snowline, 760)
  else if (cold) snowline = Math.max(snowline, 560)
  const bare = near(40, 75)
  // Under a clear, still sky the grass radiates its warmth away and runs three to five degrees
  // colder than the air, so it rimes over with the air still above freezing. Cloud sends the warmth
  // back, and wind stirs the warmer air down onto it. The frost gathers in the shade first after
  // sunset; the morning sun takes it back from the slopes it strikes first, and by noon it holds
  // only where the sun never reaches.
  const still = (1 - smooth(0.2, 0.55, sky.cover)) * (1 - smooth(0.2, 0.45, sky.wind))
  const chill =
    sky.temp === null || sky.fall !== 'none' || sky.fog ? 0 : 1 - smooth(0, 4.5, sky.temp)
  const thaw =
    al.hour < 12
      ? Math.max(smooth(0, 20, al.sun.alt), smooth(9, 12, al.hour))
      : 1 - smooth(0, -18, al.sun.alt)
  return {
    autumn: near(300, 55),
    bare,
    bloom: Math.max(0.15, 0.3 + 1.2 * near(130, 50) + 0.6 * near(200, 60) - 0.3 * bare),
    blossom: near(105, 35),
    snowline,
    ground: sky.fall === 'snow' ? 0.5 + 0.5 * sky.heavy : cold && bare > 0.3 ? 0.35 : 0,
    frost: chill * still,
    thaw,
  }
}

const tint = (a: RGB[], b: RGB[], t: number) => (t <= 0 ? a : a.map((c, k) => mix(c, b[k], t)))

// One theme's palette, turned for the season and greyed by the weather.
function weathered(p: Palette, s: Season, sky: Sky): Palette {
  const over = smooth(0.55, 1, sky.cover) * 0.6 + (sky.fog ? 0.3 : 0)
  const overcast = [p.cloud[1], p.cloud[1], p.cloud[2], p.cloud[2], p.cloud[2]]
  const haze = p.sky.slice(2)
  let hill = tint(p.hill, p.autumn, s.autumn * 0.15)
  hill = tint(hill, p.dun, s.bare * 0.45)
  return {
    ...p,
    sky: tint(p.sky, overcast, over),
    sun: tint(p.sun, p.cloud, over * 0.6),
    far: tint(p.far, haze, sky.fog ? 0.8 : 0),
    mid: tint(p.mid, haze, sky.fog ? 0.5 : 0),
    hill: tint(hill, p.snow, Math.min(0.95, s.ground * 1.15)),
    // What leaves hang on through the winter are dead ones.
    crown: tint(p.crown, p.dun, s.bare * 0.85),
    autumn: tint(p.autumn, p.dun, s.bare * 0.5),
  }
}

// The palette for this moment: night and day blended by how much daylight there is (a dark page
// only ever sees a dim day), then warmed through twilight.
function lighting(al: Almanac, s: Season, sky: Sky, dim: boolean): [Palette, number] {
  // A storm's anvil or a downpour takes a good part of the daylight with it.
  const gloom = sky.storm
    ? 0.45
    : sky.fall === 'rain' || sky.fall === 'drizzle'
      ? 0.12 + 0.25 * sky.heavy
      : 0.15 * smooth(0.8, 1, sky.cover)
  const day = (dim ? al.day * 0.3 : al.day) * (1 - gloom)
  const night = weathered(DARK, s, sky)
  const pal = blend(night, weathered(LIGHT, s, sky), day)
  const glow = al.dusk * 0.75 * (1 - smooth(0.55, 1, sky.cover) * 0.6)
  if (glow <= 0) return [pal, day]
  const out = { ...pal }
  for (const key of Object.keys(DUSK) as (keyof typeof DUSK)[])
    out[key] = blend(pal[key], blend(night[key], DUSK[key], 0.4 + 0.6 * day), glow)
  return [out, day]
}

// The sun or the moon as drawn: its centre and radius in scene units, its phase (0.5 is the full
// disc, which is how the sun is always drawn), whether the lit limb is mirrored (south of the
// equator the moon waxes from the left), and how much cloud veils it.
type Disc = { x: number; y: number; r: number; phase: number; mirror: boolean; veil: number }

// Where a body hangs over this grid: across the visible sky from its azimuth, and up from the
// skyline under it by its altitude, so it sets behind whatever ridge stands there.
function hang(b: Body, g: Grid, r: number) {
  const left = g.sx[0] + 48
  const right = g.sx[g.bw - 1] - 48
  const half = Math.min(700, (right - left) / 2 - 60)
  const x = (left + right) / 2 + b.across * half * 0.85
  const floor = skyline(x)
  const y =
    b.alt >= 0
      ? floor - (floor - 70) * Math.sin((b.alt * Math.PI) / 180) ** 0.6
      : floor - b.alt * 8 + r * 0.4
  return { x, y }
}

// 1 on the lit part of the disc, 0 on its dark side, -1 off it.
function onDisc(b: Disc, x: number, y: number) {
  const u = (x - b.x) / b.r
  const v = (y - b.y) / b.r
  if (u * u + v * v >= 1) return -1
  const rim = Math.sqrt(1 - v * v)
  const k = Math.cos(2 * Math.PI * b.phase)
  const w = b.mirror ? -u : u
  return (b.phase <= 0.5 ? w > k * rim : w < -k * rim) ? 1 : 0
}

// Octaves drift at their own speed, so coarse masses sway while fine leaves shimmer.
function windFbm(x: number, y: number, seed: number, octaves: number, wind: number) {
  let sum = 0
  let amp = 0.5
  let norm = 0
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise(x + wind * (1 + i * 0.7), y, seed + i * 17)
    norm += amp
    amp *= 0.5
    x *= 2.03
    y *= 2.03
  }
  return sum / norm
}

// Probabilistic L-system: X → FX | F[+X]X | F[-X]X | F[+X][-X]X, unrolled until every limb reaches
// the terminal width. The trunk stays bare until it thins, forks conserve cross-sectional area
// (da Vinci), and sides follow a running golden-angle phase so branching never settles into a beat.
function growTree(seed: number, x: number, y: number, lean: number, width: number, len: number) {
  const r = mulberry(seed)
  const segs: Seg[] = []
  const tips: Tip[] = []
  const up = -Math.PI / 2
  const stack = [{ x, y, a: up + lean, w: width, l: len, phase: r() * Math.PI * 2 }]
  while (stack.length) {
    const s = stack.pop()!
    if (s.w < 1.6 || segs.length > 1400) {
      // The phyllotaxis phase that picks a side also says how far the limb reaches toward us.
      tips.push([s.x, s.y, Math.sin(s.phase)])
      continue
    }
    const w1 = s.w * 0.95
    const nx = s.x + Math.cos(s.a) * s.l
    const ny = s.y + Math.sin(s.a) * s.l
    segs.push({ x0: s.x, y0: s.y, x1: nx, y1: ny, w0: s.w, w1 })
    const curl = (a: number, k: number) => a + (up - a) * k + (r() - 0.5) * 0.3
    const roll = r()
    if (roll < 0.3 || s.w > width * 0.8) {
      stack.push({ ...s, x: nx, y: ny, a: curl(s.a, 0.1), w: w1, l: s.l * 0.94 })
      continue
    }
    const both = roll > 0.79
    const phase = s.phase + 2.39996
    const side = Math.cos(phase) >= 0 ? 1 : -1
    const keep = both ? 0.45 + r() * 0.15 : 0.6 + r() * 0.2
    stack.push({ x: nx, y: ny, a: curl(s.a, 0.1), w: w1 * Math.sqrt(keep), l: s.l * 0.9, phase })
    const kids = both ? [side, -side] : [side]
    const kw = w1 * Math.sqrt((1 - keep) / kids.length)
    for (const k of kids)
      stack.push({
        x: nx,
        y: ny,
        a: curl(s.a + k * (0.35 + r() * 0.4), 0.14),
        w: kw,
        l: s.l * (0.7 + r() * 0.2),
        phase: phase + k,
      })
  }
  return { segs, tips } satisfies Tree
}

function valley(x: number, reach: number) {
  return Math.max(0, 1 - Math.abs(x - 800) / reach)
}

const farRidge = (x: number) =>
  640 - 440 * (1 - valley(x, 820) ** 1.3) + (fbm(x * 0.004, 3.1, 11, 4) - 0.5) * 150
const midRidge = (x: number) =>
  770 - 360 * (1 - valley(x, 600) ** 0.85) + (fbm(x * 0.006, 7.7, 23, 4) - 0.5) * 110
const skyline = (x: number) => Math.min(farRidge(x), midRidge(x))

// The island: a broad massif rising out of the water to two rounded summits, the gate sunk into the
// saddle between them, a shoulder on the left where the fallen stone lies and a bench on the right
// flank where the cat has settled. It is modelled the way the ranges are, spurs and gullies down
// its faces, with its crest roughened as theirs is.
export function hillTop(x: number, water: number) {
  const u = Math.max(-1, Math.min(1, (x - 800) / 700))
  const bell = (0.5 + 0.5 * Math.cos(Math.PI * u)) ** 0.7
  const bump = (c: number, w: number, h: number) => h * Math.exp(-(((x - c) / w) ** 2))
  const rise =
    235 * bell +
    bump(610, 110, 40) +
    bump(1010, 100, 45) -
    bump(800, 90, 10) +
    bump(1290, 60, 32) +
    bump(330, 80, 20) +
    (fbm(x * 0.008, 2.3, 19, 3) - 0.5) * 34 * bell
  return water - Math.max(0, rise)
}

// An animated canvas keeps one ImageData and repaints into it every tick.
type Layer = { g: Grid; ctx: CanvasRenderingContext2D; img: ImageData }

function layer(canvas: HTMLCanvasElement): Layer | null {
  const g = grid(canvas)
  if (!g) return null
  const ctx = canvas.getContext('2d')!
  return { g, ctx, img: ctx.createImageData(g.bw, g.bh) }
}

type Angle = (x: number, y: number) => number
type Keep = (i: number, j: number) => boolean

// Curved brush stroke (Hertzmann 1998): march one buffer pixel at a time along the flow in both
// directions from the seed, two pixels wide through the body and one at the tips. Flow angles
// are orientations, so each step flips to stay continuous with the last.
function brush(
  d: Uint8ClampedArray,
  g: Grid,
  x: number,
  y: number,
  len: number,
  wide: boolean,
  c: RGB,
  angle: Angle,
  keep?: Keep,
) {
  const { bw, bh } = g
  const dab = (px: number, py: number) => {
    const i = Math.round(px)
    const j = Math.round(py)
    if (i < 0 || j < 0 || i >= bw || j >= bh || (keep && !keep(i, j))) return
    put(d, (j * bw + i) * 4, c)
  }
  for (const dir of [1, -1]) {
    const n = dir > 0 ? Math.ceil(len / 2) : Math.floor(len / 2) + 1
    let cx = x
    let cy = y
    let lx = 0
    let ly = 0
    for (let s = 0; s < n; s++) {
      const a = angle(cx, cy)
      let dx = Math.cos(a) * dir
      let dy = Math.sin(a) * dir
      if (s > 0 && dx * lx + dy * ly < 0) {
        dx = -dx
        dy = -dy
      }
      if (dir > 0 || s > 0) {
        dab(cx, cy)
        if (wide && s < n - 1) dab(cx - dy, cy + dx)
      }
      lx = dx
      ly = dy
      cx += dx
      cy += dy
    }
  }
}

// Jittered lattice of stroke seeds riding a medium that has drifted `shift` buffer pixels, so a
// stroke keeps its place in the medium, and its dither threshold, while it travels. Two
// interleaved passes let later strokes overlap earlier ones without a raster grain.
function seeds(
  g: Grid,
  cell: number,
  shift: number,
  rows: [number, number],
  seed: number,
  visit: (x: number, y: number, h: number, k: number) => void,
) {
  const c0 = Math.floor(-shift / cell) - 1
  const c1 = Math.ceil((g.bw - shift) / cell) + 1
  const r0 = Math.max(0, Math.floor(rows[0] / cell) - 1)
  const r1 = Math.min(Math.ceil(g.bh / cell), Math.ceil(rows[1] / cell) + 1)
  for (const pass of [0, 1])
    for (let cj = r0; cj <= r1; cj++)
      for (let ci = c0; ci <= c1; ci++) {
        const h = hash(ci, cj, seed)
        if ((h < 0.5 ? 0 : 1) !== pass) continue
        visit(
          ci * cell + hash(ci, cj, seed + 1) * cell + shift,
          cj * cell + hash(ci, cj, seed + 2) * cell,
          frac(h * 2),
          hash(ci, cj, seed + 3),
        )
      }
}

// Sky flow. A stream function of three drifting plane waves: its curl is divergence-free, so the
// strokes eddy without pooling. Lamb–Oseen vortices on top: the sun's halo, the eddies the
// cursor sheds, and the portal's pull (a vortex plus a sink, so the sky spirals into the gate).
type Vortex = { x: number; y: number; gamma: number; age: number }
type Air = {
  drift: number
  time: number
  pull: number
  pole: Pt
  vortices: Vortex[]
  body: Disc | null
  cover: number
}

const WAVES = [
  [0.006, 0.014, 0.11, 0.4],
  [-0.013, 0.021, -0.17, 2.1],
  [0.024, -0.017, 0.23, 4.4],
].map(([kx, ky, w, phi]) => ({ kx, ky, w, phi, a: 6 / Math.hypot(kx, ky) }))

const V = { x: 0, y: 0 }

function vortex(dx: number, dy: number, gamma: number, core: number, sink: number) {
  const r2 = dx * dx + dy * dy + 1
  const k = (1 - Math.exp(-r2 / (core * core))) / (2 * Math.PI * r2)
  V.x += (-dy * gamma - dx * sink) * k
  V.y += (dx * gamma - dy * sink) * k
}

function airAngle(air: Air, x: number, y: number) {
  V.x = 24
  V.y = 2
  const u = x - air.drift
  for (const w of WAVES) {
    const c = w.a * Math.cos(w.kx * u + w.ky * y + w.w * air.time + w.phi)
    V.x += w.ky * c
    V.y -= w.kx * c
  }
  if (air.body) vortex(x - air.body.x, y - air.body.y, 42000, 70, 0)
  for (const v of air.vortices) vortex(x - v.x, y - v.y, v.gamma, 40 + Math.sqrt(v.age) * 30, 0)
  if (air.pull > 0.01)
    vortex(x - air.pole[0], y - air.pole[1], air.pull * 160000, 240, air.pull * 70000)
  return Math.atan2(V.y, V.x)
}

// What a sky stroke carries: the sun or the lit part of the moon, its banded halo, drifting cloud,
// or the sky gradient. More cloud lowers the threshold and lets the bank reach further down.
function skyInk(pal: Palette, air: Air, x: number, y: number, thr: number, mask: number) {
  const b = air.body
  if (b && b.veil < 0.85) {
    const r = Math.hypot(x - b.x, y - b.y)
    if (r < b.r && onDisc(b, x, y) === 1) return pick(pal.sun, 0.7 + 0.3 * (1 - r / b.r), thr)
    if (r >= b.r && r < b.r + 70 && b.veil < 0.4) {
      const fall = 1 - (r - b.r) / 70
      const band = 0.5 + 0.5 * Math.cos((r - b.r) * 0.21)
      if (band * fall > 0.3 + mask * 0.25) return pick(pal.sun, fall * 0.9, thr)
    }
  }
  const c = fbm((x - air.drift) * 0.0035, y * 0.012, 131, 3)
  const lo = 0.62 - 0.4 * air.cover
  const low = Math.max(0, air.cover - 0.5) * 2
  const cloud = smooth(lo, lo + 0.18, c) * (1 - smooth(230 + 260 * low, 480 + 300 * low, y))
  if (cloud > mask) return pick(pal.cloud, (c - 0.5) * 3.5, thr)
  return pick(pal.sky, smooth(60, 520, y) + (c - 0.5) * 0.35, thr)
}

// Inside an eddy the strokes take alternating light and dark bands by distance from its centre,
// so the swirl reads the way the swirls do in the Saint-Rémy night sky.
function eddyInk(pal: Palette, air: Air, x: number, y: number, mask: number) {
  let swirl = 0
  let band = 0
  const feel = (dx: number, dy: number, strength: number, reach: number, wave: number) => {
    const r2 = dx * dx + dy * dy
    const e = strength * Math.exp(-r2 / (2 * reach * reach))
    if (e <= swirl) return
    swirl = e
    band = Math.cos(Math.sqrt(r2) * wave)
  }
  for (const v of air.vortices)
    feel(x - v.x, y - v.y, Math.min(1, Math.abs(v.gamma) / 30000), 70 + Math.sqrt(v.age) * 50, 0.16)
  if (air.pull > 0.01)
    feel(x - air.pole[0], y - air.pole[1], Math.min(1, air.pull) * 0.7, 420, 0.045)
  if (swirl <= mask) return null
  return band > 0.2 ? pal.swirl[1] : band < -0.2 ? pal.swirl[0] : null
}

// The dark side of the moon keeps a trace of earthshine.
function paintSkyBase(d: Uint8ClampedArray, g: Grid, pal: Palette, floor: number, b: Disc | null) {
  const { bw, bh, sx, sy } = g
  const shown = b && b.veil < 0.85
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    if (y > floor) break
    for (let i = 0; i < bw; i++) {
      const k = (j * bw + i) * 4
      const sky = ramp(pal.sky, smooth(60, 520, y), i, j)
      const r = shown ? Math.hypot(sx[i] - b.x, y - b.y) : Infinity
      const on = shown && r < b.r ? onDisc(b, sx[i], y) : -1
      if (on === 1) put(d, k, pal.sun[2])
      else if (on === 0) put(d, k, mix(sky, pal.sun[0], 0.18))
      else if (shown && r < b.r + 18 && b.veil < 0.4 && b.phase > 0.3 && b.phase < 0.7)
        put(d, k, ramp(pal.sun, 1 - (r - b.r) / 18, i, j))
      else put(d, k, sky)
    }
  }
}

type SkyStar = { x: number; y: number; mag: number; freq: number; phase: number }

// Fewer show through cloud, and none near the moon.
function makeStars(b: Disc | null, cover: number): SkyStar[] {
  const r = mulberry(77)
  const stars: SkyStar[] = []
  const most = Math.round(18 * (1 - smooth(0.2, 0.9, cover)))
  for (let n = 0; n < 80 && stars.length < most; n++) {
    const x = 40 + r() * 1520
    const y = 30 + r() * 420
    const star = { x, y, mag: r() ** 1.6, freq: 0.6 + r() * 1.6, phase: r() * Math.PI * 2 }
    if (y < skyline(x) - 50 && (!b || Math.hypot(x - b.x, y - b.y) > 130)) stars.push(star)
  }
  return stars
}

// A broken ring of pixels, the dashes turning with `turn`: the halos of the Saint-Rémy stars.
function halo(
  plot: (x: number, y: number, c: RGB, alpha: number) => void,
  cx: number,
  cy: number,
  radius: number,
  step: number,
  c: RGB,
  alpha: number,
  turn: number,
  dashes: number,
) {
  const n = Math.ceil((2 * Math.PI * radius) / step)
  for (let k = 0; k < n; k++) {
    const u = k / n
    if (frac(u * dashes + turn) < 0.38) continue
    plot(cx + Math.cos(u * 2 * Math.PI) * radius, cy + Math.sin(u * 2 * Math.PI) * radius, c, alpha)
  }
}

function paintSky(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  air: Air,
  base: Uint8ClampedArray,
  ridge: Float32Array,
  stars: SkyStar[],
  clock: number,
) {
  d.set(base)
  const x0 = g.sx[0]
  const y0 = g.sy[0]
  const angle: Angle = (bx, by) => airAngle(air, x0 + bx * g.px, y0 + by * g.px)
  let floor = 0
  for (const y of ridge) floor = Math.max(floor, y)
  // The disc stays whole: only its own strokes may cross into it.
  const b = air.body && air.body.veil < 0.85 ? air.body : null
  const clear: Keep = (i, j) => !b || Math.hypot(g.sx[i] - b.x, g.sy[j] - b.y) > b.r
  seeds(g, 3, air.drift / g.px, [0, row(g, floor)], 131, (bx, by, h, k) => {
    const x = x0 + bx * g.px
    const y = y0 + by * g.px
    if (y > ridge[clampCol(g, bx)] + g.px * 2) return
    const c = eddyInk(pal, air, x, y, k) ?? skyInk(pal, air, x, y, h, k)
    const len = 6 + Math.floor(frac(k * 7.13) * 5)
    const near = !!b && Math.hypot(x - b.x, y - b.y) < b.r + len * g.px
    const guard = near && !pal.sun.includes(c) ? clear : undefined
    brush(d, g, bx, by, len, frac(k * 3.7) < 0.25, c, angle, guard)
  })

  const plot = (x: number, y: number, c: RGB, alpha: number) => {
    const i = col(g, x)
    const j = row(g, y)
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  for (const s of stars) {
    const twinkle = 0.6 + 0.4 * Math.sin(clock * s.freq + s.phase)
    plot(s.x, s.y, pal.star, 1)
    if (s.mag > 0.4)
      for (const [ox, oy] of CROSS) plot(s.x + ox * g.px, s.y + oy * g.px, pal.star, twinkle * 0.8)
    if (s.mag > 0.65)
      halo(
        plot,
        s.x,
        s.y,
        (2.4 + s.mag * 1.4) * g.px,
        g.px * 0.8,
        pal.sun[1],
        twinkle * 0.7,
        clock * 0.2 + s.phase,
        5,
      )
  }
}

// A storm throws a bolt every few seconds: a jagged line from the cloud base down to the ridges,
// lit for two ticks with the sky blanched round it.
function paintBolt(d: Uint8ClampedArray, g: Grid, clock: number, ridge: Float32Array) {
  const n = Math.floor(clock / 6.5)
  const t = clock - n * 6.5 - hash(n, 1, 89) * 4
  if (t < 0 || t > 0.22) return
  for (let k = 0; k < d.length; k += 4)
    if (d[k + 3]) put(d, k, mix([d[k], d[k + 1], d[k + 2]], CREAM, 0.22))
  const fork = (i: number, j: number, seed: number, reach: number) => {
    for (let s = 0; s < reach; s++) {
      const floor = row(g, ridge[clampCol(g, i)])
      if (j >= floor) return
      dot(d, g, i, j, CREAM)
      j++
      if (hash(i, j, seed) < 0.4) i += hash(j, i, seed) < 0.5 ? -1 : 1
      if (seed === n && s === 9) fork(i, j, n + 1, 14)
    }
  }
  fork(Math.floor(g.bw * (0.1 + 0.8 * hash(n, 2, 89))), row(g, 110 + 60 * hash(n, 3, 89)), n, 200)
}

const CROSS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
]

// Flame-shaped cypress: widest a third of the way up, pinched to a point, lit on the right flank,
// then painted over with upward strokes that fan toward the flanks and turn in an S as they climb.
function paintCypress(
  d: Uint8ClampedArray,
  g: Grid,
  c: Cypress,
  stops: RGB[],
  seed: number,
  sway = 0,
) {
  const { bw, bh, sx, sy } = g
  const lean = (t: number) => Math.sin(t * 3 + seed) * c.w * 0.08 + sway * t * t * c.w
  const half = (y: number, t: number) => {
    const profile = t < 0.3 ? Math.sqrt(t / 0.3) : ((1 - t) / 0.7) ** 0.85
    return Math.max(g.px * 0.5, (c.w / 2) * profile * (0.82 + 0.36 * noise(y * 0.06, seed, seed)))
  }
  const i0 = Math.max(0, col(g, c.x - c.w))
  const i1 = Math.min(bw - 1, col(g, c.x + c.w))
  const j0 = Math.max(0, row(g, c.base - c.h))
  const j1 = Math.min(bh - 1, row(g, c.base))
  for (let j = j0; j <= j1; j++) {
    const y = sy[j]
    const t = (c.base - y) / c.h
    if (t < 0 || t > 1) continue
    const hw = half(y, t)
    const mid = c.x + lean(t)
    for (let i = i0; i <= i1; i++) {
      const u = (sx[i] - mid) / hw
      if (Math.abs(u) > 1) continue
      const tone = 0.3 + u * 0.4 + (fbm(sx[i] * 0.08, y * 0.05, seed, 2) - 0.5) * 0.5
      put(d, (j * bw + i) * 4, ramp(stops, tone, i, j))
    }
  }
  let fan = 0
  const angle: Angle = (_, by) => {
    const t = clamp01((c.base - (sy[0] + by * g.px)) / c.h)
    return -Math.PI / 2 + fan * 0.5 * (1 - t) + 0.45 * Math.sin(t * 9 + seed + sway * 4)
  }
  for (let j = j0; j <= j1; j += 2)
    for (let i = i0; i <= i1; i += 2) {
      const t = (c.base - sy[j]) / c.h
      if (t < 0.03 || t > 0.97) continue
      fan = (sx[i] - c.x - lean(t)) / half(sy[j], t)
      if (Math.abs(fan) > 0.9) continue
      const h = hash(i, j, seed)
      const ink = pick(stops, 0.32 + fan * 0.4 + (h - 0.5) * 0.3, hash(i, j, seed + 1))
      brush(d, g, i + h - 0.5, j, 3 + Math.floor(h * 3), false, ink, angle)
    }
}

// How the land is lit: from which side (1 when the light comes from the right), how hard (cloud
// flattens it), and how far down the ranges the snow reaches, in scene units.
type Relief = {
  side: number
  contrast: number
  snowline: number
  dark: boolean
  frost: number
  thaw: number
}

// Whether hoarfrost holds on ground of this tone, which runs from about 0 in shade to 1 square to
// the light (the brightest turf a little past it). A marginal night frosts only the coldest shade,
// and the sun clears the brightest ground first, so both move the frost's edge down the tones
// instead of thinning it everywhere.
function rimeAt(r: Relief, tone: number) {
  if (r.frost <= 0) return 0
  const edge = -0.4 + 1.7 * r.frost * (1 - r.thaw)
  return smooth(edge + 0.08, edge - 0.08, tone)
}
// Frosted grass keeps the green under its rime; a three-stop ramp takes the upper snow stops.
const rimeOver = (stops: RGB[], snow: RGB[]) =>
  stops.map((c, k) => mix(c, snow[k + snow.length - stops.length], 0.7))

// Spurs and gullies run down each face from the crest, bending and spreading as they fall. Across
// the face a spur's crest sits at every whole number of this phase and the gully between two spurs
// at every half, so each spur is two flat planes: one turned to the light and one away from it.
// That is what gives the ranges their bulk. Rock shows along the spur crests.
const spur = (x: number, down: number, seed: number, freq: number) =>
  x * freq + fbm(x * 0.004, seed, seed, 2) * 1.4 + Math.sin(down * 0.011 + x * 0.002 + seed) * 0.3

// Spurs leave the crest square to it and fall toward the valley, so on either flank of a summit they
// fan out downhill. The ridge's slope is read over 180 units and the drift is capped, which keeps
// neighbouring spurs from crossing.
const lean = (ridge: (x: number) => number, x: number) =>
  Math.max(-1.2, Math.min(1.2, (ridge(x + 90) - ridge(x - 90)) / 180))
const spurAt = (back: boolean, x: number, down: number, tilt: number) => {
  const x0 = x - tilt * 110 * Math.tanh(down / 100)
  return back ? spur(x0, down, 17, 0.011) : spur(x0, down, 29, 0.008)
}

// Which way the face turns at a phase, +1 toward the right: sharp over a spur's crest, eased
// across the gully's floor.
const facing = (u: number) => {
  const f = frac(u)
  return f < 0.5 ? 1 - 2 * smooth(0.4, 0.6, f) : 2 * smooth(0.4, 0.6, 1 - f) - 1
}

// A glacier under the far right crest keeps its snow all year, so the lift above the trail has
// something to run on in August.
const glacier = (x: number, y: number) => x > 1372 && x < 1500 && y < 322 + (x - 1372) * 0.4

// An outcrop, as a low block with a summit (`peak`, across it from -1 to 1) and a lower shoulder.
type Rock = {
  x: number
  base: number
  w: number
  h: number
  peak: number
  shoulder: number
  mid: boolean
}

// An outcrop breaking the face: a flank turned toward the light and one turned away, split at the
// summit, with a lit edge along the sunny flank's top, a darker course where it beds into the
// slope, and snow on top above the snowline. Its top steps unevenly from column to column.
function paintRock(d: Uint8ClampedArray, g: Grid, pal: Palette, rock: Rock, relief: Relief) {
  const { bw, bh, sx, sy } = g
  const { side, contrast } = relief
  const half = rock.w / 2
  const i0 = Math.max(0, col(g, rock.x - half))
  const i1 = Math.min(bw - 1, col(g, rock.x + half))
  const j0 = Math.max(0, row(g, rock.base - rock.h * 1.2))
  const j1 = Math.min(bh - 1, row(g, rock.base))
  const tent = (t: number, at: number, h: number) =>
    Math.max(0, t < at ? ((t + 1) / (at + 1)) * h : ((1 - t) / (1 - at)) * h)
  const snowy = rock.base - rock.h < relief.snowline
  const edge = g.px / rock.h
  const seed = Math.round(rock.x)
  for (let i = i0; i <= i1; i++) {
    const t = (sx[i] - rock.x) / half
    if (Math.abs(t) >= 1) continue
    const top =
      Math.max(tent(t, rock.peak, 1), tent(t, rock.shoulder, 0.6)) +
      (hash(i, seed, 7) - 0.5) * 2 * edge
    const lit = (t < rock.peak ? -side : side) * contrast
    for (let j = j0; j <= j1; j++) {
      const up = (rock.base - sy[j]) / rock.h
      if (up >= top || up < 0) continue
      const k = (j * bw + i) * 4
      if (snowy && up > top - 0.4) put(d, k, ramp(pal.snow, 0.6 + lit * 0.35, i, j))
      else if (lit > 0 && top - up < edge) put(d, k, pal.rock[3])
      else put(d, k, ramp(pal.rock, 0.4 + lit * 0.45 - (up < edge ? 0.2 : 0), i, j))
    }
  }
}

// The ranges, laid in strokes that follow each ridge's profile and relax toward level deeper in
// the face, the way the Alpilles are combed in the Saint-Rémy canvases, then modelled by the spurs
// and capped with snow down to the snowline. The sky stays clear.
function paintLand(d: Uint8ClampedArray, g: Grid, pal: Palette, pines: Cypress[], relief: Relief) {
  const { bw, bh, sx, sy } = g
  const { side, contrast, snowline } = relief
  const far = Float32Array.from(sx, farRidge)
  const mid = Float32Array.from(sx, midRidge)
  const slope = (f: (x: number) => number, x: number, h: number) => (f(x + h) - f(x - h)) / (2 * h)
  const farLit = Float32Array.from(sx, x => slope(farRidge, x, 8) * side * contrast)
  const midLit = Float32Array.from(sx, x => slope(midRidge, x, 8) * side * contrast)
  const farAng = Float32Array.from(sx, x => Math.atan(slope(farRidge, x, 24)))
  const midAng = Float32Array.from(sx, x => Math.atan(slope(midRidge, x, 24)))
  // The plane under a point: +1 on a spur's flank square to the light, -1 on one turned away,
  // fading down the face as the spurs soften into the foothills; and how near a spur's crest it is,
  // in phase.
  const farTilt = Float32Array.from(sx, x => lean(farRidge, x))
  const midTilt = Float32Array.from(sx, x => lean(midRidge, x))
  const facet = (i: number, down: number, back: boolean) => {
    const u = spurAt(back, sx[i], down, back ? farTilt[i] : midTilt[i])
    const strength = 1 - 0.5 * smooth(0, 300, down)
    return [facing(u) * side * strength * contrast, Math.min(frac(u), 1 - frac(u))] as const
  }
  const farTone = (i: number, y: number, lit: number) =>
    0.45 +
    Math.max(-1, Math.min(1, farLit[i] * 1.6)) * 0.12 +
    lit * 0.34 +
    (fbm(sx[i] * 0.009, y * 0.003, 53) - 0.5) * 0.22 +
    smooth(30, 360, y - far[i]) * 0.35
  const midTone = (i: number, y: number, lit: number) =>
    0.5 +
    Math.max(-1, Math.min(1, midLit[i] * 1.6)) * 0.12 +
    lit * 0.4 +
    (fbm(sx[i] * 0.012, y * 0.0035, 41) - 0.5) * 0.26 +
    smooth(40, 320, y - mid[i]) * 0.22
  // Snow lies where it is cold enough, reaches lower in the shade, and slides off the spur crests.
  const snowy = (x: number, y: number, down: number, lit: number, crest: number) => {
    const edge = snowline + (fbm(x * 0.02, 5.5, 71, 2) - 0.5) * 80 - lit * 30
    if (y >= edge && !glacier(x, y)) return 0
    return crest < 0.045 && down > 6 && down < 170 ? 2 : 1
  }
  const snowTone = (lit: number, y: number, top: number) =>
    0.62 + lit * 0.36 - smooth(0, 200, y - top) * 0.12
  // Below the snow the near range's grass takes the night's frost as the island's does.
  const rimed = rimeOver(pal.mid, pal.snow)
  // What the face shows at a point: its ink for strokes and the flat underpainting alike.
  const surface = (i: number, y: number, h: number, back: boolean) => {
    const top = back ? far[i] : mid[i]
    const down = y - top
    const [lit, crest] = facet(i, down, back)
    const cover = snowy(sx[i], y, down, lit, crest)
    if (cover === 2) return pick(pal.rock, 0.35 + lit * 0.35, h)
    if (cover === 1) return pick(pal.snow, snowTone(lit, y, top), h)
    if (back) return pick(pal.far, farTone(i, y, lit), h)
    // Read off the face's tone as on the island, less the haze that lightens it toward the foot,
    // where the cold air pools.
    const frosted =
      rimeAt(relief, midTone(i, y, lit) - 0.35 * smooth(40, 320, down)) > frac(h * 7.31)
    return pick(frosted ? rimed : pal.mid, midTone(i, y, lit), h)
  }

  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    for (let i = 0; i < bw; i++) {
      const k = (j * bw + i) * 4
      const t = BAYER[(j & 3) * 4 + (i & 3)]
      // Each crest is drawn as a contour, so where the mid ridge passes in front of the far one the
      // overlap shows as a line. Slopes that fall away toward the light catch gold on the crest.
      if (y >= mid[i]) {
        if (y - mid[i] < g.px) put(d, k, midLit[i] > 0.15 ? pal.gold : pal.contour[1])
        else put(d, k, surface(i, y, t, false))
      } else if (y >= far[i]) put(d, k, y - far[i] < g.px ? pal.contour[0] : surface(i, y, t, true))
    }
  }

  const y0 = sy[0]
  let ridge = far
  let ang = farAng
  let jit = 0
  const angle: Angle = (bx, by) => {
    const i = clampCol(g, bx)
    return ang[i] * (1 - 0.6 * smooth(0, 260, y0 + by * g.px - ridge[i])) + jit
  }
  const inFar: Keep = (i, j) => sy[j] >= far[i] + g.px && sy[j] < mid[i]
  const inMid: Keep = (i, j) => sy[j] >= mid[i] + g.px
  let top = Infinity
  for (const y of far) top = Math.min(top, y)
  seeds(g, 3, 0, [row(g, top), bh], 141, (bx, by, h, k) => {
    const i = clampCol(g, bx)
    const y = y0 + by * g.px
    if (y < far[i] + g.px * 1.5 || y > mid[i] - g.px) return
    jit = (k - 0.5) * 0.4
    const ink = surface(i, y, h, true)
    brush(d, g, bx, by, 4 + Math.floor(frac(k * 9.1) * 4), false, ink, angle, inFar)
  })
  ridge = mid
  ang = midAng
  seeds(g, 3, 0, [row(g, top), bh], 143, (bx, by, h, k) => {
    const i = clampCol(g, bx)
    const y = y0 + by * g.px
    if (y < mid[i] + g.px * 1.5 || y > 900) return
    jit = (k - 0.5) * 0.5
    const ink = surface(i, y, h, false)
    brush(d, g, bx, by, 4 + Math.floor(frac(k * 9.1) * 5), frac(k * 4.3) < 0.4, ink, angle, inMid)
  })
  for (const rock of ROCKS) paintRock(d, g, pal, rock, relief)
  paintCourse(d, g, courseInk(pal), snowline)
  // The trail: a pale tread with the bench's lip a pixel under it.
  const tread = mix(pal.mid[2], CREAM, relief.dark ? 0.12 : 0.5)
  const lip = mix(pal.mid[0], INK, 0.25)
  for (let k = 1; k < TRAIL.length; k++) {
    const [i0, j0, i1, j1] = [
      col(g, TRAIL[k - 1][0]),
      row(g, TRAIL[k - 1][1]),
      col(g, TRAIL[k][0]),
      row(g, TRAIL[k][1]),
    ]
    line(d, g, i0, j0 + 1, i1, j1 + 1, lip)
    line(d, g, i0, j0, i1, j1, tread)
  }
  pines.forEach((c, n) => paintCypress(d, g, c, pal.pine, 300 + n))
}

// People on the ranges, a pixel wide and three tall: two beside the cypress on the left ridge, one
// of them waving; someone with a lantern on the right ridge; a trail runner in an orange vest on
// the switchbacks below them; and another party's fire on the far range, which by day is only a
// thread of smoke.
const FOLK = [
  { x: 246, wave: false, lamp: false },
  { x: 254, wave: true, lamp: false },
  { x: 1318, wave: false, lamp: true },
]
const CAMP = 1046
// A trail cut into the right ridge, as turns in scene units: it comes out from behind the knoll,
// switchbacks up the face (the leftward legs short, since the crest falls away toward the valley),
// and scrambles out onto the crest by the pines. A trail runner works up it leaning into the grade,
// throws both arms up on top, and runs back down, all day; the triathlete's mountain legs share it.
// The timings are the climb, the summit, the descent and the look at the watch at the bottom, in
// seconds.
const TRAIL: Pt[] = [
  [1110, 760],
  [1290, 700],
  [1230, 660],
  [1330, 600],
  [1250, 565],
  [1360, 505],
  [1400, 404],
]
const TRAIL_LEN = TRAIL.slice(1).reduce(
  (n, [x, y], k) => n + Math.hypot(x - TRAIL[k][0], y - TRAIL[k][1]),
  0,
)
const TRAIL_TIME = { up: 30, summit: 3, down: 18, foot: 2 }

// Boulders and outcrops on both ranges, clear of the trail, the people and the lift.
const ROCKS: Rock[] = (() => {
  const r = mulberry(909)
  const rocks: Rock[] = []
  const nearTrail = (x: number, y: number) =>
    TRAIL.some(([tx, ty], k) => {
      if (k === 0) return false
      const [ax, ay] = TRAIL[k - 1]
      const len2 = (tx - ax) ** 2 + (ty - ay) ** 2
      const u = Math.max(0, Math.min(1, ((x - ax) * (tx - ax) + (y - ay) * (ty - ay)) / len2))
      return Math.hypot(x - ax - (tx - ax) * u, y - ay - (ty - ay) * u) < 22
    })
  for (let n = 0; n < 1200 && rocks.length < 34; n++) {
    const mid = r() < 0.62
    const x = -40 + r() * 1680
    const down = mid ? 12 + r() * 170 : 8 + r() * 80
    // Outcrops break out along the spur crests, where the rock is nearest the surface.
    const ridge = mid ? midRidge : farRidge
    const u = spurAt(!mid, x, down, lean(ridge, x))
    if (Math.min(frac(u), 1 - frac(u)) > 0.06) continue
    const big = r() < 0.3
    const w = (mid ? 28 + r() * 22 : 16 + r() * 12) * (big ? 1.6 : 1)
    const base = ridge(x) + down
    const peak = (r() - 0.5) * 0.9
    const rock = {
      x,
      base,
      w,
      h: w * (0.45 + r() * 0.2),
      peak,
      shoulder: peak + (peak < 0 ? 0.5 : -0.5) + (r() - 0.5) * 0.3,
      mid,
    }
    if (!mid && base > midRidge(x) - 4) continue
    if (nearTrail(x, base) || FOLK.some(f => Math.abs(f.x - x) < 20) || Math.abs(x - CAMP) < 24)
      continue
    if (!mid && x > 1300 && x < 1500) continue
    rocks.push(rock)
  }
  // Nearer outcrops overlap the ones above them. None sits on the downhill line on the left hill.
  return rocks
    .filter(rock => !rock.mid || !nearCourse(rock.x, rock.base, rock.w / 2 + 10))
    .sort((a, b) => a.base - b.base)
})()

// The point `s` units up the trail, and which way that stretch runs.
function onTrail(s: number) {
  for (let k = 1; k < TRAIL.length; k++) {
    const [ax, ay] = TRAIL[k - 1]
    const [bx, by] = TRAIL[k]
    const len = Math.hypot(bx - ax, by - ay)
    if (s > len && k < TRAIL.length - 1) {
      s -= len
      continue
    }
    const u = Math.min(1, s / len)
    return { x: ax + (bx - ax) * u, y: ay + (by - ay) * u, dir: Math.sign(bx - ax) }
  }
  return { x: TRAIL[0][0], y: TRAIL[0][1], dir: 1 }
}

function paintFolk(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  base: Uint8ClampedArray,
  clock: number,
  dark: boolean,
  heads: Voices['heads'],
  park: Park,
  hill: Hill,
) {
  d.set(base)
  const beat = Math.floor(clock * 5)
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  // First row of ground under a column, so feet land on the ridge rather than near it.
  const ground = (i: number, ridge: (x: number) => number) =>
    Math.ceil((ridge(g.sx[clampCol(g, i)]) - g.sy[0]) / g.px)
  const glow = (i: number, j: number, flick: number) => {
    for (let dj = -2; dj <= 2; dj++)
      for (let di = -2; di <= 2; di++) {
        const r = Math.hypot(di, dj)
        if (r > 0) plot(i + di, j + dj, pal.glow[1], (1 - r / 2.8) * (0.55 + 0.25 * flick))
      }
  }
  FOLK.forEach((f, n) => {
    const i = col(g, f.x)
    const top = ground(i, midRidge)
    heads[`folk${n}`] = [i, top - 3]
    if (f.lamp && dark) glow(i + 1, top - 2, hash(beat, 3, 41))
    for (let k = 1; k <= 3; k++) dot(d, g, i, top - k, pal.figure.K)
    // The two by the cypress throw both arms up when a rider lands a trick off the table.
    if (park.cheer > 0 && !f.lamp) {
      const up = Math.floor(clock / 0.2) % 2
      dot(d, g, i - 1, top - 3 - up, pal.figure.K)
      dot(d, g, i + 1, top - 4 + up, pal.figure.K)
    } else if (f.wave) dot(d, g, i + 1, top - (Math.floor(clock / 0.45) % 2 ? 4 : 3), pal.figure.K)
    if (f.lamp) dot(d, g, i + 1, top - 2, hash(beat, 3, 41) < 0.3 ? FLAME[2] : FLAME[1])
  })

  // A lap starts on the summit, so a still frame (reduced motion) finds them there, arms up.
  const { up, summit, down } = TRAIL_TIME
  let lap = (clock + up) % (up + summit + down + TRAIL_TIME.foot)
  let s = 0
  let pose: 'up' | 'summit' | 'down' | 'foot' = 'foot'
  if (lap < up) {
    s = (TRAIL_LEN * lap) / up
    pose = 'up'
  } else if ((lap -= up) < summit) {
    s = TRAIL_LEN
    pose = 'summit'
  } else if ((lap -= summit) < down) {
    s = TRAIL_LEN * (1 - lap / down)
    pose = 'down'
  }
  const at = onTrail(s)
  const face = pose === 'down' ? -at.dir : at.dir
  const ri = col(g, at.x)
  const rt = row(g, at.y)
  // Head, vest and the pack behind it; the head leans a pixel into the climb.
  heads.runner = [ri + (pose === 'up' ? face : 0), rt - 3]
  dot(d, g, ri + (pose === 'up' ? face : 0), rt - 3, pal.figure.S)
  dot(d, g, ri, rt - 2, FLAME[2])
  dot(d, g, ri - face, rt - 2, pal.crew[1])
  const stride =
    pose === 'up' ? Math.floor(clock * 3) % 2 : pose === 'down' ? Math.floor(clock * 6) % 2 : 0
  if (stride) {
    dot(d, g, ri - 1, rt - 1, pal.figure.K)
    dot(d, g, ri + 1, rt - 1, pal.figure.K)
  } else dot(d, g, ri, rt - 1, pal.figure.K)
  if (pose === 'summit') {
    dot(d, g, ri - 1, rt - 4, pal.figure.S)
    dot(d, g, ri + 1, rt - 4, pal.figure.S)
  }
  if (pose === 'foot') dot(d, g, ri + face, rt - 2, pal.figure.S)

  const ci = col(g, CAMP)
  const top = ground(ci, farRidge)
  const flick = hash(beat, 7, 43)
  if (dark) glow(ci, top - 1, flick)
  dot(d, g, ci, top - 1, flick < 0.5 ? FLAME[2] : FLAME[1])
  // Its smoke leans off with the wind and thins as it climbs.
  for (let s = 1; s <= 12; s++)
    plot(
      ci + Math.round(s * 0.35 + Math.sin(s * 0.7 - clock * 1.3) * 0.6),
      top - 1 - s,
      pal.smoke[dark ? 2 : 1],
      (1 - s / 13) * (dark ? 0.5 : 0.8),
    )
  // The ski hill shows only where the far range does, under its crest and above the mid ridge.
  const onRange = (i: number, j: number) => {
    if (i < 0 || j < 0 || i >= g.bw || j >= g.bh) return false
    const y = g.sy[j]
    return y > farRidge(g.sx[i]) + g.px && y < midRidge(g.sx[i]) - g.px
  }
  paintHill(d, g, skiInk(pal), hill, onRange, dark, heads)
  paintPark(d, g, riderInk(pal), park, dark, heads)
}

// The downhill line in the ranges' colours: dirt from the bark, timber, and the lift's steel.
function courseInk(pal: Palette): CourseInk {
  return {
    tread: mix(pal.bark[2], pal.ochre, 0.3),
    rut: pal.bark[1],
    berm: mix(pal.ochre, pal.bark[2], 0.5),
    packed: pal.snow[1],
    path: mix(pal.mid[1], pal.bark[2], 0.45),
    wood: pal.bark[2],
    post: pal.bark[0],
    steel: mix(pal.rock[0], INK, 0.4),
    flag: [CREAM, INK],
  }
}

// Its riders in the crew's colours under bright helmets, on frames of any of them; the gate's
// beacon counts down in poppy and goes in sage.
function riderInk(pal: Palette): RiderInk {
  return {
    skin: pal.figure.S,
    shorts: pal.figure.D,
    tyre: pal.figure.K,
    dust: mix(pal.bark[2], CREAM, 0.65),
    lamp: CREAM,
    glow: pal.glow[1],
    steel: mix(pal.rock[0], INK, 0.4),
    flash: CREAM,
    beacon: [pal.poppy, LUME],
    jerseys: pal.crew,
    helmets: [GOLDEN, pal.poppy, CREAM, pal.cornflower, pal.crew[1], pal.crew[3]],
    frames: [pal.crew[2], pal.crew[3], pal.crew[0], CREAM, pal.crew[5], pal.rock[1]],
  }
}

// The ski hill in the ranges' colours: coats from the crew's under bright helmets, skis and boards
// of any colour, the instructor in a poppy bib, the slalom's gates poppy and cornflower, and the
// groomer golden with a flame-coloured beacon.
function skiInk(pal: Palette): SkiInk {
  return {
    snow: pal.snow,
    skin: pal.figure.S,
    pants: pal.figure.D,
    boot: pal.figure.K,
    steel: mix(pal.rock[0], INK, 0.4),
    cable: mix(pal.far[0], INK, 0.4),
    rock: [pal.rock[0], pal.rock[1]],
    coats: pal.crew,
    helmets: [GOLDEN, pal.poppy, CREAM, pal.cornflower, pal.crew[1], pal.crew[3]],
    gear: [pal.figure.K, pal.crew[2], mix(pal.rock[0], INK, 0.4), CREAM],
    boards: [GOLDEN, pal.poppy, pal.cornflower, pal.crew[1], CREAM],
    school: pal.poppy,
    gates: [pal.poppy, pal.cornflower],
    cat: GOLDEN,
    cab: mix(pal.far[0], INK, 0.4),
    lamp: CREAM,
    glow: pal.glow[1],
    flood: mix(CREAM, GOLDEN, 0.4),
    beacon: FLAME[2],
  }
}

const HILL_CYPRESSES = [
  { x: 420, h: 250, w: 46 },
  { x: 1195, h: 215, w: 40 },
]

// The island's crest line, how each column of it faces the light (already signed and scaled for
// where the light is and how hard it falls), the flow of its strokes, how its spurs lean, the
// outcrops on them, and the season's flowers and leaves.
type Meadow = {
  top: Float32Array
  lit: Float32Array
  ang: Float32Array
  tilt: Float32Array
  rocks: Rock[]
  crest: number
  relief: Relief
  bloom: number
  autumn: number
  snow: number
}

// The spur under a point of the island, as on the ranges but broader, being nearer: its phase, and
// how the plane there is lit, +1 square to the light and -1 turned away, softening toward the water.
function islandSpur(tilt: number, x: number, down: number) {
  return spur(x - tilt * 110 * Math.tanh(down / 100), down, 41, 0.0065)
}
// Under cloud the light comes from the whole sky, and the gullies, open to less of it than the
// spurs, stay the darker for it.
function islandFacet(m: Meadow, i: number, x: number, y: number) {
  const down = y - m.top[i]
  const u = islandSpur(m.tilt[i], x, down)
  const { side, contrast } = m.relief
  const open = 1 - 4 * Math.min(frac(u), 1 - frac(u))
  return (
    (facing(u) * side * contrast + open * 0.5 * (1 - contrast)) * (1 - 0.6 * smooth(0, 220, down))
  )
}

// Outcrops break out along the spur crests, clear of the gate and of everyone's footing: the
// tablet, the camp and the strand, the cypresses, the cat and the fallen stone.
function islandRocks(water: number): Rock[] {
  const top = (x: number) => hillTop(x, water)
  const clear = (x: number, y: number) =>
    Math.hypot(x - GATE.x, y - GATE.y) > 300 &&
    !(Math.abs(x - 805) < 160 && Math.abs(y - 786) < 60) &&
    y < 812 &&
    !(x > 480 && x < 720 && y > 780) &&
    Math.abs(x - 420) > 45 &&
    Math.abs(x - 1195) > 45 &&
    Math.abs(x - 1250) > 110 &&
    Math.abs(x - 1136) > 40 &&
    !(Math.abs(x - 366) < 80 && Math.abs(y - 772) < 50) &&
    !(Math.abs(x - 548) < 50 && Math.abs(y - 740) < 50)
  const rocks: Rock[] = []
  for (let x = 170; x <= 1440; x += 23)
    for (let down = 30; down <= 200; down += 17) {
      const y = top(x) + down
      const u = islandSpur(lean(top, x), x, down)
      if (Math.min(frac(u), 1 - frac(u)) > 0.05 || !clear(x, y)) continue
      if (rocks.some(r => Math.hypot(r.x - x, r.base - y) < 150)) continue
      const w = 32 + hash(x, down, 5) * 26
      const peak = (hash(x, down, 6) - 0.5) * 0.9
      rocks.push({
        x,
        base: y,
        w,
        h: w * (0.5 + hash(x, down, 7) * 0.2),
        peak,
        shoulder: peak + (peak < 0 ? 0.5 : -0.5),
        mid: true,
      })
    }
  return rocks.sort((a, b) => a.base - b.base)
}

function meadow(g: Grid, water: number, relief: Relief, s: Season): Meadow {
  const top = Float32Array.from(g.sx, x => hillTop(x, water))
  let crest = Infinity
  for (const y of top) crest = Math.min(crest, y)
  const k = relief.side * relief.contrast
  return {
    top,
    lit: Float32Array.from(g.sx, x => ((hillTop(x + 8, water) - hillTop(x - 8, water)) / 16) * k),
    ang: Float32Array.from(g.sx, x =>
      Math.atan((hillTop(x + 24, water) - hillTop(x - 24, water)) / 48),
    ),
    tilt: Float32Array.from(g.sx, x => lean(xx => hillTop(xx, water), x)),
    rocks: islandRocks(water),
    crest,
    relief,
    bloom: s.bloom,
    autumn: s.autumn,
    snow: s.ground,
  }
}

// How much hoarfrost is left at a point of the island, read off its tone, which already carries the
// spur planes, the broad mottle of the turf and the ring's shadow: the sun clears the brightest
// ground first and the shade last. The lake gives up its warmth all night, so the bank stays green.
function rimed(r: Relief, tone: number, y: number, water: number) {
  return rimeAt(r, tone) * (1 - smooth(water - 44, water - 12, y))
}

function hillTone(m: Meadow, i: number, x: number, y: number, water: number) {
  // Weighted as on the ranges: the spur planes carry the form, the broad mottle stays under them.
  let t = 0.5 + Math.max(-1, Math.min(1, m.lit[i] * 2.2)) * 0.14 + islandFacet(m, i, x, y) * 0.44
  t += (fbm(x * 0.012, y * 0.004, 61) - 0.5) * 0.26 + (fbm(x * 0.018, y * 0.07, 62) - 0.5) * 0.14
  // The turf round the foot of the ring sits in its shadow, cast down and away from the light.
  const rim = Math.hypot(x - GATE.x + 10 * m.relief.side, y - GATE.y - 6) - GATE.r - GATE.ring
  if (y > GATE.y) t -= 0.4 * (1 - smooth(0, 30, rim))
  return t - smooth(water - 16, water, y) * 0.2
}

function paintHillBase(d: Uint8ClampedArray, g: Grid, pal: Palette, water: number, m: Meadow) {
  const { bw, bh, sx, sy } = g
  const white = rimeOver(pal.hill, pal.snow)
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    if (y > water + g.px) break
    for (let i = 0; i < bw; i++) {
      if (y < m.top[i]) continue
      const k = (j * bw + i) * 4
      if (y - m.top[i] < g.px) put(d, k, m.lit[i] > 0.08 ? pal.gold : pal.contour[2])
      else {
        const t = hillTone(m, i, sx[i], y, water)
        put(d, k, ramp(dither(rimed(m.relief, t, y, water), i, j) ? white : pal.hill, t, i, j))
      }
    }
  }
}

// The field in short dashes that follow the slope and flatten toward the bank; a travelling
// wave leans them as the wind crosses, with the odd poppy, cornflower and ochre dab, and the fallen
// leaves in autumn. The outcrops stand up out of it.
function paintField(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  m: Meadow,
  base: Uint8ClampedArray,
  clock: number,
  gust: number,
) {
  d.set(base)
  const x0 = g.sx[0]
  const y0 = g.sy[0]
  const white = rimeOver(pal.hill, pal.snow)
  const sway = 0.45 + 0.55 * Math.min(1, gust)
  let jit = 0
  const angle: Angle = (bx, by) => {
    const i = clampCol(g, bx)
    const x = x0 + bx * g.px
    const y = y0 + by * g.px
    const wave = 0.36 * Math.sin(0.012 * x - 2.1 * clock + 0.035 * y) * sway
    const contour = m.ang[i] * (1 - 0.5 * smooth(0, 120, y - m.top[i]))
    return (contour + wave + jit) * (1 - smooth(water - 34, water - 8, y))
  }
  seeds(g, 3, 0, [row(g, m.crest), row(g, water)], 151, (bx, by, h, k) => {
    const i = clampCol(g, bx)
    const x = x0 + bx * g.px
    const y = y0 + by * g.px
    if (y < m.top[i] + g.px || y > water - g.px * 0.5) return
    // Behind the ring: never seen.
    if (Math.hypot(x - GATE.x, y - GATE.y) < GATE.r + GATE.ring - 10) return
    const t = hillTone(m, i, x, y, water)
    // Flowers come and go with the season; the grass goes to seed in the autumn, and the leaves
    // come down on it. Under snow the wind scours the spur crests back to the dead grass, as it
    // bares the rock along the ranges' spurs.
    const roll = frac(k * 5.31)
    const u = m.snow > 0 ? islandSpur(m.tilt[i], x, y - m.top[i]) : 0.5
    const scoured = Math.min(frac(u), 1 - frac(u)) < 0.07 * (1.3 - m.snow) + 0.03 * frac(k * 2.9)
    // Rimed blades, and among them the odd crystal catching the light where the sun is at the frost.
    const rime = rimed(m.relief, t, y, water)
    const frosted = frac(k * 4.13) < rime
    const ink = frosted
      ? frac(k * 6.07) < 0.12 * smooth(0.2, 0.6, m.relief.thaw) * (1 - m.relief.thaw)
        ? pal.snow[3]
        : pick(white, t + 0.12, h)
      : scoured
        ? pick(pal.dun, t - 0.15, h)
        : t > 0.42 && roll < 0.016 * m.bloom
          ? pal.poppy
          : t > 0.42 && roll < 0.03 * m.bloom
            ? pal.cornflower
            : t > 0.5 && roll < 0.03 * m.bloom + 0.03 + 0.05 * m.autumn
              ? pal.ochre
              : roll > 1 - 0.08 * m.autumn
                ? pal.autumn[1 + Math.floor(frac(k * 7.7) * 3)]
                : pick(pal.hill, t, h)
    jit = (frac(k * 9.73) - 0.5) * 0.7
    brush(d, g, bx, by, 3 + Math.floor(frac(k * 3.37) * 3), frac(k * 11.3) < 0.65, ink, angle)
  })
  for (const rock of m.rocks) paintRock(d, g, pal, rock, m.relief)
  HILL_CYPRESSES.forEach((c, n) =>
    paintCypress(
      d,
      g,
      { ...c, base: hillTop(c.x, water) + 6 },
      pal.leaf.slice(0, 3),
      500 + n,
      Math.sin(clock * 1.1 + n * 2) * 0.12 * sway,
    ),
  )
}

// Staffage after the woman in blue crossing the field at Clichy: cap, face, dress, stride.
type Walker = {
  x: number
  dir: number
  pause: number
  next: number
  stride: number
  walking: boolean
}

const FIGURE = ['.CC.', '.CS.', '.BB.', 'DBBB', 'DBB.', 'DBBB', 'DDBB', 'DBBB']
const LEGS = ['.K.K', '..K.', '.KK.']
const HOME = 800

// Wanders the strand right of the camp, stops now and then to look at the gate, and walks to it
// while the portal is open.
function stepWalker(w: Walker, dt: number, drawn: boolean) {
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

// Glitter on the water: horizontal dashes wherever a facet tilts the sun (or the open portal)
// toward us, widening with distance below the horizon.
function paintGlitter(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  heat: number,
  clock: number,
  body: Disc | null,
) {
  // The moon lays a path only in proportion to how much of it is lit.
  const lit = body ? (1 - Math.abs(body.phase - 0.5) * 2) * (1 - body.veil) : 0
  const { bw, bh, sx, sy } = g
  for (let j = Math.max(0, row(g, water) + 1); j < bh; j += 2) {
    const depth = sy[j] - water
    const ws = 14 + depth * 0.9
    const wp = 20 + depth * 0.5
    for (let i = (j >> 1) % 3; i < bw; i += 3) {
      const ps = body ? 0.42 * lit * Math.exp(-(((sx[i] - body.x) / ws) ** 2)) : 0
      const pp = (0.05 + 0.45 * heat) * Math.exp(-(((sx[i] - HOME) / wp) ** 2))
      if (ps + pp < 0.01) continue
      // Each facet rerolls on its own beat, so the glitter shimmers instead of strobing.
      const beat = Math.floor(clock * 3 + hash(i, j, 7) * 3)
      const roll = hash(i, j, beat)
      if (roll >= ps + pp) continue
      const ink =
        roll < ps ? pal.glint[roll < ps * 0.4 ? 0 : 1] : roll < ps + pp * 0.5 ? ROSE : GOLDEN
      const len = 2 + Math.floor(hash(i, j, beat + 1) * 3)
      for (let s = 0; s < len; s++) dot(d, g, i + s, j, ink)
    }
  }
}

type Ripple = { x: number; y: number; w: number; bright: boolean; half: number; phase: number }

function makeRipples(water: number): Ripple[] {
  const r = mulberry(99)
  return Array.from({ length: 52 }, () => {
    const y = water + 12 + r() ** 0.8 * 110
    const x = r() * 1600
    const w = 20 + r() * (30 + (y - water) * 0.6)
    return { x, y, w, bright: r() < 0.35, half: 0.8 + r() * 1.2, phase: r() * 4 }
  })
}

// The river: a plate that darkens toward the viewer, with the scene above folded into it. Near the
// bank the fold keeps the old squash (0.62); further out it steepens, so the bottom of the view
// reaches up the gate however little water the viewport shows. The fold is cut into slices that
// shear on a stop-motion beat, and wherever the reflection crosses from one mass to another (a
// ridge into sky, the ruin into the hill) it draws a contour. Then the campfire's light in broken
// streaks, ripple dashes that hop back and forth, and the bank as one row of ink.
const LIFT = 480
// Reflected layers, front to back, and the ids the contours compare: the ranges split into the mid
// ridge and the far one, and NONE for open sky.
const FIGURES = 0
const WELL = 3
const RANGES = 4
const FAR = 5
const NONE = 6
// The figures and the rift carry their own keylines, and the rift's dithered bleed would speckle.
const contoured = (a: number, b: number) =>
  a !== b && a !== FIGURES && b !== FIGURES && a !== WELL && b !== WELL && (a > 1 || b > 1)

function paintWater(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  ink: RGB,
  layers: (Uint8ClampedArray | null)[],
  ripples: Ripple[],
  clock: number,
  dark: boolean,
) {
  const { bw, bh, sx, sy } = g
  const shore = row(g, water)
  const slice = Math.floor(clock / 0.3)
  const reach = Math.max(8 * g.px, g.bottom - water)
  const knee = reach * 0.25
  const bend = Math.max(0, (LIFT - reach / 0.62) / (reach - knee) ** 2)
  const contour = mix(pal.water[0], ink, dark ? 0.5 : 0.45)
  const mid = Float32Array.from(sx, midRidge)
  let above = new Uint8Array(bw)
  let here = new Uint8Array(bw)
  for (let j = Math.max(0, shore + 1); j < bh; j++) {
    const depth = sy[j] - water
    const ys = water - depth / 0.62 - bend * Math.max(0, depth - knee) ** 2
    const src = row(g, ys)
    // Three flat bands, paler with distance from the bank.
    const pale = [0.35, 0.5, 0.65][Math.min(2, Math.floor((depth / reach) * 3))]
    const shear = Math.round((noise(Math.floor(j / 3) * 0.9, slice * 1.7, 91) - 0.5) * 6)
    for (let i = 0; i < bw; i++) {
      const k = (j * bw + i) * 4
      const plate = ramp(pal.water, 1 - depth / 180, i, j)
      put(d, k, plate)
      const si = i - shear
      let id = NONE
      if (src >= 0 && si >= 0 && si < bw) {
        const q = (src * bw + si) * 4
        for (let n = 0; n < layers.length; n++) {
          const s = layers[n]
          if (!s || s[q + 3] !== 255) continue
          id = n === RANGES && ys < mid[si] ? FAR : n
          put(d, k, mix([s[q], s[q + 1], s[q + 2]], plate, pale))
          break
        }
      }
      here[i] = id
      const up = j > shore + 1 ? above[i] : id
      const left = i > 0 ? here[i - 1] : id
      if (contoured(id, up) || contoured(id, left)) put(d, k, contour)
    }
    ;[above, here] = [here, above]
  }

  const fc = col(g, FIRE.x)
  const beat = Math.floor(clock * 6)
  for (let j = Math.max(0, shore + 1); j < bh; j++) {
    const depth = sy[j] - water
    const p = (1 - depth / 110) * (dark ? 0.85 : 0.45)
    if (p <= 0) break
    if (hash(j, beat, 29) > p) continue
    const len = 1 + Math.floor(hash(j, beat, 31) * 3)
    const off = Math.round((hash(j, beat, 37) - 0.5) * 4) - (len >> 1)
    const c = depth < 20 && hash(j, beat, 39) < 0.4 ? FLAME[0] : depth < 45 ? FLAME[1] : FLAME[2]
    for (let s = 0; s < len; s++) dot(d, g, fc + off + s, j, c)
  }

  const paper: RGB = dark ? [16, 15, 15] : [255, 252, 240]
  for (const r of ripples) {
    const j = row(g, r.y)
    if (j <= shore || j >= bh) continue
    const hop = Math.floor((clock + r.phase) / r.half) % 2 ? 10 : 0
    const c = r.bright ? paper : ink
    const a = r.bright ? 0.75 : 0.35
    const i0 = col(g, r.x + hop)
    for (let s = Math.max(1, Math.round(r.w / g.px)) - 1; s >= 0; s--) {
      const i = i0 + s
      if (i < 0 || i >= bw) continue
      const k = (j * bw + i) * 4
      for (let n = 0; n < 3; n++) d[k + n] += (c[n] - d[k + n]) * a
    }
  }

  if (shore >= 0 && shore < bh) for (let i = 0; i < bw; i++) put(d, (shore * bw + i) * 4, ink)
}

type Splash = { x: number; y: number; size: number; age: number }

// Where a letter lands: two flat rings that widen in five steps, darkening the water under them
// less at every step until they dither away.
function paintSplashes(d: Uint8ClampedArray, g: Grid, splashes: Splash[], ink: RGB) {
  const { bw, bh } = g
  for (const s of splashes)
    for (let n = 0; n < 2; n++) {
      const t = (s.age - n * 0.28) / 1.2
      if (t < 0 || t >= 1) continue
      const f = Math.floor(t * 5) / 5
      const rx = (s.size * (0.15 + 1.15 * f)) / g.px
      const ry = Math.max(1, rx * 0.2)
      const cx = (s.x - g.sx[0]) / g.px
      const cy = (s.y - g.sy[0]) / g.px
      const seen = new Set<number>()
      const steps = Math.ceil(2 * Math.PI * rx)
      for (let k = 0; k < steps; k++) {
        const a = (k / steps) * 2 * Math.PI
        const i = Math.round(cx + Math.cos(a) * rx)
        const j = Math.round(cy + Math.sin(a) * ry)
        const p = j * bw + i
        if (i < 0 || j < 0 || i >= bw || j >= bh || seen.has(p)) continue
        seen.add(p)
        if (1 - f <= BAYER[(j & 3) * 4 + (i & 3)]) continue
        for (let c = 0; c < 3; c++) d[p * 4 + c] += (ink[c] - d[p * 4 + c]) * 0.5
      }
    }
}

// Rain in streaks slanting with the wind, a fresh set every tick, the way stop-motion rain is drawn;
// snow in flakes that keep their place from tick to tick and wander as they fall, the nearest ones
// two pixels across.
function paintFall(d: Uint8ClampedArray, g: Grid, sky: Sky, clock: number, ink: RGB, flake: RGB) {
  if (sky.fall === 'none') return
  const { bw, bh } = g
  if (sky.fall === 'snow') {
    const n = Math.round(70 + 330 * sky.heavy)
    for (let k = 0; k < n; k++) {
      const speed = 6 + 10 * hash(k, 1, 61)
      const y = ((hash(k, 2, 61) * (bh + 8) + clock * speed) % (bh + 8)) - 4
      const x = hash(k, 3, 61) * bw + clock * sky.wind * 8 + Math.sin(clock * 0.8 + k) * 2
      const i = Math.round(((x % bw) + bw) % bw)
      const j = Math.round(y)
      dot(d, g, i, j, flake)
      if (hash(k, 5, 61) > 0.86) {
        dot(d, g, i + 1, j, flake)
        dot(d, g, i, j + 1, flake)
        dot(d, g, i + 1, j + 1, flake)
      }
    }
    return
  }
  const beat = Math.floor(clock * 10)
  const n = Math.round((sky.fall === 'drizzle' ? 40 : 60) + 300 * sky.heavy)
  const len = sky.fall === 'drizzle' ? 2 : 3 + Math.round(sky.heavy * 2)
  const slant = 0.15 + 0.6 * sky.wind
  for (let k = 0; k < n; k++) {
    const i0 = hash(k, beat, 67) * (bw + len) - len * slant
    const j0 = Math.floor(hash(k, beat, 71) * bh)
    for (let s = 0; s < len; s++) {
      const i = Math.round(i0 + s * slant)
      if (dither(0.8, i, j0 + s)) dot(d, g, i, j0 + s, ink)
    }
  }
}

// Where rain meets the water: a ring seen edge-on, a pale pixel either side of the drop, gone by
// the next tick.
function paintRings(
  d: Uint8ClampedArray,
  g: Grid,
  water: number,
  sky: Sky,
  clock: number,
  ink: RGB,
) {
  if (sky.fall !== 'rain' && sky.fall !== 'drizzle') return
  const { bw, bh } = g
  const shore = row(g, water) + 2
  if (shore >= bh) return
  const beat = Math.floor(clock * 10)
  const n = Math.round(8 + 40 * sky.heavy)
  for (let k = 0; k < n; k++) {
    const i = Math.floor(hash(k, beat, 73) * bw)
    const j = shore + Math.floor(hash(k, beat, 79) * (bh - shore))
    const w = 1 + Math.floor(hash(k, beat, 83) * 2)
    dot(d, g, i - w, j, ink)
    dot(d, g, i + w, j, ink)
  }
}

// How the little people are lit: by day from the sun's side, at night by the campfire when they
// are near it and otherwise by the open gate.
type Rim = { side: number; color: RGB; contrast: number; night: boolean }

// Rim light and form shadow a pixel deep: an edge facing the light takes its colour, the edge
// away from it darkens, so a figure four pixels wide reads as round. Pixel-wide limbs and inked
// pixels keep their colour.
function relight(d: Uint8ClampedArray, g: Grid, rim: Rim, heat: number) {
  const { bw, bh, sx } = g
  const open = (p: number) => d[p * 4 + 3] !== 255
  for (let j = 1; j < bh; j++)
    for (let i = 1; i < bw - 1; i++) {
      const p = j * bw + i
      const k = p * 4
      if (open(p) || d[k] + d[k + 1] + d[k + 2] < 90) continue
      let { side, color } = rim
      let strength = 0.4 * rim.contrast
      if (rim.night) {
        const x = sx[i]
        const fire = Math.abs(x - FIRE.x) < 170
        side = Math.sign((fire ? FIRE.x : GATE.x) - x) || 1
        color = fire ? FLAME[1] : ROSE
        strength = fire ? 0.45 : 0.15 + 0.3 * heat
      }
      const toward = open(p + side)
      const away = open(p - side)
      if (toward && away) continue
      const c: RGB = [d[k], d[k + 1], d[k + 2]]
      if (toward || (open(p - bw) && !away)) put(d, k, mix(c, color, strength))
      else if (away) put(d, k, mix(c, INK, 0.28 * rim.contrast))
    }
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
const FIRE = { x: 628, y: 848 }
const TENT_X = 556
const TENT = ['....S....', '...SSL...', '..SSSLL..', '.SSSLKLL.', 'SSSSLKKLL']
const SITTER = ['.CC.', '.CS.', 'BBB.', 'BBBS', 'DDDD']
const CAMPERS = [
  { x: 600, flip: false },
  { x: 658, flip: true },
]
const FLAME_ROWS = [2, 4, 7, 5, 2]
// The camp's reach along the strand at a given pixel size, from the tent's left edge to the right
// camper's: the triathlon keeps its transition out of it and rides round behind it.
const campSpan = (px: number) =>
  [
    TENT_X - ((TENT[0].length >> 1) + 1) * px,
    CAMPERS[1].x + (SITTER[0].length - (SITTER[0].length >> 1)) * px,
  ] as const

// Firelight on the ground: an ellipse that reaches further up the bank than down it, stepped
// through three warm stops, and breathing with the flames. By day it barely leaves the hearth.
function paintPool(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  clock: number,
  dark: boolean,
) {
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

function paintCamp(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  w: Walker,
  clock: number,
  dark: boolean,
) {
  const foot = row(g, FIRE.y)
  const fc = col(g, FIRE.x)
  const warmth = dark ? 0.5 : 0.2
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
  for (let o = -2; o <= 2; o++) {
    const h = Math.max(1, Math.round(FLAME_ROWS[o + 2] * (0.55 + 0.6 * hash(o, beat, 5))))
    for (let r = 0; r < h; r++) {
      const t = (r + 0.5) / h
      const c =
        t < 0.34 && Math.abs(o) < 2 ? FLAME[0] : t < 0.6 ? FLAME[1] : t < 0.85 ? FLAME[2] : FLAME[3]
      dot(d, g, fc + o, foot - 1 - r, c)
    }
  }
  // Now and then a tongue tears off the tip.
  if (hash(beat, 1, 7) < 0.35)
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
  const toast = frac(clock / 14)
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

type Puff = {
  x: number
  y: number
  vx: number
  vy: number
  age: number
  life: number
  seed: number
}
type Hearth = { smoke: Puff[]; embers: Puff[]; spawn: number; spark: number }

// Smoke rises off the fire and rides the same air as the sky strokes, so it leans with the wind,
// curls through the cursor's eddies, and spirals into the gate while it is open. Each puff also
// keeps a sideways drift of its own, so the column fans out as it climbs. Embers shoot up, slow,
// and go out.
function stepHearth(h: Hearth, air: Air, dt: number) {
  h.spawn -= dt
  while (h.spawn <= 0) {
    h.spawn += 0.08
    h.smoke.push({
      x: FIRE.x + (Math.random() - 0.5) * 6,
      y: FIRE.y - 26,
      vx: 0,
      vy: -20,
      age: 0,
      life: 5 + Math.random() * 2.5,
      seed: Math.random(),
    })
  }
  h.spark -= dt
  if (h.spark <= 0) {
    h.spark = 0.15 + Math.random() * 0.35
    h.embers.push({
      x: FIRE.x + (Math.random() - 0.5) * 10,
      y: FIRE.y - 20,
      vx: (Math.random() - 0.5) * 24,
      vy: -50 - Math.random() * 40,
      age: 0,
      life: 0.7 + Math.random() * 0.9,
      seed: Math.random(),
    })
  }
  const ease = Math.min(1, dt * 1.5)
  for (const p of h.smoke) {
    const a = airAngle(air, p.x, p.y)
    const drift = 10 + 30 * air.pull
    const rise = 24 * (1 - (0.5 * p.age) / p.life)
    p.vx += (Math.cos(a) * drift + (p.seed - 0.5) * 18 - p.vx) * ease
    p.vy += (Math.sin(a) * drift - rise - p.vy) * ease
    p.x += p.vx * dt
    p.y += p.vy * dt
    p.age += dt
  }
  for (const p of h.embers) {
    p.vy += 30 * dt
    p.vx += (Math.random() - 0.5) * 80 * dt
    p.x += p.vx * dt
    p.y += p.vy * dt
    p.age += dt
  }
  h.smoke = h.smoke.filter(p => p.age < p.life)
  h.embers = h.embers.filter(p => p.age < p.life)
}

// Smoke as short strokes along its own motion that lengthen, widen, pale and thin as they age.
function paintHearth(d: Uint8ClampedArray, g: Grid, pal: Palette, h: Hearth, clock: number) {
  const beat = Math.floor(clock * 12)
  for (const p of h.embers) {
    if (hash(Math.floor(p.seed * 1e6), beat, 3) < 0.25) continue
    const t = p.age / p.life
    dot(d, g, col(g, p.x), row(g, p.y), t < 0.35 ? FLAME[1] : t < 0.75 ? FLAME[2] : FLAME[3])
  }
  for (const p of h.smoke) {
    const t = p.age / p.life
    const alpha = Math.min(1, p.age / 0.4) * (1 - t) * 0.9
    const c = pick(pal.smoke, 0.15 + t * 0.8, p.seed)
    const len = 1 + Math.floor(t * 4.5)
    const v = Math.hypot(p.vx, p.vy) || 1
    const ux = p.vx / v
    const uy = p.vy / v
    const bx = (p.x - g.sx[0]) / g.px
    const by = (p.y - g.sy[0]) / g.px
    for (let w = t > 0.3 ? 1 : 0; w >= 0; w--)
      for (let s = 0; s < len; s++) {
        const i = Math.round(bx + ux * (s - len / 2) - uy * w)
        const j = Math.round(by + uy * (s - len / 2) + ux * w)
        if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
      }
  }
}

// Stone tone at a point in the ring's upright frame: 0.5 is the flat plate, each 0.25 is one dither
// stop. The sun sits up and to the right. Every voussoir is chamfered, lit on its sunward edges and
// shaded on the rest, and the soffit under the ring is in deep shade.
function stoneTone(x: number, y: number, side: number) {
  const dx = x - GATE.x
  const dy = y - GATE.y
  const r = Math.hypot(dx, dy)
  if (r < GATE.r) return 0.08 + 0.3 * (dx / GATE.r) ** 2
  const a = Math.atan2(dy, dx)
  const h = (Math.round((a * 6) / Math.PI) + 15) % 12
  const out = stoneOuter(h)
  if (r <= out + 2) {
    const u = (r - GATE.r) / (out - GATE.r)
    const v = (Math.atan2(Math.sin(a - hourAngle(h)), Math.cos(a - hourAngle(h))) * 12) / Math.PI
    const eu = 8 / (out - GATE.r)
    const ev = (8 * 12) / (Math.PI * r)
    const cu = u > 1 - eu ? 1 : u < eu ? -1 : 0
    const cv = v > 1 - ev ? 1 : v < ev - 1 ? -1 : 0
    const nx = Math.cos(a) * cu - Math.sin(a) * cv
    const ny = Math.sin(a) * cu + Math.cos(a) * cv
    return 0.5 + 0.45 * (nx * 0.6 * side - ny * 0.8)
  }
  return 0.5
}

const probe = (() => {
  let ctx: CanvasRenderingContext2D | null = null
  return (css: string): RGB => {
    ctx ??= document.createElement('canvas').getContext('2d', { willReadFrequently: true })!
    ctx.clearRect(0, 0, 1, 1)
    ctx.fillStyle = css
    ctx.fillRect(0, 0, 1, 1)
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data
    return [r, g, b]
  }
})()

const MATERIALS = ['nf-fill-stone', 'nf-fill-rose', 'nf-fill-sage']
// Stops 0..4 of the tone dither map onto deep, shade, plate, plate, light.
const PLATE = [0, 1, 2, 2, 3]

type Print = { ink: RGB; stone: RGB[]; rose: RGB[]; sage: RGB[] }
type Masonry = { print: Print; spill: number[]; bands: number[] }

// A point at least every scene unit along an outline, in the element's own coordinates, one list
// per subpath. The ruin is drawn with lines and circular arcs only, so the path data is walked
// directly: getPointAtLength re-measures every arc from the start of the path on each call, which
// cost 670 ms per load for the ring and the soffit.
function trace(d: string) {
  const tok = d.match(/[a-zA-Z]|-?(?:\d+\.?\d*|\.\d+)/g) ?? []
  const chains: number[][] = []
  let pts: number[] = []
  let k = 0
  let cmd = ''
  let x = 0
  let y = 0
  let x0 = 0
  let y0 = 0
  const num = () => Number(tok[k++])
  const line = (tx: number, ty: number) => {
    const n = Math.max(1, Math.ceil(Math.hypot(tx - x, ty - y)))
    for (let s = 1; s <= n; s++) pts.push(x + ((tx - x) * s) / n, y + ((ty - y) * s) / n)
    x = tx
    y = ty
  }
  // Endpoint to centre form, for a circle with no rotation (SVG 1.1 implementation notes, F.6.5).
  const arc = (r: number, large: number, sweep: number, tx: number, ty: number) => {
    const hx = (x - tx) / 2
    const hy = (y - ty) / 2
    const h2 = hx * hx + hy * hy
    if (h2 === 0) return
    r = Math.max(r, Math.sqrt(h2))
    const c = Math.sqrt(Math.max(0, (r * r - h2) / h2)) * (large === sweep ? -1 : 1)
    const cx = c * hy + (x + tx) / 2
    const cy = -c * hx + (y + ty) / 2
    const a0 = Math.atan2(y - cy, x - cx)
    let da = Math.atan2(ty - cy, tx - cx) - a0
    if (sweep && da < 0) da += 2 * Math.PI
    if (!sweep && da > 0) da -= 2 * Math.PI
    const n = Math.max(1, Math.ceil(Math.abs(da) * r))
    for (let s = 1; s <= n; s++) {
      const a = a0 + (da * s) / n
      pts.push(cx + r * Math.cos(a), cy + r * Math.sin(a))
    }
    x = tx
    y = ty
  }
  while (k < tok.length) {
    if (/[a-zA-Z]/.test(tok[k])) cmd = tok[k++]
    const rel = cmd === cmd.toLowerCase()
    const ox = rel ? x : 0
    const oy = rel ? y : 0
    switch (cmd.toUpperCase()) {
      case 'M':
        if (pts.length) chains.push(pts)
        x = x0 = ox + num()
        y = y0 = oy + num()
        pts = [x, y]
        // Pairs after a moveto are linetos.
        cmd = rel ? 'l' : 'L'
        break
      case 'L':
        line(ox + num(), oy + num())
        break
      case 'H':
        line(ox + num(), y)
        break
      case 'V':
        line(x, oy + num())
        break
      case 'A': {
        const r = num()
        k += 2
        const large = num()
        const sweep = num()
        arc(r, large, sweep, ox + num(), oy + num())
        break
      }
      case 'Z':
        line(x0, y0)
        break
      default:
        k++
    }
  }
  if (pts.length) chains.push(pts)
  return chains
}

// The pixels each subpath passes through, in order and one pixel thick: repeats are dropped, and
// so is the corner pixel of every L-shaped step, which keeps diagonals from doubling up.
function run(chains: number[][], m: DOMMatrix) {
  const out: number[] = []
  for (const pts of chains) {
    const start = out.length
    for (let s = 0; s < pts.length; s += 2) {
      const i = Math.floor(m.a * pts[s] + m.c * pts[s + 1] + m.e)
      const j = Math.floor(m.b * pts[s] + m.d * pts[s + 1] + m.f)
      const n = out.length - start
      if (n && out[start + n - 2] === i && out[start + n - 1] === j) continue
      if (
        n >= 4 &&
        Math.abs(i - out[start + n - 4]) === 1 &&
        Math.abs(j - out[start + n - 3]) === 1
      )
        out.length -= 2
      out.push(i, j)
    }
  }
  return out
}

// Prints the ruin from the hidden SVG geometry: plates dithered through the stone's lighting,
// keylines one pixel thick, dial marks wherever they cover half a pixel, and chips bitten out
// wherever an `nf-void` says, inked where they break into stone. Every stone is lit in its own
// upright place in the ring, so its chamfers travel with it when it slips and when the ring leans.
// Below `ground` the ring is buried in the island and prints nothing, so the turf's own edge is the
// line it sinks behind. The light that spills onto the lower stones and the tablet animates on top
// of this, so its pixels are returned as a list.
function rasterRuin(
  d: Uint8ClampedArray,
  g: Grid,
  svg: SVGSVGElement,
  dark: boolean,
  ground: (x: number) => number,
  side: number,
  dim: number,
) {
  const art = svg.querySelector('#nf-temple-art')
  const root = svg.getScreenCTM()
  if (!art || !root) return null
  const rootInv = root.inverse()
  const { bw, bh } = g
  const sheet = document.createElement('canvas')
  sheet.width = bw
  sheet.height = bh
  const scratch = sheet.getContext('2d', { willReadFrequently: true })!
  const stops: RGB[][] = MATERIALS.map(() => [])
  const paper: RGB = dark ? [206, 205, 195] : [255, 252, 240]
  let ink: RGB | null = null
  const toBuffer = (el: SVGGraphicsElement) => {
    const ctm = el.getScreenCTM()
    return ctm && g.toBuffer.multiply(rootInv.multiply(ctm))
  }
  const tone = (el: Element, id: number) => {
    if (stops[id].length === 0) {
      const base = mul(probe(getComputedStyle(el).fill), dim)
      stops[id] = dark
        ? [mul(base, 0.55), mul(base, 0.75), base, mix(base, paper, 0.2)]
        : [mul(base, 0.7), mul(base, 0.86), base, mix(base, paper, 0.55)]
    }
    return stops[id]
  }
  const inside = (i: number, j: number) => i >= 0 && j >= 0 && i < bw && j < bh
  // Which paint owns each pixel, stacked in document order as the SVG would stack them: 1..3 a
  // material plate, 4 ink, 0 bare. Later elements cover earlier ones, keylines included.
  const INK = 4
  const owner = new Uint8Array(bw * bh)
  const voided = new Uint8Array(bw * bh)
  // Which frame lights each plate pixel: 0 the scene, n a stone's upright place in the ring.
  const frame = new Uint8Array(bw * bh)
  const frames: DOMMatrix[] = [new DOMMatrix()]
  const frameOf = new Map<Element, number>()
  const frameFor = (el: Element) => {
    const stone = el.closest<SVGGElement>('.nf-stone')
    const m = stone?.getScreenCTM()
    if (!stone || !m) return 0
    let f = frameOf.get(stone)
    if (f === undefined) {
      f = frames.push(rootInv.multiply(m).inverse()) - 1
      frameOf.set(stone, f)
    }
    return f
  }
  const cover = (el: SVGPathElement, m: DOMMatrix, set: number, threshold: number, f = 0) => {
    const b = el.getBBox()
    const xs: number[] = []
    const ys: number[] = []
    for (const [x, y] of [
      [b.x, b.y],
      [b.x + b.width, b.y],
      [b.x, b.y + b.height],
      [b.x + b.width, b.y + b.height],
    ]) {
      const q = m.transformPoint(new DOMPoint(x, y))
      xs.push(q.x)
      ys.push(q.y)
    }
    const x0 = Math.max(0, Math.floor(Math.min(...xs)) - 1)
    const y0 = Math.max(0, Math.floor(Math.min(...ys)) - 1)
    const w = Math.min(bw, Math.ceil(Math.max(...xs)) + 1) - x0
    const h = Math.min(bh, Math.ceil(Math.max(...ys)) + 1) - y0
    if (w <= 0 || h <= 0) return
    scratch.resetTransform()
    scratch.clearRect(x0, y0, w, h)
    scratch.setTransform(m)
    scratch.fill(new Path2D(el.getAttribute('d') ?? ''))
    const a = scratch.getImageData(x0, y0, w, h).data
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        if (a[(j * w + i) * 4 + 3] < threshold) continue
        const p = (y0 + j) * bw + x0 + i
        owner[p] = set
        frame[p] = f
        if (set === 0) voided[p] = 1
      }
  }

  for (const el of art.querySelectorAll<SVGPathElement>('path')) {
    const m = toBuffer(el)
    if (!m) continue
    const cls = el.getAttribute('class') ?? ''
    const id = MATERIALS.findIndex(c => cls.includes(c))
    if (cls.includes('nf-key')) {
      ink ??= probe(getComputedStyle(el).stroke)
      const px = run(trace(el.getAttribute('d') ?? ''), m)
      for (let s = 0; s < px.length; s += 2)
        if (inside(px[s], px[s + 1])) owner[px[s + 1] * bw + px[s]] = INK
    } else if (id >= 0) {
      tone(el, id)
      cover(el, m, id + 1, 128, frameFor(el))
    } else if (cls.includes('nf-ink')) cover(el, m, INK, 110)
    else if (cls.includes('nf-void')) cover(el, m, 0, 128)
  }
  ink ??= [16, 15, 15]
  for (let j = 1; j < bh - 1; j++)
    for (let i = 1; i < bw - 1; i++) {
      const p = j * bw + i
      if (owner[p] === 0 || owner[p] === INK) continue
      if (voided[p - 1] || voided[p + 1] || voided[p - bw] || voided[p + bw]) owner[p] = INK
    }

  const spill: number[] = []
  const bands: number[] = []
  const sill = GATE.y + GATE.r * 0.7
  for (let j = 0; j < bh; j++)
    for (let i = 0; i < bw; i++) {
      const p = j * bw + i
      const o = owner[p]
      const x = g.sx[i]
      const y = g.sy[j]
      if (o === 0 || (y >= ground(x) && Math.hypot(x - GATE.x, y - GATE.y) < 270)) continue
      if (o === INK) {
        put(d, p * 4, ink)
        continue
      }
      const f = frames[frame[p]]
      const lx = f.a * x + f.c * y + f.e
      const ly = f.b * x + f.d * y + f.f
      put(d, p * 4, stops[o - 1][PLATE[step(5, stoneTone(lx, ly, side) * 0.75 + 0.125, i, j)]])
      if (o === 1 && y > sill && Math.abs(x - GATE.x) < 150) {
        spill.push(p)
        bands.push(Math.min(3, Math.floor((y - sill) / 40)))
      }
    }

  const print = { ink, stone: stops[0], rose: stops[1], sage: stops[2] }
  return { print, spill, bands } satisfies Masonry
}

// Rose light from the open gate falls on the stones under the opening and the tablet below as a
// flat wash, one step weaker for every band further down: multiplied into the stone by day,
// screened over it by night.
function paintSpill(d: Uint8ClampedArray, m: Masonry, glow: number, dark: boolean) {
  if (glow <= 0) return
  const rose = m.print.rose[2]
  for (let s = 0; s < m.spill.length; s++) {
    const k = m.spill[s] * 4
    const a = glow * (0.5 - 0.1 * m.bands[s])
    for (let c = 0; c < 3; c++) {
      const lit = dark
        ? 255 - ((255 - d[k + c]) * (255 - rose[c])) / 255
        : (d[k + c] * rose[c]) / 255
      d[k + c] += (lit - d[k + c]) * a
    }
  }
}

const HUB = 15
// Lume, in the sage of the site's palette: Super-LumiNova's C3 glows yellow-green, and it is what
// the hands turn to once the daylight goes.
const LUME = hex('#cdd597')
const luma = (c: RGB) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

// Station-clock hands on the pole star's arbor, laid into the rift pixel by pixel. The hour and
// minute hands are bars lit on their sunward edge and keylined in the rift's deepest ink, which
// holds them apart from the rose arms of the nebula; the red second hand is a pixel wide with its
// disc; the hub is an open ring so the star shows through. The rift is night in either theme, so by
// day the hands take the lighter of stone and ink, which one depending on the theme. After sunset
// a light page's stone greys with the rest of the ruin, so `lume` (0 by day, 1 at night) carries
// the hands over to lume instead.
function paintHands(d: Uint8ClampedArray, g: Grid, dial: Dial, print: Print, lume: number) {
  const { bw, bh, sx, sy } = g
  const stone = luma(print.stone[3]) >= luma(print.ink)
  const face = mix(stone ? print.stone[3] : print.ink, mix(CREAM, LUME, 0.45), lume)
  const shade = mix(stone ? print.stone[2] : mix(print.ink, RIFT[3], 0.4), LUME, lume)
  const key = RIFT[0]
  const red = print.rose[lume > 0.5 ? 3 : 2]
  const bars = [
    { deg: dial.hour, w: 14, len: 112, tail: 40, lit: face },
    { deg: dial.minute, w: 10, len: 150, tail: 44, lit: face },
    { deg: dial.second, w: 5, len: 102, tail: 52, lit: red },
  ].map(b => {
    // The hands turn on the ring's own dial, so they lean with it.
    const a = ((b.deg + TILT) * Math.PI) / 180
    const ux = Math.sin(a)
    const uy = -Math.cos(a)
    // Which side of the bar faces the sun, up and to the right.
    const sun = -uy * 0.6 - ux * 0.8 >= 0 ? 1 : -1
    const half = Math.max(b.w / 2, g.px / 2)
    const edged = b.w > g.px * 2
    // The keyline rings the broad hands a pixel out, ends included.
    return { ...b, ux, uy, sun, half, edged, rim: edged ? g.px : 0 }
  })
  const disc = [GATE.x + bars[2].ux * 114, GATE.y + bars[2].uy * 114]
  const i0 = Math.max(0, col(g, GATE.x - 160))
  const i1 = Math.min(bw - 1, col(g, GATE.x + 160))
  const j0 = Math.max(0, row(g, GATE.y - 160))
  const j1 = Math.min(bh - 1, row(g, GATE.y + 160))
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      const dx = sx[i] - GATE.x
      const dy = sy[j] - GATE.y
      let c: RGB | null = null
      for (const b of bars) {
        const u = dx * b.ux + dy * b.uy
        if (u > b.len + b.rim || u < -b.tail - b.rim || Math.abs(u) < HUB - b.rim) continue
        const v = dy * b.ux - dx * b.uy
        if (Math.abs(v) > b.half + b.rim) continue
        // Later bars lie over earlier ones, so a keyline only paints where nothing is yet.
        if (Math.abs(v) > b.half || u > b.len || u < -b.tail || Math.abs(u) < HUB) c ??= key
        else c = b.edged && v * b.sun < g.px - b.half ? (b.lit === red ? red : shade) : b.lit
      }
      if (Math.hypot(sx[i] - disc[0], sy[j] - disc[1]) <= 12) c = red
      if (Math.abs(Math.hypot(dx, dy) - HUB) <= g.px / 2) c = face
      if (c) put(d, (j * bw + i) * 4, c)
    }
  paintIvy(d, g, bars[0].ux, bars[0].uy, bars[0].half, () => true)
}

// Distance from the scene point to the segment, with the signed side for bark lighting.
function segHit(s: Seg, x: number, y: number): [number, number, number] {
  const dx = s.x1 - s.x0
  const dy = s.y1 - s.y0
  const len2 = dx * dx + dy * dy || 1
  const t = Math.max(0, Math.min(1, ((x - s.x0) * dx + (y - s.y0) * dy) / len2))
  const qx = x - (s.x0 + dx * t)
  const qy = y - (s.y0 + dy * t)
  const side = (dx * qy - dy * qx) / Math.sqrt(len2)
  return [Math.hypot(qx, qy), s.w0 + (s.w1 - s.w0) * t, side]
}

type Canopy = { trees: Tree[]; roots: number[] }

// Leaves: each tip splats density and depth; noise dragged by per-octave wind thresholds it into
// masses. Depth follows the limbs: near leaves are denser, lighter and crisp-edged, far leaves
// darker with a wider penumbra, and that penumbra is what the ordered dither renders.
function paintCanopy(
  d: Uint8ClampedArray,
  g: Grid,
  canopy: Canopy,
  pal: Palette,
  wind: number,
  drift: number,
  side: number,
  s: Season,
  sky: Sky,
) {
  const { bw, bh, sx, sy } = g
  const { bare, autumn, blossom } = s
  const snow = s.ground
  // A hard frost rimes the twigs in any weather (rime ice in freezing fog, glaze in freezing rain).
  // Hoarfrost adds to it only on clear, still, dry nights, since season() zeroes s.frost under
  // rain, snow or fog; twigs radiate their warmth away as the grass does and hold it until the
  // sun is on them.
  const rime = Math.max(
    sky.temp === null ? 0 : smooth(0, -8, sky.temp),
    smooth(0.4, 1, s.frost) * (1 - s.thaw),
  )
  const wet = sky.fall === 'rain' || sky.fall === 'drizzle' ? 0.4 + 0.6 * sky.heavy : 0
  const petal = mix(pal.snow[3], pal.poppy, 0.25)
  const wood = snow + rime > 0.05 ? new Uint8Array(bw * bh) : null
  const radius = 40
  const density = new Float32Array(bw * bh)
  const depth = new Float32Array(bw * bh)
  const toI = (x: number) => Math.floor((x - sx[0]) / g.px + 0.5)
  const toJ = (y: number) => Math.floor((y - sy[0]) / g.px + 0.5)

  for (const tree of canopy.trees) {
    for (const [tx, ty, tz] of tree.tips) {
      const i0 = Math.max(0, toI(tx - radius))
      const i1 = Math.min(bw - 1, toI(tx + radius))
      const j0 = Math.max(0, toJ(ty - radius))
      const j1 = Math.min(bh - 1, toJ(ty + radius))
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const q = 1 - Math.hypot(sx[i] - tx, sy[j] - ty) / radius
          if (q <= 0) continue
          const p = j * bw + i
          density[p] += q * q
          depth[p] += q * q * tz
        }
    }
    for (const s of tree.segs) {
      // Twigs thinner than a pixel stay hidden in the leaves instead of stair-stepping across it,
      // until the leaves are down and the twigs are all there is to see.
      if (s.w0 < g.px * (0.8 - 0.45 * bare)) continue
      const pad = Math.max(s.w0, s.w1) / 2 + g.px
      const i0 = Math.max(0, toI(Math.min(s.x0, s.x1) - pad))
      const i1 = Math.min(bw - 1, toI(Math.max(s.x0, s.x1) + pad))
      const j0 = Math.max(0, toJ(Math.min(s.y0, s.y1) - pad))
      const j1 = Math.min(bh - 1, toJ(Math.max(s.y0, s.y1) + pad))
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const [dist, w, side] = segHit(s, sx[i], sy[j])
          const half = Math.max(w / 2, g.px * 0.5)
          if (dist > half) continue
          put(d, (j * bw + i) * 4, ramp(pal.bark, 0.5 + (side / half) * 0.5, i, j))
          if (wood) wood[j * bw + i] = 1
        }
    }
  }
  // Snow lies along the tops of the limbs, a pixel deep and two in a heavy fall; in a hard frost
  // rime whitens their edges.
  if (wood)
    for (let j = 1; j < bh - 1; j++)
      for (let i = 1; i < bw - 1; i++) {
        const p = j * bw + i
        if (!wood[p]) continue
        const r = hash(i, j, 5)
        if (!wood[p - bw] && r < snow * 1.4) {
          put(d, p * 4, pal.snow[3])
          if (r < (snow - 0.6) * 2.5) put(d, (p - bw) * 4, pal.snow[3])
        } else if ((!wood[p - 1] || !wood[p + 1] || !wood[p + bw]) && r < rime * 0.7)
          put(d, p * 4, pal.snow[2])
      }

  // Continuous coverage T: 0.5 is the leaf edge, the slope is the penumbra width.
  const cover = new Float32Array(bw * bh)
  const scale = 1 / (radius * 1.6)
  for (let p = 0; p < bw * bh; p++) {
    if (density[p] < 0.05) continue
    const i = p % bw
    const j = (p - i) / bw
    const z = depth[p] / density[p]
    depth[p] = z
    const n = windFbm(sx[i] * scale, sy[j] * scale + drift, 71, 2, wind)
    // In winter most of the leaves are down and the limbs show.
    const v = density[p] * (0.35 + n) * (1 + 0.15 * z) * (1 - 0.92 * bare)
    const penumbra = 0.1 + 0.16 * (1 - z) * 0.5
    cover[p] = clamp01((v - 0.55) / penumbra + 0.5)
  }
  for (let j = 1; j < bh - 1; j++)
    for (let i = 1; i < bw - 1; i++) {
      const p = j * bw + i
      const t = cover[p]
      if (t <= BAYER[(j & 3) * 4 + (i & 3)]) continue
      const k = p * 4
      const grad = Math.max(
        Math.abs(cover[p + 1] - cover[p - 1]),
        Math.abs(cover[p + bw] - cover[p - bw]),
      )
      // Only the fringe facing the light, up and to its side, catches gold.
      const sunward = cover[p - bw + side] < t
      // A tree turns a limb at a time, so the autumn comes in patches the size of a limb.
      const leaf =
        autumn > 0 && fbm(sx[i] * 0.012, sy[j] * 0.012, 29, 2) < 0.15 + 0.55 * autumn
          ? pal.autumn
          : pal.crown
      // Snow settles on the tops of the masses, and blossom comes out in clusters.
      const cap = cover[p - bw] < 0.5 ? 1 : cover[p - 2 * bw] < 0.5 ? 2 : 0
      if (cap && hash(i, j, 9) < (cap === 1 ? snow * 1.3 : snow - 0.5))
        put(d, k, pal.snow[cap === 1 ? 3 : 2])
      else if (blossom > 0.05 && fbm(sx[i] * 0.07, sy[j] * 0.07, 37, 2) > 0.7 - 0.25 * blossom)
        put(d, k, petal)
      else if (t >= GOLD_START && t <= GOLD_END && grad > 0.3 && sunward && wet < 0.3)
        put(d, k, pal.gold)
      else if (t < 0.6 && !sunward) put(d, k, leaf[0])
      else {
        // Wet leaves go darker.
        const tone = 0.2 + fbm(sx[i] * 0.05, sy[j] * 0.05, 83, 2) * 0.5 + depth[p] * 0.18
        put(d, k, ramp(leaf, tone + (sunward ? 0.12 : 0) - 0.12 * wet, i, j))
      }
    }
}

// Cast shade from the canopy on the ground plane. Penumbra follows U = f·b/a: the farther the
// ground sits from the occluding limbs (b), the wider the soft edge, so shade is crisp at the
// roots and dissolves into dithered haze toward the temple. Fringes are gold only while sharp.
function paintDapple(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  roots: number[],
  wind: number,
  clock: number,
  water: number,
) {
  const { bw, bh, sx, sy } = g
  const [sr, sg, sb, sa] = pal.shade
  const [fr, fg, fb, fa] = pal.fringe
  const focal = 0.09
  // Shade lands only on the near ground, the hill.
  const floor = Float32Array.from(sx, x => hillTop(x, water) - 6)
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    const ground = 1 - smooth(water - 20, water, y)
    if (ground <= 0 || y < 560) continue
    for (let i = 0; i < bw; i++) {
      const x = sx[i]
      if (y < floor[i]) continue
      const b = Math.min(1, Math.min(...roots.map(r => Math.abs(x - r))) / 640)
      const cover = ground * (1 - smooth(0.25, 0.85, b))
      if (cover < 0.06) continue
      const k = (j * bw + i) * 4
      if (d[k + 3] === 255) continue
      const n = windFbm(x * 0.011, y * 0.03 + Math.sin(clock * 0.4) * 0.05, 97, 3, wind)
      const cut = 1 - cover * 0.6
      const u = 0.012 + focal * b
      const shade = smooth(cut - u, cut + u, n)
      if (shade >= GOLD_START && shade <= GOLD_END && u < 0.035) put(d, k, [fr, fg, fb], fa * 255)
      else if (shade > BAYER[(j & 3) * 4 + (i & 3)]) put(d, k, [sr, sg, sb], sa * 255)
    }
  }
}

// The rift through the gate, always night whatever the page theme: a whirlpool of brush strokes
// under a ringed pole star, which is also the arbor the clock hands turn on. Motes orbit with
// differential rotation and lay their strokes back along their own paths; hovering opens an inflow,
// so the strokes lengthen into spirals and echoes of the dial sink into the pole. Comets fall past
// on Kepler arcs.
const RIFT = [
  '#0b0b0e',
  '#10131f',
  '#172036',
  '#22305a',
  '#34427a',
  '#5d5a8c',
  '#9a7094',
  '#e3a19a',
].map(hex)
const STAR_COLORS = ['#fffcf0', '#fffcf0', '#f1d67e', '#fdb2a2', '#cdd597', '#92bfdb'].map(hex)
const CREAM = hex('#fffcf0')
const INK = hex('#100f0f')
const GOLDEN = hex('#f1d67e')
const ROSE = hex('#e3a19a')
const FLAME = ['#fffcf0', '#f1d67e', '#da702c', '#af3029'].map(hex)
const LEAK = 58
const RIM = GATE.r + LEAK + 4
const GLOW = 230
const GM = 4.5e6

type Mote = {
  r: number
  a: number
  tone: number
  thr: number
  len: number
  wide: boolean
  star: RGB | null
  mag: number
  freq: number
  phase: number
}
type Comet = {
  x: number
  y: number
  vx: number
  vy: number
  age: number
  acc: number
  trail: number[]
}
type Opening = { grid: Grid; mask: Uint8Array; i0: number; i1: number; j0: number; j1: number }
type Rift = {
  opening: Opening | null
  pole: Pt
  motes: Mote[]
  rate: number
  target: number
  inflow: number
  inflowTarget: number
  pattern: number
  heat: number
  flare: number
  comets: Comet[]
  nextComet: number
}

// Differential rotation: the core turns about twice as fast as the reference radius, the rim half.
const orbit = (r: number) => 180 / (r + 90)
// Idle stroke pitch, and the winding that makes two arms match it: tan(pitch) = 2 / WIND.
const PITCH = 0.3
const WIND = 2 / Math.tan(PITCH)

function nebula(x: number, y: number) {
  const q = fbm(x, y, 211, 3)
  const r = fbm(x + 4 * q, y + 4 * q, 223, 3)
  return fbm(x + 3.5 * r, y + 3.5 * r, 227, 3)
}

// How much of the rift shows at each buffer pixel, out of 255: all of it inside the back edge of the
// opening, a pixel generous so it tucks under the soffit and the ring. Where the eight o'clock stone
// is missing nothing holds it in, so it bleeds out through the gap and thins into dither.
function opening(g: Grid): Opening {
  const mask = new Uint8Array(g.bw * g.bh)
  const reach = GATE.r + g.px
  const gap = hourAngle(MISSING) + (TILT * Math.PI) / 180
  let [i0, i1, j0, j1] = [g.bw, -1, g.bh, -1]
  for (let j = 0; j < g.bh; j++)
    for (let i = 0; i < g.bw; i++) {
      const dx = g.sx[i] - GATE.x
      const dy = g.sy[j] - GATE.y
      const r = Math.hypot(dx, dy)
      const off = Math.atan2(dy, dx) - gap
      const bleed = r > GATE.r && Math.abs(Math.atan2(Math.sin(off), Math.cos(off))) < Math.PI / 12
      const cover = bleed
        ? Math.round(255 * Math.max(0, 1 - (r - GATE.r) / LEAK) ** 1.5)
        : Math.hypot(dx, dy - GATE.depth) <= reach
          ? 255
          : 0
      if (!cover) continue
      mask[j * g.bw + i] = cover
      i0 = Math.min(i0, i)
      i1 = Math.max(i1, i)
      j0 = Math.min(j0, j)
      j1 = Math.max(j1, j)
    }
  return { grid: g, mask, i0, i1, j0, j1 }
}

function makeRift(pole: Pt): Rift {
  const rand = mulberry(404)
  const motes = Array.from({ length: 1600 }, () => {
    const r = RIM * Math.sqrt(rand())
    const a = rand() * Math.PI * 2
    const star = rand() < 0.05 ? STAR_COLORS[Math.floor(rand() * STAR_COLORS.length)] : null
    return {
      r,
      a,
      tone: nebula(Math.cos(a) * r * 0.006, Math.sin(a) * r * 0.006),
      thr: rand(),
      len: star ? 1 : 3 + Math.floor(rand() * 4),
      wide: !star && rand() < 0.45,
      star,
      mag: rand() ** 2,
      freq: 0.8 + rand() * 2.4,
      phase: rand() * Math.PI * 2,
    }
  })
  return {
    opening: null,
    pole,
    motes,
    rate: 1.1,
    target: 1.1,
    inflow: 0,
    inflowTarget: 0,
    pattern: 0,
    heat: 0,
    flare: 0,
    comets: [],
    nextComet: 2.5,
  }
}

function paintRift(d: Uint8ClampedArray, g: Grid, rift: Rift, clock: number) {
  const { bw, sx, sy } = g
  const { pole } = rift
  if (rift.opening?.grid !== g) rift.opening = opening(g)
  const { mask, i0, i1, j0, j1 } = rift.opening
  const plot = (x: number, y: number, c: RGB, alpha: number) => {
    const i = col(g, x)
    const j = row(g, y)
    if (i < i0 || i > i1 || j < j0 || j > j1) return
    const k = j * bw + i
    if (alpha * mask[k] > BAYER[(j & 3) * 4 + (i & 3)] * 255) put(d, k * 4, c)
  }

  // Underpainting: a dark well that brightens toward the pole.
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      if (mask[j * bw + i] <= BAYER[(j & 3) * 4 + (i & 3)] * 255) continue
      const glow = 1 - Math.min(1, Math.hypot(sx[i] - pole[0], sy[j] - pole[1]) / GLOW)
      put(d, (j * bw + i) * 4, ramp(RIFT, 0.04 + glow * 0.34, i, j))
    }

  // Each stroke runs back along its mote's spiral, whose pitch opens as the inflow outruns the
  // spin. Colour mixes the nebula the mote was born in with a two-armed density wave (Lin–Shu):
  // trailing arms at the strokes' own idle pitch, turned against the spin so they stream inward
  // while the material circles through them, and nothing ever winds up.
  const spin = (rift.rate * Math.PI) / 180
  const trail = Math.min(1.1, Math.max(0, (rift.rate - 2) / 30))
  for (const m of rift.motes) {
    const x = pole[0] + Math.cos(m.a) * m.r
    const y = pole[1] + Math.sin(m.a) * m.r
    const pitch = Math.min(1.25, PITCH + Math.atan2(rift.inflow, orbit(m.r) * spin))
    const glow = 1 - Math.min(1, m.r / GLOW)
    const arm = Math.cos(2 * (m.a - rift.pattern) + WIND * Math.log(m.r / 25 + 0.05))
    const twinkle = m.star
      ? 0.75 * (0.35 + 0.65 * m.mag * (0.6 + 0.4 * Math.sin(clock * m.freq + m.phase)))
      : 1
    const crest = Math.max(0, arm) ** 4 * 0.22
    const ink = m.star ?? pick(RIFT, m.tone * 1.2 - 0.36 + glow * 0.3 + arm * 0.16 + crest, m.thr)
    const len = Math.max(m.len * g.px, m.r * trail * orbit(m.r) * 0.6)
    const n = Math.ceil(len / (g.px * 0.8))
    const cp = (Math.cos(pitch) * len) / n
    const sp = (Math.sin(pitch) * len) / n
    let r = m.r
    let a = m.a
    for (let s = 0; s < n; s++) {
      const fade = twinkle * (1 - (s / n) * trail * 0.85)
      plot(pole[0] + Math.cos(a) * r, pole[1] + Math.sin(a) * r, ink, fade)
      if (m.wide)
        plot(pole[0] + Math.cos(a) * (r + g.px), pole[1] + Math.sin(a) * (r + g.px), ink, fade)
      a -= cp / r
      r += sp
    }
    if (m.star && m.mag > 0.8)
      for (const [ox, oy] of CROSS) plot(x + ox * g.px, y + oy * g.px, m.star, twinkle * 0.8)
    if (m.star && m.mag > 0.93)
      halo(plot, x, y, g.px * 2.3, g.px * 0.8, m.star, twinkle * 0.45, clock * 0.3, 4)
  }

  // Polaris as the Saint-Rémy stars are painted: a hard core inside counter-turning broken rings.
  const breath = 0.5 + 0.5 * Math.sin(clock * 2.2)
  plot(pole[0], pole[1], CREAM, 1)
  for (const [ox, oy] of CROSS) plot(pole[0] + ox * g.px, pole[1] + oy * g.px, CREAM, 1)
  halo(
    plot,
    pole[0],
    pole[1],
    (3.2 + rift.flare * 2) * g.px,
    g.px * 0.7,
    GOLDEN,
    0.9,
    clock * 0.5,
    6,
  )
  halo(
    plot,
    pole[0],
    pole[1],
    (5.6 + breath * 0.8 + rift.flare * 4) * g.px,
    g.px * 0.7,
    ROSE,
    0.45 + 0.35 * breath,
    -clock * 0.35,
    8,
  )

  // The way back: echoes of the dial peel off the ring and sink into the pole, sixty ticks each with
  // the hours cut longer, turning with the whirlpool and wobbling as they go.
  if (rift.heat > 0.02) {
    for (let k = 0; k < 4; k++) {
      const cycle = clock / 1.7 + k / 4
      const p = frac(cycle)
      const size = (GATE.r - 6) * (1 - p) ** 1.3
      const cy = pole[1] + GATE.depth * (1 - p)
      const alpha = (1 - p) * Math.min(1, rift.heat)
      const ink = p < 0.45 ? CREAM : p < 0.75 ? GOLDEN : ROSE
      for (let t = 0; t < 60; t++) {
        // Broken ticks, recut every cycle.
        if (hash(t, k + Math.floor(cycle) * 7, 17) < 0.18) continue
        const th = (t * Math.PI) / 30 - Math.PI / 2 + p * 1.2
        const r =
          size *
          (1 +
            0.07 * Math.sin(3 * th + clock * 2.4 + k) +
            0.18 * (noise(th * 1.7 + k * 5, clock * 0.9, 311) - 0.5))
        const hour = t % 5 === 0
        const len = r * (hour ? 0.14 : 0.06)
        const n = Math.max(1, Math.ceil(len / (g.px * 0.8)))
        for (let s = 0; s <= n; s++) {
          // Each tick bends a little along the spiral as it runs inward.
          const rr = r - (len * s) / n
          const a = th + (s / n) * 0.05
          plot(pole[0] + Math.cos(a) * rr, cy + Math.sin(a) * rr, ink, alpha)
          if (hour)
            plot(
              pole[0] + Math.cos(a + g.px / rr) * rr,
              cy + Math.sin(a + g.px / rr) * rr,
              ink,
              alpha,
            )
        }
      }
    }
  }

  for (const c of rift.comets) {
    const fade = Math.min(1, (3.2 - c.age) / 0.6)
    const pts = c.trail
    const n = pts.length / 2
    for (let k = 1; k < n; k++) {
      const ax = pts[2 * k - 2]
      const ay = pts[2 * k - 1]
      const dx = pts[2 * k] - ax
      const dy = pts[2 * k + 1] - ay
      const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / (g.px * 0.8)))
      const w = k / n
      const ink = w > 0.7 ? CREAM : w > 0.35 ? GOLDEN : ROSE
      for (let s = 0; s < steps; s++)
        plot(ax + (dx * s) / steps, ay + (dy * s) / steps, ink, fade * w)
    }
    plot(c.x, c.y, CREAM, fade)
  }
}

function stepRift(rift: Rift, dt: number) {
  rift.rate += (rift.target - rift.rate) * Math.min(1, dt * 2.2)
  rift.inflow += (rift.inflowTarget - rift.inflow) * Math.min(1, dt * 1.6)
  rift.heat = Math.min(1, Math.max(0, (rift.rate - 1.1) / 24))
  rift.flare *= Math.exp(-dt * 2.5)
  const spin = (rift.rate * Math.PI) / 180
  rift.pattern -= spin * 0.6 * dt
  // Motes keep their radius so the disc never thins out; the inflow shows only as pitch.
  for (const m of rift.motes) m.a += orbit(m.r) * spin * dt

  rift.nextComet -= dt
  if (rift.nextComet <= 0) {
    // Launched from beyond the ring, mostly sideways and a little inward, so some orbit and some
    // escape.
    const a = Math.random() * Math.PI * 2
    const v = 130 + Math.random() * 170
    rift.comets.push({
      x: rift.pole[0] + Math.cos(a) * (GATE.r + 40),
      y: rift.pole[1] + Math.sin(a) * (GATE.r + 40),
      vx: -Math.sin(a) * v - Math.cos(a) * 40,
      vy: Math.cos(a) * v - Math.sin(a) * 40,
      age: 0,
      acc: 0,
      trail: [],
    })
    rift.nextComet = rift.heat > 0.5 ? 0.5 + Math.random() * 0.8 : 3 + Math.random() * 5
  }
  // Inverse-square pull toward the pole bends every path into a conic; the inflow adds drag, so
  // on hover the comets spiral in and the pole flares as it swallows them.
  const drag = Math.exp(-rift.inflow * 1.4 * dt)
  for (const c of rift.comets) {
    const dx = c.x - rift.pole[0]
    const dy = c.y - rift.pole[1]
    const r2 = dx * dx + dy * dy + 300
    const pull = GM / (r2 * Math.sqrt(r2))
    c.vx = (c.vx - dx * pull * dt) * drag
    c.vy = (c.vy - dy * pull * dt) * drag
    c.x += c.vx * dt
    c.y += c.vy * dt
    c.age += dt
    c.acc += dt
    if (c.acc >= 1 / 60) {
      c.acc = 0
      c.trail.push(c.x, c.y)
      if (c.trail.length > 44) c.trail.splice(0, 2)
    }
    if (r2 < 450) {
      c.age = 99
      rift.flare = 1
    } else if (r2 > 520 * 520) c.age = 99
  }
  rift.comets = rift.comets.filter(c => c.age < 3.2)
}

export function setupLandscape(
  scene: HTMLElement,
  reduce: boolean,
  water: number,
  dial: Dial,
  site: Site,
): Landscape | null {
  const canvas = (cls: string) => scene.querySelector<HTMLCanvasElement>(cls)
  const skyCanvas = canvas('.nf-sky')
  const back = canvas('.nf-land')
  const riftCanvas = canvas('.nf-rift')
  const hill = canvas('.nf-hill')
  const grain = canvas('.nf-grain')
  const near = canvas('.nf-near')
  const front = canvas('.nf-canopy')
  const temple = scene.querySelector<SVGSVGElement>('.nf-temple')
  if (!skyCanvas || !back || !riftCanvas || !hill || !grain || !near || !front || !temple)
    return null

  const pole: Pt = [GATE.x, GATE.y]
  const rift = makeRift(pole)
  const air: Air = { drift: 0, time: 0, pull: 0, pole, vortices: [], body: null, cover: 0.3 }
  const walker: Walker = { x: 900, dir: 1, pause: 1.5, next: 9, stride: 0, walking: false }
  // Start with the fire already smoking, so the column is standing on the first frame.
  const hearth: Hearth = { smoke: [], embers: [], spawn: 0, spark: 0 }
  for (let n = 0; n < 105; n++) stepHearth(hearth, air, 1 / 15)
  const ripples = makeRipples(water)
  const splashes: Splash[] = []
  const bank = (x: number) => hillTop(x, water)
  const overgrowth = makeOvergrowth()
  const curtain = makeCurtain()
  const cat: Cat = makeCat(1250, bank(1250))
  const crew = makeCrew()
  const tri = makeTriathlete(reduce)
  // The people on the far ridges talk in their own layer, so their bubbles travel with them.
  const far: Voices = { bubbles: [], heads: {} }
  // And the two up the left-hand tree in theirs.
  const boughs: Voices = { bubbles: [], heads: {} }
  const litter: Shed = { leaves: [], spawn: 0 }
  let bough: Bough | null = null
  let hush = 0
  // The triathlete keeps just behind the camp's footing, and follows the bank down to the water
  // where the hill runs out. Past the camp itself they swing up the bank, far enough that their
  // feet clear the tops of the tent and the campers, so the fire is all that crosses them.
  let camp: readonly [number, number] = [0, 0]
  const strand = (x: number) =>
    Math.min(water - 3, Math.max(844, bank(x) + 4)) -
    28 * smooth(camp[0] - 60, camp[0], x) * (1 - smooth(camp[1], camp[1] + 60, x))
  // The rift's angular speed at radius r, for whatever it carries round.
  const turn = (r: number) => (orbit(r) * rift.rate * Math.PI) / 180
  const inOpening = (i: number, j: number) => {
    const o = rift.opening
    return (
      !!o && i >= o.i0 && i <= o.i1 && j >= o.j0 && j <= o.j1 && o.mask[j * o.grid.bw + i] === 255
    )
  }

  const pines: Cypress[] = []
  const r = mulberry(612)
  for (const [x0, x1, n] of [
    [140, 460, 4],
    [1140, 1480, 5],
  ] as const)
    for (let k = 0; k < n; k++) {
      const x = x0 + r() * (x1 - x0)
      pines.push({ x, base: midRidge(x) + 8, h: 36 + r() * 40, w: 9 + r() * 6 })
    }

  let pal = LIGHT
  let dark = false
  let lume = 0
  // The visitor's sky: where the sun and moon stand, the weather, the season, and how the land is
  // lit as a result. Recomputed on every repaint and checked once a minute.
  let al: Almanac = almanac(new Date(), site)
  let weather: Sky = readSky(site.weather)
  let times: Season = season(al, weather)
  // The bike park's day: who turns up comes from the date, so everyone looking on a given day sees
  // the same riders. With motion it has been open a while by the time anyone looks.
  const weekend = () => {
    const day = new Date(Date.now() + (site.lon / 15) * 3.6e6).getUTCDay()
    return day === 0 || day === 6
  }
  // The ski hill's the same way, with the tracks of the day so far already on its runs; its chair
  // takes two and a half minutes, so it has been running ten by the time anyone looks. Both start
  // again once the worker says where the visitor is: the guess from the time zone can be an hour
  // and more of sun time out, enough to have warmed them up shut.
  const day = () => {
    const park = makePark(daySeed(new Date(), site.lon), parkEnv(al, weather, weekend()))
    const slopes = makeHill(daySeed(new Date(), site.lon) + 1, hillEnv(al, weather, weekend()))
    warmHill(slopes)
    if (reduce) {
      posePark(park)
      poseHill(slopes)
    } else {
      const quiet: Voices = { bubbles: [], heads: {} }
      for (let n = 0; n < 1800; n++) stepPark(park, 1 / 15, quiet)
      for (let n = 0; n < 3000; n++) stepHill(slopes, 1 / 5, quiet)
    }
    return { park, slopes }
  }
  let { park, slopes } = day()
  let relief: Relief = {
    side: 1,
    contrast: 1,
    snowline: times.snowline,
    dark,
    frost: times.frost,
    thaw: times.thaw,
  }
  let rim: Rim = { side: 1, color: CREAM, contrast: 1, night: false }
  let night = false
  let lightKey = ''
  let hung = ''
  let almanacAcc = 0
  let ranges: Layer | null = null
  let ruin: Layer | null = null
  let sky: Layer | null = null
  let field: Layer | null = null
  let glint: Layer | null = null
  let well: Layer | null = null
  let leaves: Layer | null = null
  let masonry: Masonry | null = null
  let greens: Greens | null = null
  let catInk: CatInk | null = null
  let crewInk: CrewInk | null = null
  let triInk: TriInk | null = null
  let figures = new Uint8ClampedArray(0)
  let people = new Uint8ClampedArray(0)
  let landBase = new Uint8ClampedArray(0)
  let ruinBase = new Uint8ClampedArray(0)
  let skyBase = new Uint8ClampedArray(0)
  let hillBase = new Uint8ClampedArray(0)
  let ridge = new Float32Array(0)
  let ground: Meadow | null = null
  let stars: SkyStar[] = []
  let canopy: Canopy = { trees: [], roots: [] }
  let gust = 0
  let wind = 0
  let clock = 0
  let pullTarget = 0
  let drawn = false
  let riftAcc = 0
  let grown = reduce ? Infinity : 0
  let lastGlow = -1
  let lastGrowth = -1
  let minute = NaN
  let rest: Pointer = { x: -1e4, y: -1e4 }
  let stance = ''
  let lastImpulse = -Infinity
  let lastX = NaN
  let lastY = NaN
  let shed = 0
  let sign = 1

  const flush = (l: Layer) => l.ctx.putImageData(l.img, 0, 0)
  const paintRanges = () => {
    if (!ranges) return
    const d = ranges.img.data
    paintFolk(d, ranges.g, pal, landBase, clock, dark, far.heads, park, slopes)
    far.heads.athlete =
      (triInk && paintAscent(d, ranges.g, tri, triInk, clock, u => onTrail(u * TRAIL_LEN))) ||
      undefined
    paintBubbles(d, ranges.g, far, CREAM, INK)
    flush(ranges)
  }
  // The printed ruin only changes while the moss grows or the spill steps up or down.
  const paintRuin = () => {
    if (!ruin || !masonry) return
    const glow = Math.round(rift.heat * 4) / 4
    const growth = Math.min(Math.floor(grown * 8) / 8, GROWN)
    if (glow === lastGlow && growth === lastGrowth) return
    lastGlow = glow
    lastGrowth = growth
    const d = ruin.img.data
    d.set(ruinBase)
    paintSpill(d, masonry, glow, dark)
    if (greens) paintOvergrowth(d, ruin.g, overgrowth, growth, greens)
    flush(ruin)
  }
  const paintAir = () => {
    if (!sky) return
    paintSky(sky.img.data, sky.g, pal, air, skyBase, ridge, stars, clock)
    if (weather.storm && !reduce) paintBolt(sky.img.data, sky.g, clock, ridge)
    flush(sky)
  }
  const paintMeadow = () => {
    if (!field || !ground) return
    paintField(field.img.data, field.g, pal, water, ground, hillBase, clock, gust)
    flush(field)
  }
  // Water, the figures, camp and smoke. The cat, the crew and the triathlete are painted apart
  // first, so the water can reflect them with everything else as it all stands right now. The
  // swimmer lies below the waterline, where the reflection never samples, and comes in with the
  // rest of the figures over the water.
  const paintShore = () => {
    if (!glint) return
    const { g } = glint
    const d = glint.img.data
    const ink = masonry?.print.ink ?? pal.figure.K
    const same = (l: Layer | null) => (l && l.g.bw === g.bw && l.g.bh === g.bh ? l.img.data : null)
    figures.fill(0)
    if (catInk) paintCat(figures, g, cat, catInk, clock, bank)
    // The people are lit apart from the cat, which is stone and carries its own modelling.
    people.fill(0)
    if (crewInk) paintCrew(people, g, crew, crewInk, clock, rift.heat, cat.alert, bank)
    if (triInk) paintTriathlete(people, g, tri, triInk, strand, water)
    relight(people, g, rim, rift.heat)
    for (let k = 3; k < people.length; k += 4)
      if (people[k] === 255) put(figures, k - 3, [people[k - 3], people[k - 2], people[k - 1]])
    d.fill(0)
    paintPool(d, g, pal, water, clock, dark)
    const layers = [figures, same(ruin), same(field), same(well), same(ranges)]
    paintWater(d, g, pal, water, ink, layers, ripples, clock, dark)
    paintGlitter(d, g, pal, water, rift.heat, clock, air.body)
    paintRings(d, g, water, weather, clock, mix(pal.water[2], pal.glint[0], 0.6))
    paintSplashes(d, g, splashes, ink)
    for (let k = 3; k < d.length; k += 4)
      if (figures[k] === 255) put(d, k - 3, [figures[k - 3], figures[k - 2], figures[k - 1]])
    paintCamp(d, g, pal, water, walker, clock, dark)
    paintHearth(d, g, pal, hearth, clock)
    // Everyone else on the bank can be spoken to as well.
    const foot = row(g, FIRE.y)
    CAMPERS.forEach((c, n) => (crew.heads[`camper${n}`] = [col(g, c.x), foot - SITTER.length + 1]))
    crew.heads.walker = [
      col(g, walker.x),
      row(g, Math.max(848, bank(walker.x) + 8)) - FIGURE.length,
    ]
    crew.heads.athlete = headPixel(tri, g, strand) ?? undefined
    paintBubbles(d, g, crew, CREAM, INK)
    flush(glint)
  }
  // Only the opening is ever painted, and the underpainting covers all of it each frame.
  const paintPortal = () => {
    if (!well) return
    paintRift(well.img.data, well.g, rift, clock)
    if (masonry) paintHands(well.img.data, well.g, dial, masonry.print, lume)
    paintCurtain(well.img.data, well.g, curtain, grown, inOpening)
    flush(well)
  }
  const paintFront = () => {
    if (!leaves) return
    const d = leaves.img.data
    d.fill(0)
    paintCanopy(d, leaves.g, canopy, pal, wind, clock * 0.05, relief.side, times, weather)
    paintShed(
      d,
      leaves.g,
      litter,
      [pal.autumn[1], pal.autumn[2], pal.autumn[3], pal.crown[2]],
      mix(pal.snow[3], pal.poppy, 0.25),
    )
    boughs.heads = {}
    if (bough) paintBough(d, leaves.g, bough, workFor(times), boughInk(), clock, boughs.heads)
    paintDapple(d, leaves.g, pal, canopy.roots, wind * 0.6, clock, water)
    const streak = dark ? mix(pal.water[2], pal.glint[0], 0.35) : mix(pal.cloud[0], pal.far[0], 0.4)
    paintFall(d, leaves.g, weather, clock, streak, dark ? pal.snow[3] : CREAM)
    paintBubbles(d, leaves.g, boughs, CREAM, INK)
    flush(leaves)
  }
  const boughInk = (): BoughInk => ({
    skin: pal.figure.S,
    hat: pal.crew[3],
    coats: [pal.crew[1], pal.crew[2]],
    trousers: pal.figure.D,
    boots: pal.figure.K,
    harness: pal.crew[0],
    paper: CREAM,
    apple: pal.poppy,
    steel: mix(pal.rock[2], CREAM, 0.3),
    rope: pal.rope,
    bark: pal.bark[0],
    dust: pal.bark[2],
    lights: [GOLDEN, pal.poppy, pal.cornflower, CREAM],
    glow: pal.glow[1],
    steam: pal.smoke[1],
    dark,
  })
  // What comes down out of the trees: the leaves that are left in autumn, blossom in spring, and
  // whatever the arborist shakes loose. It blows off the edge of the view or falls out of the bottom.
  const shedding = (g: Grid) => ({
    leaves: times.autumn * (1 - times.bare),
    blossom: times.blossom * (1 - times.bare),
    wind: Math.min(1, weather.wind + gust * 0.2),
    shake: bough && workFor(times) === 'shake' ? bough.anchor : null,
    box: [g.sx[0], g.sy[0], g.sx[g.bw - 1], g.sy[g.bh - 1] + 10] as [
      number,
      number,
      number,
      number,
    ],
  })

  const themed = () => document.documentElement.getAttribute('saved-theme') === 'dark'
  // Everything about the light that needs the layers repainted: the palette's step through the
  // day, the twilight glow, which side the light comes from, the snowline, and the frost's retreat.
  const keyFor = (a: Almanac) => {
    const dim = themed()
    const source = a.day > 0.2 ? a.sun : a.moon
    const s = season(a, weather)
    return [
      dim,
      Math.round((dim ? a.day * 0.3 : a.day) * 16),
      Math.round(a.dusk * 8),
      source.across >= 0,
      a.day > 0.2,
      Math.round(s.snowline / 20),
      s.frost > 0.02 ? Math.round(s.thaw * 12) : -1,
    ].join('|')
  }
  // The sun by day, the moon by night if it is up, hung over the sky layer, with the sky's base
  // coat and stars under it. Cheap, so the body can move on without a full repaint.
  const hangBody = (force: boolean) => {
    if (!sky) return
    const sunUp = al.day > 0.2
    const up = sunUp ? al.sun : al.moon.alt > -3 ? al.moon : null
    const r = sunUp ? 46 : 40
    const veil = weather.fog ? 1 : smooth(0.55, 0.95, weather.cover)
    const phase = sunUp ? 0.5 : al.moon.phase
    air.body = up ? { ...hang(up, sky.g, r), r, phase, mirror: al.south, veil } : null
    air.cover = weather.cover
    const b = air.body
    const at = b ? `${col(sky.g, b.x)},${row(sky.g, b.y)}` : ''
    if (!force && at === hung) return false
    hung = at
    let floor = 0
    for (const y of ridge) floor = Math.max(floor, y)
    skyBase = new Uint8ClampedArray(sky.img.data.length)
    paintSkyBase(skyBase, sky.g, pal, floor + 30, b)
    stars = night ? makeStars(b, weather.cover) : []
    return true
  }

  const repaint = () => {
    const dim = themed()
    al = almanac(new Date(), site)
    weather = readSky(site.weather)
    times = season(al, weather)
    park.env = parkEnv(al, weather, weekend())
    slopes.env = hillEnv(al, weather, weekend())
    if (reduce) {
      posePark(park)
      poseHill(slopes)
    }
    lightKey = keyFor(al)
    const [lit, day] = lighting(al, times, weather, dim)
    pal = lit
    dark = day < 0.5
    lume = 1 - smooth(0.25, 0.6, day)
    night = al.day < 0.35
    const sunUp = al.day > 0.2
    const side = (sunUp ? al.sun : al.moon).across >= 0 ? 1 : -1
    const over = smooth(0.55, 1, weather.cover) + (weather.fog ? 0.3 : 0)
    const moonlit = 1 - Math.abs(al.moon.phase - 0.5) * 2
    const contrast = Math.max(0.25, 1 - 0.65 * over) * (sunUp ? 1 : 0.35 + 0.4 * moonlit)
    relief = {
      side,
      contrast,
      snowline: times.snowline,
      dark,
      frost: times.frost,
      thaw: times.thaw,
    }
    rim = { side, color: mix(pal.sun[0], CREAM, 0.5), contrast, night: !sunUp }
    ranges = layer(back)
    if (ranges) {
      paintLand(ranges.img.data, ranges.g, pal, pines, relief)
      landBase = ranges.img.data.slice()
    }
    ruin = layer(grain)
    // A light page's stones stay paper-coloured by day and go grey under the moon.
    const stone = dim ? 1 : 0.5 + 0.5 * day
    masonry = ruin && rasterRuin(ruin.img.data, ruin.g, temple, dim, bank, side, stone)
    if (ruin) ruinBase = ruin.img.data.slice()
    if (masonry) {
      const { print } = masonry
      const tones = (stops: RGB[]): Tones => ({
        ink: print.ink,
        light: stops[stops.length - 1],
        plate: stops[stops.length - 2],
        shade: stops[stops.length - 3],
      })
      const flowers = [print.rose[3], CREAM]
      greens = {
        moss: tones(pal.moss),
        bark: tones(pal.bark),
        // The small tree on the gate turns early, being exposed up there.
        crown: tones(times.autumn > 0.45 ? pal.autumn : pal.crown),
        flowers,
        hair: pal.bark[dark ? 2 : 0],
      }
      catInk = {
        stone: tones(print.stone),
        moss: tones(pal.moss),
        eye: GOLDEN,
        nose: print.rose[2],
        ear: print.rose[dark ? 1 : 2],
        flowers,
      }
    }
    crewInk = {
      skin: pal.figure.S,
      trousers: pal.figure.D,
      boots: pal.figure.K,
      hat: pal.ochre,
      coats: pal.crew,
      wood: pal.rope,
      paper: pal.figure.C,
      gold: GOLDEN,
      rose: ROSE,
      fish: pal.fish,
      flame: FLAME[2],
      glow: pal.glow[2],
      ink: INK,
      dark,
    }
    triInk = {
      skin: pal.figure.S,
      hair: pal.bark[1],
      shoes: pal.figure.K,
      paper: pal.figure.C,
      suits: pal.crew,
      caps: [FLAME[2], GOLDEN, ROSE, CREAM],
      disc: mix(pal.figure.K, pal.figure.C, 0.45),
      rack: pal.rope,
      deep: mix(pal.water[0], INK, dark ? 0.5 : 0.3),
      foam: pal.glint[0],
      wake: dark ? mix(pal.water[2], pal.glint[0], 0.4) : mix(pal.water[0], INK, 0.25),
      flag: pal.poppy,
      dust: mix(pal.mid[2], CREAM, dark ? 0.3 : 0.6),
    }
    lastGlow = lastGrowth = -1
    sky = layer(skyCanvas)
    if (sky) {
      ridge = Float32Array.from(sky.g.sx, skyline)
      hangBody(true)
    }
    field = layer(hill)
    if (field) {
      ground = meadow(field.g, water, relief, times)
      hillBase = new Uint8ClampedArray(field.img.data.length)
      paintHillBase(hillBase, field.g, pal, water, ground)
    }
    glint = layer(near)
    if (glint) {
      figures = new Uint8ClampedArray(glint.img.data.length)
      people = new Uint8ClampedArray(glint.img.data.length)
      // The mountain legs run only when the flag on the crest is on screen.
      const { g } = glint
      camp = campSpan(g.px)
      layCourse(tri, g, water, camp, TRAIL[TRAIL.length - 1][0] + 8 * g.px < g.sx[g.bw - 1] - 48)
    }
    well = layer(riftCanvas)
    // The tree folk's heads and speech are in the old grid's pixels, and a resize may leave no one
    // in the tree to hail.
    bough = null
    boughs.heads = {}
    boughs.bubbles = []
    leaves = layer(front)
    if (leaves) {
      // Repoussoir trees hug whatever edges the slice fit leaves visible.
      const left = leaves.g.sx[0]
      const span = leaves.g.sx[leaves.g.bw - 1] - left
      const roots = [left + span * 0.04, left + span * 0.97]
      const k = Math.max(0.45, Math.min(1, span / 1500))
      canopy = {
        roots,
        trees: [
          growTree(4040, roots[0], 1080, -0.04, 46 * k, 90 * k),
          growTree(4041, roots[1], 1080, 0.02, 36 * k, 80 * k),
        ],
      }
      // The layers bleed past the viewport for the parallax, so the tree folk are placed within
      // what the viewport shows; and the departures board sits over the top-left corner, so nobody
      // climbs up behind it.
      const g = leaves.g
      const fr = front!.getBoundingClientRect()
      const toScene = (x: number, y: number): Pt => [
        g.sx[0] + ((x - fr.left) / fr.width) * g.bw * g.px,
        g.sy[0] + ((y - fr.top) / fr.height) * g.bh * g.px,
      ]
      const vr = front!.parentElement!.getBoundingClientRect()
      const [x0, y0] = toScene(vr.left, vr.top)
      const [x1, y1] = toScene(vr.right, vr.bottom)
      const board = scene.querySelector('.nf-board')?.getBoundingClientRect()
      const edge = board && toScene(board.right + 4, board.bottom + 4)
      bough = findBough(
        canopy.trees[0],
        g,
        [x0, y0, x1, y1],
        (x, y) => !!edge && x < edge[0] && y < edge[1],
      )
      if (!litter.leaves.length)
        for (let n = 0; n < 120; n++) stepShed(litter, canopy.trees, shedding(g), 1 / 12)
    }
    paintRanges()
    paintRuin()
    paintAir()
    paintMeadow()
    paintPortal()
    paintShore()
    paintFront()
  }
  repaint()
  // Once a minute: a repaint when the light has moved on far enough to change the palette or the
  // modelling, otherwise only the sun or the moon creeping along its arc.
  const recheck = () => {
    const next = almanac(new Date(), site)
    // The park and the lift open and close on their hours whether or not the light has changed,
    // and the groomer goes out on its own.
    const env = parkEnv(next, weather, weekend())
    const snow = hillEnv(next, weather, weekend())
    const shut =
      env.open !== park.env.open ||
      env.crowd !== park.env.crowd ||
      snow.open !== slopes.env.open ||
      snow.groom !== slopes.env.groom ||
      snow.tour !== slopes.env.tour
    park.env = env
    slopes.env = snow
    if (keyFor(next) !== lightKey) return repaint()
    if (shut && reduce) {
      posePark(park)
      poseHill(slopes)
      paintRanges()
    }
    al = next
    if (hangBody(false)) paintAir()
  }
  window.addEventListener('resize', repaint)
  document.addEventListener('themechange', repaint)

  // At most one stop-motion layer repaints per animation frame, whichever is most overdue, so the
  // ticks interleave instead of landing on the same frame.
  const jobs = [
    { run: paintAir, every: 1 / 8, acc: 0 },
    { run: paintMeadow, every: TICK, acc: 0.03 },
    { run: paintShore, every: TICK, acc: 0.06 },
    { run: paintFront, every: TICK, acc: 0.09 },
    { run: paintRanges, every: TICK, acc: 0.05 },
  ]

  return {
    step(dt, now, pointer) {
      if (reduce) return
      clock = now
      const inside = pointer.x > -1e3
      const mx = inside && !Number.isNaN(lastX) ? pointer.x - lastX : 0
      const my = inside && !Number.isNaN(lastY) ? pointer.y - lastY : 0
      const moved = Math.hypot(mx, my)
      gust = Math.min(3, gust * Math.exp(-dt * 1.2) + Math.min(Math.abs(mx), 200) * 0.004)
      wind += dt * (0.04 + 0.14 * weather.wind + gust * 0.25)
      if ((almanacAcc += dt) > 60) {
        almanacAcc = 0
        recheck()
      }
      // The cursor sheds eddies into the sky, alternating in sign on either side of its path: a
      // Kármán street, one vortex per stride.
      if (inside && pointer.y < 640 && moved / dt > 120) {
        shed += moved
        if (shed > 70) {
          shed = 0
          sign = -sign
          air.vortices.push({
            x: pointer.x - (my / moved) * 16 * sign,
            y: pointer.y + (mx / moved) * 16 * sign,
            gamma: sign * Math.min(moved / dt, 1400) * 32,
            age: 0,
          })
          if (air.vortices.length > 8) air.vortices.shift()
        }
      }
      lastX = inside ? pointer.x : NaN
      lastY = inside ? pointer.y : NaN
      for (const v of air.vortices) {
        v.age += dt
        v.gamma *= Math.exp(-dt / 2.4)
        v.x += 10 * dt
      }
      air.vortices = air.vortices.filter(v => Math.abs(v.gamma) > 600)
      air.drift += 10 * dt * (0.6 + 1.6 * weather.wind)
      air.time = now
      air.pull += (pullTarget - air.pull) * Math.min(1, dt * 1.5)

      // The moss waits for the reveal.
      if (scene.classList.contains('is-ready')) grown += dt
      // The clock's minute impulse, heard at most every 0.8 s while time is warped.
      const m = Math.round(dial.minute / 6)
      const impulse = !Number.isNaN(minute) && m !== minute && now - lastImpulse > 0.8
      if (impulse) lastImpulse = now
      minute = m
      stepRift(rift, dt)
      stepCurtain(curtain, dt, now, rift.heat, rift.inflow, turn)
      // The cat watches whichever is nearer, the pointer or the triathlete going by.
      stepTriathlete(tri, dt)
      const seen = sighting(tri, strand)
      const [hx, hy] = catHead(cat)
      const away = (p: Pointer) => Math.hypot(p.x - hx, p.y - hy)
      stepCat(cat, dt, seen && away(seen) < away(pointer) ? seen : pointer, rift.heat, impulse)
      stepCrew(crew, dt, { clock: now, heat: rift.heat, inflow: rift.inflow, impulse, turn })
      stepPark(park, dt, far)
      stepHill(slopes, dt, far)
      ageBubbles(far, dt)
      ageBubbles(boughs, dt)
      if (leaves) stepShed(litter, canopy.trees, shedding(leaves.g), dt)
      stepWalker(walker, dt, drawn)
      stepHearth(hearth, air, dt)
      for (let n = splashes.length - 1; n >= 0; n--)
        if ((splashes[n].age += dt) > 1.6) splashes.splice(n, 1)
      paintRuin()
      riftAcc += dt
      if (riftAcc >= 1 / 20) {
        riftAcc = 0
        paintPortal()
      }
      let due: (typeof jobs)[number] | null = null
      for (const job of jobs) {
        job.acc += dt
        if (job.acc >= job.every && (!due || job.acc / job.every > due.acc / due.every)) due = job
      }
      if (due) {
        due.acc = 0
        due.run()
      }
    },
    portal(mode) {
      rift.target = mode === 'enter' ? 90 : mode === 'hot' ? 26 : 1.1
      rift.inflowTarget = mode === 'enter' ? 1.8 : mode === 'hot' ? 0.22 : 0
      pullTarget = mode === 'enter' ? 2.5 : mode === 'hot' ? 1 : 0
      drawn = mode !== 'idle'
      if (reduce) {
        rift.heat = drawn ? 1 : 0
        poseCat(cat, rest, drawn)
        poseCrew(crew, drawn)
        stance = `${cat.alert}${cat.look}`
        paintRuin()
        paintPortal()
        paintShore()
      }
    },
    splash(x, y, size) {
      splashes.push({ x, y, size, age: 0 })
    },
    still(pointer) {
      rest = pointer
      poseCat(cat, pointer, drawn)
      const next = `${cat.alert}${cat.look}`
      if (next === stance) return
      stance = next
      paintShore()
    },
    refresh() {
      if ((almanacAcc += 1) >= 60) {
        almanacAcc = 0
        recheck()
      }
      paintPortal()
      paintShore()
    },
    place(next) {
      site = next
      al = almanac(new Date(), site)
      weather = readSky(site.weather)
      ;({ park, slopes } = day())
      repaint()
    },
    hail(x, y) {
      // Client pixels into each layer's buffer; the layers drift apart under parallax.
      const at = (l: Layer | null, el: HTMLCanvasElement | null) => {
        if (!l || !el) return null
        const r = el.getBoundingClientRect()
        return [((x - r.left) / r.width) * l.g.bw, ((y - r.top) / r.height) * l.g.bh] as const
      }
      const c = at(leaves, front)
      const n = at(glint, near)
      const f = at(ranges, back)
      const tree = c && heard(boughs, c[0], c[1], 3, 9)
      const bank = !tree && n ? heard(crew, n[0], n[1], 3, 10) : null
      const ridge = !tree && !bank && f ? heard(far, f[0], f[1], 2, 4) : null
      if (tree) ask(boughs, tree)
      else if (bank) ask(crew, bank)
      else if (ridge) ask(far, ridge)
      else return false
      paintFront()
      paintShore()
      paintRanges()
      // Without motion nothing ages the bubble, so it is taken down on a timer.
      if (reduce) {
        window.clearTimeout(hush)
        hush = window.setTimeout(() => {
          for (const v of [crew, far, boughs]) v.bubbles = v.bubbles.filter(b => b.glyph !== '?')
          paintFront()
          paintShore()
          paintRanges()
        }, 1800)
      }
      return true
    },
    pixel() {
      const g = glint?.g
      return g ? { px: g.px, x0: g.sx[0] - g.px / 2, y0: g.sy[0] - g.px / 2 } : null
    },
    dispose() {
      window.clearTimeout(hush)
      window.removeEventListener('resize', repaint)
      document.removeEventListener('themechange', repaint)
    },
  }
}
