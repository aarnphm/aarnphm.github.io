// Pixel landscape for the 404 ruin, painted the way Van Gogh painted the factories at Clichy:
// low-resolution canvases upscaled with `image-rendering: pixelated`, an ordered-dither
// underpainting, then short brush strokes laid along flow fields. Each stroke picks its ramp stop
// once, so the dither happens per stroke instead of per pixel. The ruin itself is printed into the
// same grid from the hidden SVG geometry: plates, one-pixel keylines, dial marks and chips, with
// the moss grown over it as blobs (404-overgrowth). The cat and the survey party (404-cat,
// 404-crew) stand in the same canvas as the water, which reflects them along with the ruin, the
// hill and the ranges. The ranges and the ruin paint once per size or theme; sky, field, water,
// camp, canopy and portal repaint in stop-motion ticks.

import { type Cat, type CatInk, makeCat, paintCat, poseCat, stepCat } from './404-cat'
import { type CrewInk, makeCrew, paintBubbles, paintCrew, poseCrew, stepCrew } from './404-crew'
import { GATE, MISSING, STEP_TOP, hourAngle, stoneOuter } from './404-gate'
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
  dot,
  fbm,
  frac,
  grid,
  hash,
  hex,
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
  // Reduced motion: poses the cat and the crew for where the pointer is, repainting only on a change.
  still(pointer: Pointer): void
  // Buffer pixel pitch and origin in scene units, for snapping vector sprites onto the grid.
  pixel(): { px: number; x0: number; y0: number } | null
  dispose(): void
}

const TICK = 1 / 10
// Gold only where the tone is mid-range and the gradient is sharp: fringes, never interiors.
const GOLD_START = 0.45
const GOLD_END = 0.7
const SUN = { x: 1150, y: 118, r: 46 }

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

// A flat-crowned knoll: the temple sits on the plateau, the shoulders run down into the water.
export function hillTop(x: number, water: number) {
  const u = Math.max(-1, Math.min(1, (x - 800) / 660))
  const bell = (0.5 + 0.5 * Math.cos(Math.PI * u)) ** 0.55
  return water - 175 * bell + (noise(x * 0.02, 1.3, 7) - 0.5) * 12 * bell
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
type Air = { drift: number; time: number; pull: number; pole: Pt; vortices: Vortex[] }

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
  vortex(x - SUN.x, y - SUN.y, 42000, 70, 0)
  for (const v of air.vortices) vortex(x - v.x, y - v.y, v.gamma, 40 + Math.sqrt(v.age) * 30, 0)
  if (air.pull > 0.01)
    vortex(x - air.pole[0], y - air.pole[1], air.pull * 160000, 240, air.pull * 70000)
  return Math.atan2(V.y, V.x)
}

// What a sky stroke carries: the sun, its banded halo, drifting smoke, or the sky gradient.
function skyInk(pal: Palette, x: number, y: number, drift: number, thr: number, mask: number) {
  const r = Math.hypot(x - SUN.x, y - SUN.y)
  if (r < SUN.r) return pick(pal.sun, 0.7 + 0.3 * (1 - r / SUN.r), thr)
  if (r < SUN.r + 70) {
    const fall = 1 - (r - SUN.r) / 70
    const band = 0.5 + 0.5 * Math.cos((r - SUN.r) * 0.21)
    if (band * fall > 0.3 + mask * 0.25) return pick(pal.sun, fall * 0.9, thr)
  }
  const c = fbm((x - drift) * 0.0035, y * 0.012, 131, 3)
  const cloud = smooth(0.5, 0.68, c) * (1 - smooth(230, 480, y))
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

function paintSkyBase(d: Uint8ClampedArray, g: Grid, pal: Palette, floor: number) {
  const { bw, bh, sx, sy } = g
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    if (y > floor) break
    for (let i = 0; i < bw; i++) {
      const r = Math.hypot(sx[i] - SUN.x, y - SUN.y)
      const k = (j * bw + i) * 4
      if (r < SUN.r) put(d, k, pal.sun[2])
      else if (r < SUN.r + 18) put(d, k, ramp(pal.sun, 1 - (r - SUN.r) / 18, i, j))
      else put(d, k, ramp(pal.sky, smooth(60, 520, y), i, j))
    }
  }
}

type SkyStar = { x: number; y: number; mag: number; freq: number; phase: number }

function makeStars(): SkyStar[] {
  const r = mulberry(77)
  const stars: SkyStar[] = []
  for (let n = 0; n < 80 && stars.length < 18; n++) {
    const x = 40 + r() * 1520
    const y = 30 + r() * 420
    const star = { x, y, mag: r() ** 1.6, freq: 0.6 + r() * 1.6, phase: r() * Math.PI * 2 }
    if (y < skyline(x) - 50 && Math.hypot(x - SUN.x, y - SUN.y) > 130) stars.push(star)
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
  // The disc stays whole: only the sun's own strokes may cross into it.
  const clear: Keep = (i, j) => Math.hypot(g.sx[i] - SUN.x, g.sy[j] - SUN.y) > SUN.r
  seeds(g, 3, air.drift / g.px, [0, row(g, floor)], 131, (bx, by, h, k) => {
    const x = x0 + bx * g.px
    const y = y0 + by * g.px
    if (y > ridge[clampCol(g, bx)] + g.px * 2) return
    const c = eddyInk(pal, air, x, y, k) ?? skyInk(pal, x, y, air.drift, h, k)
    const len = 6 + Math.floor(frac(k * 7.13) * 5)
    const near = Math.hypot(x - SUN.x, y - SUN.y) < SUN.r + len * g.px
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

// The ranges, laid in strokes that follow each ridge's profile and relax toward level deeper in
// the face, the way the Alpilles are combed in the Saint-Rémy canvases. The sky stays clear.
function paintLand(d: Uint8ClampedArray, g: Grid, pal: Palette, pines: Cypress[]) {
  const { bw, bh, sx, sy } = g
  const far = Float32Array.from(sx, farRidge)
  const mid = Float32Array.from(sx, midRidge)
  const slope = (f: (x: number) => number, x: number, h: number) => (f(x + h) - f(x - h)) / (2 * h)
  const farLit = Float32Array.from(sx, x => slope(farRidge, x, 8))
  const midLit = Float32Array.from(sx, x => slope(midRidge, x, 8))
  const farAng = Float32Array.from(sx, x => Math.atan(slope(farRidge, x, 24)))
  const midAng = Float32Array.from(sx, x => Math.atan(slope(midRidge, x, 24)))
  const farTone = (i: number, y: number) =>
    0.45 +
    Math.max(-1, Math.min(1, farLit[i] * 1.6)) * 0.28 +
    (fbm(sx[i] * 0.009, y * 0.003, 53) - 0.5) * 0.35 +
    smooth(30, 360, y - far[i]) * 0.4
  const midTone = (i: number, y: number) =>
    0.5 +
    Math.max(-1, Math.min(1, midLit[i] * 1.6)) * 0.3 +
    (fbm(sx[i] * 0.012, y * 0.0035, 41) - 0.5) * 0.45 +
    smooth(40, 320, y - mid[i]) * 0.28

  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    for (let i = 0; i < bw; i++) {
      const k = (j * bw + i) * 4
      // Each crest is drawn as a contour, so where the mid ridge passes in front of the far one the
      // overlap shows as a line. Sun-facing slopes descend to the right; their crest catches gold.
      if (y >= mid[i]) {
        if (y - mid[i] < g.px) put(d, k, midLit[i] > 0.15 ? pal.gold : pal.contour[1])
        else put(d, k, ramp(pal.mid, midTone(i, y), i, j))
      } else if (y >= far[i])
        put(d, k, y - far[i] < g.px ? pal.contour[0] : ramp(pal.far, farTone(i, y), i, j))
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
    const ink = pick(pal.far, farTone(i, y), h)
    brush(d, g, bx, by, 4 + Math.floor(frac(k * 9.1) * 4), false, ink, angle, inFar)
  })
  ridge = mid
  ang = midAng
  seeds(g, 3, 0, [row(g, top), bh], 143, (bx, by, h, k) => {
    const i = clampCol(g, bx)
    const y = y0 + by * g.px
    if (y < mid[i] + g.px * 1.5 || y > 900) return
    jit = (k - 0.5) * 0.5
    const ink = pick(pal.mid, midTone(i, y), h)
    brush(d, g, bx, by, 4 + Math.floor(frac(k * 9.1) * 5), frac(k * 4.3) < 0.4, ink, angle, inMid)
  })
  pines.forEach((c, n) => paintCypress(d, g, c, pal.pine, 300 + n))
}

// People on the ranges, a pixel wide and three tall: two beside the cypress on the left ridge, one
// of them waving; someone with a lantern on the right ridge; and another party's fire on the far
// range, which by day is only a thread of smoke.
const FOLK = [
  { x: 246, wave: false, lamp: false },
  { x: 254, wave: true, lamp: false },
  { x: 1318, wave: false, lamp: true },
]
const CAMP = 1046

function paintFolk(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  base: Uint8ClampedArray,
  clock: number,
  dark: boolean,
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
  for (const f of FOLK) {
    const i = col(g, f.x)
    const top = ground(i, midRidge)
    if (f.lamp && dark) glow(i + 1, top - 2, hash(beat, 3, 41))
    for (let k = 1; k <= 3; k++) dot(d, g, i, top - k, pal.figure.K)
    if (f.wave) dot(d, g, i + 1, top - (Math.floor(clock / 0.45) % 2 ? 4 : 3), pal.figure.K)
    if (f.lamp) dot(d, g, i + 1, top - 2, hash(beat, 3, 41) < 0.3 ? FLAME[2] : FLAME[1])
  }
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
}

const HILL_CYPRESSES = [
  { x: 420, h: 250, w: 46 },
  { x: 1195, h: 215, w: 40 },
]

type Meadow = { top: Float32Array; lit: Float32Array; ang: Float32Array; crest: number }

function meadow(g: Grid, water: number): Meadow {
  const top = Float32Array.from(g.sx, x => hillTop(x, water))
  let crest = Infinity
  for (const y of top) crest = Math.min(crest, y)
  return {
    top,
    lit: Float32Array.from(g.sx, x => (hillTop(x + 8, water) - hillTop(x - 8, water)) / 16),
    ang: Float32Array.from(g.sx, x =>
      Math.atan((hillTop(x + 24, water) - hillTop(x - 24, water)) / 48),
    ),
    crest,
  }
}

function hillTone(x: number, y: number, lit: number, water: number) {
  let t = 0.52 + Math.max(-1, Math.min(1, lit * 2.2)) * 0.28
  t += (fbm(x * 0.018, y * 0.07, 61) - 0.5) * 0.5
  // The front face under the steps sits in their shadow.
  if (x > 460 && x < 1140) t -= 0.35 * (1 - smooth(790, 812, y))
  return t - smooth(water - 16, water, y) * 0.2
}

function paintHillBase(d: Uint8ClampedArray, g: Grid, pal: Palette, water: number, m: Meadow) {
  const { bw, bh, sx, sy } = g
  for (let j = 0; j < bh; j++) {
    const y = sy[j]
    if (y > water + g.px) break
    for (let i = 0; i < bw; i++) {
      if (y < m.top[i]) continue
      const k = (j * bw + i) * 4
      if (y - m.top[i] < g.px) put(d, k, m.lit[i] > 0.08 ? pal.gold : pal.contour[2])
      else put(d, k, ramp(pal.hill, hillTone(sx[i], y, m.lit[i], water), i, j))
    }
  }
}

// The field in short dashes that follow the slope and flatten toward the bank; a travelling
// wave leans them as the wind crosses, with the odd poppy, cornflower and ochre dab.
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
    // Behind the steps and the wall: never seen.
    if (Math.abs(x - 800) < 335 && y < 792) return
    const t = hillTone(x, y, m.lit[i], water)
    const roll = frac(k * 5.31)
    const ink =
      t > 0.42 && roll < 0.016
        ? pal.poppy
        : t > 0.42 && roll < 0.03
          ? pal.cornflower
          : t > 0.5 && roll < 0.06
            ? pal.ochre
            : pick(pal.hill, t, h)
    jit = (frac(k * 9.73) - 0.5) * 0.7
    brush(d, g, bx, by, 3 + Math.floor(frac(k * 3.37) * 3), frac(k * 11.3) < 0.65, ink, angle)
  })
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
) {
  const { bw, bh, sx, sy } = g
  for (let j = Math.max(0, row(g, water) + 1); j < bh; j += 2) {
    const depth = sy[j] - water
    const ws = 14 + depth * 0.9
    const wp = 20 + depth * 0.5
    for (let i = (j >> 1) % 3; i < bw; i += 3) {
      const ps = 0.42 * Math.exp(-(((sx[i] - SUN.x) / ws) ** 2))
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

// The camp on the bank left of the steps: a rose tent, a ring of hearth stones round two logs, a
// fire in stop-motion, two people sitting either side of it (one toasting something on a stick),
// and the walker. Everything within reach of the fire takes its colour on the side facing it.
const FIRE = { x: 628, y: 848 }
const TENT = ['....S....', '...SSL...', '..SSSLL..', '.SSSLKLL.', 'SSSSLKKLL']
const SITTER = ['.CC.', '.CS.', 'BBB.', 'BBBS', 'DDDD']
const CAMPERS = [
  { x: 600, flip: false },
  { x: 658, flip: true },
]
const FLAME_ROWS = [2, 4, 7, 5, 2]

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
    col(g, 556),
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

// Stone tone at a scene point: 0.5 is the flat plate, each 0.25 is one dither stop. The sun sits
// up and to the right. Every voussoir is chamfered, lit on its sunward edges and shaded on the rest,
// the soffit under the ring is in deep shade, and the plinth and treads catch light on their noses.
function stoneTone(x: number, y: number) {
  const dx = x - GATE.x
  const dy = y - GATE.y
  const r = Math.hypot(dx, dy)
  if (r < GATE.r) return 0.08 + 0.3 * (dx / GATE.r) ** 2
  const a = Math.atan2(dy, dx)
  const h = (Math.round((a * 6) / Math.PI) + 15) % 12
  const out = stoneOuter(h)
  if (r <= out + 2 && y < 656) {
    const u = (r - GATE.r) / (out - GATE.r)
    const v = (Math.atan2(Math.sin(a - hourAngle(h)), Math.cos(a - hourAngle(h))) * 12) / Math.PI
    const eu = 8 / (out - GATE.r)
    const ev = (8 * 12) / (Math.PI * r)
    const cu = u > 1 - eu ? 1 : u < eu ? -1 : 0
    const cv = v > 1 - ev ? 1 : v < ev - 1 ? -1 : 0
    const nx = Math.cos(a) * cu - Math.sin(a) * cv
    const ny = Math.sin(a) * cu + Math.cos(a) * cv
    return 0.5 + 0.45 * (nx * 0.6 - ny * 0.8)
  }
  if (y >= 656 && y < STEP_TOP && Math.abs(dx) <= 110) {
    const v = y < 672 ? y - 656 : y - 672
    return v < 5 ? 0.92 : v > (y < 672 ? 11 : 13) ? 0.3 : 0.5
  }
  const tread = Math.floor((y - STEP_TOP) / 25)
  if (tread >= 0 && tread < 4 && Math.abs(dx) <= 180 + 50 * tread) {
    const v = (y - STEP_TOP) % 25
    return v < 7 ? 0.76 : v > 15 ? 0.3 : 0.5
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
type Masonry = { print: Print; spill: number[]; treads: number[] }

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
// wherever an `nf-void` says, inked where they break into stone. A stone that has slipped is lit
// in its own place in the ring, so its chamfers travel with it. The light that spills down the
// steps animates on top of this, so its pixels are returned as a list.
function rasterRuin(d: Uint8ClampedArray, g: Grid, svg: SVGSVGElement, dark: boolean) {
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
      const base = probe(getComputedStyle(el).fill)
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
  // Which frame lights each plate pixel: 0 the scene, n a slipped stone's place before it slipped.
  const frame = new Uint8Array(bw * bh)
  const frames: DOMMatrix[] = [new DOMMatrix()]
  const frameOf = new Map<Element, number>()
  const frameFor = (el: Element) => {
    const stone = el.closest<SVGGElement>('.nf-stone')
    const m = stone?.transform.baseVal.consolidate()?.matrix
    if (!stone || !m) return 0
    let f = frameOf.get(stone)
    if (f === undefined) {
      f = frames.push(m.inverse()) - 1
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
  const treads: number[] = []
  for (let j = 0; j < bh; j++)
    for (let i = 0; i < bw; i++) {
      const p = j * bw + i
      const o = owner[p]
      if (o === INK) put(d, p * 4, ink)
      if (o === 0 || o === INK) continue
      const x = g.sx[i]
      const y = g.sy[j]
      const f = frames[frame[p]]
      const lx = f.a * x + f.c * y + f.e
      const ly = f.b * x + f.d * y + f.f
      put(d, p * 4, stops[o - 1][PLATE[step(5, stoneTone(lx, ly) * 0.75 + 0.125, i, j)]])
      const tread = Math.floor((y - STEP_TOP) / 25)
      if (o === 1 && tread >= 0 && tread < 4 && Math.abs(x - GATE.x) <= 100 + 50 * tread) {
        spill.push(p)
        treads.push(tread)
      }
    }

  const print = { ink, stone: stops[0], rose: stops[1], sage: stops[2] }
  return { print, spill, treads } satisfies Masonry
}

// Rose light from the open gate falls down the steps as a flat wash, one step weaker on every
// tread: multiplied into the stone by day, screened over it by night.
function paintSpill(d: Uint8ClampedArray, m: Masonry, glow: number, dark: boolean) {
  if (glow <= 0) return
  const rose = m.print.rose[2]
  for (let s = 0; s < m.spill.length; s++) {
    const k = m.spill[s] * 4
    const a = glow * (0.5 - 0.1 * m.treads[s])
    for (let c = 0; c < 3; c++) {
      const lit = dark
        ? 255 - ((255 - d[k + c]) * (255 - rose[c])) / 255
        : (d[k + c] * rose[c]) / 255
      d[k + c] += (lit - d[k + c]) * a
    }
  }
}

const HUB = 15

// Station-clock hands on the pole star's arbor, laid into the rift pixel by pixel. The hour and
// minute hands are bars lit on their sunward edge; the red second hand is a pixel wide with its
// disc; the hub is an open ring so the star shows through. The rift is night in either theme, so
// the hands take whichever of stone and ink is the lighter.
function paintHands(d: Uint8ClampedArray, g: Grid, dial: Dial, print: Print, dark: boolean) {
  const { bw, bh, sx, sy } = g
  const face = dark ? print.ink : print.stone[3]
  const shade = dark ? mix(print.ink, RIFT[3], 0.4) : print.stone[2]
  const red = print.rose[dark ? 3 : 2]
  const bars = [
    { deg: dial.hour, w: 14, len: 112, tail: 40, lit: face },
    { deg: dial.minute, w: 10, len: 150, tail: 44, lit: face },
    { deg: dial.second, w: 5, len: 102, tail: 52, lit: red },
  ].map(b => {
    const a = (b.deg * Math.PI) / 180
    const ux = Math.sin(a)
    const uy = -Math.cos(a)
    // Which side of the bar faces the sun, up and to the right.
    const sun = -uy * 0.6 - ux * 0.8 >= 0 ? 1 : -1
    return { ...b, ux, uy, sun, half: Math.max(b.w / 2, g.px / 2), edged: b.w > g.px * 2 }
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
        if (u > b.len || u < -b.tail || Math.abs(u) < HUB) continue
        const v = dy * b.ux - dx * b.uy
        if (Math.abs(v) > b.half) continue
        c = b.edged && v * b.sun < g.px - b.half ? (b.lit === red ? red : shade) : b.lit
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
) {
  const { bw, bh, sx, sy } = g
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
      // Twigs thinner than a pixel stay hidden in the leaves instead of stair-stepping across it.
      if (s.w0 < g.px * 0.8) continue
      const pad = Math.max(s.w0, s.w1) / 2 + g.px
      const i0 = Math.max(0, toI(Math.min(s.x0, s.x1) - pad))
      const i1 = Math.min(bw - 1, toI(Math.max(s.x0, s.x1) + pad))
      const j0 = Math.max(0, toJ(Math.min(s.y0, s.y1) - pad))
      const j1 = Math.min(bh - 1, toJ(Math.max(s.y0, s.y1) + pad))
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const [dist, w, side] = segHit(s, sx[i], sy[j])
          const half = Math.max(w / 2, g.px * 0.5)
          if (dist <= half)
            put(d, (j * bw + i) * 4, ramp(pal.bark, 0.5 + (side / half) * 0.5, i, j))
        }
    }
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
    const v = density[p] * (0.35 + n) * (1 + 0.15 * z)
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
      // The sun sits up and to the right: only the fringe facing it catches gold.
      const sunward = cover[p - bw + 1] < t
      if (t >= GOLD_START && t <= GOLD_END && grad > 0.3 && sunward) put(d, k, pal.gold)
      else if (t < 0.6 && !sunward) put(d, k, pal.leaf[0])
      else {
        const tone = 0.2 + fbm(sx[i] * 0.05, sy[j] * 0.05, 83, 2) * 0.5 + depth[p] * 0.18
        put(d, k, ramp(pal.leaf, tone + (sunward ? 0.12 : 0), i, j))
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
  // Shade lands only on the near ground: the hill and the temple's lower courses.
  const floor = Float32Array.from(sx, x => (Math.abs(x - 800) < 330 ? 600 : hillTop(x, water) - 6))
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
  const gap = hourAngle(MISSING)
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
  const air: Air = { drift: 0, time: 0, pull: 0, pole, vortices: [] }
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
  let figures = new Uint8ClampedArray(0)
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
    paintFolk(ranges.img.data, ranges.g, pal, landBase, clock, dark)
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
    flush(sky)
  }
  const paintMeadow = () => {
    if (!field || !ground) return
    paintField(field.img.data, field.g, pal, water, ground, hillBase, clock, gust)
    flush(field)
  }
  // Water, the cat and the crew, camp and smoke. The cat and the crew are painted apart first, so
  // the water can reflect them with everything else as it all stands right now.
  const paintShore = () => {
    if (!glint) return
    const { g } = glint
    const d = glint.img.data
    const ink = masonry?.print.ink ?? pal.figure.K
    const same = (l: Layer | null) => (l && l.g.bw === g.bw && l.g.bh === g.bh ? l.img.data : null)
    figures.fill(0)
    if (catInk) paintCat(figures, g, cat, catInk, clock, bank)
    if (crewInk) paintCrew(figures, g, crew, crewInk, clock, rift.heat, cat.alert, bank)
    d.fill(0)
    paintPool(d, g, pal, water, clock, dark)
    const layers = [figures, same(ruin), same(field), same(well), same(ranges)]
    paintWater(d, g, pal, water, ink, layers, ripples, clock, dark)
    paintGlitter(d, g, pal, water, rift.heat, clock)
    paintSplashes(d, g, splashes, ink)
    for (let k = 3; k < d.length; k += 4)
      if (figures[k] === 255) put(d, k - 3, [figures[k - 3], figures[k - 2], figures[k - 1]])
    paintCamp(d, g, pal, water, walker, clock, dark)
    paintHearth(d, g, pal, hearth, clock)
    paintBubbles(d, g, crew, CREAM, INK)
    flush(glint)
  }
  // Only the opening is ever painted, and the underpainting covers all of it each frame.
  const paintPortal = () => {
    if (!well) return
    paintRift(well.img.data, well.g, rift, clock)
    if (masonry) paintHands(well.img.data, well.g, dial, masonry.print, dark)
    paintCurtain(well.img.data, well.g, curtain, grown, inOpening)
    flush(well)
  }
  const paintFront = () => {
    if (!leaves) return
    const d = leaves.img.data
    d.fill(0)
    paintCanopy(d, leaves.g, canopy, pal, wind, clock * 0.05)
    paintDapple(d, leaves.g, pal, canopy.roots, wind * 0.6, clock, water)
    flush(leaves)
  }

  const repaint = () => {
    dark = document.documentElement.getAttribute('saved-theme') === 'dark'
    pal = dark ? DARK : LIGHT
    ranges = layer(back)
    if (ranges) {
      paintLand(ranges.img.data, ranges.g, pal, pines)
      landBase = ranges.img.data.slice()
    }
    ruin = layer(grain)
    masonry = ruin && rasterRuin(ruin.img.data, ruin.g, temple, dark)
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
        crown: tones(pal.leaf),
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
    lastGlow = lastGrowth = -1
    sky = layer(skyCanvas)
    if (sky) {
      ridge = Float32Array.from(sky.g.sx, skyline)
      let floor = 0
      for (const y of ridge) floor = Math.max(floor, y)
      skyBase = new Uint8ClampedArray(sky.img.data.length)
      paintSkyBase(skyBase, sky.g, pal, floor + 30)
      stars = dark ? makeStars() : []
    }
    field = layer(hill)
    if (field) {
      ground = meadow(field.g, water)
      hillBase = new Uint8ClampedArray(field.img.data.length)
      paintHillBase(hillBase, field.g, pal, water, ground)
    }
    glint = layer(near)
    if (glint) figures = new Uint8ClampedArray(glint.img.data.length)
    well = layer(riftCanvas)
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
  window.addEventListener('resize', repaint)
  document.addEventListener('themechange', repaint)

  // At most one stop-motion layer repaints per animation frame, whichever is most overdue, so the
  // ticks interleave instead of landing on the same frame.
  const jobs = [
    { run: paintAir, every: 1 / 8, acc: 0 },
    { run: paintMeadow, every: TICK, acc: 0.03 },
    { run: paintShore, every: TICK, acc: 0.06 },
    { run: paintFront, every: TICK, acc: 0.09 },
    { run: paintRanges, every: 0.2, acc: 0.12 },
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
      wind += dt * (0.05 + gust * 0.25)
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
      air.drift += 10 * dt
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
      stepCat(cat, dt, pointer, rift.heat, impulse)
      stepCrew(crew, dt, { clock: now, heat: rift.heat, inflow: rift.inflow, impulse, turn })
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
      paintPortal()
      paintShore()
    },
    pixel() {
      const g = glint?.g
      return g ? { px: g.px, x0: g.sx[0] - g.px / 2, y0: g.sy[0] - g.px / 2 } : null
    },
    dispose() {
      window.removeEventListener('resize', repaint)
      document.removeEventListener('themechange', repaint)
    },
  }
}
