// Pixel landscape for the 404 ruin, painted the way Van Gogh painted the factories at Clichy:
// low-resolution canvases upscaled with `image-rendering: pixelated`, an ordered-dither
// underpainting, then short brush strokes laid along flow fields. Each stroke picks its ramp stop
// once, so the dither happens per stroke instead of per pixel. The ranges and the grain over the
// temple paint once per size or theme; sky, field, water, canopy and portal repaint in
// stop-motion ticks.

import { GATE, MISSING, STEP_TOP, hourAngle, stoneOuter } from './404-gate'

// Leaf tip: position plus depth toward (+1) or away from (-1) the viewer.
type Tip = [number, number, number]
type Pt = [number, number]
type RGB = [number, number, number]
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
}

export type Landscape = {
  step(dt: number, clock: number, pointer: Pointer): void
  portal(mode: Mode): void
  dispose(): void
}

const W = 1600
const H = 1000
const TICK = 1 / 10
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16)
// Gold only where the tone is mid-range and the gradient is sharp: fringes, never interiors.
const GOLD_START = 0.45
const GOLD_END = 0.7
const SUN = { x: 1150, y: 118, r: 46 }

const hex = (h: string): RGB => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)) as RGB

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
}

function mulberry(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hash(x: number, y: number, seed: number) {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 144269504)) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

function noise(x: number, y: number, seed: number) {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const xf = x - xi
  const yf = y - yi
  const u = xf * xf * (3 - 2 * xf)
  const v = yf * yf * (3 - 2 * yf)
  const a = hash(xi, yi, seed)
  const b = hash(xi + 1, yi, seed)
  const c = hash(xi, yi + 1, seed)
  const d = hash(xi + 1, yi + 1, seed)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

function fbm(x: number, y: number, seed: number, octaves = 3) {
  let sum = 0
  let amp = 0.5
  let norm = 0
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise(x, y, seed + i * 17)
    norm += amp
    amp *= 0.5
    x *= 2.03
    y *= 2.03
  }
  return sum / norm
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

const clamp01 = (t: number) => (t < 0 ? 0 : t > 0.9999 ? 0.9999 : t)
const smooth = (a: number, b: number, t: number) => {
  const x = clamp01((t - a) / (b - a))
  return x * x * (3 - 2 * x)
}
const frac = (t: number) => t - Math.floor(t)
const mul = (c: RGB, k: number): RGB => [c[0] * k, c[1] * k, c[2] * k]
const mix = (a: RGB, b: RGB, k: number): RGB => [0, 1, 2].map(i => a[i] + (b[i] - a[i]) * k) as RGB

// Ordered dither between neighbouring ramp stops.
function ramp(stops: RGB[], t: number, bx: number, by: number): RGB {
  return stops[step(stops.length, t, bx, by)]
}

function step(n: number, t: number, bx: number, by: number) {
  const p = clamp01(t) * (n - 1)
  const i = Math.floor(p)
  return p - i > BAYER[(by & 3) * 4 + (bx & 3)] ? i + 1 : i
}

// Stroke-level dither: the stroke, not the pixel, chooses between neighbouring stops.
function pick(stops: RGB[], t: number, threshold: number): RGB {
  const p = clamp01(t) * (stops.length - 1)
  const i = Math.floor(p)
  return stops[p - i > threshold ? i + 1 : i]
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

type Grid = {
  bw: number
  bh: number
  sx: Float32Array
  sy: Float32Array
  // scene units per buffer pixel
  px: number
  toBuffer: DOMMatrix
}

// Buffer pixel centres mapped into scene units with the same xMidYMid-slice fit as the SVG layers.
function grid(canvas: HTMLCanvasElement): Grid | null {
  const cw = canvas.clientWidth
  const ch = canvas.clientHeight
  if (!cw || !ch) return null
  const px = Math.max(3, Math.min(6, Math.round(cw / 380)))
  const bw = Math.ceil(cw / px)
  const bh = Math.ceil(ch / px)
  const scale = Math.max(cw / W, ch / H)
  const ox = (cw - W * scale) / 2
  const oy = (ch - H * scale) / 2
  const sx = new Float32Array(bw)
  const sy = new Float32Array(bh)
  for (let i = 0; i < bw; i++) sx[i] = ((i + 0.5) * px - ox) / scale
  for (let j = 0; j < bh; j++) sy[j] = ((j + 0.5) * px - oy) / scale
  canvas.width = bw
  canvas.height = bh
  const toBuffer = new DOMMatrix([scale / px, 0, 0, scale / px, ox / px, oy / px])
  return { bw, bh, sx, sy, px: px / scale, toBuffer }
}

const col = (g: Grid, x: number) => Math.round((x - g.sx[0]) / g.px)
const row = (g: Grid, y: number) => Math.round((y - g.sy[0]) / g.px)
const clampCol = (g: Grid, i: number) => Math.max(0, Math.min(g.bw - 1, Math.round(i)))

function put(data: Uint8ClampedArray, k: number, c: RGB | readonly number[], a = 255) {
  data[k] = c[0]
  data[k + 1] = c[1]
  data[k + 2] = c[2]
  data[k + 3] = a
}

function dot(d: Uint8ClampedArray, g: Grid, i: number, j: number, c: RGB) {
  if (i >= 0 && j >= 0 && i < g.bw && j < g.bh) put(d, (j * g.bw + i) * 4, c)
}

function paint(canvas: HTMLCanvasElement, draw: (d: Uint8ClampedArray, g: Grid) => void) {
  const g = grid(canvas)
  if (!g) return null
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(g.bw, g.bh)
  draw(img.data, g)
  ctx.putImageData(img, 0, 0)
  return g
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
      if (y >= mid[i]) {
        // Sun-facing slopes descend to the right; the lit crest catches gold.
        if (y - mid[i] < g.px * 1.2 && midLit[i] > 0.15) put(d, k, pal.gold)
        else put(d, k, ramp(pal.mid, midTone(i, y), i, j))
      } else if (y >= far[i]) put(d, k, ramp(pal.far, farTone(i, y), i, j))
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
  const inFar: Keep = (i, j) => sy[j] >= far[i] && sy[j] < mid[i]
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
      if (y - m.top[i] < g.px * 1.2 && m.lit[i] > 0.08) put(d, k, pal.gold)
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

// Wanders the strand in front of the steps, stops now and then to look at the gate, and walks
// to it while the portal is open.
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
    if (w.x < 250) w.dir = 1
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
// toward us, widening with distance below the horizon, then the walker and her reflection,
// folded about the waterline with the temple mirror's squash.
function paintNear(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  water: number,
  w: Walker,
  heat: number,
  clock: number,
) {
  d.fill(0)
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

  const rows = [...FIGURE, LEGS[w.walking ? Math.floor(w.stride / 0.22) % 2 : 2]]
  const foot = row(g, Math.max(848, hillTop(w.x, water) + 8))
  const ci = col(g, w.x)
  rows.forEach((line, r) => {
    const j = foot - rows.length + 1 + r
    const mirror = water + 0.62 * (water - (sy[0] + j * g.px))
    const mj = row(g, mirror)
    const wobble = Math.round(Math.sin(mj * 1.7 + clock * 5) * 0.7)
    for (let c = 0; c < 4; c++) {
      const ink = line[w.dir > 0 ? c : 3 - c]
      if (ink === '.') continue
      const i = ci - 2 + c
      dot(d, g, i, j, pal.figure[ink as Ink])
      if (mirror > water + g.px && (mj + i) % 2 === 0)
        dot(d, g, i + wobble, mj, pal.figure[ink as Ink])
    }
  })
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

// Grain over the temple's flat plates: its own fill and keyline paths are rasterized into
// material and exclusion masks, then only the off-plate dither pixels are painted, so the SVG
// fills show through untouched and the ink keylines stay crisp.
function paintGrain(d: Uint8ClampedArray, g: Grid, svg: SVGSVGElement, dark: boolean) {
  const art = svg.querySelector('#nf-temple-art')
  const root = svg.getScreenCTM()
  if (!art || !root) return
  const rootInv = root.inverse()
  const canvasFor = () => {
    const c = document.createElement('canvas')
    c.width = g.bw
    c.height = g.bh
    return c.getContext('2d', { willReadFrequently: true })!
  }
  const mat = canvasFor()
  const key = canvasFor()
  const stops: RGB[][] = MATERIALS.map(() => [])
  const channel = ['#ff0000', '#00ff00', '#0000ff']
  const paper: RGB = dark ? [206, 205, 195] : [255, 252, 240]

  for (const el of art.querySelectorAll<SVGPathElement | SVGRectElement>('path, rect')) {
    const cls = el.getAttribute('class') ?? ''
    if (cls.includes('nf-light-spill')) continue
    const ctm = el.getScreenCTM()
    if (!ctm) continue
    const path =
      el instanceof SVGPathElement
        ? new Path2D(el.getAttribute('d') ?? '')
        : new Path2D(
            `M${el.x.baseVal.value} ${el.y.baseVal.value}h${el.width.baseVal.value}v${el.height.baseVal.value}h${-el.width.baseVal.value}Z`,
          )
    const m = g.toBuffer.multiply(rootInv.multiply(ctm))
    const id = MATERIALS.findIndex(c => cls.includes(c))
    if (id >= 0) {
      if (stops[id].length === 0) {
        const base = probe(getComputedStyle(el).fill)
        stops[id] = dark
          ? [mul(base, 0.55), mul(base, 0.75), base, mix(base, paper, 0.2)]
          : [mul(base, 0.7), mul(base, 0.86), base, mix(base, paper, 0.55)]
      }
      mat.setTransform(m)
      mat.fillStyle = channel[id]
      mat.fill(path)
    } else {
      key.setTransform(m)
      if (cls.includes('nf-key') || cls.includes('nf-stem')) {
        key.lineWidth = Math.max(2.5, g.px * 1.1)
        key.stroke(path)
      } else key.fill(path)
    }
  }

  const md = mat.getImageData(0, 0, g.bw, g.bh).data
  const kd = key.getImageData(0, 0, g.bw, g.bh).data
  for (let j = 0; j < g.bh; j++)
    for (let i = 0; i < g.bw; i++) {
      const k = (j * g.bw + i) * 4
      if (kd[k + 3] > 60 || md[k + 3] < 128) continue
      const id = md[k] >= md[k + 1] && md[k] >= md[k + 2] ? 0 : md[k + 1] >= md[k + 2] ? 1 : 2
      const s = step(5, stoneTone(g.sx[i], g.sy[j]) * 0.75 + 0.125, i, j)
      // Stops 0..4 map onto deep, shade, plate, plate, light; the plate never gets painted.
      if (s === 2 || s === 3) continue
      put(d, k, stops[id][s === 4 ? 3 : s])
    }
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
const GOLDEN = hex('#f1d67e')
const ROSE = hex('#e3a19a')
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
  const walker: Walker = { x: 560, dir: 1, pause: 1.5, next: 9, stride: 0, walking: false }

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
  let sky: Layer | null = null
  let field: Layer | null = null
  let glint: Layer | null = null
  let well: Layer | null = null
  let leaves: Layer | null = null
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
  let lastX = NaN
  let lastY = NaN
  let shed = 0
  let sign = 1

  const flush = (l: Layer) => l.ctx.putImageData(l.img, 0, 0)
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
  const paintShore = () => {
    if (!glint) return
    paintNear(glint.img.data, glint.g, pal, water, walker, rift.heat, clock)
    flush(glint)
  }
  // Only the opening is ever painted, and the underpainting covers all of it each frame.
  const paintPortal = () => {
    if (!well) return
    paintRift(well.img.data, well.g, rift, clock)
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
    const dark = document.documentElement.getAttribute('saved-theme') === 'dark'
    pal = dark ? DARK : LIGHT
    paint(back, (d, g) => paintLand(d, g, pal, pines))
    paint(grain, (d, g) => paintGrain(d, g, temple, dark))
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
    paintAir()
    paintMeadow()
    paintShore()
    paintPortal()
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

      stepRift(rift, dt)
      stepWalker(walker, dt, drawn)
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
        paintPortal()
      }
    },
    dispose() {
      window.removeEventListener('resize', repaint)
      document.removeEventListener('themechange', repaint)
    },
  }
}
