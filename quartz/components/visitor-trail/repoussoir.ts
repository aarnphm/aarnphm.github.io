import type { ForestSprite } from './forest-sprites'
import type { Landscape, Point } from './landscape'
import { BAYER } from '../../util/bayer'
import { fbm, mulberry } from '../scripts/404-pixel'

// The two big trees from the 404, grown the same way and drawn at the trail's 2 px cell: one at
// each lower corner of the scene, cropped by its edge, with the long tapering limbs and the leaf
// masses of the 404's repoussoir pair. The markup keeps the trail tree's structure (motion group,
// canopy clusters, snow, hollow) so wind, seasons, snow and the depth order treat it as a tree.

type Seg = { x0: number; y0: number; x1: number; y1: number; w0: number; w1: number }
type Tip = [number, number, number]
type Tone = 'light' | 'crown' | 'dark' | 'gold'
type Cell = { x: number; y: number; tone: Tone | 'trunk' | 'trunk-lit' }

const pixel = 2
const cellKey = (x: number, y: number) => (x + 4096) * 8192 + y + 4096

// Probabilistic L-system from the 404: X → FX | F[+X]X | F[-X]X | F[+X][-X]X, unrolled until every
// limb reaches the terminal width. The trunk stays bare until it thins, forks conserve
// cross-sectional area (da Vinci), and sides follow a running golden-angle phase.
function growTree(seed: number, lean: number, width: number, len: number) {
  const r = mulberry(seed)
  const segs: Seg[] = []
  const tips: Tip[] = []
  const up = -Math.PI / 2
  const stack = [{ x: 0, y: 0, a: up + lean, w: width, l: len, phase: r() * Math.PI * 2 }]
  while (stack.length) {
    const s = stack.pop()!
    if (s.w < 1.6 || segs.length > 1400) {
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
  return { segs, tips }
}

// Distance from a point to the segment, the width there, and the signed side for bark lighting.
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

const rect = (x: number, y: number) => `M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`
const cap = (x: number, y: number) =>
  `M${x} ${y}h${pixel}v${Math.max(1, Math.floor(pixel / 2))}h${-pixel}Z`

export function repoussoirTree(seed: number, height: number, lean: number) {
  // The 404 grows its pair at width 46 and 36 for a 1500 px span; the shape is set by those
  // absolute widths (the terminal width is 1.6), so grow at the 404's size and scale to height.
  const grown = growTree(seed, lean, 44, 88)
  let top = 0
  for (const s of grown.segs) top = Math.min(top, s.y1)
  for (const [, y] of grown.tips) top = Math.min(top, y)
  const scale = height / Math.max(1, -top)
  const segs = grown.segs.map(s => ({
    x0: s.x0 * scale,
    y0: s.y0 * scale,
    x1: s.x1 * scale,
    y1: s.y1 * scale,
    w0: s.w0 * scale,
    w1: s.w1 * scale,
  }))
  const tips: Tip[] = grown.tips.map(([x, y, z]) => [x * scale, y * scale, z])
  const cells = new Map<number, Cell>()
  const mark = (cx: number, cy: number, tone: Cell['tone']) =>
    cells.set(cellKey(cx, cy), { x: cx * pixel, y: cy * pixel, tone })
  let trunkRadius = pixel
  // Limbs: the lit side of each limb faces up and left, with the valley's light.
  for (const s of segs) {
    if (s.w0 < pixel * 0.6) continue
    const pad = Math.max(s.w0, s.w1) / 2 + pixel
    const i0 = Math.floor((Math.min(s.x0, s.x1) - pad) / pixel)
    const i1 = Math.ceil((Math.max(s.x0, s.x1) + pad) / pixel)
    const j0 = Math.floor((Math.min(s.y0, s.y1) - pad) / pixel)
    const j1 = Math.ceil((Math.max(s.y0, s.y1) + pad) / pixel)
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const x = (i + 0.5) * pixel
        const y = (j + 0.5) * pixel
        const [dist, w, side] = segHit(s, x, y)
        const half = Math.max(w / 2, pixel * 0.5)
        if (dist > half) continue
        mark(i, j, side / half < -0.35 ? 'trunk-lit' : 'trunk')
        if (y >= -pixel * 2) trunkRadius = Math.max(trunkRadius, Math.abs(x) + pixel / 2)
      }
  }
  // Leaves: each tip splats density and depth; noise thresholds it into masses, and the ordered
  // dither renders the penumbra. Near leaves are lighter and crisp, far ones darker and soft.
  // The 404 splats radius 40 on trees ~300 units tall, so the crown radius is an eighth of the
  // height; a smaller one reads as a tuft on a stick at the meadow's scale.
  const radius = height * 0.12
  let left = Infinity
  let right = -Infinity
  let crownTop = Infinity
  let crownBottom = -Infinity
  for (const [x, y] of tips) {
    left = Math.min(left, x - radius)
    right = Math.max(right, x + radius)
    crownTop = Math.min(crownTop, y - radius)
    crownBottom = Math.max(crownBottom, y + radius)
  }
  const i0 = Math.floor(left / pixel) - 1
  const j0 = Math.floor(crownTop / pixel) - 1
  const bw = Math.ceil(right / pixel) + 2 - i0
  const bh = Math.ceil(crownBottom / pixel) + 2 - j0
  const density = new Float32Array(bw * bh)
  const depth = new Float32Array(bw * bh)
  for (const [tx, ty, tz] of tips) {
    const ia = Math.max(0, Math.floor((tx - radius) / pixel) - i0)
    const ib = Math.min(bw - 1, Math.ceil((tx + radius) / pixel) - i0)
    const ja = Math.max(0, Math.floor((ty - radius) / pixel) - j0)
    const jb = Math.min(bh - 1, Math.ceil((ty + radius) / pixel) - j0)
    for (let j = ja; j <= jb; j++)
      for (let i = ia; i <= ib; i++) {
        const q = 1 - Math.hypot((i + i0 + 0.5) * pixel - tx, (j + j0 + 0.5) * pixel - ty) / radius
        if (q <= 0) continue
        density[j * bw + i] += q * q
        depth[j * bw + i] += q * q * tz
      }
  }
  const cover = new Float32Array(bw * bh)
  const k = 1 / (radius * 1.6)
  for (let p = 0; p < bw * bh; p++) {
    if (density[p] < 0.05) continue
    const i = p % bw
    const j = (p - i) / bw
    const z = depth[p] / density[p]
    depth[p] = z
    const n = fbm((i + i0) * pixel * k, (j + j0) * pixel * k, 71 + seed, 2)
    const v = density[p] * (0.35 + n) * (1 + 0.15 * z)
    const penumbra = 0.1 + 0.08 * (1 - z)
    cover[p] = Math.max(0, Math.min(0.9999, (v - 0.55) / penumbra + 0.5))
  }
  const side = -1
  for (let j = 1; j < bh - 1; j++)
    for (let i = 1; i < bw - 1; i++) {
      const p = j * bw + i
      const t = cover[p]
      if (t <= BAYER[(j & 3) * 4 + (i & 3)]) continue
      const grad = Math.max(
        Math.abs(cover[p + 1] - cover[p - 1]),
        Math.abs(cover[p + bw] - cover[p - bw]),
      )
      const sunward = cover[p - bw + side] < t
      let tone: Tone
      if (t >= 0.45 && t <= 0.7 && grad > 0.3 && sunward) tone = 'gold'
      else if (t < 0.6 && !sunward) tone = 'dark'
      else {
        const shade =
          0.2 +
          fbm((i + i0) * pixel * 0.05, (j + j0) * pixel * 0.05, 83 + seed, 2) * 0.5 +
          depth[p] * 0.18 +
          (sunward ? 0.12 : 0)
        tone = shade < 0.42 ? 'dark' : shade < 0.68 ? 'crown' : 'light'
      }
      mark(i + i0, j + j0, tone)
    }
  // Markup in the trail tree's grammar: limbs that stay through the winter, the canopy in 3 × 3
  // wind clusters over a shaded copy, snow along every exposed top.
  const paths: Record<'trunk' | 'trunk-lit', string[]> = { trunk: [], 'trunk-lit': [] }
  const woodySnow: string[] = []
  const foliage = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity }
  let extent = 1
  for (const { x, y, tone } of cells.values()) {
    extent = Math.max(extent, -y)
    if (tone === 'trunk' || tone === 'trunk-lit') {
      paths[tone].push(rect(x, y))
      if (y < -height * 0.3 && !cells.has(cellKey(x / pixel, y / pixel - 1)))
        woodySnow.push(cap(x, y))
      continue
    }
    foliage.left = Math.min(foliage.left, x)
    foliage.right = Math.max(foliage.right, x + pixel)
    foliage.top = Math.min(foliage.top, y)
    foliage.bottom = Math.max(foliage.bottom, y + pixel)
  }
  type Cluster = { marks: Record<Tone, string[]>; snow: string[]; flip: string[]; sprout: string[] }
  const clusters = new Map<number, Cluster>()
  const hollow: string[] = []
  const binWidth = Math.max(pixel, (foliage.right - foliage.left) / 3)
  const binHeight = Math.max(pixel, (foliage.bottom - foliage.top) / 3)
  for (const { x, y, tone } of cells.values()) {
    if (tone === 'trunk' || tone === 'trunk-lit') continue
    const column = Math.min(2, Math.floor((x - foliage.left) / binWidth))
    const row = Math.min(2, Math.floor((y - foliage.top) / binHeight))
    const key = row * 3 + column
    let cluster = clusters.get(key)
    if (!cluster) {
      cluster = {
        marks: { light: [], crown: [], dark: [], gold: [] },
        snow: [],
        flip: [],
        sprout: [],
      }
      clusters.set(key, cluster)
    }
    const cell = rect(x, y)
    cluster.marks[tone].push(cell)
    hollow.push(cell)
    if (!cells.has(cellKey(x / pixel, y / pixel - 1))) cluster.snow.push(cap(x, y))
    let hash = Math.imul(x / pixel, 374761393) + Math.imul(y / pixel, 668265263)
    hash = Math.imul(hash ^ (hash >>> 13), 1274126177)
    hash = (hash ^ (hash >>> 16)) >>> 0
    if (tone !== 'light' && hash % 7 === 0) cluster.flip.push(cell)
    if ((hash >>> 3) % 3 === 0) cluster.sprout.push(cell)
  }
  const canopy = [...clusters.entries()]
    .sort(([a], [b]) => a - b)
    .map(([key, cluster]) => {
      const centreX = foliage.left + ((key % 3) + 0.5) * binWidth
      const centreY = foliage.top + (Math.floor(key / 3) + 0.5) * binHeight
      const lift = Math.max(0.15, Math.min(1, -centreY / extent)) ** 1.5
      return (
        `<g class="trail-tree-leaves" style="--leaf-x:${centreX.toFixed(1)};--leaf-y:${centreY.toFixed(1)};--leaf-lift:${lift.toFixed(2)};--leaf-cell:${pixel}px">` +
        (['light', 'crown', 'dark', 'gold'] as const)
          .filter(tone => cluster.marks[tone].length)
          .map(tone => `<path class="trail-tree-${tone}" d="${cluster.marks[tone].join('')}"/>`)
          .join('') +
        (cluster.flip.length
          ? `<path class="trail-tree-flip" d="${cluster.flip.join('')}"/>`
          : '') +
        (cluster.sprout.length
          ? `<path class="trail-tree-sprout" d="${cluster.sprout.join('')}"/>`
          : '') +
        `<path class="trail-tree-snow" d="${cluster.snow.join('')}"/>` +
        '</g>'
      )
    })
    .join('')
  const shadow = Math.round(trunkRadius * 2.4)
  const markup =
    `<path class="trail-tree-shadow" d="M${-shadow} 0h${shadow * 2}v2h-3v2h${-shadow * 2 + 6}V2h-3Z"/>` +
    '<g class="trail-tree-motion">' +
    `<path class="trail-tree-trunk" d="${paths.trunk.join('')}"/>` +
    `<path class="trail-tree-trunk-lit" d="${paths['trunk-lit'].join('')}"/>` +
    `<path class="trail-tree-snow" d="${woodySnow.join('')}"/>` +
    `<g class="trail-tree-canopy"><path class="trail-tree-hollow" d="${hollow.join('')}"/>${canopy}</g>` +
    '</g>'
  return { markup, trunkRadius, extent, foliage }
}

const shapes = new Map<string, ReturnType<typeof repoussoirTree>>()

function grow(treeSeed: number, tall: number, lean: number) {
  const key = `${treeSeed}|${tall}|${lean.toFixed(3)}`
  let shape = shapes.get(key)
  if (!shape) {
    shape = repoussoirTree(treeSeed, tall, lean)
    if (shapes.size > 4) shapes.clear()
    shapes.set(key, shape)
  }
  return { key, shape }
}

export type Clearing = Point & { radius: number }

// One tree at each lower corner, rooted just outside the scene's bottom corners and leaning out
// like the 404 pair, a third of the scene tall. The crown hangs half over the edge, and slides
// further out until it clears every camp; a tree with less than a crown's quarter left on screen
// is dropped. They stand in front of everything on the ground.
export function repoussoirSprites(
  landscape: Landscape,
  seed: number,
  clearings: Clearing[],
): ForestSprite[] {
  const { width, height } = landscape
  const tall = Math.round(Math.max(200, Math.min(400, height * 0.32)))
  const sprites: ForestSprite[] = []
  for (const side of [0, 1]) {
    const random = mulberry(seed ^ Math.imul(side + 1, 0x7f4a7c15))
    const out = side ? 1 : -1
    const lean = out * (0.02 + random() * 0.03)
    const treeSeed = (seed ^ Math.imul(0x404 + side, 0x9e3779b9)) >>> 0
    const { key, shape } = grow(treeSeed, tall, lean)
    const { foliage, markup, trunkRadius } = shape
    const y = height - 4
    let x = side ? width + 4 + random() * 10 : -4 - random() * 10
    const inner = side ? foliage.left : foliage.right
    for (const clearing of clearings) {
      const top = y + foliage.top
      const bottom = y + foliage.bottom
      if (clearing.y + clearing.radius < top || clearing.y - clearing.radius > bottom) continue
      const edge = clearing.x + out * clearing.radius
      const overlap = out * (x + inner - edge)
      if (overlap < 0) x += out * -overlap
    }
    const shown = side ? width - (x + foliage.left) : x + foliage.right
    if (shown < (foliage.right - foliage.left) / 4) continue
    x = Math.round(x)
    const render = () =>
      `<g class="trail-tree" data-tree-id="tree-repoussoir-${side}" data-tree="beech" data-tree-placement="standalone" data-tree-depth="meadow" data-tree-scale="1.000" data-base-trunk-radius="${trunkRadius.toFixed(3)}" data-trunk-radius="${trunkRadius.toFixed(3)}" data-growth-limit="1.3" data-ground-x="${x}" data-ground-y="${y}" data-tree-form="broad" data-tree-tone="sage" data-tree-repoussoir="${side ? 'right' : 'left'}" style="--trail-tree-haze: 0%" transform="translate(${x} ${y})"><g class="trail-tree-perspective" transform="scale(1)"><g class="trail-tree-growth" transform="scale(1)"><g class="trail-tree-community"><g class="trail-tree-touch">${markup}</g></g></g></g></g>`
    sprites.push({ id: `tree-repoussoir-${side}`, bottom: y + 12, shape: `${key}|${x}`, render })
  }
  return sprites
}
