import type { ForestSprite } from './forest-sprites'
import { bayerThreshold } from '../../util/bayer'
import { forestYear, groveStorms, shadeTolerance, slotOccupant } from './forest-succession'
import {
  creekDistance,
  creekSection,
  groundOffset,
  onDryLand,
  pondDistance,
  type Landscape,
} from './landscape'
import { trailRandom } from './random'
import { forestLimits, sceneSettings, type ForestSettings } from './scene-settings'
import { trailDeadfall, trailUnderstory } from './understory'
import { trailSnowBank } from './winter'

export type TreeKind =
  | 'pine'
  | 'spruce'
  | 'fir'
  | 'hemlock'
  | 'cypress'
  | 'larch'
  | 'birch'
  | 'aspen'
  | 'oak'
  | 'maple'
  | 'willow'
  | 'sequoia'
  | 'snag'
type Crown = 'round' | 'cone' | 'spire' | 'column' | 'tuft' | 'tier' | 'droop'
type Species = {
  // A single leader with whorled side branches, or a crown that keeps forking as it grows.
  leader: boolean
  // Needles neither flutter nor show pale undersides in a gust.
  needles: boolean
  crown: Crown
  spread: number
  radius: number
  limb: number
  // Papery white bark with dark bands every few rows (birch, aspen).
  bands: number
  // Share of crown cells left open.
  sparse: number
  // Share of the height, from the ground, that carries no foliage.
  clear: number
}
type TreeForm = 'upright' | 'leaning' | 'forked' | 'open' | 'broad'
type TreeTone = 'sage' | 'lichen' | 'ochre'
type Point = { x: number; y: number }
type Tip = Point & { level: number }
type Turtle = Point & { heading: number; bend: number }
type Tone = 'trunk' | 'bark' | 'light' | 'crown' | 'dark' | 'gold'
// Eastern white pine holds its needles in flat tufts; hemlock in drooping tiers; tamarack is a
// thin conifer that turns gold; aspen is a narrow crown on pale bark; willow hangs in curtains.
const species: Record<TreeKind, Species> = {
  pine: {
    leader: true,
    needles: true,
    crown: 'tuft',
    spread: 1.15,
    radius: 4,
    limb: 1.3,
    bands: 0,
    sparse: 0.1,
    clear: 0,
  },
  spruce: {
    leader: true,
    needles: true,
    crown: 'cone',
    spread: 0.7,
    radius: 4,
    limb: 1,
    bands: 0,
    sparse: 0,
    clear: 0,
  },
  fir: {
    leader: true,
    needles: true,
    crown: 'spire',
    spread: 0.5,
    radius: 3.5,
    limb: 0.9,
    bands: 0,
    sparse: 0,
    clear: 0,
  },
  hemlock: {
    leader: true,
    needles: true,
    crown: 'tier',
    spread: 0.95,
    radius: 4.5,
    limb: 1,
    bands: 0,
    sparse: 0.04,
    clear: 0,
  },
  cypress: {
    leader: true,
    needles: true,
    crown: 'column',
    spread: 0.45,
    radius: 4,
    limb: 0.9,
    bands: 0,
    sparse: 0,
    clear: 0,
  },
  larch: {
    leader: true,
    needles: true,
    crown: 'cone',
    spread: 0.8,
    radius: 3.5,
    limb: 0.85,
    bands: 0,
    sparse: 0.32,
    clear: 0,
  },
  birch: {
    leader: false,
    needles: false,
    crown: 'round',
    spread: 1,
    radius: 5,
    limb: 0.85,
    bands: 5,
    sparse: 0.04,
    clear: 0,
  },
  aspen: {
    leader: false,
    needles: false,
    crown: 'column',
    spread: 0.6,
    radius: 4.5,
    limb: 0.8,
    bands: 7,
    sparse: 0.06,
    clear: 0,
  },
  oak: {
    leader: false,
    needles: false,
    crown: 'round',
    spread: 1.2,
    radius: 5.5,
    limb: 1.5,
    bands: 0,
    sparse: 0.02,
    clear: 0,
  },
  maple: {
    leader: false,
    needles: false,
    crown: 'round',
    spread: 1.05,
    radius: 6,
    limb: 1.3,
    bands: 0,
    sparse: 0,
    clear: 0,
  },
  willow: {
    leader: false,
    needles: false,
    crown: 'droop',
    spread: 1.3,
    radius: 5,
    limb: 1.35,
    bands: 0,
    sparse: 0,
    clear: 0,
  },
  // Giant sequoia: a cinnamon column bare for half its height under a few rounded crowns. A snag is a
  // standing dead conifer, silver, with no foliage at all.
  sequoia: {
    leader: true,
    needles: true,
    crown: 'round',
    spread: 0.55,
    radius: 3.2,
    limb: 2.1,
    bands: 0,
    sparse: 0.04,
    clear: 0.5,
  },
  snag: {
    leader: true,
    needles: true,
    crown: 'cone',
    spread: 0.75,
    radius: 1,
    limb: 1,
    bands: 0,
    sparse: 1,
    clear: 1,
  },
}
const kinds = Object.keys(species) as TreeKind[]
// Trees that grow together, one stand per grove: boreal, pine–hemlock, sugar-maple hardwood, and
// Sierra mixed conifer (sequoia with ponderosa, incense cedar and black oak).
const stands: readonly TreeKind[][] = [
  ['spruce', 'fir', 'birch', 'larch'],
  ['pine', 'hemlock', 'birch', 'aspen'],
  ['maple', 'oak', 'aspen', 'maple'],
  ['sequoia', 'pine', 'cypress', 'sequoia'],
]
const foothillKinds: readonly TreeKind[] = ['pine', 'spruce', 'fir', 'larch']
// Broadleaves and the larch drop their leaves for the winter.
const bareInWinter = new Set<TreeKind>(['maple', 'oak', 'aspen', 'birch', 'larch', 'willow'])
const woodlandKinds: readonly TreeKind[] = [
  'spruce',
  'fir',
  'pine',
  'larch',
  'birch',
  'hemlock',
  'maple',
  'aspen',
]
const forms: TreeForm[] = ['upright', 'leaning', 'forked', 'open', 'broad']
const tones: TreeTone[] = ['sage', 'lichen', 'ochre']

function canopyTone(x: number, y: number, radius: number, pixel: number, dapple: number): Tone {
  const brightness = Math.max(0, Math.min(1, 0.52 - (x / radius) * 0.23 - (y / radius) * 0.16))
  const column = ((Math.round(x / pixel) % 4) + 4) % 4
  const row = ((Math.round(y / pixel) % 4) + 4) % 4
  const threshold = bayerThreshold(column, row)
  const rim = Math.hypot(x, y) / radius
  if (rim > 0.72 && x - y < 0 && brightness > 0.5 && threshold < dapple * 0.32) return 'gold'
  const level = brightness * 2
  const lower = Math.floor(level)
  const upper = level - lower > 0.5 * (1 - dapple) + threshold * dapple
  const index = Math.min(2, lower + Number(upper))
  return index === 0 ? 'dark' : index === 1 ? 'crown' : 'light'
}

// The rows of a crown around one bud: where it starts and ends, how wide each row is, how far the
// whole crown sits above the bud, and how much its edges sag.
function crownRows(crown: Crown, r: number) {
  const ellipse = (y: number, ry: number, rx: number) =>
    rx * Math.sqrt(Math.max(0, 1 - (y / ry) ** 2))
  switch (crown) {
    case 'cone':
      return { top: -r, bottom: r, lift: -r, sag: 0, span: (y: number) => (y + r) * 0.65 }
    case 'spire':
      return {
        top: -1.6 * r,
        bottom: r,
        lift: -r,
        sag: 0,
        span: (y: number) => (y + 1.6 * r) * 0.36,
      }
    case 'column':
      return {
        top: -1.5 * r,
        bottom: 1.5 * r,
        lift: -0.4 * r,
        sag: 0,
        span: (y: number) => ellipse(y, 1.5 * r, 0.55 * r),
      }
    case 'tuft':
      return {
        top: -0.6 * r,
        bottom: 0.4 * r,
        lift: -0.2 * r,
        sag: 0.15,
        span: (y: number) => ellipse(y + 0.1 * r, 0.6 * r, 1.4 * r),
      }
    case 'tier':
      return {
        top: -0.35 * r,
        bottom: 0.3 * r,
        lift: 0,
        sag: 0.35,
        span: (y: number) => ellipse(y, 0.4 * r, 1.3 * r),
      }
    case 'droop':
      return {
        top: -0.7 * r,
        bottom: 0,
        lift: 0,
        sag: 0,
        span: (y: number) => ellipse(y, 0.7 * r, 1.1 * r),
      }
    default:
      return { top: -r, bottom: r, lift: 0, sag: 0, span: (y: number) => ellipse(y, r, r) }
  }
}

function bounded(value: number, field: keyof ForestSettings) {
  const { min, max } = forestLimits[field]
  return Math.max(min, Math.min(max, Math.round(value)))
}

function nearestDistance(point: Point, points: Point[]) {
  let distance = Infinity
  for (const other of points) {
    const squared = (point.x - other.x) ** 2 + (point.y - other.y) ** 2
    if (squared < distance) distance = squared
  }
  return Math.sqrt(distance)
}

// Raster cells are keyed by one number; a tree spans far fewer than 4,096 cells either way.
const cellKey = (x: number, y: number) => (x + 4096) * 8192 + y + 4096

function tree(
  kind: TreeKind,
  seed: number,
  height: number,
  settings: ForestSettings,
  minPixel = 1,
  // A young tree has run fewer rewrites of the same word, so it is the old tree's early form.
  grown = forestLimits.depth.max,
) {
  const random = trailRandom(seed)
  const growth = trailRandom(seed ^ 0x9e3779b9)
  const leaves = trailRandom(seed ^ 0xa46f315b)
  const habit = species[kind]
  // Some pines and cedars keep the narrow cone the forest grew before species had their own crowns.
  const classic = (kind === 'pine' || kind === 'cypress') && trailRandom(seed ^ 0x5bd1e995)() < 0.45
  const crownShape: Crown = classic ? 'cone' : habit.crown
  const spread = classic && kind === 'pine' ? 0.85 : habit.spread
  const depth = Math.min(bounded(settings.depth, 'depth'), Math.max(forestLimits.depth.min, grown))
  const probability = bounded(settings.branching, 'branching') / 100
  const variation = bounded(settings.variation, 'variation') / 100
  const dapple = bounded(settings.dapple, 'dapple') / 100
  let form = variation ? forms[Math.floor(random() * forms.length)] : 'upright'
  if (form === 'forked' && growth() >= probability) form = 'upright'
  const tone = tones[Math.floor(random() * tones.length)]
  const handedness = random() > 0.5 ? 1 : -1
  const lean =
    handedness * variation * (form === 'leaning' ? 0.15 + random() * 0.2 : random() * 0.08)
  const breadth = 1 + variation * (form === 'broad' ? 0.45 : random() * 0.2 - 0.1)
  const branchBias = (random() - 0.5) * variation * 0.3
  type Growth = number | 'X' | '[' | ']' | '+' | '-'
  let word: Growth[] =
    form === 'forked' ? [2, '[', '+', 1, 'X', ']', '[', '-', 1, 'X', ']'] : [1, 'X']
  // Count consecutive forward steps while expanding. Growth visits X in the same order,
  // so both random streams and every final branch coordinate stay unchanged.
  for (let iteration = 0; iteration < depth; iteration++) {
    const next: Growth[] = []
    for (const symbol of word) {
      if (typeof symbol === 'number') next.push(symbol * 2)
      else if (symbol !== 'X') next.push(symbol)
      else {
        const bias = branchBias * probability * (1 - probability)
        const left = growth() < probability + bias
        const right = growth() < probability - bias
        next.push(1)
        if (left) next.push('[', '+', 'X', ']')
        if (habit.leader) {
          if (right) next.push('[', '-', 'X', ']')
          next.push(1, 'X')
        } else {
          next.push(1)
          if (right) next.push('[', '-', 'X', ']')
          next.push('+', 'X')
        }
      }
    }
    word = next
  }
  const angle = (bounded(settings.angle, 'angle') * spread * breadth * Math.PI) / 180
  // Segment counts grow with depth; keep curvature per branch at the same scale.
  const curl = (bounded(settings.curl, 'curl') * Math.PI) / (180 * 2 ** depth)
  let turtle: Turtle = { x: 0, y: 0, heading: -Math.PI / 2 + lean, bend: handedness * 0.4 }
  const stack: Turtle[] = []
  // Each segment as five numbers: from x, from y, to x, to y, and its depth in the branching.
  const branches: number[] = []
  const tips: Tip[] = []
  let top = 0
  for (let index = 0; index < word.length; index++) {
    const symbol = word[index]
    if (typeof symbol === 'number') {
      for (let step = 0; step < symbol; step++) {
        const fromX = turtle.x
        const fromY = turtle.y
        turtle.heading = Math.max(
          -Math.PI + 0.1,
          Math.min(-0.1, turtle.heading + turtle.bend * curl),
        )
        const length = 0.78 ** stack.length * (1 + (random() - 0.5) * variation * 0.35)
        turtle.x += Math.cos(turtle.heading) * length
        turtle.y += Math.sin(turtle.heading) * length
        top = Math.min(top, turtle.y)
        branches.push(fromX, fromY, turtle.x, turtle.y, stack.length)
      }
    } else if (symbol === '+' || symbol === '-') {
      const turn = handedness * (symbol === '+' ? 1 : -1)
      turtle.heading += angle * turn * (1 + (random() - 0.5) * variation * 0.65)
      turtle.bend = turn * (0.6 + random() * 0.4)
    } else if (symbol === '[') {
      stack.push({ ...turtle })
    } else if (symbol === ']') {
      const previous = stack.pop()
      if (previous) turtle = previous
    } else if (symbol === 'X') {
      tips.push({ x: turtle.x, y: turtle.y, level: stack.length })
    }
  }
  const size = (bounded(settings.size, 'size') / 100) * (1 + (random() - 0.5) * variation * 0.25)
  const scale = ((height - 8) * size) / Math.max(1, -top)
  const pixel = Math.max(minPixel, Math.ceil((height * size) / 54))
  const cells = new Map<number, { x: number; y: number; tone: Tone }>()
  const mark = (x: number, y: number, tone: Tone) => {
    const cellX = Math.round(x / pixel)
    const cellY = Math.round(y / pixel)
    const key = cellKey(cellX, cellY)
    const cell = cells.get(key)
    if (cell) cell.tone = tone
    else cells.set(key, { x: cellX * pixel, y: cellY * pixel, tone })
  }
  // A broadleaf loses its crown in winter. Its bare frame is kept apart: the trunk, and limbs
  // grown from it to fill the crown's extent (the L-system's own limbs are short under a full crown,
  // and at full depth they fill its outline).
  const deciduous = bareInWinter.has(kind)
  const limbs = new Map<number, { x: number; y: number; tone: Tone }>()
  const buds = new Set<number>()
  const limb = (x: number, y: number, tone: Tone) => {
    mark(x, y, tone)
    if (!deciduous) return
    const cellX = Math.round(x / pixel)
    const cellY = Math.round(y / pixel)
    limbs.set(cellKey(cellX, cellY), { x: cellX * pixel, y: cellY * pixel, tone })
  }
  for (let index = 0; index < branches.length; index += 5) {
    const fromX = branches[index]
    const fromY = branches[index + 1]
    const toX = branches[index + 2]
    const toY = branches[index + 3]
    const level = branches[index + 4]
    const steps = Math.max(1, Math.ceil((Math.hypot(toX - fromX, toY - fromY) * scale) / pixel))
    const girth = Math.max(pixel, 3.4 * size * habit.limb * 0.66 ** level)
    for (let step = 0; step <= steps; step++) {
      const fraction = step / steps
      const x = (fromX + (toX - fromX) * fraction) * scale
      const y = (fromY + (toY - fromY) * fraction) * scale
      // The trunk flares where its roots meet the ground.
      const flare = level === 0 && y > -height * size * 0.08 ? 1.6 : 1
      const thickness = girth * flare
      const bark = habit.bands > 0 && Math.floor(-y / pixel) % habit.bands !== 0
      for (let offset = -thickness / 2; offset <= thickness / 2; offset += pixel)
        (level === 0 ? limb : mark)(x + offset, y, bark ? 'bark' : 'trunk')
    }
  }
  const crowns = new Set<number>()
  tips.forEach(tip => {
    // Buds within one raster cell at the same depth share a crown.
    const crown =
      cellKey(Math.round((tip.x * scale) / pixel), Math.round((tip.y * scale) / pixel)) * 16 +
      tip.level
    if (crowns.has(crown)) return
    crowns.add(crown)
    if (tip.y >= top * habit.clear) return
    if (leaves() < (form === 'open' ? 0.3 : 0.06) * variation) return
    const radius = (habit.radius + leaves() * habit.radius * 0.6) * size * 0.9 ** tip.level
    const fullness = breadth * (0.9 + leaves() * variation * 0.35)
    const rows = crownRows(crownShape, radius)
    const leaf = (x: number, y: number, rim: boolean) => {
      if (leaves() < habit.sparse + variation * (rim ? 0.16 : form === 'open' ? 0.08 : 0.015))
        return
      mark(
        tip.x * scale + x,
        tip.y * scale + y + rows.lift,
        canopyTone(x, y, radius, pixel, dapple),
      )
    }
    for (let y = rows.top; y <= rows.bottom; y += pixel) {
      const edge = 1 + Math.sin(y * 0.8 + tip.x) * variation * 0.12
      const span = rows.span(y) * fullness * edge
      for (let x = -span; x <= span; x += pixel)
        leaf(x, y + rows.sag * radius * (x / Math.max(pixel, span)) ** 2, Math.abs(x) > span * 0.82)
    }
    // Willow strands hang from the dome in curtains, a gap every few columns.
    if (habit.crown === 'droop')
      for (let x = -radius * 1.1; x <= radius * 1.1; x += pixel) {
        const strand = Math.imul(Math.round(x / pixel) + tip.level * 7, 2654435761) >>> 0
        if (strand % 4 === 0) continue
        const length = radius * (1.3 + ((strand >>> 4) % 5) * 0.22)
        for (let y = 0; y <= length; y += pixel) leaf(x, y, y > length - pixel * 2)
      }
  })
  const paths: Record<'trunk' | 'bark', string[]> = { trunk: [], bark: [] }
  let crownTop = 0
  let trunkRadius = 1
  const woodySnow: string[] = []
  const foliage = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity }
  for (const { x, y, tone } of cells.values()) {
    crownTop = Math.min(crownTop, y)
    if ((tone === 'trunk' || tone === 'bark') && y >= -pixel * 2)
      trunkRadius = Math.max(trunkRadius, Math.abs(x), Math.abs(x + pixel))
    if (tone === 'trunk' || tone === 'bark') {
      paths[tone].push(`M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`)
      if (y < -height * size * 0.3 && !cells.has(cellKey(x / pixel, y / pixel - 1)))
        woodySnow.push(`M${x} ${y}h${pixel}v${Math.max(1, Math.floor(pixel / 2))}h${-pixel}Z`)
      continue
    }
    foliage.left = Math.min(foliage.left, x)
    foliage.right = Math.max(foliage.right, x + pixel)
    foliage.top = Math.min(foliage.top, y)
    foliage.bottom = Math.max(foliage.bottom, y + pixel)
  }
  if (deciduous && foliage.top < foliage.bottom) {
    const bare = trailRandom(seed ^ 0x6c8e9cf5)
    // Birch and aspen keep their limbs close and upright, larch holds them out level in tiers,
    // willow hangs its twigs, and oak forks wide and crooked.
    const fan =
      habit.crown === 'column' ? 0.34 : habit.crown === 'cone' ? 0.95 : kind === 'oak' ? 0.8 : 0.6
    const crook = kind === 'oak' ? 0.5 : 0.22
    const stem = Math.max(pixel, 3.4 * size * habit.limb * 0.7)
    const inside = (x: number, y: number) =>
      x >= foliage.left - pixel &&
      x <= foliage.right + pixel &&
      y >= foliage.top - pixel &&
      y <= foliage.bottom
    const put = (x: number, y: number) => {
      const cellX = Math.round(x / pixel)
      const cellY = Math.round(y / pixel)
      const key = cellKey(cellX, cellY)
      if (!limbs.has(key)) limbs.set(key, { x: cellX * pixel, y: cellY * pixel, tone: 'trunk' })
      return key
    }
    const twigs = (x: number, y: number, heading: number) => {
      const hang = habit.crown === 'droop'
      const count = (hang ? 1 : 2) + Math.floor(bare() * 2)
      for (let index = 0; index < count; index++) {
        const turn = hang
          ? Math.PI / 2 + (bare() - 0.5) * 0.5
          : heading + (index - (count - 1) / 2) * 0.7 + (bare() - 0.5) * 0.4
        const length = (hang ? 4 + bare() * 6 : 2 + bare() * 3) * pixel
        let end: number | null = null
        for (let along = pixel; along <= length; along += pixel)
          end = put(x + Math.cos(turn) * along, y + Math.sin(turn) * along)
        // One twig end in three opens a flower or catkin in early spring.
        if (end !== null && bare() < 0.34) buds.add(end)
      }
    }
    const grow = (x: number, y: number, heading: number, length: number, order: number) => {
      const girth = Math.max(pixel, stem * 0.55 ** order)
      let endX = x
      let endY = y
      for (let along = 0; along <= length; along += pixel) {
        const px = x + Math.cos(heading) * along
        const py = y + Math.sin(heading) * along
        if (!inside(px, py)) break
        for (let offset = -girth / 2; offset < girth / 2 || offset === -girth / 2; offset += pixel)
          put(px + offset, py)
        endX = px
        endY = py
      }
      if (order >= 3) return twigs(endX, endY, heading)
      const count = 2 + (bare() < 0.35 ? 1 : 0)
      for (let index = 0; index < count; index++) {
        const turn = heading + (index - (count - 1) / 2) * fan + (bare() - 0.5) * crook
        grow(
          endX,
          endY,
          Math.max(-Math.PI + 0.12, Math.min(-0.12, turn)),
          length * (0.55 + bare() * 0.2),
          order + 1,
        )
      }
    }
    // Limbs leave the trunk where it enters the crown, alternating sides up to the leader's top,
    // which carries on toward the crown top.
    const leader: Point[] = []
    for (let index = 0; index < branches.length; index += 5)
      if (branches[index + 4] === 0 && branches[index + 3] * scale < foliage.bottom)
        leader.push({ x: branches[index + 2] * scale, y: branches[index + 3] * scale })
    const height = foliage.bottom - foliage.top
    const forks = 2 + Math.floor(bare() * 3)
    for (let index = 0; index < forks && leader.length; index++) {
      const at = leader[Math.floor(((index + 0.5) / forks) * leader.length)]
      const side = index % 2 ? 1 : -1
      const turn = -Math.PI / 2 + side * (fan * 1.1 + bare() * 0.3)
      grow(at.x, at.y, turn, height * (0.45 - (index / forks) * 0.15), 1)
    }
    const crest = leader.at(-1) ?? { x: 0, y: foliage.bottom }
    grow(crest.x, crest.y, -Math.PI / 2 + (bare() - 0.5) * 0.3, height * 0.4, 1)
  }
  // Wind moves leaves and twigs; the trunk and limbs stay put. The canopy splits into a 3 × 3 grid
  // of clusters, each shifting by whole cells, so a gust can cross the crown one cluster at a time.
  type Cluster = {
    marks: Record<Exclude<Tone, 'trunk' | 'bark'>, string[]>
    snow: string[]
    flip: string[]
    sprout: string[]
  }
  const clusters = new Map<number, Cluster>()
  // A still copy of the crown in its shade tone sits under the clusters, so a cluster that shifts
  // uncovers shaded leaves rather than the paper, at the seams and along the silhouette alike. The
  // gaps in a sparse crown are not on it and stay open.
  const hollow: string[] = []
  const binWidth = Math.max(pixel, (foliage.right - foliage.left) / 3)
  const binHeight = Math.max(pixel, (foliage.bottom - foliage.top) / 3)
  for (const { x, y, tone } of cells.values()) {
    if (tone === 'trunk' || tone === 'bark') continue
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
    const cell = `M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`
    cluster.marks[tone].push(cell)
    hollow.push(cell)
    const exposed = !cells.has(cellKey(x / pixel, y / pixel - 1))
    if (exposed)
      cluster.snow.push(`M${x} ${y}h${pixel}v${Math.max(1, Math.floor(pixel / 2))}h${-pixel}Z`)
    // Broadleaves show their paler undersides when a gust turns them; needles do not.
    let hash = Math.imul(x / pixel, 374761393) + Math.imul(y / pixel, 668265263)
    hash = Math.imul(hash ^ (hash >>> 13), 1274126177)
    hash = (hash ^ (hash >>> 16)) >>> 0
    if (!habit.needles && tone !== 'light' && hash % (kind === 'aspen' ? 4 : 7) === 0)
      cluster.flip.push(cell)
    // The first leaves of spring, and the last of autumn, are a third of a broadleaf crown. A
    // conifer's new growth comes at the tips of its shoots, on top of the old needles.
    if (habit.needles ? exposed && (hash >>> 3) % 2 === 0 : (hash >>> 3) % 3 === 0)
      cluster.sprout.push(cell)
  }
  const extent = Math.max(1, -crownTop)
  const canopy = [...clusters.entries()]
    .sort(([a], [b]) => a - b)
    .map(([key, cluster]) => {
      const centreX = foliage.left + ((key % 3) + 0.5) * binWidth
      const centreY = foliage.top + (Math.floor(key / 3) + 0.5) * binHeight
      // Displacement grows toward the crown top, like a cantilever loaded along its length.
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
  // Bare limbs: the trunk the crown hid and the frame grown into it, with snow along their tops.
  const bare: Record<'trunk' | 'bark', string[]> = { trunk: [], bark: [] }
  const bareSnow: string[] = []
  for (const [key, { x, y, tone }] of limbs) {
    const shown = cells.get(key)
    if (shown && (shown.tone === 'trunk' || shown.tone === 'bark')) continue
    bare[tone === 'bark' ? 'bark' : 'trunk'].push(`M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`)
    if (!limbs.has(cellKey(x / pixel, y / pixel - 1)))
      bareSnow.push(`M${x} ${y}h${pixel}v${Math.max(1, Math.floor(pixel / 2))}h${-pixel}Z`)
  }
  // Early spring opens flowers and catkins at the twig ends, before the leaves.
  const bloom = [...buds]
    .map(key => {
      const { x, y } = limbs.get(key)!
      return `M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`
    })
    .join('')
  const limbMarkup = deciduous
    ? `<g class="trail-tree-limbs">${(['trunk', 'bark'] as const)
        .filter(tone => bare[tone].length)
        .map(tone => `<path class="trail-tree-${tone}" d="${bare[tone].join('')}"/>`)
        .join(
          '',
        )}<path class="trail-tree-snow" d="${bareSnow.join('')}"/><path class="trail-tree-bud" d="${bloom}"/></g>`
    : ''
  const shadow = Math.round(8 + size * breadth * 4)
  return {
    form,
    tone,
    extent,
    trunkRadius,
    markup:
      `<path class="trail-tree-shadow" d="M${-shadow} 0h${shadow * 2}v2h-3v2h${-shadow * 2 + 6}V2h-3Z"/>` +
      '<g class="trail-tree-motion">' +
      (['trunk', 'bark'] as const)
        .map(tone => `<path class="trail-tree-${tone}" d="${paths[tone].join('')}"/>`)
        .join('') +
      `<path class="trail-tree-snow" d="${woodySnow.join('')}"/>` +
      limbMarkup +
      `<g class="trail-tree-canopy"><path class="trail-tree-hollow" d="${hollow.join('')}"/>${canopy}</g>` +
      '</g>',
  }
}

export function forestShape(settings: ForestSettings) {
  return [
    settings.depth,
    settings.angle,
    settings.size,
    settings.curl,
    settings.branching,
    settings.variation,
    settings.dapple,
  ].join('|')
}

export function createTreePainter(): typeof tree {
  let signature = ''
  const shapes = new Map<string, ReturnType<typeof tree>>()
  return (kind, seed, height, settings, minPixel = 1, grown = forestLimits.depth.max) => {
    const next = forestShape(settings)
    if (next !== signature) {
      shapes.clear()
      signature = next
    }
    const key = `${kind}|${seed}|${height}|${minPixel}|${grown}`
    let shape = shapes.get(key)
    if (!shape) {
      shape = tree(kind, seed, height, settings, minPixel, grown)
      shapes.set(key, shape)
    }
    return shape
  }
}

export type TreePlacement = {
  id: string
  x: number
  y: number
  standalone: boolean
  foothill: boolean
  cluster: number
  kind: TreeKind
  treeSeed: number
  height: number
  // Share of full height reached, the forest year's age of the tree, and the fallen trunk under it.
  growth: number
  generation: number
  // Forest years since the tree rooted, when succession placed it.
  age?: number
  log: number | null
}

export function treeSprite(
  landscape: Landscape,
  placement: TreePlacement,
  settings: ForestSettings,
  paint: typeof tree,
  patch = '',
) {
  const { meadowTop, camp, treeline, alpine } = landscape
  const { id, x, y, standalone, foothill, cluster, kind, treeSeed, height } = placement
  const { growth, generation } = placement
  const grown = Math.round(
    forestLimits.depth.min + (forestLimits.depth.max - forestLimits.depth.min) * growth ** 0.6,
  )
  const { markup, form, tone, extent, trunkRadius } = paint(
    kind,
    treeSeed,
    height,
    settings,
    foothill ? 1 : 2,
    grown,
  )
  const depth = Math.max(0, Math.min(1, (y - meadowTop) / (camp.y - meadowTop)))
  const foothillDepth = Math.max(0, Math.min(1, (y - treeline) / (alpine.base - treeline)))
  const scale = foothill ? 0.2 + foothillDepth * 0.1 : 0.55 + depth * 0.6
  const fit = foothill ? Math.min(1, (y - alpine.top - 4) / (extent * scale)) : 1
  const haze = foothill ? 44 - foothillDepth * 12 : 20 - depth * 15
  const grouping = standalone
    ? 'data-tree-placement="standalone"'
    : `data-tree-placement="grove" data-tree-cluster="${cluster}"`
  const snow =
    !foothill && trailRandom(treeSeed ^ 0x16c73)() < 0.45
      ? trailSnowBank(treeSeed, (standalone ? 112 : 70) * scale, (18 + depth * 10) * scale)
      : ''
  return `<g class="trail-tree" ${patch ? `data-trail-chunk="${patch}"` : ''} data-tree-id="${id}" data-tree="${kind}" ${grouping} data-tree-depth="${foothill ? 'foothill' : 'meadow'}" data-tree-scale="${scale.toFixed(3)}" data-base-trunk-radius="${(trunkRadius * scale * fit).toFixed(3)}" data-trunk-radius="${(trunkRadius * scale * fit).toFixed(3)}" data-growth-limit="${foothill ? Math.max(1, Math.min(1.3, (y - treeline - 1) / (extent * scale * fit))).toFixed(4) : '1.3'}" data-ground-x="${x}" data-ground-y="${y}" data-tree-form="${form}" data-tree-tone="${tone}" data-tree-growth="${growth.toFixed(2)}" data-tree-generation="${generation}" ${placement.age === undefined ? '' : `data-tree-age="${Math.floor(placement.age)}"`} style="--trail-tree-haze: ${haze.toFixed(0)}%" transform="translate(${x} ${y})">${snow}<g class="trail-tree-perspective" transform="scale(${scale.toFixed(3)})"><g class="trail-tree-growth" transform="scale(${fit.toFixed(4)})"><g class="trail-tree-community"><g class="trail-tree-touch">${markup}</g></g></g></g></g>`
}

export function* trailPatchTrees(
  landscape: Landscape,
  seed: number,
  region: { left: number; top: number; right: number; bottom: number; id: string },
  settings: ForestSettings,
  neighbours: Point[],
  walks: Point[],
  paint: typeof tree,
) {
  const random = trailRandom(seed)
  const top = Math.max(region.top, landscape.meadowTop + 64)
  const span = Math.max(0, region.bottom - top - 30)
  const population = (density: number) =>
    Math.min(
      24,
      Math.round(
        (density * (region.right - region.left) * span) /
          (1280 * Math.max(1, landscape.height - landscape.meadowTop)),
      ),
    )
  const count = population(settings.density)
  const capacity = population(forestLimits.density.max)
  const roots = [...neighbours]
  const center = {
    x: region.left + 50 + random() * (region.right - region.left - 100),
    y: top + span * (0.25 + random() * 0.5),
  }
  for (let index = 0; index < capacity + 3; index++) {
    const foothill = index >= capacity && region.top <= landscape.treeline
    if (index >= capacity && !foothill) continue
    const standalone = index % 4 === 3 || foothill
    let root: Point | undefined
    for (let attempt = 0; attempt < 48; attempt++) {
      const angle = random() * Math.PI * 2
      const spread = Math.sqrt(random())
      const x =
        standalone || attempt > 24
          ? region.left + 16 + random() * (region.right - region.left - 32)
          : center.x + Math.cos(angle) * 60 * spread
      const y = foothill
        ? landscape.treeline + (landscape.alpine.base - landscape.treeline) * (0.4 + random() * 0.5)
        : standalone || attempt > 24
          ? top + random() * span
          : center.y + Math.sin(angle) * 50 * spread
      const candidate = { x: Math.round(x), y: Math.round(y) }
      if (
        x < region.left + 12 ||
        x > region.right - 12 ||
        (!foothill && (y < top || y > region.bottom - 28))
      )
        continue
      if (
        pondDistance(candidate, landscape.pond) < 1.9 ||
        creekDistance(candidate, landscape) <
          Math.max(18, creekSection(candidate, landscape).waterHalf + 12) ||
        Math.hypot(x - landscape.camp.x, y - landscape.camp.y) < 110 ||
        nearestDistance(candidate, roots) < (standalone ? 26 : 16) ||
        nearestDistance(candidate, walks) < 20
      )
        continue
      root = candidate
      break
    }
    if (!root) continue
    roots.push(root)
    const choices = foothill ? foothillKinds : stands[Math.floor(random() * stands.length)]
    const placement: TreePlacement = {
      ...root,
      id: `tree-patch-${region.id}-${index}`,
      standalone,
      foothill,
      cluster: seed,
      kind: choices[Math.floor(random() * choices.length)],
      treeSeed: Math.floor(random() * 4294967296),
      height: foothill ? 24 + random() * 10 : 32 + random() * 22,
      growth: 1,
      generation: 0,
      log: null,
    }
    if (index < count || foothill)
      yield {
        id: placement.id,
        bottom: placement.y + 12,
        shape: forestShape(settings),
        render: () => treeSprite(landscape, placement, settings, paint, region.id),
      }
  }
}

export function planTrailTrees(
  landscape: Landscape,
  seed: number,
  walks: Point[] = [],
  clearings: (Point & { radius: number })[] = [],
  now = Date.now(),
  water: ReadonlyMap<string, readonly { count: number; at: number }[]> = new Map(),
): TreePlacement[] {
  const { width, meadowTop, camp, groves, pond, alpine, treeline } = landscape
  const clearing = camp.y - meadowTop
  const random = trailRandom(seed ^ 0x7f4a7c15)
  // The groves keep their size on a wide meadow, so a wide screen takes up to half again as many
  // trees to keep the stands as thick. Plan the maximum population once so density
  // changes select stable positions without rerolling edge trees or the ground cover.
  const count = Math.round(forestLimits.density.max * Math.min(1.5, Math.max(1, width / 1280)))
  const roots: Point[] = []
  const placements = Array.from({ length: count }, (_, index) => {
    // One tree in four stands alone; the rest fill the groves, so more trees thicken the stands.
    const standalone = index % 4 === 3
    const foothill = standalone && index < 24
    const cluster = (index - Math.floor(index / 4)) % groves.length
    const grove = groves[cluster]
    let root: Point = grove
    if (foothill) {
      root = {
        x:
          alpine.left +
          alpine.width * (0.12 + Math.floor(index / 4) * 0.145 + (random() - 0.5) * 0.03),
        y: treeline + (alpine.base - treeline) * (0.55 + random() * 0.4),
      }
    } else {
      let bestClearance = -Infinity
      for (let attempt = 0; attempt < 140; attempt++) {
        const angle = Math.floor(index / groves.length) * 2.399963 + (random() - 0.5) * 1.8
        const radius = Math.sqrt(random()) * (standalone ? 1 : 0.65) * (attempt < 70 ? 1 : 1.8)
        const spread = standalone || attempt >= 100
        const x = spread
          ? 26 + random() * (width - 56)
          : grove.x + Math.cos(angle) * grove.rx * radius
        const y = spread
          ? meadowTop + 64 + random() * Math.max(1, clearing - 100)
          : grove.y + Math.sin(angle) * grove.ry * radius
        const candidate = onDryLand({ x, y: y + groundOffset(x, landscape, y) }, landscape, 1.9)
        if (
          pondDistance(candidate, pond) <= 1.85 ||
          creekDistance(candidate, landscape) <=
            Math.max(16, creekSection(candidate, landscape).waterHalf + 12) ||
          Math.hypot(candidate.x - camp.x, candidate.y - camp.y) <= 102 ||
          clearings.some(
            site => Math.hypot(candidate.x - site.x, candidate.y - site.y) < site.radius,
          )
        )
          continue
        const rootDistance = nearestDistance(candidate, roots)
        const walkDistance = nearestDistance(candidate, walks)
        const groveDistance = standalone
          ? Math.min(
              ...groves.map(grove =>
                Math.hypot((candidate.x - grove.x) / grove.rx, (candidate.y - grove.y) / grove.ry),
              ),
            )
          : Infinity
        const clearance = Math.min(
          rootDistance / (standalone ? 28 : 16),
          walkDistance / 15,
          groveDistance,
        )
        if (clearance > bestClearance) {
          root = candidate
          bestClearance = clearance
        }
        if (
          rootDistance > (attempt < 70 ? (standalone ? 28 : 16) : 8) &&
          walkDistance > (attempt < 70 ? 15 : 6) &&
          groveDistance > (attempt < 70 ? 1.1 : 0.9)
        ) {
          root = candidate
          break
        }
      }
    }
    roots.push(root)
    const identity = trailRandom(0x777034b3 ^ Math.imul(index + 1, 16807))
    const pick = identity()
    // Willows take the pond shore; groves keep to their stand; the foothills are montane conifer.
    // One lone tree in eight is a snag left standing after a burn.
    const upland = kinds.filter(kind => kind !== 'willow' && kind !== 'snag')
    const kind: TreeKind = foothill
      ? foothillKinds[Math.floor(pick * foothillKinds.length)]
      : pondDistance(root, pond) < 2.8 && pick < 0.7
        ? 'willow'
        : standalone
          ? pick < 0.125
            ? 'snag'
            : upland[Math.floor(((pick - 0.125) / 0.875) * upland.length)]
          : stands[cluster % stands.length][Math.floor(pick * 4)]
    const treeSeed = Math.floor(identity() * 4294967296)
    const height = foothill ? 20 + identity() * 10 : 30 + identity() * 26
    return {
      pool: foothill
        ? null
        : kind === 'willow'
          ? waterside
          : standalone || kind === 'snag'
            ? upland
            : stands[cluster % stands.length],
      id: `tree-${index + 1}`,
      x: Math.round(root.x),
      y: foothill
        ? Math.min(Math.floor(alpine.base), Math.max(Math.ceil(treeline), Math.round(root.y)))
        : Math.max(Math.ceil(meadowTop + 64), Math.round(root.y)),
      standalone,
      foothill,
      cluster,
      kind,
      treeSeed,
      height,
      growth: 1,
      generation: 0,
      log: null,
    }
  })
  // Conifers stand at both ends of the meadow, where the groves leave open ground, thickest on the
  // far half below the range.
  const edge = trailRandom(seed ^ 0x2f6b1c3d)
  const perSide = 3 + Math.round(width / 500)
  for (const side of [0, 1])
    for (let index = 0; index < perSide; index++) {
      for (let attempt = 0; attempt < 40; attempt++) {
        const reach = 16 + edge() * width * 0.12
        const x = side ? width - reach : reach
        const y = meadowTop + 64 + edge() ** 1.4 * clearing * 0.6
        const candidate = onDryLand({ x, y: y + groundOffset(x, landscape, y) }, landscape, 1.9)
        if (
          pondDistance(candidate, pond) <= 1.85 ||
          creekDistance(candidate, landscape) <=
            Math.max(16, creekSection(candidate, landscape).waterHalf + 12) ||
          Math.hypot(candidate.x - camp.x, candidate.y - camp.y) <= 102 ||
          clearings.some(
            site => Math.hypot(candidate.x - site.x, candidate.y - site.y) < site.radius,
          ) ||
          nearestDistance(candidate, roots) <= 14 ||
          nearestDistance(candidate, walks) <= 15
        )
          continue
        roots.push(candidate)
        const pick = edge()
        placements.push({
          id: `tree-edge-${side}-${index + 1}`,
          x: Math.round(candidate.x),
          y: Math.max(Math.ceil(meadowTop + 64), Math.round(candidate.y)),
          standalone: true,
          foothill: false,
          cluster: 0,
          kind: pick < 0.5 ? 'pine' : pick < 0.75 ? 'spruce' : 'fir',
          treeSeed: Math.floor(edge() * 4294967296),
          height: 30 + edge() * 26,
          growth: 1,
          generation: 0,
          log: null,
          pool: edgeKinds,
        })
        break
      }
    }
  return grow(placements, landscape, forestYear(now), water)
}

const waterside: readonly TreeKind[] = ['willow', 'willow', 'birch', 'aspen']
const edgeKinds: readonly TreeKind[] = ['pine', 'pine', 'spruce', 'fir']

// Age every slot to the forest year, then let the canopy shade its understory: a sapling under a
// taller crown grows slowly, the more so the less shade it tolerates.
function grow(
  planned: (TreePlacement & { pool: readonly TreeKind[] | null })[],
  landscape: Landscape,
  year: number,
  water: ReadonlyMap<string, readonly { count: number; at: number }[]>,
) {
  const storms = landscape.groves.map((_, grove) => groveStorms(grove, Math.floor(year)))
  const placements = planned.map(({ pool, ...placement }, slot) => {
    if (!pool || year < 0) return placement
    const edge = placement.id.startsWith('tree-edge-')
    const occupant = slotOccupant(
      placement,
      slot,
      pool,
      placement.standalone || edge ? [] : storms[placement.cluster],
      year,
      placement.standalone && !edge,
      water.get(placement.id),
    )
    return { ...placement, ...occupant }
  })
  for (const sapling of placements) {
    if (sapling.foothill || sapling.growth >= 0.8) continue
    let shade = 0
    for (const canopy of placements) {
      if (canopy === sapling || canopy.foothill || canopy.growth <= sapling.growth + 0.2) continue
      const reach = canopy.height * 0.7
      if (Math.abs(canopy.x - sapling.x) < reach && Math.abs(canopy.y - sapling.y) < reach * 0.45)
        shade++
    }
    if (!shade) continue
    const slowed = 1 - Math.min(0.6, 0.3 * shade) * (1 - shadeTolerance(sapling.kind))
    sapling.growth *= slowed
    sapling.height = Math.round(16 + (sapling.height - 16) * slowed)
  }
  return placements
}

export function* trailTreeSprites(
  landscape: Landscape,
  seed: number,
  settings: ForestSettings = sceneSettings.forest,
  walks: Point[] = [],
  clearings: (Point & { radius: number })[] = [],
  paint: typeof tree = tree,
  nearFirst = false,
  placements = planTrailTrees(landscape, seed, walks, clearings),
): Generator<ForestSprite> {
  const count = Math.round(
    bounded(settings.density, 'density') * Math.min(1.5, Math.max(1, landscape.width / 1280)),
  )
  const shape = forestShape(settings)
  const wanted = (placement: TreePlacement) =>
    placement.id.startsWith('tree-edge-') || Number(placement.id.slice(5)) <= count
  const sprites: ForestSprite[] = placements
    .filter(wanted)
    .map(placement => ({
      id: placement.id,
      bottom: placement.y + 12,
      shape: `${shape}|${placement.x}|${placement.y}|${placement.kind}|${placement.treeSeed}|${placement.height.toFixed(1)}`,
      render: () => treeSprite(landscape, placement, settings, paint),
    }))
  for (const placement of placements)
    if (placement.log !== null && !placement.foothill && wanted(placement)) {
      const sprite = trailDeadfall(landscape, placement, placement.log)
      sprites.push({
        id: sprite.id,
        bottom: sprite.bottom,
        shape: sprite.markup,
        render: () => sprite.markup,
      })
    }
  for (const sprite of trailUnderstory(landscape, seed, placements, walks, clearings))
    sprites.push({
      id: sprite.id,
      bottom: sprite.bottom,
      shape: sprite.markup,
      render: () => sprite.markup,
    })
  sprites.sort((a, b) => (nearFirst ? b.bottom - a.bottom : a.bottom - b.bottom))
  yield* sprites
}

export function trailTrees(
  landscape: Landscape,
  seed: number,
  settings: ForestSettings = sceneSettings.forest,
  walks: Point[] = [],
  clearings: (Point & { radius: number })[] = [],
  paint: typeof tree = tree,
) {
  return Array.from(trailTreeSprites(landscape, seed, settings, walks, clearings, paint), sprite =>
    sprite.render(),
  ).join('')
}

export function trailFoothillWoodland(
  landscape: Landscape,
  seed: number,
  settings: ForestSettings,
  walks: Point[],
  paint: typeof tree = tree,
  neighbours: Point[] = [],
) {
  const { width, alpine, meadowTop, camp, pond } = landscape
  // Further trees need fewer branches at their screen size.
  const distant = { ...settings, depth: Math.min(5, settings.depth) }
  const random = trailRandom(seed ^ 0x5a83c917)
  const near = walks.filter(point => point.y >= alpine.base - 14 && point.y < meadowTop + 64)
  const trackPoints: Point[] = []
  const tracks = [18, 50].map((offset, row) => {
    const anchors = Array.from({ length: Math.max(3, Math.ceil(width / 280)) }, (_, index) => {
      const target = {
        x: (width * (index + 0.5)) / Math.max(3, Math.ceil(width / 280)),
        y: alpine.base + offset,
      }
      return near.reduce(
        (best, point) =>
          Math.hypot(point.x - target.x, point.y - target.y) <
          Math.hypot(best.x - target.x, best.y - target.y)
            ? point
            : best,
        near[0] ?? target,
      )
    }).sort((left, right) => left.x - right.x)
    let d = `M${anchors[0].x} ${anchors[0].y}`
    for (let index = 1; index < anchors.length; index++) {
      const from = anchors[index - 1]
      const to = anchors[index]
      const bend = { x: (from.x + to.x) / 2, y: alpine.base + offset + (random() - 0.5) * 12 }
      d += `Q${bend.x} ${bend.y} ${to.x} ${to.y}`
      const segments = Math.max(1, Math.ceil(Math.hypot(to.x - from.x, to.y - from.y) / 6))
      for (let step = 0; step <= segments; step++) {
        const t = step / segments
        trackPoints.push({
          x: (1 - t) ** 2 * from.x + 2 * t * (1 - t) * bend.x + t * t * to.x,
          y: (1 - t) ** 2 * from.y + 2 * t * (1 - t) * bend.y + t * t * to.y,
        })
      }
    }
    return {
      id: `woodland-track-${row}`,
      bottom: alpine.base + offset,
      shape: d,
      render: () => `<path class="trail-mountain-track" data-woodland-track="${row}" d="${d}"/>`,
    }
  })
  const trees: { x: number; y: number; sprite: ForestSprite; active: boolean }[] = []
  const woodland = trailRandom(seed ^ 0x492671b3)
  const band = Math.min(148, Math.max(60, (camp.y - alpine.base) * 0.3))
  const floor = alpine.base + 6
  const front = alpine.base + band
  const perCluster = Math.max(2, Math.round(settings.density / 12))
  const plant = (center: Point, rx: number, ry: number, cluster: string, active = true) => {
    let root: Point | undefined
    for (let attempt = 0; attempt < 32; attempt++) {
      const angle = woodland() * Math.PI * 2
      const spread = Math.sqrt(woodland())
      const x =
        cluster === 'standalone'
          ? 12 + woodland() * (width - 24)
          : center.x + Math.cos(angle) * rx * spread
      const y = center.y + Math.sin(angle) * ry * spread
      const candidate = { x, y: y + groundOffset(x, landscape, y) * 0.3 }
      if (
        candidate.x < 12 ||
        candidate.x > width - 12 ||
        candidate.y < floor ||
        candidate.y > front
      )
        continue
      const depth = (candidate.y - floor) / (front - floor)
      if (
        pondDistance(candidate, pond) > 1.9 &&
        creekDistance(candidate, landscape) > creekSection(candidate, landscape).waterHalf + 12 &&
        Math.hypot(candidate.x - camp.x, candidate.y - camp.y) > 102 &&
        nearestDistance(candidate, walks) > 24 + depth * 8 &&
        nearestDistance(candidate, trackPoints) > 10 &&
        nearestDistance(candidate, neighbours) > 22 &&
        nearestDistance(candidate, trees) > (cluster === 'standalone' ? 30 : 6 + depth * 5)
      ) {
        root = candidate
        break
      }
    }
    if (!root) return
    const variant = Math.floor(woodland() * 8)
    const kind = woodlandKinds[variant % woodlandKinds.length]
    const depth = (root.y - floor) / (front - floor)
    const depthScale = 0.28 + depth * 0.37
    const size = 0.7 + woodland() * 0.6
    const scale = depthScale * size
    const point = root
    const id = `woodland-${trees.length}`
    trees.push({
      x: root.x,
      y: root.y,
      active,
      sprite: {
        id,
        bottom: root.y + 12,
        shape: `${forestShape(distant)}|${point.x}|${point.y}|${scale}`,
        render: () => {
          const shape = paint(
            kind,
            0x24c039af ^ Math.imul(variant + 1, 16807),
            20 + (variant % 3) * 5,
            distant,
            2,
          )
          return `<g class="trail-woodland-tree" data-trail-layer="${id}" data-woodland-tree="" data-woodland-id="${id}" data-woodland-cluster="${cluster}" data-tree="${kind}" data-tree-tone="${shape.tone}" data-tree-scale="${scale.toFixed(3)}" data-tree-depth-scale="${depthScale.toFixed(3)}" data-tree-size="${size.toFixed(3)}" data-trunk-radius="${(shape.trunkRadius * scale).toFixed(3)}" data-ground-x="${point.x.toFixed(2)}" data-ground-y="${point.y.toFixed(2)}" style="--trail-tree-haze:${(44 - depth * 28).toFixed(0)}%" transform="translate(${point.x.toFixed(2)} ${point.y.toFixed(2)}) scale(${scale.toFixed(3)})">${shape.markup}</g>`
        },
      },
    })
  }
  let x = 32 + woodland() * 76
  let cluster = 0
  const depthPhase = woodland()
  const depthStep = (Math.sqrt(5) - 1) / 2
  while (x < width - 12) {
    // Spread the groves over the slope even when an opening has only a few clusters.
    const depth = (depthPhase + cluster * depthStep + (woodland() - 0.5) * 0.16 + 1) % 1
    const center = { x, y: floor + depth * (front - floor) }
    const rx = 22 + woodland() * 44
    const ry = 12 + woodland() * 24
    const extra = Math.floor(woodland() * 5)
    const count = Math.max(3, perCluster + extra)
    const capacity = Math.max(3, Math.round(forestLimits.density.max / 12) + extra)
    for (let index = 0; index < capacity; index++)
      plant(center, rx, ry, String(cluster), index < count)
    // Stands follow each other closely along the foot of the range, with the odd wide clearing.
    x += woodland() < 0.2 ? 180 + woodland() * 120 : 60 + woodland() * 80
    cluster++
  }
  const singles = Math.max(1, Math.round(width / 430))
  for (let index = 0; index < singles; index++)
    plant(
      {
        x: 16 + woodland() * (width - 32),
        y: floor + ((index + woodland()) / singles) * (front - floor),
      },
      20,
      16,
      'standalone',
    )
  return [
    ...tracks,
    ...trees
      .filter(tree => tree.active)
      .sort((left, right) => left.y - right.y)
      .map(tree => tree.sprite),
  ]
}
