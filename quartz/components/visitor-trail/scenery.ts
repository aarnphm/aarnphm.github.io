import { trailRandom } from './random'

export type ForestSettings = {
  density: number
  depth: number
  angle: number
  size: number
  curl: number
  branching: number
  variation: number
  dapple: number
}
export const defaultForestSettings: ForestSettings = {
  density: 24,
  depth: 3,
  angle: 26,
  size: 150,
  curl: 4,
  branching: 85,
  variation: 70,
  dapple: 65,
}
export const forestLimits: Record<
  keyof ForestSettings,
  { min: number; max: number; step: number }
> = {
  density: { min: 8, max: 40, step: 1 },
  depth: { min: 2, max: 5, step: 1 },
  angle: { min: 15, max: 45, step: 1 },
  size: { min: 60, max: 240, step: 5 },
  curl: { min: 0, max: 40, step: 1 },
  branching: { min: 0, max: 100, step: 5 },
  variation: { min: 0, max: 100, step: 5 },
  dapple: { min: 0, max: 100, step: 5 },
}

type TreeKind = 'pine' | 'spruce' | 'cypress' | 'birch' | 'oak'
type TreeForm = 'upright' | 'leaning' | 'forked' | 'open' | 'broad'
type TreeTone = 'sage' | 'lichen' | 'ochre'
type Point = { x: number; y: number }
type Tip = Point & { level: number }
type Turtle = Point & { heading: number; bend: number }
type Branch = { from: Point; to: Point; level: number }
type Tone = 'trunk' | 'bark' | 'light' | 'crown' | 'dark' | 'gold'
const kinds: TreeKind[] = ['pine', 'spruce', 'cypress', 'birch', 'oak']
const forms: TreeForm[] = ['upright', 'leaning', 'forked', 'open', 'broad']
const tones: TreeTone[] = ['sage', 'lichen', 'ochre']
const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]

function canopyTone(x: number, y: number, radius: number, pixel: number, dapple: number): Tone {
  const brightness = Math.max(0, Math.min(1, 0.52 - (x / radius) * 0.23 - (y / radius) * 0.16))
  const column = ((Math.round(x / pixel) % 4) + 4) % 4
  const row = ((Math.round(y / pixel) % 4) + 4) % 4
  const threshold = (bayer[row * 4 + column] + 0.5) / 16
  const rim = Math.hypot(x, y) / radius
  if (rim > 0.72 && x - y < 0 && brightness > 0.5 && threshold < dapple * 0.32) return 'gold'
  const level = brightness * 2
  const lower = Math.floor(level)
  const upper = level - lower > 0.5 * (1 - dapple) + threshold * dapple
  const index = Math.min(2, lower + Number(upper))
  return index === 0 ? 'dark' : index === 1 ? 'crown' : 'light'
}

function bounded(value: number, field: keyof ForestSettings) {
  const { min, max } = forestLimits[field]
  return Math.max(min, Math.min(max, Math.round(value)))
}

function tree(kind: TreeKind, seed: number, height: number, settings: ForestSettings) {
  const random = trailRandom(seed)
  const growth = trailRandom(seed ^ 0x9e3779b9)
  const leaves = trailRandom(seed ^ 0xa46f315b)
  const evergreen = kind !== 'birch' && kind !== 'oak'
  const depth = bounded(settings.depth, 'depth')
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
  const rule = () => {
    const bias = branchBias * probability * (1 - probability)
    const left = growth() < probability + bias ? '[+X]' : ''
    const right = growth() < probability - bias ? '[-X]' : ''
    return evergreen ? `F${left}${right}FX` : `F${left}F${right}+X`
  }
  let word = form === 'forked' ? 'FF[+FX][-FX]' : 'FX'
  for (let iteration = 0; iteration < depth; iteration++) {
    word = Array.from(word, symbol =>
      symbol === 'X' ? rule() : symbol === 'F' ? 'FF' : symbol,
    ).join('')
  }
  const spread = kind === 'cypress' ? 0.45 : kind === 'spruce' ? 0.7 : kind === 'pine' ? 0.85 : 1
  const angle = (bounded(settings.angle, 'angle') * spread * breadth * Math.PI) / 180
  // Segment counts grow with depth; keep curvature per branch at the same scale.
  const curl = (bounded(settings.curl, 'curl') * Math.PI) / (180 * 2 ** depth)
  let turtle: Turtle = { x: 0, y: 0, heading: -Math.PI / 2 + lean, bend: handedness * 0.4 }
  const stack: Turtle[] = []
  const branches: Branch[] = []
  const tips: Tip[] = []
  for (const symbol of word) {
    if (symbol === 'F') {
      const from = { x: turtle.x, y: turtle.y }
      turtle.heading = Math.max(-Math.PI + 0.1, Math.min(-0.1, turtle.heading + turtle.bend * curl))
      const length = 0.78 ** stack.length * (1 + (random() - 0.5) * variation * 0.35)
      turtle.x += Math.cos(turtle.heading) * length
      turtle.y += Math.sin(turtle.heading) * length
      branches.push({ from, to: { x: turtle.x, y: turtle.y }, level: stack.length })
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
  const top = Math.min(...branches.map(branch => branch.to.y))
  const size = (bounded(settings.size, 'size') / 100) * (1 + (random() - 0.5) * variation * 0.25)
  const scale = ((height - 8) * size) / Math.max(1, -top)
  const pixel = Math.max(1, Math.ceil((height * size) / 54))
  const cells = new Map<string, { x: number; y: number; tone: Tone }>()
  const mark = (x: number, y: number, tone: Tone) => {
    const cellX = Math.round(x / pixel)
    const cellY = Math.round(y / pixel)
    cells.set(`${cellX},${cellY}`, { x: cellX * pixel, y: cellY * pixel, tone })
  }
  branches.forEach(({ from, to, level }) => {
    const steps = Math.max(1, Math.ceil((Math.hypot(to.x - from.x, to.y - from.y) * scale) / pixel))
    const thickness = Math.max(pixel, 2.6 * size * 0.62 ** level)
    for (let step = 0; step <= steps; step++) {
      const fraction = step / steps
      const x = (from.x + (to.x - from.x) * fraction) * scale
      const y = (from.y + (to.y - from.y) * fraction) * scale
      const bark = kind === 'birch' && Math.floor(-y / pixel) % 5 !== 0
      for (let offset = -thickness / 2; offset <= thickness / 2; offset += pixel)
        mark(x + offset, y, bark ? 'bark' : 'trunk')
    }
  })
  tips.forEach(tip => {
    if (leaves() < (form === 'open' ? 0.3 : 0.06) * variation) return
    const radius = (evergreen ? 4 + leaves() * 2.5 : 5 + leaves() * 3.5) * size * 0.9 ** tip.level
    const fullness = breadth * (0.9 + leaves() * variation * 0.35)
    for (let y = -radius; y <= radius; y += pixel) {
      const edge = 1 + Math.sin(y * 0.8 + tip.x) * variation * 0.12
      const span =
        (evergreen ? (y + radius) * 0.65 : Math.sqrt(Math.max(0, radius * radius - y * y))) *
        fullness *
        edge
      for (let x = -span; x <= span; x += pixel) {
        const rim = Math.abs(x) > span * 0.82
        if (leaves() < variation * (rim ? 0.16 : form === 'open' ? 0.08 : 0.015)) continue
        mark(
          tip.x * scale + x,
          tip.y * scale + y - (evergreen ? radius : 0),
          canopyTone(x, y, radius, pixel, dapple),
        )
      }
    }
  })
  const paths: Record<Tone, string[]> = {
    trunk: [],
    bark: [],
    light: [],
    crown: [],
    dark: [],
    gold: [],
  }
  for (const { x, y, tone } of cells.values()) {
    paths[tone].push(`M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`)
  }
  const shadow = Math.round(8 + size * breadth * 4)
  return {
    form,
    tone,
    markup:
      `<path class="trail-tree-shadow" d="M${-shadow} 0h${shadow * 2}v2h-3v2h${-shadow * 2 + 6}V2h-3Z"/>` +
      Object.entries(paths)
        .map(([tone, marks]) => `<path class="trail-tree-${tone}" d="${marks.join('')}"/>`)
        .join(''),
  }
}

export function trailTrees(
  width: number,
  bottom: number,
  camp: Point,
  seed: number,
  settings: ForestSettings = defaultForestSettings,
) {
  const clearing = camp.y - bottom
  const random = trailRandom(seed ^ 0x7f4a7c15)
  const count = bounded(settings.density, 'density')
  const columns = width < 700 ? 5 : 8
  const rows = Math.ceil(count / columns)
  return Array.from({ length: count }, (_, index) => {
    const x = width * (0.07 + (((index % columns) + 0.2 + random() * 0.6) / columns) * 0.86)
    let y =
      bottom +
      clearing * (0.24 + ((Math.floor(index / columns) + 0.25 + random() * 0.5) / rows) * 0.76)
    if (Math.abs(x - camp.x) < 90 && Math.abs(y - camp.y) < 85) y -= 100
    const kind = kinds[(index + Math.floor(random() * kinds.length)) % kinds.length]
    const treeSeed = Math.floor(random() * 4294967296)
    const height = (width < 700 ? 24 : 30) + random() * 26
    return { x: Math.round(x), y: Math.round(y), kind, treeSeed, height }
  })
    .sort((a, b) => a.y - b.y)
    .map(({ x, y, kind, treeSeed, height }) => {
      const { markup, form, tone } = tree(kind, treeSeed, height, settings)
      const depth = Math.max(0, Math.min(1, (y - bottom) / clearing))
      return `<g class="trail-tree" data-tree="${kind}" data-tree-form="${form}" data-tree-tone="${tone}" style="--trail-tree-opacity: ${(0.8 + depth * 0.15).toFixed(2)}" transform="translate(${x} ${y})">${markup}</g>`
    })
    .join('')
}
