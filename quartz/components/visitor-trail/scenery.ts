import { groundOffset, onDryLand, pondDistance, type Landscape } from './landscape'
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
  density: 48,
  depth: 9,
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
  density: { min: 8, max: 72, step: 1 },
  depth: { min: 2, max: 9, step: 1 },
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

function nearestDistance(point: Point, points: Point[]) {
  let distance = Infinity
  for (const other of points) {
    const squared = (point.x - other.x) ** 2 + (point.y - other.y) ** 2
    if (squared < distance) distance = squared
  }
  return Math.sqrt(distance)
}

function tree(
  kind: TreeKind,
  seed: number,
  height: number,
  settings: ForestSettings,
  minPixel = 1,
) {
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
    word = word.replace(/[XF]/g, symbol => (symbol === 'X' ? rule() : 'FF'))
  }
  const spread = kind === 'cypress' ? 0.45 : kind === 'spruce' ? 0.7 : kind === 'pine' ? 0.85 : 1
  const angle = (bounded(settings.angle, 'angle') * spread * breadth * Math.PI) / 180
  // Segment counts grow with depth; keep curvature per branch at the same scale.
  const curl = (bounded(settings.curl, 'curl') * Math.PI) / (180 * 2 ** depth)
  let turtle: Turtle = { x: 0, y: 0, heading: -Math.PI / 2 + lean, bend: handedness * 0.4 }
  const stack: Turtle[] = []
  const branches: Branch[] = []
  const tips: Tip[] = []
  let top = 0
  for (const symbol of word) {
    if (symbol === 'F') {
      const from = { x: turtle.x, y: turtle.y }
      turtle.heading = Math.max(-Math.PI + 0.1, Math.min(-0.1, turtle.heading + turtle.bend * curl))
      const length = 0.78 ** stack.length * (1 + (random() - 0.5) * variation * 0.35)
      turtle.x += Math.cos(turtle.heading) * length
      turtle.y += Math.sin(turtle.heading) * length
      top = Math.min(top, turtle.y)
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
  const size = (bounded(settings.size, 'size') / 100) * (1 + (random() - 0.5) * variation * 0.25)
  const scale = ((height - 8) * size) / Math.max(1, -top)
  const pixel = Math.max(minPixel, Math.ceil((height * size) / 54))
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
  const crowns = new Set<string>()
  tips.forEach(tip => {
    // Buds within one raster cell at the same depth share a crown.
    const crown = `${Math.round((tip.x * scale) / pixel)},${Math.round((tip.y * scale) / pixel)},${tip.level}`
    if (crowns.has(crown)) return
    crowns.add(crown)
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
  let crownTop = 0
  let trunkRadius = 1
  const snow: string[] = []
  for (const { x, y, tone } of cells.values()) {
    crownTop = Math.min(crownTop, y)
    if ((tone === 'trunk' || tone === 'bark') && y >= -pixel * 2)
      trunkRadius = Math.max(trunkRadius, Math.abs(x), Math.abs(x + pixel))
    paths[tone].push(`M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`)
    const woody = tone === 'trunk' || tone === 'bark'
    if ((!woody || y < -height * size * 0.3) && !cells.has(`${x / pixel},${y / pixel - 1}`))
      snow.push(`M${x} ${y}h${pixel}v${Math.max(1, Math.floor(pixel / 2))}h${-pixel}Z`)
  }
  const shadow = Math.round(8 + size * breadth * 4)
  return {
    form,
    tone,
    extent: Math.max(1, -crownTop),
    trunkRadius,
    markup:
      `<path class="trail-tree-shadow" d="M${-shadow} 0h${shadow * 2}v2h-3v2h${-shadow * 2 + 6}V2h-3Z"/>` +
      '<g class="trail-tree-motion">' +
      Object.entries(paths)
        .filter(([tone]) => tone === 'trunk' || tone === 'bark')
        .map(([tone, marks]) => `<path class="trail-tree-${tone}" d="${marks.join('')}"/>`)
        .join('') +
      '<g class="trail-tree-canopy">' +
      Object.entries(paths)
        .filter(([tone]) => tone !== 'trunk' && tone !== 'bark')
        .map(([tone, marks]) => `<path class="trail-tree-${tone}" d="${marks.join('')}"/>`)
        .join('') +
      `<path class="trail-tree-snow" d="${snow.join('')}"/>` +
      '</g>' +
      '</g>',
  }
}

export function createTreePainter(): typeof tree {
  let signature = ''
  const shapes = new Map<string, ReturnType<typeof tree>>()
  return (kind, seed, height, settings, minPixel = 1) => {
    const next = [
      settings.depth,
      settings.angle,
      settings.size,
      settings.curl,
      settings.branching,
      settings.variation,
      settings.dapple,
    ].join('|')
    if (next !== signature) {
      shapes.clear()
      signature = next
    }
    const key = `${kind}|${seed}|${height}|${minPixel}`
    let shape = shapes.get(key)
    if (!shape) {
      shape = tree(kind, seed, height, settings, minPixel)
      shapes.set(key, shape)
    }
    return shape
  }
}

export function* trailTreeSprites(
  landscape: Landscape,
  seed: number,
  settings: ForestSettings = defaultForestSettings,
  walks: Point[] = [],
  clearings: (Point & { radius: number })[] = [],
  paint: typeof tree = tree,
) {
  const { width, meadowTop, camp, groves, pond, alpine, treeline } = landscape
  const clearing = camp.y - meadowTop
  const random = trailRandom(seed ^ 0x7f4a7c15)
  const count = bounded(settings.density, 'density')
  const roots: Point[] = []
  const placements = Array.from({ length: count }, (_, index) => {
    const standalone = index % 4 === 3 || index >= 32
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
        const candidate = onDryLand({ x, y: y + groundOffset(x, landscape) * 0.3 }, landscape, 1.9)
        if (
          pondDistance(candidate, pond) <= 1.85 ||
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
    const kind =
      kinds[
        foothill
          ? Math.floor(identity() * 2)
          : standalone
            ? Math.floor(identity() * kinds.length)
            : (cluster + Math.floor(identity() * 2)) % kinds.length
      ]
    const treeSeed = Math.floor(identity() * 4294967296)
    const height = foothill ? 20 + identity() * 10 : 30 + identity() * 26
    return {
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
    }
  }).sort((a, b) => a.y - b.y)
  for (const { id, x, y, standalone, foothill, cluster, kind, treeSeed, height } of placements) {
    const { markup, form, tone, extent, trunkRadius } = paint(
      kind,
      treeSeed,
      height,
      settings,
      foothill ? 4 : 1,
    )
    const depth = Math.max(0, Math.min(1, (y - meadowTop) / clearing))
    const foothillDepth = Math.max(0, Math.min(1, (y - treeline) / (alpine.base - treeline)))
    const scale = foothill ? 0.2 + foothillDepth * 0.1 : 0.55 + depth * 0.6
    const fit = foothill ? Math.min(1, (y - treeline - 1) / (extent * scale)) : 1
    const opacity = foothill ? 0.56 + foothillDepth * 0.12 : 0.8 + depth * 0.15
    const placement = standalone
      ? 'data-tree-placement="standalone"'
      : `data-tree-placement="grove" data-tree-cluster="${cluster}"`
    yield `<g class="trail-tree" data-tree-id="${id}" data-tree="${kind}" ${placement} data-tree-depth="${foothill ? 'foothill' : 'meadow'}" data-tree-scale="${scale.toFixed(3)}" data-base-trunk-radius="${(trunkRadius * scale * fit).toFixed(3)}" data-trunk-radius="${(trunkRadius * scale * fit).toFixed(3)}" data-growth-limit="${foothill ? Math.max(1, Math.min(1.3, (y - treeline - 1) / (extent * scale * fit))).toFixed(4) : '1.3'}" data-ground-x="${x}" data-ground-y="${y}" data-tree-form="${form}" data-tree-tone="${tone}" style="--trail-tree-opacity: ${opacity.toFixed(2)}" transform="translate(${x} ${y})"><g class="trail-tree-perspective" transform="scale(${scale.toFixed(3)})"><g class="trail-tree-growth" transform="scale(${fit.toFixed(4)})"><g class="trail-tree-community"><g class="trail-tree-touch">${markup}</g></g></g></g></g>`
  }
}

export function trailTrees(
  landscape: Landscape,
  seed: number,
  settings: ForestSettings = defaultForestSettings,
  walks: Point[] = [],
  clearings: (Point & { radius: number })[] = [],
  paint: typeof tree = tree,
) {
  return [...trailTreeSprites(landscape, seed, settings, walks, clearings, paint)].join('')
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
  const tracks = [18, 50]
    .map((offset, row) => {
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
      return `<path class="trail-mountain-track" data-woodland-track="${row}" d="${d}"/>`
    })
    .join('')
  const trees: { x: number; y: number; markup: string }[] = []
  const woodland = trailRandom(seed ^ 0x492671b3)
  const band = Math.min(148, Math.max(60, (camp.y - alpine.base) * 0.3))
  const floor = alpine.base + 6
  const front = alpine.base + band
  const perCluster = Math.max(2, Math.round(settings.density / 12))
  const plant = (center: Point, rx: number, ry: number, cluster: string) => {
    let root: Point | undefined
    for (let attempt = 0; attempt < 32; attempt++) {
      const angle = woodland() * Math.PI * 2
      const spread = Math.sqrt(woodland())
      const candidate = {
        x:
          cluster === 'standalone'
            ? 12 + woodland() * (width - 24)
            : center.x + Math.cos(angle) * rx * spread,
        y: center.y + Math.sin(angle) * ry * spread,
      }
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
    const kind = kinds[variant % kinds.length]
    const shape = paint(
      kind,
      0x24c039af ^ Math.imul(variant + 1, 16807),
      20 + (variant % 3) * 5,
      distant,
      2,
    )
    const depth = (root.y - floor) / (front - floor)
    const depthScale = 0.28 + depth * 0.37
    const size = 0.7 + woodland() * 0.6
    const scale = depthScale * size
    trees.push({
      x: root.x,
      y: root.y,
      markup: `<g class="trail-woodland-tree" data-woodland-tree="" data-woodland-id="woodland-${trees.length}" data-woodland-cluster="${cluster}" data-tree="${kind}" data-tree-tone="${shape.tone}" data-tree-scale="${scale.toFixed(3)}" data-tree-depth-scale="${depthScale.toFixed(3)}" data-tree-size="${size.toFixed(3)}" data-trunk-radius="${(shape.trunkRadius * scale).toFixed(3)}" data-ground-x="${root.x.toFixed(2)}" data-ground-y="${root.y.toFixed(2)}" style="--trail-tree-opacity:${(0.56 + depth * 0.28).toFixed(2)}" transform="translate(${root.x.toFixed(2)} ${root.y.toFixed(2)}) scale(${scale.toFixed(3)})">${shape.markup}</g>`,
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
    const count = Math.max(2, perCluster - 2 + Math.floor(woodland() * 5))
    for (let index = 0; index < count; index++) plant(center, rx, ry, String(cluster))
    x += woodland() < 0.35 ? 235 + woodland() * 160 : 105 + woodland() * 110
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
  return (
    tracks +
    trees
      .sort((left, right) => left.y - right.y)
      .map(tree => tree.markup)
      .join('')
  )
}
