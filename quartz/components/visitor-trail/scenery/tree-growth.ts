import { trailRandom } from '../random'
import { forestLimits, type ForestSettings } from '../scene-settings'
import {
  bareInWinter,
  bounded,
  canopyTone,
  cellKey,
  crownRows,
  forms,
  species,
  tones,
  type Crown,
  type Tip,
  type Tone,
  type TreeKind,
  type Turtle,
} from './species'
import { dressTree } from './tree-canopy'

export function tree(
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
  const lengths = Array.from({ length: depth + 2 }, (_, level) => 0.78 ** level)
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
        const length = lengths[stack.length] * (1 + (random() - 0.5) * variation * 0.35)
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
  const girths = lengths.map((_, level) => Math.max(pixel, 3.4 * size * habit.limb * 0.66 ** level))
  // Every unscaled step is shorter than 1.2; this conservative bound avoids measuring subpixels.
  const shortSegments = 1.2 * Math.abs(scale) < pixel / 2
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
    const steps = shortSegments
      ? 1
      : Math.max(1, Math.ceil((Math.hypot(toX - fromX, toY - fromY) * scale) / pixel))
    const girth = girths[level]
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
  return dressTree({
    kind,
    seed,
    height,
    habit,
    form,
    tone,
    breadth,
    branches,
    size,
    scale,
    pixel,
    cells,
    deciduous,
    limbs,
    buds,
  })
}
