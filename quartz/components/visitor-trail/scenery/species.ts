import { bayerThreshold } from '../../../util/bayer'
import { alpineFoot, type Landscape } from '../landscape'
import { forestLimits, type ForestSettings } from '../scene-settings'

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
export type Crown = 'round' | 'cone' | 'spire' | 'column' | 'tuft' | 'tier' | 'droop'
export type Species = {
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
export type TreeForm = 'upright' | 'leaning' | 'forked' | 'open' | 'broad'
export type TreeTone = 'sage' | 'lichen' | 'ochre'
export type Point = { x: number; y: number }
export type Tip = Point & { level: number }
export type Turtle = Point & { heading: number; bend: number }
export type Tone = 'trunk' | 'bark' | 'light' | 'crown' | 'dark' | 'gold'
// Eastern white pine holds its needles in flat tufts; hemlock in drooping tiers; tamarack is a
// thin conifer that turns gold; aspen is a narrow crown on pale bark; willow hangs in curtains.
export const species: Record<TreeKind, Species> = {
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
export const kinds = Object.keys(species) as TreeKind[]
// Trees that grow together, one stand per grove: boreal, pine–hemlock, sugar-maple hardwood, and
// Sierra mixed conifer (sequoia with ponderosa, incense cedar and black oak).
export const stands: readonly TreeKind[][] = [
  ['spruce', 'fir', 'birch', 'larch'],
  ['pine', 'hemlock', 'birch', 'aspen'],
  ['maple', 'oak', 'aspen', 'maple'],
  ['sequoia', 'pine', 'cypress', 'sequoia'],
]
export const foothillKinds: readonly TreeKind[] = ['pine', 'spruce', 'fir', 'larch']

// A foothill tree roots on the lip where the ranges meet the valley, at most 2 px back. Higher, it
// stands in the haze over the pale moss shelf with nothing under its trunk; lower, on the bank
// face, the ground layer's snow blanket (from the lip down) covers its trunk. The blanket and the
// bank sample the lip once per 4 px cell, at its edge and its middle, so the trunk takes the
// highest sample of the cells it stands in. `roll` is in [0, 1).
export const foothillFoot = (x: number, landscape: Landscape, roll: number) => {
  const cells = [-2, 2].map(dx => Math.floor((Math.round(x) + dx) / 4) * 4)
  const lip = Math.min(
    ...cells.flatMap(cell => [cell, cell + 2].map(at => alpineFoot(at, landscape))),
  )
  return Math.floor(lip) - Math.floor(roll * 2) * 2
}

// The stand a wide meadow adds in front: white pine with hemlock and spruce under it.
export const standKinds: Record<'pine' | 'spruce', readonly TreeKind[]> = {
  pine: ['pine', 'pine', 'hemlock', 'spruce'],
  spruce: ['spruce', 'spruce', 'fir', 'pine'],
}
// Trees planned for an added stand, on top of the home groves' count.
export const standSize = 12
// Broadleaves and the larch drop their leaves for the winter.
export const bareInWinter = new Set<TreeKind>(['maple', 'oak', 'aspen', 'birch', 'larch', 'willow'])
export const woodlandKinds: readonly TreeKind[] = [
  'spruce',
  'fir',
  'pine',
  'larch',
  'birch',
  'hemlock',
  'maple',
  'aspen',
]
export const forms: TreeForm[] = ['upright', 'leaning', 'forked', 'open', 'broad']
export const tones: TreeTone[] = ['sage', 'lichen', 'ochre']

export function canopyTone(
  x: number,
  y: number,
  radius: number,
  pixel: number,
  dapple: number,
): Tone {
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
export function crownRows(crown: Crown, r: number) {
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

export function bounded(value: number, field: keyof ForestSettings) {
  const { min, max } = forestLimits[field]
  return Math.max(min, Math.min(max, Math.round(value)))
}

export function nearestDistance(point: Point, points: Point[]) {
  let distance = Infinity
  for (const other of points) {
    const squared = (point.x - other.x) ** 2 + (point.y - other.y) ** 2
    if (squared < distance) distance = squared
  }
  return Math.sqrt(distance)
}

// Raster cells are keyed by one number; a tree spans far fewer than 4,096 cells either way.
export const cellKey = (x: number, y: number) => (x + 4096) * 8192 + y + 4096
