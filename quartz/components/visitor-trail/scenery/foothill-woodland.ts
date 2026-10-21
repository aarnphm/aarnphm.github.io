import type { ForestSprite } from '../forest-sprites'
import {
  creekDistance,
  creekSection,
  groundOffset,
  pondDistance,
  type Landscape,
} from '../landscape'
import { trailRandom } from '../random'
import { forestLimits, type ForestSettings } from '../scene-settings'
import { nearestDistance, woodlandKinds, type Point } from './species'
import { tree } from './tree-growth'
import { forestShape } from './tree-sprites'

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
