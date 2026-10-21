import type { TreePlacement } from './forest-plan'
import { creekDistance, creekSection, pondDistance, type Landscape } from '../landscape'
import { trailRandom } from '../random'
import { forestLimits, type ForestSettings } from '../scene-settings'
import { trailSnowBank } from '../winter'
import { foothillFoot, foothillKinds, nearestDistance, stands, type Point } from './species'
import { tree } from './tree-growth'

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

export function treeGrowthDepth(growth: number) {
  return Math.round(
    forestLimits.depth.min + (forestLimits.depth.max - forestLimits.depth.min) * growth ** 0.6,
  )
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
  const grown = treeGrowthDepth(growth)
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
        ? foothillFoot(x, landscape, random())
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
