import type { ForestSprite } from './forest-sprites'
import type { Landscape } from './landscape'
import { sceneSettings, type ForestSettings } from './scene-settings'
import { planTrailTrees, type TreePlacement } from './scenery/forest-plan'
import { bounded, type Point } from './scenery/species'
import { tree } from './scenery/tree-growth'
import { forestShape, treeSprite } from './scenery/tree-sprites'
import { trailDeadfall, trailUnderstory } from './understory'
export type { TreeKind } from './scenery/species'
export { planTrailTrees, planTrailTreeSteps } from './scenery/forest-plan'
export type { TreePlacement } from './scenery/forest-plan'
export {
  createTreePainter,
  trailPatchTrees,
  treeGrowthDepth,
  treeSprite,
} from './scenery/tree-sprites'
export { trailFoothillWoodland } from './scenery/foothill-woodland'

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
    bounded(settings.density, 'density') * Math.min(2.4, Math.max(1, landscape.width / 1280)),
  )
  const shape = forestShape(settings)
  // Edge conifers and an added stand are drawn whole; the density setting thins the home groves.
  const wanted = (placement: TreePlacement) =>
    placement.id.startsWith('tree-edge-') ||
    placement.id.startsWith('tree-stand-') ||
    Number(placement.id.slice(5)) <= count
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
