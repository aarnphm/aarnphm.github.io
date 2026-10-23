import type { ForestSprite } from './forest-sprites'
import type { Landscape, Point } from './landscape'
import type { RepoussoirShape } from './repoussoir/markup'
import { mulberry } from '../scripts/404-pixel'
import { repoussoirTreeSteps } from './repoussoir/shape'

// The two big trees from the 404, grown the same way and drawn at the trail's 2 px cell: one at
// each lower corner of the scene, cropped by its edge, with the long tapering limbs and the leaf
// masses of the 404's repoussoir pair. The markup keeps the trail tree's structure (motion group,
// canopy clusters, snow, hollow) so wind, seasons, snow and the depth order treat it as a tree.

function finish<T>(search: Generator<void, T, undefined>): T {
  let step = search.next()
  while (!step.done) step = search.next()
  return step.value
}

export function repoussoirTree(seed: number, height: number, lean: number): RepoussoirShape {
  return finish(repoussoirTreeSteps(seed, height, lean))
}

const shapes = new Map<string, ReturnType<typeof repoussoirTree>>()

function* growSteps(treeSeed: number, tall: number, lean: number) {
  const key = `${treeSeed}|${tall}|${lean.toFixed(3)}`
  let shape = shapes.get(key)
  if (!shape) {
    shape = yield* repoussoirTreeSteps(treeSeed, tall, lean)
    if (shapes.size > 12) shapes.clear()
    shapes.set(key, shape)
  }
  return { key, shape }
}

export type Clearing = Point & { radius: number }

// One tree at each lower corner, rooted far enough inside the bottom corners to leave the page's
// corner readouts clear, leaning out like the 404 pair so the crown hangs over the edge, a third
// of the scene tall. Where the crown or the trunk would reach a camp the tree grows shorter, down
// to two thirds of the height; a tree that cannot clear the camps even then is left out. They
// stand in front of everything on the ground.
export function repoussoirSprites(
  landscape: Landscape,
  seed: number,
  clearings: Clearing[],
): ForestSprite[] {
  return finish(repoussoirSpriteSteps(landscape, seed, clearings))
}

export function* repoussoirSpriteSteps(
  landscape: Landscape,
  seed: number,
  clearings: Clearing[],
): Generator<void, ForestSprite[], undefined> {
  const { width, height } = landscape
  const base = Math.max(160, Math.min(420, height * 0.34))
  const sprites: ForestSprite[] = []
  for (const side of [0, 1]) {
    const random = mulberry(seed ^ Math.imul(side + 1, 0x7f4a7c15))
    const out = side ? 1 : -1
    const lean = out * (0.07 + random() * 0.04)
    // 118–134 px in: past the lower corner readouts (waterings + mascot, local time).
    const edge = Math.round(118 + random() * 16)
    const x = side ? width - edge : edge
    const y = height - 4
    const treeSeed = (seed ^ Math.imul(0x404 + side, 0x9e3779b9)) >>> 0
    for (const scale of [1, 0.88, 0.77, 0.67]) {
      const tall = Math.round(base * scale)
      const { key, shape } = yield* growSteps(treeSeed, tall, lean)
      const { foliage, markup, trunkRadius } = shape
      const top = y + foliage.top
      const bottom = y + foliage.bottom
      const inner = x + (side ? foliage.left : foliage.right)
      // The trunk is a segment from the root up to the crown's base along the lean.
      const trunkTop = y + foliage.bottom
      const trunkTipX = x + Math.sin(lean) * (y - trunkTop)
      const blocked = clearings.some(clearing => {
        const reach = clearing.radius + trunkRadius
        const t = Math.max(0, Math.min(1, (y - clearing.y) / Math.max(1, y - trunkTop)))
        const near = Math.hypot(
          x + (trunkTipX - x) * t - clearing.x,
          y - (y - trunkTop) * t - clearing.y,
        )
        if (near < reach) return true
        if (clearing.y + clearing.radius < top || clearing.y - clearing.radius > bottom)
          return false
        return out * (inner - (clearing.x + out * clearing.radius)) < 0
      })
      if (blocked) continue
      const render = () =>
        `<g class="trail-tree" data-tree-id="tree-repoussoir-${side}" data-tree="beech" data-tree-placement="standalone" data-tree-depth="meadow" data-tree-scale="1.000" data-base-trunk-radius="${trunkRadius.toFixed(3)}" data-trunk-radius="${trunkRadius.toFixed(3)}" data-growth-limit="1.3" data-ground-x="${x}" data-ground-y="${y}" data-tree-form="broad" data-tree-tone="sage" data-tree-repoussoir="${side ? 'right' : 'left'}" style="--trail-tree-haze: 0%" transform="translate(${x} ${y})"><g class="trail-tree-perspective" transform="scale(1)"><g class="trail-tree-growth" transform="scale(1)"><g class="trail-tree-community"><g class="trail-tree-touch">${markup}</g></g></g></g></g>`
      sprites.push({ id: `tree-repoussoir-${side}`, bottom: y + 12, shape: `${key}|${x}`, render })
      break
    }
  }
  return sprites
}
