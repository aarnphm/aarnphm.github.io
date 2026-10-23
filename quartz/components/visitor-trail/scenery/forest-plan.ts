import { forestYear, groveStorms, shadeTolerance, slotOccupant } from '../forest-succession'
import {
  creekDistance,
  creekSection,
  groundOffset,
  onDryLand,
  pondDistance,
  type Landscape,
} from '../landscape'
import { trailRandom } from '../random'
import { behindCrest, contourFaces, onSteepGround } from '../relief'
import { forestLimits } from '../scene-settings'
import {
  foothillFoot,
  foothillKinds,
  kinds,
  nearestDistance,
  standKinds,
  stands,
  standSize,
  type Point,
  type TreeKind,
} from './species'

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
  // Forest years since the tree rooted, and who roots after it, when succession placed it.
  age?: number
  successor?: TreeKind | null
  succeeds?: number
  log: number | null
}

export function planTrailTrees(
  landscape: Landscape,
  seed: number,
  walks: Point[] = [],
  clearings: (Point & { radius: number })[] = [],
  now = Date.now(),
  water: ReadonlyMap<string, readonly { count: number; at: number }[]> = new Map(),
): TreePlacement[] {
  const steps = planTrailTreeSteps(landscape, seed, walks, clearings, now, water)
  let step = steps.next()
  while (!step.done) step = steps.next()
  return step.value
}

// Each step keeps random draws and root insertion ordered. Callers can stop between slots without
// exposing partial geometry or changing the next plan.
export function* planTrailTreeSteps(
  landscape: Landscape,
  seed: number,
  walks: Point[] = [],
  clearings: (Point & { radius: number })[] = [],
  now = Date.now(),
  water: ReadonlyMap<string, readonly { count: number; at: number }[]> = new Map(),
): Generator<void, TreePlacement[]> {
  const { width, meadowTop, camp, groves, pond, alpine } = landscape
  const clearing = camp.y - meadowTop
  const random = trailRandom(seed ^ 0x7f4a7c15)
  // The groves keep their size on a wide meadow, so a wide screen takes more trees to keep the
  // stands as thick, up to 2.4 times the base count at 4k. Plan the maximum population once so density
  // changes select stable positions without rerolling edge trees or the ground cover.
  const count = Math.round(forestLimits.density.max * Math.min(2.4, Math.max(1, width / 1280)))
  // The home groves share the count; a stand the wide meadow adds is planned after them, from
  // the same generator, so the home trees keep their draws and the stand only adds its own.
  const home = groves.filter(grove => !grove.stand)
  const added = groves.filter(grove => grove.stand)
  const roots: Point[] = []
  const faces = contourFaces(landscape)
  const placements: (TreePlacement & { pool: readonly TreeKind[] | null })[] = []
  for (let index = 0; index < count + added.length * standSize; index++) {
    const stand = index >= count ? added[Math.floor((index - count) / standSize)] : null
    // One tree in four stands alone; the rest fill the groves, so more trees thicken the stands.
    const standalone = !stand && index % 4 === 3
    const foothill = standalone && index < 24
    const cluster = stand
      ? groves.indexOf(stand)
      : (index - Math.floor(index / 4)) % Math.max(1, home.length)
    const grove = groves[cluster]
    let root: Point = grove
    if (foothill) {
      const x =
        alpine.left +
        alpine.width * (0.12 + Math.floor(index / 4) * 0.145 + (random() - 0.5) * 0.03)
      root = { x, y: foothillFoot(x, landscape, random()) }
    } else {
      let bestClearance = -Infinity
      let open: Point | null = null
      for (let attempt = 0; attempt < 140; attempt++) {
        const angle =
          Math.floor((stand ? index - count : index) / Math.max(1, home.length)) * 2.399963 +
          (random() - 0.5) * 1.8
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
          onSteepGround(faces, candidate) ||
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
          // A lone tree looks forty times for open ground just behind a crest, where the hill
          // hides its foot and the crown above the crest line gives the ground its depth, before
          // it settles for the first open ground it found.
          if (!standalone || attempt >= 40 || behindCrest(faces, candidate)) {
            open = candidate
            break
          }
          open ??= candidate
        }
      }
      if (open) root = open
    }
    roots.push(root)
    const identity = trailRandom(0x777034b3 ^ Math.imul(index + 1, 16807))
    const pick = identity()
    // Willows take the pond shore; groves keep to their stand; the foothills are montane conifer.
    // One lone tree in eight is a snag left standing after a burn.
    const upland = kinds.filter(kind => kind !== 'willow' && kind !== 'snag')
    const standPool = stand?.stand ? standKinds[stand.stand] : stands[cluster % stands.length]
    // An added stand keeps to its own kinds even near the pond shore.
    const kind: TreeKind = foothill
      ? foothillKinds[Math.floor(pick * foothillKinds.length)]
      : !stand && pondDistance(root, pond) < 2.8 && pick < 0.7
        ? 'willow'
        : standalone
          ? pick < 0.125
            ? 'snag'
            : upland[Math.floor(((pick - 0.125) / 0.875) * upland.length)]
          : standPool[Math.floor(pick * 4)]
    const treeSeed = Math.floor(identity() * 4294967296)
    const height = foothill ? 20 + identity() * 10 : 30 + identity() * 26
    placements.push({
      pool: foothill
        ? null
        : kind === 'willow'
          ? waterside
          : standalone || kind === 'snag'
            ? upland
            : standPool,
      id: stand ? `tree-stand-${cluster}-${index - count + 1}` : `tree-${index + 1}`,
      x: Math.round(root.x),
      y: foothill ? root.y : Math.max(Math.ceil(meadowTop + 64), Math.round(root.y)),
      standalone,
      foothill,
      cluster,
      kind,
      treeSeed,
      height,
      growth: 1,
      generation: 0,
      log: null,
    })
    yield
  }
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
          onSteepGround(faces, candidate) ||
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
      yield
    }
  return yield* grow(placements, landscape, forestYear(now), water)
}

const waterside: readonly TreeKind[] = ['willow', 'willow', 'birch', 'aspen']
const edgeKinds: readonly TreeKind[] = ['pine', 'pine', 'spruce', 'fir']

// Age every slot to the forest year, then let the canopy shade its understory: a sapling under a
// taller crown grows slowly, the more so the less shade it tolerates.
function* grow(
  planned: (TreePlacement & { pool: readonly TreeKind[] | null })[],
  landscape: Landscape,
  year: number,
  water: ReadonlyMap<string, readonly { count: number; at: number }[]>,
): Generator<void, TreePlacement[]> {
  const storms = landscape.groves.map((_, grove) => groveStorms(grove, Math.floor(year)))
  const placements: TreePlacement[] = []
  for (const [slot, plannedPlacement] of planned.entries()) {
    const { pool, ...placement } = plannedPlacement
    if (!pool || year < 0) placements.push(placement)
    else {
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
      placements.push({ ...placement, ...occupant })
    }
    yield
  }
  for (const sapling of placements) {
    yield
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
