import { trailRandom } from '../random'
import {
  cellKey,
  type Point,
  type Species,
  type Tone,
  type TreeForm,
  type TreeKind,
  type TreeTone,
} from './species'

// What the grown tree hands to its markup: the raster cells, the bare frame and the traits a
// sprite reports back.
type GrownTree = {
  kind: TreeKind
  seed: number
  height: number
  habit: Species
  form: TreeForm
  tone: TreeTone
  breadth: number
  branches: number[]
  size: number
  scale: number
  pixel: number
  cells: Map<number, { x: number; y: number; tone: Tone }>
  deciduous: boolean
  limbs: Map<number, { x: number; y: number; tone: Tone }>
  buds: Set<number>
}

export function dressTree(grown: GrownTree) {
  const {
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
  } = grown
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
      `<path class="trail-tree-shadow" data-shadow-radius="${shadow}" d="M${-shadow} 0h${shadow * 2}v2h-3v2h${-shadow * 2 + 6}V2h-3Z"/>` +
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
