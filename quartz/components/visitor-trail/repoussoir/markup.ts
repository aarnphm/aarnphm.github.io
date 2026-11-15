import { leafClumps } from '../scenery/leaf-clumps'

export type Tone = 'light' | 'crown' | 'dark' | 'gold'
export type Cell = { x: number; y: number; tone: Tone | 'trunk' | 'trunk-lit' }

export const pixel = 2
export const cellKey = (x: number, y: number) => (x + 4096) * 8192 + y + 4096

const rect = (x: number, y: number) => `M${x} ${y}h${pixel}v${pixel}h${-pixel}Z`
const cap = (x: number, y: number) =>
  `M${x} ${y}h${pixel}v${Math.max(1, Math.floor(pixel / 2))}h${-pixel}Z`

export type RepoussoirShape = {
  markup: string
  trunkRadius: number
  extent: number
  foliage: { left: number; right: number; top: number; bottom: number }
}

export function* repoussoirMarkupSteps(
  cells: ReadonlyMap<number, Cell>,
  height: number,
  trunkRadius: number,
  work: number,
): Generator<void, RepoussoirShape, undefined> {
  // Markup in the trail tree's grammar: limbs that stay through the winter, the canopy in leaf
  // clumps over a shaded copy, snow along every exposed top.
  const paths: Record<'trunk' | 'trunk-lit', string[]> = { trunk: [], 'trunk-lit': [] }
  const woodySnow: string[] = []
  const foliage = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity }
  let extent = 1
  for (const { x, y, tone } of cells.values()) {
    if (++work % 128 === 0) yield
    extent = Math.max(extent, -y)
    if (tone === 'trunk' || tone === 'trunk-lit') {
      paths[tone].push(rect(x, y))
      if (y < -height * 0.3 && !cells.has(cellKey(x / pixel, y / pixel - 1)))
        woodySnow.push(cap(x, y))
      continue
    }
    foliage.left = Math.min(foliage.left, x)
    foliage.right = Math.max(foliage.right, x + pixel)
    foliage.top = Math.min(foliage.top, y)
    foliage.bottom = Math.max(foliage.bottom, y + pixel)
  }
  type Cluster = { marks: Record<Tone, string[]>; snow: string[]; flip: string[]; sprout: string[] }
  const clusters = new Map<number, Cluster>()
  const hollow: string[] = []
  const leafy = [...cells.values()].filter(
    ({ tone }) => tone !== 'trunk' && tone !== 'trunk-lit',
  ) as { x: number; y: number; tone: Tone }[]
  yield
  const plan = leafClumps(leafy, pixel, false)
  yield
  for (let index = 0; index < leafy.length; index++) {
    if (++work % 128 === 0) yield
    const { x, y, tone } = leafy[index]
    const key = plan.owner[index]
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
    const cell = rect(x, y)
    cluster.marks[tone].push(cell)
    hollow.push(cell)
    if (!cells.has(cellKey(x / pixel, y / pixel - 1))) cluster.snow.push(cap(x, y))
    let hash = Math.imul(x / pixel, 374761393) + Math.imul(y / pixel, 668265263)
    hash = Math.imul(hash ^ (hash >>> 13), 1274126177)
    hash = (hash ^ (hash >>> 16)) >>> 0
    if (tone !== 'light' && hash % 7 === 0) cluster.flip.push(cell)
    if ((hash >>> 3) % 3 === 0) cluster.sprout.push(cell)
  }
  const canopy = [...clusters.entries()]
    .sort(([a], [b]) => a - b)
    .map(([key, cluster]) => {
      const { x: centreX, y: centreY, loose } = plan.clumps[key]
      const lift = Math.max(0.15, Math.min(1, -centreY / extent)) ** 1.5
      return (
        `<g class="trail-tree-leaves" data-leaf-x="${centreX.toFixed(1)}" data-leaf-y="${centreY.toFixed(1)}" data-leaf-lift="${lift.toFixed(2)}" data-leaf-loose="${loose.toFixed(2)}" data-leaf-cell="${pixel}">` +
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
  const shadow = Math.round(trunkRadius * 2.4)
  const markup =
    `<path class="trail-tree-shadow" data-shadow-radius="${shadow}" d="M${-shadow} 0h${shadow * 2}v2h-3v2h${-shadow * 2 + 6}V2h-3Z"/>` +
    '<g class="trail-tree-motion">' +
    `<path class="trail-tree-trunk" d="${paths.trunk.join('')}"/>` +
    `<path class="trail-tree-trunk-lit" d="${paths['trunk-lit'].join('')}"/>` +
    `<path class="trail-tree-snow" d="${woodySnow.join('')}"/>` +
    `<g class="trail-tree-canopy" data-leaf-span="${Math.max(0, foliage.right - foliage.left).toFixed(0)}"><path class="trail-tree-hollow" d="${hollow.join('')}"/>${canopy}</g>` +
    '</g>'
  return { markup, trunkRadius, extent, foliage }
}
