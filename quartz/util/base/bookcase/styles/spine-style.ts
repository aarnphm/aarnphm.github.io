import type { Built, StyleContext, StyleLayer, StyleSolid } from './style-model'
import { bottomUp, boxPart, halo, shelfLayer, solids, standing } from './style-primitives'

/**
 * One steel pole on a base plate with small leaves up it; books lie flat with spines out. Leaves
 * alternate right and left of the pole without rotating, since a turned leaf would show a pulled
 * cover upside down. The pole stands near the back of the leaves, between the two lanes, and each
 * leaf ends against a collar on the pole. A plane through the pole separates the lanes, so the left
 * lane is painted first, then the pole, then the right lane.
 */
export function spine(ctx: StyleContext): Built {
  const { shelves, shelfWidth: run, depth: d, bookHeight } = ctx.layout
  const n = shelves.length
  const POLE = 14
  const LEAF = 6
  const COLLAR = 10
  const leafW = bookHeight + 16
  const step = run + LEAF + 14
  const baseZ = -30
  const topZ = (n - 1) * step + run + 24
  const py = d - POLE - 6
  const half = Math.round(leafW * 0.6)
  const leaf = ctx.pick('leaf')
  const pole = ctx.pick('pole')
  const collar = ctx.pick('collar')
  const placements = shelves.map((_, i) =>
    i % 2 === 0
      ? standing(i, [POLE / 2 + 8, 0, (n - 1 - i) * step], { pose: 'flat', align: 'head' })
      : standing(i, [-POLE / 2 - 8, 0, (n - 1 - i) * step], {
          pose: 'flat',
          align: 'tail',
          lane: 1,
        }),
  )
  const layers: StyleLayer[] = [
    solids(
      [
        boxPart('base', ctx.pick('base'), [-half, -16, baseZ - 12], [2 * half, d + 32, 12], {
          top: 's3',
        }),
      ],
      [
        halo('case', leaf, [
          [-half - 10, -26, baseZ - 12],
          [half + 10, -26, baseZ - 12],
          [half + 10, d + 26, baseZ - 12],
          [-half - 10, d + 26, baseZ - 12],
        ]),
      ],
    ),
  ]
  const leafLayers = (i: number): StyleLayer[] => {
    const p = placements[i]
    const [ox, , z] = p.origin
    const right = p.lane === 0
    const x0 = right ? POLE / 2 : -POLE / 2 - leafW
    // An empty leaf keeps a contact as long as a full-height book.
    const span = Math.max(
      ...ctx.runs[i].map(index => ctx.layout.books[index].height),
      ctx.runs[i].length > 0 ? 0 : bookHeight,
    )
    const sx = right ? ox - 4 : ox - span - 6
    const ex = right ? ox + span + 6 : ox + 4
    return [
      solids([boxPart('leaf', leaf, [x0, 0, z - LEAF], [leafW, d, LEAF], { top: 's3' })]),
      shelfLayer(ctx, p, {
        material: leaf,
        board: [x0 - ox, x0 + leafW - ox],
        contact: [
          [sx, d - 14, z],
          [ex, d - 14, z],
          [ex, d, z],
          [sx, d, z],
        ],
        plateZ: [-LEAF - 7, -LEAF + 0.5],
      }),
    ]
  }
  const order = bottomUp(n)
  for (const i of order) if (placements[i].lane === 1) layers.push(...leafLayers(i))
  // The pole runs in segments between collars, bottom up, so no two parts of it overlap.
  const column: StyleSolid[] = []
  let from = baseZ
  for (const i of order) {
    const z = placements[i].origin[2]
    column.push(
      boxPart('pole', pole, [-POLE / 2, py, from], [POLE, POLE, z - LEAF - COLLAR - from]),
      boxPart(
        'collar',
        collar,
        [-POLE / 2, py - 4, z - LEAF - COLLAR],
        [POLE, POLE + 8, COLLAR + LEAF],
        { front: 's2', side: 's1', top: 's3' },
      ),
    )
    from = z
  }
  column.push(
    boxPart('pole', pole, [-POLE / 2, py, from], [POLE, POLE, topZ - from]),
    boxPart('pole', pole, [-POLE / 2 - 3, py - 3, topZ], [POLE + 6, POLE + 6, 6]),
  )
  layers.push(solids(column))
  for (const i of order) if (placements[i].lane === 0) layers.push(...leafLayers(i))
  return { placements, layers }
}
