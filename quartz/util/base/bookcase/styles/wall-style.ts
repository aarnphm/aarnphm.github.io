import type { Built, StyleContext, StyleLayer } from './style-model'
import { bottomUp, boxPart, halo, shelfLayer, solids, standing } from './style-primitives'

/**
 * A tall panel with short blades in two columns, staggered by half a pitch. The column gap is at
 * least the depth, so the right column never overlaps the left one on screen and is painted after it.
 */
export function wallHung(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  const BLADE = 6
  const gap = Math.max(160, d + 6)
  const topZ = ((n - 1) * pitch) / 2 + pitch + 10
  const px0 = -30
  const px1 = 2 * w + gap + 30
  const blade = ctx.pick('blade')
  const panel = ctx.pick('panel')
  const placements = shelves.map((_, i) =>
    standing(i, [(i % 2) * (w + gap), 0, ((n - 1 - i) * pitch) / 2], { lane: i % 2 }),
  )
  const layers: StyleLayer[] = [
    solids(
      [
        boxPart('panel', panel, [px0, d, -60], [px1 - px0, 6, topZ + 60], {
          front: 's1',
          side: 's0',
          top: 's2',
        }),
      ],
      [
        halo('case', panel, [
          [px0 + 6, d - 2, -70],
          [px1 + 6, d - 2, -70],
          [px1 + 6, d - 2, -40],
          [px0 + 6, d - 2, -40],
        ]),
      ],
    ),
  ]
  for (const lane of [0, 1]) {
    for (const i of bottomUp(n).filter(i => i % 2 === lane)) {
      const p = placements[i]
      const z = p.origin[2]
      const x0 = p.origin[0] - 6
      const len = shelves[i].end + 14
      layers.push(
        solids(
          [boxPart('blade', blade, [x0, 0, z - BLADE], [len, d, BLADE], { top: 's3' })],
          [
            halo('shadow', panel, [
              [x0, d - 0.5, z - BLADE - 18],
              [x0 + len, d - 0.5, z - BLADE - 18],
              [x0 + len, d - 0.5, z - BLADE],
              [x0, d - 0.5, z - BLADE],
            ]),
          ],
        ),
        shelfLayer(ctx, p, {
          material: blade,
          board: [x0 - p.origin[0], x0 + len - p.origin[0]],
          plateZ: [-BLADE - 7, -BLADE + 0.5],
        }),
      )
    }
  }
  return { placements, layers }
}
