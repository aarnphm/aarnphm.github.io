import {
  BOARD,
  type Built,
  type StyleContext,
  type StyleLayer,
  type StyleSolid,
} from './style-model'
import {
  bottomUp,
  boxPart,
  halo,
  shelfLayer,
  solidOf,
  solids,
  standing,
  withLines,
} from './style-primitives'
import { at, seg } from './world-space'

/**
 * A leaning ladder: boards share a back plane and grow deeper toward the floor, so their fronts
 * step forward without tilting a face. The rails are the only slanted solids; they meet the board
 * ends with no gap.
 */
export function ladder(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  // 22 units per tier up to six tiers; more tiers share the same total lean.
  const STEP = n > 6 ? 110 / (n - 1) : 22
  const RW = 12
  const RT = 16
  const FLOOR = -30
  const L = w + 8
  const yBack = d + (n - 1) * STEP
  // The stretcher clears the tallest book on the top tier.
  const zTop = (n - 1) * pitch + ctx.layout.bookHeight + 40
  const board = ctx.pick('board')
  const railMaterial = ctx.pick('rail')
  // Rail front: through every board's front edge, 2 units proud.
  const yAt = (z: number) => yBack - d - 2 - (STEP / pitch) * ((n - 1) * pitch - z)
  const placements = shelves.map((_, i) =>
    standing(i, [0, yBack - (d + i * STEP), (n - 1 - i) * pitch], { depth: d + i * STEP }),
  )
  const rail = (x: number): StyleSolid => {
    const y0 = yAt(FLOOR)
    const y1 = yAt(zTop)
    const [A, B, C, D, E, F, N] = (
      [
        [x, y0, FLOOR],
        [x + RW, y0, FLOOR],
        [x + RW, y0 + RT, FLOOR],
        [x + RW, y1 + RT, zTop],
        [x, y1 + RT, zTop],
        [x, y1, zTop],
        [x + RW, y1, zTop],
      ] as const
    ).map(at)
    const solid = solidOf('rail', railMaterial)
    solid.faces = [
      { points: [A, B, N, F], tone: 'face', o: 'front', tall: true },
      { points: [B, C, D, N], tone: 's2', o: 'side', tall: true },
      { points: [F, N, D, E], tone: 'top', o: 'top', tall: true },
    ]
    solid.lines = [
      { kind: 'highlight', segments: [seg([x + 2, y0, FLOOR + 2], [x + 2, y1, zTop - 2])] },
    ]
    solid.creases = [
      [N, B],
      [N, D],
      [N, F],
    ]
    solid.outline = [A, B, C, D, E, F]
    return solid
  }
  const layers: StyleLayer[] = [
    solids(
      [rail(-RW)],
      [
        halo('case', railMaterial, [
          [-RW - 4, yAt(FLOOR) - 4, FLOOR],
          [L + RW + 4, yAt(FLOOR) - 4, FLOOR],
          [L + RW + 4, yBack + 4, FLOOR],
          [-RW - 4, yBack + 4, FLOOR],
        ]),
      ],
    ),
  ]
  for (const i of bottomUp(n)) {
    const p = placements[i]
    const [, y, z] = p.origin
    layers.push(
      solids([
        withLines(
          boxPart('board', board, [0, y, z - BOARD], [L, p.depth!, BOARD], { top: 's3' }),
          'bevel',
          [seg([2, y, z - 2], [L - 2, y, z - 2])],
        ),
      ]),
      shelfLayer(ctx, p, { material: board, board: [0, L] }),
    )
  }
  // The stretcher spans between the rails, flush with their tops and fronts; the right rail is
  // nearer than its end, so it is painted after it.
  layers.push(
    solids([
      boxPart('stretcher', ctx.pick('stretcher'), [0, yAt(zTop), zTop - 12], [L, RT, 12]),
      rail(L),
    ]),
  )
  return { placements, layers }
}
