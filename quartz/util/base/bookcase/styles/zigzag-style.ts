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
  solids,
  standing,
  withLines,
} from './style-primitives'
import { seg } from './world-space'

/**
 * The S shelf: boards at the straight pitch, an upright in each gap alternating left and right, a
 * short return under each board opposite its upright. A tier under a left upright starts its run
 * clear of it.
 */
export function zigzag(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  const UP = 44
  const RET = 12
  const RET_H = 44
  const L = w + UP + 12
  const board = ctx.pick('board')
  const upright = ctx.pick('upright')
  // Gap k lies under tier k; its upright is on the left for even k.
  const leftGap = (k: number) => k % 2 === 0
  const placements = shelves.map((_, i) =>
    standing(i, [i > 0 && leftGap(i - 1) ? UP + 2 : 0, 0, (n - 1 - i) * pitch]),
  )
  const layers: StyleLayer[] = [
    solids(
      [8, L - 38].map(x =>
        boxPart('foot', ctx.pick('foot'), [x, 10, -BOARD - 14], [30, d - 20, 14]),
      ),
      [
        halo('case', board, [
          [-4, -4, -BOARD - 14],
          [L + 4, -4, -BOARD - 14],
          [L + 4, d + 4, -BOARD - 14],
          [-4, d + 4, -BOARD - 14],
        ]),
      ],
    ),
  ]
  for (const i of bottomUp(n)) {
    const p = placements[i]
    const z = p.origin[2]
    const parts: StyleSolid[] = []
    // Every inner board meets an upright at both ends, so only the top board has a free end; its
    // return hangs there. A return under an inner board would stub out below the upright above it.
    // The return is painted before its board, which covers the return's top face.
    if (i === 0 && n > 1) {
      parts.push(
        boxPart(
          'return',
          ctx.pick('return'),
          [leftGap(0) ? L - RET : 0, 0, z - BOARD - RET_H],
          [RET, d, RET_H],
        ),
      )
    }
    parts.push(
      withLines(boxPart('board', board, [0, 0, z - BOARD], [L, d, BOARD]), 'bevel', [
        seg([2, 0, z - 2], [L - 2, 0, z - 2]),
      ]),
    )
    const above = i > 0 ? (leftGap(i - 1) ? 'left' : 'right') : null
    const rise = i > 0 ? placements[i - 1].origin[2] - BOARD - z : 0
    // Uprights are flush with the board ends and front, so each corner closes on one edge.
    if (above === 'left') parts.push(boxPart('upright', upright, [0, 0, z], [UP, d, rise]))
    layers.push(
      solids(parts),
      shelfLayer(ctx, p, { material: board, board: [-p.origin[0], L - p.origin[0]] }),
    )
    // The right upright stands past the last book, so it is painted after the books.
    if (above === 'right')
      layers.push(solids([boxPart('upright', upright, [L - UP, 0, z], [UP, d, rise])]))
  }
  return { placements, layers }
}
