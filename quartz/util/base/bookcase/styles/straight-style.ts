import {
  BOARD,
  CROWN,
  OVERHANG,
  PLINTH,
  WALL,
  type Built,
  type StyleContext,
  type StyleLayer,
  type World,
} from './style-model'
import {
  bottomUp,
  boxPart,
  halo,
  panelPart,
  shelfLayer,
  solids,
  standing,
  withLines,
} from './style-primitives'
import { at, chain, seg } from './world-space'

export function straight(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  const board = ctx.pick('board')
  const wall = ctx.pick('wall')
  const crown = ctx.pick('crown')
  const ceiling = n * pitch - BOARD
  const floor = -PLINTH
  const x0 = -WALL - OVERHANG
  const x1 = w + WALL + OVERHANG
  const top = ceiling + CROWN
  const inset = 8
  const placements = shelves.map((_, i) => standing(i, [0, 0, (n - 1 - i) * pitch]))
  const layers: StyleLayer[] = [
    solids(
      [
        boxPart(
          'plinth',
          ctx.pick('plinth'),
          [-WALL + 6, 6, floor],
          [w + 2 * WALL - 12, d - 6, PLINTH - BOARD],
          { front: 's1', side: 's0' },
        ),
        panelPart(
          'panel',
          ctx.pick('panel'),
          [
            [0, d, -BOARD],
            [w, d, -BOARD],
            [w, d, ceiling],
            [0, d, ceiling],
          ],
          's0',
        ),
        panelPart(
          'wall',
          wall,
          [
            [0, 0, floor],
            [0, d, floor],
            [0, d, ceiling],
            [0, 0, ceiling],
          ],
          's1',
        ),
      ],
      [
        halo('case', board, [
          [x0, -OVERHANG, floor],
          [x1, -OVERHANG, floor],
          [x1, d, floor],
          [x0, d, floor],
        ]),
      ],
    ),
  ]
  for (const i of bottomUp(n)) {
    const z = placements[i].origin[2]
    layers.push(
      solids([
        withLines(
          boxPart('board', board, [0, 0, z - BOARD], [w, d, BOARD], { top: 's3' }),
          'bevel',
          [seg([0, 0, z - 2], [w, 0, z - 2])],
        ),
      ]),
      shelfLayer(ctx, placements[i], { material: board, board: [0, w] }),
    )
  }
  const loop: World[] = [
    [x0 + inset, -OVERHANG + inset, top],
    [x1 - inset, -OVERHANG + inset, top],
    [x1 - inset, d - inset, top],
    [x0 + inset, d - inset, top],
    [x0 + inset, -OVERHANG + inset, top],
  ]
  layers.push(
    solids([
      withLines(boxPart('wall', wall, [w, 0, floor], [WALL, d, ceiling - floor]), 'bevel', [
        seg([w + 2, 0, floor], [w + 2, 0, ceiling]),
      ]),
      withLines(
        panelPart(
          'wall',
          wall,
          [
            [-WALL, 0, floor],
            [0, 0, floor],
            [0, 0, ceiling],
            [-WALL, 0, ceiling],
          ],
          'face',
          true,
        ),
        'bevel',
        [seg([-WALL + 2, 0, floor], [-WALL + 2, 0, ceiling])],
      ),
      {
        ...withLines(
          boxPart('crown', crown, [x0, -OVERHANG, ceiling], [x1 - x0, d + OVERHANG, CROWN]),
          'bevel',
          [seg([x0, -OVERHANG, top - 3], [x1, -OVERHANG, top - 3]), ...chain(loop)],
        ),
        screws: [
          at([x0 + 4, -OVERHANG + 4, top]),
          at([x1 - 4, -OVERHANG + 4, top]),
          at([x1 - 4, d - 4, top]),
          at([x0 + 4, d - 4, top]),
        ],
      },
    ]),
  )
  return { placements, layers }
}
