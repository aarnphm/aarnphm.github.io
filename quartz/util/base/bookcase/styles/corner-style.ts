import { project } from '../geometry'
import {
  BOARD,
  CROWN,
  PLINTH,
  WALL,
  type Built,
  type Plan,
  type StyleContext,
  type StyleLayer,
  type World,
} from './style-model'
import {
  bottomUp,
  boxPart,
  halo,
  panelPart,
  prismPart,
  shelfLayer,
  solids,
  standing,
  withLines,
} from './style-primitives'
import { chain, seg } from './world-space'

/**
 * An L-shaped case. Wing A is the straight case along +x; wing B turns 90° and runs along +y at
 * x ≤ 0. The inner corner opens toward the viewer, so neither wing hides the other: a view ray from
 * wing A keeps x > 0, one from wing B keeps y < 0.
 */
export function corner(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const levels = Math.ceil(shelves.length / 2)
  const L = w + 8
  const top = levels * pitch
  const board = ctx.pick('board')
  const wall = ctx.pick('wall')
  const panel = ctx.pick('panel')
  const placements = shelves.map((_, i) => {
    const z = (levels - 1 - Math.floor(i / 2)) * pitch
    return i % 2 === 0 ? standing(i, [0, -L, z], { rot: 90, lane: 1 }) : standing(i, [0, 0, z])
  })
  const crownPlan: Plan = [
    [-d - 5, d],
    [-d - 5, -L - WALL - 5],
    [5, -L - WALL - 5],
    [5, -5],
    [L + WALL + 5, -5],
    [L + WALL + 5, d],
  ]
  const layers: StyleLayer[] = [
    solids(
      [
        prismPart(
          'plinth',
          ctx.pick('plinth'),
          [
            [-d, d],
            [-d, -L - WALL + 6],
            [-6, -L - WALL + 6],
            [-6, 6],
            [L + WALL - 6, 6],
            [L + WALL - 6, d],
          ],
          -PLINTH,
          PLINTH - BOARD,
          { front: 's1', side: 's0', top: 's0' },
        ),
        panelPart(
          'panel',
          panel,
          [
            [-d, d, -BOARD],
            [L, d, -BOARD],
            [L, d, top - BOARD],
            [-d, d, top - BOARD],
          ],
          's0',
        ),
        panelPart(
          'panel',
          panel,
          [
            [-d, -L, -BOARD],
            [-d, d, -BOARD],
            [-d, d, top - BOARD],
            [-d, -L, top - BOARD],
          ],
          's1',
        ),
      ],
      [
        halo(
          'case',
          board,
          crownPlan.map(([x, y]) => [x, y, -PLINTH] as World),
        ),
      ],
    ),
  ]
  for (const level of bottomUp(levels)) {
    const z = (levels - 1 - level) * pitch
    layers.push(
      solids([
        withLines(
          prismPart(
            'board',
            board,
            [
              [-d, d],
              [-d, -L],
              [0, -L],
              [0, 0],
              [L, 0],
              [L, d],
            ],
            z - BOARD,
            BOARD,
            { top: 's3' },
          ),
          'bevel',
          // One bevel along both fronts, through the inner corner.
          chain([
            [0, -L, z - 2],
            [0, 0, z - 2],
            [L, 0, z - 2],
          ]),
        ),
      ]),
    )
    for (const i of [2 * level + 1, 2 * level]) {
      if (i < shelves.length) {
        // Each wing's plate stops 4 short of the inner corner, clear of the other wing's plate.
        layers.push(
          shelfLayer(ctx, placements[i], {
            material: board,
            board: placements[i].rot === 90 ? [0, L - 4] : [4, L],
          }),
        )
      }
    }
  }
  const ct = top - BOARD + CROWN
  layers.push(
    solids([
      withLines(boxPart('wall', wall, [L, 0, -BOARD], [WALL, d, top]), 'bevel', [
        seg([L + 2, 0, -BOARD + 2], [L + 2, 0, top - BOARD - 2]),
      ]),
      withLines(boxPart('wall', wall, [-d, -L - WALL, -BOARD], [d, WALL, top]), 'bevel', [
        seg([-d + 2, -L - WALL, -BOARD + 2], [-d + 2, -L - WALL, top - BOARD - 2]),
      ]),
      {
        ...withLines(
          prismPart('crown', ctx.pick('crown'), crownPlan, top - BOARD, CROWN),
          'bevel',
          [
            ...chain([
              [5, -L - WALL - 5, ct - 3],
              [5, -5, ct - 3],
              [L + WALL + 5, -5, ct - 3],
            ]),
            ...chain([
              [-d + 3, d - 8, ct],
              [-d + 3, -L - WALL + 3, ct],
              [-3, -L - WALL + 3, ct],
              [-3, 3, ct],
              [L + WALL - 3, 3, ct],
              [L + WALL - 3, d - 8, ct],
              [-d + 3, d - 8, ct],
            ]),
          ],
        ),
        screws: (
          [
            [-d - 1, d - 4],
            [-d - 1, -L - WALL - 1],
            [1, -L - WALL - 1],
            [L + WALL + 1, -1],
            [L + WALL + 1, d - 4],
          ] as const
        ).map(([x, y]) => project(x, y, ct)),
      },
    ]),
  )
  return { placements, layers }
}
