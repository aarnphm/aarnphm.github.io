type TreeKind = 'pine' | 'spruce' | 'cypress' | 'birch' | 'oak'
type TreePlacement = { x: number; y: number; pixel: 1 | 2 | 3; kind: TreeKind; variant?: number }

// Two-pixel foliage and bark use the same stepped edges as the 404 hearth.
const crowns: Record<TreeKind, string[][]> = {
  pine: [
    [
      '........#........',
      '.......###.......',
      '......#####......',
      '.....#######.....',
      '.......###.......',
      '......#####......',
      '.....#######.....',
      '....#########....',
      '...###########...',
      '......#####......',
      '.....#######.....',
      '....#########....',
      '...###########...',
      '..#############..',
      '....#########....',
      '...###########...',
      '..#############..',
      '.###############.',
      '#################',
      '.......BBB.......',
      '.......BBB.......',
      '.......BBB.......',
    ],
    [
      '......#......',
      '.....###.....',
      '....#####....',
      '...#######...',
      '.....###.....',
      '....#####....',
      '...#######...',
      '..#########..',
      '....#####....',
      '...#######...',
      '..#########..',
      '.###########.',
      '...#######...',
      '..#########..',
      '.###########.',
      '#############',
      '.....BB......',
      '.....BB......',
    ],
  ],
  spruce: [
    [
      '.......#.......',
      '......###......',
      '......###......',
      '.....#####.....',
      '....#######....',
      '.....#####.....',
      '....#######....',
      '...#########...',
      '....#######....',
      '...#########...',
      '..###########..',
      '...#########...',
      '..###########..',
      '.#############.',
      '..###########..',
      '.#############.',
      '###############',
      '......BBB......',
      '......BBB......',
    ],
  ],
  cypress: [
    [
      '....#....',
      '....##...',
      '...###...',
      '...###...',
      '...####..',
      '..#####..',
      '..#####..',
      '..######.',
      '.#######.',
      '.#######.',
      '.########',
      '#########',
      '#########',
      '.########',
      '.#######.',
      '..######.',
      '..#####..',
      '...###...',
      '...BBB...',
      '...BBB...',
    ],
  ],
  birch: [
    [
      '.....###.......',
      '...#######.....',
      '..###########..',
      '.#############.',
      '..#############',
      '###############',
      '.##############',
      '..###########..',
      '...####W####...',
      '.....##W##.....',
      '....B..W..B....',
      '.....B.W.B.....',
      '......BWB......',
      '.......W.......',
      '.......B.......',
      '.......W.......',
      '.......W.......',
      '.......B.......',
      '.......W.......',
    ],
  ],
  oak: [
    [
      '.....####.........',
      '..########..###...',
      '.###############..',
      '.################.',
      '##################',
      '#################.',
      '.#################',
      '..###############.',
      '...#######B####...',
      '.....##B##B##.....',
      '.......BBBB.......',
      '........BB........',
      '........BB........',
      '........BB........',
      '.......BBBB.......',
    ],
  ],
}

const tones = ['light', 'crown', 'dark', 'trunk', 'bark']

function tree(kind: TreeKind, variant: number, pixel: number) {
  const rows = crowns[kind][variant % crowns[kind].length]
  const width = Math.max(...rows.map(row => row.length))
  const paths = tones.map(() => '')
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const cell = row[x]
      if (cell === '.') continue
      const grain = (x * 7 + y * 11) % 9
      const tone =
        cell === 'B'
          ? 3
          : cell === 'W'
            ? 4
            : grain === 0 || (x < width / 2 && grain < 5)
              ? 0
              : x > width * 0.65 && grain > 2
                ? 2
                : 1
      paths[tone] +=
        `M${(x - Math.floor(width / 2)) * pixel} ${(y - rows.length) * pixel}h${pixel}v${pixel}h${-pixel}Z`
    }
  })
  return (
    `<path class="trail-tree-shadow" d="M${-7 * pixel} 0h${14 * pixel}v${pixel}h${-2 * pixel}v${pixel}h${-10 * pixel}V${pixel}h${-2 * pixel}Z"/>` +
    paths.map((d, index) => `<path class="trail-tree-${tones[index]}" d="${d}"/>`).join('') +
    (kind === 'pine'
      ? `<path class="trail-tree-trunk" d="M${-4 * pixel} ${-7 * pixel}h${pixel}v${2 * pixel}h${-pixel}ZM${3 * pixel} ${-11 * pixel}h${pixel}v${2 * pixel}h${-pixel}Z"/>`
      : '')
  )
}

export function trailTrees(width: number, bottom: number, campY: number) {
  const clearing = campY - bottom
  const trees: TreePlacement[] = [
    { x: 0.16, y: 0.35, pixel: 1, kind: 'pine', variant: 1 },
    { x: 0.12, y: 0.48, pixel: 2, kind: 'cypress' },
    { x: 0.2, y: 0.52, pixel: 2, kind: 'pine' },
    { x: 0.53, y: 0.8, pixel: 2, kind: 'birch' },
    { x: 0.57, y: 0.9, pixel: 2, kind: 'oak' },
    { x: 0.82, y: 0.72, pixel: 2, kind: 'spruce' },
    { x: 0.86, y: 0.86, pixel: 3, kind: 'cypress' },
    { x: 0.88, y: 0.96, pixel: 2, kind: 'pine', variant: 1 },
    { x: 0.9, y: 0.6, pixel: 1, kind: 'birch' },
  ]
  return trees
    .sort((a, b) => a.y - b.y)
    .map(
      ({ x, y, pixel, kind, variant = 0 }) =>
        `<g class="trail-tree" data-tree="${kind}" data-variant="${variant}" transform="translate(${Math.round(width * x)} ${Math.round(bottom + clearing * y)})">${tree(kind, variant, pixel)}</g>`,
    )
    .join('')
}
