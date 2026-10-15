import { creekDistance, grassGrows, pondDistance, type Landscape } from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'

type Point = { x: number; y: number }
type Plant =
  | 'fern'
  | 'shrub'
  | 'sedge'
  | 'sapling'
  | 'flowers'
  | 'log'
  | 'stump'
  | 'stone'
  | 'grass'
  | 'manzanita'
  | 'lupine'
  | 'poppy'
  | 'boulder'
// Plants low enough to read as grass; they keep to the middle and far meadow.
const low = new Set<Plant>([
  'fern',
  'shrub',
  'sedge',
  'flowers',
  'grass',
  'manzanita',
  'lupine',
  'poppy',
])
export type UnderstoryTree = Point & { kind: string; standalone: boolean; foothill: boolean }

// Sprites stand on their bottom row. Rows above `rooted` sway with the leaf clusters; the rest stay.
const sprites: Record<Plant, { rows: readonly string[]; rooted: number; lift: number }> = {
  fern: {
    rows: [
      '.....L.L.....',
      '..LL..L..LL..',
      '.L..L.D.L..L.',
      'L....LDL....L',
      '.LL..DDD..LL.',
      '...LDD.DDL...',
      '.....DDD.....',
    ],
    rooted: 2,
    lift: 0.7,
  },
  shrub: {
    rows: [
      '...LLLL....',
      '.LLLLDLLL..',
      'LLDLLLLDLL.',
      'LDLLBLDLLLL',
      'DLDDLDDLBDL',
      '.DDDDDDDDD.',
    ],
    rooted: 3,
    lift: 0.45,
  },
  sedge: { rows: ['.L...L.', '.L.L.L.', 'L.LLL.L', '.LDLDL.', '..DDD..'], rooted: 2, lift: 0.9 },
  sapling: {
    rows: ['..L..', '.LDL.', 'LDLDL', '.LDL.', 'LDDDL', '..T..', '..T..', '..T..'],
    rooted: 3,
    lift: 0.6,
  },
  flowers: { rows: ['.P..Y..P', 'PSPYSY.S', '.S..S..S', 'DSD.SDDS'], rooted: 2, lift: 0.8 },
  log: {
    rows: ['.MMMMMMMMMMMMMMM..', 'OCOTTTTKTTTTTTTTT.', 'OOOTTTTTTTTTKTTTTM', '.TTTTTTTTTTTTTTTT.'],
    rooted: 4,
    lift: 0,
  },
  stump: { rows: ['.MOOOO.', 'OCCCCCO', 'TTKTTTT', 'TTTTKTT', 'TT.T.TT'], rooted: 5, lift: 0 },
  stone: { rows: ['..MMMM..', '.MMSSMM.', 'SSSSSSSE', '.EEEEEE.'], rooted: 4, lift: 0 },
  // Sierra meadow: bunchgrass that cures gold by midsummer, red-barked manzanita, lupine and
  // poppy drifts, and granite erratics left by the glaciers.
  grass: { rows: ['.G..G.G.', 'G.G.GG.G', '.GGGG.G.', '..GDDG..'], rooted: 1, lift: 0.9 },
  manzanita: {
    rows: ['..LLL.LL..', '.LLLLLLLL.', 'LLRLLLRLLL', '.LR.LRLL..', '..RR.R....', '...RRR....'],
    rooted: 2,
    lift: 0.4,
  },
  lupine: { rows: ['.U...U.', '.U.U.U.', 'UU.U.UU', '.S.U.S.', '.SSSSS.'], rooted: 1, lift: 0.8 },
  poppy: { rows: ['Q..Q.Q', 'S.QS.S', 'S.S.SS'], rooted: 1, lift: 0.8 },
  boulder: {
    rows: [
      '....HHHH....',
      '..HHHHHNNN..',
      '.HHHNNNNNNN.',
      'HHNNNNNNNNNE',
      'NNNNNNNNNNEE',
      '.EEEEEEEEEE.',
    ],
    rooted: 6,
    lift: 0,
  },
}
const tones: Record<string, string> = {
  L: 'trail-under-light',
  D: 'trail-under-dark',
  B: 'trail-under-berry',
  T: 'trail-under-wood',
  K: 'trail-under-knot',
  O: 'trail-under-ring',
  C: 'trail-under-heart',
  M: 'trail-under-moss',
  S: 'trail-under-stem',
  P: 'trail-under-petal',
  Y: 'trail-under-pollen',
  E: 'trail-under-stone-edge',
  G: 'trail-under-grass',
  R: 'trail-under-manzanita',
  U: 'trail-under-lupine',
  Q: 'trail-under-poppy',
  H: 'trail-under-granite-lit',
  N: 'trail-under-granite',
}
const caps = ['.RRR..RR', 'RWRRR.RR', '..S....S']
const needled = new Set(['pine', 'spruce', 'fir', 'hemlock', 'cypress', 'larch', 'sequoia', 'snag'])
const granite = [
  sprites.boulder.rows,
  [
    '.....HH...',
    '...HHHHN..',
    '..HHHNNNN.',
    '.HHNNNNNNE',
    'HHNNNNNNEE',
    'NNNNNNNEEE',
    '.EEEEEEEE.',
  ],
  ['..HHHHHH......', '.HHHNNNNN..HH.', 'HHNNNNNNNEHHNN', 'NNNNNNNNEENNNE', '.EEEEEEEEE.EE.'],
]

// Conifer floors are shaded and acid: ferns, moss and deadfall. Hardwood floors take shrubs,
// saplings and spring flowers. Open ground under a lone tree is mostly sedge.
function choose(tree: UnderstoryTree, roll: number): Plant {
  if (tree.standalone) return roll < 0.55 ? 'sedge' : roll < 0.8 ? 'flowers' : 'stone'
  if (needled.has(tree.kind))
    return roll < 0.5
      ? 'fern'
      : roll < 0.62
        ? 'sedge'
        : roll < 0.72
          ? 'log'
          : roll < 0.8
            ? 'stump'
            : roll < 0.9
              ? 'stone'
              : 'sapling'
  return roll < 0.32
    ? 'shrub'
    : roll < 0.5
      ? 'fern'
      : roll < 0.64
        ? 'sapling'
        : roll < 0.78
          ? 'flowers'
          : roll < 0.86
            ? 'log'
            : roll < 0.93
              ? 'sedge'
              : 'stump'
}

function nearest(point: Point, points: readonly Point[]) {
  let distance = Infinity
  for (const other of points)
    distance = Math.min(distance, (point.x - other.x) ** 2 + (point.y - other.y) ** 2)
  return Math.sqrt(distance)
}

function plant(
  kind: Plant,
  point: Point,
  depth: number,
  scale: number,
  random: () => number,
  index: number | string,
) {
  const sprite = sprites[kind]
  const rows = kind === 'boulder' ? granite[Math.floor(random() * granite.length)] : sprite.rows
  const cell = depth > 0.45 ? 2 : 1
  const wide = rows[0].length * cell
  const tall = rows.length * cell
  const split = kind === 'boulder' ? 0 : rows.length - sprite.rooted
  const mirrored = random() < 0.5 ? -1 : 1
  const sway =
    sprite.lift && split > 0
      ? `<g class="trail-tree-motion"><g class="trail-tree-leaves" style="--leaf-x:0;--leaf-y:${-tall};--leaf-lift:${sprite.lift};--leaf-cell:${cell}px">${pixels(rows.slice(0, split), tones, cell)}</g></g>`
      : ''
  const base = pixels(rows.slice(split), tones, cell)
  const fungi =
    (kind === 'log' || kind === 'stump') && random() < 0.6
      ? `<g class="trail-under-fungi" transform="translate(${wide - 3 * cell} ${tall - 3 * cell})">${pixels(caps, { R: 'trail-under-cap', W: 'trail-under-petal', S: 'trail-under-stem' }, cell)}</g>`
      : ''
  const solid =
    kind === 'boulder'
      ? `data-camp-solid="boulder-${index}" data-solid-radius="${(wide * scale * 0.32).toFixed(2)}"`
      : ''
  return `<g class="trail-understory" data-plant="${kind}" data-understory-id="understory-${index}" ${solid} data-ground-x="${point.x}" data-ground-y="${point.y}" transform="translate(${point.x} ${point.y})"><g transform="scale(${(mirrored * scale).toFixed(3)} ${scale.toFixed(3)}) translate(${-wide / 2} ${-tall})">${sway}<g transform="translate(0 ${split * cell})">${base}</g>${fungi}</g></g>`
}

export function trailBoulder(point: Point, seed: number, index: number) {
  return plant('boulder', point, 1, 1, trailRandom(seed), index)
}

// The trunk a slot's last tree dropped. Its successor roots on the log, so the log lies across the
// young trunk's foot until it rots.
export function trailDeadfall(landscape: Landscape, tree: Point & { id: string }, seed: number) {
  const { meadowTop, camp } = landscape
  const depth = Math.max(0, Math.min(1, (tree.y - meadowTop) / Math.max(1, camp.y - meadowTop)))
  const random = trailRandom(seed)
  const point = { x: Math.round(tree.x + (random() - 0.5) * 12), y: tree.y - 1 }
  return {
    id: `deadfall-${tree.id}`,
    bottom: point.y + 11,
    markup: plant('log', point, depth, 0.55 + depth * 0.6, random, `deadfall-${tree.id}`),
  }
}

export function* trailUnderstory(
  landscape: Landscape,
  seed: number,
  trees: readonly UnderstoryTree[],
  walks: readonly Point[],
  clearings: readonly (Point & { radius: number })[],
) {
  const { width, meadowTop, camp, pond } = landscape
  const clearing = camp.y - meadowTop
  const random = trailRandom(seed ^ 0x6d2b79f5)
  const placed: Point[] = []
  const trunks = trees.filter(tree => !tree.foothill)
  let index = 0
  for (const tree of trunks) {
    const count = tree.standalone ? 1 + Math.floor(random() * 2) : 2 + Math.floor(random() * 3)
    for (let attempt = 0; attempt < count; attempt++) {
      const depth = Math.max(0, Math.min(1, (tree.y - meadowTop) / clearing))
      const scale = 0.55 + depth * 0.6
      const angle = random() * Math.PI * 2
      const reach = (9 + random() * 22) * scale
      // The ground plane is foreshortened, so the ring around a trunk is an ellipse.
      const point = {
        x: Math.round(tree.x + Math.cos(angle) * reach),
        y: Math.round(tree.y + Math.sin(angle) * reach * 0.42 + 2),
      }
      const kind = choose(tree, random())
      if (
        (low.has(kind) && !grassGrows(point.y, landscape, random())) ||
        point.x < 10 ||
        point.x > width - 10 ||
        point.y < meadowTop + 60 ||
        pondDistance(point, pond) < 1.95 ||
        creekDistance(point, landscape) < 12 ||
        Math.hypot(point.x - camp.x, point.y - camp.y) < 96 ||
        clearings.some(site => Math.hypot(point.x - site.x, point.y - site.y) < site.radius) ||
        nearest(point, walks) < 10 ||
        nearest(point, placed) < (kind === 'log' ? 16 : 7) ||
        nearest(point, trunks) < 5
      )
        continue
      placed.push(point)
      yield {
        id: `understory-${index}`,
        bottom: point.y + 12,
        markup: plant(kind, point, depth, scale, random, index++),
      }
    }
  }
  // The open meadow between groves: bunchgrass in patches, lupine and poppy in drifts, manzanita
  // and granite erratics scattered alone, sedge and fern along the creek.
  const meadow = trailRandom(seed ^ 0x3c6ef372)
  const phase = [meadow() * 6, meadow() * 6, meadow() * 6]
  const patch = (x: number, y: number) =>
    Math.sin(x * 0.012 + phase[0]) * Math.sin(y * 0.045 + phase[1]) +
    0.6 * Math.sin(x * 0.029 - y * 0.018 + phase[2])
  const drift = (x: number, y: number) =>
    Math.sin(x * 0.021 + phase[1]) * Math.sin(y * 0.06 - phase[0])
  const candidates = Math.round((width * Math.max(1, clearing)) / 1700)
  for (let attempt = 0; attempt < candidates; attempt++) {
    const point = {
      x: Math.round(14 + meadow() * (width - 28)),
      y: Math.round(meadowTop + 62 + meadow() * Math.max(1, landscape.height - meadowTop - 80)),
    }
    const roll = meadow()
    const wet = creekDistance(point, landscape)
    const kind: Plant =
      wet < 34
        ? roll < 0.55
          ? 'sedge'
          : roll < 0.8
            ? 'fern'
            : 'stone'
        : patch(point.x, point.y) > 0.35
          ? roll < 0.78
            ? 'grass'
            : 'sedge'
          : drift(point.x, point.y) > 0.55
            ? roll < 0.55
              ? 'lupine'
              : 'poppy'
            : roll < 0.28
              ? 'manzanita'
              : roll < 0.46
                ? 'boulder'
                : roll < 0.54
                  ? 'log'
                  : roll < 0.78
                    ? 'grass'
                    : 'stone'
    const room = kind === 'boulder' || kind === 'log' ? 20 : kind === 'manzanita' ? 12 : 8
    if (
      (low.has(kind) && !grassGrows(point.y, landscape, meadow())) ||
      pondDistance(point, pond) < 1.6 ||
      wet < 12 ||
      Math.hypot(point.x - camp.x, point.y - camp.y) < 92 ||
      clearings.some(site => Math.hypot(point.x - site.x, point.y - site.y) < site.radius) ||
      nearest(point, walks) < (kind === 'boulder' ? room + 8 : 9) ||
      nearest(point, trunks) < 7 ||
      nearest(point, placed) < room
    )
      continue
    placed.push(point)
    const depth = Math.max(0, Math.min(1, (point.y - meadowTop) / clearing))
    const scale = (0.55 + depth * 0.6) * (kind === 'boulder' ? 1.2 + meadow() * 0.8 : 1)
    yield {
      id: `understory-${index}`,
      bottom: point.y + 12,
      markup: plant(kind, point, depth, scale, meadow, index++),
    }
  }
}
