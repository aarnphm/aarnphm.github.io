import type { Landscape, Point } from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'

// Pond inhabitants beyond the geese and the beaver. Every sprite is drawn in the 2 px register
// and faces right; a facing group mirrors it. The water group is inside `.trail-pond`, under the
// fauna layer, so the geese swim over the ducks and the fish; the heron stands in the shallows
// from the pasture, where the depth sort puts it among the trees. Ice hides all of them but the
// log, night hides the day birds and the insects, and the turtle only basks on a fair day.
const cell = 2

const duckTones = {
  G: 'trail-duck-head',
  W: 'trail-duck-white',
  R: 'trail-duck-breast',
  S: 'trail-duck-flank',
  K: 'trail-duck-dark',
  H: 'trail-duck-hen',
  M: 'trail-duck-mottle',
  B: 'trail-goose-bill',
  E: 'trail-animal-eye',
}
const drake = ['......GG..', '......GEBB', '......GG..', '.K.....W..', '.KSSSSSRR.', '..SSSSSR..']
const hen = ['......HH..', '......HEBB', '......HH..', '.M.....H..', '.MHMHHHHH.', '..HHMHHH..']
// Tail up, head under: the dabbling frame swaps in at the far end of each cruise.
const tipped = (body: string, dark: string) => [
  '..........',
  `...${dark}......`,
  `..${dark}${dark}......`,
  `..${body}${body}${body}${body}....`,
  `.${body}${body}${body}${body}${body}${body}...`,
  `..${body}${body}${body}${body}....`,
]

const fishTones = {
  S: 'trail-river-fish-silver',
  F: 'trail-river-fish-flank',
  D: 'trail-river-fish-fin',
  E: 'trail-animal-eye',
}
const fish = ['.SSS..D', 'EFFFSDD', '.SSS..D']
const ring = ['..WWWW..', '.W....W.', 'W......W', '.W....W.', '..WWWW..']
const shade = ['..DDDD..', 'DDDDDDDD', '..DDDD..']

const heronTones = {
  A: 'trail-heron-plume',
  K: 'trail-jay-ink',
  W: 'trail-jay-white',
  Y: 'trail-heron-bill',
  L: 'trail-heron-leg',
  E: 'trail-animal-eye',
}
const heronStand = [
  '.....KKAA....',
  '.....AAAEYYYY',
  '.....AAW.....',
  '....AAW......',
  '....AW.......',
  '...AAA.......',
  '..AAAAAA.....',
  '.AAAAAAAA....',
  '..KAAAAAA....',
  '...AAAAA.....',
  '....L..L.....',
  '....L..L.....',
]
const heronStrike = [
  '.............',
  '.............',
  '.............',
  '.............',
  '.............',
  '...AAA.......',
  '..AAAAAAW....',
  '.AAAAAAAAW...',
  '..KAAAAAAAW..',
  '...AAAAA.AAK.',
  '....L..L..AEY',
  '....L..L...YY',
]

const blackbirdTones = { K: 'trail-jay-ink', R: 'trail-woodpecker-nape', B: 'trail-heron-leg' }
const blackbird = ['...KKB', '.RKKK.', '.KKKK.', 'KK.K..']

const logTones = { L: 'trail-tree-bark', T: 'trail-tree-trunk' }
const log = ['.LLLLLLLLLL', 'TTTTTTTTTTT']
const turtleTones = { S: 'trail-turtle-shell', L: 'trail-turtle-skin' }
const turtleBody = ['..SSSS.', '.SSSSSS', '.L.L..L']
const turtleHead = ['LL']

const dragonflyTones = { K: 'trail-dragonfly-body', W: 'trail-dragonfly-wing' }
const dragonfly = ['..K..', 'WWKWW', 'WWKWW', '..K..', '..K..']

const at = (x: number, y: number) => `translate(${Math.round(x)} ${Math.round(y)})`

function duck(
  kind: 'drake' | 'hen',
  point: Point,
  dx: number,
  dy: number,
  cycle: number,
  delay: number,
) {
  const rows = kind === 'drake' ? drake : hen
  const tip = kind === 'drake' ? tipped('S', 'K') : tipped('H', 'M')
  const face = dx < 0 ? -1 : 1
  return `<g class="trail-duck trail-duck-${kind}" transform="${at(point.x, point.y)}" style="--pond-dx:${dx.toFixed(1)}px;--pond-dy:${dy.toFixed(1)}px;--pond-face:${face};--pond-cycle:${cycle.toFixed(1)}s;--pond-delay:${delay.toFixed(1)}s"><g class="trail-pond-cruise"><g class="trail-pond-facing"><g transform="translate(${-5 * cell} ${-6 * cell})"><path class="trail-pond-wake" d="M-4 ${6 * cell}h${12 * cell}v1H-4Z"/><g class="trail-duck-up">${pixels(rows, duckTones, cell)}</g><g class="trail-duck-tip">${pixels(tip, duckTones, cell)}</g></g></g></g></g>`
}

function jumper(point: Point, cycle: number, delay: number) {
  return `<g class="trail-pond-fish" transform="${at(point.x, point.y)}" style="--pond-cycle:${cycle.toFixed(1)}s;--pond-delay:${delay.toFixed(1)}s"><g class="trail-pond-fish-ring" transform="translate(${-4 * cell} ${-2.5 * cell})">${pixels(ring, { W: 'trail-pond-ripple' }, cell)}</g><g class="trail-pond-fish-jump"><g transform="translate(${-3.5 * cell} ${-3 * cell})">${pixels(fish, fishTones, cell)}</g></g></g>`
}

function glider(point: Point, dx: number, dy: number, cycle: number, delay: number) {
  const face = dx < 0 ? -1 : 1
  return `<g class="trail-pond-fish-shade" transform="${at(point.x, point.y)}" style="--pond-dx:${dx.toFixed(1)}px;--pond-dy:${dy.toFixed(1)}px;--pond-face:${face};--pond-cycle:${cycle.toFixed(1)}s;--pond-delay:${delay.toFixed(1)}s"><g class="trail-pond-cruise"><g class="trail-pond-facing"><g transform="translate(${-4 * cell} ${-1.5 * cell})">${pixels(shade, { D: 'trail-pond-shade' }, cell)}</g></g></g></g>`
}

// Perched on the head of a reed: `size` is the reed's cell, and the origin is the reed group's.
export function pondBlackbird(size: number, cycle: number, delay: number) {
  const x = 4 * size + size / 2 - 3 * cell
  const y = size - 4 * cell
  return `<g class="trail-pond-blackbird" transform="translate(${x} ${y})" style="--pond-cycle:${cycle.toFixed(1)}s;--pond-delay:${delay.toFixed(1)}s"><g class="trail-pond-hop"><g class="trail-pond-facing" transform="translate(${3 * cell} 0)"><g transform="translate(${-3 * cell} 0)">${pixels(blackbird, blackbirdTones, cell)}</g></g></g></g>`
}

export function trailPondLife(landscape: Landscape, seed: number) {
  const { pond, width } = landscape
  const narrow = width < 700
  const spot = (angle: number, radius: number): Point => ({
    x: pond.x + Math.cos(angle) * pond.rx * radius,
    y: pond.y + Math.sin(angle) * pond.ry * radius,
  })

  // Ducks cruise a chord across the far right of the water, clear of the lodge on the far left
  // and of the beaver's route to the near right shore.
  const duckRandom = trailRandom(seed ^ 0x5a1d3c7b)
  const start = { x: pond.x + pond.rx * (0.02 + duckRandom() * 0.08), y: pond.y - pond.ry * 0.42 }
  const reach = pond.rx * (0.42 + duckRandom() * 0.14)
  const drift = pond.ry * (0.1 + duckRandom() * 0.1)
  const cruise = 34 + duckRandom() * 10
  const lead = -duckRandom() * cruise
  const ducks =
    duck('drake', start, reach, drift, cruise, lead) +
    (narrow
      ? ''
      : duck(
          'hen',
          { x: start.x - 16, y: start.y + 3 },
          reach * 0.9,
          drift * 0.8,
          cruise,
          lead - 1.8,
        ))

  // Rises in the near water, where the viewer looks into it, and shadows gliding beneath.
  const fishRandom = trailRandom(seed ^ 0x2f7c9e15)
  const jumps = Array.from({ length: narrow ? 2 : 3 }, (_, index) =>
    jumper(
      spot(0.9 + index * 0.55 + fishRandom() * 0.3, 0.28 + fishRandom() * 0.26),
      16 + fishRandom() * 10,
      -fishRandom() * 26,
    ),
  ).join('')
  const shadows = Array.from({ length: narrow ? 1 : 2 }, (_, index) => {
    const from = spot(1.1 + index * 1.1 + fishRandom() * 0.4, 0.45 + fishRandom() * 0.2)
    const side = index % 2 ? -1 : 1
    return glider(
      from,
      side * pond.rx * (0.18 + fishRandom() * 0.16),
      (fishRandom() - 0.5) * pond.ry * 0.3,
      28 + fishRandom() * 14,
      -fishRandom() * 40,
    )
  }).join('')

  // A dead log on the far bank, its near end in the water, and a turtle up on it in the sun.
  const logRandom = trailRandom(seed ^ 0x7b3e1a92)
  const rest = spot(-0.85 + (logRandom() - 0.5) * 0.2, 0.98)
  const logCycle = 18 + logRandom() * 8
  const perch = `<g class="trail-pond-log" transform="${at(rest.x - 11 * cell, rest.y - 2 * cell)}">${pixels(log, logTones, cell)}<g class="trail-turtle" style="--pond-cycle:${logCycle.toFixed(1)}s;--pond-delay:${(-logRandom() * logCycle).toFixed(1)}s" transform="translate(${2 * cell} ${-3 * cell})">${pixels(turtleBody, turtleTones, cell)}<g class="trail-turtle-head" transform="translate(${7 * cell} ${cell})">${pixels(turtleHead, turtleTones, cell)}</g></g></g>`

  // Dragonflies hunt over the reeds on the near shore through the summer.
  const flyRandom = trailRandom(seed ^ 0x1c4a8d37)
  const dragonflies = Array.from({ length: narrow ? 2 : 3 }, (_, index) => {
    const hover = spot(0.5 + index * 0.75 + flyRandom() * 0.4, 1.02 + flyRandom() * 0.16)
    const cycle = 9 + flyRandom() * 5
    return `<g class="trail-dragonfly" transform="${at(hover.x, hover.y - 10)} scale(${index % 2 ? -1 : 1} 1)" style="--pond-cycle:${cycle.toFixed(1)}s;--pond-delay:${(-flyRandom() * cycle).toFixed(1)}s"><g class="trail-dragonfly-dart"><g transform="translate(-2.5 -2.5)">${pixels(dragonfly, dragonflyTones, 1)}</g></g></g>`
  }).join('')

  // The heron wades at the near left shore, feet in the water, among the reeds.
  const heronRandom = trailRandom(seed ^ 0x3d9f6b21)
  const stand = spot(2.05 + heronRandom() * 0.35, 0.94)
  const heronCycle = 22 + heronRandom() * 8
  const foot = { x: Math.round(stand.x), y: Math.round(stand.y) }
  const heron = `<g class="trail-wetland-animal trail-heron" data-wetland-animal="heron" data-trail-collision="animal" data-collision-id="animal-heron-0" data-collision-radius="10" data-ground-x="${foot.x}" data-ground-y="${foot.y}" data-phase="resting" transform="${at(foot.x, foot.y)}" style="--pond-cycle:${heronCycle.toFixed(1)}s;--pond-delay:${(-heronRandom() * heronCycle).toFixed(1)}s;--pond-face:${stand.x > pond.x ? -1 : 1}"><g class="trail-pond-facing"><g transform="translate(${-6 * cell} ${-12 * cell})"><g class="trail-heron-stand">${pixels(heronStand, heronTones, cell)}</g><g class="trail-heron-strike">${pixels(heronStrike, heronTones, cell)}<path class="trail-heron-splash trail-pond-ripple" d="M${10 * cell} ${12 * cell}h${2 * cell}v1h-${2 * cell}ZM${13 * cell} ${11 * cell}h${cell}v1h-${cell}Z"/></g></g></g></g>`

  return {
    under: shadows,
    water: ducks + jumps + perch + dragonflies,
    shore: heron,
    sites: [foot, { x: foot.x, y: foot.y + 20 }],
  }
}
