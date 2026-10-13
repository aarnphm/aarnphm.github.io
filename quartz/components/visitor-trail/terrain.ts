import { grassGrows, type Landscape, type Pond } from './landscape'
import { trailMeadow } from './meadow'
import { pixels } from './pixels'
import { trailRandom } from './random'
import { trailBeavers, trailFords, trailRavine } from './ravine'
import { trailWayside } from './wayside'
import { trailGroundSnow } from './winter'

function oval(pond: Pond, margin: number, grid: number) {
  const rows: string[] = []
  for (let y = -pond.ry - margin; y <= pond.ry + margin; y += grid) {
    const span = (pond.rx + margin) * Math.sqrt(Math.max(0, 1 - (y / (pond.ry + margin)) ** 2))
    const padding = Math.max(0, Math.floor((pond.rx + margin - span) / grid))
    rows.push('.'.repeat(padding) + 'W'.repeat(Math.max(1, Math.round((span * 2) / grid))))
  }
  return pixels(rows, { W: 'trail-water' }, grid)
}

export function trailTerrain(landscape: Landscape, seed: number, routes: readonly string[] = []) {
  const { width, height, contours, pond } = landscape
  const random = trailRandom(seed ^ 0x94d049bb)
  const terraces = contours
    .map((points, index) => {
      const [first, ...rest] = points
      const path = `M${first.x} ${first.y}${rest.map(point => `H${point.x}V${point.y}`).join('')}V${height}H0Z`
      return `<path class="trail-terrace trail-terrace-${index}" d="${path}"/>`
    })
    .join('')
  const grass = contours
    .flatMap(points =>
      points
        .filter((_, index) => index % 5 === 2)
        .map(point => {
          const x = Math.round(point.x + random() * 12)
          const y = point.y + 10 + Math.round(random() * 12)
          if (!grassGrows(y, landscape, random())) return ''
          return `<path class="trail-meadow-tuft" d="M${x} ${y}h2v-4h-2ZM${x + 4} ${y + 2}h2v-6h-2Z"/>`
        }),
    )
    .join('')
  const ripples = Array.from({ length: 8 }, () => {
    const x = pond.x + (random() - 0.5) * pond.rx * 1.2
    const y = pond.y + (random() - 0.5) * pond.ry * 1.1
    const length = 6 + Math.round(random() * 8)
    return `<path class="trail-pond-ripple" d="M${Math.round(x)} ${Math.round(y)}h${length}v2h-${length}Z"/>`
  }).join('')
  const reeds = Array.from({ length: 10 }, (_, index) => {
    const angle = index * 2.399963 + random() * 0.24
    const x = Math.round(pond.x + Math.cos(angle) * pond.rx * 1.08)
    const y = Math.round(pond.y + Math.sin(angle) * pond.ry * 1.08)
    const size = index % 3 ? 1 : 2
    return `<g class="trail-pond-reeds" data-pond-plant="reeds" data-ground-x="${x}" data-ground-y="${y}" transform="translate(${x - size * 3} ${y - size * 6})">${pixels(['..L...', '.LL.C.', '.LL.C.', 'L.L.C.', 'L.L.D.', '..L.D.'], { L: 'trail-meadow-reed', C: 'trail-reed-head', D: 'trail-tree-trunk' }, size)}</g>`
  }).join('')
  const bankPlants = Array.from({ length: 14 }, (_, index) => {
    const angle = index * 2.399963 + 0.8
    const radius = 1.2 + random() * 0.2
    const x = Math.round(pond.x + Math.cos(angle) * pond.rx * radius)
    const y = Math.round(pond.y + Math.sin(angle) * pond.ry * radius)
    const rows =
      index % 3
        ? ['...L..', '.L.LL.', '..LLLL', '.LLL..']
        : ['.C..C.', '.LL.L.', 'LLLLL.', '.LLL..']
    return `<g class="trail-pond-bank-plant" data-pond-plant="bank" data-ground-x="${x}" data-ground-y="${y}" transform="translate(${x - 6} ${y - 8})">${pixels(rows, { L: 'trail-meadow-reed', C: 'trail-reed-head' }, 2)}</g>`
  }).join('')
  const lilyPads = Array.from({ length: 5 }, (_, index) => {
    const angle = index * 2.399963 + 0.4
    const radius = 0.72 + random() * 0.16
    const x = Math.round(pond.x + Math.cos(angle) * pond.rx * radius)
    const y = Math.round(pond.y + Math.sin(angle) * pond.ry * radius)
    return `<g class="trail-pond-lily" data-pond-plant="lily" data-ground-x="${x}" data-ground-y="${y}" transform="translate(${x - 4} ${y - 2})">${pixels(['.LLL.', 'LLLL.', '.LL.L'], { L: 'trail-meadow-reed' }, 2)}</g>`
  }).join('')
  const ice = Array.from({ length: 7 }, (_, index) => {
    const angle = index * 2.399963
    const x = Math.round(pond.x + Math.cos(angle) * pond.rx * 0.58)
    const y = Math.round(pond.y + Math.sin(angle) * pond.ry * 0.58)
    return `M${x} ${y}h8l4 -2h6m-10 2l4 4h5`
  }).join('')
  const grid = width < 700 ? 3 : 4
  const bounds = {
    alpine: landscape.alpine,
    treeline: landscape.treeline,
    meadowTop: landscape.meadowTop,
  }
  return `<g class="trail-terrain" data-trail-terrain='${JSON.stringify(bounds)}'>${terraces}${trailGroundSnow(landscape, seed)}${trailRavine(landscape, seed, routes)}${trailFords(landscape, routes)}${grass}${trailMeadow(landscape, seed, routes)}${trailWayside(landscape, routes, seed)}</g>
    <g class="trail-pond" data-trail-pond='${JSON.stringify(pond)}'>
      <g class="trail-pond-bank" transform="translate(${pond.x - pond.rx - 8} ${pond.y - pond.ry - 8})">${oval(pond, 8, grid)}</g>
      <g transform="translate(${pond.x - pond.rx} ${pond.y - pond.ry})">${oval(pond, 0, grid)}</g>
      <path class="trail-pond-ice" d="${ice}"/>
      ${ripples}${lilyPads}${reeds}${bankPlants}
      ${trailBeavers(landscape, seed)}
    </g>`
}
