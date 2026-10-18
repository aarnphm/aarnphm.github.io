import { grassGrows, type Landscape, type Pond } from './landscape'
import { trailMeadow } from './meadow'
import { pixels } from './pixels'
import { pondBlackbird, trailPondLife } from './pond-life'
import { trailRandom } from './random'
import { trailBeavers, trailFords, trailRavine } from './ravine'
import { reliefStrengths } from './relief'
import { trailWayside } from './wayside'
import { trailPuddles } from './wetlands'
import { trailSnowBase } from './winter'

function oval(pond: Pond, margin: number, grid: number) {
  const rows: string[] = []
  for (let y = -pond.ry - margin; y <= pond.ry + margin; y += grid) {
    const span = (pond.rx + margin) * Math.sqrt(Math.max(0, 1 - (y / (pond.ry + margin)) ** 2))
    const padding = Math.max(0, Math.floor((pond.rx + margin - span) / grid))
    rows.push('.'.repeat(padding) + 'W'.repeat(Math.max(1, Math.round((span * 2) / grid))))
  }
  return pixels(rows, { W: 'trail-water' }, grid)
}

// Each terrace steps down toward the viewer along its contour, and the step shows as a ledge under
// the edge, the way a top-down pixel map draws a drop: a line of riser, and on the two near
// contours a face of two or four pixels with a line of shadow on the field below, in the same
// planes as the landforms. The ledge fades out and in over long stretches, so it reads as a
// contour that the flats swallow, never as a rule across the page. Each contour is its own group,
// so the far ledges are fainter and thinner than the near ones.
function terraceRisers(landscape: Landscape, seed: number) {
  const random = trailRandom(seed ^ 0x27d4eb2f)
  const groups: string[] = []
  landscape.contours.forEach((points, level) => {
    const rims: string[] = []
    const faces: string[] = []
    const shadows: string[] = []
    const drop = level < 2 ? 0 : level === 2 ? 2 : 4
    const phase = random() * Math.PI * 2
    const cover = (x: number) => 0.72 + 0.26 * Math.sin(x / 131 + phase)
    for (let p = 0; p + 1 < points.length; p++) {
      const a = points[p]
      const b = points[p + 1]
      for (let x = a.x; x < b.x; ) {
        const length = Math.min(b.x - x, 32 + Math.floor(random() * 16) * 4)
        if (random() < cover(x + length / 2)) {
          rims.push(`M${x} ${a.y}h${length}v2h-${length}Z`)
          if (drop) {
            faces.push(`M${x} ${a.y + 2}h${length}v${drop}h-${length}Z`)
            shadows.push(`M${x} ${a.y + 2 + drop}h${length}v2h-${length}Z`)
          }
        }
        x += length
      }
      if (b.y !== a.y)
        rims.push(
          `M${b.x - (b.y > a.y ? 2 : 0)} ${Math.min(a.y, b.y)}h2v${Math.abs(b.y - a.y) + 2 + drop}h-2Z`,
        )
    }
    if (!rims.length) return
    const path = (tone: string, runs: string[]) =>
      runs.length ? `<path class="${tone}" d="${runs.join('')}"/>` : ''
    groups.push(
      `<g class="trail-terrace-ledge" style="--trail-landform-field:var(--trail-field-${level});--trail-relief-strength:${reliefStrengths[level] ?? 1}">${path(`trail-terrace-riser trail-terrace-riser-${level}`, rims)}${path('trail-ravine-face', faces)}${path('trail-ravine-shadow', shadows)}</g>`,
    )
  })
  return groups.join('')
}

export function trailTerraces(landscape: Landscape, seed: number) {
  const { height, contours } = landscape
  const terraces = contours
    .map((points, index) => {
      const [first, ...rest] = points
      const path = `M${first.x} ${first.y}${rest.map(point => `H${point.x}V${point.y}`).join('')}V${height}H0Z`
      return `<path class="trail-terrace trail-terrace-${index}" d="${path}"/>`
    })
    .join('')
  return terraces + terraceRisers(landscape, seed) + trailSnowBase(landscape, seed)
}

export function trailTerrain(landscape: Landscape, seed: number, routes: readonly string[] = []) {
  const { width, contours, pond } = landscape
  const random = trailRandom(seed ^ 0x94d049bb)
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
  // A red-winged blackbird keeps the fourth reed, a tall one on the near right shore.
  const reeds = Array.from({ length: 10 }, (_, index) => {
    const angle = index * 2.399963 + random() * 0.24
    const x = Math.round(pond.x + Math.cos(angle) * pond.rx * 1.08)
    const y = Math.round(pond.y + Math.sin(angle) * pond.ry * 1.08)
    const size = index % 3 ? 1 : 2
    const guest = index === 3 ? pondBlackbird(size, 14 + random() * 6, -random() * 20) : ''
    return `<g class="trail-pond-reeds" data-pond-plant="reeds" data-ground-x="${x}" data-ground-y="${y}" transform="translate(${x - size * 3} ${y - size * 6})">${pixels(['..L...', '.LL.C.', '.LL.C.', 'L.L.C.', 'L.L.D.', '..L.D.'], { L: 'trail-meadow-reed', C: 'trail-reed-head', D: 'trail-tree-trunk' }, size)}${guest}</g>`
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
  const life = trailPondLife(landscape, seed)
  // The pet's keyboard walk keeps to this ground and off the pond.
  const bounds = {
    alpine: landscape.alpine,
    treeline: landscape.treeline,
    meadowTop: landscape.meadowTop,
    pond,
  }
  return `<g class="trail-terrain" data-trail-terrain='${JSON.stringify(bounds)}'>${trailRavine(landscape, seed, routes)}${trailPuddles(landscape, seed, routes)}${trailFords(landscape, routes)}${grass}${trailMeadow(landscape, seed, routes)}${trailWayside(landscape, routes, seed)}</g>
    <g class="trail-pond" data-trail-pond='${JSON.stringify(pond)}'>
      <g class="trail-pond-bank" transform="translate(${pond.x - pond.rx - 8} ${pond.y - pond.ry - 8})">${oval(pond, 8, grid)}</g>
      <g transform="translate(${pond.x - pond.rx} ${pond.y - pond.ry})">${oval(pond, 0, grid)}</g>
      <path class="trail-pond-ice" d="${ice}"/>
      ${life.under}${ripples}${lilyPads}${reeds}${bankPlants}
      ${trailBeavers(landscape, seed)}${life.water}
    </g>`
}
