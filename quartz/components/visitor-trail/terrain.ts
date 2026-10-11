import type { Landscape, Pond } from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'

function oval(pond: Pond, margin: number, grid: number) {
  const rows: string[] = []
  for (let y = -pond.ry - margin; y <= pond.ry + margin; y += grid) {
    const span = (pond.rx + margin) * Math.sqrt(Math.max(0, 1 - (y / (pond.ry + margin)) ** 2))
    const padding = Math.max(0, Math.floor((pond.rx + margin - span) / grid))
    rows.push('.'.repeat(padding) + 'W'.repeat(Math.max(1, Math.round((span * 2) / grid))))
  }
  return pixels(rows, { W: 'trail-water' }, grid)
}

export function trailTerrain(landscape: Landscape, seed: number) {
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
  const reeds = [-0.75, -0.25, 0.55, 0.85]
    .map((fraction, index) => {
      const x = Math.round(pond.x + pond.rx * fraction)
      const y = Math.round(pond.y + pond.ry * Math.sqrt(1 - fraction ** 2) + 4)
      return `<g transform="translate(${x} ${y})">${pixels(['..L...', '.LL.C.', '.LL.C.', 'L.L.C.', 'L.L.D.', '..L.D.'], { L: 'trail-meadow-reed', C: 'trail-reed-head', D: 'trail-tree-trunk' }, index % 2 ? 1 : 2)}</g>`
    })
    .join('')
  const grid = width < 700 ? 3 : 4
  const bounds = {
    alpine: landscape.alpine,
    treeline: landscape.treeline,
    meadowTop: landscape.meadowTop,
  }
  return `<g class="trail-terrain" data-trail-terrain='${JSON.stringify(bounds)}'>${terraces}${grass}</g>
    <g class="trail-pond" data-trail-pond='${JSON.stringify(pond)}'>
      <g class="trail-pond-bank" transform="translate(${pond.x - pond.rx - 8} ${pond.y - pond.ry - 8})">${oval(pond, 8, grid)}</g>
      <g transform="translate(${pond.x - pond.rx} ${pond.y - pond.ry})">${oval(pond, 0, grid)}</g>
      ${ripples}${reeds}
    </g>`
}
