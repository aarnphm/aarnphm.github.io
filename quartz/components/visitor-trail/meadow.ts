import {
  creekDistance,
  grassGrows,
  pondDistance,
  trailGroundScale,
  type Landscape,
} from './landscape'
import { trailRandom } from './random'
import { routePoints } from './ravine'

type Runs = Map<string, string[]>

function run(runs: Runs, tone: string, x: number, y: number, width: number, height = 2) {
  const list = runs.get(tone) ?? []
  list.push(`M${x} ${y}h${width}v${height}h${-width}Z`)
  runs.set(tone, list)
}

const paths = (runs: Runs) =>
  [...runs.entries()]
    .map(([tone, list]) => `<path class="trail-meadow-${tone}" d="${list.join('')}"/>`)
    .join('')

// The open meadow fills with drifts of tall grass and wildflowers, thick in some places and sparse
// in others. Each drift is a cluster of blades whose tips lean over, a few pixels at a time, as a
// gust front reaches it; flowers and seed heads follow the season.
export function trailMeadow(landscape: Landscape, seed: number, routes: readonly string[]) {
  const random = trailRandom(seed ^ 0x6a09e667)
  const { width, height, meadowTop, camp, pond } = landscape
  const trail = routes.flatMap(route => routePoints(route).filter((_, index) => index % 3 === 0))
  // Low-frequency noise sets where drifts gather, so the meadow has open lawns and thick swathes.
  const phases = [random(), random(), random()].map(value => value * Math.PI * 2)
  const thickness = (x: number, y: number) =>
    0.5 +
    0.3 * Math.sin(x / 170 + phases[0]) * Math.cos(y / 90 + phases[1]) +
    0.2 * Math.sin((x + y) / 230 + phases[2])
  const drifts: string[] = []
  const step = width < 700 ? 44 : 36
  for (let gy = meadowTop + 36; gy < height - 18; gy += step * 0.6)
    for (let gx = 8; gx < width - 8; gx += step) {
      const x = Math.round((gx + (random() - 0.5) * step) / 2) * 2
      const y = Math.round((gy + (random() - 0.5) * step * 0.5) / 2) * 2
      const thick = thickness(x, y)
      // Only the thickest swathes carry drifts; the rest of the meadow stays open lawn.
      if (random() > (thick - 0.55) * 1.6) continue
      if (!grassGrows(y, landscape, random())) continue
      const point = { x, y }
      if (
        pondDistance(point, pond) < 1.9 ||
        creekDistance(point, landscape) < 22 ||
        Math.hypot(x - camp.x, (y - camp.y) * 1.6) < 150 ||
        trail.some(stop => Math.abs(stop.x - x) < 14 && Math.abs(stop.y - y) < 10)
      )
        continue
      const depth = trailGroundScale(y, landscape)
      const size = depth > 0.85 ? 2 : 1
      const runs: Runs = new Map()
      const tips: Runs = new Map()
      const blades = Math.round((3 + thick * 6) * (0.6 + depth * 0.4))
      const spread = Math.round((4 + thick * 8) * (0.6 + depth * 0.4))
      for (let blade = 0; blade < blades; blade++) {
        const bx = Math.round((random() - 0.5) * spread * 2) * size
        const by = Math.round((random() - 0.5) * spread * 0.35) * size
        const tall = Math.max(2, Math.round((2 + random() * 3) * (0.6 + depth * 0.4))) * size
        const tone = random() < 0.3 ? 'blade-dark' : 'blade'
        // The lower half stands still; the tip leans with the gust.
        const stem = Math.ceil(tall / 2 / size) * size
        run(runs, tone, bx, by - stem, size, stem)
        run(tips, tone, bx, by - tall, size, tall - stem)
        const bloom = random()
        if (bloom < 0.03 + thick * 0.05) {
          const kind = bloom < 0.035 ? 'bloom-a' : bloom < 0.06 ? 'bloom-b' : 'bloom-c'
          run(tips, kind, bx - size / 2, by - tall - size, size * 2, size * 2)
        } else if (bloom > 0.9) run(tips, 'seed', bx, by - tall - size, size, size * 2)
      }
      // A broken shadow where the drift meets the ground.
      for (let dx = -spread; dx < spread; dx += 2)
        if (random() < 0.35) run(runs, 'ground', dx * size, size, 2 * size, size)
      drifts.push(
        `<g class="trail-meadow-drift" transform="translate(${x} ${y})" style="--mx:${x};--my:${y}">${paths(runs)}<g class="trail-meadow-tips">${paths(tips)}</g></g>`,
      )
    }
  return `<g class="trail-meadow" aria-hidden="true">${drifts.join('')}</g>`
}
