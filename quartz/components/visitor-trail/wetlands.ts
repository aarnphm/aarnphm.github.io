import {
  creekDistance,
  creekSection,
  pondDistance,
  trailGroundScale,
  type Landscape,
  type Point,
} from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'
import { contourFaces, faceAt, routePoints } from './ravine'
import { wetlandArt } from './wetland-art'

const tones = {
  G: 'trail-wetland-frog-back',
  D: 'trail-wetland-frog-leg',
  L: 'trail-wetland-belly',
  E: 'trail-animal-eye',
  C: 'trail-wetland-otter-coat',
  B: 'trail-wetland-otter-tail',
}

function habitat(landscape: Landscape, walks: readonly Point[], seed: number) {
  const random = trailRandom(seed ^ 0x3c8da671)
  const faces = contourFaces(landscape)
  const line = landscape.creek[0] ?? []
  const candidate = (shore: boolean, margin: number): Point => {
    if (shore || line.length < 3) {
      const angle = random() * Math.PI * 2
      return {
        x: landscape.pond.x + Math.cos(angle) * (landscape.pond.rx + margin),
        y: landscape.pond.y + Math.sin(angle) * (landscape.pond.ry + margin * 0.6),
      }
    }
    const index = 1 + Math.floor(random() * (line.length - 2))
    const a = line[index - 1],
      b = line[index]
    const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
    const side = random() < 0.5 ? 1 : -1
    const section = creekSection(b, landscape)
    const offset =
      side * (section.waterHalf + (side > 0 ? section.nearBank : section.farBank) + margin)
    return {
      x: b.x - ((b.y - a.y) / length) * offset,
      y: b.y + ((b.x - a.x) / length) * offset * 0.6,
    }
  }
  const clear = (point: Point, radius: number, occupied: readonly Point[]) =>
    point.x > 24 &&
    point.x < landscape.width - 28 &&
    point.y > landscape.meadowTop + 32 &&
    point.y < landscape.height - 48 &&
    Math.hypot(point.x - landscape.camp.x, point.y - landscape.camp.y) > 120 &&
    pondDistance(point, landscape.pond) > 1.08 &&
    creekDistance(point, landscape) > 10 &&
    !faceAt(faces, point) &&
    walks.every(walk => Math.hypot(point.x - walk.x, point.y - walk.y) > radius) &&
    occupied.every(other => Math.hypot(point.x - other.x, point.y - other.y) > radius)
  return { random, candidate, clear }
}

export function trailPuddles(landscape: Landscape, seed: number, routes: readonly string[]) {
  const { random, candidate, clear } = habitat(landscape, routes.flatMap(routePoints), seed)
  const sites: Point[] = []
  const permanent: string[] = [],
    rain: string[] = []
  const limit = landscape.width < 700 ? 3 : 6
  for (let attempt = 0; attempt < 120 && sites.length < limit; attempt++) {
    const point = candidate(attempt % 3 === 0, 18 + random() * 18)
    if (
      !clear(point, 22, sites) ||
      pondDistance(point, landscape.pond) < 1.4 ||
      creekDistance(point, landscape) < 18
    )
      continue
    const scale = trailGroundScale(point.y, landscape)
    const half = 5 + Math.round(random() * 6),
      tall = 2 + Math.floor(random() * 2)
    const edge = random() < 0.5 ? 1 : -1
    const rows = [
      '.'.repeat(2) + 'W'.repeat(half * 2 - 4),
      'W'.repeat(half * 2),
      '.'.repeat(edge > 0 ? 3 : 1) + 'W'.repeat(half * 2 - 4),
    ]
    const shape = pixels(rows, { W: 'trail-wetland-water' }, 1)
    const markup = `<g class="trail-wetland-puddle" data-ground-x="${point.x.toFixed(1)}" data-ground-y="${point.y.toFixed(1)}" transform="translate(${point.x.toFixed(1)} ${point.y.toFixed(1)}) scale(${scale.toFixed(3)} ${((tall / 3) * scale).toFixed(3)}) translate(-${half} -1)"><path class="trail-wetland-mud" d="M2 -1h${half * 2 - 4}v1H2ZM0 3h${half * 2 - 1}v1H0Z"/>${shape}<path class="trail-wetland-glint" d="M3 1h${2 + (sites.length % 3)}v1H3Z"/></g>`
    const pools = sites.length < 2 ? permanent : rain
    pools.push(markup)
    sites.push(point)
  }
  return `<g class="trail-wetland-pools" aria-hidden="true">${permanent.join('')}<g class="trail-wetland-rain-pools">${rain.join('')}</g></g>`
}

export function trailBankWildlife(
  landscape: Landscape,
  walks: readonly Point[],
  seed: number,
  occupied: readonly Point[],
) {
  const { random, candidate, clear } = habitat(landscape, walks, seed ^ 0x71fc354b)
  const sites: Point[] = [],
    sprites: string[] = []
  const frogs = landscape.width < 700 ? 1 : 3
  for (const kind of ['frog', 'otter'] satisfies (keyof typeof wetlandArt)[]) {
    const count = kind === 'frog' ? frogs : 1
    for (let index = 0; index < count; index++) {
      let point: Point | undefined
      for (let attempt = 0; attempt < 80; attempt++) {
        const next = candidate(
          kind === 'frog' && (index < 2 || attempt % 3 === 0),
          kind === 'frog' ? 9 : 14,
        )
        if (
          clear(next, kind === 'frog' ? 36 : 48, [...occupied, ...sites]) &&
          landscape.groves.every(
            grove => Math.hypot((next.x - grove.x) / grove.rx, (next.y - grove.y) / grove.ry) > 0.6,
          )
        ) {
          point = next
          break
        }
      }
      if (!point) continue
      const scale = trailGroundScale(point.y, landscape)
      const rows = wetlandArt[kind],
        duration = kind === 'frog' ? 26 + random() * 8 : 38 + random() * 10
      sprites.push(
        `<g class="trail-wetland-animal trail-wetland-${kind}" data-wetland-animal="${kind}" data-trail-collision="animal" data-collision-id="animal-wetland-${kind}-${index}" data-collision-radius="${((kind === 'frog' ? 9 : 13) * scale).toFixed(2)}" data-ground-x="${point.x.toFixed(1)}" data-ground-y="${point.y.toFixed(1)}" data-phase="resting" transform="translate(${point.x.toFixed(1)} ${point.y.toFixed(1)})" style="--wetland-cycle:${duration.toFixed(2)}s;--wetland-delay:${(-random() * duration).toFixed(2)}s"><g transform="scale(${scale.toFixed(3)})"><g class="trail-wetland-${kind}-motion"><g transform="translate(-${rows[0].length / 2} -${rows.length})">${pixels(rows, tones, 1)}</g></g></g></g>`,
      )
      sites.push(
        point,
        { x: point.x, y: point.y + 20 },
        { x: point.x - 10, y: point.y + 36 },
        { x: point.x + 10, y: point.y + 36 },
      )
    }
  }
  return { markup: sprites.join(''), sites }
}
