import { catPixels, dogPixels, goosePixels, type TrailAnimalKind } from './animals'
import { cowPixels } from './cows'
import { onDryLand, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

export type RouteSites = { id: string; samples: Point[] }

export function trailPasture(landscape: Landscape, routes: RouteSites[], seed: number) {
  const { width, meadowTop, camp, pond } = landscape
  const random = trailRandom(seed ^ 0x243f6a88)
  const narrow = width < 700
  const walks = routes.flatMap(route =>
    route.samples.filter(
      (point, index) =>
        index % 8 === 0 &&
        point.y >= meadowTop + 64 &&
        Math.hypot(point.x - camp.x, point.y - camp.y) > 110,
    ),
  )
  const occupied: Point[] = []
  const site = () => {
    let point: Point = { x: width * 0.5, y: meadowTop + 64 }
    for (let attempt = 0; attempt < 100; attempt++) {
      const anchor = walks[Math.floor(random() * walks.length)] ?? point
      const side = random() < 0.5 ? -1 : 1
      point = onDryLand(
        { x: anchor.x + side * (24 + random() * 32), y: anchor.y + (random() - 0.5) * 32 },
        landscape,
        1.8,
      )
      if (
        occupied.every(
          other => Math.hypot(point.x - other.x, point.y - other.y) > (narrow ? 32 : 44),
        ) &&
        Math.hypot(point.x - camp.x, point.y - camp.y) > 94
      )
        break
    }
    occupied.push(point)
    return point
  }
  const animal = (
    kind: TrailAnimalKind,
    index: number,
    point: Point,
    size: number,
    flip: number,
    sprite: string,
    anchor: number,
    tone = '',
  ) => {
    const x = Math.round(point.x)
    const y = Math.round(point.y)
    return `<g class="trail-${kind}${tone}" data-trail-wander="${kind}-${index}" data-trail-animal="${kind}" data-animal-home-x="${x}" data-animal-home-y="${y}" data-animal-size="${size}" data-phase="waiting" data-motion="paused" data-stride="0" transform="translate(${x} ${y})" style="--trail-animal-step:${size}px;--trail-cow-delay:${index * -2700}ms;--trail-animal-delay:${index * -1700}ms"><g class="trail-animal-facing" transform="scale(${flip} 1)"><g transform="translate(${-size * anchor} ${-size * 8})">${sprite}</g></g></g>`
  }
  const animals: { y: number; markup: string }[] = []
  for (let index = 0; index < (narrow ? 6 : 9); index++) {
    const point = site()
    const size = index % 4 === 3 ? 1 : 2
    const flip = random() < 0.5 ? -1 : 1
    animals.push({
      y: point.y,
      markup: animal(
        'cow',
        index,
        point,
        size,
        flip,
        cowPixels(size),
        9,
        index % 3 === 1 ? ' trail-cow-russet' : '',
      ),
    })
  }
  for (const kind of ['dog', 'cat'] satisfies TrailAnimalKind[]) {
    for (let index = 0; index < (narrow ? 2 : 3); index++) {
      const point = site()
      const size = 2
      const flip = random() < 0.5 ? -1 : 1
      animals.push({
        y: point.y,
        markup: animal(
          kind,
          index,
          point,
          size,
          flip,
          kind === 'dog' ? dogPixels(size) : catPixels(size),
          5,
          ` trail-animal-tone-${index}`,
        ),
      })
    }
  }
  const geese = Array.from({ length: narrow ? 4 : 5 }, (_, index) => {
    const x = pond.x + pond.rx * (-0.62 + index * (narrow ? 0.36 : 0.29))
    const y = pond.y + (index % 2 ? -0.24 : 0.28) * pond.ry
    const size = index === 4 ? 1 : narrow ? 1 : 2
    const flip = index % 2 ? -1 : 1
    return animal('goose', index, { x, y }, size, flip, goosePixels(size), 5)
  }).join('')
  return {
    markup: `<g class="trail-pasture">${animals
      .sort((a, b) => a.y - b.y)
      .map(animal => animal.markup)
      .join('')}${geese}</g>`,
    sites: occupied,
  }
}
