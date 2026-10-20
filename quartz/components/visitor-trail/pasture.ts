import { bearPixels, catPixels, dogPixels, goosePixels, type TrailAnimalKind } from './animals'
import { cowPixels } from './cows'
import { onDryLand, type Landscape, type Point } from './landscape'
import { trailPondLife } from './pond-life'
import { trailRandom } from './random'
import { riverBear, riverClearing, riverFish, riverFishingSites } from './river-life'
import { sceneSettings, type AnimalSettings } from './scene-settings'
import { trailBankWildlife } from './wetlands'

export type RouteSites = { id: string; samples: Point[] }
export function trailPasture(
  landscape: Landscape,
  routes: RouteSites[],
  seed: number,
  settings: AnimalSettings = sceneSettings.animals,
) {
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
  const fishing = riverFishingSites(
    landscape,
    routes.flatMap(route => route.samples),
    seed,
  )
  const fishingBear = settings.bear > 0 ? fishing[0] : undefined
  if (fishingBear) occupied.push(...riverClearing(fishingBear))
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
  fishing.forEach((site, index) => animals.push({ y: site.fish.y, markup: riverFish(site, index) }))
  if (fishingBear) animals.push({ y: fishingBear.bank.y, markup: riverBear(fishingBear) })
  for (let index = 0; index < settings.cow; index++) {
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
    for (let index = 0; index < settings[kind]; index++) {
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
  const geese = Array.from({ length: settings.goose }, (_, index) => {
    const angle = index * 2.399963
    const radius = settings.goose === 1 ? 0 : 0.72 * Math.sqrt((index + 0.5) / settings.goose)
    const x = pond.x + Math.cos(angle) * pond.rx * radius
    const y = pond.y + Math.sin(angle) * pond.ry * radius
    const size = narrow || settings.goose > 4 || index % 5 === 4 ? 1 : 2
    const flip = index % 2 ? -1 : 1
    return animal('goose', index, { x, y }, size, flip, goosePixels(size), 5)
  }).join('')
  for (let index = fishingBear ? 1 : 0; index < settings.bear; index++) {
    const point =
      index % 2
        ? {
            x: Math.max(30, Math.min(width - 30, width * (0.25 + random() * 0.5))),
            y: landscape.alpine.base - 18 + random() * 8,
          }
        : site()
    if (index % 2) occupied.push(point)
    animals.push({
      y: point.y,
      markup: animal('bear', index, point, 2, random() < 0.5 ? -1 : 1, bearPixels(2), 9),
    })
  }
  // The heron's shore spot is kept clear before the frogs and the otter take theirs.
  const life = trailPondLife(landscape, seed)
  occupied.push(...life.sites)
  const bank = trailBankWildlife(
    landscape,
    routes.flatMap(route => route.samples),
    seed,
    occupied,
  )
  return {
    markup: `<g class="trail-pasture">${animals
      .sort((a, b) => a.y - b.y)
      .map(animal => animal.markup)
      .join('')}${geese}${life.shore}${bank.markup}</g>`,
    sites: [...occupied, ...bank.sites],
  }
}
