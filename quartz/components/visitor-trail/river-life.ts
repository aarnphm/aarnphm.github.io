import {
  creekSection,
  pondDistance,
  trailGroundScale,
  type Landscape,
  type Point,
} from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'
import { riverArt } from './river-art'

type FishingSite = { fish: Point; bank: Point; scale: number; delay: number; duration: number }

export function riverFishingSites(landscape: Landscape, walks: readonly Point[], seed: number) {
  const random = trailRandom(seed ^ 0x71b48c29)
  const sites: FishingSite[] = []
  const line = landscape.creek[0] ?? []
  const offset = Math.floor(random() * 7)
  for (let index = 8 + offset; index < line.length - 3 && sites.length < 2; index++) {
    const fish = line[index]
    const scale = trailGroundScale(fish.y, landscape)
    const bank = { x: fish.x - 17 * scale, y: fish.y - 8 * scale }
    if (
      fish.x < 48 ||
      fish.x > landscape.width - 64 ||
      fish.y < landscape.meadowTop + 72 ||
      creekSection(fish, landscape).waterHalf < 2.1 ||
      pondDistance(fish, landscape.pond) < 1.7 ||
      pondDistance(bank, landscape.pond) < 1.7 ||
      Math.hypot(bank.x - landscape.camp.x, bank.y - landscape.camp.y) < 130 ||
      landscape.groves.some(
        grove =>
          Math.hypot((bank.x - grove.x) / (grove.rx + 24), (bank.y - grove.y) / (grove.ry + 24)) <
          0.6,
      ) ||
      sites.some(site => Math.hypot(site.fish.x - fish.x, site.fish.y - fish.y) < 180) ||
      walks.some(point => Math.hypot(point.x - bank.x, point.y - bank.y) < 52)
    )
      continue
    sites.push({ fish, bank, scale, delay: random() * -18, duration: 18 + random() * 6 })
  }
  return sites
}

export function riverClearing(site: FishingSite) {
  return [
    site.bank,
    ...[20, 40, 60].flatMap(y =>
      [-16, 0, 16].map(x => ({ x: site.bank.x + x, y: site.bank.y + y })),
    ),
  ]
}

const fishTones = {
  S: 'trail-river-fish-silver',
  F: 'trail-river-fish-flank',
  D: 'trail-river-fish-fin',
  E: 'trail-animal-eye',
}
const bearTones = {
  C: 'trail-bear-coat',
  D: 'trail-bear-shade',
  E: 'trail-animal-eye',
  N: 'trail-bear-muzzle',
  B: 'trail-bear-paw',
}

function timing(site: FishingSite) {
  return `--river-cycle:${site.duration.toFixed(2)}s;--river-delay:${site.delay.toFixed(2)}s`
}

export function riverFish(site: FishingSite) {
  return `<g class="trail-river-fish" data-ground-y="${site.fish.y.toFixed(1)}" transform="translate(${site.fish.x.toFixed(1)} ${site.fish.y.toFixed(1)}) scale(${site.scale.toFixed(3)})" style="${timing(site)}"><g class="trail-river-jump"><g transform="translate(-4.5 -2.5)">${pixels(riverArt.fish, fishTones, 1)}</g></g><path class="trail-river-splash trail-ravine-foam" d="M-4 0h2v1h-2ZM1 1h3v1H1ZM5 -1h1v1H5Z"/></g>`
}

// One of the configured bears waits on a clear bank. Its collision body and forest exclusion
// share this anchor; the paw reaches the water without moving that body into a hiking route.
export function riverBear(site: FishingSite) {
  const radius = 16 * site.scale
  return `<g class="trail-bear trail-river-bear" data-trail-animal="bear" data-trail-collision="animal" data-collision-id="animal-bear-0" data-collision-radius="${radius.toFixed(2)}" data-ground-x="${site.bank.x.toFixed(1)}" data-ground-y="${site.bank.y.toFixed(1)}" data-phase="resting" transform="translate(${site.bank.x.toFixed(1)} ${site.bank.y.toFixed(1)})" style="${timing(site)}"><g transform="scale(${site.scale.toFixed(3)}) translate(-15 -16.5)">${pixels(riverArt.bear, bearTones, 1.5)}<g class="trail-river-paw">${pixels(riverArt.paw, bearTones, 1.5)}</g></g></g>`
}
