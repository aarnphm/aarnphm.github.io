import type { TrailSeason, TrailWeather } from '../../../util/visitor-trail-weather'
import type { Almanac } from '../../scripts/404-almanac'
import { bayerThreshold } from '../../../util/bayer'
import { holidaySprite } from '../holiday-sprites'
import { creekDistance, groundOffset, pondDistance, type Landscape } from '../landscape'
import { trailRandom } from '../random'
import { decoration } from './conditions'

export function weatherMarks(
  landscape: Landscape,
  seed: number,
  weather: TrailWeather | null,
  festiveSnow: boolean,
  season: TrailSeason | null,
  al?: Almanac,
  seasonOverride?: TrailSeason | null,
) {
  const random = trailRandom(seed ^ 0x34c4f69b)
  const { width, height, bottom, groves } = landscape
  const { rain, snow, fog, snowCover, gust, dx, dy } = decoration(weather, al, seasonOverride)
  const marks: string[] = []
  if (fog) marks.push(fogVeil(landscape))
  if (snowCover || festiveSnow) {
    const patches: string[] = []
    const count = width < 700 ? 28 : 56
    for (let index = 0; index < count; index++) {
      const x = Math.round(20 + random() * Math.max(1, width - 40))
      const base =
        landscape.meadowTop + 24 + random() * Math.max(1, height - landscape.meadowTop - 54)
      const y = Math.round(base + groundOffset(x, landscape, base))
      const span = 10 + Math.floor(random() * 12) * 2
      const samples = [x - 4, x + span / 2, x + span + 4].flatMap(sx => [
        { x: sx, y },
        { x: sx, y: y + 4 },
      ])
      if (
        y >= height - 12 ||
        samples.some(
          point =>
            pondDistance(point, landscape.pond) < 1.1 || creekDistance(point, landscape) < 12,
        )
      )
        continue
      patches.push(`M${x} ${y}h${span}v2h4v2h-${span + 8}v-2h4Z`)
    }
    marks.push(`<path class="trail-weather-snow-ground" d="${patches.join('')}"/>`)
  }
  marks.push(`<g class="trail-weather-night">
    ${Array.from({ length: 6 }, () => {
      const x = Math.round(20 + random() * (width - 40))
      const y = Math.round(bottom + 22 + random() * 70)
      return `<path class="trail-weather-star" d="M${x} ${y}h2v2h-2Z"/>`
    }).join('')}
  </g>`)
  marks.push('<g class="trail-weather-daylight">')
  for (let index = 0; groves.length && index < 20; index++) {
    const grove = groves[index % groves.length]
    const x = Math.round((grove.x + (random() - 0.5) * grove.rx * 2.5) / 4) * 4
    const y = Math.round((grove.y + (random() - 0.5) * grove.ry * 2) / 4) * 4
    const span = 16 + Math.floor(random() * 8) * 4
    marks.push(
      `<path class="trail-weather-dapple" style="--trail-weather-delay: ${(-random() * 10).toFixed(2)}s" d="M${x} ${y}h${span}v4h-8v4h-${span}v-4h8Z"/>`,
    )
  }
  marks.push('</g>')
  if (gust > 0.025) {
    // Streamlines lie on the ground plane, so the north–south component is foreshortened. Each
    // one is mirrored rather than turned past vertical, keeping its curl on the upper side.
    const ground = Math.atan2(dy * 0.5, Math.abs(dx))
    const mirror = dx < 0 ? -1 : 1
    const period = 9 - gust * 3
    const front = 55 + gust * 90
    const count = Math.round((width < 700 ? 5 : 9) * (0.6 + gust * 0.8))
    for (let index = 0; index < count; index++) {
      const x = Math.round(24 + random() * Math.max(1, width - 70))
      const y = Math.round(bottom + 55 + random() * Math.max(1, height - bottom - 100))
      const length = Math.round(34 + gust * 60 + random() * 24)
      const wave = 1.5 + gust * 3 + random()
      const bends = 2 + Math.floor(random() * 2)
      let d = 'M0 0'
      for (let bend = 1; bend <= bends; bend++) {
        const end = (length * bend) / bends
        const crest = end - length / bends / 2
        d += `Q${crest.toFixed(1)} ${((bend % 2 ? -1 : 1) * wave).toFixed(1)} ${end.toFixed(1)} 0`
      }
      // A gust strong enough to eddy rolls over at the end of its line.
      if (gust > 0.4 && random() < 0.6) {
        const r = (3 + gust * 4).toFixed(1)
        d += `a${r} ${r} 0 1 0 0 ${-2 * Number(r)}a${r} ${r} 0 0 0 ${-Number(r)} ${r}`
      }
      const travel = (x * dx + y * dy) / front + random() * 0.6
      const phase = ((travel % period) + period) % period
      marks.push(
        `<g transform="translate(${x} ${y}) scale(${mirror} 1) rotate(${((ground * 180) / Math.PI).toFixed(1)})"><path class="trail-weather-gust" pathLength="100" style="--trail-weather-delay: ${(phase - period).toFixed(2)}s" d="${d}"/></g>`,
      )
    }
    // On the ground a gust is grass laid flat in a crescent that runs downwind, and cat's paws
    // that darken the pond as the front crosses it. Both keep to whole pixels.
    const delay = (x: number, y: number) => {
      const travel = (x * dx + y * dy) / front
      return (((travel % period) + period) % period) - period
    }
    const along = { x: Math.cos(ground) * mirror, y: Math.sin(ground) }
    const across = { x: -along.y, y: along.x }
    const lean = mirror > 0 ? 'h2v-2h2v-2h-2v2h-2Z' : 'h-2v-2h-2v-2h2v2h2Z'
    const { pond, meadowTop } = landscape
    const patches = Math.round((width < 700 ? 6 : 12) * (0.5 + gust))
    for (let placed = 0, tries = 0; placed < patches && tries < patches * 6; tries++) {
      const cx = 20 + random() * Math.max(1, width - 40)
      const cy = meadowTop + 16 + random() * Math.max(1, height - meadowTop - 40)
      if (pondDistance({ x: cx, y: cy }, pond) < 1.4) continue
      placed++
      const blades = 7 + Math.floor(gust * 8 + random() * 4)
      let d = ''
      for (let blade = 0; blade < blades; blade++) {
        const side = (blade / (blades - 1) - 0.5) * (14 + gust * 18)
        const bow = 4 - (side * side) / 40
        const x = Math.round(cx + along.x * bow + across.x * side + (random() - 0.5) * 2)
        const y = Math.round(cy + along.y * bow + across.y * side * 0.5)
        d += `M${x} ${y}${lean}`
      }
      const run = 6 + gust * 10
      marks.push(
        `<path class="trail-weather-sweep" style="--trail-weather-delay: ${delay(cx, cy).toFixed(2)}s; --trail-sweep-x: ${(along.x * run).toFixed(1)}px; --trail-sweep-y: ${(along.y * run).toFixed(1)}px" d="${d}"/>`,
      )
    }
    const paws = 3 + Math.round(gust * 4)
    for (let paw = 0; paw < paws; paw++) {
      const angle = random() * Math.PI * 2
      const reach = Math.sqrt(random()) * 0.7
      const cx = pond.x + Math.cos(angle) * pond.rx * reach
      const cy = pond.y + Math.sin(angle) * pond.ry * reach
      let d = ''
      for (let row = 0; row < 3; row++) {
        const length = 3 + Math.floor(random() * 5)
        const x = Math.round(cx + (random() - 0.5) * 8)
        d += `M${x} ${Math.round(cy) + row * 2}h${length}v1h${-length}Z`
      }
      marks.push(
        `<path class="trail-weather-paw" style="--trail-weather-delay: ${delay(cx, cy).toFixed(2)}s" d="${d}"/>`,
      )
    }
  }
  if (season === 'autumn' && gust > 0.06 && !snow && !festiveSnow) {
    const count = width < 700 ? 6 : 12
    for (let index = 0; index < count; index++) {
      const grove = groves[index % groves.length]
      const x = Math.round(grove.x + (random() - 0.5) * grove.rx * 1.7)
      const y = Math.round(grove.y - 20 - random() * 30)
      marks.push(
        `<g transform="translate(${x} ${y})"><path class="trail-weather-leaf" style="--trail-weather-delay: ${(-random() * 9).toFixed(2)}s" d="M0 0h3v2H2v1H0Z"/></g>`,
      )
    }
  }
  // Forecast precipitation spans the homepage. Keep holiday-only snow within the trail.
  if (festiveSnow && !snow) {
    const count = rain ? (width < 700 ? 12 : 18) : width < 700 ? 28 : 48
    for (let index = 0; index < count; index++) {
      const x = Math.round(12 + random() * Math.max(1, width - 28))
      const top = Math.max(20, landscape.alpine.top - 40)
      const y = Math.round(top + random() * Math.max(1, height - top - 52))
      const art = index % 4 === 0 ? holidaySprite('snowflake') : '<path d="M0 0h2v2H0Z"/>'
      marks.push(
        `<g transform="translate(${x} ${y})"><g class="trail-weather-snow" style="--trail-weather-delay: ${(-random() * 6).toFixed(2)}s">${art}</g></g>`,
      )
    }
  }
  return marks.join('')
}

// Fog settles in the valley and over the near ranges: a paper veil through the Bayer screen,
// thickest at the foot of the mountains and thinning upward, so the crests show through it.
function fogVeil(landscape: Landscape) {
  const { width, alpine } = landscape
  const cell = width < 700 ? 2 : 4
  const top = alpine.top - 24
  const depth = alpine.base - top + alpine.height * 0.35
  // Below its densest row the veil thins out over the meadow instead of ending on one level edge.
  const tail = alpine.height * 0.4
  const runs: string[] = []
  for (let row = 0; row * cell < depth + tail; row++) {
    const y = top + row * cell
    const level =
      row * cell < depth
        ? 0.2 + 0.5 * ((row * cell) / depth) ** 0.8
        : 0.7 * (1 - (row * cell - depth) / tail) ** 1.5
    let start = -1
    for (let column = 0; column * cell <= width; column++) {
      const on = level > bayerThreshold(column, row)
      if (on && start < 0) start = column * cell
      if (!on && start >= 0) {
        runs.push(`M${start} ${y}h${column * cell - start}v${cell}h${start - column * cell}Z`)
        start = -1
      }
    }
    if (start >= 0) runs.push(`M${start} ${y}H${width + cell}v${cell}H${start}Z`)
  }
  return `<path class="trail-weather-fog" d="${runs.join('')}"/>`
}
