import { bayerThreshold } from '../../util/bayer'
import { lunarPhase } from '../../util/lunar-phase'
import { sceneNow, subscribeSceneClock } from '../../util/scene-clock'
import { readTrailSeason, subscribeTrailSeason } from '../../util/trail-season'
import {
  sampleVisitorDaylight,
  subscribeVisitorDaylight,
  type VisitorDaylight,
} from '../../util/visitor-daylight'
import { subscribeVisitorMoon, type VisitorMoon } from '../../util/visitor-lunar'
import { stormDelay, visitorPrecipitation, wetFall } from '../../util/visitor-precipitation'
import {
  subscribeVisitorTrailWeather,
  trailSite,
  trailSky,
  type TrailSeason,
  type TrailWeather,
} from '../../util/visitor-trail-weather'
import { season, seasonBucket } from '../../util/visitor-world'
import { campSleeps, fireOut, herd, leafRate, mossClimate, night } from '../../util/world-schedule'
import { almanac, guessSite, type Almanac, type Body } from '../scripts/404-almanac'
import { paintLunarDisc, stopLunarTransition } from '../scripts/lunar-disc'
import { holidaySprite } from './holiday-sprites'
import { creekDistance, groundOffset, pondDistance, type Landscape } from './landscape'
import { buildMountainRanges, mountainCrestAt, type MountainRange } from './mountain-profile'
import { trailRandom } from './random'
import { skyBand } from './sky-band'
import { trailWinter } from './winter'

const isSeason = (value: unknown): value is TrailSeason =>
  value === 'spring' || value === 'summer' || value === 'autumn' || value === 'winter'

function decoration(
  weather: TrailWeather | null,
  al?: Almanac,
  seasonOverride?: TrailSeason | null,
) {
  const precipitation = visitorPrecipitation(weather?.code)
  const rain = wetFall(precipitation.fall) ? precipitation.heavy : 0
  const snow = precipitation.fall === 'snow' ? 1 : 0
  const { fog, storm } = precipitation
  const snowCover = trailWinter(weather, false, al, seasonOverride).cover
  const sun =
    weather?.day && weather.code >= 0 && weather.code <= 2 && !rain && !snow
      ? Math.max(0, (0.85 - weather.cloud) / 0.85)
      : 0
  // A valley always has a breeze. Without a reading, or with a calm or variable one, gusts still
  // drift through from the west at a light strength.
  const angle = ((weather?.windFromDegrees ?? 270) * Math.PI) / 180
  const wind = weather ? Math.min(1, weather.windKmh / 55) : 0
  const gust = Math.max(
    0.22,
    weather ? Math.min(1, Math.max(weather.windKmh, weather.gustKmh ?? weather.windKmh) / 55) : 0,
  )
  // Meteorological directions describe where the wind comes from; SVG y grows southward.
  const dx = -Math.sin(angle)
  const dy = Math.cos(angle)
  const mode = !weather
    ? 'quiet'
    : rain
      ? 'rain'
      : snow
        ? 'snow'
        : weather?.day === false
          ? 'night'
          : sun
            ? 'sun'
            : 'cloud'
  return { rain, snow, fog, storm, snowCover, sun, wind, gust, dx, dy, mode }
}

function weatherMarks(
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

const sunRadius = 6
const moonRadius = 10

// Where a body hangs over the trail: across the width by its bearing from the equator-facing view
// (±120° reaches the edges), and up from the crest under it by its altitude. Below the horizon it
// sinks behind that crest, a radius for every 1.5°.
function hang(body: Body, radius: number, landscape: Landscape, ranges: readonly MountainRange[]) {
  const { width, bottom, alpine } = landscape
  const x = width / 2 + (body.bearing / 120) * width * 0.5
  const crest = mountainCrestAt(x, ranges) ?? alpine.base
  const ceiling = bottom + 14 + radius
  const y =
    body.alt >= 0
      ? crest - (crest - ceiling) * Math.sin((Math.min(90, body.alt) * Math.PI) / 180) ** 0.6
      : crest - (body.alt / 1.5) * radius
  const visible = y - radius < crest && x > -radius * 2 && x < width + radius * 2
  return { x: Math.round(x), y: Math.round(y), visible }
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

// A storm's bolt drops from the cloud deck to a crest behind the ranges in 2 px steps, and shows
// only at the flash (see trail-weather.scss), on the same clock as the page's flash.
function boltMarks(landscape: Landscape, ranges: readonly MountainRange[], seed: number) {
  const random = trailRandom(seed ^ 0x5b0c71a3)
  const { width, bottom, alpine } = landscape
  // The left half of the ranges, clear of the far gate on the right.
  let x = Math.round((width * (0.15 + random() * 0.3)) / 2) * 2
  const crest = mountainCrestAt(x, ranges) ?? alpine.base
  let d = ''
  for (let y = bottom + 10; y < crest; y += 6) {
    d += `M${x} ${y}h2v6h-2Z`
    x += (random() < 0.5 ? -2 : 2) * (random() < 0.3 ? 2 : 1)
  }
  return `<path class="trail-sky-bolt" d="${d}"/>`
}

// The 404's gate, seen from the trail: a stone ring on the flattest stretch of the right-hand
// ranges' skyline, its portal dark and its hands keeping the visitor's time. It is drawn in front
// of the ranges with its cells cut along the crest, so the ridge buries its foot the way the hill
// buries the gate on the 404; behind the ranges it floated, because the crest band is near-paper
// rock. Drawn in 2 px cells, redrawn with the sky each minute. The scene lays a link to the 404
// over the ring.
export type FarGateSite = { cx: number; cy: number; radius: number; cell: number }

export function farGateSite(landscape: Landscape, ranges: readonly MountainRange[]): FarGateSite {
  const { width, alpine } = landscape
  const cell = 2
  const radius = width < 700 ? 6 : 8
  // The flattest window three ring widths wide between 72% and 90% of the scene: the smallest
  // crest rise and fall, the higher stretch on a tie, so the ring stands on a shelf or a saddle
  // instead of a peak.
  const half = radius * cell * 3
  const step = 4
  let x = Math.round(width * 0.8)
  let crest = mountainCrestAt(x, ranges) ?? alpine.base
  let flattest = Infinity
  for (let centre = Math.round(width * 0.72); centre <= width * 0.9; centre += step) {
    let low = Infinity
    let high = -Infinity
    for (let probe = centre - half; probe <= centre + half; probe += step) {
      const y = mountainCrestAt(probe, ranges)
      if (y === undefined) {
        low = Infinity
        break
      }
      low = Math.min(low, y)
      high = Math.max(high, y)
    }
    if (low === Infinity) continue
    const spread = high - low
    const y = mountainCrestAt(centre, ranges) ?? alpine.base
    if (spread < flattest - 0.5 || (Math.abs(spread - flattest) <= 0.5 && y < crest))
      [x, crest, flattest] = [centre, y, spread]
  }
  const cx = Math.round(x / cell) * cell
  // The ranges' tops dissolve into dither, so the ring sinks four cells to stand in solid rock.
  const cy = Math.round((crest - (radius - 4) * cell) / cell) * cell
  return { cx, cy, radius, cell }
}

function farGate(site: FarGateSite, ranges: readonly MountainRange[], now: number) {
  const { cx, cy, radius, cell } = site
  const tones: Record<'stone' | 'joint' | 'portal' | 'star' | 'hand', string[]> = {
    stone: [],
    joint: [],
    portal: [],
    star: [],
    hand: [],
  }
  // A cell whose centre lies under the skyline is rock, so the ring's foot stays buried.
  const buried = (i: number, j: number) =>
    cy + (j + 0.5) * cell > (mountainCrestAt(cx + (i + 0.5) * cell, ranges) ?? Infinity)
  const plot = (tone: keyof typeof tones, i: number, j: number) => {
    if (buried(i, j)) return
    tones[tone].push(`M${cx + i * cell} ${cy + j * cell}h${cell}v${cell}h${-cell}Z`)
  }
  const random = trailRandom(0x404)
  for (let j = -radius; j < radius; j++)
    for (let i = -radius; i < radius; i++) {
      const distance = Math.hypot(i + 0.5, j + 0.5)
      if (distance > radius) continue
      if (distance > radius - 2.2) {
        // Twelve dressed stones with a dark joint between each.
        const turn = (Math.atan2(j + 0.5, i + 0.5) / (2 * Math.PI) + 1) * 12
        plot(turn % 1 < 0.14 ? 'joint' : 'stone', i, j)
      } else plot(random() < 0.06 ? 'star' : 'portal', i, j)
    }
  const time = new Date(now)
  const minute = time.getMinutes() + time.getSeconds() / 60
  const hour = (time.getHours() % 12) + minute / 60
  const hand = (turns: number, length: number) => {
    const angle = turns * 2 * Math.PI
    const seen = new Set<string>()
    for (let t = 0; t <= length; t += 0.5) {
      const i = Math.round(Math.sin(angle) * t - 0.5)
      const j = Math.round(-Math.cos(angle) * t - 0.5)
      if (seen.has(`${i},${j}`)) continue
      seen.add(`${i},${j}`)
      plot('hand', i, j)
    }
  }
  hand(minute / 60, radius - 3)
  hand(hour / 12, (radius - 3) * 0.6)
  const paths = Object.entries(tones)
    .map(([tone, runs]) => `<path class="trail-far-gate-${tone}" d="${runs.join('')}"/>`)
    .join('')
  return `<g class="trail-far-gate" data-far-gate="${time.getHours()}:${String(time.getMinutes()).padStart(2, '0')}">${paths}</g>`
}

// Twilight lights the sky behind the ranges, strongest low over the sun's bearing. A 4 × 4 Bayer
// screen turns the glow into whole cells, two tones, merged into runs per row.
function duskGlow(landscape: Landscape, sunX: number, strength: number) {
  if (strength < 0.04) return ''
  const { width, alpine } = landscape
  const cell = width < 700 ? 2 : 4
  const top = alpine.top - 48
  const depth = alpine.height * 0.75 + 48
  const span = width * 0.45
  const runs: Record<'inner' | 'outer', string[]> = { inner: [], outer: [] }
  for (let row = 0; row * cell < depth; row++) {
    const y = top + row * cell
    const low = (row * cell) / depth
    let tone: 'inner' | 'outer' | null = null
    let start = 0
    for (let column = 0; column * cell <= width; column++) {
      const x = column * cell
      const near = Math.max(0, 1 - Math.abs(x - sunX) / span) ** 1.5
      const level = strength * near * (0.25 + low * 0.75)
      const threshold = bayerThreshold(column, row)
      const next = level > threshold ? (level > 0.55 ? 'inner' : 'outer') : null
      if (next === tone) continue
      if (tone) runs[tone].push(`M${start} ${y}h${x - start}v${cell}h${start - x}Z`)
      tone = next
      start = x
    }
    if (tone) runs[tone].push(`M${start} ${y}H${width + cell}v${cell}H${start}Z`)
  }
  return `<path class="trail-sky-glow trail-sky-glow-outer" d="${runs.outer.join('')}"/><path class="trail-sky-glow trail-sky-glow-inner" d="${runs.inner.join('')}"/>`
}

function skyMarks(
  landscape: Landscape,
  ranges: readonly MountainRange[],
  weather: TrailWeather | null,
  now: number,
  seed: number,
) {
  const site =
    weather?.lat !== null && weather?.lat !== undefined && weather.lon !== null
      ? { lat: weather.lat, lon: weather.lon, place: null, weather: null }
      : guessSite()
  const sky = almanac(new Date(now), site)
  const { rain, snow, storm, wind, dx } = decoration(weather)
  const clear = rain || snow ? 0 : 1 - (weather?.cloud ?? 0.3)
  const sun = hang(sky.sun, sunRadius, landscape, ranges)
  const moon = hang(sky.moon, moonRadius, landscape, ranges)
  const marks = [
    skyBand(
      landscape,
      seed,
      { now, clear, cloud: weather?.cloud ?? 0.3, rain, snow, wind, dx, sunAltitude: sky.sun.alt },
      x => mountainCrestAt(x, ranges) ?? landscape.alpine.base,
    ),
    duskGlow(landscape, sun.x, sky.dusk * (0.35 + clear * 0.65)),
  ]
  if (storm) marks.push(boltMarks(landscape, ranges, seed))
  if (sun.visible && clear > 0.15) {
    const rays = sky.sun.alt > 4 ? 'M-1-12h2v3h-2ZM-1 9h2v3h-2ZM-12-1h3v2h-3ZM9-1h3v2H9Z' : ''
    marks.push(
      `<g class="trail-sky-sun" data-sun-altitude="${sky.sun.alt.toFixed(1)}" style="--trail-sun-low:${Math.max(0, Math.min(1, 1 - sky.sun.alt / 12)).toFixed(2)};--trail-sun-clear:${clear.toFixed(2)}" transform="translate(${sun.x} ${sun.y})"><path d="M-4-6h8v2h2v8h-2v2h-8V4h-2V-4h2Z${rays}"/></g>`,
    )
  }
  if (moon.visible)
    marks.push(
      `<g class="trail-sky-moon" data-moon-altitude="${sky.moon.alt.toFixed(1)}" transform="translate(${moon.x} ${moon.y})${sky.south ? ' scale(-1 1)' : ''}">
      <g class="trail-moon-motion">
      <path class="trail-moon-shadow" d="M-4-10h8v2h4v4h2v8H8v4H4v2h-8V8h-4V4h-2v-8h2v-4h4Z"/>
      <path class="trail-moon-lit" data-lunar-disc/>
      <path class="trail-moon-crater" d="M3-5h2v2H3ZM4 3h3v2H4Z"/>
      </g>
    </g>`,
    )
  return marks.join('')
}

export function mountTrailWeather(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-weather')
  const skyLayer = scene.querySelector<SVGGElement>('.trail-sky')
  const gateLayer = scene.querySelector<SVGGElement>('.trail-far-gate-layer')
  const motion = view?.matchMedia('(prefers-reduced-motion: reduce)')
  // The ring is drawn in the composited canvas, with a native link over its painted position.
  const gateLink = document.createElement('a')
  gateLink.className = 'trail-far-gate-link'
  gateLink.href = '/404'
  gateLink.dataset.routerIgnore = ''
  gateLink.hidden = true
  scene.append(gateLink)
  let landscape: Landscape | null = null
  let seed = 0
  let weather: TrailWeather | null = null
  let festiveSnow = scene.dataset.holiday === 'christmas'
  let seasonOverride = readTrailSeason(document.defaultView)
  // A chosen winter brings the holiday's snow with it.
  const snowy = () => festiveSnow || seasonOverride === 'winter'
  let moon: VisitorMoon = { ...lunarPhase(Date.now()), at: Date.now() }
  let paused = host.hidden
  let ranges: MountainRange[] = []
  let skyTimer = 0
  let skyAt = -Infinity
  let wetUntil = 0
  let daylight: VisitorDaylight = sampleVisitorDaylight(null)
  const updateDamp = () => {
    const damp = String(Date.now() < wetUntil)
    if (scene.dataset.groundDamp !== damp) scene.dataset.groundDamp = damp
  }

  const paintMoon = (animate: boolean) => {
    scene.dataset.moonPhase = String(moon.phase)
    const path = skyLayer?.querySelector<SVGPathElement>('[data-lunar-disc]')
    if (path)
      paintLunarDisc(
        path,
        moon.phase,
        10,
        2,
        animate && !paused && !document.hidden && !motion?.matches,
      )
  }

  const updateLighting = () => {
    const moonlit = document.documentElement.getAttribute('saved-theme') === 'dark'
    const automatic = document.documentElement.getAttribute('data-theme-mode') === 'daylight'
    const state = decoration(weather)
    const clear = (1 - (weather?.cloud ?? 0.3)) * (1 - state.rain * 0.85 - state.snow * 0.85)
    const sun = automatic
      ? daylight.sunlight * Math.max(0, Math.min(1, daylight.altitude / 6))
      : moonlit
        ? 0
        : clear
    const glow = automatic ? 1 - daylight.daylight : moonlit ? 1 : 0
    const lighting = moonlit ? 'moonlit' : 'day'
    if (scene.dataset.lighting !== lighting) scene.dataset.lighting = lighting
    const dayLayer = layer?.querySelector<SVGGElement>('.trail-weather-daylight')
    if (dayLayer) dayLayer.style.opacity = sun.toFixed(3)
    for (const fire of scene.querySelectorAll<SVGGElement>('.trail-firelight'))
      fire.style.opacity = `calc(${glow.toFixed(3)} * var(--trail-fire-strength, 1) * var(--trail-fire-night-dim, 1))`

    // The sun clock moves existing foot shadows, leaving crowns and their raster textures intact.
    const angle = ((automatic ? daylight.bearing : -25) * Math.PI) / 180
    const dx = -Math.sin(angle)
    const dy = Math.cos(angle) * 0.35
    const low = automatic ? Math.max(0, Math.min(1, 1 - daylight.altitude / 55)) : 0.15
    const scale = 0.75 + low * 1.85 * Math.abs(dx)
    const shadowSun = sun * (automatic ? Math.max(0, Math.min(1, daylight.altitude / 5)) : 1)
    // The generator knows each shadow's radius before the hidden forest enters layout.
    for (const shadow of scene.querySelectorAll<SVGPathElement>('.trail-tree-shadow')) {
      const radius = Number(shadow.dataset.shadowRadius)
      const reach = Math.min(36, radius * low * 1.65)
      const skew = dy * low * 0.4
      const shift = dx * reach
      shadow.style.transform = `matrix(${scale.toFixed(3)}, ${skew.toFixed(3)}, 0, 1, ${shift.toFixed(2)}, ${((skew * shift) / scale).toFixed(2)})`
      shadow.style.opacity = (shadowSun * (0.28 + low * 0.24)).toFixed(3)
    }
    // The canopy shade under the groves is cast from the same sun; the frame rebuilds its cells.
    scene.style.setProperty('--trail-sunlight-strength', shadowSun.toFixed(3))
    host.dispatchEvent(new CustomEvent('trail:sun', { detail: { dx, dy, low } }))
  }
  const updateMotion = () => {
    scene.dataset.weatherMotion =
      paused || host.hidden || document.hidden || motion?.matches ? 'paused' : 'running'
    if (scene.dataset.weatherMotion === 'paused') {
      const path = skyLayer?.querySelector<SVGPathElement>('[data-lunar-disc]')
      if (path) stopLunarTransition(path)
    }
  }
  const refreshForest = () => {
    const state = decoration(weather)
    const period = 9 - state.gust * 3
    scene.querySelectorAll<SVGGElement>('.trail-tree-motion').forEach(tree => {
      const root = tree.closest<SVGGElement>('.trail-tree, .trail-woodland-tree, .trail-understory')
      const random = trailRandom(
        seed ^
          Math.imul(Number(root?.dataset.groundX ?? 0), 16807) ^
          Math.imul(Number(root?.dataset.groundY ?? 0), 48271),
      )
      const alongWind =
        Number(root?.dataset.groundX ?? 0) * state.dx +
        Number(root?.dataset.groundY ?? 0) * state.dy
      const travel = alongWind / (55 + state.gust * 90) + random() * 0.8
      const phase = ((travel % period) + period) % period
      const strength = (0.55 + random() * 0.45).toFixed(2)
      const delay = `${(phase - period).toFixed(2)}s`
      if (tree.style.getPropertyValue('--trail-tree-wind-scale') !== strength)
        tree.style.setProperty('--trail-tree-wind-scale', strength)
      if (tree.style.getPropertyValue('--trail-weather-delay') !== delay)
        tree.style.setProperty('--trail-weather-delay', delay)
    })
    updateLighting()
  }
  const placeGateLink = (gate: FarGateSite, now: number) => {
    const pad = 4
    const edge = gate.radius * gate.cell + pad
    gateLink.style.left = `${gate.cx - edge}px`
    gateLink.style.top = `${gate.cy - edge}px`
    gateLink.style.width = `${edge * 2}px`
    gateLink.style.height = `${edge * 2}px`
    const time = new Date(now)
    const clock = `${time.getHours()}:${String(time.getMinutes()).padStart(2, '0')}`
    const label = `Stone clock on the far ridge, ${clock}. Step through to the 404`
    gateLink.setAttribute('aria-label', label)
    gateLink.title = label
    gateLink.hidden = false
  }
  // The shared world schedule decides what the herd, the campers and the trees do; the modules
  // that draw them read these attributes and listen for trail:climate.
  const updateSchedule = (now: number) => {
    const al = almanac(new Date(now), trailSite(weather, document))
    const sky = trailSky(weather)
    const cover = trailWinter(weather, snowy(), al, seasonOverride).cover
    const world = season(al, sky, seasonOverride)
    const next = {
      herd: herd(al, { ground: Math.max(world.ground, cover) }, sky),
      night: String(night(al)),
      campSleeps: String(campSleeps(al)),
      fireWeather: fireOut(sky) ? 'wet' : 'dry',
    }
    const changed = Object.entries(next).some(([key, value]) => scene.dataset[key] !== value)
    Object.assign(scene.dataset, next)
    scene.dataset.leafRate = leafRate(world, sky).toFixed(4)
    // Moss moisture, growth, snow and rime, and the wind across the clumps (-1 blowing left .. 1
    // right, the leaves' direction), in tenths so the clumps repaint only on a real change.
    const moss = mossClimate(al, { ...world, ground: cover }, sky, Date.now() < wetUntil)
    const lean = -Math.sin(((weather?.windFromDegrees ?? 270) * Math.PI) / 180) * sky.wind
    const mossState = [moss.moisture, moss.growth, moss.buried, moss.rime, lean]
      .map(value => (Math.round(value * 10) / 10).toFixed(1))
      .join(' ')
    if (scene.dataset.moss !== mossState) scene.dataset.moss = mossState
    return changed
  }
  // The sun and moon move about a pixel a minute on a wide screen.
  const renderSky = (now = sceneNow(document)) => {
    view?.clearTimeout(skyTimer)
    updateDamp()
    if (updateSchedule(now)) host.dispatchEvent(new CustomEvent('trail:climate'))
    if (!skyLayer || !landscape || signal.aborted) return
    skyAt = now
    skyLayer.innerHTML = skyMarks(landscape, ranges, weather, now, seed)
    const gate = farGateSite(landscape, ranges)
    if (gateLayer) gateLayer.innerHTML = farGate(gate, ranges, now)
    placeGateLink(gate, now)
    paintMoon(false)
    if (view) skyTimer = view.setTimeout(() => renderSky(), 60_000)
  }
  const render = () => {
    const now = sceneNow(document)
    const sky = almanac(new Date(now), trailSite(weather, document))
    const bucket = seasonBucket(sky.winterDay)
    const state = decoration(weather, sky, seasonOverride)
    if (state.rain > 0) wetUntil = Date.now() + 2 * 60 * 60 * 1000
    const { cover, forecastCover, frozen, forecastFrozen, source } = trailWinter(
      weather,
      snowy(),
      sky,
      seasonOverride,
    )
    const changedIce = scene.dataset.waterFrozen !== String(frozen)
    scene.dataset.waterFrozen = String(frozen)
    scene.dataset.forecastWaterFrozen = String(forecastFrozen)
    for (const wader of scene.querySelectorAll<SVGGElement>(
      '[data-wetland-animal="frog"], [data-wetland-animal="heron"]',
    ))
      wader.dataset.collisionActive = String(!frozen)
    scene.dataset.weatherMode = state.mode
    scene.dataset.weatherFog = String(state.fog)
    scene.dataset.weatherStorm = String(state.storm)
    scene.style.setProperty('--trail-storm-delay', stormDelay())
    scene.dataset.snowSource = source
    scene.dataset.festiveSnow = String(festiveSnow)
    scene.dataset.forecastSnowCover = forecastCover.toFixed(3)
    scene.dataset.weatherDay = weather?.day === null || !weather ? 'unknown' : String(weather.day)
    scene.dataset.weatherWind = String(weather?.windKmh ?? 0)
    scene.dataset.weatherGust = String(weather?.gustKmh ?? 'unknown')
    scene.dataset.weatherDirection = String(weather?.windFromDegrees ?? 'unknown')
    scene.dataset.weatherWindy = String(state.gust > 0.025)
    scene.dataset.season = seasonOverride ?? bucket.season
    scene.dataset.seasonOverride = seasonOverride ?? 'auto'
    // The winter illustration keeps its snow and ice separate from the local forecast.
    const foliage = snowy() ? 'winter' : (seasonOverride ?? scene.dataset.season)
    scene.dataset.foliageSeason = foliage
    // Early, mid or late in the season, by the third of the quarter it is in: leaves come out over
    // the spring and fall over the autumn. A chosen season is shown at its height.
    scene.dataset.seasonPhase = seasonOverride ? 'mid' : bucket.phase
    scene.dataset.snowCover = cover.toFixed(3)
    scene.style.setProperty('--trail-rain-strength', state.rain.toFixed(3))
    scene.style.setProperty('--trail-wind-strength', state.wind.toFixed(3))
    scene.style.setProperty('--trail-gust-strength', state.gust.toFixed(3))
    scene.style.setProperty('--trail-snow-cover', cover.toFixed(3))
    scene.style.setProperty('--trail-snow-mix', `${(cover * 75).toFixed(1)}%`)
    scene.style.setProperty('--trail-wind-x', `${(state.dx * state.wind * 18).toFixed(2)}px`)
    scene.style.setProperty('--trail-wind-y', `${(state.dy * state.wind * 18).toFixed(2)}px`)
    scene.style.setProperty('--trail-rain-x', `${(state.dx * state.wind * 12).toFixed(2)}px`)
    scene.style.setProperty('--trail-wind-period', `${(9 - state.gust * 3).toFixed(2)}s`)
    // Leaf clusters move in tree cells: a gust peak, a steady lean, and how fast the gust front
    // crosses a crown (seconds per tree unit, matching the per-tree delays in refreshForest).
    const flex = 1 - cover * 0.45
    const push = state.gust > 0.025 ? Math.min(2.2, 0.6 + state.gust * 1.8) * flex : 0
    const rest = state.wind * 1.1 * flex
    const front = 55 + state.gust * 90
    scene.style.setProperty('--trail-leaf-push-x', (state.dx * push).toFixed(3))
    scene.style.setProperty('--trail-leaf-push-y', (state.dy * push * 0.35).toFixed(3))
    scene.style.setProperty('--trail-leaf-rest-x', (state.dx * rest).toFixed(3))
    scene.style.setProperty('--trail-leaf-rest-y', (state.dy * rest * 0.35).toFixed(3))
    scene.style.setProperty('--trail-leaf-flip', Math.min(1, 0.35 + state.gust).toFixed(3))
    scene.style.setProperty('--trail-leaf-lag-x', `${(state.dx / front).toFixed(4)}s`)
    scene.style.setProperty('--trail-leaf-lag-y', `${(state.dy / front).toFixed(4)}s`)
    scene.style.setProperty('--trail-leaf-travel-x', `${(state.dx * state.gust * 45).toFixed(2)}px`)
    scene.style.setProperty(
      '--trail-leaf-travel-y',
      `${(24 + state.dy * state.gust * 14).toFixed(2)}px`,
    )
    if (layer) {
      layer.dataset.trailWeather = state.mode
      if (landscape)
        layer.innerHTML = weatherMarks(
          landscape,
          seed,
          weather,
          snowy(),
          isSeason(scene.dataset.foliageSeason) ? scene.dataset.foliageSeason : null,
          sky,
          seasonOverride,
        )
    }
    renderSky(now)
    updateLighting()
    updateMotion()
    refreshForest()
    if (changedIce) host.dispatchEvent(new CustomEvent('trail:climate'))
  }
  document.addEventListener('themechange', updateLighting, { signal })
  subscribeSceneClock(document, () => render(), signal)
  subscribeTrailSeason(
    document,
    next => {
      if (next === seasonOverride) return
      seasonOverride = next
      render()
    },
    signal,
  )
  host.addEventListener(
    'trail:holiday',
    event => {
      const next = event.detail === 'christmas'
      if (next === festiveSnow) return
      festiveSnow = next
      render()
    },
    { signal },
  )
  document.addEventListener(
    'visibilitychange',
    () => {
      if (!document.hidden && !paused && !host.hidden) renderSky()
      updateMotion()
    },
    { signal },
  )
  motion?.addEventListener('change', updateMotion, { signal })
  subscribeVisitorDaylight(
    document,
    current => {
      if (!current) return
      daylight = current
      if (Math.floor(current.at / 60_000) !== Math.floor(skyAt / 60_000)) renderSky(current.at)
      updateLighting()
    },
    signal,
  )
  subscribeVisitorMoon(
    document,
    current => {
      moon = current
      paintMoon(scene.dataset.lighting === 'moonlit')
    },
    signal,
  )
  subscribeVisitorTrailWeather(
    document,
    conditions => {
      weather = conditions
      render()
    },
    signal,
  )
  render()
  signal.addEventListener(
    'abort',
    () => {
      scene.dataset.weatherMotion = 'paused'
      scene.style.removeProperty('--trail-snow-cover')
      scene.style.removeProperty('--trail-snow-mix')
      delete scene.dataset.snowCover
      delete scene.dataset.snowSource
      delete scene.dataset.festiveSnow
      delete scene.dataset.forecastSnowCover
      delete scene.dataset.forecastWaterFrozen
      delete scene.dataset.waterFrozen
      delete scene.dataset.groundDamp
      delete scene.dataset.herd
      delete scene.dataset.night
      delete scene.dataset.campSleeps
      delete scene.dataset.fireWeather
      delete scene.dataset.leafRate
      delete scene.dataset.moss
      view?.clearTimeout(skyTimer)
      layer?.replaceChildren()
      skyLayer?.replaceChildren()
      gateLayer?.replaceChildren()
      gateLink.remove()
    },
    { once: true },
  )
  return {
    update(current: Landscape, currentSeed: number) {
      landscape = current
      seed = currentSeed
      ranges = buildMountainRanges(current.alpine, currentSeed)
      render()
    },
    refreshForest,
    pause() {
      paused = true
      updateMotion()
    },
    resume() {
      paused = false
      if (Math.abs(sceneNow(document) - skyAt) >= 60_000) renderSky()
      updateMotion()
    },
  }
}
