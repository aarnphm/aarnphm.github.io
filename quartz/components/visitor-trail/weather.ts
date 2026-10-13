import { lunarPhase } from '../../util/lunar-phase'
import { subscribeVisitorMoon, type VisitorMoon } from '../../util/visitor-lunar'
import { visitorPrecipitation } from '../../util/visitor-precipitation'
import { subscribeVisitorTrailWeather, type TrailWeather } from '../../util/visitor-trail-weather'
import { almanac, guessSite, type Body } from '../scripts/404-almanac'
import { paintLunarDisc, stopLunarTransition } from '../scripts/lunar-disc'
import { holidaySprite } from './holiday-sprites'
import { creekDistance, groundOffset, pondDistance, type Landscape } from './landscape'
import { buildMountainRanges, mountainCrestAt, type MountainRange } from './mountain-profile'
import { trailRandom } from './random'
import { skyBand } from './sky-band'

function decoration(weather: TrailWeather | null) {
  const precipitation = visitorPrecipitation(weather?.code)
  const rain = precipitation.kind === 'rain' ? precipitation.strength : 0
  const snow = precipitation.kind === 'snow' ? 1 : 0
  const recentSnow = weather?.snowCm ?? 0
  const snowCover = Math.max(
    snow ? (weather?.code === 75 || weather?.code === 86 ? 0.95 : 0.65) : 0,
    weather?.temperatureC !== null &&
      weather?.temperatureC !== undefined &&
      weather.temperatureC <= 2
      ? Math.min(1, recentSnow / 2.5)
      : 0,
  )
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
  return { rain, snow, snowCover, sun, wind, gust, dx, dy, mode }
}

function weatherMarks(
  landscape: Landscape,
  seed: number,
  weather: TrailWeather | null,
  festiveSnow: boolean,
) {
  const random = trailRandom(seed ^ 0x34c4f69b)
  const { width, height, bottom, groves } = landscape
  const { rain, snow, snowCover, sun, gust, dx, dy } = decoration(weather)
  const marks: string[] = []
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
  if (sun) {
    for (let index = 0; index < 20; index++) {
      const grove = groves[index % groves.length]
      const x = Math.round((grove.x + (random() - 0.5) * grove.rx * 2.5) / 4) * 4
      const y = Math.round((grove.y + (random() - 0.5) * grove.ry * 2) / 4) * 4
      const span = 16 + Math.floor(random() * 8) * 4
      marks.push(
        `<path class="trail-weather-dapple" style="--trail-weather-delay: ${(-random() * 10).toFixed(2)}s" d="M${x} ${y}h${span}v4h-8v4h-${span}v-4h8Z"/>`,
      )
    }
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
  if (weather?.season === 'autumn' && gust > 0.06 && !snow) {
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

const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
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
      const threshold = (bayer[(row % 4) * 4 + (column % 4)] + 0.5) / 16
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
  const { rain, snow, wind, dx } = decoration(weather)
  const clear = rain || snow ? 0 : 1 - (weather?.cloud ?? 0.3)
  const sun = hang(sky.sun, sunRadius, landscape, ranges)
  const moon = hang(sky.moon, moonRadius, landscape, ranges)
  const marks = [
    skyBand(landscape, seed, {
      now,
      clear,
      cloud: weather?.cloud ?? 0.3,
      rain,
      snow,
      wind,
      dx,
      sunAltitude: sky.sun.alt,
    }),
    duskGlow(landscape, sun.x, sky.dusk * (0.35 + clear * 0.65)),
  ]
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
  const motion = view?.matchMedia('(prefers-reduced-motion: reduce)')
  let landscape: Landscape | null = null
  let seed = 0
  let weather: TrailWeather | null = null
  let festiveSnow = scene.dataset.holiday === 'christmas'
  let moon: VisitorMoon = { ...lunarPhase(Date.now()), at: Date.now() }
  let paused = host.hidden
  let ranges: MountainRange[] = []
  let skyTimer = 0

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
    // Theme changes the illustration; the forecast fields retain their actual values.
    scene.dataset.lighting =
      document.documentElement.getAttribute('saved-theme') === 'dark' || weather?.day === false
        ? 'moonlit'
        : 'day'
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
    const random = trailRandom(seed ^ 0x385ab251)
    const state = decoration(weather)
    const period = 9 - state.gust * 3
    scene.querySelectorAll<SVGGElement>('.trail-tree-motion').forEach(tree => {
      const root = tree.closest<SVGGElement>('.trail-tree, .trail-woodland-tree, .trail-understory')
      const alongWind =
        Number(root?.dataset.groundX ?? 0) * state.dx +
        Number(root?.dataset.groundY ?? 0) * state.dy
      const travel = alongWind / (55 + state.gust * 90) + random() * 0.8
      const phase = ((travel % period) + period) % period
      tree.style.setProperty('--trail-tree-wind-scale', (0.55 + random() * 0.45).toFixed(2))
      tree.style.setProperty('--trail-weather-delay', `${(phase - period).toFixed(2)}s`)
    })
  }
  // The sun and moon move about a pixel a minute on a wide screen.
  const renderSky = () => {
    view?.clearTimeout(skyTimer)
    if (!skyLayer || !landscape || signal.aborted) return
    skyLayer.innerHTML = skyMarks(landscape, ranges, weather, Date.now(), seed)
    paintMoon(false)
    if (view) skyTimer = view.setTimeout(renderSky, 60_000)
  }
  const render = () => {
    const state = decoration(weather)
    const cover = Math.max(state.snowCover, festiveSnow ? 0.9 : 0)
    scene.dataset.weatherMode = state.mode
    scene.dataset.snowSource = festiveSnow ? 'holiday' : 'forecast'
    scene.dataset.festiveSnow = String(festiveSnow)
    scene.dataset.forecastSnowCover = state.snowCover.toFixed(3)
    scene.dataset.weatherDay = weather?.day === null || !weather ? 'unknown' : String(weather.day)
    scene.dataset.weatherWind = String(weather?.windKmh ?? 0)
    scene.dataset.weatherGust = String(weather?.gustKmh ?? 'unknown')
    scene.dataset.weatherDirection = String(weather?.windFromDegrees ?? 'unknown')
    scene.dataset.weatherWindy = String(state.gust > 0.025)
    scene.dataset.season = weather?.season ?? 'unknown'
    // Early, mid or late in the season, by the month of the three it is in: leaves come out over
    // the spring and fall over the autumn.
    const month = new Date().getMonth() + 1 + ((weather?.lat ?? 0) < 0 ? 6 : 0)
    scene.dataset.seasonPhase = ['early', 'mid', 'late'][(month % 12) % 3]
    scene.dataset.snowCover = cover.toFixed(3)
    scene.style.setProperty('--trail-sunlight', state.sun.toFixed(3))
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
    const flex = 1 - state.snowCover * 0.45
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
      if (landscape) layer.innerHTML = weatherMarks(landscape, seed, weather, festiveSnow)
    }
    renderSky()
    updateLighting()
    updateMotion()
    refreshForest()
  }
  document.addEventListener('themechange', updateLighting, { signal })
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
  document.addEventListener('visibilitychange', updateMotion, { signal })
  motion?.addEventListener('change', updateMotion, { signal })
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
      view?.clearTimeout(skyTimer)
      layer?.replaceChildren()
      skyLayer?.replaceChildren()
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
      updateMotion()
    },
  }
}
