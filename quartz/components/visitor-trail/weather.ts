import type { Landscape } from './landscape'
import { lunarPhase } from '../../util/lunar-phase'
import { subscribeVisitorMoon, type VisitorMoon } from '../../util/visitor-lunar'
import { subscribeVisitorTrailWeather, type TrailWeather } from '../../util/visitor-trail-weather'
import { paintLunarDisc, stopLunarTransition } from '../scripts/lunar-disc'
import { trailRandom } from './random'

const rainCodes = new Map([
  [51, 0.3],
  [53, 0.45],
  [55, 0.55],
  [56, 0.35],
  [57, 0.55],
  [61, 0.55],
  [63, 0.75],
  [65, 0.95],
  [66, 0.6],
  [67, 0.9],
  [80, 0.6],
  [81, 0.8],
  [82, 0.98],
  [95, 0.8],
  [96, 0.9],
  [99, 0.95],
])
const snowCodes = new Set([71, 73, 75, 77, 85, 86])

function decoration(weather: TrailWeather | null) {
  const rain = weather ? (rainCodes.get(weather.code) ?? 0) : 0
  const snow = weather && snowCodes.has(weather.code) ? 1 : 0
  const sun =
    weather?.day && weather.code >= 0 && weather.code <= 2 && !rain && !snow
      ? Math.max(0, (0.85 - weather.cloud) / 0.85)
      : 0
  const angle = ((weather?.windFromDegrees ?? 0) * Math.PI) / 180
  const wind = weather ? Math.min(1, weather.windKmh / 70) : 0
  // Meteorological directions describe where the wind comes from; SVG y grows southward.
  const dx = weather?.windFromDegrees === null ? 0 : -Math.sin(angle)
  const dy = weather?.windFromDegrees === null ? 0 : Math.cos(angle)
  const mode = !weather
    ? 'quiet'
    : rain
      ? 'rain'
      : snow
        ? 'snow'
        : weather.day === false
          ? 'night'
          : sun
            ? 'sun'
            : 'cloud'
  return { rain, snow, sun, wind, dx, dy, mode }
}

function weatherMarks(landscape: Landscape, seed: number, weather: TrailWeather | null) {
  const random = trailRandom(seed ^ 0x34c4f69b)
  const { width, height, bottom, groves } = landscape
  const { rain, snow, sun, wind, dx, dy } = decoration(weather)
  const marks: string[] = []
  const skyX = Math.round(width * 0.9)
  const skyY = bottom + 86
  marks.push(`<g class="trail-weather-night">
    <g class="trail-weather-moon" transform="translate(${skyX} ${skyY})">
      <g class="trail-moon-motion">
      <path class="trail-moon-shadow" d="M-4-10h8v2h4v4h2v8H8v4H4v2h-8V8h-4V4h-2v-8h2v-4h4Z"/>
      <path class="trail-moon-lit" data-lunar-disc/>
      <path class="trail-moon-crater" d="M3-5h2v2H3ZM4 3h3v2H4Z"/>
      </g>
    </g>
    ${Array.from({ length: 6 }, () => {
      const x = Math.round(20 + random() * (width - 40))
      const y = Math.round(bottom + 22 + random() * 70)
      return `<path class="trail-weather-star" d="M${x} ${y}h2v2h-2Z"/>`
    }).join('')}
  </g>`)
  marks.push('<g class="trail-weather-daylight">')
  if (sun) {
    marks.push(
      `<g class="trail-weather-sun" transform="translate(${skyX} ${skyY})"><path d="M-4-6h8v2h2v8h-2v2h-8V4h-2V-4h2ZM-1-12h2v3h-2ZM-1 9h2v3h-2ZM-12-1h3v2h-3ZM9-1h3v2H9Z"/></g>`,
    )
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
  if (wind > 0.025 && weather?.windFromDegrees !== null) {
    const count = width < 700 ? 6 : 10
    for (let index = 0; index < count; index++) {
      const x = Math.round(24 + random() * Math.max(1, width - 70))
      const y = Math.round(bottom + 55 + random() * Math.max(1, height - bottom - 100))
      const rotation = (Math.atan2(dy, dx) * 180) / Math.PI
      marks.push(
        `<g transform="translate(${x} ${y}) rotate(${rotation.toFixed(1)})"><path class="trail-weather-gust" style="--trail-weather-delay: ${(-random() * 8).toFixed(2)}s" d="M0 0h12m4 0h8m-5-3h9"/></g>`,
      )
    }
  }
  if (rain || snow) {
    const count = width < 700 ? 28 : 48
    for (let index = 0; index < count; index++) {
      const x = Math.round(12 + random() * Math.max(1, width - 28))
      const y = Math.round(20 + random() * Math.max(1, height - 72))
      const stroke = snow
        ? 'M0 0h2v2H0Z'
        : `M0 0l${(dx * wind * 5).toFixed(1)} ${5 + Math.round(rain * 3)}`
      const kind = snow ? 'snow' : 'rain'
      marks.push(
        `<g transform="translate(${x} ${y})"><path class="trail-weather-${kind}" style="--trail-weather-delay: ${(-random() * 6).toFixed(2)}s" d="${stroke}"/></g>`,
      )
    }
  }
  return marks.join('')
}

export function mountTrailWeather(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-weather')
  const motion = view?.matchMedia('(prefers-reduced-motion: reduce)')
  let landscape: Landscape | null = null
  let seed = 0
  let weather: TrailWeather | null = null
  let moon: VisitorMoon = { ...lunarPhase(Date.now()), at: Date.now() }
  let paused = host.hidden

  const paintMoon = (animate: boolean) => {
    scene.dataset.moonPhase = String(moon.phase)
    const path = layer?.querySelector<SVGPathElement>('[data-lunar-disc]')
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
      const path = layer?.querySelector<SVGPathElement>('[data-lunar-disc]')
      if (path) stopLunarTransition(path)
    }
  }
  const refreshForest = () => {
    const random = trailRandom(seed ^ 0x385ab251)
    scene.querySelectorAll<SVGGElement>('.trail-tree-motion').forEach(tree => {
      tree.style.setProperty('--trail-tree-wind-scale', (0.55 + random() * 0.45).toFixed(2))
      tree.style.setProperty('--trail-weather-delay', `${(-random() * 8).toFixed(2)}s`)
    })
  }
  const render = () => {
    const state = decoration(weather)
    scene.dataset.weatherMode = state.mode
    scene.dataset.weatherDay = weather?.day === null || !weather ? 'unknown' : String(weather.day)
    scene.dataset.weatherWind = String(weather?.windKmh ?? 0)
    scene.dataset.weatherDirection = String(weather?.windFromDegrees ?? 'unknown')
    scene.dataset.weatherWindy = String(state.wind > 0.025)
    scene.style.setProperty('--trail-sunlight', state.sun.toFixed(3))
    scene.style.setProperty('--trail-rain-strength', state.rain.toFixed(3))
    scene.style.setProperty('--trail-wind-strength', state.wind.toFixed(3))
    scene.style.setProperty('--trail-wind-x', `${(state.dx * state.wind * 18).toFixed(2)}px`)
    scene.style.setProperty('--trail-wind-y', `${(state.dy * state.wind * 18).toFixed(2)}px`)
    scene.style.setProperty('--trail-rain-x', `${(state.dx * state.wind * 12).toFixed(2)}px`)
    scene.style.setProperty('--trail-wind-period', `${(8 - state.wind * 3).toFixed(2)}s`)
    const lean = state.dx * state.wind * 1.5
    scene.style.setProperty('--trail-sway-low', `${(lean - state.wind * 0.6).toFixed(2)}deg`)
    scene.style.setProperty('--trail-sway-high', `${(lean + state.wind * 1.4).toFixed(2)}deg`)
    if (layer) {
      layer.dataset.trailWeather = state.mode
      if (landscape) layer.innerHTML = weatherMarks(landscape, seed, weather)
    }
    paintMoon(false)
    updateLighting()
    updateMotion()
  }
  document.addEventListener('themechange', updateLighting, { signal })
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
      layer?.replaceChildren()
    },
    { once: true },
  )
  return {
    update(current: Landscape, currentSeed: number) {
      landscape = current
      seed = currentSeed
      render()
      refreshForest()
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
