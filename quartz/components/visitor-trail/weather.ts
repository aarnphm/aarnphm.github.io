import type { Landscape } from './landscape'
import { lunarPhase } from '../../util/lunar-phase'
import { sceneNow, subscribeSceneClock } from '../../util/scene-clock'
import { readTrailSeason, subscribeTrailSeason } from '../../util/trail-season'
import {
  sampleVisitorDaylight,
  subscribeVisitorDaylight,
  type VisitorDaylight,
} from '../../util/visitor-daylight'
import { subscribeVisitorMoon, type VisitorMoon } from '../../util/visitor-lunar'
import { stormDelay } from '../../util/visitor-precipitation'
import {
  subscribeVisitorTrailWeather,
  trailSite,
  trailSky,
  type TrailSeason,
  type TrailWeather,
} from '../../util/visitor-trail-weather'
import { season, seasonBucket } from '../../util/visitor-world'
import { campSleeps, fireOut, herd, leafRate, mossClimate, night } from '../../util/world-schedule'
import { almanac } from '../scripts/404-almanac'
import { paintLunarDisc, stopLunarTransition } from '../scripts/lunar-disc'
import { buildMountainRanges, type MountainRange } from './mountain-profile'
import { decoration } from './weather/conditions'
import { createForestWeather } from './weather/forest'
import { farGate, farGateSite, type FarGateSite } from './weather/gate'
import { weatherMarks } from './weather/marks'
import { skyMarks } from './weather/sky'
import { trailWinter } from './winter'

export { farGateSite, type FarGateSite } from './weather/gate'

const isSeason = (value: unknown): value is TrailSeason =>
  value === 'spring' || value === 'summer' || value === 'autumn' || value === 'winter'

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

  const { updateLighting, refreshForest } = createForestWeather(scene, host, () => ({
    weather,
    daylight,
    seed,
  }))
  const updateMotion = () => {
    scene.dataset.weatherMotion =
      paused || host.hidden || document.hidden || motion?.matches ? 'paused' : 'running'
    if (scene.dataset.weatherMotion === 'paused') {
      const path = skyLayer?.querySelector<SVGPathElement>('[data-lunar-disc]')
      if (path) stopLunarTransition(path)
    }
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
