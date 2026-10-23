import type { VisitorDaylight } from '../../../util/visitor-daylight'
import type { TrailWeather } from '../../../util/visitor-trail-weather'
import { trailRandom } from '../random'
import { decoration } from './conditions'

type ForestWeather = { weather: TrailWeather | null; daylight: VisitorDaylight; seed: number }

export function createForestWeather(
  scene: HTMLElement,
  host: HTMLElement,
  read: () => ForestWeather,
) {
  const document = scene.ownerDocument
  const layer = scene.querySelector<SVGGElement>('.trail-weather')
  const updateLighting = () => {
    const { weather, daylight } = read()
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
  const refreshForest = () => {
    const { weather, seed } = read()
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
      const delay = (phase - period).toFixed(2)
      if (tree.dataset.windScale !== strength) tree.dataset.windScale = strength
      if (tree.dataset.windDelay !== delay) tree.dataset.windDelay = delay
    })
    updateLighting()
  }
  return { updateLighting, refreshForest }
}
