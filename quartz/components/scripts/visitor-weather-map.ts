import { mapboxStyleUrl } from '../../util/mapbox-style'
import { applyMonochromeMapPalette, loadMapbox } from './mapbox-client'

interface WeatherMap {
  getStyle(): { layers?: { id: string; type: string }[] } | undefined
  setPaintProperty(layer: string, property: string, value: string | number): void
  setStyle(style: string): void
  once(event: 'style.load' | 'load', callback: () => void): void
  on(event: 'error', callback: () => void): void
  resize(): void
  remove(): void
}
interface WeatherMapLibrary {
  Map: new (options: {
    container: HTMLElement
    style: string
    center: [number, number]
    zoom: number
    interactive: boolean
    attributionControl: boolean | { compact: boolean }
  }) => WeatherMap
}

export function mountVisitorWeatherMap(container: HTMLElement, lat: number, lon: number) {
  const status = container.parentElement?.querySelector<HTMLButtonElement>(
    '[data-weather-map-status]',
  )
  const controller = new AbortController()
  let map: WeatherMap | null = null
  let generation = 0
  const theme = () =>
    document.documentElement.getAttribute('saved-theme') === 'dark' ? 'dark' : 'light'
  const fail = () => {
    if (controller.signal.aborted || !status) return
    status.hidden = false
    status.disabled = false
    status.textContent = 'map unavailable · retry'
  }
  const load = async () => {
    const current = ++generation
    map?.remove()
    map = null
    if (status) {
      status.hidden = false
      status.disabled = true
      status.textContent = 'loading map'
    }
    try {
      const library: WeatherMapLibrary | null = await loadMapbox()
      if (controller.signal.aborted || generation !== current) return
      if (!library) {
        fail()
        return
      }
      map = new library.Map({
        container,
        style: mapboxStyleUrl('mono', theme()),
        center: [lon, lat],
        zoom: 9.5,
        interactive: false,
        attributionControl: { compact: true },
      })
      const active = map
      active.once('style.load', () => applyMonochromeMapPalette(active, theme()))
      active.once('load', () => {
        if (status) status.hidden = true
        container.dataset.mapState = 'ready'
      })
      active.on('error', () => {
        if (!status?.hidden) fail()
      })
    } catch {
      fail()
    }
  }
  status?.addEventListener('click', () => void load(), { signal: controller.signal })
  const observer = new MutationObserver(() => {
    if (!map) return
    const active = map
    active.once('style.load', () => applyMonochromeMapPalette(active, theme()))
    active.setStyle(mapboxStyleUrl('mono', theme()))
  })
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['saved-theme'] })
  void load()
  return {
    resize: () => map?.resize(),
    dispose: () => {
      controller.abort()
      generation++
      observer.disconnect()
      map?.remove()
      map = null
    },
  }
}
