import { mapboxStyleUrl, type MapboxTheme } from '../../util/mapbox-style'
import { applyMonochromeMapPalette, loadMapbox } from './mapbox-client'

interface VisitorMap {
  getStyle(): { layers?: { id: string; type: string }[] } | undefined
  setPaintProperty(layer: string, property: string, value: string | number): void
  setLayoutProperty(layer: string, property: string, value: string): void
  on(type: 'style.load' | 'idle' | 'error', listener: () => void): void
  setStyle(style: string): void
  setPadding(padding: { top: number; right: number; bottom: number; left: number }): void
  resize(): void
  remove(): void
}

interface VisitorMapLibrary {
  Map: new (options: {
    container: HTMLElement
    style: string
    center: [number, number]
    zoom: number
    pitch: number
    bearing: number
    interactive: boolean
    attributionControl: boolean
  }) => VisitorMap
}

export function mountVisitorWeatherMap(container: HTMLElement, lat: number, lon: number) {
  const status = container.parentElement?.querySelector<HTMLButtonElement>(
    '[data-weather-map-status]',
  )
  const controller = new AbortController()
  let generation = 0
  let map: VisitorMap | null = null
  container.inert = true
  const theme = (): MapboxTheme =>
    document.documentElement.getAttribute('saved-theme') === 'dark' ? 'dark' : 'light'
  const fail = () => {
    if (controller.signal.aborted || !status) return
    container.dataset.mapState = 'unavailable'
    status.hidden = false
    status.disabled = false
    status.textContent = 'map unavailable · retry'
  }
  const resize = () => {
    if (!map || !container.clientWidth || !container.clientHeight) return
    map.resize()
    // Centre the camera in the visible crop of the taller GL canvas.
    map.setPadding({
      top: 0,
      right: 0,
      bottom: Math.max(0, container.clientHeight - (container.parentElement?.clientHeight ?? 0)),
      left: 0,
    })
  }
  const paint = (current: VisitorMap) => {
    const dark = theme() === 'dark'
    const paper = getComputedStyle(document.documentElement).getPropertyValue('--light').trim()
    applyMonochromeMapPalette(current, theme())
    for (const { id, type } of current.getStyle()?.layers ?? []) {
      if (type === 'symbol') current.setLayoutProperty(id, 'visibility', 'none')
      else if (type === 'background') current.setPaintProperty(id, 'background-color', paper)
      else if (type === 'fill') {
        current.setPaintProperty(
          id,
          'fill-color',
          id.includes('water') ? (dark ? '#23221f' : '#e8e5da') : paper,
        )
        current.setPaintProperty(id, 'fill-opacity', 1)
      } else if (type === 'line') {
        current.setPaintProperty(id, 'line-color', dark ? '#878580' : '#b6afa2')
        current.setPaintProperty(id, 'line-opacity', 0.7)
      }
    }
  }
  const load = async () => {
    const revision = ++generation
    map?.remove()
    map = null
    container.dataset.mapState = 'loading'
    if (status) {
      status.hidden = false
      status.disabled = true
      status.textContent = 'loading map'
    }
    const library: VisitorMapLibrary | null = await loadMapbox()
    if (controller.signal.aborted || revision !== generation) return
    if (!library) {
      fail()
      return
    }
    try {
      const current = new library.Map({
        container,
        style: mapboxStyleUrl('mono', theme()),
        center: [lon, lat],
        zoom: 8.5,
        pitch: 0,
        bearing: 0,
        interactive: false,
        attributionControl: true,
      })
      map = current
      current.on('style.load', () => {
        if (map === current && !controller.signal.aborted) paint(current)
      })
      current.on('idle', () => {
        if (map !== current || controller.signal.aborted) return
        container.dataset.mapState = 'ready'
        if (status) status.hidden = true
      })
      current.on('error', () => {
        if (map === current) fail()
      })
      resize()
    } catch {
      fail()
    }
  }
  status?.addEventListener('click', () => void load(), { signal: controller.signal })
  const themeObserver = new MutationObserver(() => map?.setStyle(mapboxStyleUrl('mono', theme())))
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['saved-theme'],
  })
  const sizeObserver = new ResizeObserver(resize)
  sizeObserver.observe(container)
  void load()
  return {
    resize,
    dispose: () => {
      controller.abort()
      generation++
      themeObserver.disconnect()
      sizeObserver.disconnect()
      map?.remove()
      map = null
    },
  }
}
