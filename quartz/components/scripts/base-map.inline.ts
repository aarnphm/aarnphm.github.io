import {
  applyMapElevation,
  applyMapTerrain,
  type MapboxOverlayMap,
} from '../../util/mapbox-overlays'
import { mapboxStyleUrl, type MapboxTheme } from '../../util/mapbox-style'
import { isRecord, readString } from '../../util/type-guards'
import { createPopupContent, markerColor, readBaseMapData, type MarkerData } from './base-map-data'
import { mountBaseMapView } from './base-map-view'
import { applyMonochromeMapPalette, loadMapbox } from './mapbox-client'

type Coordinates = [number, number]
interface MapBounds {
  extend(coordinates: Coordinates): void
  getCenter(): { toArray(): number[] }
}
interface MapFeature {
  properties?: unknown
  geometry?: { coordinates?: unknown }
}
interface MapLayerEvent {
  point: unknown
  features?: MapFeature[]
}
interface MarkerSource {
  setData(data: unknown): void
  getClusterExpansionZoom(
    clusterId: number,
    callback: (error: unknown, zoom?: number) => void,
  ): void
}
interface BaseMapInstance extends MapboxOverlayMap {
  once(type: 'load', listener: () => void): void
  on(type: 'style.load', listener: () => void): void
  on(type: 'click', layer: string, listener: (event: MapLayerEvent) => void): void
  on(type: 'mouseenter' | 'mouseleave', layer: string, listener: () => void): void
  queryRenderedFeatures(point: unknown, options: { layers: string[] }): MapFeature[]
  getSource(id: string): MarkerSource | undefined
  easeTo(options: { center?: Coordinates; zoom?: number; pitch?: number; duration: number }): void
  getCanvas(): HTMLCanvasElement
  getZoom(): number
  isStyleLoaded(): boolean
  fitBounds(
    bounds: MapBounds,
    options: { padding: number; maxZoom: number; duration: number },
  ): void
  setPaintProperty(layer: string, property: string, value: string | number): void
  setStyle(style: string): void
  resize(): void
  remove(): void
}
interface BaseMapPopup {
  setLngLat(coordinates: Coordinates): BaseMapPopup
  setHTML(content: string): BaseMapPopup
  addTo(map: BaseMapInstance): BaseMapPopup
  remove(): void
}
interface BaseMapLibrary {
  Map: new (options: {
    container: HTMLElement
    style: string
    center: Coordinates
    zoom: number
    pitch: number
    attributionControl: boolean
  }) => BaseMapInstance
  Popup: new (options: { offset: number; maxWidth: string; className: string }) => BaseMapPopup
  LngLatBounds: new () => MapBounds
  Marker: new (options: { element: HTMLElement; anchor: string }) => {
    setLngLat(coordinates: Coordinates): { addTo(map: BaseMapInstance): void }
  }
}
interface BaseMapState {
  controller: AbortController
  map?: BaseMapInstance
}
const mapStates = new Map<HTMLElement, BaseMapState>()
let initializationTimer: number | undefined

function renderEmpty(container: HTMLElement, message: string): void {
  const empty = document.createElement('div')
  empty.className = 'base-map-empty'
  empty.textContent = message
  container.replaceChildren(empty)
}
function isCurrentState(container: HTMLElement, state: BaseMapState): boolean {
  return (
    container.isConnected && !state.controller.signal.aborted && mapStates.get(container) === state
  )
}
function disposeMap(container: HTMLElement, state: BaseMapState): void {
  state.controller.abort()
  state.map?.remove()
  if (mapStates.get(container) === state) mapStates.delete(container)
}
function coordinatesFromFeature(feature: MapFeature | undefined): Coordinates | undefined {
  const coordinates = feature?.geometry?.coordinates
  if (
    !Array.isArray(coordinates) ||
    coordinates.length < 2 ||
    typeof coordinates[0] !== 'number' ||
    typeof coordinates[1] !== 'number' ||
    !Number.isFinite(coordinates[0]) ||
    !Number.isFinite(coordinates[1])
  )
    return undefined
  return [coordinates[0], coordinates[1]]
}
function readPreference(key: string): boolean {
  try {
    return localStorage.getItem(key) === 'true'
  } catch {
    return false
  }
}
function savePreference(key: string, enabled: boolean): void {
  try {
    localStorage.setItem(key, String(enabled))
  } catch {
    /* The map still works when storage is unavailable. */
  }
}
function theme(): MapboxTheme {
  return document.documentElement.getAttribute('saved-theme') === 'dark' ? 'dark' : 'light'
}

async function initializeMap(container: HTMLElement, state: BaseMapState): Promise<void> {
  const data = readBaseMapData({
    markers: container.dataset.markers,
    config: container.dataset.config,
    currentSlug: container.dataset.currentSlug,
    properties: container.dataset.properties,
  })
  if (!data || data.markers.length === 0) {
    renderEmpty(container, data ? 'no locations to display' : 'map unavailable')
    return
  }
  let satellite = readPreference('base-map-satellite')
  let elevation = readPreference('base-map-elevation')
  let threeDimensional = readPreference('base-map-3d')
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let map: BaseMapInstance | undefined
  let popup: BaseMapPopup | undefined
  const iconMarkers = new Map<string, HTMLButtonElement>()
  const geojson = (markers: MarkerData[]) => ({
    type: 'FeatureCollection',
    features: markers.map(marker => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [marker.lon, marker.lat] },
      properties: { slug: marker.slug, title: marker.title, color: markerColor(marker) },
    })),
  })
  const style = () => (satellite ? 'satellite' : 'mono')
  const fit = () => {
    if (!map || !library || !view.markers().length) return
    const bounds = new library.LngLatBounds()
    for (const marker of view.markers()) bounds.extend([marker.lon, marker.lat])
    map.fitBounds(bounds, { padding: 48, maxZoom: 15, duration: reduce ? 0 : 450 })
  }
  const select = (marker: MarkerData, move: boolean) => {
    if (!map || !library) return
    popup?.remove()
    popup = new library.Popup({ offset: 14, maxWidth: '300px', className: 'base-map-spot-popup' })
      .setLngLat([marker.lon, marker.lat])
      .setHTML(createPopupContent(marker, data.currentSlug, data.properties))
      .addTo(map)
    if (move)
      map.easeTo({
        center: [marker.lon, marker.lat],
        zoom: Math.max(map.getZoom(), 15),
        duration: reduce ? 0 : 450,
      })
  }
  const view = mountBaseMapView(
    container,
    data,
    {
      select: marker => select(marker, true),
      clear: () => popup?.remove(),
      filter: markers => {
        if (map?.isStyleLoaded()) map.getSource('markers')?.setData(geojson(markers))
        for (const [slug, element] of iconMarkers) {
          element.hidden = !markers.some(marker => marker.slug === slug)
        }
        popup?.remove()
      },
      fit,
      control: control => {
        if (!map) return
        if (control === 'satellite') {
          satellite = !satellite
          savePreference('base-map-satellite', satellite)
          view.pressed(control, satellite)
          map.setStyle(mapboxStyleUrl(style(), theme()))
        } else if (control === 'elevation') {
          elevation = !elevation
          savePreference('base-map-elevation', elevation)
          view.pressed(control, elevation)
          if (map.isStyleLoaded())
            applyMapElevation(map, elevation, style(), theme(), 'metric', 'base')
        } else {
          threeDimensional = !threeDimensional
          savePreference('base-map-3d', threeDimensional)
          view.pressed(control, threeDimensional)
          if (map.isStyleLoaded()) applyMapTerrain(map, threeDimensional, style(), theme(), 'base')
          map.easeTo({ pitch: threeDimensional ? 55 : 0, duration: reduce ? 0 : 240 })
        }
      },
    },
    state.controller.signal,
  )
  view.pressed('satellite', satellite)
  view.pressed('elevation', elevation)
  view.pressed('3d', threeDimensional)
  const library: BaseMapLibrary | null = await loadMapbox()
  if (!isCurrentState(container, state)) return
  if (!library || !view.canvas) {
    renderEmpty(container, 'map unavailable')
    disposeMap(container, state)
    return
  }
  const bounds = new library.LngLatBounds()
  for (const marker of data.markers) bounds.extend([marker.lon, marker.lat])
  const [lon = 0, lat = 0] = bounds.getCenter().toArray()
  const center: Coordinates = data.config.defaultCenter
    ? [data.config.defaultCenter[1], data.config.defaultCenter[0]]
    : [lon, lat]
  const current = new library.Map({
    container: view.canvas,
    style: mapboxStyleUrl(style(), theme()),
    center,
    zoom: data.config.defaultZoom,
    pitch: threeDimensional ? 55 : 0,
    attributionControl: false,
  })
  map = current
  state.map = current
  const clustered = data.config.clustering && data.markers.length > 10
  if (!clustered) {
    for (const marker of data.markers) {
      if (!marker.icon) continue
      const element = document.createElement('button')
      element.type = 'button'
      element.className = 'base-map-marker'
      element.setAttribute('aria-label', marker.title)
      element.style.color = markerColor(marker)
      const match = marker.icon.match(/^<i class="([a-z0-9_ -]+)" aria-hidden="true"><\/i>$/i)
      if (match) {
        const glyph = document.createElement('i')
        glyph.classList.add(...match[1].split(/\s+/).filter(Boolean))
        glyph.setAttribute('aria-hidden', 'true')
        element.append(glyph)
      } else element.textContent = marker.icon
      element.addEventListener(
        'click',
        () => {
          view.show(marker)
          select(marker, false)
        },
        { signal: state.controller.signal },
      )
      iconMarkers.set(marker.slug, element)
      element.hidden = !view.markers().some(item => item.slug === marker.slug)
      new library.Marker({ element, anchor: 'bottom' })
        .setLngLat([marker.lon, marker.lat])
        .addTo(current)
    }
  }
  const installLayers = () => {
    if (!isCurrentState(container, state)) return
    if (!satellite) applyMonochromeMapPalette(current, theme())
    applyMapTerrain(current, threeDimensional, style(), theme(), 'base')
    applyMapElevation(current, elevation, style(), theme(), 'metric', 'base')
    if (!current.getSource('markers'))
      current.addSource('markers', {
        type: 'geojson',
        data: geojson(view.markers()),
        cluster: clustered,
        clusterMaxZoom: 14,
        clusterRadius: 44,
      })
    if (current.getLayer('clusters')) return
    current.addLayer({
      id: 'clusters',
      type: 'circle',
      source: 'markers',
      filter: ['has', 'point_count'],
      paint: {
        'circle-color': theme() === 'dark' ? '#b7a58b' : '#5c5141',
        'circle-radius': ['step', ['get', 'point_count'], 17, 10, 23, 30, 30],
        'circle-stroke-width': 2,
        'circle-stroke-color': theme() === 'dark' ? '#100f0f' : '#fff9f3',
      },
    })
    current.addLayer({
      id: 'cluster-count',
      type: 'symbol',
      source: 'markers',
      filter: ['has', 'point_count'],
      layout: {
        'text-field': '{point_count_abbreviated}',
        'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
        'text-size': 12,
      },
      paint: { 'text-color': theme() === 'dark' ? '#100f0f' : '#fff9f3' },
    })
    current.addLayer({
      id: 'unclustered-point',
      type: 'circle',
      source: 'markers',
      filter: ['!', ['has', 'point_count']],
      paint: {
        'circle-color': ['get', 'color'],
        'circle-radius': 7,
        'circle-stroke-width': 2,
        'circle-stroke-color': '#fff9f3',
        'circle-opacity': [
          'case',
          ['in', ['get', 'slug'], ['literal', [...iconMarkers.keys()]]],
          0,
          1,
        ],
        'circle-stroke-opacity': [
          'case',
          ['in', ['get', 'slug'], ['literal', [...iconMarkers.keys()]]],
          0,
          1,
        ],
      },
    })
    current.addLayer({
      id: 'spot-labels',
      type: 'symbol',
      source: 'markers',
      filter: ['!', ['has', 'point_count']],
      minzoom: 13,
      layout: {
        'text-field': ['get', 'title'],
        'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
        'text-size': 11,
        'text-anchor': 'top',
        'text-offset': [0, 1],
        'text-max-width': 15,
      },
      paint: {
        'text-color': satellite || theme() === 'dark' ? '#fff9f3' : '#2b2418',
        'text-halo-color': satellite || theme() === 'dark' ? '#100f0f' : '#fff9f3',
        'text-halo-width': 1.5,
      },
    })
    view.ready()
  }
  current.on('style.load', installLayers)
  current.once('load', () => {
    if (!isCurrentState(container, state)) return
    if (!data.config.defaultCenter && data.markers.length > 1) fit()
  })
  current.on('click', 'clusters', event => {
    const feature = current.queryRenderedFeatures(event.point, { layers: ['clusters'] })[0]
    const coordinates = coordinatesFromFeature(feature)
    if (!coordinates || !isRecord(feature?.properties)) return
    const clusterId = Number(feature.properties.cluster_id)
    if (!Number.isFinite(clusterId)) return
    current.getSource('markers')?.getClusterExpansionZoom(clusterId, (error, zoom) => {
      if (!error && zoom !== undefined && isCurrentState(container, state))
        current.easeTo({ center: coordinates, zoom, duration: reduce ? 0 : 450 })
    })
  })
  current.on('click', 'unclustered-point', event => {
    const properties = event.features?.[0]?.properties
    if (!isRecord(properties)) return
    const slug = readString(properties, 'slug')
    const marker = data.markers.find(item => item.slug === slug)
    if (marker) {
      view.show(marker)
      select(marker, false)
    }
  })
  for (const layer of ['clusters', 'unclustered-point']) {
    current.on('mouseenter', layer, () => {
      current.getCanvas().style.cursor = 'pointer'
    })
    current.on('mouseleave', layer, () => {
      current.getCanvas().style.cursor = ''
    })
  }
  const resize = new ResizeObserver(() => current.resize())
  resize.observe(view.canvas)
  document.addEventListener(
    'themechange',
    () => current.setStyle(mapboxStyleUrl(style(), theme())),
    { signal: state.controller.signal },
  )
  state.controller.signal.addEventListener(
    'abort',
    () => {
      resize.disconnect()
      popup?.remove()
    },
    { once: true },
  )
}

function initBaseMaps(): void {
  for (const [container, state] of mapStates)
    if (!container.isConnected) disposeMap(container, state)
  for (const container of document.querySelectorAll<HTMLElement>('.base-map')) {
    if (mapStates.has(container)) continue
    const state: BaseMapState = { controller: new AbortController() }
    mapStates.set(container, state)
    void initializeMap(container, state).catch(error => {
      if (!isCurrentState(container, state)) return
      disposeMap(container, state)
      renderEmpty(container, 'map unavailable')
      console.error(error)
    })
  }
}
document.addEventListener('nav', () => {
  if (initializationTimer === undefined)
    initializationTimer = window.setTimeout(() => {
      initializationTimer = undefined
      initBaseMaps()
    }, 100)
  const observer = new MutationObserver(() => {
    for (const [container, state] of mapStates)
      if (!container.isConnected) disposeMap(container, state)
  })
  observer.observe(document.documentElement, { childList: true, subtree: true })
  window.addCleanup(() => {
    observer.disconnect()
    if (initializationTimer !== undefined) {
      window.clearTimeout(initializationTimer)
      initializationTimer = undefined
    }
    for (const [container, state] of mapStates) disposeMap(container, state)
  })
})
