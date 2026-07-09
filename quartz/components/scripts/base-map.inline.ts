import { isFullSlug } from '../../util/path'
import { isRecord, readString } from '../../util/type-guards'
import {
  createPopupContent,
  type MapProperties,
  type MarkerData,
  readBaseMapData,
  readPopupFields,
} from './base-map-data'
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

interface ClusterSource {
  getClusterExpansionZoom(
    clusterId: number,
    callback: (error: unknown, zoom?: number) => void,
  ): void
}

interface BaseMapInstance {
  addSource(id: string, source: unknown): void
  addLayer(layer: unknown): void
  once(type: 'load', listener: () => void): void
  on(type: 'click', layer: string, listener: (event: MapLayerEvent) => void): void
  on(type: 'mouseenter' | 'mouseleave', layer: string, listener: () => void): void
  queryRenderedFeatures(point: unknown, options: { layers: string[] }): MapFeature[]
  getSource(id: string): ClusterSource | undefined
  easeTo(options: { center: Coordinates; zoom: number }): void
  getCanvas(): HTMLCanvasElement
  fitBounds(
    bounds: MapBounds,
    options: {
      padding: { top: number; bottom: number; left: number; right: number }
      maxZoom: number
    },
  ): void
  remove(): void
}

interface BaseMapState {
  controller: AbortController
  map?: BaseMapInstance
}

const mapStates = new Map<HTMLElement, BaseMapState>()
let initializationTimer: number | undefined

function disposeRemovedMaps(node: Node): void {
  if (!(node instanceof Element)) return
  const containers = node.matches('.base-map')
    ? [node]
    : Array.from(node.querySelectorAll<HTMLElement>('.base-map'))
  for (const container of containers) {
    if (!(container instanceof HTMLElement)) continue
    const state = mapStates.get(container)
    if (state) disposeMap(container, state)
  }
}

const removalObserver = new MutationObserver(records => {
  for (const record of records) {
    for (const node of record.removedNodes) disposeRemovedMaps(node)
  }
})
removalObserver.observe(document.documentElement, { childList: true, subtree: true })

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
  try {
    state.map?.remove()
  } catch (error) {
    console.error(error)
  }
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
  ) {
    return undefined
  }
  return [coordinates[0], coordinates[1]]
}

function markerFromFeature(feature: MapFeature | undefined): MarkerData | undefined {
  const coordinates = coordinatesFromFeature(feature)
  if (!coordinates || !isRecord(feature?.properties)) return undefined

  const title = readString(feature.properties, 'title')
  const slug = readString(feature.properties, 'slug')
  if (!title || !slug || !isFullSlug(slug)) return undefined

  return {
    lat: coordinates[1],
    lon: coordinates[0],
    title,
    slug,
    icon: readString(feature.properties, 'icon'),
    color: readString(feature.properties, 'color'),
    popupFields: readPopupFields(feature.properties.popupFields),
  }
}

function clusterIdFromFeature(feature: MapFeature | undefined): number | undefined {
  if (!isRecord(feature?.properties)) return undefined
  const value = feature.properties.cluster_id
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value !== 'string' || value.trim().length === 0) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function addClusteredMarkers(
  mapboxgl: typeof window.mapboxgl,
  map: BaseMapInstance,
  markers: MarkerData[],
  currentSlug: MarkerData['slug'],
  properties: MapProperties,
  active: () => boolean,
): void {
  const geojson = {
    type: 'FeatureCollection',
    features: markers.map(marker => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [marker.lon, marker.lat] },
      properties: {
        title: marker.title,
        slug: marker.slug,
        icon: marker.icon,
        color: marker.color,
        popupFields: JSON.stringify(marker.popupFields),
      },
    })),
  }

  map.addSource('markers', {
    type: 'geojson',
    data: geojson,
    cluster: true,
    clusterMaxZoom: 14,
    clusterRadius: 50,
  })
  map.addLayer({
    id: 'clusters',
    type: 'circle',
    source: 'markers',
    filter: ['has', 'point_count'],
    paint: {
      'circle-color': '#2b2418',
      'circle-radius': ['step', ['get', 'point_count'], 20, 10, 30, 30, 40],
      'circle-opacity': 0.8,
    },
  })
  map.addLayer({
    id: 'cluster-count',
    type: 'symbol',
    source: 'markers',
    filter: ['has', 'point_count'],
    layout: {
      'text-field': '{point_count_abbreviated}',
      'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
      'text-size': 12,
    },
    paint: { 'text-color': '#fff9f3' },
  })
  map.addLayer({
    id: 'unclustered-point',
    type: 'circle',
    source: 'markers',
    filter: ['!', ['has', 'point_count']],
    paint: {
      'circle-color': '#2b2418',
      'circle-radius': 8,
      'circle-stroke-width': 2,
      'circle-stroke-color': '#fff9f3',
    },
  })

  map.on('click', 'clusters', event => {
    const feature = map.queryRenderedFeatures(event.point, { layers: ['clusters'] })[0]
    const coordinates = coordinatesFromFeature(feature)
    const clusterId = clusterIdFromFeature(feature)
    const source = map.getSource('markers')
    if (!coordinates || clusterId === undefined || !source) return

    source.getClusterExpansionZoom(clusterId, (error, zoom) => {
      if (error || zoom === undefined || !active()) return
      map.easeTo({ center: coordinates, zoom })
    })
  })

  map.on('click', 'unclustered-point', event => {
    const marker = markerFromFeature(event.features?.[0])
    if (!marker || !active()) return

    const popupContent = createPopupContent(marker, currentSlug, properties)
    new mapboxgl.Popup().setLngLat([marker.lon, marker.lat]).setHTML(popupContent).addTo(map)
  })

  map.on('mouseenter', 'clusters', () => {
    map.getCanvas().style.cursor = 'pointer'
  })
  map.on('mouseleave', 'clusters', () => {
    map.getCanvas().style.cursor = ''
  })
  map.on('mouseenter', 'unclustered-point', () => {
    map.getCanvas().style.cursor = 'pointer'
  })
  map.on('mouseleave', 'unclustered-point', () => {
    map.getCanvas().style.cursor = ''
  })
}

function addMarkerIcon(element: HTMLElement, icon: string): void {
  const match = icon.match(/^<i class="([a-z0-9_ -]+)" aria-hidden="true"><\/i>$/i)
  if (!match) {
    element.textContent = icon
    return
  }

  const classes = match[1].split(/\s+/).filter(Boolean)
  const iconElement = document.createElement('i')
  iconElement.classList.add(...classes)
  iconElement.setAttribute('aria-hidden', 'true')
  element.append(iconElement)
}

function addIndividualMarkers(
  mapboxgl: typeof window.mapboxgl,
  map: BaseMapInstance,
  markers: MarkerData[],
  currentSlug: MarkerData['slug'],
  properties: MapProperties,
): void {
  for (const marker of markers) {
    const element = document.createElement('div')
    element.className = 'base-map-marker'
    if (marker.icon) addMarkerIcon(element, marker.icon)
    if (marker.color && CSS.supports('color', marker.color)) element.style.color = marker.color

    const popup = new mapboxgl.Popup({ offset: 25 }).setHTML(
      createPopupContent(marker, currentSlug, properties),
    )
    new mapboxgl.Marker({ element, anchor: 'bottom' })
      .setLngLat([marker.lon, marker.lat])
      .setPopup(popup)
      .addTo(map)
  }
}

async function initializeMap(container: HTMLElement, state: BaseMapState): Promise<void> {
  const data = readBaseMapData({
    markers: container.dataset.markers,
    config: container.dataset.config,
    currentSlug: container.dataset.currentSlug,
    properties: container.dataset.properties,
  })
  if (!data) {
    renderEmpty(container, 'map unavailable')
    return
  }
  if (data.markers.length === 0) {
    renderEmpty(container, 'no locations to display')
    return
  }

  const mapboxgl = await loadMapbox()
  if (!isCurrentState(container, state)) return
  if (!mapboxgl) {
    renderEmpty(container, 'map unavailable')
    disposeMap(container, state)
    return
  }

  const bounds: MapBounds = new mapboxgl.LngLatBounds()
  for (const marker of data.markers) bounds.extend([marker.lon, marker.lat])

  let center: Coordinates
  if (data.config.defaultCenter) {
    center = [data.config.defaultCenter[1], data.config.defaultCenter[0]]
  } else {
    const [lon = 0, lat = 0] = bounds.getCenter().toArray()
    center = [lon, lat]
  }

  const map: BaseMapInstance = new mapboxgl.Map({
    container,
    style: 'mapbox://styles/mapbox/light-v11',
    center,
    zoom: data.config.defaultZoom,
    attributionControl: false,
  })
  state.map = map

  const clustered = data.config.clustering && data.markers.length > 10
  if (!clustered) {
    addIndividualMarkers(mapboxgl, map, data.markers, data.currentSlug, data.properties)
  }

  map.once('load', () => {
    if (!isCurrentState(container, state)) return
    applyMonochromeMapPalette(map)
    if (clustered) {
      addClusteredMarkers(mapboxgl, map, data.markers, data.currentSlug, data.properties, () =>
        isCurrentState(container, state),
      )
    }
    if (!data.config.defaultCenter && data.markers.length > 1) {
      map.fitBounds(bounds, { padding: { top: 50, bottom: 50, left: 50, right: 50 }, maxZoom: 15 })
    }
  })
}

function initBaseMaps(): void {
  for (const [container, state] of mapStates) {
    if (!container.isConnected) disposeMap(container, state)
  }

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

function scheduleBaseMaps(): void {
  if (initializationTimer !== undefined) return
  initializationTimer = window.setTimeout(() => {
    initializationTimer = undefined
    initBaseMaps()
  }, 100)
}

function cleanupBaseMaps(): void {
  if (initializationTimer !== undefined) {
    window.clearTimeout(initializationTimer)
    initializationTimer = undefined
  }
  for (const [container, state] of mapStates) disposeMap(container, state)
}

document.addEventListener('nav', () => {
  scheduleBaseMaps()
  window.addCleanup(cleanupBaseMaps)
})
