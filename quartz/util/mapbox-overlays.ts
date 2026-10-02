import type { MapboxStyle, MapboxTheme } from './mapbox-style'
import { M_TO_FT } from './triathlon-card'

export interface MapboxOverlayMap {
  addLayer(layer: Record<string, unknown>, beforeId?: string): void
  addSource(id: string, source: Record<string, unknown>): void
  getLayer(id: string): unknown
  getSource(id: string): unknown
  getStyle(): { layers?: { id: string; type: string }[] } | undefined
  removeLayer(id: string): void
  removeSource(id: string): void
  setLayoutProperty(layer: string, property: string, value: unknown): void
  setTerrain(terrain: { source: string; exaggeration: number } | null): void
}

export const applyMapTerrain = (
  map: MapboxOverlayMap,
  enabled: boolean,
  style: MapboxStyle,
  theme: MapboxTheme,
  prefix = 'tri',
): void => {
  if (!enabled) {
    map.setTerrain(null)
    if (map.getLayer(`${prefix}-3d-buildings`)) map.removeLayer(`${prefix}-3d-buildings`)
    if (map.getSource(`${prefix}-terrain`)) map.removeSource(`${prefix}-terrain`)
    return
  }
  if (!map.getSource(`${prefix}-terrain`))
    map.addSource(`${prefix}-terrain`, {
      type: 'raster-dem',
      url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
      tileSize: 512,
      maxzoom: 14,
    })
  map.setTerrain({ source: `${prefix}-terrain`, exaggeration: 1.25 })
  if (map.getLayer(`${prefix}-3d-buildings`)) return
  const beforeId = map.getStyle()?.layers?.find(layer => layer.type === 'symbol')?.id
  map.addLayer(
    {
      id: `${prefix}-3d-buildings`,
      source: 'composite',
      'source-layer': 'building',
      filter: ['==', ['get', 'extrude'], 'true'],
      type: 'fill-extrusion',
      minzoom: 14.5,
      paint: {
        'fill-extrusion-color':
          style === 'satellite' ? '#b8b5af' : theme === 'dark' ? '#34312d' : '#d8cec1',
        'fill-extrusion-height': [
          'interpolate',
          ['linear'],
          ['zoom'],
          14.5,
          0,
          14.75,
          ['coalesce', ['get', 'height'], 3],
        ],
        'fill-extrusion-base': [
          'interpolate',
          ['linear'],
          ['zoom'],
          14.5,
          0,
          14.75,
          ['coalesce', ['get', 'min_height'], 0],
        ],
        'fill-extrusion-opacity': 0.72,
        'fill-extrusion-vertical-gradient': true,
      },
    },
    beforeId,
  )
}

export const applyMapElevation = (
  map: MapboxOverlayMap,
  enabled: boolean,
  style: MapboxStyle,
  theme: MapboxTheme,
  distance: 'metric' | 'imperial',
  prefix = 'tri',
): void => {
  const sourceId = `${prefix}-elevation`
  const hillshadeId = `${prefix}-elevation-hillshade`
  const contourId = `${prefix}-elevation-contours`
  const labelId = `${prefix}-elevation-labels`
  if (!enabled) {
    for (const id of [labelId, contourId, hillshadeId]) {
      if (map.getLayer(id)) map.removeLayer(id)
    }
    if (map.getSource(sourceId)) map.removeSource(sourceId)
    return
  }

  if (!map.getSource(sourceId)) {
    map.addSource(sourceId, { type: 'vector', url: 'mapbox://mapbox.mapbox-terrain-v2' })
  }
  const layers = map.getStyle()?.layers ?? []
  const firstRoad = layers.find(layer =>
    ['line', 'symbol', 'fill-extrusion'].includes(layer.type),
  )?.id
  const firstRouteOrLabel = layers.find(
    layer => layer.id === `${prefix}-heat-casing` || layer.type === 'symbol',
  )?.id
  const satellite = style === 'satellite'
  const dark = theme === 'dark'
  const color = satellite ? '#fff9f3' : dark ? '#b7a58b' : '#85745c'
  const halo = satellite ? '#242820' : dark ? '#100f0f' : '#fff9f3'

  if (!map.getLayer(hillshadeId)) {
    map.addLayer(
      {
        id: hillshadeId,
        type: 'fill',
        source: sourceId,
        'source-layer': 'hillshade',
        paint: {
          'fill-color': ['match', ['get', 'class'], 'highlight', '#ffffff', '#000000'],
          'fill-opacity': ['interpolate', ['linear'], ['zoom'], 13, 0.16, 17, 0],
          'fill-antialias': false,
        },
      },
      firstRoad,
    )
  }
  if (!map.getLayer(contourId)) {
    map.addLayer(
      {
        id: contourId,
        type: 'line',
        source: sourceId,
        'source-layer': 'contour',
        minzoom: 9,
        layout: { 'line-join': 'round' },
        paint: {
          'line-color': color,
          'line-opacity': satellite ? 0.65 : 0.5,
          'line-width': ['case', ['>=', ['get', 'index'], 5], 1, 0.5],
        },
      },
      firstRouteOrLabel,
    )
  }
  const textField = [
    'concat',
    ['to-string', ['round', ['*', ['get', 'ele'], distance === 'imperial' ? M_TO_FT : 1]]],
    distance === 'imperial' ? ' ft' : ' m',
  ]
  if (!map.getLayer(labelId)) {
    map.addLayer(
      {
        id: labelId,
        type: 'symbol',
        source: sourceId,
        'source-layer': 'contour',
        minzoom: 11,
        layout: {
          'symbol-placement': 'line',
          'symbol-spacing': 350,
          'text-field': textField,
          'text-font': ['DIN Offc Pro Regular', 'Arial Unicode MS Regular'],
          'text-size': 10,
          'text-padding': 4,
        },
        paint: { 'text-color': color, 'text-halo-color': halo, 'text-halo-width': 1.5 },
      },
      firstRouteOrLabel,
    )
  } else {
    map.setLayoutProperty(labelId, 'text-field', textField)
  }
}
