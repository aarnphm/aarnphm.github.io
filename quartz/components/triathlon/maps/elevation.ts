import type { DistanceSystem } from '../../../util/triathlon-presentation'
import type { TriMapStyle, TriMapTheme } from '../runtime/preferences'
import type { TriathlonMapboxMap } from './mapbox'
import { M_TO_FT } from '../../../util/triathlon-card'

const sourceId = 'tri-elevation'
const hillshadeId = 'tri-elevation-hillshade'
const contourId = 'tri-elevation-contours'
const labelId = 'tri-elevation-labels'

export const applyMapElevation = (
  map: TriathlonMapboxMap,
  enabled: boolean,
  style: TriMapStyle,
  theme: TriMapTheme,
  distance: DistanceSystem,
): void => {
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
    layer => layer.id === 'tri-heat-casing' || layer.type === 'symbol',
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
