import type { TriMapStyle, TriMapTheme } from '../runtime/preferences'
import type { TriathlonMapboxMap } from './mapbox'

export const applyMapTerrain = (
  map: TriathlonMapboxMap,
  enabled: boolean,
  style: TriMapStyle,
  theme: TriMapTheme,
): void => {
  if (!enabled) {
    map.setTerrain(null)
    if (map.getLayer('tri-3d-buildings')) map.removeLayer('tri-3d-buildings')
    if (map.getSource('tri-terrain')) map.removeSource('tri-terrain')
    return
  }
  if (!map.getSource('tri-terrain'))
    map.addSource('tri-terrain', {
      type: 'raster-dem',
      url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
      tileSize: 512,
      maxzoom: 14,
    })
  map.setTerrain({ source: 'tri-terrain', exaggeration: 1.25 })
  if (map.getLayer('tri-3d-buildings')) return
  const beforeId = map.getStyle()?.layers?.find(layer => layer.type === 'symbol')?.id
  map.addLayer(
    {
      id: 'tri-3d-buildings',
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
