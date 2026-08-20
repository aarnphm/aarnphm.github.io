import { loadMapbox } from '../../scripts/mapbox-client'

export interface MapboxFeature {
  properties?: Record<string, unknown>
}

export interface MapboxLayerEvent {
  features?: MapboxFeature[]
  point: { x: number; y: number }
}

export interface MapboxLayer {
  id: string
  type: string
  'source-layer'?: string
}

export interface MapboxSource {
  setData(data: unknown): void
}

export interface TriathlonMapboxMap {
  addLayer(layer: Record<string, unknown>, beforeId?: string): void
  addSource(id: string, source: Record<string, unknown>): void
  easeTo(options: { pitch: number; duration: number }): void
  fitBounds(
    bounds: [[number, number], [number, number]],
    options: {
      padding: number | { top: number; right: number; bottom: number; left: number }
      maxZoom: number
      duration: number
    },
  ): void
  getCanvas(): HTMLCanvasElement
  getLayer(id: string): unknown
  getSource(id: string): MapboxSource | undefined
  getStyle(): { layers?: MapboxLayer[] } | undefined
  on(type: 'mousemove' | 'click', layer: string, listener: (event: MapboxLayerEvent) => void): void
  on(type: 'mouseleave', layer: string, listener: () => void): void
  on(type: 'moveend', listener: () => void): void
  once(type: 'idle' | 'load' | 'style.load', listener: () => void): void
  queryRenderedFeatures(options: { layers: string[] }): MapboxFeature[]
  removeLayer(id: string): void
  removeSource(id: string): void
  remove(): void
  resize(): void
  setPaintProperty(layer: string, property: string, value: unknown): void
  setStyle(style: string): void
  setTerrain(terrain: { source: string; exaggeration: number } | null): void
}

interface MapboxLibrary {
  Map: new (options: {
    container: HTMLElement
    style: string
    center: [number, number]
    zoom: number
    attributionControl: boolean
    pitch: number
  }) => TriathlonMapboxMap
}

export const createMapboxMap = async (
  container: HTMLElement,
  style: string,
  threeDimensional: boolean,
): Promise<TriathlonMapboxMap | null> => {
  const library: MapboxLibrary | null = await loadMapbox()
  if (!library) return null
  return new library.Map({
    container,
    style,
    center: [-79.4, 43.7],
    zoom: 9,
    attributionControl: false,
    pitch: threeDimensional ? 55 : 0,
  })
}
