import type { StravaActivityDetail, StravaMapPoint } from '../../../plugins/stores/strava'
import type { TriathlonMapboxMap } from '../maps/mapbox'
import type { GeoFC } from '../maps/model'
import { activityCompareColor } from '../../../util/triathlon-card'
import { mapRoutePointAtDistance } from '../../../util/triathlon-map-route'
import { applyMonochromeMapPalette } from '../../scripts/mapbox-client'
import { createMapboxMap, type MapboxPointerEvent } from '../maps/mapbox'
import { emptyFC, fcBounds, gpsSegments, segmentFeatures } from '../maps/model'
import {
  mapboxStyleUrl,
  readTriMapStyle,
  readTriMapTheme,
  TRI_MAP_STYLE_EVENT,
} from '../runtime/preferences'

export interface ActivityComparisonMapController {
  setVisible(activityId: string, visible: boolean): void
  showCursors(distanceKm: number): void
  hideCursors(): void
  destroy(): void
}

interface ComparisonRoute {
  id: string
  color: string
  dash: readonly number[] | null
  segments: readonly (readonly StravaMapPoint[])[]
  startKm: number
  endKm: number
}

const ROUTE_DASHES: readonly (readonly number[] | null)[] = [
  null,
  [2.2, 1.1],
  [0.5, 1.1],
  [3.2, 0.9, 0.5, 0.9],
]

const routeSourceId = (route: ComparisonRoute): string => `tri-cmp-${route.id}`

const routeFeatures = (route: ComparisonRoute): GeoFC => ({
  type: 'FeatureCollection',
  features: segmentFeatures(route.segments),
})

export const mountActivityComparisonMap = (
  container: HTMLElement,
  activities: readonly StravaActivityDetail[],
  options: { unavailableText: string; onScrub: (distanceKm: number) => void; onLeave: () => void },
): ActivityComparisonMapController => {
  const routes: ComparisonRoute[] = activities.flatMap((activity, index) => {
    const segments = gpsSegments(activity)
    if (segments.length === 0) return []
    const distances = segments.flatMap(segment => segment.map(point => point.d))
    return [
      {
        id: `${activity.id}`,
        color: activityCompareColor(index),
        dash: ROUTE_DASHES[index % ROUTE_DASHES.length],
        segments,
        startKm: Math.min(...distances),
        endKm: Math.max(...distances),
      },
    ]
  })
  const hidden = new Set<string>()
  let map: TriathlonMapboxMap | null = null
  let ready = false
  let disposed = false
  let styleSequence = 0
  let cursorDistanceKm: number | null = null
  let scrubFrame = 0
  let pendingScrub: MapboxPointerEvent | null = null
  let pendingFit = false

  const casingColor = (): string =>
    readTriMapStyle() === 'satellite'
      ? '#fff9f3'
      : readTriMapTheme() === 'dark'
        ? '#100f0f'
        : '#fff9f3'

  const cursorFeatures = (): GeoFC => {
    const distanceKm = cursorDistanceKm
    if (distanceKm == null) return emptyFC()
    return {
      type: 'FeatureCollection',
      features: routes.flatMap(route => {
        if (hidden.has(route.id) || distanceKm < route.startKm || distanceKm > route.endKm)
          return []
        const point = mapRoutePointAtDistance(route.segments, distanceKm)
        return point
          ? [
              {
                type: 'Feature',
                properties: { color: route.color },
                geometry: { type: 'Point', coordinates: [point.lng, point.lat] },
              },
            ]
          : []
      }),
    }
  }

  const drawCursors = (): void => {
    if (ready) map?.getSource('tri-cmp-dot')?.setData(cursorFeatures())
  }

  const installLayers = (): void => {
    const current = map
    if (!current) return
    const casing = casingColor()
    if (readTriMapStyle() === 'mono') applyMonochromeMapPalette(current, readTriMapTheme())
    for (const route of routes) {
      const sourceId = routeSourceId(route)
      if (!current.getSource(sourceId))
        current.addSource(sourceId, { type: 'geojson', data: routeFeatures(route) })
      if (!current.getLayer(`${sourceId}-casing`))
        current.addLayer({
          id: `${sourceId}-casing`,
          type: 'line',
          source: sourceId,
          layout: {
            'line-cap': 'round',
            'line-join': 'round',
            visibility: hidden.has(route.id) ? 'none' : 'visible',
          },
          paint: { 'line-color': casing, 'line-width': 4.6, 'line-opacity': 0.6 },
        })
      if (!current.getLayer(sourceId))
        current.addLayer({
          id: sourceId,
          type: 'line',
          source: sourceId,
          layout: {
            'line-cap': route.dash ? 'butt' : 'round',
            'line-join': 'round',
            visibility: hidden.has(route.id) ? 'none' : 'visible',
            ...(route.dash ? { 'line-dasharray': route.dash } : {}),
          },
          paint: { 'line-color': route.color, 'line-width': 2.4 },
        })
    }
    if (!current.getSource('tri-cmp-dot'))
      current.addSource('tri-cmp-dot', { type: 'geojson', data: emptyFC() })
    if (!current.getLayer('tri-cmp-dot'))
      current.addLayer({
        id: 'tri-cmp-dot',
        type: 'circle',
        source: 'tri-cmp-dot',
        paint: {
          'circle-radius': 4,
          'circle-color': ['get', 'color'],
          'circle-stroke-width': 1.6,
          'circle-stroke-color': casing,
        },
      })
    ready = true
    drawCursors()
  }

  const nearestDistanceKm = (event: MapboxPointerEvent): number | null => {
    const scale = Math.cos((event.lngLat.lat * Math.PI) / 180)
    let nearest: number | null = null
    let nearestSquared = Infinity
    for (const route of routes) {
      if (hidden.has(route.id)) continue
      for (const segment of route.segments)
        for (const point of segment) {
          const dx = (point.lng - event.lngLat.lng) * scale
          const dy = point.lat - event.lngLat.lat
          const squared = dx * dx + dy * dy
          if (squared >= nearestSquared) continue
          nearestSquared = squared
          nearest = point.d
        }
    }
    return nearest
  }

  const onPointerMove = (event: MapboxPointerEvent): void => {
    pendingScrub = event
    if (scrubFrame) return
    scrubFrame = window.requestAnimationFrame(() => {
      scrubFrame = 0
      const latest = pendingScrub
      pendingScrub = null
      if (!latest || disposed) return
      const distanceKm = nearestDistanceKm(latest)
      if (distanceKm != null) options.onScrub(distanceKm)
    })
  }

  const onPointerOut = (): void => {
    pendingScrub = null
    options.onLeave()
  }

  const applyStyle = (): void => {
    const current = map
    if (!current) return
    const sequence = ++styleSequence
    ready = false
    current.setStyle(mapboxStyleUrl(readTriMapStyle(), readTriMapTheme()))
    current.once('style.load', () => {
      if (disposed || sequence !== styleSequence) return
      installLayers()
    })
  }

  const fitRoutes = (): void => {
    const current = map
    const bounds = fcBounds({
      type: 'FeatureCollection',
      features: routes.flatMap(route => routeFeatures(route).features),
    })
    if (!current || !bounds) return
    if (container.clientWidth === 0 || container.clientHeight === 0) {
      pendingFit = true
      return
    }
    pendingFit = false
    current.fitBounds(bounds, { padding: 32, maxZoom: 15, duration: 0 })
  }

  const onThemeChange = (): void => {
    if (readTriMapStyle() !== 'satellite') applyStyle()
  }
  const observer = new ResizeObserver(() => {
    map?.resize()
    if (pendingFit) fitRoutes()
  })
  observer.observe(container)
  window.addEventListener(TRI_MAP_STYLE_EVENT, applyStyle)
  document.addEventListener('themechange', onThemeChange)

  void createMapboxMap(container, mapboxStyleUrl(readTriMapStyle(), readTriMapTheme()), false).then(
    created => {
      if (disposed) {
        created?.remove()
        return
      }
      if (!created) {
        container.classList.add('tri-compare-map--down')
        container.textContent = options.unavailableText
        return
      }
      map = created
      created.on('mousemove', onPointerMove)
      created.on('mouseout', onPointerOut)
      created.once('load', () => {
        if (disposed) return
        installLayers()
        const bounds = fcBounds({
          type: 'FeatureCollection',
          features: routes.flatMap(route => routeFeatures(route).features),
        })
        if (bounds) created.fitBounds(bounds, { padding: 32, maxZoom: 15, duration: 0 })
      })
    },
  )

  return {
    setVisible: (activityId, visible) => {
      if (visible) hidden.delete(activityId)
      else hidden.add(activityId)
      const current = map
      const route = routes.find(candidate => candidate.id === activityId)
      if (ready && current && route) {
        const sourceId = routeSourceId(route)
        const visibility = visible ? 'visible' : 'none'
        current.setLayoutProperty(sourceId, 'visibility', visibility)
        current.setLayoutProperty(`${sourceId}-casing`, 'visibility', visibility)
      }
      drawCursors()
    },
    showCursors: distanceKm => {
      cursorDistanceKm = distanceKm
      drawCursors()
    },
    hideCursors: () => {
      cursorDistanceKm = null
      drawCursors()
    },
    destroy: () => {
      disposed = true
      observer.disconnect()
      window.removeEventListener(TRI_MAP_STYLE_EVENT, applyStyle)
      document.removeEventListener('themechange', onThemeChange)
      if (scrubFrame) window.cancelAnimationFrame(scrubFrame)
      map?.remove()
      map = null
      ready = false
    },
  }
}
