import type { StravaActivityDetail, StravaMapPoint } from '../../../../plugins/stores/strava'
import type { ActivitySelectionSummary } from '../../../../util/triathlon-card'
import { activityGpsSegments, clipMapRoute } from '../../../../util/triathlon-map-route'
import { analysisRouteIndex } from '../../activity/analysis'
import { HEAT_RAMP } from '../palette'

export type GeoFC = { type: 'FeatureCollection'; features: unknown[] }

export const emptyFC = (): GeoFC => ({ type: 'FeatureCollection', features: [] })

export type GeoCoord = [number, number]

export const gpsSegments = activityGpsSegments

export const lineFeatures = (
  route: readonly StravaMapPoint[],
  props: Record<string, unknown> = {},
) =>
  route.length >= 2
    ? [
        {
          type: 'Feature',
          properties: props,
          geometry: { type: 'LineString', coordinates: route.map<GeoCoord>(p => [p.lng, p.lat]) },
        },
      ]
    : []

export const segmentFeatures = (
  segments: readonly (readonly StravaMapPoint[])[],
  props: Record<string, unknown> = {},
) => segments.flatMap(segment => lineFeatures(segment, props))

export const routeFC = (d: StravaActivityDetail, color?: string): GeoFC => ({
  type: 'FeatureCollection',
  features: segmentFeatures(gpsSegments(d), color ? { color } : {}),
})

export interface MapAnalysisRange extends ActivitySelectionSummary {
  kind: 'lap' | 'segment' | 'climb' | null
  id: string | null
  label: string
}

export const rangeFC = (d: StravaActivityDetail, range: MapAnalysisRange): GeoFC => {
  return {
    type: 'FeatureCollection',
    features: segmentFeatures(
      clipMapRoute(gpsSegments(d), range.startDistanceKm, range.endDistanceKm),
    ),
  }
}

export const sameAnalysisRange = (
  left: MapAnalysisRange | null,
  right: MapAnalysisRange | null,
): boolean =>
  left === right ||
  (left != null &&
    right != null &&
    left.startDistanceKm === right.startDistanceKm &&
    left.endDistanceKm === right.endDistanceKm)

export const pointFC = (lng: number, lat: number): GeoFC => ({
  type: 'FeatureCollection',
  features: [
    { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [lng, lat] } },
  ],
})

export const metricValueAtDistance = (
  d: StravaActivityDetail,
  pick: (p: StravaActivityDetail['route'][number], i: number) => number,
  distanceKm: number,
): number => {
  const index = analysisRouteIndex(d.route, distanceKm)
  const point = d.route[index]
  if (!point) return 0
  const otherIndex =
    point.d <= distanceKm ? Math.min(d.route.length - 1, index + 1) : Math.max(0, index - 1)
  const other = d.route[otherIndex]
  if (!other || other.d === point.d) return pick(point, index)
  const fraction = (distanceKm - point.d) / (other.d - point.d)
  return pick(point, index) + (pick(other, otherIndex) - pick(point, index)) * fraction
}

export const metricRouteFC = (
  d: StravaActivityDetail,
  spec: {
    pick: (point: StravaActivityDetail['route'][number], index: number) => number
    ramp: readonly string[]
    zeroGap?: boolean
    valid?: (point: StravaActivityDetail['route'][number], index: number) => boolean
  },
): GeoFC => {
  const route = d.route
  const routeLength = route.length
  const fallbackColor = spec.ramp[6] ?? HEAT_RAMP[6]
  if (routeLength === 0) return routeFC(d, fallbackColor)
  const pick = spec.pick
  const vals = new Float64Array(routeLength)
  const validFlags = spec.valid ? new Uint8Array(routeLength) : null
  let lo = Infinity
  let hi = -Infinity
  let poolCount = 0
  for (let i = 0; i < routeLength; i++) {
    const p = route[i]
    const v = pick(p, i)
    vals[i] = v
    const isValid = validFlags ? (validFlags[i] = spec.valid!(p, i) ? 1 : 0) === 1 : true
    if (isValid && (!spec.zeroGap || v > 0)) {
      poolCount++
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  if (poolCount === 0) return routeFC(d, fallbackColor)
  const range = hi > lo ? hi - lo : 1
  const features: unknown[] = []
  for (const segment of gpsSegments(d)) {
    let runColor = ''
    let coordinates: GeoCoord[] = []
    const flush = (): void => {
      if (coordinates.length >= 2)
        features.push({
          type: 'Feature',
          properties: { color: runColor },
          geometry: { type: 'LineString', coordinates },
        })
      coordinates = []
    }
    for (let index = 1; index < segment.length; index++) {
      const a = segment[index - 1]
      const b = segment[index]
      const distanceKm = (a.d + b.d) * 0.5
      const routeIndex = analysisRouteIndex(route, distanceKm)
      const point = route[routeIndex]
      let value = vals[routeIndex]
      if (point) {
        const otherIndex =
          point.d <= distanceKm
            ? Math.min(routeLength - 1, routeIndex + 1)
            : Math.max(0, routeIndex - 1)
        const other = route[otherIndex]
        if (other && other.d !== point.d) {
          const fraction = (distanceKm - point.d) / (other.d - point.d)
          value = value + (vals[otherIndex] - value) * fraction
        }
      }
      const valid = validFlags ? validFlags[routeIndex] === 1 : true
      const bucket = Math.min(6, Math.max(0, Math.floor(((value - lo) / range) * 7)))
      const color = !valid || (spec.zeroGap && value <= 0) ? 'rgba(0,0,0,0)' : spec.ramp[bucket]
      const start: GeoCoord = [a.lng, a.lat]
      const end: GeoCoord = [b.lng, b.lat]
      if (color !== runColor) {
        flush()
        runColor = color
        coordinates = [start]
      } else if (coordinates.length === 0) coordinates.push(start)
      coordinates.push(end)
    }
    flush()
  }
  return { type: 'FeatureCollection', features }
}
