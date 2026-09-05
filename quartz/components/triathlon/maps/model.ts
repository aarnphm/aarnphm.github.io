import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { StravaMapPoint } from '../../../plugins/stores/strava'
import type { ActivitySelectionSummary } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { StreetMapActivity } from '../../scripts/triathlon-map-heat'
import type { StreetMetricValues } from '../../scripts/triathlon-map-heat'
import type { DetailPayload } from '../activity/data'
import { none, type Cmd } from '../../../functional'
import { ROUTE_SPORTS, type ActivityKind } from '../../../plugins/stores/strava'
import { activityAverageCadence } from '../../../util/triathlon-card'
import { activityCadenceUnit } from '../../../util/triathlon-card'
import { clock } from '../../../util/triathlon-card'
import { KM_TO_MI } from '../../../util/triathlon-card'
import { powerViewActivity } from '../../../util/triathlon-card'
import { clipMapRoute } from '../../../util/triathlon-map-route'
import { isRecord } from '../../../util/type-guards'
import { analysisRouteIndex } from '../activity/analysis'
import { CAD_RAMP, HEAT_RAMP, HR_RAMP, SPD_RAMP } from './palette'

export type GeoFC = { type: 'FeatureCollection'; features: unknown[] }

export const emptyFC = (): GeoFC => ({ type: 'FeatureCollection', features: [] })

export type GeoCoord = [number, number]

export const gpsSegments = (d: StravaActivityDetail): readonly (readonly StravaMapPoint[])[] =>
  d.mapRoute.some(segment => segment.length >= 2)
    ? d.mapRoute.filter(segment => segment.length >= 2)
    : d.route.length >= 2
      ? [d.route]
      : []

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

export type OverviewMode = 'heat' | 'w' | 'hr' | 'cad' | 'spd'

export const readOverviewMode = (value: string | undefined): OverviewMode | null =>
  value === 'heat' || value === 'w' || value === 'hr' || value === 'cad' || value === 'spd'
    ? value
    : null

export const readRouteSport = (value: string | undefined): ActivityKind | null =>
  value === 'run' || value === 'bike' || value === 'swim' || value === 'walk' ? value : null

export interface OverviewLegend {
  lo: string
  hi: string
}

export interface Overview {
  streetActivities: StreetMapActivity[]
  maximumVisits: number
  traces: GeoFC
  legend: Record<OverviewMode, OverviewLegend | null>
}

export interface OverviewDrawOptions {
  fit?: boolean
  redrawStreetMap?: boolean
}

export const OVERVIEW_METRICS = ['w', 'hr', 'cad', 'spd'] as const

export const OVERVIEW_CELL = 0.0008

export const overviewCellKey = (lng: number, lat: number): string =>
  `${Math.round(lng / OVERVIEW_CELL)},${Math.round(lat / OVERVIEW_CELL)}`

export const stampSegment = (a: StravaMapPoint, b: StravaMapPoint, into: Set<string>) => {
  const steps = Math.max(
    1,
    Math.ceil(Math.max(Math.abs(b.lng - a.lng), Math.abs(b.lat - a.lat)) / (OVERVIEW_CELL / 2)),
  )
  for (let s = 0; s <= steps; s++) {
    const t = s / steps
    into.add(overviewCellKey(a.lng + (b.lng - a.lng) * t, a.lat + (b.lat - a.lat) * t))
  }
}

export const pctRange = (vals: number[]): [number, number] => {
  const sorted = [...vals].sort((x, y) => x - y)
  const lo = sorted[Math.floor(0.1 * (sorted.length - 1))]
  const hi = sorted[Math.ceil(0.9 * (sorted.length - 1))]
  return hi > lo ? [lo, hi] : [lo, lo + 1]
}

export const overviewMetric = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  k: (typeof OVERVIEW_METRICS)[number],
) => {
  const d = powerViewActivity(presentation, activity)
  if (k === 'w') return d.deviceWatts && d.avgWatts ? d.avgWatts : null
  if (k === 'hr') return d.avgHr
  if (k === 'cad') return activityAverageCadence(d)
  return d.movingTimeS > 0 ? d.distanceKm / (d.movingTimeS / 3600) : null
}

export const overviewFmt = (
  k: (typeof OVERVIEW_METRICS)[number],
  sport: ActivityKind,
  v: number,
): string => {
  if (k === 'w') return `${Math.round(v)} W`
  if (k === 'hr') return `${Math.round(v)} bpm`
  if (k === 'cad') return `${Math.round(v)} ${activityCadenceUnit(sport)}`
  if (sport === 'bike') return `${(v * KM_TO_MI).toFixed(1)} mph`
  if (sport === 'swim') return `${clock(360 / v)} /100m`
  return `${clock(3600 / (v * KM_TO_MI))} /mi`
}

export const normalizedOverviewMetrics = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  ranges: ReadonlyMap<string, [number, number]>,
): StreetMetricValues => {
  const normalized = (metric: (typeof OVERVIEW_METRICS)[number]): number => {
    const value = overviewMetric(presentation, activity, metric)
    const range = ranges.get(`${metric}:${activity.sport}`)
    return value != null && value > 0 && range
      ? Math.min(1, Math.max(0, (value - range[0]) / (range[1] - range[0])))
      : -1
  }
  return {
    w: normalized('w'),
    hr: normalized('hr'),
    cad: normalized('cad'),
    spd: normalized('spd'),
  }
}

export const buildOverview = (
  presentation: TriathlonPresentation,
  dp: DetailPayload | null,
  enabled: ReadonlySet<ActivityKind>,
): Overview => {
  const acts: StravaActivityDetail[] = []
  const det = dp?.details ?? {}
  for (const k in det) {
    const d = det[k]
    if (enabled.has(d.sport) && gpsSegments(d).length > 0) acts.push(d)
  }
  const counts = new Map<string, number>()
  for (const d of acts) {
    if (d.sport === 'swim') continue
    const cells = new Set<string>()
    for (const r of gpsSegments(d))
      for (let i = 0; i < r.length - 1; i++) stampSegment(r[i], r[i + 1], cells)
    for (const c of cells) counts.set(c, (counts.get(c) ?? 0) + 1)
  }
  let maxCount = 1
  for (const c of counts.values()) if (c > maxCount) maxCount = c
  const ranges = new Map<string, [number, number]>()
  for (const k of OVERVIEW_METRICS)
    for (const sport of ROUTE_SPORTS) {
      const vals: number[] = []
      for (const d of acts)
        if (d.sport === sport) {
          const v = overviewMetric(presentation, d, k)
          if (v != null && v > 0) vals.push(v)
        }
      if (vals.length) ranges.set(`${k}:${sport}`, pctRange(vals))
    }
  const traceFeatures: unknown[] = []
  const streetActivities: StreetMapActivity[] = []
  for (const d of acts) {
    const segments = gpsSegments(d)
    const metrics = normalizedOverviewMetrics(presentation, d, ranges)
    if (d.sport !== 'swim') streetActivities.push({ id: d.id, segments, metrics })
    traceFeatures.push(...segmentFeatures(segments, { id: d.id, sport: d.sport, ...metrics }))
  }
  const legend: Record<OverviewMode, OverviewLegend | null> = {
    heat: { lo: '$1\\times$', hi: `$${maxCount}\\times$` },
    w: null,
    hr: null,
    cad: null,
    spd: null,
  }
  for (const k of OVERVIEW_METRICS) {
    const present = ROUTE_SPORTS.filter(s => ranges.has(`${k}:${s}`))
    if (present.length === 1) {
      const range = ranges.get(`${k}:${present[0]}`)
      if (!range) continue
      const [lo, hi] = range
      legend[k] = { lo: overviewFmt(k, present[0], lo), hi: overviewFmt(k, present[0], hi) }
    }
  }
  return {
    streetActivities,
    maximumVisits: maxCount,
    traces: { type: 'FeatureCollection', features: traceFeatures },
    legend,
  }
}

export const heatColorExpr: unknown[] = (() => {
  const e: unknown[] = ['interpolate', ['linear'], ['get', 'heat']]
  HEAT_RAMP.forEach((c, i) => e.push(i + 1, c))
  return e
})()

export const heatOpacityExpr: unknown[] = [
  'interpolate',
  ['linear'],
  ['get', 'heat'],
  1,
  0.5,
  7,
  0.95,
]

const heatWidthExpression = (casing: number): unknown[] => {
  const w = (base: number, k: number) => ['+', base + casing, ['*', k, ['-', ['get', 'heat'], 1]]]
  return ['interpolate', ['linear'], ['zoom'], 10, w(0.55, 0.08), 14, w(0.9, 0.14), 16, w(1.3, 0.2)]
}

export const heatWidthExpr = heatWidthExpression(0)
export const heatCasingWidthExpr = heatWidthExpression(2.4)

export const overviewRamp = (m: OverviewMode): string[] =>
  m === 'hr' ? HR_RAMP : m === 'cad' ? CAD_RAMP : m === 'spd' ? SPD_RAMP : HEAT_RAMP

export const streetMetricColorExpr = (k: OverviewMode): unknown[] => {
  const ramp: unknown[] = ['interpolate', ['linear'], ['get', k]]
  overviewRamp(k).forEach((c, i) => ramp.push(i / 6, c))
  return ['case', ['<', ['get', k], 0], '#b7b3ac', ramp]
}

export const streetMetricOpacityExpr = (k: OverviewMode): unknown[] => [
  'case',
  ['<', ['get', k], 0],
  0.12,
  0.6,
]

const streetMetricWidthExpression = (casing: number): unknown[] => [
  'interpolate',
  ['linear'],
  ['zoom'],
  10,
  0.75 + casing,
  14,
  1.2 + casing,
  16,
  1.8 + casing,
]

export const streetMetricWidthExpr = streetMetricWidthExpression(0)
export const streetMetricCasingWidthExpr = streetMetricWidthExpression(2.4)

export const routeFC = (d: StravaActivityDetail): GeoFC => ({
  type: 'FeatureCollection',
  features: segmentFeatures(gpsSegments(d)),
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
  if (d.route.length === 0) return emptyFC()
  const pick = spec.pick
  const vals = d.route.map((p, i) => pick(p, i))
  const pool = vals.filter((value, index) =>
    spec.valid && !spec.valid(d.route[index], index) ? false : !spec.zeroGap || value > 0,
  )
  let lo = Infinity
  let hi = -Infinity
  for (const v of pool.length ? pool : vals) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
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
      const value = metricValueAtDistance(d, pick, (a.d + b.d) / 2)
      const routeIndex = analysisRouteIndex(d.route, (a.d + b.d) / 2)
      const valid = spec.valid?.(d.route[routeIndex], routeIndex) ?? true
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

export const fcBounds = (fc: GeoFC): [[number, number], [number, number]] | null => {
  let minLng = Infinity
  let minLat = Infinity
  let maxLng = -Infinity
  let maxLat = -Infinity
  for (const f of fc.features) {
    if (!isRecord(f) || !isRecord(f.geometry) || !Array.isArray(f.geometry.coordinates)) continue
    const coords: GeoCoord[] = []
    for (const coordinate of f.geometry.coordinates) {
      if (
        Array.isArray(coordinate) &&
        coordinate.length >= 2 &&
        typeof coordinate[0] === 'number' &&
        typeof coordinate[1] === 'number'
      )
        coords.push([coordinate[0], coordinate[1]])
    }
    for (const [lng, lat] of coords) {
      if (lng < minLng) minLng = lng
      if (lng > maxLng) maxLng = lng
      if (lat < minLat) minLat = lat
      if (lat > maxLat) maxLat = lat
    }
  }
  return Number.isFinite(minLng)
    ? [
        [minLng, minLat],
        [maxLng, maxLat],
      ]
    : null
}

export type MapLoadStatus = 'idle' | 'loading' | 'ready' | 'failed'
export type MapStyle = 'mono' | 'streets' | 'satellite'

export interface MapModel {
  status: MapLoadStatus
  request: number
  mode: OverviewMode
  enabledSports: ReadonlySet<ActivityKind>
  selectedRouteId: string | null
  metric: number
  analysisRange: MapAnalysisRange | null
  style: MapStyle
}

export type MapMessage =
  | { type: 'load' }
  | { type: 'loaded'; request: number }
  | { type: 'failed'; request: number }
  | { type: 'set-mode'; mode: OverviewMode }
  | { type: 'toggle-sport'; sport: ActivityKind }
  | { type: 'set-sports'; sports: ReadonlySet<ActivityKind> }
  | { type: 'select-route'; id: string; metric?: number }
  | { type: 'select-metric'; metric: number }
  | { type: 'select-range'; range: MapAnalysisRange | null }
  | { type: 'clear-route' }
  | { type: 'set-style'; style: MapStyle }
  | { type: 'reset' }

export type MapEffect =
  | { type: 'load-artifacts'; request: number }
  | { type: 'draw-overview'; options?: OverviewDrawOptions }
  | { type: 'draw-route'; id: string; metric: number }
  | { type: 'apply-metric'; id: string; metric: number }
  | { type: 'draw-range'; range: MapAnalysisRange | null }
  | { type: 'apply-mode'; mode: OverviewMode }
  | { type: 'apply-style'; style: MapStyle }

export const initialMapModel = (style: MapStyle = 'mono'): MapModel => ({
  status: 'idle',
  request: 0,
  mode: 'heat',
  enabledSports: new Set(ROUTE_SPORTS),
  selectedRouteId: null,
  metric: 0,
  analysisRange: null,
  style,
})

export const updateMap = (
  model: MapModel,
  message: MapMessage,
): { model: MapModel; effects: Cmd<MapEffect> } => {
  switch (message.type) {
    case 'load': {
      const request = model.request + 1
      return {
        model: { ...model, status: 'loading', request },
        effects: [{ type: 'load-artifacts', request }],
      }
    }
    case 'loaded':
      return message.request === model.request
        ? { model: { ...model, status: 'ready' }, effects: [{ type: 'draw-overview' }] }
        : { model, effects: none() }
    case 'failed':
      return message.request === model.request
        ? { model: { ...model, status: 'failed' }, effects: none() }
        : { model, effects: none() }
    case 'set-mode':
      return {
        model: { ...model, mode: message.mode },
        effects: [{ type: 'apply-mode', mode: message.mode }],
      }
    case 'toggle-sport': {
      const enabledSports = new Set(model.enabledSports)
      if (enabledSports.has(message.sport)) enabledSports.delete(message.sport)
      else enabledSports.add(message.sport)
      return {
        model: { ...model, enabledSports },
        effects: [
          {
            type: 'draw-overview',
            options: { fit: false, redrawStreetMap: message.sport !== 'swim' },
          },
        ],
      }
    }
    case 'set-sports': {
      if (
        message.sports.size === model.enabledSports.size &&
        [...message.sports].every(sport => model.enabledSports.has(sport))
      )
        return { model, effects: none() }
      return {
        model: { ...model, enabledSports: new Set(message.sports) },
        effects: [{ type: 'draw-overview' }],
      }
    }
    case 'select-route': {
      const metric = message.metric ?? 0
      return {
        model: { ...model, selectedRouteId: message.id, metric, analysisRange: null },
        effects: [{ type: 'draw-route', id: message.id, metric }],
      }
    }
    case 'select-metric':
      return model.selectedRouteId
        ? {
            model: { ...model, metric: message.metric },
            effects: [{ type: 'apply-metric', id: model.selectedRouteId, metric: message.metric }],
          }
        : { model: { ...model, metric: message.metric }, effects: none() }
    case 'select-range':
      return {
        model: { ...model, analysisRange: message.range },
        effects: [{ type: 'draw-range', range: message.range }],
      }
    case 'clear-route':
      return {
        model: { ...model, selectedRouteId: null, metric: 0, analysisRange: null },
        effects: [{ type: 'draw-overview', options: { fit: false } }],
      }
    case 'set-style':
      return {
        model: { ...model, style: message.style },
        effects: [{ type: 'apply-style', style: message.style }],
      }
    case 'reset':
      return {
        model: {
          ...model,
          mode: 'heat',
          enabledSports: new Set(ROUTE_SPORTS),
          selectedRouteId: null,
          metric: 0,
          analysisRange: null,
        },
        effects: [{ type: 'draw-overview' }],
      }
  }
}
