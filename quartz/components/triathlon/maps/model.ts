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
import { activityGpsSegments, clipMapRoute } from '../../../util/triathlon-map-route'
import { isRecord } from '../../../util/type-guards'
import { analysisRouteIndex } from '../activity/analysis'
import { CAD_RAMP, HEAT_RAMP, HR_RAMP, SPD_RAMP } from './palette'

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
const OVERVIEW_KEY_OFFSET = 1_048_576
const OVERVIEW_KEY_SCALE = 2_097_152

export const overviewCellKey = (lng: number, lat: number): string =>
  `${Math.round(lng / OVERVIEW_CELL)},${Math.round(lat / OVERVIEW_CELL)}`

const overviewCellIntKey = (lng: number, lat: number): number =>
  (Math.round(lng / OVERVIEW_CELL) + OVERVIEW_KEY_OFFSET) * OVERVIEW_KEY_SCALE +
  (Math.round(lat / OVERVIEW_CELL) + OVERVIEW_KEY_OFFSET)

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

const stampSegmentInt = (a: StravaMapPoint, b: StravaMapPoint, into: Set<number>) => {
  const dLng = b.lng - a.lng
  const dLat = b.lat - a.lat
  const steps = Math.max(
    1,
    Math.ceil(Math.max(Math.abs(dLng), Math.abs(dLat)) / (OVERVIEW_CELL / 2)),
  )
  const invSteps = 1 / steps
  for (let s = 0; s <= steps; s++) {
    const t = s * invSteps
    into.add(overviewCellIntKey(a.lng + dLng * t, a.lat + dLat * t))
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
    if (!d.virtual && enabled.has(d.sport) && gpsSegments(d).length > 0) acts.push(d)
  }
  const counts = new Map<number, number>()
  for (const d of acts) {
    if (d.sport === 'swim') continue
    const cells = new Set<number>()
    for (const r of gpsSegments(d))
      for (let i = 0; i < r.length - 1; i++) stampSegmentInt(r[i], r[i + 1], cells)
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
  const w = (base: number, k: number) => [
    '+',
    base + casing,
    ['*', k, ['-', ['coalesce', ['get', 'heat'], 1], 1]],
  ]
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

export const fcBounds = (fc: GeoFC): [[number, number], [number, number]] | null => {
  let minLng = Infinity
  let minLat = Infinity
  let maxLng = -Infinity
  let maxLat = -Infinity
  for (const f of fc.features) {
    if (!isRecord(f) || !isRecord(f.geometry) || !Array.isArray(f.geometry.coordinates)) continue
    for (const coordinate of f.geometry.coordinates) {
      if (
        Array.isArray(coordinate) &&
        coordinate.length >= 2 &&
        typeof coordinate[0] === 'number' &&
        typeof coordinate[1] === 'number'
      ) {
        const lng = coordinate[0]
        const lat = coordinate[1]
        if (lng < minLng) minLng = lng
        if (lng > maxLng) maxLng = lng
        if (lat < minLat) minLat = lat
        if (lat > maxLat) maxLat = lat
      }
    }
  }
  return Number.isFinite(minLng)
    ? [
        [minLng, minLat],
        [maxLng, maxLat],
      ]
    : null
}

export const DEFAULT_OVERVIEW_CENTER: [number, number] = [-79.4, 43.7]

export const DEFAULT_OVERVIEW_BOUNDS: [[number, number], [number, number]] = [
  [-80.102531, 43.249998],
  [-78.292016, 44.606476],
]

export const OVERVIEW_CLUSTER_RADIUS_DEG = 1.5

interface ActivityBoundsBox {
  minLng: number
  minLat: number
  maxLng: number
  maxLat: number
}

export const overviewBounds = (fc: GeoFC): [[number, number], [number, number]] | null => {
  const byActivity = new Map<string, ActivityBoundsBox>()
  let anonymousIndex = 0
  for (const f of fc.features) {
    if (!isRecord(f) || !isRecord(f.geometry) || !Array.isArray(f.geometry.coordinates)) continue
    const rawId = isRecord(f.properties) ? f.properties.id : undefined
    const key =
      typeof rawId === 'number' || typeof rawId === 'string'
        ? String(rawId)
        : `__anon_${anonymousIndex++}`
    let box = byActivity.get(key)
    for (const coordinate of f.geometry.coordinates) {
      if (
        Array.isArray(coordinate) &&
        coordinate.length >= 2 &&
        typeof coordinate[0] === 'number' &&
        typeof coordinate[1] === 'number' &&
        Number.isFinite(coordinate[0]) &&
        Number.isFinite(coordinate[1])
      ) {
        const lng = coordinate[0]
        const lat = coordinate[1]
        if (!box) {
          box = { minLng: lng, minLat: lat, maxLng: lng, maxLat: lat }
          byActivity.set(key, box)
        } else {
          if (lng < box.minLng) box.minLng = lng
          if (lng > box.maxLng) box.maxLng = lng
          if (lat < box.minLat) box.minLat = lat
          if (lat > box.maxLat) box.maxLat = lat
        }
      }
    }
  }
  const boxes = [...byActivity.values()]
  if (boxes.length === 0) return null
  if (boxes.length === 1) {
    const b = boxes[0]
    return [
      [b.minLng, b.minLat],
      [b.maxLng, b.maxLat],
    ]
  }
  const visited = new Uint8Array(boxes.length)
  let bestCount = -1
  let bestCenterDistSq = Infinity
  let bestBounds: [[number, number], [number, number]] | null = null
  const radius = OVERVIEW_CLUSTER_RADIUS_DEG
  for (let i = 0; i < boxes.length; i++) {
    if (visited[i]) continue
    visited[i] = 1
    const queue = [i]
    let minLng = boxes[i].minLng
    let minLat = boxes[i].minLat
    let maxLng = boxes[i].maxLng
    let maxLat = boxes[i].maxLat
    for (let head = 0; head < queue.length; head++) {
      const current = boxes[queue[head]]
      if (current.minLng < minLng) minLng = current.minLng
      if (current.maxLng > maxLng) maxLng = current.maxLng
      if (current.minLat < minLat) minLat = current.minLat
      if (current.maxLat > maxLat) maxLat = current.maxLat
      for (let j = 0; j < boxes.length; j++) {
        if (visited[j]) continue
        const other = boxes[j]
        if (
          current.minLng - radius <= other.maxLng &&
          other.minLng - radius <= current.maxLng &&
          current.minLat - radius <= other.maxLat &&
          other.minLat - radius <= current.maxLat
        ) {
          visited[j] = 1
          queue.push(j)
        }
      }
    }
    const count = queue.length
    const centerLng = (minLng + maxLng) * 0.5
    const centerLat = (minLat + maxLat) * 0.5
    const dLng = centerLng - DEFAULT_OVERVIEW_CENTER[0]
    const dLat = centerLat - DEFAULT_OVERVIEW_CENTER[1]
    const centerDistSq = dLng * dLng + dLat * dLat
    if (count > bestCount || (count === bestCount && centerDistSq < bestCenterDistSq)) {
      bestCount = count
      bestCenterDistSq = centerDistSq
      bestBounds = [
        [minLng, minLat],
        [maxLng, maxLat],
      ]
    }
  }
  return bestBounds
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
