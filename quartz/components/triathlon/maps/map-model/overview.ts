import type { StravaActivityDetail, StravaMapPoint } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { StreetMapActivity, StreetMetricValues } from '../../../scripts/triathlon-map-heat'
import type { DetailPayload } from '../../activity/data'
import { ROUTE_SPORTS, type ActivityKind } from '../../../../plugins/stores/strava'
import {
  activityAverageCadence,
  activityCadenceUnit,
  clock,
  KM_TO_MI,
  powerViewActivity,
} from '../../../../util/triathlon-card'
import { CAD_RAMP, HEAT_RAMP, HR_RAMP, SPD_RAMP } from '../palette'
import { gpsSegments, segmentFeatures, type GeoFC } from './geo'

export type OverviewMode = 'heat' | 'w' | 'hr' | 'cad' | 'spd'

export const readOverviewMode = (value: string | undefined): OverviewMode | null =>
  value === 'heat' || value === 'w' || value === 'hr' || value === 'cad' || value === 'spd'
    ? value
    : null

export const readRouteSport = (value: string | undefined): ActivityKind | null =>
  value === 'run' || value === 'bike' || value === 'swim' || value === 'walk' || value === 'hike'
    ? value
    : null

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
