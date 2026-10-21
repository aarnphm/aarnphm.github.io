import type { ActivityKind, StravaActivityDetail, StravaMapPoint } from '../plugins/stores/strava'
import type { FullSlug } from './path'
import { activityGpsSegments, simplifyMapRoute } from './triathlon-map-route'
import { isRecord } from './type-guards'

// The hub link preview: the last 60 days of routes drawn on one plane, so overlaps read as heat.
// Like the day previews, it carries shapes in a box, never coordinates.
export const TRIATHLON_HUB_PREVIEW_KIND = 'triathlon-hub-preview-v1'
export const HUB_PREVIEW_DAYS = 60
/** 16:10, the popover frame's aspect; integer coordinates keep the payload small. */
export const HUB_BOX = { w: 1600, h: 1000 } as const
// Routes whose centroid sits this far from the densest cluster's are a trip; drawn, they would
// shrink home to a dot.
const CLUSTER_RADIUS_M = 30_000
// Streets must stay streets: ~20 m tolerance on a 50 km sheet, 320 points a route; 90 routes
// come to about 60 KB before compression.
const ROUTE_MAX_POINTS = 320
const ROUTE_RESOLUTION = 2500
const PAD = 0.06
// The window is the dense core, not the bounding box: the long rides' far ends run off the sheet
// and home streets get the pixels. Quantiles weight every route the same, so a 100 km ride's
// thousands of samples do not outvote a 5 km run.
const CORE_QUANTILE = 0.08
const METRES_PER_DEGREE = 111_320
const HUB_SPORTS: readonly ActivityKind[] = ['swim', 'bike', 'run', 'walk']

export interface TriathlonHubRoute {
  sport: ActivityKind
  d: string
}

export interface TriathlonHubPreview {
  kind: typeof TRIATHLON_HUB_PREVIEW_KIND
  from: string
  to: string
  box: { w: number; h: number }
  routes: TriathlonHubRoute[]
  /** Every session in the window, with or without a route. */
  sessions: number
  km: Partial<Record<ActivityKind, number>>
  hours: number
  /** Routes outside the drawn cluster. */
  away: number
}

export const triathlonHubPreviewSlug = (): FullSlug => 'static/triathlon/preview/hub' as FullSlug

export const triathlonHubPreviewUrl = (reference: string): URL =>
  new URL(`/${triathlonHubPreviewSlug()}.json`, reference)

const shiftDate = (iso: string, days: number): string => {
  const date = new Date(`${iso}T00:00:00Z`)
  date.setUTCDate(date.getUTCDate() + days)
  return date.toISOString().slice(0, 10)
}

interface Traced {
  detail: StravaActivityDetail
  segments: readonly (readonly StravaMapPoint[])[]
  lat: number
  lng: number
}

const traceOf = (detail: StravaActivityDetail): Traced | null => {
  if (detail.virtual) return null
  const segments = activityGpsSegments(detail)
  if (segments.length === 0) return null
  let lat = 0
  let lng = 0
  let n = 0
  for (const segment of segments)
    for (const point of segment) {
      lat += point.lat
      lng += point.lng
      n += 1
    }
  return { detail, segments, lat: lat / n, lng: lng / n }
}

/** The [q, 1 - q] weighted quantile span of (value, weight) samples. */
function weightedCore(samples: [number, number][]): [number, number] {
  samples.sort((left, right) => left[0] - right[0])
  const total = samples.reduce((sum, [, weight]) => sum + weight, 0)
  let low = samples[0][0]
  let high = samples[samples.length - 1][0]
  let seen = 0
  let lowFound = false
  for (const [value, weight] of samples) {
    seen += weight
    if (!lowFound && seen >= total * CORE_QUANTILE) {
      low = value
      lowFound = true
    }
    if (seen >= total * (1 - CORE_QUANTILE)) {
      high = value
      break
    }
  }
  return high > low ? [low, high] : [samples[0][0], samples[samples.length - 1][0]]
}

/** Members of the densest cluster of route centroids (planar distance at the first route's latitude). */
function homeCluster(traced: readonly Traced[]): Traced[] {
  if (traced.length === 0) return []
  const scaleX = Math.cos((traced[0].lat * Math.PI) / 180) * METRES_PER_DEGREE
  const near = (a: Traced, b: Traced): boolean =>
    Math.hypot((a.lng - b.lng) * scaleX, (a.lat - b.lat) * METRES_PER_DEGREE) <= CLUSTER_RADIUS_M
  let best: Traced[] = []
  for (const centre of traced) {
    const members = traced.filter(other => near(centre, other))
    if (members.length > best.length) best = members
  }
  return best
}

export function buildTriathlonHubPreview(
  details: readonly StravaActivityDetail[],
  today: string,
): TriathlonHubPreview {
  const from = shiftDate(today, 1 - HUB_PREVIEW_DAYS)
  const window = details
    .filter(detail => detail.date >= from && detail.date <= today)
    .sort((left, right) => left.start.localeCompare(right.start))
  const km: Partial<Record<ActivityKind, number>> = {}
  let seconds = 0
  for (const detail of window) {
    if (detail.distanceKm > 0)
      km[detail.sport] = Math.round(((km[detail.sport] ?? 0) + detail.distanceKm) * 10) / 10
    seconds += detail.movingTimeS
  }
  const traced = window.flatMap(detail => {
    const trace = HUB_SPORTS.includes(detail.sport) ? traceOf(detail) : null
    return trace ? [trace] : []
  })
  const drawn = homeCluster(traced)
  const routes: TriathlonHubRoute[] = []
  if (drawn.length > 0) {
    const scaleX = Math.cos((drawn[0].lat * Math.PI) / 180) * METRES_PER_DEGREE
    const xs: [number, number][] = []
    const ys: [number, number][] = []
    for (const { segments } of drawn) {
      const count = segments.reduce((sum, segment) => sum + segment.length, 0)
      for (const segment of segments)
        for (const point of segment) {
          xs.push([point.lng * scaleX, 1 / count])
          ys.push([-point.lat * METRES_PER_DEGREE, 1 / count])
        }
    }
    const [minX, maxX] = weightedCore(xs)
    const [minY, maxY] = weightedCore(ys)
    const extentX = (maxX - minX) * (1 + 2 * PAD)
    const extentY = (maxY - minY) * (1 + 2 * PAD)
    const scale = Math.min(HUB_BOX.w / extentX, HUB_BOX.h / extentY)
    const offsetX = (HUB_BOX.w - (maxX - minX) * scale) / 2
    const offsetY = (HUB_BOX.h - (maxY - minY) * scale) / 2
    const extent = Math.max(extentX, extentY)
    for (const { detail, segments } of drawn) {
      const commands: string[] = []
      for (const segment of segments) {
        // Coarsen until the route fits the point budget; a 60-day sheet holds dozens of routes.
        let tolerance = Math.max(extent / ROUTE_RESOLUTION, 5)
        let points = simplifyMapRoute(segment, tolerance)
        for (let round = 0; points.length > ROUTE_MAX_POINTS && round < 8; round += 1) {
          tolerance *= 1.6
          points = simplifyMapRoute(segment, tolerance)
        }
        if (points.length < 2) continue
        points.forEach((point, index) => {
          const x = Math.round((point.lng * scaleX - minX) * scale + offsetX)
          const y = Math.round((-point.lat * METRES_PER_DEGREE - minY) * scale + offsetY)
          commands.push(`${index === 0 ? 'M' : 'L'}${x} ${y}`)
        })
      }
      if (commands.length > 1) routes.push({ sport: detail.sport, d: commands.join('') })
    }
  }
  return {
    kind: TRIATHLON_HUB_PREVIEW_KIND,
    from,
    to: today,
    box: { ...HUB_BOX },
    routes,
    sessions: window.length,
    km,
    hours: Math.round((seconds / 3600) * 10) / 10,
    away: traced.length - drawn.length,
  }
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

export function readTriathlonHubPreview(value: unknown): TriathlonHubPreview | null {
  if (
    !isRecord(value) ||
    value.kind !== TRIATHLON_HUB_PREVIEW_KIND ||
    typeof value.from !== 'string' ||
    typeof value.to !== 'string' ||
    !isRecord(value.box) ||
    !isFiniteNumber(value.box.w) ||
    !isFiniteNumber(value.box.h) ||
    !Array.isArray(value.routes) ||
    !value.routes.every(
      route => isRecord(route) && typeof route.sport === 'string' && typeof route.d === 'string',
    ) ||
    !isFiniteNumber(value.sessions) ||
    !isRecord(value.km) ||
    !Object.values(value.km).every(isFiniteNumber) ||
    !isFiniteNumber(value.hours) ||
    !isFiniteNumber(value.away)
  )
    return null
  return value as unknown as TriathlonHubPreview
}
