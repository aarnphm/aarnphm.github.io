import { isRecord } from './type-guards'

export type OverlayKey = 'card' | 'grid' | 'stack' | 'receipt' | 'chips' | 'rail' | 'route'
export const OVERLAY_OPTIONS: ReadonlyArray<readonly [OverlayKey, string]> = [
  ['card', 'carte'],
  ['grid', 'grille'],
  ['stack', 'colonne'],
  ['receipt', 'ticket'],
  ['chips', 'pastilles'],
  ['rail', 'bord vertical'],
  ['route', 'parcours seul'],
]
export const isOverlayKey = (value: string): value is OverlayKey =>
  OVERLAY_OPTIONS.some(([key]) => key === value)

export const ACTIVITY_CARD_STATS = [
  'distance',
  'time',
  'pace',
  'elevation',
  'heartRate',
  'power',
] as const
export type ActivityCardStatKey = (typeof ACTIVITY_CARD_STATS)[number]

export interface ActivityCardStat {
  key: string
  label: string
  value: string
  provenance?: string
}

export type ActivityCardChartKey =
  | 'cyclingPower'
  | 'powerCurve'
  | 'cyclingLaps'
  | 'runPace'
  | 'runLaps'
export type PowerWindow = 30 | 300
export interface ActivityCardChartPoint {
  x: number
  y: number | null
  label: string
  value: string
  range?: string
  end?: number
  detail?: string
  split?: {
    distanceKm: number
    elevationDifferenceM: number | null
    heartRate: number | null
    heartRateSource: 'strava' | null
  }
}
export interface ActivityCardChart {
  key: ActivityCardChartKey
  title: string
  kind: 'line' | 'bars' | 'intervals'
  axis: string
  note: string
  points: ActivityCardChartPoint[]
  series?: { seconds: PowerWindow; points: ActivityCardChartPoint[] }[]
  window?: PowerWindow
}

export interface ActivityCardPoint {
  lat: number
  lng: number
}

export interface ActivityCardData {
  url: string
  title: string
  date: string
  sport: string
  stats: ActivityCardStat[]
  route: ActivityCardPoint[][]
  charts: ActivityCardChart[]
}

export const activityCardStatKey = (label: string): ActivityCardStatKey | null => {
  switch (label) {
    case 'distance':
      return 'distance'
    case 'time':
    case 'duration':
      return 'time'
    case 'pace':
    case 'speed':
      return 'pace'
    case 'elevation':
    case 'elevation gain':
      return 'elevation'
    case 'avg hr':
      return 'heartRate'
    case 'avg power':
    case 'est power':
    case 'drag power index':
    case 'walking power':
      return 'power'
    default:
      return null
  }
}

const isStatKey = (value: unknown): value is ActivityCardStatKey =>
  typeof value === 'string' && ACTIVITY_CARD_STATS.some(key => key === value)

const isStat = (value: unknown): value is ActivityCardStat =>
  isRecord(value) &&
  typeof value.key === 'string' &&
  typeof value.label === 'string' &&
  typeof value.value === 'string' &&
  (value.provenance === undefined || typeof value.provenance === 'string')

export const isActivityCardStatKey = isStatKey
export const isActivityCardChartKey = (value: string): value is ActivityCardChartKey =>
  value === 'cyclingPower' ||
  value === 'powerCurve' ||
  value === 'cyclingLaps' ||
  value === 'runPace' ||
  value === 'runLaps'

const isPowerWindow = (value: unknown): value is PowerWindow => value === 30 || value === 300
const isChartPoint = (point: unknown): point is ActivityCardChartPoint =>
  isRecord(point) &&
  typeof point.x === 'number' &&
  Number.isFinite(point.x) &&
  (point.y === null || (typeof point.y === 'number' && Number.isFinite(point.y))) &&
  typeof point.label === 'string' &&
  typeof point.value === 'string' &&
  (point.split === undefined ||
    (isRecord(point.split) &&
      typeof point.split.distanceKm === 'number' &&
      Number.isFinite(point.split.distanceKm) &&
      point.split.distanceKm > 0 &&
      (point.split.elevationDifferenceM === null ||
        (typeof point.split.elevationDifferenceM === 'number' &&
          Number.isFinite(point.split.elevationDifferenceM))) &&
      (point.split.heartRate === null ||
        (typeof point.split.heartRate === 'number' &&
          Number.isFinite(point.split.heartRate) &&
          point.split.heartRate > 0)) &&
      (point.split.heartRateSource === null || point.split.heartRateSource === 'strava'))) &&
  (point.range === undefined || typeof point.range === 'string') &&
  (point.detail === undefined || typeof point.detail === 'string') &&
  (point.end === undefined ||
    (typeof point.end === 'number' && Number.isFinite(point.end) && point.end > point.x))

const isChart = (value: unknown): value is ActivityCardChart =>
  isRecord(value) &&
  typeof value.key === 'string' &&
  isActivityCardChartKey(value.key) &&
  (value.kind === 'line' || value.kind === 'bars' || value.kind === 'intervals') &&
  typeof value.title === 'string' &&
  typeof value.axis === 'string' &&
  typeof value.note === 'string' &&
  Array.isArray(value.points) &&
  value.points.every(isChartPoint) &&
  (value.window === undefined || isPowerWindow(value.window)) &&
  (value.series === undefined ||
    (Array.isArray(value.series) &&
      value.series.every(
        series =>
          isRecord(series) &&
          isPowerWindow(series.seconds) &&
          Array.isArray(series.points) &&
          series.points.every(isChartPoint),
      )))

const isPoint = (value: unknown): value is ActivityCardPoint =>
  isRecord(value) &&
  typeof value.lat === 'number' &&
  Number.isFinite(value.lat) &&
  Math.abs(value.lat) <= 85.0511 &&
  typeof value.lng === 'number' &&
  Number.isFinite(value.lng) &&
  Math.abs(value.lng) <= 180

export function activityCardRoute(
  segments: readonly (readonly ActivityCardPoint[])[],
): ActivityCardPoint[][] {
  const route: ActivityCardPoint[][] = []
  for (const segment of segments) {
    let current: ActivityCardPoint[] = []
    const flush = () => {
      if (current.length >= 2) route.push(current)
      current = []
    }
    for (const point of segment) {
      if (isPoint(point)) current.push({ lat: point.lat, lng: point.lng })
      else flush()
    }
    flush()
  }
  return route
}

function activityCardData(value: unknown): ActivityCardData | null {
  if (
    !isRecord(value) ||
    typeof value.url !== 'string' ||
    typeof value.title !== 'string' ||
    typeof value.date !== 'string' ||
    typeof value.sport !== 'string' ||
    !Array.isArray(value.stats) ||
    !value.stats.every(isStat) ||
    !Array.isArray(value.route) ||
    !Array.isArray(value.charts) ||
    !value.charts.every(isChart)
  )
    return null
  const route: ActivityCardPoint[][] = []
  for (const segment of value.route) {
    if (!Array.isArray(segment) || !segment.every(isPoint)) return null
    if (segment.length >= 2) route.push(segment)
  }
  return {
    url: value.url,
    title: value.title,
    date: value.date,
    sport: value.sport,
    stats: value.stats,
    route,
    charts: value.charts,
  }
}

function decodeData(raw: string | undefined): unknown {
  if (!raw) return null
  try {
    return JSON.parse(decodeURIComponent(raw))
  } catch {
    return null
  }
}

export function parseActivityCardData(raw: string | undefined): ActivityCardData | null {
  return activityCardData(decodeData(raw))
}

export function parseActivityCardList(raw: string | undefined): ActivityCardData[] {
  const value = decodeData(raw)
  return Array.isArray(value)
    ? value.flatMap(item => {
        const data = activityCardData(item)
        return data ? [data] : []
      })
    : []
}
