export type Sport = 'swim' | 'bike' | 'run'

export const SPORT_ORDER: readonly Sport[] = ['swim', 'bike', 'run']

const DAY_MS = 86_400_000
const WINDOW_DAYS = 364

export interface StravaAuth {
  refreshToken: string
  obtainedAt: number
}

export interface RawStravaActivity {
  id: number
  name: string
  sportType: string
  distance: number
  movingTime: number
  elapsedTime: number
  totalElevationGain: number
  startDate: string
  startDateLocal: string
  averageSpeed: number
  averageHeartrate?: number
}

export interface StravaStreams {
  latlng: [number, number][]
  altitude: number[]
  distance: number[]
}

export interface StravaRawCache {
  athleteId: number
  auth: StravaAuth
  lastSync: number
  lastActivityStart: number
  activities: Record<string, RawStravaActivity>
  streams?: Record<string, StravaStreams>
}

export interface StravaSportTotals {
  sport: Sport
  count: number
  distanceKm: number
  movingTimeS: number
  elevationM: number
}

export interface StravaDayItem {
  id: number
  sport: Sport
  distanceKm: number
  durationS: number
}

export interface StravaDay {
  date: string
  durationS: number
  items: StravaDayItem[]
  dominant: Sport | null
}

export interface StravaRoutePoint {
  x: number
  y: number
  d: number
  alt: number
}

export interface StravaActivityDetail {
  id: number
  sport: Sport
  name: string
  date: string
  distanceKm: number
  movingTimeS: number
  elevationM: number
  avgHr: number | null
  route: StravaRoutePoint[]
  minAlt: number
  maxAlt: number
}

export interface StravaPayload {
  generatedAt: number
  athleteId: number
  totalKm: number
  totalTimeS: number
  totalCount: number
  totals: StravaSportTotals[]
  days: StravaDay[]
  details: Record<string, StravaActivityDetail>
}

export function normalizeSport(sportType: string): Sport | null {
  switch (sportType) {
    case 'Run':
    case 'TrailRun':
    case 'VirtualRun':
      return 'run'
    case 'Ride':
    case 'VirtualRide':
    case 'MountainBikeRide':
    case 'GravelRide':
    case 'EBikeRide':
      return 'bike'
    case 'Swim':
    case 'OpenWaterSwim':
      return 'swim'
    default:
      return null
  }
}

function round(value: number, dp: number): number {
  const f = 10 ** dp
  return Math.round(value * f) / f
}

function emptyTotals(): StravaSportTotals[] {
  return SPORT_ORDER.map(sport => ({
    sport,
    count: 0,
    distanceKm: 0,
    movingTimeS: 0,
    elevationM: 0,
  }))
}

function projectDetail(
  a: RawStravaActivity,
  sport: Sport,
  streams: StravaStreams | undefined,
): StravaActivityDetail {
  const route: StravaRoutePoint[] = []
  let minAlt = 0
  let maxAlt = 0
  const latlng = streams?.latlng ?? []
  if (latlng.length >= 2) {
    const altitude = streams!.altitude
    const distance = streams!.distance
    const n = latlng.length
    const stride = Math.max(1, Math.ceil(n / 140))
    const idx: number[] = []
    for (let i = 0; i < n; i += stride) idx.push(i)
    if (idx[idx.length - 1] !== n - 1) idx.push(n - 1)
    let sumLat = 0
    let sumLng = 0
    for (const i of idx) {
      sumLat += latlng[i][0]
      sumLng += latlng[i][1]
    }
    const meanLat = sumLat / idx.length
    const meanLng = sumLng / idx.length
    const cosLat = Math.cos((meanLat * Math.PI) / 180)
    const xs = idx.map(i => (latlng[i][1] - meanLng) * cosLat)
    const ys = idx.map(i => latlng[i][0] - meanLat)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const span = Math.max(maxX - minX, maxY - minY) || 1
    const offX = (1 - (maxX - minX) / span) / 2
    const offY = (1 - (maxY - minY) / span) / 2
    const alts = idx.map(i => altitude[i] ?? 0)
    minAlt = round(Math.min(...alts), 1)
    maxAlt = round(Math.max(...alts), 1)
    idx.forEach((i, k) => {
      route.push({
        x: round((xs[k] - minX) / span + offX, 4),
        y: round((ys[k] - minY) / span + offY, 4),
        d: round((distance[i] ?? 0) / 1000, 3),
        alt: round(alts[k], 1),
      })
    })
  }
  return {
    id: a.id,
    sport,
    name: a.name,
    date: a.startDateLocal.slice(0, 10),
    distanceKm: round(a.distance / 1000, 1),
    movingTimeS: a.movingTime,
    elevationM: Math.round(a.totalElevationGain),
    avgHr: a.averageHeartrate ? Math.round(a.averageHeartrate) : null,
    route,
    minAlt,
    maxAlt,
  }
}

export function emptyPayload(athleteId = 0): StravaPayload {
  return {
    generatedAt: 0,
    athleteId,
    totalKm: 0,
    totalTimeS: 0,
    totalCount: 0,
    totals: emptyTotals(),
    days: [],
    details: {},
  }
}

export function buildPayload(cache: StravaRawCache | null, since?: string): StravaPayload {
  if (!cache) return emptyPayload()

  const sinceDay = since && /^\d{4}-\d{2}-\d{2}$/.test(since) ? since : null
  const activities = Object.values(cache.activities)
    .map(a => ({ a, sport: normalizeSport(a.sportType) }))
    .filter((x): x is { a: RawStravaActivity; sport: Sport } => x.sport !== null)
    .filter(x => !sinceDay || x.a.startDateLocal.slice(0, 10) >= sinceDay)
    .sort((p, q) => p.a.startDateLocal.localeCompare(q.a.startDateLocal))

  if (activities.length === 0) return emptyPayload(cache.athleteId)

  const totals = emptyTotals()
  const byDate = new Map<string, StravaDayItem[]>()
  for (const { a, sport } of activities) {
    const t = totals.find(x => x.sport === sport)!
    t.count += 1
    t.distanceKm += a.distance / 1000
    t.movingTimeS += a.movingTime
    t.elevationM += a.totalElevationGain

    const date = a.startDateLocal.slice(0, 10)
    const items = byDate.get(date) ?? []
    items.push({
      id: a.id,
      sport,
      distanceKm: round(a.distance / 1000, 1),
      durationS: a.movingTime,
    })
    byDate.set(date, items)
  }

  const dayMs = (iso: string): number => Date.parse(`${iso}T00:00:00Z`)
  const firstMs = dayMs(activities[0].a.startDateLocal.slice(0, 10))
  const lastActMs = dayMs(activities[activities.length - 1].a.startDateLocal.slice(0, 10))
  const end = cache.lastSync
    ? dayMs(new Date(cache.lastSync).toISOString().slice(0, 10))
    : lastActMs
  const start = sinceDay ? dayMs(sinceDay) : Math.max(firstMs, end - (WINDOW_DAYS - 1) * DAY_MS)
  const days: StravaDay[] = []
  for (let ms = start; ms <= end; ms += DAY_MS) {
    const date = new Date(ms).toISOString().slice(0, 10)
    const items = byDate.get(date) ?? []
    const dominant = items.reduce<StravaDayItem | null>(
      (best, item) => (item.distanceKm > (best?.distanceKm ?? -1) ? item : best),
      null,
    )
    days.push({
      date,
      durationS: items.reduce((s, item) => s + item.durationS, 0),
      items,
      dominant: dominant?.sport ?? null,
    })
  }

  const finalized = totals.map(t => ({
    ...t,
    distanceKm: round(t.distanceKm, 1),
    elevationM: Math.round(t.elevationM),
  }))

  const details: Record<string, StravaActivityDetail> = {}
  for (const { a, sport } of activities) {
    details[String(a.id)] = projectDetail(a, sport, cache.streams?.[String(a.id)])
  }

  return {
    generatedAt: cache.lastSync,
    athleteId: cache.athleteId,
    totalKm: round(
      finalized.reduce((s, t) => s + t.distanceKm, 0),
      1,
    ),
    totalTimeS: finalized.reduce((s, t) => s + t.movingTimeS, 0),
    totalCount: finalized.reduce((s, t) => s + t.count, 0),
    totals: finalized,
    days,
    details,
  }
}
