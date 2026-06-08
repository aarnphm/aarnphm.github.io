import type { GarminCache, GarminFueling } from './garmin'
import type { OuraCache, OuraDaily } from './oura'
import { matchGarminFueling } from './garmin'

export type Sport = 'swim' | 'bike' | 'run'

export const SPORT_ORDER: readonly Sport[] = ['swim', 'bike', 'run']

export const SPORT_ICON: Record<Sport, string[]> = {
  run: [
    'M15 7C16.1046 7 17 6.10457 17 5C17 3.89543 16.1046 3 15 3C13.8954 3 13 3.89543 13 5C13 6.10457 13.8954 7 15 7Z',
    'M12.6129 8.26709L9.30469 12.4023L13.4399 16.5376L11.3723 21.0863',
    'M6.41016 9.50741L9.79704 6.19922L12.613 8.26683L15.5078 11.5751H19.2295',
    'M8.89055 15.7104L7.64998 16.5375H4.3418',
  ],
  bike: [
    'M9 17.5a3.5 3.5 0 1 0-7 0a3.5 3.5 0 1 0 7 0',
    'M22 17.5a3.5 3.5 0 1 0-7 0a3.5 3.5 0 1 0 7 0',
    'M16 5a1 1 0 1 0-2 0a1 1 0 1 0 2 0',
    'M12 17.5V14l-3-3 4-3 2 3h2',
  ],
  swim: [
    'M18 6a2 2 0 1 0-4 0a2 2 0 1 0 4 0',
    'M3 13l6-2 4 2.5',
    'M2 18c1.5 1.4 3 1.4 4.5 0s3-1.4 4.5 0 3 1.4 4.5 0',
  ],
}

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
  maxHeartrate?: number
  averageWatts?: number
  weightedAverageWatts?: number
  maxWatts?: number
  kilojoules?: number
  deviceWatts?: boolean
  averageCadence?: number
  sufferScore?: number
  averageTemp?: number
  calories?: number
}

export interface StravaStreams {
  latlng: [number, number][]
  altitude: number[]
  distance: number[]
  watts?: number[]
  heartrate?: number[]
  cadence?: number[]
}

export interface StravaZones {
  hr: number[]
  power: number[]
  ftp: number | null
}

export interface StravaRawCache {
  version?: number
  athleteId: number
  auth: StravaAuth
  lastSync: number
  lastActivityStart: number
  activities: Record<string, RawStravaActivity>
  streams?: Record<string, StravaStreams>
  geo?: Record<string, string>
  zones?: StravaZones
}

export interface PowerCurvePoint {
  s: number
  w: number
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
  w: number
  hr: number
  cad: number
}

export type ActivityHealth = Omit<OuraDaily, 'date'>

export interface StravaActivityDetail {
  id: number
  sport: Sport
  name: string
  date: string
  distanceKm: number
  movingTimeS: number
  elevationM: number
  avgHr: number | null
  maxHr: number | null
  avgWatts: number | null
  npWatts: number | null
  maxWatts: number | null
  kilojoules: number | null
  deviceWatts: boolean
  avgCadence: number | null
  sufferScore: number | null
  calories: number | null
  avgTemp: number | null
  location: string | null
  fueling: GarminFueling | null
  route: StravaRoutePoint[]
  minAlt: number
  maxAlt: number
  descentM: number
  hrZones: number[] | null
  powerZones: number[] | null
  powerHist: number[] | null
  powerCurve: PowerCurvePoint[] | null
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
  health: Record<string, ActivityHealth>
  zones: StravaZones
  powerCurveRef: PowerCurvePoint[]
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

export function round(value: number, dp: number): number {
  const f = 10 ** dp
  return Math.round(value * f) / f
}

function cleanAltitude(alt: number[]): number[] {
  const n = alt.length
  const filled = alt.slice()
  let last = filled.find(x => x > 0.5) ?? 0
  for (let i = 0; i < n; i++) {
    if (filled[i] > 0.5) last = filled[i]
    else filled[i] = last
  }
  const w = 4
  const out = filled.slice()
  for (let i = 0; i < n; i++) {
    let sum = 0
    let count = 0
    for (let j = Math.max(0, i - w); j <= Math.min(n - 1, i + w); j++) {
      sum += filled[j]
      count++
    }
    out[i] = sum / count
  }
  return out
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

function toHealth(o: OuraDaily): ActivityHealth {
  return {
    readiness: o.readiness,
    sleepScore: o.sleepScore,
    hrv: o.hrv,
    rhr: o.rhr,
    sleepDurationS: o.sleepDurationS,
    tempDeviationC: o.tempDeviationC,
    totalCalories: o.totalCalories,
    activeCalories: o.activeCalories,
  }
}

function avgPos(arr: number[]): number | null {
  let sum = 0
  let count = 0
  for (const x of arr) {
    if (x > 0) {
      sum += x
      count++
    }
  }
  return count ? Math.round(sum / count) : null
}

function maxPos(arr: number[]): number | null {
  let m = 0
  for (const x of arr) if (x > m) m = x
  return m > 0 ? Math.round(m) : null
}

const CURVE_SECS = [1, 5, 10, 15, 30, 60, 120, 180, 300, 600, 1200, 1800, 3600, 5400, 7200, 10800]

function meanMaxCurve(w: number[]): PowerCurvePoint[] {
  const n = w.length
  if (n < 1) return []
  const pre = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + (w[i] > 0 ? w[i] : 0)
  const out: PowerCurvePoint[] = []
  for (const d of CURVE_SECS) {
    if (d > n) break
    let best = 0
    for (let i = 0; i + d <= n; i++) {
      const avg = (pre[i + d] - pre[i]) / d
      if (avg > best) best = avg
    }
    out.push({ s: d, w: Math.round(best) })
  }
  return out
}

function zoneTimes(stream: number[], uppers: number[], countZero: boolean): number[] {
  const counts = Array.from({ length: uppers.length + 1 }, () => 0)
  for (const raw of stream) {
    if (raw <= 0 && !countZero) continue
    const v = raw > 0 ? raw : 0
    let z = uppers.length
    for (let i = 0; i < uppers.length; i++)
      if (v <= uppers[i]) {
        z = i
        break
      }
    counts[z]++
  }
  return counts
}

function powerHistogram(w: number[], bin = 25): number[] {
  let maxB = 0
  for (const raw of w) {
    const b = Math.floor((raw > 0 ? raw : 0) / bin)
    if (b > maxB) maxB = b
  }
  const out = Array.from({ length: maxB + 1 }, () => 0)
  for (const raw of w) out[Math.floor((raw > 0 ? raw : 0) / bin)]++
  return out
}

function deriveHrBounds(hrmax: number): number[] {
  return [0.6, 0.7, 0.8, 0.9].map(p => Math.round(hrmax * p))
}

function derivePowerBounds(ftp: number): number[] {
  return [0.55, 0.75, 0.9, 1.05, 1.2, 1.5].map(p => Math.round(ftp * p))
}

function mergeMaxCurves(curves: PowerCurvePoint[][]): PowerCurvePoint[] {
  const best = new Map<number, number>()
  for (const c of curves) for (const p of c) best.set(p.s, Math.max(best.get(p.s) ?? 0, p.w))
  return CURVE_SECS.filter(s => best.has(s)).map(s => ({ s, w: best.get(s)! }))
}

function projectDetail(
  a: RawStravaActivity,
  sport: Sport,
  streams: StravaStreams | undefined,
  geo: string | undefined,
  fueling: GarminFueling | null,
  hrBounds: number[],
  powerBounds: number[],
): StravaActivityDetail {
  const route: StravaRoutePoint[] = []
  let minAlt = 0
  let maxAlt = 0
  let ascentM = 0
  let descentM = 0
  const latlng = streams?.latlng ?? []
  const hrStream = streams?.heartrate ?? []
  const cadStream = streams?.cadence ?? []
  if (latlng.length >= 2) {
    const altitude = cleanAltitude(streams!.altitude)
    const distance = streams!.distance
    const watts = streams!.watts ?? []
    let ascent = 0
    let descent = 0
    for (let i = 1; i < altitude.length; i++) {
      const delta = altitude[i] - altitude[i - 1]
      if (delta > 0) ascent += delta
      else descent -= delta
    }
    ascentM = Math.round(ascent)
    descentM = Math.round(descent)
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
        w: Math.round(watts[i] ?? 0),
        hr: Math.round(hrStream[i] ?? 0),
        cad: Math.round(cadStream[i] ?? 0),
      })
    })
  }
  const wFull = streams?.watts ?? []
  const hasHr = hrStream.some(v => v > 0)
  const hasW = wFull.some(v => v > 0)
  return {
    id: a.id,
    sport,
    name: a.name,
    date: a.startDateLocal.slice(0, 10),
    distanceKm: round(a.distance / 1000, 1),
    movingTimeS: a.movingTime,
    elevationM: ascentM,
    avgHr: a.averageHeartrate ? Math.round(a.averageHeartrate) : avgPos(hrStream),
    maxHr: a.maxHeartrate ? Math.round(a.maxHeartrate) : maxPos(hrStream),
    avgWatts: a.averageWatts != null ? Math.round(a.averageWatts) : null,
    npWatts: a.weightedAverageWatts != null ? Math.round(a.weightedAverageWatts) : null,
    maxWatts: a.maxWatts != null ? Math.round(a.maxWatts) : null,
    kilojoules: a.kilojoules != null ? Math.round(a.kilojoules) : null,
    deviceWatts: a.deviceWatts === true,
    avgCadence: a.averageCadence != null ? Math.round(a.averageCadence) : avgPos(cadStream),
    sufferScore: a.sufferScore != null ? Math.round(a.sufferScore) : null,
    calories: a.calories != null ? Math.round(a.calories) : null,
    avgTemp: a.averageTemp != null ? Math.round(a.averageTemp) : null,
    location: geo ?? null,
    fueling,
    route,
    minAlt,
    maxAlt,
    descentM,
    hrZones: hasHr && hrBounds.length > 0 ? zoneTimes(hrStream, hrBounds, false) : null,
    powerZones: hasW && powerBounds.length > 0 ? zoneTimes(wFull, powerBounds, true) : null,
    powerHist: hasW ? powerHistogram(wFull) : null,
    powerCurve: hasW ? meanMaxCurve(wFull) : null,
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
    health: {},
    zones: { hr: [], power: [], ftp: null },
    powerCurveRef: [],
  }
}

export function buildPayload(
  cache: StravaRawCache | null,
  oura: OuraCache | null,
  garmin: GarminCache | null,
  since?: string,
): StravaPayload {
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

  let hrmax = 0
  for (const { a } of activities) if ((a.maxHeartrate ?? 0) > hrmax) hrmax = a.maxHeartrate ?? 0
  if (hrmax < 100) hrmax = 190
  const recentCut = end - 42 * DAY_MS
  let best20 = 0
  const recentCurves: PowerCurvePoint[][] = []
  for (const { a } of activities) {
    const w = cache.streams?.[String(a.id)]?.watts
    if (!w || !w.some(v => v > 0)) continue
    const c = meanMaxCurve(w)
    const p20 = c.find(p => p.s === 1200)
    if (p20 && p20.w > best20) best20 = p20.w
    if (dayMs(a.startDateLocal.slice(0, 10)) >= recentCut) recentCurves.push(c)
  }
  const ftp = cache.zones?.ftp ?? (best20 > 0 ? Math.round(best20 * 0.95) : null)
  const hrBounds = cache.zones?.hr?.length ? cache.zones.hr : deriveHrBounds(hrmax)
  const powerBounds = cache.zones?.power?.length
    ? cache.zones.power
    : ftp != null
      ? derivePowerBounds(ftp)
      : []

  const details: Record<string, StravaActivityDetail> = {}
  for (const { a, sport } of activities) {
    details[String(a.id)] = projectDetail(
      a,
      sport,
      cache.streams?.[String(a.id)],
      cache.geo?.[String(a.id)],
      matchGarminFueling(a, sport, garmin),
      hrBounds,
      powerBounds,
    )
  }

  const health: Record<string, ActivityHealth> = {}
  if (oura) for (const [date, o] of Object.entries(oura.days)) health[date] = toHealth(o)

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
    health,
    zones: { hr: hrBounds, power: powerBounds, ftp },
    powerCurveRef: mergeMaxCurves(recentCurves),
  }
}
