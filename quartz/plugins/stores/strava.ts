import type { GarminActivityMatch, GarminCache, GarminFueling, GarminStreams } from './garmin'
import type { OuraCache, OuraDaily } from './oura'
import type { WeatherCache } from './weather'
import { localIsoDay } from '../../util/local-date'
import { matchGarminActivity, matchGarminFueling, matchGarminHeartRateActivity } from './garmin'

export type Sport = 'swim' | 'bike' | 'run'

export type ActivityKind = Sport | 'strength' | 'walk' | 'yoga' | 'treatment'

export const SPORT_ORDER: readonly Sport[] = ['swim', 'bike', 'run']

export const SPORT_ICON: Record<ActivityKind, string[]> = {
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
  strength: [
    'M7.4 7H4.6C4.26863 7 4 7.26863 4 7.6V16.4C4 16.7314 4.26863 17 4.6 17H7.4C7.73137 17 8 16.7314 8 16.4V7.6C8 7.26863 7.73137 7 7.4 7Z',
    'M19.4 7H16.6C16.2686 7 16 7.26863 16 7.6V16.4C16 16.7314 16.2686 17 16.6 17H19.4C19.7314 17 20 16.7314 20 16.4V7.6C20 7.26863 19.7314 7 19.4 7Z',
    'M1 14.4V9.6C1 9.26863 1.26863 9 1.6 9H3.4C3.73137 9 4 9.26863 4 9.6V14.4C4 14.7314 3.73137 15 3.4 15H1.6C1.26863 15 1 14.7314 1 14.4Z',
    'M23 14.4V9.6C23 9.26863 22.7314 9 22.4 9H20.6C20.2686 9 20 9.26863 20 9.6V14.4C20 14.7314 20.2686 15 20.6 15H22.4C22.7314 15 23 14.7314 23 14.4Z',
    'M8 12H16',
  ],
  walk: [
    'M14 4a1 1 0 1 0-2 0a1 1 0 1 0 2 0',
    'M7 21l3-4',
    'M16 21l-2-4l-3-3l1-6',
    'M6 12l2-3l4-1l3 3l3 1',
  ],
  yoga: [
    'M16.22 23H5.5a5.978 5.978 0 01-2.265-.443C1.928 22.021.878 21.03.354 19.766a4.593 4.593 0 01.161-3.881l5.991-12.98a.983.983 0 01.02-.042C7.159 1.646 8.289.737 9.634.296a6.02 6.02 0 014.132.147c1.307.536 2.357 1.527 2.881 2.791a4.592 4.592 0 01-.161 3.881L16.076 8h5.861a2 2 0 011.816 2.838l-4.809 10.42A3 3 0 0116.22 23zM13.006 2.293a4.02 4.02 0 00-2.75-.096c-.887.29-1.573.867-1.944 1.569L3.957 13.2a6.02 6.02 0 013.808.243c1.093.448 2.007 1.216 2.584 2.195l4.329-9.38.02-.043a2.594 2.594 0 00.1-2.215c-.3-.727-.93-1.353-1.792-1.707zM8.983 17.708A2.63 2.63 0 008.8 17c-.302-.727-.932-1.354-1.793-1.707a4.02 4.02 0 00-2.75-.096c-.89.291-1.579.872-1.949 1.577A2.594 2.594 0 002.201 19c.302.727.932 1.354 1.793 1.707.473.194.987.293 1.506.293h10.72a1 1 0 00.908-.58L21.938 10h-6.785l-3.516 7.619a4.099 4.099 0 01-7.388.115l-.143-.287 1.788-.894.144.287a2.099 2.099 0 002.945.868z',
  ],
  treatment: [
    'M12.5 0a3.5 3.5 0 012.51 5.936c.654 2.723.711 5.556.162 8.303l-.019.091 4.48 1.493A2 2 0 0121 17.721V24h-2v-6.28l-6.153-2.05.364-1.823a17.335 17.335 0 00-.026-6.915 3.513 3.513 0 01-1.665-.073 39.226 39.226 0 01-1.949 14.588l-.394 1.186A2 2 0 017.279 24H1v-2h6.28l.395-1.186a37.226 37.226 0 001.903-12.46L5.71 11.032 8.6 13.2l-1.2 1.6-2.89-2.167a2 2 0 01.061-3.245l5.25-3.635A3.5 3.5 0 0112.5 0zm0 2a1.5 1.5 0 100 3 1.5 1.5 0 000-3z',
  ],
}

const DAY_MS = 86_400_000
const WINDOW_DAYS = 364
const ROUTE_POINTS = 140
const MAP_ROUTE_POINTS = 1600

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
  sport: ActivityKind
  distanceKm: number
  durationS: number
}

export interface StravaDay {
  date: string
  durationS: number
  items: StravaDayItem[]
  dominant: ActivityKind | null
}

export interface StravaRoutePoint {
  x: number
  y: number
  d: number
  alt: number
  w: number
  hr: number
  cad: number
  lat: number
  lng: number
}

export interface StravaMapPoint {
  lat: number
  lng: number
}

export type ActivityHealth = Omit<OuraDaily, 'date'> & {
  windKph?: number | null
  windDir?: string | null
  windDirDeg?: number | null
  windGustKph?: number | null
}

export function emptyHealth(): ActivityHealth {
  return {
    readiness: null,
    sleepScore: null,
    hrv: null,
    rhr: null,
    sleepDurationS: null,
    tempDeviationC: null,
    totalCalories: null,
    activeCalories: null,
  }
}

export interface GarminVerification {
  activityId: string
  name: string | null
  sourceDevice: string | null
  startDate: string
  startDiffS: number
  distanceM: number | null
  distanceDeltaM: number | null
  distanceDeltaPct: number | null
  movingTimeS: number | null
  movingTimeDeltaS: number | null
  elapsedTimeS: number | null
  elapsedTimeDeltaS: number | null
  totalCalories: number | null
  caloriesDelta: number | null
  avgHeartRate: number | null
  avgHeartRateDelta: number | null
  avgPower: number | null
  avgPowerDelta: number | null
  normalizedPower: number | null
  maxPower: number | null
  totalWorkKJ: number | null
  totalWorkDeltaKJ: number | null
  trainingStressScore: number | null
  intensityFactor: number | null
}

export interface ActivityHeartRate {
  avgHr: number | null
  maxHr: number | null
  stream: number[]
}

export interface StravaActivityDetail {
  id: number
  sport: ActivityKind
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
  windKph: number | null
  windDir: string | null
  windDirDeg: number | null
  windGustKph: number | null
  location: string | null
  fueling: GarminFueling | null
  garmin: GarminVerification | null
  route: StravaRoutePoint[]
  mapRoute?: StravaMapPoint[]
  minAlt: number
  maxAlt: number
  descentM: number
  hrZones: number[] | null
  powerZones: number[] | null
  powerHist: number[] | null
  powerCurve: PowerCurvePoint[] | null
  strokes?: Record<string, number> | null
}

export interface StravaPayload {
  generatedAt: number
  athleteId: number
  totalKm: number
  totalTimeS: number
  totalCount: number
  totals: StravaSportTotals[]
  strengthTotal: { count: number; movingTimeS: number }
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

export function normalizeKind(sportType: string): ActivityKind | null {
  switch (sportType) {
    case 'WeightTraining':
    case 'Workout':
    case 'Crossfit':
      return 'strength'
    case 'Walk':
    case 'Hike':
      return 'walk'
    case 'Yoga':
    case 'Pilates':
      return 'yoga'
    default:
      return normalizeSport(sportType)
  }
}

const TREATMENT_TYPES = new Set(['PhysicalTherapy', 'Physiotherapy'])
const TREATMENT_NAME_RE = /\b(physio|physiotherapy|physical[ -]?therapy|treatment|rehab|massage)\b/i

export function isTreatment(sportType: string, name: string | null | undefined): boolean {
  return TREATMENT_TYPES.has(sportType) || TREATMENT_NAME_RE.test(name ?? '')
}

export function round(value: number, dp: number): number {
  const f = 10 ** dp
  return Math.round(value * f) / f
}

export function haversineMeters(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLat = (lat2 - lat1) * 111320
  const dLng = (lng2 - lng1) * 111320 * Math.cos((lat1 * Math.PI) / 180)
  return Math.hypot(dLat, dLng)
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const s = [...xs].sort((p, q) => p - q)
  const m = Math.floor(s.length / 2)
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2
}

function homeStart(starts: [number, number][]): [number, number] | null {
  if (starts.length < 6) return null
  const seedLat = median(starts.map(p => p[0]))
  const seedLng = median(starts.map(p => p[1]))
  const near = starts.filter(p => haversineMeters(p[0], p[1], seedLat, seedLng) <= 200)
  if (near.length < 6) return null
  return [
    near.reduce((s, p) => s + p[0], 0) / near.length,
    near.reduce((s, p) => s + p[1], 0) / near.length,
  ]
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
  if (n <= w * 2 + 1) return filled
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

function sampleIndices(lo: number, hi: number, maxPoints: number): number[] {
  if (hi < lo) return []
  const span = hi - lo + 1
  const stride = Math.max(1, Math.ceil(span / maxPoints))
  const idx: number[] = []
  for (let i = lo; i <= hi; i += stride) idx.push(i)
  if (idx[idx.length - 1] !== hi) idx.push(hi)
  return idx
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

function roundPos(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? Math.round(value) : null
}

function hasPositive(values: number[] | undefined): boolean {
  return values?.some(value => value > 0) ?? false
}

function positiveCount(values: number[] | undefined): number {
  return values?.filter(value => value > 0).length ?? 0
}

function streamQuality(streams: StravaStreams | GarminStreams | undefined): number {
  if (!streams) return 0
  const channels =
    (streams.latlng.length >= 2 ? 1 : 0) +
    (streams.altitude.length > 0 ? 1 : 0) +
    (streams.distance.length > 0 ? 1 : 0) +
    (hasPositive(streams.heartrate) ? 1 : 0) +
    (hasPositive(streams.cadence) ? 1 : 0) +
    (hasPositive(streams.watts) ? 1 : 0)
  return (
    channels * 10_000 +
    streams.latlng.length +
    streams.altitude.length +
    streams.distance.length +
    positiveCount(streams.heartrate) +
    positiveCount(streams.cadence) +
    positiveCount(streams.watts)
  )
}

function selectStreams(
  strava: StravaStreams | undefined,
  match: GarminActivityMatch | null,
  garmin: GarminCache | null,
): StravaStreams | GarminStreams | undefined {
  const fromGarmin = match ? garmin?.streams?.[match.activity.id] : undefined
  return streamQuality(fromGarmin) > streamQuality(strava) ? fromGarmin : strava
}

export function resolveActivityHeartRate(
  a: RawStravaActivity,
  sport: ActivityKind,
  selectedStreams: StravaStreams | GarminStreams | undefined,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
): ActivityHeartRate {
  const selectedHr = selectedStreams?.heartrate ?? []
  const stravaAvg = roundPos(a.averageHeartrate) ?? avgPos(selectedHr)
  const stravaMax = roundPos(a.maxHeartrate) ?? maxPos(selectedHr)
  const garminStream = garminMatch
    ? (garmin?.streams?.[garminMatch.activity.id]?.heartrate ?? [])
    : []
  const metrics = garminMatch?.activity.metrics
  const garminAvg = roundPos(metrics?.avgHeartRate) ?? avgPos(garminStream)
  const garminMax = roundPos(metrics?.maxHeartRate) ?? maxPos(garminStream)

  if (sport === 'run' && (garminAvg != null || garminMax != null))
    return {
      avgHr: garminAvg ?? stravaAvg,
      maxHr: garminMax ?? stravaMax,
      stream: hasPositive(garminStream) ? garminStream : selectedHr,
    }

  return { avgHr: stravaAvg, maxHr: stravaMax, stream: selectedHr }
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

function durationZoneTimes(stream: number[], uppers: number[], movingTimeS: number): number[] {
  const counts = zoneTimes(stream, uppers, false)
  const total = counts.reduce((sum, count) => sum + count, 0)
  if (total <= 0 || movingTimeS <= 0) return counts
  const scale = movingTimeS / total
  return counts.map(count => Math.round(count * scale))
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

function delta(
  garmin: number | null | undefined,
  strava: number | null | undefined,
): number | null {
  return garmin != null && strava != null ? Math.round(garmin - strava) : null
}

function deltaFloat(
  garmin: number | null | undefined,
  strava: number | null | undefined,
  dp: number,
): number | null {
  return garmin != null && strava != null ? round(garmin - strava, dp) : null
}

function garminVerification(
  a: RawStravaActivity,
  match: GarminActivityMatch | null,
): GarminVerification | null {
  if (!match) return null
  const activity = match.activity
  const metrics = activity.metrics
  const distanceDeltaM = delta(activity.distanceM, a.distance)
  return {
    activityId: activity.id,
    name: activity.name,
    sourceDevice: activity.sourceDevice,
    startDate: activity.startDate,
    startDiffS: Math.round(match.startDiffMs / 1000),
    distanceM: activity.distanceM,
    distanceDeltaM,
    distanceDeltaPct:
      distanceDeltaM != null && a.distance > 0
        ? round((distanceDeltaM / a.distance) * 100, 1)
        : null,
    movingTimeS: activity.movingTimeS,
    movingTimeDeltaS: delta(activity.movingTimeS, a.movingTime),
    elapsedTimeS: activity.elapsedTimeS,
    elapsedTimeDeltaS: delta(activity.elapsedTimeS, a.elapsedTime),
    totalCalories: metrics.totalCalories,
    caloriesDelta: delta(metrics.totalCalories, a.calories),
    avgHeartRate: metrics.avgHeartRate,
    avgHeartRateDelta: delta(metrics.avgHeartRate, a.averageHeartrate),
    avgPower: metrics.avgPower,
    avgPowerDelta: delta(metrics.avgPower, a.averageWatts),
    normalizedPower: metrics.normalizedPower,
    maxPower: metrics.maxPower,
    totalWorkKJ: metrics.totalWorkKJ,
    totalWorkDeltaKJ: deltaFloat(metrics.totalWorkKJ, a.kilojoules, 1),
    trainingStressScore: metrics.trainingStressScore,
    intensityFactor: metrics.intensityFactor,
  }
}

function projectDetail(
  a: RawStravaActivity,
  sport: ActivityKind,
  streams: StravaStreams | GarminStreams | undefined,
  heartRate: ActivityHeartRate,
  weather: WeatherCache['activities'][string] | undefined,
  geo: string | undefined,
  fueling: GarminFueling | null,
  garmin: GarminVerification | null,
  hrBounds: number[],
  powerBounds: number[],
  home: [number, number] | null,
): StravaActivityDetail {
  const route: StravaRoutePoint[] = []
  const mapRoute: StravaMapPoint[] = []
  let minAlt = 0
  let maxAlt = 0
  let ascentM = 0
  let descentM = 0
  const latlng = streams?.latlng ?? []
  const alignedHrStream = streams?.heartrate ?? []
  const routeHrStream =
    heartRate.stream.length === latlng.length ? heartRate.stream : alignedHrStream
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
    let lo0 = 0
    let hi0 = n - 1
    if (home) {
      while (lo0 < n && haversineMeters(latlng[lo0][0], latlng[lo0][1], home[0], home[1]) <= 200)
        lo0++
      while (hi0 > lo0 && haversineMeters(latlng[hi0][0], latlng[hi0][1], home[0], home[1]) <= 200)
        hi0--
    }
    if (hi0 - lo0 < 1) {
      lo0 = 0
      hi0 = n - 1
    }
    const idx = sampleIndices(lo0, hi0, ROUTE_POINTS)
    for (const i of sampleIndices(lo0, hi0, MAP_ROUTE_POINTS))
      mapRoute.push({ lat: round(latlng[i][0], 5), lng: round(latlng[i][1], 5) })
    const d0 = distance[lo0] ?? 0
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
        d: round(((distance[i] ?? 0) - d0) / 1000, 3),
        alt: round(alts[k], 1),
        w: Math.round(watts[i] ?? 0),
        hr: Math.round(routeHrStream[i] ?? 0),
        cad: Math.round(cadStream[i] ?? 0),
        lat: round(latlng[i][0], 5),
        lng: round(latlng[i][1], 5),
      })
    })
  }
  const wFull = streams?.watts ?? []
  const hasHr = heartRate.stream.some(v => v > 0)
  const hasW = wFull.some(v => v > 0)
  return {
    id: a.id,
    sport,
    name: a.name,
    date: a.startDateLocal.slice(0, 10),
    distanceKm: round(a.distance / 1000, 1),
    movingTimeS: a.movingTime,
    elevationM: ascentM,
    avgHr: heartRate.avgHr,
    maxHr: heartRate.maxHr,
    avgWatts: a.averageWatts != null ? Math.round(a.averageWatts) : null,
    npWatts: a.weightedAverageWatts != null ? Math.round(a.weightedAverageWatts) : null,
    maxWatts: a.maxWatts != null ? Math.round(a.maxWatts) : null,
    kilojoules: a.kilojoules != null ? Math.round(a.kilojoules) : null,
    deviceWatts: a.deviceWatts === true,
    avgCadence: a.averageCadence != null ? Math.round(a.averageCadence) : avgPos(cadStream),
    sufferScore: a.sufferScore != null ? Math.round(a.sufferScore) : null,
    calories: a.calories ? Math.round(a.calories) : (garmin?.totalCalories ?? null),
    avgTemp: a.averageTemp != null ? Math.round(a.averageTemp) : null,
    windKph: weather?.windKph ?? null,
    windDir: weather?.windDir ?? null,
    windDirDeg: weather?.windDirDeg ?? null,
    windGustKph: weather?.windGustKph ?? null,
    location: geo ?? null,
    fueling,
    garmin,
    route,
    mapRoute,
    minAlt,
    maxAlt,
    descentM,
    hrZones:
      hasHr && hrBounds.length > 0
        ? durationZoneTimes(heartRate.stream, hrBounds, a.movingTime)
        : null,
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
    strengthTotal: { count: 0, movingTimeS: 0 },
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
  weather?: WeatherCache | null,
  inputFtp?: number | null,
  inputHrBounds?: number[] | null,
  timeZone?: string,
): StravaPayload {
  if (!cache) return emptyPayload()

  const sinceDay = since && /^\d{4}-\d{2}-\d{2}$/.test(since) ? since : null
  const activities = Object.values(cache.activities)
    .map(a => ({
      a,
      sport: isTreatment(a.sportType, a.name)
        ? ('treatment' as ActivityKind)
        : normalizeKind(a.sportType),
    }))
    .filter((x): x is { a: RawStravaActivity; sport: ActivityKind } => x.sport !== null)
    .filter(x => !sinceDay || x.a.startDateLocal.slice(0, 10) >= sinceDay)
    .sort((p, q) => p.a.startDateLocal.localeCompare(q.a.startDateLocal))

  if (activities.length === 0) return emptyPayload(cache.athleteId)

  const garminMatches = new Map<string, GarminActivityMatch | null>()
  const garminHeartRateMatches = new Map<string, GarminActivityMatch | null>()
  const selectedStreams = new Map<string, StravaStreams | GarminStreams | undefined>()
  const heartRates = new Map<string, ActivityHeartRate>()
  for (const { a, sport } of activities) {
    const id = String(a.id)
    const match = matchGarminActivity(a, sport, garmin)
    const hrMatch = matchGarminHeartRateActivity(a, sport, garmin)
    const streams = selectStreams(cache.streams?.[id], match, garmin)
    garminMatches.set(id, match)
    garminHeartRateMatches.set(id, hrMatch)
    selectedStreams.set(id, streams)
    heartRates.set(id, resolveActivityHeartRate(a, sport, streams, hrMatch, garmin))
  }

  const totals = emptyTotals()
  const strengthTotal = { count: 0, movingTimeS: 0 }
  const byDate = new Map<string, StravaDayItem[]>()
  for (const { a, sport } of activities) {
    const t = totals.find(x => x.sport === sport)
    if (t) {
      t.count += 1
      t.distanceKm += a.distance / 1000
      t.movingTimeS += a.movingTime
      t.elevationM += a.totalElevationGain
    }
    if (sport === 'strength') {
      strengthTotal.count += 1
      strengthTotal.movingTimeS += a.movingTime
    }

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
  const end = cache.lastSync ? dayMs(localIsoDay(cache.lastSync, timeZone)) : lastActMs
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
  for (const { a } of activities) {
    const maxHr = heartRates.get(String(a.id))?.maxHr
    if ((maxHr ?? 0) > hrmax) hrmax = maxHr ?? 0
  }
  if (hrmax < 100) hrmax = 190
  const recentCut = end - 42 * DAY_MS
  let best20 = 0
  const recentCurves: PowerCurvePoint[][] = []
  for (const { a, sport } of activities) {
    if (sport !== 'bike') continue
    const w = selectedStreams.get(String(a.id))?.watts
    if (!w || !w.some(v => v > 0)) continue
    const c = meanMaxCurve(w)
    const p20 = c.find(p => p.s === 1200)
    if (p20 && p20.w > best20) best20 = p20.w
    if (dayMs(a.startDateLocal.slice(0, 10)) >= recentCut) recentCurves.push(c)
  }
  const ftp = inputFtp ?? cache.zones?.ftp ?? (best20 > 0 ? Math.round(best20 * 0.95) : null)
  const hrBounds = inputHrBounds?.length
    ? inputHrBounds
    : cache.zones?.hr?.length
      ? cache.zones.hr
      : deriveHrBounds(hrmax)
  const powerBounds =
    inputFtp != null
      ? derivePowerBounds(inputFtp)
      : cache.zones?.power?.length
        ? cache.zones.power
        : ftp != null
          ? derivePowerBounds(ftp)
          : []

  const starts: [number, number][] = []
  for (const { a } of activities) {
    const ll = selectedStreams.get(String(a.id))?.latlng
    if (ll && ll.length >= 2) starts.push([ll[0][0], ll[0][1]])
  }
  const home = homeStart(starts)

  const details: Record<string, StravaActivityDetail> = {}
  for (const { a, sport } of activities) {
    const id = String(a.id)
    const garminMatch = garminMatches.get(id) ?? null
    details[String(a.id)] = projectDetail(
      a,
      sport,
      selectedStreams.get(id),
      heartRates.get(id) ??
        resolveActivityHeartRate(
          a,
          sport,
          selectedStreams.get(id),
          garminHeartRateMatches.get(id) ?? null,
          garmin,
        ),
      weather?.activities[id],
      cache.geo?.[String(a.id)],
      matchGarminFueling(a, sport, garmin),
      garminVerification(a, garminMatch),
      hrBounds,
      powerBounds,
      home,
    )
  }

  const health: Record<string, ActivityHealth> = {}
  if (oura) for (const [date, o] of Object.entries(oura.days)) health[date] = toHealth(o)
  if (weather)
    for (const [date, w] of Object.entries(weather.days)) {
      const h = health[date] ?? emptyHealth()
      health[date] = {
        ...h,
        windKph: h.windKph ?? w.windKph,
        windDir: h.windDir ?? w.windDir,
        windDirDeg: h.windDirDeg ?? w.windDirDeg,
        windGustKph: h.windGustKph ?? w.windGustKph,
      }
    }

  return {
    generatedAt: cache.lastSync,
    athleteId: cache.athleteId,
    totalKm: round(
      finalized.reduce((s, t) => s + t.distanceKm, 0),
      1,
    ),
    totalTimeS: activities.reduce((s, { a }) => s + a.movingTime, 0),
    totalCount: activities.length,
    totals: finalized,
    strengthTotal,
    days,
    details,
    health,
    zones: { hr: hrBounds, power: powerBounds, ftp },
    powerCurveRef: mergeMaxCurves(recentCurves),
  }
}
