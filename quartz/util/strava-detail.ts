import type { CriticalPowerEstimate } from '../plugins/stores/critical-power'
import type { MatchedRidesBlock } from '../plugins/stores/matched-rides'
import type { MatchedRunsBlock } from '../plugins/stores/matched-runs'
import type {
  ActivityHealth,
  PowerCurvePoint,
  StravaActivityDetail,
  StravaZones,
  SwimTrendPoint,
} from '../plugins/stores/strava'
import type { TriathlonDailyAnalytics } from './triathlon-day-analytics'

export const STRAVA_DETAIL_INDEX_KIND = 'strava-detail-index-v1'
export const STRAVA_DETAIL_SHARD_MAX_BYTES = 20 * 1024 * 1024

const STRAVA_DETAIL_SHARD_PATH = /^strava-detail\/\d{4}-\d{2}(?:-\d+)?\.json$/
const STRAVA_DETAIL_DATE = /^(\d{4}-\d{2})-\d{2}$/
const SHARD_PREFIX = '{"details":{'
const SHARD_SUFFIX = '}}'
const textEncoder = new TextEncoder()

export type StravaDetailPayload<TDetail extends { date: string } = StravaActivityDetail> = {
  details: Record<string, TDetail>
  swimTrend?: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
  dailyAnalytics?: TriathlonDailyAnalytics
  zones?: StravaZones
  powerCurveRef?: PowerCurvePoint[]
  powerCurveYearRef?: PowerCurvePoint[]
  runPowerCurveRef?: PowerCurvePoint[]
  runPowerCurveYearRef?: PowerCurvePoint[]
  powerCurveYear?: number | null
  criticalPower?: CriticalPowerEstimate | null
  criticalPowerYear?: CriticalPowerEstimate | null
  ftp?: number | null
  goalFtp?: number | null
  vt1Hr?: number | null
  matchedRuns?: MatchedRunsBlock
  matchedRides?: MatchedRidesBlock
}

export type StravaDetailIndex = Omit<StravaDetailPayload, 'details'> & {
  kind: typeof STRAVA_DETAIL_INDEX_KIND
  shards: string[]
}

export type StravaDetailShard<TDetail extends { date: string } = StravaActivityDetail> = {
  details: Record<string, TDetail>
}

export type SerializedStravaDetailShard = {
  name: string
  path: string
  content: string
  bytes: number
}

export type SerializedStravaDetails = { manifest: string; shards: SerializedStravaDetailShard[] }

const jsonBytes = (value: string): number => textEncoder.encode(value).byteLength

export const isStravaDetailShardPath = (value: unknown): value is string =>
  typeof value === 'string' && STRAVA_DETAIL_SHARD_PATH.test(value)

export function serializeStravaDetails<TDetail extends { date: string }>(
  payload: StravaDetailPayload<TDetail>,
  maxShardBytes = STRAVA_DETAIL_SHARD_MAX_BYTES,
): SerializedStravaDetails {
  if (!Number.isInteger(maxShardBytes) || maxShardBytes <= jsonBytes(SHARD_PREFIX + SHARD_SUFFIX))
    throw new Error('Strava detail shard byte limit must fit a JSON object')

  const entriesByMonth = new Map<string, string[]>()
  for (const [id, detail] of Object.entries(payload.details)) {
    const month = STRAVA_DETAIL_DATE.exec(detail.date)?.[1]
    if (!month) throw new Error(`Strava detail ${id} has invalid date ${detail.date}`)
    const serializedDetail = JSON.stringify(detail)
    if (!serializedDetail) throw new Error(`Strava detail ${id} is not JSON serializable`)
    const entry = `${JSON.stringify(id)}:${serializedDetail}`
    if (jsonBytes(SHARD_PREFIX + entry + SHARD_SUFFIX) > maxShardBytes)
      throw new Error(`Strava detail ${id} exceeds the shard byte limit`)
    const entries = entriesByMonth.get(month) ?? []
    entries.push(entry)
    entriesByMonth.set(month, entries)
  }

  const shards: SerializedStravaDetailShard[] = []
  for (const [month, entries] of [...entriesByMonth].sort(([left], [right]) =>
    right.localeCompare(left),
  )) {
    let chunk: string[] = []
    let chunkBytes = jsonBytes(SHARD_PREFIX + SHARD_SUFFIX)
    const flush = (): void => {
      if (chunk.length === 0) return
      const index = shards.filter(
        shard => shard.name === month || shard.name.startsWith(`${month}-`),
      ).length
      const name = index === 0 ? month : `${month}-${index + 1}`
      const content = SHARD_PREFIX + chunk.join(',') + SHARD_SUFFIX
      shards.push({ name, path: `strava-detail/${name}.json`, content, bytes: jsonBytes(content) })
      chunk = []
      chunkBytes = jsonBytes(SHARD_PREFIX + SHARD_SUFFIX)
    }

    for (const entry of entries) {
      const entryBytes = jsonBytes(entry) + (chunk.length > 0 ? 1 : 0)
      if (chunkBytes + entryBytes > maxShardBytes) flush()
      chunk.push(entry)
      chunkBytes += jsonBytes(entry) + (chunk.length > 1 ? 1 : 0)
    }
    flush()
  }

  const manifest: StravaDetailIndex = {
    kind: STRAVA_DETAIL_INDEX_KIND,
    shards: shards.map(shard => shard.path),
    swimTrend: payload.swimTrend,
    health: payload.health,
    dailyAnalytics: payload.dailyAnalytics,
    zones: payload.zones,
    powerCurveRef: payload.powerCurveRef,
    powerCurveYearRef: payload.powerCurveYearRef,
    runPowerCurveRef: payload.runPowerCurveRef,
    runPowerCurveYearRef: payload.runPowerCurveYearRef,
    powerCurveYear: payload.powerCurveYear,
    criticalPower: payload.criticalPower,
    criticalPowerYear: payload.criticalPowerYear,
    ftp: payload.ftp,
    goalFtp: payload.goalFtp,
    vt1Hr: payload.vt1Hr,
    matchedRuns: payload.matchedRuns,
    matchedRides: payload.matchedRides,
  }
  return { manifest: JSON.stringify(manifest), shards }
}

export const STRAVA_MAP_PAYLOAD_KIND = 'strava-map-v1'

export interface StravaMapActivityEntry {
  id: number
  name: string
  date: string
  start: string
  sport: StravaActivityDetail['sport']
  virtual?: boolean
  commute?: boolean
  distanceKm: number
  movingTimeS: number
  elapsedTimeS: number
  elevationM: number
  avgHr: number | null
  maxHr: number | null
  avgWatts: number | null
  npWatts: number | null
  maxWatts: number | null
  deviceWatts: boolean
  avgCadence: number | null
  calories: number | null
  sufferScore: number | null
  location: string | null
  device: StravaActivityDetail['device']
  avgWattsNoZero: number | null
  segs: number[][]
}

export interface StravaMapPayload {
  kind: typeof STRAVA_MAP_PAYLOAD_KIND
  activities: StravaMapActivityEntry[]
}

export function serializeStravaMapPayload(details: Record<string, StravaActivityDetail>): string {
  const activities: StravaMapActivityEntry[] = []
  for (const detail of Object.values(details)) {
    const rawSegments = detail.mapRoute.some(seg => seg.length >= 2)
      ? detail.mapRoute.filter(seg => seg.length >= 2)
      : detail.route.length >= 2
        ? [detail.route]
        : []
    if (rawSegments.length === 0) continue
    const segs = rawSegments.map(seg => {
      const flat: number[] = []
      let pLat = 0
      let pLng = 0
      let pDist = 0
      for (const point of seg) {
        const lat = Math.round(point.lat * 1_000_000)
        const lng = Math.round(point.lng * 1_000_000)
        const dist = Math.round(point.d * 10_000)
        flat.push(lat - pLat, lng - pLng, dist - pDist)
        pLat = lat
        pLng = lng
        pDist = dist
      }
      return flat
    })
    activities.push({
      id: detail.id,
      name: detail.name,
      date: detail.date,
      start: detail.start,
      sport: detail.sport,
      virtual: detail.virtual,
      commute: detail.commute,
      distanceKm: detail.distanceKm,
      movingTimeS: detail.movingTimeS,
      elapsedTimeS: detail.elapsedTimeS,
      elevationM: detail.elevationM,
      avgHr: detail.avgHr,
      maxHr: detail.maxHr,
      avgWatts: detail.avgWatts,
      npWatts: detail.npWatts,
      maxWatts: detail.maxWatts,
      deviceWatts: detail.deviceWatts,
      avgCadence:
        detail.sport === 'walk' && detail.garmin?.avgCadence != null
          ? detail.garmin.avgCadence / 2
          : detail.avgCadence,
      calories: detail.calories,
      sufferScore: detail.sufferScore,
      location: detail.location,
      device: detail.device,
      avgWattsNoZero: detail.powerWithoutZeros?.avgWatts ?? null,
      segs,
    })
  }
  const payload: StravaMapPayload = { kind: STRAVA_MAP_PAYLOAD_KIND, activities }
  return JSON.stringify(payload)
}

export function hydrateStravaMapPayload(value: unknown): StravaDetailPayload | null {
  if (
    typeof value !== 'object' ||
    value === null ||
    (value as { kind?: unknown }).kind !== STRAVA_MAP_PAYLOAD_KIND ||
    !Array.isArray((value as { activities?: unknown }).activities)
  )
    return null
  const details: Record<string, StravaActivityDetail> = {}
  for (const raw of (value as StravaMapPayload).activities) {
    if (!raw || typeof raw.id !== 'number' || !Array.isArray(raw.segs)) continue
    const mapRoute = raw.segs.map(flat => {
      const length = Math.floor(flat.length / 3) * 3
      const segment: { lat: number; lng: number; d: number }[] = []
      let lat = 0
      let lng = 0
      let dist = 0
      for (let i = 0; i < length; i += 3) {
        lat += flat[i]
        lng += flat[i + 1]
        dist += flat[i + 2]
        segment.push({ lat: lat / 1_000_000, lng: lng / 1_000_000, d: dist / 10_000 })
      }
      return segment
    })
    details[String(raw.id)] = {
      id: raw.id,
      sport: raw.sport,
      name: raw.name,
      date: raw.date,
      start: raw.start,
      virtual: raw.virtual,
      commute: raw.commute,
      distanceKm: raw.distanceKm,
      movingTimeS: raw.movingTimeS,
      elapsedTimeS: raw.elapsedTimeS,
      maxSpeedKph: null,
      elevationM: raw.elevationM ?? 0,
      avgHr: raw.avgHr,
      maxHr: raw.maxHr,
      avgWatts: raw.avgWatts,
      npWatts: raw.npWatts ?? null,
      maxWatts: raw.maxWatts,
      kilojoules: null,
      deviceWatts: raw.deviceWatts,
      avgCadence: raw.avgCadence,
      sufferScore: raw.sufferScore,
      calories: raw.calories,
      deviceTemperatureC: null,
      ambientTemperatureC: null,
      windKph: null,
      windDir: null,
      windDirDeg: null,
      windGustKph: null,
      averageRelativeHumidityPct: null,
      relativeHumidityProvenance: null,
      location: raw.location,
      fueling: null,
      strength: null,
      sauna: null,
      garmin: null,
      computer: null,
      device: raw.device,
      staminaTrace: null,
      performanceConditionTrace: null,
      calculatedIntensityFactor: null,
      calculatedExerciseLoad: null,
      anaerobicPowerEstimate: null,
      calculatedTrainingEffect: null,
      gearShifts: [],
      cyclingDynamics: null,
      runWalk: null,
      route: [],
      heartRateTrace: [],
      mapRoute,
      analysisRanges: [],
      runSplitsMetric: [],
      runSplitsStandard: [],
      runPaceZones: null,
      minAlt: 0,
      maxAlt: 0,
      descentM: 0,
      hrZones: null,
      powerZones: null,
      powerHist: null,
      powerWithoutZeros:
        raw.avgWattsNoZero != null
          ? { avgWatts: raw.avgWattsNoZero, powerZones: null, powerHist: null }
          : null,
      powerCurve: null,
      activityCriticalPower: null,
      bestEfforts: null,
      strokeCount: null,
      strokeRateSpm: null,
      swimPaceSPer100m: null,
      swimPaceSource: null,
      swimDurationS: null,
      swimIntervals: [],
      swimLocation: null,
      waterTemperatureC: null,
      analyses: {
        native: { pelotan: null, myWindsock: null, myWindsockRoute: null, myWindsockArchive: null },
        derived: { environment: null, uvScore: null, apparentWind: null },
      },
    }
  }
  return { details, health: {} }
}
