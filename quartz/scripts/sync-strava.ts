import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { AdaptiveRateLimiter, fetchWithRetry } from '../plugins/stores/citations'
import {
  hasFetchedActivityDetail,
  normalizeKind,
  RawStravaActivity,
  RawStravaActivityDetail,
  RawStravaAnalysisRange,
  RawStravaRunSplit,
  StravaRawCache,
  StravaStreams,
  StravaZones,
} from '../plugins/stores/strava'
import { upsertEnvLine } from '../util/env-file'
import { joinSegments, QUARTZ } from '../util/path'
import { readStravaCacheFile, writeStravaCacheFile } from '../util/strava-cache-file'
import { reconcileStravaActivities, stravaFetchAfter } from '../util/strava-sync-window'
import { stravaSyncRefreshDays } from '../util/sync-refresh-window'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord, readNumber, readString } from '../util/type-guards'

const TOKEN_URL = 'https://www.strava.com/oauth/token'
const DEFAULT_API_BASE_URL = 'https://www.strava.com/api/v3'
const API = normalizeApiBaseUrl(process.env.STRAVA_API_BASE_URL ?? DEFAULT_API_BASE_URL)
const PER_PAGE = 200
const CACHE_VERSION = 5
const ENV_FILE = '.env'
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const limiter = new AdaptiveRateLimiter(400, 60_000)
const geoLimiter = new AdaptiveRateLimiter(1100, 30_000)
const CONCURRENCY = 5
const DETAIL_BACKFILL_BATCH = 50
const READ_REQUEST_RESERVE = 5
const HEADERLESS_READ_REQUEST_LIMIT = 90
const configuredReadRequestLimit = Number(process.env.STRAVA_MAX_READ_REQUESTS)
const MAX_READ_REQUESTS =
  Number.isFinite(configuredReadRequestLimit) && configuredReadRequestLimit >= 1
    ? Math.floor(configuredReadRequestLimit)
    : null

interface StravaRatePair {
  short: number
  daily: number
}

let readLimit: StravaRatePair | null = null
let readUsage: StravaRatePair | null = null
let readRequests = 0
let readsInFlight = 0
let rateDeferred = false

function ratePair(value: string | null): StravaRatePair | null {
  const values = value?.split(',').map(part => Number(part.trim())) ?? []
  if (values.length !== 2 || !values.every(item => Number.isInteger(item) && item >= 0)) return null
  return { short: values[0], daily: values[1] }
}

function recordReadRate(response: Response): void {
  readLimit = ratePair(response.headers.get('X-ReadRateLimit-Limit')) ?? readLimit
  readUsage = ratePair(response.headers.get('X-ReadRateLimit-Usage')) ?? readUsage
}

function canRead(): boolean {
  if (MAX_READ_REQUESTS != null && readRequests >= MAX_READ_REQUESTS) return false
  if (!readLimit || !readUsage) return readRequests < HEADERLESS_READ_REQUEST_LIMIT
  return (
    readLimit.short - readUsage.short > READ_REQUEST_RESERVE + readsInFlight &&
    readLimit.daily - readUsage.daily > READ_REQUEST_RESERVE + readsInFlight
  )
}

async function fetchStrava(url: string, options: RequestInit): Promise<Response | null> {
  if (!canRead()) {
    rateDeferred = true
    return null
  }
  readRequests += 1
  readsInFlight += 1
  try {
    const response = await fetchWithRetry(url, options, limiter)
    if (response) recordReadRate(response)
    return response
  } finally {
    readsInFlight -= 1
  }
}

async function mapPool<T>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<void>,
): Promise<void> {
  let cursor = 0
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++
      await worker(items[index], index)
    }
  })
  await Promise.all(runners)
}

interface TokenResponse {
  access_token: string
  refresh_token: string
}

function normalizeApiBaseUrl(value: string): string {
  return new URL(value).toString().replace(/\/+$/, '')
}

function apiUrl(path: string, params: Record<string, string | number> = {}): string {
  const url = new URL(`${API}${path}`)
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, String(value))
  return url.toString()
}

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` }
}

async function readTokenResponse(res: Response): Promise<TokenResponse> {
  const raw: unknown = await res.json()
  if (!isRecord(raw)) throw new Error('token refresh returned a non-object response')
  const accessToken = readString(raw, 'access_token')
  const refreshToken = readString(raw, 'refresh_token')
  if (!accessToken || !refreshToken) throw new Error('token refresh response missing tokens')
  return { access_token: accessToken, refresh_token: refreshToken }
}

async function readCache(): Promise<StravaRawCache | null> {
  return readStravaCacheFile(cacheFile)
}

async function refresh(
  clientId: string,
  clientSecret: string,
  refreshToken: string,
): Promise<TokenResponse> {
  const body = new URLSearchParams({
    client_id: clientId,
    client_secret: clientSecret,
    grant_type: 'refresh_token',
    refresh_token: refreshToken,
  })
  const res = await fetchWithRetry(TOKEN_URL, { method: 'POST', body }, limiter)
  if (!res) throw new Error('token refresh failed (network/auth)')
  return readTokenResponse(res)
}

async function resolveToken(
  prev: StravaRawCache | null,
): Promise<{ access: string; refreshToken: string }> {
  const clientId = process.env.STRAVA_CLIENT_ID
  const clientSecret = process.env.STRAVA_CLIENT_SECRET
  const envRefreshToken = process.env.STRAVA_REFRESH_TOKEN
  const refreshToken = prev?.auth.refreshToken || envRefreshToken
  if (clientId && clientSecret && refreshToken) {
    const token = await refresh(clientId, clientSecret, refreshToken)
    if (token.refresh_token !== refreshToken) console.log('[strava] refresh token rotated')
    if (token.refresh_token !== envRefreshToken) {
      await upsertEnvLine(ENV_FILE, 'STRAVA_REFRESH_TOKEN', token.refresh_token)
      console.log('[strava] STRAVA_REFRESH_TOKEN updated in .env')
    }
    return { access: token.access_token, refreshToken: token.refresh_token }
  }
  const direct = process.env.STRAVA_ACCESS_TOKEN
  if (direct) {
    console.log('[strava] using STRAVA_ACCESS_TOKEN directly (no client_id for refresh flow)')
    return { access: direct, refreshToken: refreshToken ?? '' }
  }
  throw new Error(
    'need STRAVA_CLIENT_ID + STRAVA_CLIENT_SECRET + STRAVA_REFRESH_TOKEN, or STRAVA_ACCESS_TOKEN',
  )
}

function mapActivity(raw: Record<string, unknown>): RawStravaActivity {
  return {
    id: raw.id as number,
    name: String(raw.name ?? ''),
    sportType: String(raw.sport_type ?? raw.type ?? ''),
    distance: Number(raw.distance ?? 0),
    movingTime: Number(raw.moving_time ?? 0),
    elapsedTime: Number(raw.elapsed_time ?? 0),
    totalElevationGain: Number(raw.total_elevation_gain ?? 0),
    startDate: String(raw.start_date ?? ''),
    startDateLocal: String(raw.start_date_local ?? raw.start_date ?? ''),
    averageSpeed: Number(raw.average_speed ?? 0),
    averageHeartrate:
      raw.average_heartrate === undefined ? undefined : Number(raw.average_heartrate),
    maxHeartrate: raw.max_heartrate === undefined ? undefined : Number(raw.max_heartrate),
    averageWatts: raw.average_watts === undefined ? undefined : Number(raw.average_watts),
    weightedAverageWatts:
      raw.weighted_average_watts === undefined ? undefined : Number(raw.weighted_average_watts),
    maxWatts: raw.max_watts === undefined ? undefined : Number(raw.max_watts),
    kilojoules: raw.kilojoules === undefined ? undefined : Number(raw.kilojoules),
    deviceWatts: raw.device_watts === undefined ? undefined : Boolean(raw.device_watts),
    averageCadence: raw.average_cadence === undefined ? undefined : Number(raw.average_cadence),
    sufferScore: raw.suffer_score === undefined ? undefined : Number(raw.suffer_score),
    averageTemp: raw.average_temp === undefined ? undefined : Number(raw.average_temp),
  }
}

async function fetchActivities(
  token: string,
  after: number,
): Promise<{ activities: RawStravaActivity[]; athleteId: number }> {
  const headers = authHeaders(token)
  const activities: RawStravaActivity[] = []
  let athleteId = 0
  for (let page = 1; ; page++) {
    const url = apiUrl('/athlete/activities', { after, per_page: PER_PAGE, page })
    const res = await fetchStrava(url, { headers })
    if (!res) throw new Error(`activity fetch failed at page ${page}`)
    const batch = (await res.json()) as Record<string, unknown>[]
    if (!Array.isArray(batch) || batch.length === 0) break
    for (const raw of batch) {
      const athlete = raw.athlete as { id?: number } | undefined
      if (athlete?.id) athleteId = athlete.id
      activities.push(mapActivity(raw))
    }
    console.log(`[strava] page ${page}: ${batch.length} activities`)
    if (batch.length < PER_PAGE) break
  }
  return { activities, athleteId }
}

async function fetchStreams(token: string, id: number): Promise<StravaStreams | null> {
  const headers = authHeaders(token)
  const url = apiUrl(`/activities/${id}/streams`, {
    keys: 'time,latlng,altitude,distance,watts,heartrate,cadence',
    key_by_type: 'true',
  })
  const res = await fetchStrava(url, { headers })
  if (!res) return null
  const data = (await res.json()) as Record<string, { data?: unknown[] }>
  return {
    time: (data.time?.data as number[]) ?? [],
    latlng: (data.latlng?.data as [number, number][]) ?? [],
    altitude: (data.altitude?.data as number[]) ?? [],
    distance: (data.distance?.data as number[]) ?? [],
    watts: (data.watts?.data as number[]) ?? [],
    heartrate: (data.heartrate?.data as number[]) ?? [],
    cadence: (data.cadence?.data as number[]) ?? [],
  }
}

async function fetchAthleteFtp(token: string): Promise<number | null> {
  const res = await fetchStrava(apiUrl('/athlete'), { headers: authHeaders(token) })
  if (!res || !res.ok) return null
  const data = (await res.json()) as { ftp?: number | null }
  return typeof data.ftp === 'number' && data.ftp > 0 ? Math.round(data.ftp) : null
}

async function fetchZones(token: string): Promise<StravaZones | null> {
  const res = await fetchStrava(apiUrl('/athlete/zones'), { headers: authHeaders(token) })
  const ftp = await fetchAthleteFtp(token)
  if (!res || !res.ok) return ftp != null ? { hr: [], power: [], ftp } : null
  const data = (await res.json()) as {
    heart_rate?: { zones?: { min: number; max: number }[] }
    power?: { zones?: { min: number; max: number }[] }
  }
  const bounds = (zones: { max: number }[] | undefined): number[] =>
    (zones ?? []).map(z => z.max).filter(m => m > 0)
  const hr = bounds(data.heart_rate?.zones)
  const power = bounds(data.power?.zones)
  if (hr.length === 0 && power.length === 0 && ftp == null) return null
  return { hr, power, ftp }
}

function nullableNumber(record: Record<string, unknown>, key: string): number | null {
  const value = readNumber(record, key)
  return value != null && Number.isFinite(value) ? value : null
}

function parseAnalysisRange(
  value: unknown,
  fallbackKind: 'lap' | 'segment',
  index: number,
): RawStravaAnalysisRange | null {
  if (!isRecord(value)) return null
  const elapsedTime = nullableNumber(value, 'elapsed_time')
  const distance = nullableNumber(value, 'distance')
  if (elapsedTime == null || elapsedTime < 0 || distance == null || distance < 0) return null
  const rawId = value.id
  const id =
    typeof rawId === 'number' || typeof rawId === 'string'
      ? String(rawId)
      : `${fallbackKind}-${index + 1}`
  const name =
    readString(value, 'name')?.trim() ||
    `${fallbackKind === 'lap' ? 'Lap' : 'Segment'} ${index + 1}`
  return {
    id,
    name,
    elapsedTime,
    movingTime: nullableNumber(value, 'moving_time') ?? elapsedTime,
    startDate: readString(value, 'start_date') ?? null,
    distance,
    startIndex: nullableNumber(value, 'start_index'),
    endIndex: nullableNumber(value, 'end_index'),
    totalElevationGain: nullableNumber(value, 'total_elevation_gain'),
    averageSpeed: nullableNumber(value, 'average_speed'),
    averageHeartrate: nullableNumber(value, 'average_heartrate'),
    averageWatts: nullableNumber(value, 'average_watts'),
    averageCadence: nullableNumber(value, 'average_cadence'),
  }
}

function parseAnalysisRanges(value: unknown, kind: 'lap' | 'segment'): RawStravaAnalysisRange[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item, index) => {
    const range = parseAnalysisRange(item, kind, index)
    return range ? [range] : []
  })
}

function parseRunSplit(value: unknown, index: number): RawStravaRunSplit | null {
  if (!isRecord(value)) return null
  const distance = nullableNumber(value, 'distance')
  const elapsedTime = nullableNumber(value, 'elapsed_time')
  const movingTime = nullableNumber(value, 'moving_time') ?? elapsedTime
  if (
    distance == null ||
    distance <= 0 ||
    elapsedTime == null ||
    elapsedTime <= 0 ||
    movingTime == null ||
    movingTime <= 0
  )
    return null
  const averageSpeed = nullableNumber(value, 'average_speed') ?? distance / movingTime
  if (!Number.isFinite(averageSpeed) || averageSpeed <= 0) return null
  return {
    split: Math.max(1, Math.round(nullableNumber(value, 'split') ?? index + 1)),
    distance,
    elapsedTime,
    movingTime,
    averageSpeed,
    elevationDifference: nullableNumber(value, 'elevation_difference'),
    paceZone: nullableNumber(value, 'pace_zone'),
  }
}

export function parseRunSplits(value: unknown): RawStravaRunSplit[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item, index) => {
    const split = parseRunSplit(item, index)
    return split ? [split] : []
  })
}

async function fetchActivityDetail(
  token: string,
  id: number,
): Promise<RawStravaActivityDetail | null> {
  const res = await fetchStrava(apiUrl(`/activities/${id}`, { include_all_efforts: 'true' }), {
    headers: authHeaders(token),
  })
  if (!res) return null
  const data: unknown = await res.json()
  if (!isRecord(data)) return null
  return {
    description: readString(data, 'description') ?? null,
    fetchedAt: Date.now(),
    calories: nullableNumber(data, 'calories'),
    laps: parseAnalysisRanges(data.laps, 'lap'),
    segmentEfforts: parseAnalysisRanges(data.segment_efforts, 'segment'),
    splitsMetric: parseRunSplits(data.splits_metric),
    splitsStandard: parseRunSplits(data.splits_standard),
  }
}

async function fetchCity(lat: number, lon: number): Promise<string | null> {
  const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&zoom=10&addressdetails=1`
  const res = await fetchWithRetry(
    url,
    { headers: { 'User-Agent': 'aarnphm-garden-strava-sync/1.0' } },
    geoLimiter,
  )
  if (!res) return null
  const data = (await res.json()) as { address?: Record<string, string> }
  const a = data.address ?? {}
  return a.city || a.town || a.village || a.municipality || a.county || a.state || null
}

function progress(label: string, done: number, total: number): void {
  const width = 22
  const filled = Math.round((done / total) * width)
  const bar = '█'.repeat(filled) + '░'.repeat(width - filled)
  process.stdout.write(`\r[strava] ${label.padEnd(8)} [${bar}] ${done}/${total}`)
  if (done >= total) process.stdout.write('\n')
}

async function main(): Promise<void> {
  const prev = await readCache()
  const { access, refreshToken } = await resolveToken(prev)
  const stale = (prev?.version ?? 0) < CACHE_VERSION
  if (stale && prev)
    console.log('[strava] cache schema bumped → re-pulling all summaries to backfill')
  const refreshWindowDays = stravaSyncRefreshDays()
  const after = stravaFetchAfter(prev?.lastActivityStart, stale, refreshWindowDays)
  if (!stale && after < (prev?.lastActivityStart ?? 0))
    console.log(`[strava] refreshing ${refreshWindowDays}d recent activity overlap`)
  const { activities, athleteId } = await fetchActivities(access, after)

  const reconciled = reconcileStravaActivities(prev?.activities, activities, after)
  const merged = reconciled.activities

  let lastActivityStart = 0
  for (const a of Object.values(merged)) {
    const epoch = Math.floor(Date.parse(a.startDate) / 1000)
    if (Number.isFinite(epoch) && epoch > lastActivityStart) lastActivityStart = epoch
  }

  const streams: Record<string, StravaStreams> = { ...prev?.streams }
  const geo: Record<string, string> = { ...prev?.geo }
  const activityDetails: Record<string, RawStravaActivityDetail> = { ...prev?.activityDetails }
  for (const id of reconciled.removedIds) {
    delete streams[id]
    delete geo[id]
    delete activityDetails[id]
  }
  for (const id of Object.keys(activityDetails))
    if (merged[id] === undefined) delete activityDetails[id]
  let zones: StravaZones | undefined = prev?.zones
  const writeCache = async (): Promise<void> => {
    const cache: StravaRawCache = {
      version: CACHE_VERSION,
      athleteId: athleteId || prev?.athleteId || 0,
      auth: { refreshToken, obtainedAt: Date.now() },
      lastSync: Date.now(),
      lastActivityStart,
      activities: Object.fromEntries(
        Object.entries(merged).sort(([a], [b]) => Number(a) - Number(b)),
      ),
      activityDetails,
      streams,
      geo,
      zones,
    }
    await writeStravaCacheFile(cacheFile, cache)
  }
  let writing: Promise<void> | null = null
  const checkpoint = (): void => {
    if (writing) return
    writing = writeCache().finally(() => {
      writing = null
    })
  }

  const fetchedZones = await fetchZones(access)
  if (fetchedZones) zones = fetchedZones
  else if (!zones) console.log('[strava] no athlete zones (needs profile:read_all) — deriving')

  const needStreams = Object.values(merged)
    .filter(a => {
      if (normalizeKind(a.sportType) === null) return false
      const s = streams[String(a.id)]
      return !s || s.heartrate === undefined || !s.time?.length
    })
    .sort((x, y) => y.startDate.localeCompare(x.startDate))
  let si = 0
  await mapPool(needStreams, CONCURRENCY, async a => {
    const s = await fetchStreams(access, a.id)
    if (s) streams[String(a.id)] = s
    progress('streams', ++si, needStreams.length)
    if (si % 16 === 0) checkpoint()
  })

  const needGeo = Object.values(merged).filter(a => {
    const s = streams[String(a.id)]
    return (
      normalizeKind(a.sportType) !== null &&
      s !== undefined &&
      s.latlng.length >= 2 &&
      !geo[String(a.id)]
    )
  })
  let gi = 0
  for (const a of needGeo) {
    const s = streams[String(a.id)]!
    const city = await fetchCity(s.latlng[0][0], s.latlng[0][1])
    if (city) geo[String(a.id)] = city
    progress('geocode', ++gi, needGeo.length)
    if (gi % 8 === 0) checkpoint()
  }

  const detailCandidates = Object.values(merged)
    .filter(a => normalizeKind(a.sportType) !== null)
    .sort((left, right) => right.startDate.localeCompare(left.startDate))
  const recentDetailCutoff = Date.now() - refreshWindowDays * 86_400_000
  const recentDetails = detailCandidates
    .filter(activity => Date.parse(activity.startDate) >= recentDetailCutoff)
    .sort(
      (left, right) =>
        (activityDetails[String(left.id)]?.fetchedAt ?? 0) -
          (activityDetails[String(right.id)]?.fetchedAt ?? 0) ||
        right.startDate.localeCompare(left.startDate),
    )
  const recentIds = new Set(recentDetails.map(activity => activity.id))
  const historicalBackfill = detailCandidates
    .filter(
      activity =>
        !recentIds.has(activity.id) &&
        !hasFetchedActivityDetail(activityDetails[String(activity.id)]),
    )
    .slice(0, DETAIL_BACKFILL_BATCH)
  const needDetails = [...recentDetails, ...historicalBackfill]
  let di = 0
  await mapPool(needDetails, CONCURRENCY, async a => {
    const detail = await fetchActivityDetail(access, a.id)
    if (detail) {
      activityDetails[String(a.id)] = detail
      if (detail.calories != null) merged[String(a.id)].calories = detail.calories
    }
    progress('details', ++di, needDetails.length)
    if (di % 16 === 0) checkpoint()
  })

  if (writing) await writing
  await writeCache()
  await refreshTriathlonRouteSource()
  if (rateDeferred) {
    const usage =
      readLimit && readUsage
        ? `; Strava read usage ${readUsage.short}/${readLimit.short} short, ${readUsage.daily}/${readLimit.daily} daily`
        : ''
    console.log(
      `[strava] deferred requests after ${readRequests} reads${MAX_READ_REQUESTS == null ? '' : `/${MAX_READ_REQUESTS}`}${usage}`,
    )
  }
  console.log(
    `[strava] wrote ${Object.keys(merged).length} activities (+${activities.length} new), ${Object.keys(streams).length} streams, ${Object.keys(activityDetails).length} details, ${Object.keys(geo).length} located → ${cacheFile}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[strava] sync failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
