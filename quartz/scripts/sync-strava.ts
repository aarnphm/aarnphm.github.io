import fs from 'node:fs/promises'
import { AdaptiveRateLimiter, fetchWithRetry } from '../plugins/stores/citations'
import {
  normalizeSport,
  RawStravaActivity,
  StravaRawCache,
  StravaStreams,
} from '../plugins/stores/strava'
import { upsertEnvLine } from '../util/env-file'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, readString } from '../util/type-guards'

const TOKEN_URL = 'https://www.strava.com/oauth/token'
const DEFAULT_API_BASE_URL = 'https://www.strava.com/api/v3'
const API = normalizeApiBaseUrl(process.env.STRAVA_API_BASE_URL ?? DEFAULT_API_BASE_URL)
const PER_PAGE = 200
const CACHE_VERSION = 1
const ENV_FILE = '.env'
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const limiter = new AdaptiveRateLimiter(1500, 60_000)

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
  try {
    return JSON.parse(await fs.readFile(cacheFile, 'utf8')) as StravaRawCache
  } catch {
    return null
  }
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
    const res = await fetchWithRetry(url, { headers }, limiter)
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
    keys: 'latlng,altitude,distance,watts,heartrate,cadence',
    key_by_type: 'true',
  })
  const res = await fetchWithRetry(url, { headers }, limiter)
  if (!res) return null
  const data = (await res.json()) as Record<string, { data?: unknown[] }>
  return {
    latlng: (data.latlng?.data as [number, number][]) ?? [],
    altitude: (data.altitude?.data as number[]) ?? [],
    distance: (data.distance?.data as number[]) ?? [],
    watts: (data.watts?.data as number[]) ?? [],
    heartrate: (data.heartrate?.data as number[]) ?? [],
    cadence: (data.cadence?.data as number[]) ?? [],
  }
}

async function fetchCity(lat: number, lon: number): Promise<string | null> {
  const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&zoom=10&addressdetails=1`
  const res = await fetchWithRetry(
    url,
    { headers: { 'User-Agent': 'aarnphm-garden-strava-sync/1.0' } },
    limiter,
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
  process.stdout.write(`\r[strava] ${label.padEnd(7)} [${bar}] ${done}/${total}`)
  if (done >= total) process.stdout.write('\n')
}

async function main(): Promise<void> {
  const prev = await readCache()
  const { access, refreshToken } = await resolveToken(prev)
  const stale = (prev?.version ?? 0) < CACHE_VERSION
  if (stale && prev)
    console.log('[strava] cache schema bumped → re-pulling all summaries to backfill')
  const { activities, athleteId } = await fetchActivities(
    access,
    stale ? 0 : (prev?.lastActivityStart ?? 0),
  )

  const merged: Record<string, RawStravaActivity> = { ...prev?.activities }
  for (const a of activities) merged[String(a.id)] = a

  let lastActivityStart = prev?.lastActivityStart ?? 0
  for (const a of Object.values(merged)) {
    const epoch = Math.floor(Date.parse(a.startDate) / 1000)
    if (Number.isFinite(epoch) && epoch > lastActivityStart) lastActivityStart = epoch
  }

  const streams: Record<string, StravaStreams> = { ...prev?.streams }
  const geo: Record<string, string> = { ...prev?.geo }
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
      streams,
      geo,
    }
    await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
    await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  }

  const needStreams = Object.values(merged)
    .filter(a => {
      if (normalizeSport(a.sportType) === null) return false
      const s = streams[String(a.id)]
      return !s || s.heartrate === undefined
    })
    .sort((x, y) => y.startDate.localeCompare(x.startDate))
  let si = 0
  for (const a of needStreams) {
    const s = await fetchStreams(access, a.id)
    if (s) streams[String(a.id)] = s
    progress('streams', ++si, needStreams.length)
    if (si % 8 === 0) await writeCache()
  }

  const needGeo = Object.values(merged).filter(a => {
    const s = streams[String(a.id)]
    return (
      normalizeSport(a.sportType) !== null &&
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
    if (gi % 8 === 0) await writeCache()
  }

  await writeCache()
  console.log(
    `[strava] wrote ${Object.keys(merged).length} activities (+${activities.length} new), ${Object.keys(streams).length} streams, ${Object.keys(geo).length} located → ${cacheFile}`,
  )
}

main().catch(err => {
  console.error(`[strava] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
