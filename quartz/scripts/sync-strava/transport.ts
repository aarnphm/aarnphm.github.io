import { AdaptiveRateLimiter, fetchWithRetry } from '../../plugins/stores/citations'
import { StravaRawCache } from '../../plugins/stores/strava'
import { upsertEnvLine } from '../../util/env-file'
import { isRecord, readString } from '../../util/type-guards'

const TOKEN_URL = 'https://www.strava.com/oauth/token'

const DEFAULT_API_BASE_URL = 'https://www.strava.com/api/v3'

const API = normalizeApiBaseUrl(process.env.STRAVA_API_BASE_URL ?? DEFAULT_API_BASE_URL)

const ENV_FILE = '.env'

const limiter = new AdaptiveRateLimiter(400, 60_000)

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

export async function fetchStrava(url: string, options: RequestInit): Promise<Response | null> {
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

interface TokenResponse {
  access_token: string
  refresh_token: string
}

function normalizeApiBaseUrl(value: string): string {
  return new URL(value).toString().replace(/\/+$/, '')
}

export function apiUrl(path: string, params: Record<string, string | number> = {}): string {
  const url = new URL(`${API}${path}`)
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, String(value))
  return url.toString()
}

export function authHeaders(token: string): HeadersInit {
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

export async function resolveToken(
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

export function logDeferredReads(): void {
  if (rateDeferred) {
    const usage =
      readLimit && readUsage
        ? `; Strava read usage ${readUsage.short}/${readLimit.short} short, ${readUsage.daily}/${readLimit.daily} daily`
        : ''
    console.log(
      `[strava] deferred requests after ${readRequests} reads${MAX_READ_REQUESTS == null ? '' : `/${MAX_READ_REQUESTS}`}${usage}`,
    )
  }
}

export const geoLimiter = new AdaptiveRateLimiter(1100, 30_000)

export async function mapPool<T>(
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

export async function fetchCity(lat: number, lon: number): Promise<string | null> {
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
