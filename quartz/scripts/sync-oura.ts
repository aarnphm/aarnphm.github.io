import fs from 'node:fs/promises'
import { AdaptiveRateLimiter, fetchWithRetry } from '../plugins/stores/citations'
import {
  emptyOuraDaily,
  OuraCache,
  OuraDaily,
  ouraSleepCalendarDay,
  OuraUser,
} from '../plugins/stores/oura'
import { localIsoDayOffset } from '../util/local-date'
import { joinSegments, QUARTZ } from '../util/path'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'

const API = 'https://api.ouraring.com/v2/usercollection'
const TOKEN_URL = 'https://api.ouraring.com/oauth/token'
const CACHE_VERSION = 2
const LOOKBACK_DAYS = 365
const REFRESH_DAYS = 30
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
const limiter = new AdaptiveRateLimiter(1500, 60_000)

type Row = Record<string, unknown>

interface TokenResponse {
  access_token: string
  refresh_token: string
}

async function readCache(): Promise<OuraCache | null> {
  try {
    return JSON.parse(await fs.readFile(cacheFile, 'utf8')) as OuraCache
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
  const res = await fetch(TOKEN_URL, { method: 'POST', body })
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 300)}`)
  return (await res.json()) as TokenResponse
}

async function resolveToken(
  prev: OuraCache | null,
): Promise<{ access: string; refreshToken: string }> {
  const clientId = process.env.OURA_CLIENT_ID
  const clientSecret = process.env.OURA_CLIENT_SECRET
  const cacheTok = prev?.auth?.refreshToken
  const envTok = process.env.OURA_REFRESH_TOKEN
  if (clientId && clientSecret && (cacheTok || envTok)) {
    const sources: [string, string][] = []
    if (cacheTok) sources.push(['cache', cacheTok])
    if (envTok && envTok !== cacheTok) sources.push(['.env', envTok])
    let lastErr: unknown
    for (const [src, rt] of sources) {
      try {
        console.log(`[oura] refreshing access token (refresh_token from ${src})`)
        const token = await refresh(clientId, clientSecret, rt)
        return { access: token.access_token, refreshToken: token.refresh_token }
      } catch (err) {
        lastErr = err
        console.warn(
          `[oura] ${src} refresh_token rejected: ${err instanceof Error ? err.message : err}`,
        )
      }
    }
    throw lastErr instanceof Error ? lastErr : new Error('oura token refresh failed')
  }
  const direct = process.env.OURA_PERSONAL_ACCESS_TOKEN
  if (direct) {
    console.log('[oura] using OURA_PERSONAL_ACCESS_TOKEN directly (no client_id for refresh flow)')
    return { access: direct, refreshToken: envTok ?? '' }
  }
  throw new Error(
    'need OURA_CLIENT_ID + OURA_CLIENT_SECRET + OURA_REFRESH_TOKEN (run pnpm oura:auth), or OURA_PERSONAL_ACCESS_TOKEN',
  )
}

async function fetchRange(
  token: string,
  endpoint: string,
  start: string,
  end: string,
): Promise<Row[]> {
  const headers = { Authorization: `Bearer ${token}` }
  const rows: Row[] = []
  let nextToken = ''
  for (;;) {
    const q = new URLSearchParams({ start_date: start, end_date: end })
    if (nextToken) q.set('next_token', nextToken)
    const res = await fetchWithRetry(`${API}/${endpoint}?${q}`, { headers }, limiter)
    if (!res) throw new Error(`oura ${endpoint} fetch failed`)
    const json = (await res.json()) as { data?: Row[]; next_token?: string | null }
    if (Array.isArray(json.data)) rows.push(...json.data)
    if (!json.next_token) break
    nextToken = json.next_token
  }
  return rows
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)

async function fetchPersonalInfo(token: string): Promise<OuraUser> {
  const res = await fetchWithRetry(
    `${API}/personal_info`,
    { headers: { Authorization: `Bearer ${token}` } },
    limiter,
  )
  if (!res) return { id: null, email: null }
  const info = (await res.json()) as Record<string, unknown>
  const data = (info.data as Record<string, unknown> | undefined) ?? info
  return { id: str(data.id), email: str(data.email) }
}

async function main(): Promise<void> {
  const prev = await readCache()
  const { access, refreshToken } = await resolveToken(prev)
  const stale = (prev?.version ?? 0) < CACHE_VERSION
  const now = Date.now()
  const start = localIsoDayOffset(stale ? -LOOKBACK_DAYS : -REFRESH_DAYS, now)
  const end = localIsoDayOffset(0, now)

  const days: Record<string, OuraDaily> = {}
  if (prev?.days) for (const [k, v] of Object.entries(prev.days)) days[k] = { ...v }
  const ensure = (day: string): OuraDaily => (days[day] ??= emptyOuraDaily(day))
  let user: OuraUser = prev?.user ?? { id: null, email: null }
  const writeCache = async (): Promise<void> => {
    const cache: OuraCache = {
      version: CACHE_VERSION,
      auth: { refreshToken, obtainedAt: now },
      user,
      lastSync: now,
      days,
    }
    await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
    await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  }
  await writeCache()

  const info = await fetchPersonalInfo(access)
  if (info.id) {
    const pin = process.env.OURA_USER_ID
    if (pin && pin !== info.id)
      throw new Error(`oura account mismatch: token belongs to ${info.id}, but OURA_USER_ID=${pin}`)
    if (prev?.user?.id && prev.user.id !== info.id)
      console.warn(`[oura] account changed since last sync: ${prev.user.id} → ${info.id}`)
    user = info
    console.log(`[oura] authorized as ${info.email ?? '(email scope off)'} (id ${info.id})`)
  } else {
    console.log('[oura] personal_info unavailable; keeping prior identity')
  }

  const readiness = await fetchRange(access, 'daily_readiness', start, end)
  const dailySleep = await fetchRange(access, 'daily_sleep', start, end)
  const sleep = await fetchRange(access, 'sleep', start, end)
  const activity = await fetchRange(access, 'daily_activity', start, end)

  for (const r of readiness) {
    const day = str(r.day)
    if (!day) continue
    const d = ensure(day)
    d.readiness = num(r.score)
    d.tempDeviationC = num(r.temperature_deviation)
  }
  for (const r of dailySleep) {
    const day = str(r.day)
    if (!day) continue
    ensure(day).sleepScore = num(r.score)
  }
  const mainSleep: Record<string, Row> = {}
  for (const r of sleep) {
    const day = ouraSleepCalendarDay(r)
    if (!day) continue
    if (r.type !== undefined && r.type !== 'long_sleep') continue
    const cur = mainSleep[day]
    if (!cur || (num(r.total_sleep_duration) ?? 0) > (num(cur.total_sleep_duration) ?? 0))
      mainSleep[day] = r
  }
  for (const [day, r] of Object.entries(mainSleep)) {
    const d = ensure(day)
    d.hrv = num(r.average_hrv)
    d.rhr = num(r.lowest_heart_rate)
    d.sleepDurationS = num(r.total_sleep_duration)
  }
  for (const r of activity) {
    const day = str(r.day)
    if (!day) continue
    const d = ensure(day)
    d.totalCalories = num(r.total_calories)
    d.activeCalories = num(r.active_calories)
  }

  await writeCache()
  await refreshTriathlonRouteSource()
  const withReadiness = Object.values(days).filter(d => d.readiness != null).length
  console.log(
    `[oura] wrote ${Object.keys(days).length} days (${start} → ${end}), ${withReadiness} with readiness → ${cacheFile}`,
  )
}

main().catch(err => {
  console.error(`[oura] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
