import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { AdaptiveRateLimiter, fetchWithRetry } from '../plugins/stores/citations'
import {
  emptyOuraDaily,
  OuraCache,
  OuraDaily,
  OuraDayDetail,
  OuraHeartRateSample,
  OuraHeartRateSource,
  OuraNap,
  OuraSeries,
  ouraSleepCalendarDay,
  OuraUser,
} from '../plugins/stores/oura'
import { localDayStartUtcMs, localIsoDayOffset } from '../util/local-date'
import {
  applyOuraHealthRow,
  emptyOuraHealth,
  ouraHealthCollections,
  type OuraHealthCollection,
} from '../util/oura-health'
import { joinSegments, QUARTZ } from '../util/path'
import { calendarRefreshStart, syncRefreshDays } from '../util/sync-refresh-window'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord } from '../util/type-guards'

const API = 'https://api.ouraring.com/v2/usercollection'
const TOKEN_URL = 'https://api.ouraring.com/oauth/token'
const CACHE_VERSION = 7
const LOOKBACK_DAYS = 365
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
    const json: unknown = await res.json()
    if (!isRecord(json)) throw new Error(`oura ${endpoint} returned an invalid response`)
    if (Array.isArray(json.data)) rows.push(...json.data.filter(isRecord))
    if (typeof json.next_token !== 'string' || !json.next_token) break
    nextToken = json.next_token
  }
  return rows
}

async function fetchDateTimeRange(
  token: string,
  endpoint: string,
  startMs: number,
  endMs: number,
): Promise<Row[]> {
  const headers = { Authorization: `Bearer ${token}` }
  const rows: Row[] = []
  let nextToken = ''
  for (;;) {
    const q = new URLSearchParams({
      start_datetime: new Date(startMs).toISOString(),
      end_datetime: new Date(endMs).toISOString(),
    })
    if (nextToken) q.set('next_token', nextToken)
    const res = await fetchWithRetry(`${API}/${endpoint}?${q}`, { headers }, limiter)
    if (!res) throw new Error(`oura ${endpoint} fetch failed`)
    const json: unknown = await res.json()
    if (!isRecord(json)) throw new Error(`oura ${endpoint} returned an invalid response`)
    if (Array.isArray(json.data)) rows.push(...json.data.filter(isRecord))
    if (typeof json.next_token !== 'string' || !json.next_token) break
    nextToken = json.next_token
  }
  return rows
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const positive = (v: unknown): number | null => {
  const value = num(v)
  return value != null && value > 0 ? value : null
}
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)

const HEART_RATE_SOURCES: readonly OuraHeartRateSource[] = [
  'awake',
  'workout',
  'rest',
  'sleep',
  'live',
  'session',
]

const isHeartRateSource = (value: string): value is OuraHeartRateSource =>
  HEART_RATE_SOURCES.some(source => source === value)

function heartRateSample(value: Row): OuraHeartRateSample | null {
  const timestamp = str(value.timestamp)
  const bpm = num(value.bpm)
  const source = str(value.source)
  if (
    !timestamp ||
    !Number.isFinite(Date.parse(timestamp)) ||
    bpm == null ||
    bpm <= 0 ||
    !source ||
    !isHeartRateSource(source)
  )
    return null
  return { timestamp, bpm, source }
}

function sampleSeries(v: unknown): OuraSeries | null {
  if (!isRecord(v)) return null
  const startTs = str(v.timestamp)
  const intervalS = num(v.interval)
  if (!startTs || intervalS == null || !Array.isArray(v.items)) return null
  return { startTs, intervalS, items: v.items.map(positive) }
}

function contributors(v: unknown): Record<string, number | null> | null {
  if (!isRecord(v)) return null
  const entries = Object.entries(v)
  if (!entries.length) return null
  const out: Record<string, number | null> = {}
  for (const [k, val] of entries) out[k] = num(val)
  return out
}

function emptyDetail(date: string): OuraDayDetail {
  return {
    date,
    bedtimeStart: null,
    bedtimeEnd: null,
    phase5Min: null,
    efficiency: null,
    latencyS: null,
    timeInBedS: null,
    totalSleepS: null,
    deepS: null,
    lightS: null,
    remS: null,
    awakeS: null,
    avgBreath: null,
    avgHr: null,
    avgHrv: null,
    lowestHr: null,
    restlessPeriods: null,
    hrv: null,
    hr: null,
    readinessScore: null,
    readinessContrib: null,
    sleepScore: null,
    sleepContrib: null,
  }
}

const detailEmpty = (d: OuraDayDetail): boolean =>
  Object.entries(d).every(([k, v]) => k === 'date' || v == null)

function sleepDetail(day: string, r: Row): OuraDayDetail {
  return {
    ...emptyDetail(day),
    bedtimeStart: str(r.bedtime_start),
    bedtimeEnd: str(r.bedtime_end),
    phase5Min: str(r.sleep_phase_5_min),
    phase30Sec: str(r.sleep_phase_30_sec),
    movement30Sec: str(r.movement_30_sec),
    lowBatteryAlert: r.low_battery_alert === true,
    sleepAlgorithmVersion: str(r.sleep_algorithm_version),
    efficiency: num(r.efficiency),
    latencyS: num(r.latency),
    timeInBedS: num(r.time_in_bed),
    totalSleepS: num(r.total_sleep_duration),
    deepS: num(r.deep_sleep_duration),
    lightS: num(r.light_sleep_duration),
    remS: num(r.rem_sleep_duration),
    awakeS: num(r.awake_time),
    avgBreath: positive(r.average_breath),
    avgHr: positive(r.average_heart_rate),
    avgHrv: positive(r.average_hrv),
    lowestHr: positive(r.lowest_heart_rate),
    restlessPeriods: num(r.restless_periods),
    hrv: sampleSeries(r.hrv),
    hr: sampleSeries(r.heart_rate),
  }
}

export function applyOuraSleepRows(
  rows: readonly Row[],
  days: Record<string, OuraDaily>,
  details: Record<string, OuraDayDetail>,
  start: string,
  end: string,
): void {
  const ensureDetail = (day: string): OuraDayDetail => (details[day] ??= emptyDetail(day))
  // Replace the refreshed window so deleted/reclassified naps disappear on the next sync.
  for (const [day, detail] of Object.entries(details))
    if (day >= start && day <= end) detail.naps = []
  const mainSleep = new Map<string, Row>()
  const naps = new Map<string, OuraNap>()
  for (const row of rows) {
    const day = ouraSleepCalendarDay(row)
    if (!day || day < start || day > end) continue
    ensureDetail(day).naps ??= []
    if (row.type === 'sleep' || row.type === 'late_nap') {
      const detail = sleepDetail(day, row)
      const id = str(row.id)
      if (
        !id ||
        !detail.bedtimeStart ||
        !detail.bedtimeEnd ||
        !Number.isFinite(Date.parse(detail.bedtimeStart)) ||
        Date.parse(detail.bedtimeEnd) <= Date.parse(detail.bedtimeStart) ||
        !Number.isFinite(Date.parse(detail.bedtimeEnd)) ||
        (detail.totalSleepS ?? 0) <= 0
      )
        continue
      naps.set(id, {
        ...detail,
        id,
        type: row.type,
        reportedDay: str(row.day),
        sleepScoreDelta: num(row.sleep_score_delta),
        readinessScoreDelta: num(row.readiness_score_delta),
      })
      continue
    }
    if (row.type !== undefined && row.type !== null && row.type !== 'long_sleep') continue
    const current = mainSleep.get(day)
    if (!current || (num(row.total_sleep_duration) ?? 0) > (num(current.total_sleep_duration) ?? 0))
      mainSleep.set(day, row)
  }
  for (const [day, row] of mainSleep) {
    const detail = ensureDetail(day)
    const { readinessScore, readinessContrib, sleepScore, sleepContrib, health } = detail
    Object.assign(detail, sleepDetail(day, row), {
      readinessScore,
      readinessContrib,
      sleepScore,
      sleepContrib,
      health,
    })
    const daily = (days[day] ??= emptyOuraDaily(day))
    daily.hrv = detail.avgHrv
    daily.rhr = detail.lowestHr
    daily.sleepDurationS = detail.totalSleepS
  }
  for (const nap of [...naps.values()].sort(
    (a, b) => Date.parse(a.bedtimeStart ?? '') - Date.parse(b.bedtimeStart ?? ''),
  )) {
    days[nap.date] ??= emptyOuraDaily(nap.date)
    ensureDetail(nap.date).naps?.push(nap)
  }
}

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

export interface OuraRefreshRange {
  start: string
  end: string
  endExclusive: string
  heartRateStart: string
}

export function ouraRefreshRange(
  stale: boolean,
  refreshWindowDays: number,
  now = Date.now(),
): OuraRefreshRange {
  return {
    start: stale
      ? localIsoDayOffset(-LOOKBACK_DAYS, now)
      : calendarRefreshStart(refreshWindowDays, now),
    end: localIsoDayOffset(0, now),
    endExclusive: localIsoDayOffset(1, now),
    heartRateStart: calendarRefreshStart(refreshWindowDays, now),
  }
}

async function main(): Promise<void> {
  const prev = await readCache()
  const { access, refreshToken } = await resolveToken(prev)
  const stale = (prev?.version ?? 0) < CACHE_VERSION
  const now = Date.now()
  const refreshWindowDays = syncRefreshDays()
  const { start, end, endExclusive, heartRateStart } = ouraRefreshRange(
    stale,
    refreshWindowDays,
    now,
  )

  const days: Record<string, OuraDaily> = {}
  if (prev?.days) for (const [k, v] of Object.entries(prev.days)) days[k] = { ...v }
  const ensure = (day: string): OuraDaily => (days[day] ??= emptyOuraDaily(day))
  const details: Record<string, OuraDayDetail> = {}
  if (prev?.details) for (const [k, v] of Object.entries(prev.details)) details[k] = { ...v }
  const ensureDetail = (day: string): OuraDayDetail => (details[day] ??= emptyDetail(day))
  const applyHealth = (collection: OuraHealthCollection, rows: readonly Row[]): void => {
    for (const [date, detail] of Object.entries(details)) {
      if (date < start || date > end || !detail.health) continue
      applyOuraHealthRow(detail.health, collection, {})
      detail.health.failedCollections = detail.health.failedCollections?.filter(
        key => key !== collection,
      )
    }
    for (const row of rows) {
      const date = str(row.day)
      if (!date || date < start || date > end) continue
      const detail = ensureDetail(date)
      applyOuraHealthRow((detail.health ??= emptyOuraHealth(date)), collection, row)
      days[date] ??= emptyOuraDaily(date)
    }
  }
  let heartRate = prev?.heartRate?.slice() ?? []
  let user: OuraUser = prev?.user ?? { id: null, email: null }
  let cacheVersion = prev?.version
  let lastSync = prev?.lastSync ?? 0
  const writeCache = async (): Promise<void> => {
    const cache: OuraCache = {
      version: cacheVersion,
      auth: { refreshToken, obtainedAt: now },
      user,
      lastSync,
      days,
      details,
      heartRate,
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
  const sleep = await fetchRange(access, 'sleep', start, endExclusive)
  const activity = await fetchRange(access, 'daily_activity', start, endExclusive)
  const heartRateStartMs = localDayStartUtcMs(heartRateStart)
  const heartRateEndMs = now
  const refreshedHeartRate = (
    await fetchDateTimeRange(access, 'heartrate', heartRateStartMs, heartRateEndMs)
  )
    .map(heartRateSample)
    .filter((sample): sample is OuraHeartRateSample => sample != null)
  const heartRateByTimestamp = new Map(
    heartRate
      .filter(sample => Date.parse(sample.timestamp) < heartRateStartMs)
      .map(sample => [sample.timestamp, sample]),
  )
  for (const sample of refreshedHeartRate) heartRateByTimestamp.set(sample.timestamp, sample)
  heartRate = [...heartRateByTimestamp.values()].sort((left, right) =>
    left.timestamp.localeCompare(right.timestamp),
  )

  for (const r of readiness) {
    const day = str(r.day)
    if (!day) continue
    const d = ensure(day)
    d.readiness = num(r.score)
    d.tempDeviationC = num(r.temperature_deviation)
    const dd = ensureDetail(day)
    dd.readinessScore = num(r.score)
    dd.readinessContrib = contributors(r.contributors)
  }
  for (const r of dailySleep) {
    const day = str(r.day)
    if (!day) continue
    ensure(day).sleepScore = num(r.score)
    const dd = ensureDetail(day)
    dd.sleepScore = num(r.score)
    dd.sleepContrib = contributors(r.contributors)
  }
  applyOuraSleepRows(sleep, days, details, start, end)
  for (const r of activity) {
    const day = str(r.day)
    if (!day) continue
    const d = ensure(day)
    d.totalCalories = num(r.total_calories)
    d.activeCalories = num(r.active_calories)
  }

  applyHealth('daily_activity', activity)
  applyHealth('daily_readiness', readiness)
  for (const collection of ouraHealthCollections) {
    try {
      const rows = await fetchRange(access, collection, start, endExclusive)
      applyHealth(collection, rows)
      console.log(`[oura] ${collection}: ${rows.length} records`)
    } catch (error) {
      console.warn(
        `[oura] ${collection}: ${error instanceof Error ? error.message : 'refresh failed'}; keeping cached values`,
      )
      for (const [date, detail] of Object.entries(details)) {
        if (date < start || date > end) continue
        const health = (detail.health ??= emptyOuraHealth(date))
        health.failedCollections = [...new Set([...(health.failedCollections ?? []), collection])]
      }
    }
  }

  for (const [day, dd] of Object.entries(details)) if (detailEmpty(dd)) delete details[day]

  cacheVersion = CACHE_VERSION
  lastSync = now
  await writeCache()
  await refreshTriathlonRouteSource()
  const withReadiness = Object.values(days).filter(d => d.readiness != null).length
  console.log(
    `[oura] wrote ${Object.keys(days).length} days (${start} → ${end}), ${withReadiness} with readiness, ${heartRate.length} heart-rate samples → ${cacheFile}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[oura] sync failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
