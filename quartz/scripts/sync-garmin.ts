import fs from 'node:fs/promises'
import type {
  GarminActivity,
  GarminCache,
  GarminStreams,
  GarminVo2Day,
} from '../plugins/stores/garmin'
import { browserCookieHeaders } from '../util/browser-cookie'
import {
  garminConnectActivities,
  garminConnectActivity,
  garminConnectStreams,
  garminConnectVo2,
  type GarminConnectActivityListItem,
} from '../util/garmin-connect'
import { joinSegments, QUARTZ } from '../util/path'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord, type UnknownRecord } from '../util/type-guards'

const CACHE_VERSION = 3
const CONNECT_ORIGIN = 'https://connect.garmin.com'
const DEFAULT_CONNECT_BASE = `${CONNECT_ORIGIN}/gc-api`
const DEFAULT_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.7827.53 Safari/537.36'
const DEFAULT_PAGE_SIZE = 100
const DEFAULT_DELAY_MS = 1200
const DAY_MS = 86_400_000
const MAX_DOCUMENT_REDIRECTS = 8
const TRIATHLON_PAGE = joinSegments(QUARTZ, '..', 'content', 'triathlon.md')
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')

interface GarminConnectSession {
  cookie: string
  csrf: string
}

interface CookieCandidate {
  cookie: string
  source: string
  type: 'env' | 'file' | 'browser'
}

function cleanBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

function envNumber(name: string, fallback: number): number {
  const value = process.env[name]
  if (!value?.trim()) return fallback
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${name} must be nonnegative`)
  return parsed
}

function envFlag(name: string, fallback: boolean): boolean {
  const value = process.env[name]?.trim()
  if (!value) return fallback
  if (value === '0' || value.toLowerCase() === 'false') return false
  if (value === '1' || value.toLowerCase() === 'true') return true
  throw new Error(`${name} must be true/false or 1/0`)
}

function isoDay(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10)
}

function cleanDay(value: string | undefined): string | null {
  if (!value?.trim()) return null
  const day = value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`${value} is not YYYY-MM-DD`)
  return day
}

async function readTriathlonStart(): Promise<string | null> {
  try {
    const content = await fs.readFile(TRIATHLON_PAGE, 'utf8')
    const match = /^strava:\s*['"]?(\d{4}-\d{2}-\d{2})['"]?\s*$/m.exec(content)
    return match?.[1] ?? null
  } catch {
    return null
  }
}

async function startDate(): Promise<string> {
  return (
    cleanDay(process.env.GARMIN_CONNECT_START_DATE) ??
    cleanDay(process.env.GARMIN_CONNECT_SINCE) ??
    (await readTriathlonStart()) ??
    isoDay(Date.now() - 90 * DAY_MS)
  )
}

function endDate(): string {
  return cleanDay(process.env.GARMIN_CONNECT_END_DATE) ?? isoDay(Date.now() + DAY_MS)
}

async function readCookieCandidates(): Promise<CookieCandidate[]> {
  const inline = process.env.GARMIN_CONNECT_COOKIE?.trim()
  if (inline) return [{ cookie: inline, source: 'env', type: 'env' }]
  const file = process.env.GARMIN_CONNECT_COOKIE_FILE?.trim()
  if (file) {
    const cookie = (await fs.readFile(file, 'utf8')).trim()
    if (cookie) return [{ cookie, source: file, type: 'file' }]
  }
  const candidates = await browserCookieHeaders()
  if (candidates.length > 0)
    return candidates.map(candidate => ({
      cookie: candidate.cookie,
      source: `${candidate.browser}:${candidate.db}`,
      type: 'browser',
    }))
  throw new Error(
    'set GARMIN_CONNECT_COOKIE/GARMIN_CONNECT_COOKIE_FILE, or log in to Garmin Connect in a supported browser profile',
  )
}

async function readSession(): Promise<GarminConnectSession> {
  const candidates = await readCookieCandidates()
  const csrf = process.env.GARMIN_CONNECT_CSRF_TOKEN?.trim()
  let last: unknown = null
  for (const candidate of candidates) {
    const session: GarminConnectSession = { cookie: candidate.cookie, csrf: '' }
    try {
      session.csrf = csrf || (await fetchCsrf(session))
      if (candidate.type === 'browser')
        console.log('[garmin] using Garmin Connect cookies from browser profile')
      return session
    } catch (err) {
      last = err
      if (candidates.length === 1) break
      console.warn(
        `[garmin] skipped Garmin Connect cookie jar ${candidate.source}: ${errorMessage(err)}`,
      )
    }
  }
  throw last instanceof Error ? last : new Error(errorMessage(last))
}

function requestHeaders(session: GarminConnectSession, contentType?: string): HeadersInit {
  return {
    Accept: 'application/json, text/plain, */*',
    Cookie: session.cookie,
    Origin: CONNECT_ORIGIN,
    Referer: `${CONNECT_ORIGIN}/app/home`,
    'User-Agent': process.env.GARMIN_CONNECT_USER_AGENT?.trim() || DEFAULT_USER_AGENT,
    'Accept-Language': process.env.GARMIN_CONNECT_ACCEPT_LANGUAGE?.trim() || 'en-GB,en;q=0.9',
    'Connect-Csrf-Token': session.csrf,
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    ...(contentType ? { 'Content-Type': contentType } : {}),
  }
}

function documentHeaders(session: GarminConnectSession): HeadersInit {
  return {
    Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    Cookie: session.cookie,
    'User-Agent': process.env.GARMIN_CONNECT_USER_AGENT?.trim() || DEFAULT_USER_AGENT,
    'Accept-Language': process.env.GARMIN_CONNECT_ACCEPT_LANGUAGE?.trim() || 'en-GB,en;q=0.9',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-User': '?1',
    'Sec-Fetch-Dest': 'document',
    'Upgrade-Insecure-Requests': '1',
  }
}

function errorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value)
}

function cookiePairs(header: string): Map<string, string> {
  const pairs = new Map<string, string>()
  for (const part of header.split('; ')) {
    const index = part.indexOf('=')
    if (index <= 0) continue
    pairs.set(part.slice(0, index), part.slice(index + 1))
  }
  return pairs
}

function splitSetCookie(header: string): string[] {
  return header.split(/,(?=\s*[^;,=\s]+=)/g).map(cookie => cookie.trim())
}

function setCookieHeaders(headers: Headers): string[] {
  const withSetCookie = headers as Headers & { getSetCookie?: () => string[] }
  const values = withSetCookie.getSetCookie?.()
  if (values?.length) return values
  const header = headers.get('set-cookie')
  return header ? splitSetCookie(header) : []
}

function applySetCookies(session: GarminConnectSession, headers: Headers): void {
  const updates = setCookieHeaders(headers)
  if (updates.length === 0) return
  const pairs = cookiePairs(session.cookie)
  for (const header of updates) {
    const first = header.split(';', 1)[0]
    const index = first.indexOf('=')
    if (index <= 0) continue
    const name = first.slice(0, index)
    const value = first.slice(index + 1)
    if (!value || /;\s*Max-Age=0(?:;|$)/i.test(header)) pairs.delete(name)
    else pairs.set(name, value)
  }
  session.cookie = [...pairs].map(([name, value]) => `${name}=${value}`).join('; ')
}

async function fetchCsrf(session: GarminConnectSession): Promise<string> {
  let last: string | null = null
  for (const path of ['/app/home', '/app/activities']) {
    const shell = await fetchDocument(session, `${CONNECT_ORIGIN}${path}`)
    const csrf = /<meta\s+name=["']csrf-token["']\s+content=["']([^"']+)["']/i.exec(shell.text)?.[1]
    if (csrf) return csrf
    if (!shell.res.ok) last = `Garmin Connect app shell failed: ${shell.res.status}`
    else if (isSignInUrl(shell.url))
      last = 'Garmin Connect app shell landed on sign-in; refresh the browser session'
    else last = 'Garmin Connect app shell did not expose a CSRF token'
  }
  throw new Error(last ?? 'Garmin Connect app shell did not expose a CSRF token')
}

function isRedirect(status: number): boolean {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308
}

function redirectTarget(current: string, location: string | null): string | null {
  if (!location) return null
  let target: URL
  try {
    target = new URL(location, current)
  } catch {
    return null
  }
  if (target.protocol !== 'https:') return null
  if (target.hostname !== 'connect.garmin.com' && !target.hostname.endsWith('.garmin.com'))
    return null
  return target.toString()
}

function isSignInUrl(value: string): boolean {
  const url = new URL(value)
  return url.hostname === 'sso.garmin.com' || /(?:sign-?in|login)/i.test(url.pathname)
}

async function fetchDocument(
  session: GarminConnectSession,
  start: string,
): Promise<{ res: Response; text: string; url: string }> {
  let url = start
  for (let redirects = 0; redirects <= MAX_DOCUMENT_REDIRECTS; redirects++) {
    const res = await fetch(url, { headers: documentHeaders(session), redirect: 'manual' })
    applySetCookies(session, res.headers)
    if (!isRedirect(res.status)) return { res, text: await res.text(), url }
    const next = redirectTarget(url, res.headers.get('location'))
    if (!next) return { res, text: await res.text(), url }
    url = next
  }
  throw new Error('Garmin Connect app shell exceeded redirect limit')
}

function urlFor(base: string, path: string, params?: URLSearchParams): string {
  const url = new URL(`${base}${path}`)
  if (params) for (const [key, value] of params) url.searchParams.set(key, value)
  return url.toString()
}

function responseSummary(res: Response, text: string): string {
  const type = res.headers.get('content-type') ?? 'unknown content-type'
  if (type.includes('text/html') || text.trimStart().startsWith('<'))
    return `${type} (${text.length} bytes HTML)`
  return `${type} ${text.trim().slice(0, 300)}`
}

async function getJson(
  session: GarminConnectSession,
  base: string,
  path: string,
  params?: URLSearchParams,
  init?: RequestInit,
): Promise<unknown> {
  const res = await fetch(urlFor(base, path, params), {
    ...init,
    headers: requestHeaders(session, init?.body ? 'application/json' : undefined),
  })
  const text = await res.text()
  applySetCookies(session, res.headers)
  if (res.status === 401 || res.status === 403)
    throw new Error(`Garmin Connect session rejected (${res.status}); refresh the cookie`)
  if (!res.ok)
    throw new Error(`Garmin Connect request failed: ${res.status} ${responseSummary(res, text)}`)
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('application/json'))
    throw new Error(`Garmin Connect returned non-JSON: ${responseSummary(res, text)}`)
  return JSON.parse(text) as unknown
}

function activityStartMs(item: GarminConnectActivityListItem): number | null {
  const start = garminConnectActivity(null, item.record, 0)?.startDate
  if (!start) return null
  const ms = Date.parse(start)
  return Number.isFinite(ms) ? ms : null
}

function dayStartMs(day: string): number {
  return Date.parse(`${day}T00:00:00.000Z`)
}

function dayEndMs(day: string): number {
  return Date.parse(`${day}T23:59:59.999Z`)
}

async function fetchActivities(
  session: GarminConnectSession,
  base: string,
  start: string,
  end: string,
  pageSize: number,
  maxActivities: number,
): Promise<GarminConnectActivityListItem[]> {
  const out: GarminConnectActivityListItem[] = []
  const seen = new Set<string>()
  const startMs = dayStartMs(start)
  const endMs = dayEndMs(end)
  for (let offset = 0; ; offset += pageSize) {
    const raw = await getJson(session, base, '/graphql-gateway/graphql', undefined, {
      method: 'POST',
      body: JSON.stringify({
        query: `query{searchActivitiesScalar(start:${offset}, limit:${pageSize}, excludedActivitySubTypes:["assistance"])}`,
      }),
    })
    const page = garminConnectActivities(raw)
    let oldestMs: number | null = null
    for (const item of page) {
      if (seen.has(item.id)) continue
      seen.add(item.id)
      const ms = activityStartMs(item)
      if (ms != null) {
        oldestMs = oldestMs == null ? ms : Math.min(oldestMs, ms)
        if (ms < startMs || ms > endMs) continue
      }
      out.push(item)
      if (maxActivities > 0 && out.length >= maxActivities) return out
    }
    if (page.length < pageSize) return out
    if (oldestMs != null && oldestMs < startMs) return out
  }
}

async function fetchActivityDetail(
  session: GarminConnectSession,
  base: string,
  id: string,
): Promise<UnknownRecord | null> {
  const raw = await getJson(session, base, `/activity-service/activity/${encodeURIComponent(id)}`)
  return isRecord(raw) ? raw : null
}

async function fetchActivityStreamDetail(
  session: GarminConnectSession,
  base: string,
  id: string,
): Promise<UnknownRecord | null> {
  const raw = await getJson(
    session,
    base,
    `/activity-service/activity/${encodeURIComponent(id)}/details`,
  )
  return isRecord(raw) ? raw : null
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function main(): Promise<void> {
  const session = await readSession()
  const base = cleanBaseUrl(process.env.GARMIN_CONNECT_BASE_URL?.trim() || DEFAULT_CONNECT_BASE)
  const pageSize = Math.max(1, envNumber('GARMIN_CONNECT_PAGE_SIZE', DEFAULT_PAGE_SIZE))
  const delayMs = envNumber('GARMIN_CONNECT_DELAY_MS', DEFAULT_DELAY_MS)
  const maxActivities = envNumber('GARMIN_CONNECT_MAX_ACTIVITIES', 0)
  const fetchStreams = envFlag('GARMIN_CONNECT_FETCH_STREAMS', true)
  const start = await startDate()
  const end = endDate()

  console.log(`[garmin] fetching Garmin Connect activities ${start} -> ${end}`)
  const list = await fetchActivities(session, base, start, end, pageSize, maxActivities)
  list.sort((a, b) => {
    const left = garminConnectActivity(null, a.record, 0)?.startDate ?? ''
    const right = garminConnectActivity(null, b.record, 0)?.startDate ?? ''
    return left.localeCompare(right)
  })
  console.log(`[garmin] found ${list.length} activities`)

  const activities: Record<string, GarminActivity> = {}
  const streams: Record<string, GarminStreams> = {}
  let details = 0
  let streamDetails = 0
  let skipped = 0
  for (let i = 0; i < list.length; i++) {
    const item = list[i]
    let detail: UnknownRecord | null = null
    try {
      detail = await fetchActivityDetail(session, base, item.id)
      if (detail) details++
    } catch (err) {
      console.warn(`[garmin] detail ${item.id} failed: ${err instanceof Error ? err.message : err}`)
    }
    const activity = garminConnectActivity(detail, item.record, i)
    if (activity) {
      activities[activity.id] = activity
      if (fetchStreams) {
        try {
          const streamDetail = await fetchActivityStreamDetail(session, base, item.id)
          const stream = garminConnectStreams(streamDetail)
          if (stream) streams[activity.id] = stream
          if (streamDetail) streamDetails++
        } catch (err) {
          console.warn(
            `[garmin] stream ${item.id} failed: ${err instanceof Error ? err.message : err}`,
          )
        }
      }
    } else skipped++
    if (delayMs > 0) await sleep(delayMs)
  }

  const vo2max: Record<string, GarminVo2Day> = {}
  try {
    const raw = await getJson(
      session,
      base,
      `/metrics-service/metrics/maxmet/daily/${encodeURIComponent(start)}/${encodeURIComponent(end)}`,
    )
    for (const day of garminConnectVo2(raw)) vo2max[day.date] = day
    console.log(`[garmin] vo2max days: ${Object.keys(vo2max).length}`)
  } catch (err) {
    console.warn(`[garmin] vo2max fetch failed: ${err instanceof Error ? err.message : err}`)
  }

  const sorted: Record<string, GarminActivity> = {}
  for (const activity of Object.values(activities).sort((a, b) =>
    a.startDate.localeCompare(b.startDate),
  )) {
    sorted[activity.id] = activity
  }

  const now = Date.now()
  const sortedStreams: Record<string, GarminStreams> = {}
  for (const id of Object.keys(sorted)) if (streams[id]) sortedStreams[id] = streams[id]
  const cache: GarminCache = {
    version: CACHE_VERSION,
    lastSync: now,
    activities: sorted,
    streams: sortedStreams,
    vo2max,
  }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  await refreshTriathlonRouteSource()
  console.log(
    `[garmin] wrote ${Object.keys(sorted).length} activities (${details} detail responses, ${streamDetails} stream responses, ${skipped} skipped) -> ${cacheFile}`,
  )
}

main().catch(err => {
  console.error(`[garmin] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
