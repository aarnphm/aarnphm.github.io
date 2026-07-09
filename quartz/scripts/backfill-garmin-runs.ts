import fs from 'node:fs/promises'
import { AdaptiveRateLimiter, fetchWithRetry } from '../plugins/stores/citations'
import { matchGarminActivity, type GarminCache } from '../plugins/stores/garmin'
import {
  normalizeKind,
  type RawStravaActivity,
  type Sport,
  type StravaRawCache,
} from '../plugins/stores/strava'
import { upsertEnvLine } from '../util/env-file'
import {
  applyGarminSetCookies,
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  DEFAULT_GARMIN_IMPORT_BASE,
  garminConnectRequestHeaders,
  garminResponseSummary,
  garminUrlFor,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import { updateGarminActivityTitle } from '../util/garmin-title-update'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, readNumber, readString, type UnknownRecord } from '../util/type-guards'

const TOKEN_URL = 'https://www.strava.com/oauth/token'
const DEFAULT_API_BASE_URL = 'https://www.strava.com/api/v3'
const STRAVA_API = new URL(process.env.STRAVA_API_BASE_URL ?? DEFAULT_API_BASE_URL)
  .toString()
  .replace(/\/+$/, '')
const ENV_FILE = '.env'
const CACHE_DIR = joinSegments(QUARTZ, '.quartz-cache')
const STRAVA_CACHE = joinSegments(CACHE_DIR, 'strava.json')
const GARMIN_CACHE = joinSegments(CACHE_DIR, 'garmin.json')
const DEFAULT_OUTPUT_DIR = joinSegments(CACHE_DIR, 'garmin-run-backfill')
const STRAVA_LIMITER = new AdaptiveRateLimiter(400, 60_000)
const DEFAULT_UPLOAD_DELAY_MS = 1500

interface Args {
  write: boolean
  sport: BackfillSport
  since: string | null
  limit: number
  ids: Set<string>
  outputDir: string | null
  uploadDelayMs: number
}

type BackfillSport = Extract<Sport, 'run' | 'swim'>

interface TokenResponse {
  access_token: string
  refresh_token: string
}

interface StravaToken {
  access: string
  refreshToken: string
}

interface BackfillCandidate {
  activity: RawStravaActivity
  reason: string
}

interface TimedStravaStreams {
  time: number[]
  latlng: [number, number][]
  altitude: number[]
  distance: number[]
  heartrate: number[]
  cadence: number[]
  watts: number[]
  temp: number[]
}

interface UploadResult {
  status: number
  body: string
  json: unknown
}

function usage(): string {
  return [
    'usage: pnpm garmin:backfill-runs -- [--write] [--sport run|swim] [--since YYYY-MM-DD] [--limit N] [--id ID]',
    '',
    'defaults to dry-run. --write uploads generated TCX files to Garmin Connect.',
  ].join('\n')
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    write: false,
    sport: 'run',
    since: null,
    limit: 0,
    ids: new Set(),
    outputDir: null,
    uploadDelayMs: envNumber('GARMIN_BACKFILL_UPLOAD_DELAY_MS', DEFAULT_UPLOAD_DELAY_MS),
  }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--') continue
    if (arg === '--write') args.write = true
    else if (arg === '--dry-run') args.write = false
    else if (arg === '--sport') args.sport = parseSport(readArgValue(argv, ++i, arg))
    else if (arg === '--since') args.since = readArgValue(argv, ++i, arg)
    else if (arg === '--limit') args.limit = positiveInteger(readArgValue(argv, ++i, arg), arg)
    else if (arg === '--id') args.ids.add(readArgValue(argv, ++i, arg))
    else if (arg === '--ids') {
      for (const id of readArgValue(argv, ++i, arg).split(','))
        if (id.trim()) args.ids.add(id.trim())
    } else if (arg === '--out') args.outputDir = readArgValue(argv, ++i, arg)
    else if (arg === '--delay-ms')
      args.uploadDelayMs = nonnegativeInteger(readArgValue(argv, ++i, arg), arg)
    else if (arg === '--help' || arg === '-h') {
      console.log(usage())
      process.exit(0)
    } else throw new Error(`unknown argument ${arg}\n${usage()}`)
  }
  if (args.since && !/^\d{4}-\d{2}-\d{2}$/.test(args.since))
    throw new Error(`--since must be YYYY-MM-DD, got ${args.since}`)
  return args
}

function parseSport(value: string): BackfillSport {
  if (value === 'run' || value === 'swim') return value
  throw new Error(`--sport must be run or swim, got ${value}`)
}

function readArgValue(argv: string[], index: number, flag: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`)
  return value
}

function envNumber(name: string, fallback: number): number {
  const value = process.env[name]
  if (!value?.trim()) return fallback
  return nonnegativeInteger(value, name)
}

function nonnegativeInteger(value: string, name: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0)
    throw new Error(`${name} must be a nonnegative integer`)
  return parsed
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed <= 0)
    throw new Error(`${name} must be a positive integer`)
  return parsed
}

async function readJsonFile<T>(path: string): Promise<T> {
  return JSON.parse(await fs.readFile(path, 'utf8')) as T
}

function apiUrl(path: string, params: Record<string, string | number> = {}): string {
  const url = new URL(`${STRAVA_API}${path}`)
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, String(value))
  return url.toString()
}

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` }
}

async function readTokenResponse(res: Response): Promise<TokenResponse> {
  const raw: unknown = await res.json()
  if (!isRecord(raw)) throw new Error('Strava token refresh returned a non-object response')
  const accessToken = readString(raw, 'access_token')
  const refreshToken = readString(raw, 'refresh_token')
  if (!accessToken || !refreshToken) throw new Error('Strava token refresh response missing tokens')
  return { access_token: accessToken, refresh_token: refreshToken }
}

async function refreshStravaToken(
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
  const res = await fetchWithRetry(TOKEN_URL, { method: 'POST', body }, STRAVA_LIMITER)
  if (!res) throw new Error('Strava token refresh failed')
  return readTokenResponse(res)
}

async function resolveStravaToken(cache: StravaRawCache): Promise<StravaToken> {
  const clientId = process.env.STRAVA_CLIENT_ID
  const clientSecret = process.env.STRAVA_CLIENT_SECRET
  const envRefreshToken = process.env.STRAVA_REFRESH_TOKEN
  const refreshToken = cache.auth.refreshToken || envRefreshToken
  if (clientId && clientSecret && refreshToken) {
    const token = await refreshStravaToken(clientId, clientSecret, refreshToken)
    if (token.refresh_token !== envRefreshToken) {
      await upsertEnvLine(ENV_FILE, 'STRAVA_REFRESH_TOKEN', token.refresh_token)
      console.log('[strava] STRAVA_REFRESH_TOKEN updated in .env')
    }
    return { access: token.access_token, refreshToken: token.refresh_token }
  }
  const direct = process.env.STRAVA_ACCESS_TOKEN
  if (direct) return { access: direct, refreshToken: refreshToken ?? '' }
  throw new Error('need Strava OAuth env or cached refresh token')
}

function isBackfillSport(activity: RawStravaActivity, sport: BackfillSport): boolean {
  return normalizeKind(activity.sportType) === sport
}

function startDay(activity: RawStravaActivity): string {
  return (activity.startDateLocal || activity.startDate).slice(0, 10)
}

function selectCandidates(
  strava: StravaRawCache,
  garmin: GarminCache,
  args: Args,
): BackfillCandidate[] {
  const candidates: BackfillCandidate[] = []
  const activities = Object.values(strava.activities)
    .filter(activity => isBackfillSport(activity, args.sport))
    .filter(activity => args.ids.size === 0 || args.ids.has(String(activity.id)))
    .filter(activity => !args.since || startDay(activity) >= args.since)
    .sort((left, right) => left.startDate.localeCompare(right.startDate))
  for (const activity of activities) {
    const match = matchGarminActivity(activity, args.sport, garmin)
    if (match) continue
    candidates.push({
      activity,
      reason: `no Garmin ${args.sport} matched by start time, distance, and duration`,
    })
  }
  return args.limit > 0 ? candidates.slice(0, args.limit) : candidates
}

function streamRecord(raw: unknown, key: string): UnknownRecord | null {
  if (!isRecord(raw)) return null
  const value = raw[key]
  return isRecord(value) ? value : null
}

function streamData(raw: unknown, key: string): unknown[] {
  const record = streamRecord(raw, key)
  const data = record?.data
  return Array.isArray(data) ? data : []
}

function numberArray(raw: unknown, key: string): number[] {
  const out: number[] = []
  for (const value of streamData(raw, key)) {
    if (typeof value === 'number' && Number.isFinite(value)) out.push(value)
  }
  return out
}

function latlngArray(raw: unknown): [number, number][] {
  const out: [number, number][] = []
  for (const value of streamData(raw, 'latlng')) {
    if (!Array.isArray(value) || value.length < 2) continue
    const lat = value[0]
    const lng = value[1]
    if (
      typeof lat === 'number' &&
      typeof lng === 'number' &&
      Number.isFinite(lat) &&
      Number.isFinite(lng) &&
      lat >= -90 &&
      lat <= 90 &&
      lng >= -180 &&
      lng <= 180
    )
      out.push([lat, lng])
  }
  return out
}

async function fetchTimedStreams(token: string, id: number): Promise<TimedStravaStreams> {
  const res = await fetchWithRetry(
    apiUrl(`/activities/${id}/streams`, {
      keys: 'time,latlng,altitude,distance,heartrate,cadence,watts,temp',
      key_by_type: 'true',
    }),
    { headers: authHeaders(token) },
    STRAVA_LIMITER,
  )
  if (!res) throw new Error(`Strava stream fetch failed for ${id}`)
  const raw: unknown = await res.json()
  return {
    time: numberArray(raw, 'time'),
    latlng: latlngArray(raw),
    altitude: numberArray(raw, 'altitude'),
    distance: numberArray(raw, 'distance'),
    heartrate: numberArray(raw, 'heartrate'),
    cadence: numberArray(raw, 'cadence'),
    watts: numberArray(raw, 'watts'),
    temp: numberArray(raw, 'temp'),
  }
}

function xml(value: string | number): string {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

function positive(value: number | undefined): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? value : null
}

function pointNumber(values: number[], index: number): number | null {
  return positive(values[index])
}

function pointDistance(values: number[], index: number): number | null {
  const value = values[index]
  return value != null && Number.isFinite(value) && value >= 0 ? value : null
}

function pointTime(startMs: number, elapsedS: number): string {
  return new Date(startMs + Math.round(elapsedS * 1000)).toISOString()
}

function average(values: number[]): number | null {
  const positives = values.filter(value => value > 0 && Number.isFinite(value))
  if (positives.length === 0) return null
  return positives.reduce((sum, value) => sum + value, 0) / positives.length
}

function max(values: number[]): number | null {
  const positives = values.filter(value => value > 0 && Number.isFinite(value))
  if (positives.length === 0) return null
  return Math.max(...positives)
}

function integer(value: number | null): number | null {
  return value == null ? null : Math.round(value)
}

function byte(value: number | null): number | null {
  if (value == null) return null
  return Math.max(0, Math.min(254, Math.round(value)))
}

function element(name: string, value: string | number | null): string {
  return value == null ? '' : `<${name}>${xml(value)}</${name}>`
}

function heartRateElement(value: number | null): string {
  return value == null ? '' : `<HeartRateBpm><Value>${Math.round(value)}</Value></HeartRateBpm>`
}

function trackpoint(
  activity: RawStravaActivity,
  streams: TimedStravaStreams,
  index: number,
  startMs: number,
): string {
  const time = streams.time[index]
  const latlng = streams.latlng[index]
  const altitude = pointNumber(streams.altitude, index)
  const distance = pointDistance(streams.distance, index)
  const heartrate = pointNumber(streams.heartrate, index)
  const cadence = byte(pointNumber(streams.cadence, index))
  const watts = integer(pointNumber(streams.watts, index))
  const temp = integer(pointNumber(streams.temp, index))
  const parts = [`<Time>${pointTime(startMs, time)}</Time>`]
  if (latlng)
    parts.push(
      `<Position><LatitudeDegrees>${latlng[0]}</LatitudeDegrees><LongitudeDegrees>${latlng[1]}</LongitudeDegrees></Position>`,
    )
  parts.push(element('AltitudeMeters', altitude == null ? null : altitude.toFixed(1)))
  parts.push(element('DistanceMeters', distance == null ? null : distance.toFixed(1)))
  parts.push(heartRateElement(heartrate))
  if (cadence != null) parts.push(element('Cadence', cadence))
  const tpx = [
    watts == null ? '' : `<ns3:Watts>${watts}</ns3:Watts>`,
    cadence == null ? '' : `<ns3:RunCadence>${cadence}</ns3:RunCadence>`,
    temp == null ? '' : `<ns3:Temp>${temp}</ns3:Temp>`,
  ].filter(Boolean)
  if (tpx.length > 0) parts.push(`<Extensions><ns3:TPX>${tpx.join('')}</ns3:TPX></Extensions>`)
  if (parts.length <= 1) throw new Error(`no TCX samples available for ${activity.id}`)
  return `<Trackpoint>${parts.join('')}</Trackpoint>`
}

function tcxSport(sport: BackfillSport): string {
  if (sport === 'run') return 'Running'
  return 'Other'
}

function buildTcx(
  activity: RawStravaActivity,
  streams: TimedStravaStreams,
  sport: BackfillSport,
): string {
  if (streams.time.length < 2) throw new Error(`Strava activity ${activity.id} has no timed stream`)
  const startMs = Date.parse(activity.startDate)
  if (!Number.isFinite(startMs))
    throw new Error(`Strava activity ${activity.id} has invalid startDate`)
  const lastElapsed = streams.time[streams.time.length - 1]
  const totalTimeS = Math.max(lastElapsed, activity.elapsedTime || activity.movingTime)
  const avgHr = integer(average(streams.heartrate))
  const maxHr = integer(max(streams.heartrate))
  const calories = integer(activity.calories ?? null) ?? 0
  const points = streams.time
    .map((_, index) => trackpoint(activity, streams, index, startMs))
    .join('')
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2 http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd">',
    '<Activities>',
    `<Activity Sport="${tcxSport(sport)}">`,
    `<Id>${new Date(startMs).toISOString()}</Id>`,
    `<Lap StartTime="${new Date(startMs).toISOString()}">`,
    element('TotalTimeSeconds', totalTimeS.toFixed(1)),
    element('DistanceMeters', activity.distance.toFixed(1)),
    element('Calories', calories),
    '<Intensity>Active</Intensity>',
    heartRateElement(avgHr),
    maxHr == null ? '' : `<MaximumHeartRateBpm><Value>${maxHr}</Value></MaximumHeartRateBpm>`,
    '<TriggerMethod>Manual</TriggerMethod>',
    `<Track>${points}</Track>`,
    '</Lap>',
    `<Notes>${xml(`Strava ${activity.id}: ${activity.name}`)}</Notes>`,
    `<Creator xsi:type="Device_t"><Name>${xml(`Strava ${sport === 'swim' ? 'Swim' : 'Run'} Backfill`)}</Name><UnitId>0</UnitId><ProductID>0</ProductID></Creator>`,
    '</Activity>',
    '</Activities>',
    '</TrainingCenterDatabase>',
  ].join('')
}

function safeFilename(activity: RawStravaActivity): string {
  return `${activity.startDate.slice(0, 10)}-${activity.id}.tcx`
}

function outputDir(args: Args): string {
  if (args.outputDir) return args.outputDir
  if (args.sport === 'run') return DEFAULT_OUTPUT_DIR
  return joinSegments(CACHE_DIR, 'garmin-swim-backfill')
}

function uploadHeaders(session: GarminConnectSession): HeadersInit {
  const headers = new Headers(garminConnectRequestHeaders(session))
  headers.set('NK', 'NT')
  headers.set(
    'Origin',
    process.env.GARMIN_CONNECT_IMPORT_ORIGIN?.trim() || 'https://sso.garmin.com',
  )
  headers.set(
    'User-Agent',
    process.env.GARMIN_CONNECT_IMPORT_USER_AGENT?.trim() || 'GCM-iOS-5.7.2.1',
  )
  headers.delete('Content-Type')
  return headers
}

function parseJsonOrNull(text: string): unknown {
  if (!text.trim()) return null
  try {
    return JSON.parse(text) as unknown
  } catch {
    return null
  }
}

async function uploadTcx(
  session: GarminConnectSession,
  base: string,
  filename: string,
  content: string,
): Promise<UploadResult> {
  const form = new FormData()
  form.set('file', new Blob([content], { type: 'application/octet-stream' }), `"${filename}"`)
  const res = await fetch(garminUrlFor(base, '/upload-service/upload/tcx'), {
    method: 'POST',
    headers: uploadHeaders(session),
    body: form,
  })
  const body = await res.text()
  applyGarminSetCookies(session, res.headers)
  const json = parseJsonOrNull(body)
  if (res.status === 409) return { status: res.status, body, json }
  if (!res.ok)
    throw new Error(`Garmin upload failed: ${res.status} ${garminResponseSummary(res, body)}`)
  return { status: res.status, body, json }
}

function resultIds(result: unknown): string[] {
  const out: string[] = []
  const queue: unknown[] = [result]
  for (let i = 0; i < queue.length; i++) {
    const item = queue[i]
    if (Array.isArray(item)) {
      queue.push(...item)
      continue
    }
    if (!isRecord(item)) continue
    const id =
      readNumber(item, 'internalId') ??
      readNumber(item, 'activityId') ??
      readNumber(item, 'externalId')
    if (id != null) out.push(String(id))
    for (const value of Object.values(item))
      if (isRecord(value) || Array.isArray(value)) queue.push(value)
  }
  return [...new Set(out)]
}

function describe(activity: RawStravaActivity): string {
  const km = activity.distance / 1000
  const minutes = Math.round(activity.movingTime / 60)
  return `${activity.startDateLocal || activity.startDate} | ${activity.name} | ${km.toFixed(2)}km | ${minutes}min | ${activity.id}`
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const strava = await readJsonFile<StravaRawCache>(STRAVA_CACHE)
  const garmin = await readJsonFile<GarminCache>(GARMIN_CACHE)
  const candidates = selectCandidates(strava, garmin, args)
  console.log(
    `[garmin-backfill] ${args.write ? 'write' : 'dry-run'} ${candidates.length} candidate ${args.sport}s${args.since ? ` since ${args.since}` : ''}`,
  )
  for (const candidate of candidates)
    console.log(`[garmin-backfill] candidate ${describe(candidate.activity)}`)
  if (!args.write || candidates.length === 0) return

  const targetDir = outputDir(args)
  await fs.mkdir(targetDir, { recursive: true })
  const token = await resolveStravaToken(strava)
  const session = await readGarminConnectSession()
  const uploadBase = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_IMPORT_BASE_URL?.trim() || DEFAULT_GARMIN_IMPORT_BASE,
  )
  const titleBase = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_TITLE_BASE_URL?.trim() ||
      process.env.GARMIN_CONNECT_BASE_URL?.trim() ||
      DEFAULT_GARMIN_CONNECT_BASE,
  )

  let uploaded = 0
  let duplicates = 0
  let renamed = 0
  for (const candidate of candidates) {
    const activity = candidate.activity
    const filename = safeFilename(activity)
    const streams = await fetchTimedStreams(token.access, activity.id)
    const tcx = buildTcx(activity, streams, args.sport)
    const path = joinSegments(targetDir, filename)
    await fs.writeFile(path, tcx)
    const result = await uploadTcx(session, uploadBase, filename, tcx)
    const ids = resultIds(result.json)
    if (result.status === 409) {
      duplicates++
      console.log(
        `[garmin-backfill] duplicate ${activity.id} ${filename}${ids.length ? ` -> ${ids.join(',')}` : ''}`,
      )
    } else {
      uploaded++
      console.log(
        `[garmin-backfill] uploaded ${activity.id} ${filename}${ids.length ? ` -> ${ids.join(',')}` : ''}`,
      )
    }
    for (const id of ids) {
      await updateGarminActivityTitle(session, titleBase, id, activity.name)
      renamed++
      console.log(`[garmin-backfill] renamed ${id} -> ${activity.name}`)
    }
    if (args.uploadDelayMs > 0) await sleep(args.uploadDelayMs)
  }
  console.log(
    `[garmin-backfill] done uploaded=${uploaded} duplicate=${duplicates} renamed=${renamed}`,
  )
}

main().catch(err => {
  console.error(`[garmin-backfill] failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
