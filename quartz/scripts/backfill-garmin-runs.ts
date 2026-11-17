import fs from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import type { AppleCache, AppleSwim } from '../plugins/stores/apple'
import type { Args, BackfillSport } from './backfill-garmin-runs/options'
import { AdaptiveRateLimiter, fetchWithRetry } from '../plugins/stores/citations'
import { matchGarminActivity, type GarminCache } from '../plugins/stores/garmin'
import {
  normalizeKind,
  type RawStravaActivity,
  type StravaRawCache,
} from '../plugins/stores/strava'
import { matchAppleSwims } from '../util/apple-swim-match'
import { upsertEnvLine } from '../util/env-file'
import { encodeGarminSwimFit, type GarminFitEncoding } from '../util/garmin-fit'
import {
  assertGarminResponseAuthorized,
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
import { readStravaCacheFile } from '../util/strava-cache-file'
import { stravaActivityTcx, type TimedStravaStreams } from '../util/strava-tcx'
import { isRecord, readNumber, readString, type UnknownRecord } from '../util/type-guards'
import { parseArgs } from './backfill-garmin-runs/options'
import {
  applePoolSwimProjection,
  garminSwimFitInput,
  shouldDeferPoolSwimForApple,
} from './backfill-garmin-runs/swim'
import { uploadGarminFit } from './garmin-fit-upload'

const TOKEN_URL = 'https://www.strava.com/oauth/token'

const DEFAULT_API_BASE_URL = 'https://www.strava.com/api/v3'

const STRAVA_API = new URL(process.env.STRAVA_API_BASE_URL ?? DEFAULT_API_BASE_URL)
  .toString()
  .replace(/\/+$/, '')

const ENV_FILE = '.env'

const CACHE_DIR = joinSegments(QUARTZ, '.quartz-cache')

const STRAVA_CACHE = joinSegments(CACHE_DIR, 'strava.json')

const GARMIN_CACHE = joinSegments(CACHE_DIR, 'garmin.json')

const APPLE_CACHE = joinSegments(CACHE_DIR, 'apple-health.json')

const DEFAULT_OUTPUT_DIR = joinSegments(CACHE_DIR, 'garmin-run-backfill')

const STRAVA_LIMITER = new AdaptiveRateLimiter(400, 60_000)

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

interface UploadResult {
  status: number
  body: string
  json: unknown
}

async function readJsonFile<T>(path: string): Promise<T> {
  return JSON.parse(await fs.readFile(path, 'utf8')) as T
}

async function readAppleCache(): Promise<AppleCache | null> {
  try {
    return await readJsonFile<AppleCache>(APPLE_CACHE)
  } catch {
    return null
  }
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

async function fetchTimedStreams(token: string, id: string): Promise<TimedStravaStreams> {
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

export function encodeGarminSwimBackfill(
  activity: RawStravaActivity,
  streams: TimedStravaStreams,
  appleSwim?: AppleSwim | null,
): GarminFitEncoding {
  return encodeGarminSwimFit(garminSwimFitInput(activity, streams, appleSwim))
}

export function garminBackfillFilename(activity: RawStravaActivity, sport: BackfillSport): string {
  const extension = sport === 'swim' ? 'fit' : 'tcx'
  return `${activity.startDate.slice(0, 10)}-${activity.id}.${extension}`
}

function outputDir(args: Args): string {
  if (args.outputDir) return args.outputDir
  if (args.sport === 'run') return DEFAULT_OUTPUT_DIR
  return joinSegments(CACHE_DIR, 'garmin-swim-backfill')
}

function parseJsonOrNull(text: string): unknown {
  if (!text.trim()) return null
  try {
    const value: unknown = JSON.parse(text)
    return value
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
  form.set('file', new Blob([content], { type: 'application/octet-stream' }), filename)
  const res = await fetch(garminUrlFor(base, '/upload-service/upload'), {
    method: 'POST',
    headers: garminConnectRequestHeaders(session),
    body: form,
  })
  assertGarminResponseAuthorized(res)
  const body = await res.text()
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
  const strava = await readStravaCacheFile(STRAVA_CACHE)
  if (!strava) throw new Error('Strava cache is missing')
  const garmin = await readJsonFile<GarminCache>(GARMIN_CACHE)
  const candidates = selectCandidates(strava, garmin, args)
  const apple = args.sport === 'swim' ? await readAppleCache() : null
  const appleMatches = matchAppleSwims(
    Object.values(apple?.swims ?? {}),
    candidates.map(({ activity }) => ({
      id: activity.id,
      date: startDay(activity),
      start: activity.startDate,
      distanceM: activity.distance,
    })),
  )
  console.log(
    `[garmin-backfill] ${args.write ? 'write' : 'dry-run'} ${candidates.length} candidate ${args.sport}s${args.since ? ` since ${args.since}` : ''}`,
  )
  for (const candidate of candidates) {
    const appleSwim = appleMatches.get(candidate.activity.id)
    const appleStatus =
      args.sport === 'swim'
        ? ` | Apple ${applePoolSwimProjection(candidate.activity, appleSwim) ? 'lengths' : appleSwim ? 'summary' : 'pending'}`
        : ''
    console.log(`[garmin-backfill] candidate ${describe(candidate.activity)}${appleStatus}`)
  }
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
  let deferred = 0
  for (const candidate of candidates) {
    const activity = candidate.activity
    const filename = garminBackfillFilename(activity, args.sport)
    const streams = await fetchTimedStreams(token.access, activity.id)
    const pathname = joinSegments(targetDir, filename)
    let ids: string[]
    if (args.sport === 'swim') {
      const input = garminSwimFitInput(activity, streams, appleMatches.get(activity.id))
      if (shouldDeferPoolSwimForApple(activity, input)) {
        deferred++
        console.log(
          `[garmin-backfill] deferred ${activity.id}: waiting for complete Apple pool lengths`,
        )
        continue
      }
      const encoded = encodeGarminSwimFit(input)
      if (!encoded.validation.valid)
        throw new Error(
          `FIT validation failed for ${activity.id}: ${encoded.validation.errors.join('; ')}`,
        )
      await fs.writeFile(pathname, encoded.bytes)
      ids = [await uploadGarminFit(session, uploadBase, filename, encoded.bytes)]
      uploaded++
      console.log(`[garmin-backfill] uploaded ${activity.id} ${filename} -> ${ids.join(',')}`)
    } else {
      const tcx = stravaActivityTcx(activity, streams, args.sport)
      await fs.writeFile(pathname, tcx)
      const result = await uploadTcx(session, uploadBase, filename, tcx)
      ids = resultIds(result.json)
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
    }
    for (const id of ids) {
      await updateGarminActivityTitle(session, titleBase, id, activity.name)
      renamed++
      console.log(`[garmin-backfill] renamed ${id} -> ${activity.name}`)
    }
    if (args.uploadDelayMs > 0) await sleep(args.uploadDelayMs)
  }
  console.log(
    `[garmin-backfill] done uploaded=${uploaded} duplicate=${duplicates} renamed=${renamed} deferred=${deferred}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[garmin-backfill] failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
export {
  applePoolSwimProjection,
  garminSwimFitInput,
  shouldDeferPoolSwimForApple,
} from './backfill-garmin-runs/swim'
