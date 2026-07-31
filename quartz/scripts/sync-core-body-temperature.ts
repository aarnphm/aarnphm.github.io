import matter from 'gray-matter'
import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import { homedir } from 'node:os'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import {
  mergeCoreBodyTemperatureSamples,
  parseCoreBodyTemperatureApiSamples,
  parseCoreBodyTemperatureCache,
  parseCoreBodyTemperatureCsv,
  type CoreBodyTemperatureCache,
  type CoreBodyTemperatureSample,
} from '../plugins/stores/core-body-temperature'
import { joinSegments, QUARTZ } from '../util/path'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord, stringValue } from '../util/type-guards'

const CACHE_VERSION = 1
const DAY_MS = 86_400_000
const CLOUD_WINDOW_DAYS = 31
const CLOUD_OVERLAP_DAYS = 35
const CORE_API_BASE = 'https://core-api.corebodytemp.com/api/v1'
const FIREBASE_AUTH_URL = 'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword'
const TRIATHLON_PAGE = joinSegments(QUARTZ, '..', 'content', 'triathlon.md')
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'core-body-temperature.json')
const importFile = 'core-body-temperature.csv'
const execFileAsync = promisify(execFile)
const keychainServices = ['garden-core-body-temperature', 'CORE', 'corebodytemp']

interface CoreCloudAuth {
  idToken: string
  userId: string
}

interface CoreCloudCredentials {
  email: string
  password: string
}

function firebaseApiKey(): string {
  const value = process.env.CORE_BODY_TEMP_FIREBASE_API_KEY?.trim()
  if (!value) throw new Error('CORE_BODY_TEMP_FIREBASE_API_KEY is required for cloud sync')
  return value
}

export interface CoreCloudWindow {
  start: string
  end: string
}

function unquotePath(path: string): string {
  const trimmed = path.trim()
  if (trimmed.length < 2) return trimmed
  const first = trimmed[0]
  const last = trimmed[trimmed.length - 1]
  return (first === '"' && last === '"') || (first === "'" && last === "'")
    ? trimmed.slice(1, -1)
    : trimmed
}

export function coreBodyTemperatureICloudPath(home: string): string {
  return joinSegments(
    home,
    'Library',
    'Mobile Documents',
    'com~apple~CloudDocs',
    'CORE',
    importFile,
  )
}

export function expandCoreBodyTemperaturePath(path: string, home = homedir()): string {
  const normalized = unquotePath(path)
  if (normalized === '~' || normalized === '$HOME' || normalized === '${HOME}') return home
  if (normalized === joinSegments('iCloud Drive', 'CORE', importFile))
    return coreBodyTemperatureICloudPath(home)
  if (normalized.startsWith('~/')) return joinSegments(home, normalized.slice(2))
  if (normalized.startsWith('$HOME/')) return joinSegments(home, normalized.slice(6))
  if (normalized.startsWith('${HOME}/')) return joinSegments(home, normalized.slice(8))
  return normalized
}

export function coreBodyTemperatureImportCandidates(
  envFile: string | undefined,
  home = homedir(),
): string[] {
  if (envFile?.trim()) return [expandCoreBodyTemperaturePath(envFile, home)]
  return [
    coreBodyTemperatureICloudPath(home),
    joinSegments(QUARTZ, '.quartz-cache', 'core-body-temperature-import.csv'),
  ]
}

function cleanDay(value: string | undefined): string | null {
  if (!value?.trim()) return null
  const day = value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`${value} is not YYYY-MM-DD`)
  return day
}

async function triathlonStart(): Promise<string | null> {
  try {
    const parsed = matter(await fs.readFile(TRIATHLON_PAGE, 'utf8'))
    return typeof parsed.data.strava === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(parsed.data.strava)
      ? parsed.data.strava
      : null
  } catch {
    return null
  }
}

export function coreCloudWindows(
  start: string,
  end: string,
  windowDays = CLOUD_WINDOW_DAYS,
): CoreCloudWindow[] {
  const startMs = Date.parse(start)
  const endMs = Date.parse(end)
  if (
    !Number.isFinite(startMs) ||
    !Number.isFinite(endMs) ||
    endMs < startMs ||
    !Number.isInteger(windowDays) ||
    windowDays < 1
  )
    return []
  const windows: CoreCloudWindow[] = []
  for (let windowStart = startMs; windowStart <= endMs; windowStart += windowDays * DAY_MS) {
    const windowEnd = Math.min(endMs, windowStart + windowDays * DAY_MS - 1)
    windows.push({
      start: new Date(windowStart).toISOString(),
      end: new Date(windowEnd).toISOString(),
    })
  }
  return windows
}

export function coreCloudDataUrl(sensorId: string, window: CoreCloudWindow): string {
  const query = new URLSearchParams({ startGte: window.start, startLte: window.end })
  return `${CORE_API_BASE}/devices/device-data/${encodeURIComponent(sensorId)}/?${query.toString()}`
}

async function keychainPassword(accounts: string[]): Promise<string | null> {
  for (const service of keychainServices)
    for (const account of accounts) {
      try {
        const { stdout } = await execFileAsync(
          '/usr/bin/security',
          ['find-generic-password', '-w', '-s', service, '-a', account],
          { timeout: 5_000 },
        )
        const value = stdout.trim()
        if (value) return value
      } catch {}
    }
  return null
}

async function coreCloudCredentials(): Promise<CoreCloudCredentials | null> {
  const email =
    process.env.CORE_BODY_TEMP_EMAIL?.trim() ??
    (await keychainPassword(['CORE_BODY_TEMP_EMAIL', 'email']))
  const password =
    process.env.CORE_BODY_TEMP_PASSWORD?.trim() ??
    (await keychainPassword([...(email ? [email] : []), 'CORE_BODY_TEMP_PASSWORD', 'password']))
  if (!email && !password) return null
  if (!email || !password)
    throw new Error('CORE_BODY_TEMP_EMAIL and CORE_BODY_TEMP_PASSWORD must both be configured')
  return { email, password }
}

async function requestJson(url: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(url, init)
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail =
      isRecord(body) && typeof body.error === 'string'
        ? body.error
        : isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string'
          ? body.error.message
          : `${response.status} ${response.statusText}`
    throw new Error(detail)
  }
  return body
}

async function authenticateCoreCloud(credentials: CoreCloudCredentials): Promise<CoreCloudAuth> {
  const body = await requestJson(
    `${FIREBASE_AUTH_URL}?key=${encodeURIComponent(firebaseApiKey())}`,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        email: credentials.email,
        password: credentials.password,
        returnSecureToken: true,
      }),
    },
  )
  if (!isRecord(body)) throw new Error('CORE authentication returned an invalid response')
  const idToken = stringValue(body.idToken)
  const userId = stringValue(body.localId)
  if (!idToken || !userId) throw new Error('CORE authentication returned no identity token')
  return { idToken, userId }
}

async function coreApiRequest(path: string, idToken: string): Promise<unknown> {
  return requestJson(`${CORE_API_BASE}/${path.replace(/^\/+/, '')}`, {
    headers: { Authorization: `Bearer ${idToken}` },
  })
}

function deviceIds(raw: unknown): string[] {
  if (!Array.isArray(raw)) return []
  return raw
    .map(value => (isRecord(value) ? stringValue(value.deviceId) : undefined))
    .filter((value): value is string => value != null && value.length > 0)
}

async function coreSensorIds(auth: CoreCloudAuth): Promise<string[]> {
  const explicit = process.env.CORE_BODY_TEMP_SENSOR_IDS?.split(',')
    .map(value => value.trim())
    .filter(Boolean)
  if (explicit?.length) return [...new Set(explicit)]
  const [owned, shared] = await Promise.all([
    coreApiRequest(`users/${encodeURIComponent(auth.userId)}/owned-devices/`, auth.idToken),
    coreApiRequest(`users/${encodeURIComponent(auth.userId)}/shared-devices/`, auth.idToken),
  ])
  return [...new Set([...deviceIds(owned), ...deviceIds(shared)])]
}

async function cloudStart(previous: CoreBodyTemperatureCache | null): Promise<string> {
  const explicit = cleanDay(process.env.CORE_BODY_TEMP_SINCE)
  if (explicit) return `${explicit}T00:00:00.000Z`
  const latest = previous?.samples.at(-1)?.time
  if (latest) return new Date(Date.parse(latest) - CLOUD_OVERLAP_DAYS * DAY_MS).toISOString()
  const configuredStart = await triathlonStart()
  return configuredStart
    ? `${configuredStart}T00:00:00.000Z`
    : new Date(Date.now() - 365 * DAY_MS).toISOString()
}

async function readCoreCloud(
  previous: CoreBodyTemperatureCache | null,
): Promise<CoreBodyTemperatureSample[] | null> {
  const credentials = await coreCloudCredentials()
  if (!credentials) return null
  const auth = await authenticateCoreCloud(credentials)
  const sensors = await coreSensorIds(auth)
  if (sensors.length === 0) throw new Error('CORE Cloud account has no sensors')
  const start = await cloudStart(previous)
  const end = new Date().toISOString()
  let samples: CoreBodyTemperatureSample[] = []
  for (const sensor of sensors)
    for (const window of coreCloudWindows(start, end)) {
      const raw = await requestJson(coreCloudDataUrl(sensor, window), {
        headers: { Authorization: `Bearer ${auth.idToken}` },
      })
      samples = mergeCoreBodyTemperatureSamples(samples, parseCoreBodyTemperatureApiSamples(raw))
    }
  console.log(
    `[core] read ${samples.length} cloud samples from ${sensors.length} sensor${sensors.length === 1 ? '' : 's'}`,
  )
  return samples
}

async function readCache(): Promise<CoreBodyTemperatureCache | null> {
  try {
    return parseCoreBodyTemperatureCache(JSON.parse(await fs.readFile(cacheFile, 'utf8')))
  } catch {
    return null
  }
}

async function main(): Promise<void> {
  const previous = await readCache()
  let samples = previous?.samples ?? []
  let sources = 0
  let imported = 0
  const cloudSamples = await readCoreCloud(previous)
  if (cloudSamples) {
    sources += 1
    imported += cloudSamples.length
    samples = mergeCoreBodyTemperatureSamples(samples, cloudSamples)
  }
  for (const path of coreBodyTemperatureImportCandidates(process.env.CORE_BODY_TEMP_FILE)) {
    try {
      await fs.stat(path)
    } catch {
      continue
    }
    const incoming = parseCoreBodyTemperatureCsv(await fs.readFile(path, 'utf8'))
    if (incoming.length === 0) throw new Error(`${path} contains no recognized CORE samples`)
    sources += 1
    imported += incoming.length
    samples = mergeCoreBodyTemperatureSamples(samples, incoming)
    console.log(`[core] read ${incoming.length} samples from ${path}`)
  }

  if (sources === 0) {
    console.log(
      `[core] no cloud credentials or import found. set CORE_BODY_TEMP_EMAIL and CORE_BODY_TEMP_PASSWORD, save them under the garden-core-body-temperature keychain service, export CORE Cloud CSV to iCloud Drive/CORE/${importFile}, set CORE_BODY_TEMP_FILE=<export.csv>, or drop one at quartz/.quartz-cache/core-body-temperature-import.csv`,
    )
    if (!previous) return
    const latest = previous.samples.at(-1)?.time ?? 'none'
    console.log(
      `[core] keeping previous cache (${new Date(previous.lastSync).toISOString()}, latest sample ${latest})`,
    )
    return
  }

  const cache: CoreBodyTemperatureCache = { version: CACHE_VERSION, lastSync: Date.now(), samples }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  await refreshTriathlonRouteSource()
  console.log(`[core] merged ${imported} samples → ${samples.length} samples → ${cacheFile}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(`[core] sync failed: ${error instanceof Error ? error.message : error}`)
    process.exit(1)
  })
}
