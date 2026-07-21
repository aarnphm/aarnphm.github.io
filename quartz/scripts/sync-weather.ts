import matter from 'gray-matter'
import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import { promisify } from 'node:util'
import type { RawStravaActivity } from '../plugins/stores/strava'
import { normalizeKind } from '../plugins/stores/strava'
import {
  parseWeatherCache,
  summarizeWeatherDays,
  weatherActivityFromHours,
  type WeatherActivity,
  type WeatherActivityCandidate,
  type WeatherCache,
  type WeatherHour,
} from '../plugins/stores/weather'
import { localIsoDayOffset } from '../util/local-date'
import { joinSegments, QUARTZ } from '../util/path'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord, readNumber, readString } from '../util/type-guards'
import {
  fetchWeatherKitHours,
  WeatherKitRequestError,
  type WeatherKitConfig,
} from '../util/weather-kit'

const CACHE_VERSION = 2
const HOUR_MS = 3_600_000
const TRIATHLON_PAGE = joinSegments(QUARTZ, '..', 'content', 'triathlon.md')
const stravaCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'weather.json')
const execFileAsync = promisify(execFile)
const KEYCHAIN_SERVICES = ['garden-weatherkit', 'WeatherKit', 'weatherkit']

interface StravaWeatherSource {
  activities: RawStravaActivity[]
  streams: Record<string, { latlng: [number, number][] }>
}

function cleanDay(value: string | undefined): string | null {
  if (!value?.trim()) return null
  const day = value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`${value} is not YYYY-MM-DD`)
  return day
}

function envFlag(name: string, fallback: boolean): boolean {
  const value = process.env[name]?.trim()
  if (!value) return fallback
  if (value === '0' || value.toLowerCase() === 'false') return false
  if (value === '1' || value.toLowerCase() === 'true') return true
  throw new Error(`${name} must be true/false or 1/0`)
}

function envNumber(name: string, fallback: number): number {
  const value = process.env[name]
  if (!value?.trim()) return fallback
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${name} must be nonnegative`)
  return parsed
}

async function readTriathlonStart(): Promise<string | null> {
  try {
    const parsed = matter(await fs.readFile(TRIATHLON_PAGE, 'utf8'))
    const strava = parsed.data.strava
    return typeof strava === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(strava) ? strava : null
  } catch {
    return null
  }
}

async function startDate(): Promise<string> {
  return (
    cleanDay(process.env.WEATHERKIT_START_DATE) ??
    cleanDay(process.env.WEATHERKIT_SINCE) ??
    (await readTriathlonStart()) ??
    localIsoDayOffset(-90)
  )
}

function endDate(): string {
  return cleanDay(process.env.WEATHERKIT_END_DATE) ?? localIsoDayOffset(0)
}

async function readPrivateKey(): Promise<string | null> {
  const inline = process.env.WEATHERKIT_PRIVATE_KEY?.trim()
  if (inline) return inline.replaceAll('\\n', '\n')
  const file = process.env.WEATHERKIT_PRIVATE_KEY_FILE?.trim()
  if (!file) return null
  return fs.readFile(file, 'utf8')
}

async function keychainPassword(accounts: string[]): Promise<string | null> {
  for (const service of KEYCHAIN_SERVICES)
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

async function envOrKeychain(name: string, aliases: string[]): Promise<string | null> {
  const value = process.env[name]?.trim()
  if (value) return value
  return keychainPassword([name, ...aliases])
}

async function readMachinePrivateKey(): Promise<string | null> {
  const inline = await keychainPassword(['WEATHERKIT_PRIVATE_KEY', 'privateKey', 'private-key'])
  if (inline) return inline.replaceAll('\\n', '\n')
  const file = await keychainPassword([
    'WEATHERKIT_PRIVATE_KEY_FILE',
    'privateKeyFile',
    'private-key-file',
  ])
  if (!file) return null
  return fs.readFile(file, 'utf8')
}

async function weatherKitConfig(): Promise<WeatherKitConfig | null> {
  const teamId = await envOrKeychain('WEATHERKIT_TEAM_ID', ['teamId', 'team-id'])
  const serviceId = await envOrKeychain('WEATHERKIT_SERVICE_ID', ['serviceId', 'service-id'])
  const keyId = await envOrKeychain('WEATHERKIT_KEY_ID', ['keyId', 'key-id'])
  const privateKey = (await readPrivateKey()) ?? (await readMachinePrivateKey())
  if (!teamId || !serviceId || !keyId || !privateKey) return null
  return { teamId, serviceId, keyId, privateKey }
}

function readActivity(value: unknown): RawStravaActivity | null {
  if (!isRecord(value)) return null
  const id = readNumber(value, 'id')
  const name = readString(value, 'name')
  const sportType = readString(value, 'sportType')
  const distance = readNumber(value, 'distance')
  const movingTime = readNumber(value, 'movingTime')
  const elapsedTime = readNumber(value, 'elapsedTime')
  const totalElevationGain = readNumber(value, 'totalElevationGain')
  const startDate = readString(value, 'startDate')
  const startDateLocal = readString(value, 'startDateLocal')
  const averageSpeed = readNumber(value, 'averageSpeed')
  if (
    id == null ||
    name == null ||
    sportType == null ||
    distance == null ||
    movingTime == null ||
    elapsedTime == null ||
    totalElevationGain == null ||
    startDate == null ||
    startDateLocal == null ||
    averageSpeed == null
  )
    return null
  return {
    id,
    name,
    sportType,
    distance,
    movingTime,
    elapsedTime,
    totalElevationGain,
    startDate,
    startDateLocal,
    averageSpeed,
    averageHeartrate: readNumber(value, 'averageHeartrate'),
    maxHeartrate: readNumber(value, 'maxHeartrate'),
    averageWatts: readNumber(value, 'averageWatts'),
    weightedAverageWatts: readNumber(value, 'weightedAverageWatts'),
    maxWatts: readNumber(value, 'maxWatts'),
    kilojoules: readNumber(value, 'kilojoules'),
    deviceWatts: typeof value.deviceWatts === 'boolean' ? value.deviceWatts : undefined,
    averageCadence: readNumber(value, 'averageCadence'),
    sufferScore: readNumber(value, 'sufferScore'),
    averageTemp: readNumber(value, 'averageTemp'),
    calories: readNumber(value, 'calories'),
  }
}

function coordinate(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null
  const lat = value[0]
  const lng = value[1]
  return typeof lat === 'number' &&
    typeof lng === 'number' &&
    Number.isFinite(lat) &&
    Number.isFinite(lng)
    ? [lat, lng]
    : null
}

function readStreams(value: unknown): Record<string, { latlng: [number, number][] }> {
  if (!isRecord(value)) return {}
  const out: Record<string, { latlng: [number, number][] }> = {}
  for (const [id, raw] of Object.entries(value)) {
    if (!isRecord(raw) || !Array.isArray(raw.latlng)) continue
    const latlng = raw.latlng.map(coordinate).filter(point => point !== null)
    if (latlng.length >= 2) out[id] = { latlng }
  }
  return out
}

async function readStravaSource(): Promise<StravaWeatherSource | null> {
  try {
    const raw: unknown = JSON.parse(await fs.readFile(stravaCacheFile, 'utf8'))
    if (!isRecord(raw) || !isRecord(raw.activities)) return null
    const activities = Object.values(raw.activities)
      .map(readActivity)
      .filter(activity => activity !== null)
    return { activities, streams: readStreams(raw.streams) }
  } catch {
    return null
  }
}

async function readWeatherCache(): Promise<WeatherCache | null> {
  try {
    return parseWeatherCache(JSON.parse(await fs.readFile(cacheFile, 'utf8')))
  } catch {
    return null
  }
}

function routeCenter(latlng: [number, number][]): { latitude: number; longitude: number } {
  const stride = Math.max(1, Math.floor(latlng.length / 200))
  let latitude = 0
  let longitude = 0
  let count = 0
  for (let i = 0; i < latlng.length; i += stride) {
    latitude += latlng[i][0]
    longitude += latlng[i][1]
    count += 1
  }
  if (count === 0) throw new Error('route must contain at least one coordinate')
  return { latitude: latitude / count, longitude: longitude / count }
}

function candidate(
  activity: RawStravaActivity,
  latlng: [number, number][],
): WeatherActivityCandidate | null {
  const sport = normalizeKind(activity.sportType)
  if (!sport || sport === 'strength') return null
  const startMs = Date.parse(activity.startDate)
  const durationS = activity.elapsedTime > 0 ? activity.elapsedTime : activity.movingTime
  if (!Number.isFinite(startMs) || durationS <= 0) return null
  const end = new Date(startMs + durationS * 1000).toISOString()
  const center = routeCenter(latlng)
  return {
    activityId: activity.id,
    date: activity.startDateLocal.slice(0, 10),
    start: new Date(startMs).toISOString(),
    end,
    latitude: center.latitude,
    longitude: center.longitude,
    durationS,
  }
}

function floorHour(ms: number): string {
  return new Date(Math.floor(ms / HOUR_MS) * HOUR_MS).toISOString()
}

function ceilHour(ms: number): string {
  return new Date(Math.ceil(ms / HOUR_MS) * HOUR_MS).toISOString()
}

function requestWindow(candidate: WeatherActivityCandidate): {
  hourlyStart: string
  hourlyEnd: string
} {
  const startMs = Date.parse(candidate.start)
  const endMs = Date.parse(candidate.end)
  return { hourlyStart: floorHour(startMs - HOUR_MS), hourlyEnd: ceilHour(endMs + HOUR_MS) }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function main(): Promise<void> {
  const config = await weatherKitConfig()
  if (!config) {
    console.log(
      '[weather] missing WeatherKit env vars or Keychain items for team id, service id, key id, and private key',
    )
    return
  }

  const source = await readStravaSource()
  if (!source) {
    console.log('[weather] no Strava cache found. run pnpm strava:sync first')
    return
  }

  const since = await startDate()
  const until = endDate()
  const timezone = process.env.WEATHERKIT_TIMEZONE?.trim() || 'America/Toronto'
  const language = process.env.WEATHERKIT_LANGUAGE?.trim() || 'en'
  const delayMs = envNumber('WEATHERKIT_DELAY_MS', 250)
  const force = envFlag('WEATHERKIT_FORCE', false)
  const prev = await readWeatherCache()
  const activities: Record<string, WeatherActivity> = { ...prev?.activities }
  const candidates = source.activities
    .filter(activity => {
      const date = activity.startDateLocal.slice(0, 10)
      return date >= since && date <= until
    })
    .map(activity => {
      const stream = source.streams[String(activity.id)]
      return stream ? candidate(activity, stream.latlng) : null
    })
    .filter(item => item !== null)
    .sort((a, b) => a.start.localeCompare(b.start))

  let fetched = 0
  let skipped = 0
  for (const item of candidates) {
    const key = String(item.activityId)
    if (
      !force &&
      activities[key]?.start === item.start &&
      (activities[key]?.temperatureSeries?.length ?? 0) >= 2
    ) {
      skipped += 1
      continue
    }
    const window = requestWindow(item)
    let hours: WeatherHour[]
    try {
      hours = await fetchWeatherKitHours(config, {
        latitude: item.latitude,
        longitude: item.longitude,
        hourlyStart: window.hourlyStart,
        hourlyEnd: window.hourlyEnd,
        timezone,
        language,
      })
    } catch (err) {
      if (err instanceof WeatherKitRequestError && (err.status === 401 || err.status === 403))
        throw err
      console.warn(
        `[weather] ${item.date} ${item.activityId}: ${err instanceof Error ? err.message : err}`,
      )
      continue
    }
    const weather = weatherActivityFromHours(item, hours)
    if (weather) {
      activities[key] = weather
      fetched += 1
      console.log(
        `[weather] ${item.date} ${item.activityId}: ${weather.windKph ?? 'n/a'} km/h ${weather.windDir ?? ''}`,
      )
    } else {
      console.warn(`[weather] ${item.date} ${item.activityId}: no overlapping hourly wind`)
    }
    if (delayMs > 0) await sleep(delayMs)
  }

  const sortedActivities: Record<string, WeatherActivity> = {}
  for (const activity of Object.values(activities).sort((a, b) => a.start.localeCompare(b.start)))
    sortedActivities[String(activity.activityId)] = activity

  const cache: WeatherCache = {
    version: CACHE_VERSION,
    lastSync: Date.now(),
    activities: sortedActivities,
    days: summarizeWeatherDays(sortedActivities),
  }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  await refreshTriathlonRouteSource()
  console.log(
    `[weather] fetched ${fetched}, skipped ${skipped}, cached ${Object.keys(cache.activities).length} activities -> ${cacheFile}`,
  )
}

main().catch(err => {
  console.error(`[weather] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
