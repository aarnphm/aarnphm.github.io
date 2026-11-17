import matter from 'gray-matter'
import fs from 'node:fs/promises'
import type { RouteWeatherCandidate } from './sync-weather/source'
import { normalizeKind } from '../plugins/stores/strava'
import {
  parseWeatherCache,
  summarizeWeatherDays,
  weatherActivityFromRouteHours,
  weatherActivityHasCompleteRouteHours,
  weatherRouteHourFromForecast,
  weatherSnapshotFromHours,
  type WeatherActivity,
  type WeatherCache,
  type WeatherHour,
  type WeatherRouteHour,
  type WeatherSnapshot,
} from '../plugins/stores/weather'
import { buildActivityEnvironment } from '../util/activity-environment'
import { parseActivityProviderReports } from '../util/activity-provider-reports'
import {
  auditGardenUvCalibration,
  calibrateGardenUvScore,
  type GardenUvCalibrationPair,
} from '../util/activity-uv-score'
import { localIsoDayOffset } from '../util/local-date'
import { joinSegments, QUARTZ } from '../util/path'
import { weatherSyncRefreshDays } from '../util/sync-refresh-window'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import {
  fetchWeatherKitAttribution,
  fetchWeatherKitHours,
  WeatherKitRequestError,
} from '../util/weather-kit'
import {
  routeWeatherNeedsRefresh,
  selectRouteHour,
  type RouteHourQuery,
} from '../util/weather-route-hours'
import { weatherKitConfig } from './sync-weather/config'
import { candidate, readStravaSource } from './sync-weather/source'

const CACHE_VERSION = 5

const HOUR_MS = 3_600_000

const TRIATHLON_PAGE = joinSegments(QUARTZ, '..', 'content', 'triathlon.md')

const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'weather.json')

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

async function readWeatherCache(): Promise<WeatherCache | null> {
  try {
    return parseWeatherCache(JSON.parse(await fs.readFile(cacheFile, 'utf8')))
  } catch {
    return null
  }
}

function floorHour(ms: number): string {
  return new Date(Math.floor(ms / HOUR_MS) * HOUR_MS).toISOString()
}

function ceilHour(ms: number): string {
  return new Date(Math.ceil(ms / HOUR_MS) * HOUR_MS).toISOString()
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
  const maxCalls = Math.floor(envNumber('WEATHERKIT_MAX_CALLS', 500))
  const force = envFlag('WEATHERKIT_FORCE', false)
  const prev = await readWeatherCache()
  const activities: Record<string, WeatherActivity> = { ...prev?.activities }
  const refreshDays = weatherSyncRefreshDays()
  const latestCachedWeatherMs = Object.values(activities).reduce<number | null>((latest, item) => {
    const startMs = Date.parse(item.start)
    if (!Number.isFinite(startMs)) return latest
    return latest === null || startMs > latest ? startMs : latest
  }, null)
  const windowCutoffMs = Date.now() - refreshDays * 86_400_000
  const refreshWindowStartMs =
    latestCachedWeatherMs === null
      ? windowCutoffMs
      : Math.min(windowCutoffMs, latestCachedWeatherMs - refreshDays * 86_400_000)
  let uvCalibration = prev?.uvCalibration ?? null
  let attribution = prev?.attribution ?? null
  try {
    attribution = await fetchWeatherKitAttribution(config, language)
  } catch (err) {
    if (err instanceof WeatherKitRequestError && (err.status === 401 || err.status === 403))
      throw err
    console.warn(`[weather] attribution: ${err instanceof Error ? err.message : err}`)
  }
  const sourceActivities = new Map(source.activities.map(activity => [activity.id, activity]))
  const sourceIds = new Set(source.activities.map(activity => String(activity.id)))
  for (const id of Object.keys(activities)) if (!sourceIds.has(id)) delete activities[id]
  const allCandidates = source.activities
    .map(activity => {
      const stream = source.streams[String(activity.id)]
      return stream ? candidate(activity, stream) : null
    })
    .filter((item): item is RouteWeatherCandidate => item !== null && item.queries.length > 0)
    .sort((a, b) => a.start.localeCompare(b.start))
  const candidates = allCandidates.filter(item => item.date >= since && item.date <= until)

  const refreshCandidates = candidates.filter(item =>
    routeWeatherNeedsRefresh(
      activities[String(item.activityId)],
      item.routeFingerprint,
      prev?.version,
      CACHE_VERSION,
      item.start,
      refreshWindowStartMs,
      force,
    ),
  )
  const recent = refreshCandidates
    .filter(item => Date.parse(item.start) >= refreshWindowStartMs)
    .sort((left, right) => right.start.localeCompare(left.start))
  let historical = refreshCandidates
    .filter(item => Date.parse(item.start) < refreshWindowStartMs)
    .sort((left, right) => left.start.localeCompare(right.start))
  const prefetched = new Map<string, WeatherHour>()
  let calls = 0
  const fetchQuery = async (item: RouteWeatherCandidate, query: RouteHourQuery) => {
    if (calls >= maxCalls) throw new Error(`WeatherKit call budget ${maxCalls} exhausted`)
    calls += 1
    const hours = await fetchWeatherKitHours(config, {
      latitude: query.latitude,
      longitude: query.longitude,
      hourlyStart: query.hourlyStart,
      hourlyEnd: query.hourlyEnd,
      timezone,
      language,
    })
    if (delayMs > 0) await sleep(delayMs)
    const hour = selectRouteHour(hours, query.forecastStart)
    if (!hour)
      throw new Error(
        `WeatherKit returned no ${query.forecastStart} hour for activity ${item.activityId}`,
      )
    return hour
  }

  const oldestHistorical = historical[0]
  const probe = oldestHistorical?.queries[0]
  if (oldestHistorical && probe && maxCalls > 0) {
    try {
      prefetched.set(
        `${oldestHistorical.activityId}:${probe.forecastStart}`,
        await fetchQuery(oldestHistorical, probe),
      )
      console.log(
        `[weather] historical availability confirmed at ${probe.forecastStart} for ${oldestHistorical.activityId}`,
      )
    } catch (err) {
      if (err instanceof WeatherKitRequestError && (err.status === 401 || err.status === 403))
        throw err
      historical = []
      console.warn(
        `[weather] historical backfill unavailable at ${probe.forecastStart}: ${err instanceof Error ? err.message : err}`,
      )
    }
  }

  let fetched = 0
  const skipped = candidates.length - refreshCandidates.length
  let current: WeatherSnapshot | null = prev?.current ?? null
  const sortedActivities = (): Record<string, WeatherActivity> =>
    Object.fromEntries(
      Object.values(activities)
        .sort((left, right) => left.start.localeCompare(right.start))
        .map(activity => [String(activity.activityId), activity]),
    )
  const persist = async (): Promise<void> => {
    const ordered = sortedActivities()
    const cache: WeatherCache = {
      version: CACHE_VERSION,
      lastSync: Date.now(),
      current,
      attribution,
      uvCalibration,
      activities: ordered,
      days: summarizeWeatherDays(ordered),
    }
    await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
    await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  }

  for (const item of [...recent, ...historical]) {
    const key = String(item.activityId)
    const prefetchedCount = item.queries.filter(query =>
      prefetched.has(`${item.activityId}:${query.forecastStart}`),
    ).length
    if (item.queries.length - prefetchedCount > maxCalls - calls) {
      console.warn(
        `[weather] call budget leaves ${refreshCandidates.length - fetched} activities for the next checkpointed run`,
      )
      break
    }
    const routeHours: WeatherRouteHour[] = []
    let failed = false
    try {
      for (const query of item.queries) {
        const prefetchKey = `${item.activityId}:${query.forecastStart}`
        const hour = prefetched.get(prefetchKey) ?? (await fetchQuery(item, query))
        prefetched.delete(prefetchKey)
        const routeHour = weatherRouteHourFromForecast(item, hour, query)
        if (!routeHour) throw new Error(`invalid overlap for ${query.forecastStart}`)
        routeHours.push(routeHour)
      }
    } catch (err) {
      if (err instanceof WeatherKitRequestError && (err.status === 401 || err.status === 403))
        throw err
      failed = true
      console.warn(
        `[weather] ${item.date} ${item.activityId}: ${err instanceof Error ? err.message : err}`,
      )
    }
    const weather = failed
      ? null
      : weatherActivityFromRouteHours(item, routeHours, Date.now(), activities[key]?.surfaceCurrent)
    if (weather && routeHours.length === item.queries.length) {
      activities[key] = weather
      fetched += 1
      console.log(
        `[weather] ${item.date} ${item.activityId}: ${routeHours.length} route-hours, ${weather.windKph ?? 'n/a'} km/h ${weather.windDir ?? ''}`,
      )
      if (fetched % 10 === 0) await persist()
    } else if (!failed) {
      console.warn(`[weather] ${item.date} ${item.activityId}: no overlapping hourly weather`)
    }
  }

  for (const [id, activity] of Object.entries(activities)) {
    const fingerprint = activity.routeFingerprint
    if (!fingerprint || !weatherActivityHasCompleteRouteHours(activity, fingerprint))
      delete activities[id]
  }

  const latestLocation = Object.values(activities)
    .sort((a, b) => a.start.localeCompare(b.start))
    .at(-1)
  if (latestLocation && calls < maxCalls) {
    const nowMs = Date.now()
    try {
      calls += 1
      const hours = await fetchWeatherKitHours(config, {
        latitude: latestLocation.latitude,
        longitude: latestLocation.longitude,
        hourlyStart: floorHour(nowMs - HOUR_MS),
        hourlyEnd: ceilHour(nowMs + HOUR_MS),
        timezone,
        language,
      })
      current = weatherSnapshotFromHours(latestLocation, hours, nowMs)
      if (current)
        console.log(
          `[weather] current ${current.forecastStart}: ${current.temperatureC ?? 'n/a'} C, ${current.precipitationChance == null ? 'n/a' : `${Math.round(current.precipitationChance * 100)}%`} precipitation`,
        )
    } catch (err) {
      if (err instanceof WeatherKitRequestError && (err.status === 401 || err.status === 403))
        throw err
      console.warn(`[weather] current: ${err instanceof Error ? err.message : err}`)
    }
  }

  const calibrationPairs: GardenUvCalibrationPair[] = []
  for (const item of allCandidates) {
    const activityWeather = activities[String(item.activityId)]
    const detail = source.details[String(item.activityId)]
    if (!activityWeather || !detail) continue
    const pelotan = parseActivityProviderReports(
      detail.description,
      item.activityId,
      detail.fetchedAt,
    ).pelotan
    if (pelotan?.score == null || pelotan.severity == null) continue
    const environment = buildActivityEnvironment({
      activityId: item.activityId,
      elapsedTimeS: item.durationS,
      movingTimeS: sourceActivities.get(item.activityId)?.movingTime ?? item.durationS,
      timeS: item.stream.timeS,
      distanceM: item.stream.distanceM,
      latlng: item.stream.latlng,
      openWater: normalizeKind(sourceActivities.get(item.activityId)?.sportType ?? '') === 'swim',
      weather: activityWeather,
      attribution,
      computedAt: Date.now(),
    }).environment
    if (
      environment?.doseClocks.elapsedSed == null ||
      environment.doseClocks.movingTelemetrySed == null
    )
      continue
    calibrationPairs.push({
      activityId: item.activityId,
      date: item.date,
      score: pelotan.score,
      severity: pelotan.severity,
      elapsedSed: environment.doseClocks.elapsedSed,
      movingTelemetrySed: environment.doseClocks.movingTelemetrySed,
    })
  }
  uvCalibration =
    uvCalibration?.status === 'active'
      ? auditGardenUvCalibration(uvCalibration, calibrationPairs)
      : uvCalibration?.status === 'rejected' || uvCalibration?.status === 'suspended'
        ? uvCalibration
        : calibrateGardenUvScore(calibrationPairs)
  console.log(
    `[weather] UV calibration ${uvCalibration.status}: ${calibrationPairs.length} exact pairs`,
  )

  await persist()
  await refreshTriathlonRouteSource()
  console.log(
    `[weather] fetched ${fetched}, skipped ${skipped}, used ${calls}/${maxCalls} calls, cached ${Object.keys(activities).length} activities -> ${cacheFile}`,
  )
}

main().catch(err => {
  console.error(`[weather] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
