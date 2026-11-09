import { randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type { GarminCache } from '../plugins/stores/garmin'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import { shiftIsoDay } from '../util/local-date'
import { syncRefreshDays } from '../util/sync-refresh-window'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { fetchGarminHealthRange } from './garmin-health-sync'
import { syncGarminActivities } from './sync-garmin/activities'
import {
  fetchGarminSleepRange,
  fetchRunningLactateThreshold,
  syncGarminBiometrics,
} from './sync-garmin/biometrics'
import {
  CACHE_VERSION,
  cacheFile,
  prepareGarminRefreshWindows,
  readCache,
  replaceGarminSleepCoverage,
  resolveGarminFetch,
} from './sync-garmin/refresh-window'
export {
  garminHealthRefreshStart,
  garminLactateThresholdRefreshStart,
  garminRefreshStart,
  garminSleepRefreshStart,
  resolveGarminFetch,
} from './sync-garmin/refresh-window'
export {
  fetchGarminSleepRange,
  mergeGarminLactateThresholdHistory,
  mergeGarminVo2Range,
  mergeGarminWeightRange,
  resolveGarminWeightDay,
} from './sync-garmin/biometrics'
export {
  mergeGarminFitTrainingEffect,
  needsGarminActivityFit,
  reconcileGarminActivities,
} from './sync-garmin/activities'

async function writeCache(cache: GarminCache): Promise<void> {
  const target = await fs.realpath(cacheFile).catch(() => cacheFile)
  const temporary = `${target}.${randomUUID()}.tmp`
  await fs.mkdir(dirname(target), { recursive: true })
  try {
    await fs.writeFile(temporary, JSON.stringify(cache, null, 2))
    await fs.rename(temporary, target)
  } finally {
    await fs.rm(temporary, { force: true })
  }
}

function sleepCacheFields(
  previous: GarminCache | null,
  refresh: Awaited<ReturnType<typeof fetchGarminSleepRange>> | undefined,
  start: string,
  end: string,
  fetchedAt: number,
): Pick<GarminCache, 'sleep' | 'sleepLastSync' | 'sleepCoverage'> {
  const { sleep: cached, sleepLastSync, sleepCoverage } = previous ?? {}
  if (!refresh) return { sleep: cached, sleepLastSync, sleepCoverage }
  const sleep = { ...cached }
  for (const range of refresh.coverage) {
    for (let date = range.start; date <= range.end; date = shiftIsoDay(date, 1)) {
      const row = refresh.sleep[date]
      if (row) sleep[date] = row
      else delete sleep[date]
    }
  }
  return {
    sleep: Object.fromEntries(
      Object.entries(sleep).sort(([left], [right]) => left.localeCompare(right)),
    ),
    sleepLastSync: refresh.fetchedDays > 0 ? fetchedAt : sleepLastSync,
    sleepCoverage: replaceGarminSleepCoverage(sleepCoverage ?? [], refresh.coverage, start, end),
  }
}

async function prepareGarminSyncScope(): Promise<{
  flags: string[]
  sleepOnly: boolean
  previous: GarminCache | null
  session: GarminConnectSession
  base: string
  refreshWindowDays: number
  stale: boolean
}> {
  const flags = process.argv.slice(2)
  for (const flag of flags)
    if (flag !== '--sleep-only' && flag !== '--lactate-threshold-only' && flag !== '--health-only')
      throw new Error(`unknown flag ${flag}`)
  if (flags.length > 1) throw new Error('Choose only one Garmin sync scope')
  const sleepOnly = flags.includes('--sleep-only')
  const previous = await readCache()
  const session = await readGarminConnectSession()
  const base = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )
  const refreshWindowDays = syncRefreshDays()
  const stale = !previous || (previous.version ?? 0) < CACHE_VERSION
  return { flags, sleepOnly, previous, session, base, refreshWindowDays, stale }
}

async function main(): Promise<void> {
  const { flags, sleepOnly, previous, session, base, refreshWindowDays, stale } =
    await prepareGarminSyncScope()
  if (process.argv.includes('--lactate-threshold-only')) {
    if (!previous) throw new Error('Run a Garmin sync before refreshing lactate threshold only')
    const runningLactateThreshold = await fetchRunningLactateThreshold(
      session,
      base,
      previous.runningLactateThreshold,
      refreshWindowDays,
      stale,
    )
    const latest = await readCache()
    if (!latest) throw new Error('Garmin cache disappeared during lactate threshold refresh')
    await writeCache({ ...latest, runningLactateThreshold, lactateThresholdLastSync: Date.now() })
    await refreshTriathlonRouteSource()
    console.log('[garmin] refreshed running lactate threshold', runningLactateThreshold)
    return
  }
  const { pageSize, delayMs, maxActivities, fetchStreams, start, end, healthStart, sleepStart } =
    await prepareGarminRefreshWindows(previous, refreshWindowDays)

  let health = previous?.health
  let healthLastSync = previous?.healthLastSync
  if (!sleepOnly) {
    const result = await fetchGarminHealthRange(
      session,
      base,
      health ?? {},
      healthStart,
      end,
      delayMs,
    )
    health = result.health
    if (result.responses > 0) healthLastSync = Date.now()
    if (flags.includes('--health-only')) {
      const latest = await readCache()
      if (previous && !latest) throw new Error('Garmin cache disappeared during health refresh')
      const cache: GarminCache = {
        ...(latest ?? { version: CACHE_VERSION, lastSync: 0, activities: {} }),
        health,
        healthLastSync,
      }
      await writeCache(cache)
      await refreshTriathlonRouteSource()
      console.log(
        `[garmin] health: ${result.responses} responses, ${result.failures} failures, ${Object.keys(health).length} days`,
      )
      if (result.failures)
        throw new Error(`Garmin health refresh incomplete: ${result.failures} failed requests`)
      return
    }
  }

  let sleepRefresh: Awaited<ReturnType<typeof fetchGarminSleepRange>> | undefined
  console.log(`[garmin] fetching Garmin Connect sleep ${sleepStart} -> ${end}`)
  try {
    sleepRefresh = await fetchGarminSleepRange(
      session,
      base,
      previous?.sleep ?? {},
      sleepStart,
      end,
      delayMs,
    )
    console.log(
      `[garmin] sleep: ${sleepRefresh.fetchedDays} responses, ${sleepRefresh.failedDays.length} failed, ${Object.keys(sleepRefresh.sleep).length} dates with measurements`,
    )
    if (sleepOnly && sleepRefresh.fetchedDays === 0)
      throw new Error('No Garmin sleep dates fetched')
  } catch (error) {
    if (sleepOnly) throw error
    console.warn(`[garmin] sleep fetch failed: ${error instanceof Error ? error.message : error}`)
  }
  const sleepFetchedAt = Date.now()
  if (sleepOnly) {
    const latest = await readCache()
    if (previous && !latest) throw new Error('Garmin cache disappeared during sleep refresh')
    const cache: GarminCache = {
      ...(latest ?? { version: CACHE_VERSION, lastSync: 0, activities: {} }),
      ...sleepCacheFields(latest, sleepRefresh, sleepStart, end, sleepFetchedAt),
    }
    await writeCache(cache)
    await refreshTriathlonRouteSource()
    console.log(
      `[garmin] wrote ${Object.keys(cache.sleep ?? {}).length} sleep dates -> ${cacheFile}`,
    )
    if (sleepRefresh?.failedDays.length)
      throw new Error(`Garmin sleep refresh incomplete for ${sleepRefresh.failedDays.join(', ')}`)
    return
  }

  const {
    sorted,
    sortedStreams,
    sortedGearShifts,
    sortedCyclingDynamics,
    sortedFitTrainingEffects,
    sortedSwims,
    sortedClimbs,
    sortedRunWalks,
    details,
    streamDetails,
    activityArchives,
    climbDetails,
    runWalkDetails,
    skipped,
  } = await syncGarminActivities(
    session,
    base,
    previous,
    start,
    end,
    pageSize,
    maxActivities,
    fetchStreams,
    delayMs,
  )

  const { lactateThresholdOutcome, vo2max, weight } = await syncGarminBiometrics(
    session,
    base,
    previous,
    start,
    end,
    refreshWindowDays,
    stale,
    delayMs,
  )

  const now = Date.now()
  const cache: GarminCache = {
    version: CACHE_VERSION,
    lastSync: now,
    activities: sorted,
    streams: sortedStreams,
    gearShifts: sortedGearShifts,
    cyclingDynamics: sortedCyclingDynamics,
    fitTrainingEffects: sortedFitTrainingEffects,
    swims: sortedSwims,
    climbs: sortedClimbs,
    runWalks: sortedRunWalks,
    vo2max,
    runningLactateThreshold: resolveGarminFetch(
      lactateThresholdOutcome,
      previous?.runningLactateThreshold,
    ),
    lactateThresholdLastSync: lactateThresholdOutcome.ok ? now : previous?.lactateThresholdLastSync,
    weight,
    ...sleepCacheFields(
      (await readCache()) ?? previous,
      sleepRefresh,
      sleepStart,
      end,
      sleepFetchedAt,
    ),
    health,
    healthLastSync,
  }
  await writeCache(cache)
  await refreshTriathlonRouteSource()
  console.log(
    `[garmin] wrote ${Object.keys(sorted).length} activities (${details} detail responses, ${streamDetails} stream responses, ${activityArchives} activity archives, ${Object.values(sortedGearShifts).reduce((sum, shifts) => sum + shifts.length, 0)} shift states, ${Object.values(sortedCyclingDynamics).reduce((sum, dynamics) => sum + dynamics.time.length, 0)} cycling dynamics samples, ${Object.values(sortedSwims).reduce((sum, swim) => sum + swim.lengths.length, 0)} swim lengths, ${climbDetails} climb responses, ${Object.values(sortedClimbs).reduce((sum, segments) => sum + segments.length, 0)} climbs, ${runWalkDetails} run/walk responses, ${Object.values(sortedRunWalks).reduce((sum, runWalk) => sum + runWalk.segments.length, 0)} run/walk segments, ${skipped} skipped) -> ${cacheFile}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[garmin] sync failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
