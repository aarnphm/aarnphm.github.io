import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type { GarminCache } from '../plugins/stores/garmin'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import { joinSegments, QUARTZ } from '../util/path'
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
    await fs.writeFile(
      cacheFile,
      JSON.stringify(
        { ...latest, runningLactateThreshold, lactateThresholdLastSync: Date.now() },
        null,
        2,
      ),
    )
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
      await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
      await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
      await refreshTriathlonRouteSource()
      console.log(
        `[garmin] health: ${result.responses} responses, ${result.failures} failures, ${Object.keys(health).length} days`,
      )
      if (result.failures)
        throw new Error(`Garmin health refresh incomplete: ${result.failures} failed requests`)
      return
    }
  }

  let overnightSleep = previous?.sleep ?? {}
  let sleepLastSync = previous?.sleepLastSync
  let sleepFailedDays: string[] = []
  console.log(`[garmin] fetching Garmin Connect sleep ${sleepStart} -> ${end}`)
  try {
    const result = await fetchGarminSleepRange(
      session,
      base,
      overnightSleep,
      sleepStart,
      end,
      delayMs,
    )
    overnightSleep = result.sleep
    sleepFailedDays = result.failedDays
    if (result.fetchedDays > 0) sleepLastSync = Date.now()
    console.log(
      `[garmin] sleep: ${result.fetchedDays} responses, ${result.failedDays.length} failed, ${Object.keys(overnightSleep).length} nights with measurements`,
    )
    if (sleepOnly && result.fetchedDays === 0) throw new Error('No Garmin sleep dates fetched')
  } catch (error) {
    if (sleepOnly) throw error
    console.warn(`[garmin] sleep fetch failed: ${error instanceof Error ? error.message : error}`)
  }
  if (sleepOnly) {
    const latest = await readCache()
    if (previous && !latest) throw new Error('Garmin cache disappeared during sleep refresh')
    const cache: GarminCache = {
      ...(latest ?? { version: CACHE_VERSION, lastSync: 0, activities: {} }),
      sleep: overnightSleep,
      sleepLastSync,
    }
    await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
    await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
    await refreshTriathlonRouteSource()
    console.log(`[garmin] wrote ${Object.keys(overnightSleep).length} sleep nights -> ${cacheFile}`)
    if (sleepFailedDays.length > 0)
      throw new Error(`Garmin sleep refresh incomplete for ${sleepFailedDays.join(', ')}`)
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
    sleep: overnightSleep,
    sleepLastSync,
    health,
    healthLastSync,
  }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
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
