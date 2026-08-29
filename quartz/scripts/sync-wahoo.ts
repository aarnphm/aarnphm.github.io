import fs from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import {
  normalizeWahooSport,
  parseWahooCache,
  WAHOO_CACHE_VERSION,
  type WahooActivity,
  type WahooCache,
  type WahooMetrics,
  type WahooSummary,
} from '../plugins/stores/wahoo'
import { joinSegments, QUARTZ } from '../util/path'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import {
  isWahooOriginatedSummary,
  isWahooRestrictedWorkoutSummaryError,
  WahooApiError,
  WahooCloudClient,
  wahooCloudClientFromEnv,
  type WahooWorkoutDto,
  type WahooWorkoutSummaryDto,
} from '../util/wahoo-cloud'
import { decodeWahooFit, wahooFitSha256, type WahooFitData } from '../util/wahoo-fit'

export const WAHOO_CACHE_FILE = joinSegments(QUARTZ, '.quartz-cache', 'wahoo.json')

function first<T>(preferred: T | null, fallback: T | null): T | null {
  return preferred ?? fallback
}

function localStartDate(startDate: string, timeZone: string | null): string {
  if (!timeZone) return startDate
  let formatter: Intl.DateTimeFormat
  try {
    formatter = new Intl.DateTimeFormat('en-CA', {
      timeZone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hourCycle: 'h23',
    })
  } catch {
    throw new Error(`Wahoo summary contains invalid time zone ${timeZone}`)
  }
  const parts = new Map(
    formatter
      .formatToParts(new Date(startDate))
      .filter(part => part.type !== 'literal')
      .map(part => [part.type, part.value]),
  )
  const year = parts.get('year')
  const month = parts.get('month')
  const day = parts.get('day')
  const hour = parts.get('hour')
  const minute = parts.get('minute')
  const second = parts.get('second')
  if (!year || !month || !day || !hour || !minute || !second)
    throw new Error(`Wahoo could not format start time in ${timeZone}`)
  return `${year}-${month}-${day}T${hour}:${minute}:${second}`
}

function mergedMetrics(summary: WahooWorkoutSummaryDto, fit: WahooFitData): WahooMetrics {
  return {
    totalCalories: first(summary.calories, fit.metrics.totalCalories),
    avgHeartRate: first(summary.heartRateAvg, fit.metrics.avgHeartRate),
    maxHeartRate: fit.metrics.maxHeartRate,
    avgPower: first(summary.powerAvgW, fit.metrics.avgPower),
    normalizedPower: first(summary.normalizedPowerW, fit.metrics.normalizedPower),
    maxPower: fit.metrics.maxPower,
    avgCadence: first(summary.cadenceAvg, fit.metrics.avgCadence),
    totalAscentM: first(summary.ascentM, fit.metrics.totalAscentM),
    totalDescentM: fit.metrics.totalDescentM,
    totalWorkKJ: first(
      summary.workJ == null ? null : summary.workJ / 1000,
      fit.metrics.totalWorkKJ,
    ),
    trainingStressScore: first(summary.trainingStressScore, fit.metrics.trainingStressScore),
    intensityFactor: fit.metrics.intensityFactor,
    avgSpeedMps: first(summary.speedAvgMps, fit.metrics.avgSpeedMps),
    maxSpeedMps: fit.metrics.maxSpeedMps,
    avgTemperatureC: fit.metrics.avgTemperatureC,
  }
}

function normalizedSummary(summary: WahooWorkoutSummaryDto): WahooSummary {
  if (summary.fitnessAppId == null)
    throw new Error(`Wahoo summary ${summary.id} has no fitness_app_id`)
  return {
    id: summary.id,
    name: summary.name,
    timeZone: summary.timeZone,
    manual: summary.manual ?? false,
    edited: summary.edited ?? false,
    fitnessAppId: summary.fitnessAppId,
    durationPausedS: summary.durationPausedS,
    createdAt: summary.createdAt,
    updatedAt: summary.updatedAt,
  }
}

export function normalizeWahooActivity(
  workout: WahooWorkoutDto,
  summary: WahooWorkoutSummaryDto,
  fit: WahooFitData,
  fitBytes: Uint8Array,
): WahooActivity {
  if (!summary.fileUrl) throw new Error(`Wahoo summary ${summary.id} has no FIT URL`)
  const id = `wahoo:${workout.id}`
  const startDate = fit.startDate
  return {
    id,
    workoutId: workout.id,
    workoutTypeId: workout.workoutTypeId,
    workoutUpdatedAt: workout.updatedAt,
    name: workout.name ?? summary.name,
    sport: normalizeWahooSport(workout.workoutTypeId, fit.sport),
    startDate,
    startDateLocal: localStartDate(startDate, summary.timeZone),
    distanceM: first(summary.distanceM, fit.distanceM),
    movingTimeS: first(summary.durationActiveS, fit.movingTimeS),
    elapsedTimeS: first(summary.durationTotalS, fit.elapsedTimeS),
    sourceDevice: fit.sourceDevice,
    sourceFile: {
      url: summary.fileUrl,
      sha256: wahooFitSha256(fitBytes),
      byteLength: fitBytes.byteLength,
      profileVersion: fit.profileVersion,
    },
    sweatLoss: fit.sweatLoss,
    metrics: mergedMetrics(summary, fit),
    summary: normalizedSummary(summary),
  }
}

export type WahooWorkoutSummaryResolution =
  | { kind: 'available'; summary: WahooWorkoutSummaryDto }
  | { kind: 'missing' }
  | { kind: 'restricted' }

export async function resolveWahooWorkoutSummary(
  client: WahooCloudClient,
  workout: WahooWorkoutDto,
): Promise<WahooWorkoutSummaryResolution> {
  if (workout.summary?.fileUrl && workout.summary.fitnessAppId != null)
    return { kind: 'available', summary: workout.summary }
  try {
    const summary = await client.getWorkoutSummary(workout.id)
    return summary ? { kind: 'available', summary } : { kind: 'missing' }
  } catch (error) {
    if (error instanceof WahooApiError && (error.status === 404 || error.status === 410))
      return { kind: 'missing' }
    if (isWahooRestrictedWorkoutSummaryError(error)) return { kind: 'restricted' }
    throw error
  }
}

export async function fetchWahooCache(
  client: WahooCloudClient,
  previous: WahooCache | null = null,
): Promise<WahooCache> {
  const workouts = await client.listWorkouts()
  const activities: Record<string, WahooActivity> = {}
  const streams: WahooCache['streams'] = {}
  const gearShifts: WahooCache['gearShifts'] = {}
  const cyclingDynamics: WahooCache['cyclingDynamics'] = {}
  let skippedThirdParty = 0
  let skippedIncomplete = 0
  let skippedRestricted = 0
  for (const workout of workouts) {
    const id = `wahoo:${workout.id}`
    const previousActivity = previous?.activities[id]
    const previousStreams = previous?.streams[id]
    const previousGearShifts = previous?.gearShifts[id]
    const previousCyclingDynamics = previous?.cyclingDynamics[id]
    if (
      workout.summary == null &&
      workout.updatedAt != null &&
      previousActivity?.workoutUpdatedAt === workout.updatedAt &&
      previousStreams &&
      previousGearShifts &&
      previousCyclingDynamics
    ) {
      activities[id] = previousActivity
      streams[id] = previousStreams
      gearShifts[id] = previousGearShifts
      cyclingDynamics[id] = previousCyclingDynamics
      continue
    }
    const resolution = await resolveWahooWorkoutSummary(client, workout)
    if (resolution.kind === 'restricted') {
      skippedRestricted++
      continue
    }
    if (resolution.kind === 'missing') {
      skippedIncomplete++
      continue
    }
    const { summary } = resolution
    if (summary.manual === true || !summary.fileUrl) {
      skippedIncomplete++
      continue
    }
    if (!isWahooOriginatedSummary(summary)) {
      skippedThirdParty++
      continue
    }
    const bytes = await client.downloadFit(summary.fileUrl)
    const fit = decodeWahooFit(bytes)
    const activity = normalizeWahooActivity(workout, summary, fit, bytes)
    activities[activity.id] = activity
    streams[activity.id] = fit.streams
    gearShifts[activity.id] = fit.gearShifts
    cyclingDynamics[activity.id] = fit.cyclingDynamics
    console.log(`[wahoo] decoded ${activity.id} ${activity.startDate} ${bytes.byteLength} bytes`)
  }
  console.log(
    `[wahoo] retained ${Object.keys(activities).length}/${workouts.length} completed Wahoo workouts, skipped incomplete=${skippedIncomplete} restricted=${skippedRestricted} third-party=${skippedThirdParty}`,
  )
  return {
    version: WAHOO_CACHE_VERSION,
    lastSync: Date.now(),
    activities,
    streams,
    gearShifts,
    cyclingDynamics,
  }
}

async function readPreviousCache(): Promise<WahooCache | null> {
  try {
    const value: unknown = JSON.parse(await fs.readFile(WAHOO_CACHE_FILE, 'utf8'))
    return parseWahooCache(value)
  } catch {
    return null
  }
}

export async function writeWahooCache(cache: WahooCache, path = WAHOO_CACHE_FILE): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`
  await fs.mkdir(dirname(path), { recursive: true })
  await fs.writeFile(temporary, JSON.stringify(cache, null, 2))
  await fs.rename(temporary, path)
}

async function main(): Promise<void> {
  const client = await wahooCloudClientFromEnv()
  const cache = await fetchWahooCache(client, await readPreviousCache())
  await writeWahooCache(cache)
  await refreshTriathlonRouteSource()
  console.log(
    `[wahoo] wrote ${Object.keys(cache.activities).length} activities → ${WAHOO_CACHE_FILE}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(`[wahoo] sync failed: ${error instanceof Error ? error.message : error}`)
    process.exit(1)
  })
}
