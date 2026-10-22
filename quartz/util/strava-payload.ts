import { statSync } from 'node:fs'
import { resolve } from 'node:path'
import type { AppleCache } from '../plugins/stores/apple'
import type { GarminCache } from '../plugins/stores/garmin'
import type { OuraCache } from '../plugins/stores/oura'
import type {
  ActivityTrackingEntry,
  ManualFuelingEntry,
  ManualMovesEntry,
  ManualSaunaEntry,
  ManualStrengthEntry,
} from '../plugins/stores/tracking'
import {
  ATHLETE,
  buildAnalytics,
  hrZoneUppers,
  parseVo2Lab,
  type Analytics,
  type AnalyticsInputs,
} from '../plugins/stores/analytics'
import {
  parseCoreBodyTemperatureCache,
  type CoreBodyTemperatureCache,
} from '../plugins/stores/core-body-temperature'
import {
  applyActivityTracking,
  applyManualFueling,
  applyManualMoves,
  applyManualSauna,
  applyManualStrength,
  buildPayload,
  emptyHealth,
  type StravaPayload,
  type StravaRawCache,
} from '../plugins/stores/strava'
import { parseWahooCache, type WahooCache, type WahooData } from '../plugins/stores/wahoo'
import { parseWeatherCache, type WeatherCache } from '../plugins/stores/weather'
import { applyHeartRatePhysiology } from './heart-rate-physiology'
import { joinSegments, QUARTZ } from './path'
import { latestProviderSync } from './provider-sync'
import {
  pairedActivityTracking,
  readRouvyWahooPairs,
  rouvyWahooPairsPath,
  withoutPairedWahooDuplicates,
} from './rouvy-wahoo-pairs'
import { applyStaminaLedger, garminRecoveryReadings } from './stamina-ledger'
import { readStravaCacheFileSync } from './strava-cache-file'
import {
  enrichActivityDevices,
  enrichRouteLessHeartRate,
  enrichRunDynamics,
} from './strava-payload/apple-workout-enrichment'
import { enrichCoreBodyTemperature } from './strava-payload/core-temperature-enrichment'
import { enrichSwimMetrics } from './strava-payload/swim-enrichment'
import {
  enrichCalculatedExerciseLoads,
  enrichCalculatedIntensityFactors,
  enrichCalculatedTrainingEffects,
  enrichRunPaceZones,
} from './strava-payload/training-load-enrichment'
import {
  enrichMyWindsockRoutes,
  readJson,
  windArchiveStamp,
} from './strava-payload/wind-enrichment'
import { enrichTrainingPeaksStrength } from './strength-trainingpeaks'
import { applySwimPhysiology } from './swim-physiology'
import { readSwimProvenance, swimProvenanceStamp } from './swim-provenance'
import { parseTrainingPeaksCalendar, type TrainingPeaksCalendar } from './trainingpeaks-calendar'
import { TRAININGPEAKS_CALENDAR_CACHE } from './trainingpeaks-calendar-cache'
import {
  buildTriathlonDailyAnalytics,
  type TriathlonDailyAnalytics,
} from './triathlon-day-analytics'
import { loadTrackedWahooFits, wahooTrackingStamp } from './wahoo-tracking'

// The tests exercise these enrichment steps directly; production callers use buildStravaData.
export {
  enrichActivityDevices,
  enrichRouteLessHeartRate,
  enrichRunDynamics,
} from './strava-payload/apple-workout-enrichment'
export { enrichCoreBodyTemperature } from './strava-payload/core-temperature-enrichment'
export { enrichSwimMetrics, swimActivityIntervals } from './strava-payload/swim-enrichment'
export {
  enrichCalculatedExerciseLoads,
  enrichCalculatedIntensityFactors,
  enrichCalculatedTrainingEffects,
  enrichRunPaceZones,
} from './strava-payload/training-load-enrichment'

export const stravaCachePath = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const ouraCachePath = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
const garminCachePath = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
const wahooCachePath = joinSegments(QUARTZ, '.quartz-cache', 'wahoo.json')
const appleCachePath = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')
const coreBodyTemperatureCachePath = joinSegments(
  QUARTZ,
  '.quartz-cache',
  'core-body-temperature.json',
)
const weatherCachePath = joinSegments(QUARTZ, '.quartz-cache', 'weather.json')

const stamp = (path: string): number => {
  try {
    return statSync(path).mtimeMs
  } catch {
    return 0
  }
}

function readWahooCache(): WahooCache | null {
  const value = readJson<unknown>(wahooCachePath)
  if (value == null) return null
  try {
    return parseWahooCache(value)
  } catch {
    return null
  }
}

export type LoadedStravaPayload = StravaPayload & { dailyAnalytics: TriathlonDailyAnalytics }

type StravaPayloadAnalyticsInputs = Pick<AnalyticsInputs, 'weights' | 'events' | 'dexa' | 'vo2labs'>

interface ManualActivityTracking {
  readonly activities: readonly ActivityTrackingEntry[]
  readonly fueling: readonly ManualFuelingEntry[]
  readonly strength: readonly ManualStrengthEntry[]
  readonly moves?: readonly ManualMovesEntry[]
  readonly sauna: readonly ManualSaunaEntry[]
}

export function applyManualActivityTracking(
  payload: StravaPayload,
  tracking: ManualActivityTracking | null | undefined,
  oura: OuraCache | null,
  weather: WeatherCache | null,
  garmin: GarminCache | null,
): void {
  applyManualFueling(payload, tracking?.fueling ?? [])
  applyManualMoves(payload, tracking?.moves ?? [])
  applyManualSauna(
    payload,
    tracking?.sauna ?? [],
    oura?.heartRate ?? [],
    undefined,
    weather,
    garmin,
  )
  applyManualStrength(payload, tracking?.strength ?? [])
}

export interface StravaDataSources {
  trainingPeaks?: TrainingPeaksCalendar | null
  strava: StravaRawCache | null
  oura: OuraCache | null
  garmin: GarminCache | null
  wahoo: WahooCache | null
  apple: AppleCache | null
  core: CoreBodyTemperatureCache | null
  weather: WeatherCache | null
}

export interface LoadedStravaData {
  payload: LoadedStravaPayload
  analytics: Analytics
  trackedCache: StravaRawCache | null
  sources: StravaDataSources
  wahooActivities: WahooData['activities']
  generatedAt: number
}

type PayloadMemo = {
  sourceSalt: string
  stamps: string
  entries: Map<string, { manualKey: string; data: LoadedStravaData }>
}

declare global {
  var __quartzStravaPayloadMemo: PayloadMemo | undefined
  var __quartzStravaPayloadSourceInputs: string[] | undefined
}

// Watch mode reuses one payload across bundle generations. The CLI publishes this module's import
// closure; an edit to any of those files changes the salt and drops the payload.
const payloadMemo = (globalThis.__quartzStravaPayloadMemo ??= {
  sourceSalt: '',
  stamps: '',
  entries: new Map(),
})
// Capture this bundle's inputs before parsing starts; watched edits can arrive before its first
// payload read. A lazy fingerprint would mark the old calculation as current for the next bundle.
const payloadSourceSalt = (globalThis.__quartzStravaPayloadSourceInputs ?? [])
  .map(fp => {
    try {
      const info = statSync(fp)
      return `${fp}:${info.mtimeMs}:${info.size}`
    } catch {
      return `${fp}:missing`
    }
  })
  .join('|')

export function loadStravaPayloadSync(
  since: string | undefined,
  manualTracking: ManualActivityTracking | null | undefined,
  analyticsInputs: StravaPayloadAnalyticsInputs = {},
  contentDirectory = 'content',
): LoadedStravaPayload {
  return loadStravaDataSync(since, manualTracking, analyticsInputs, contentDirectory).payload
}

export function loadStravaDataSync(
  since: string | undefined,
  manualTracking: ManualActivityTracking | null | undefined,
  analyticsInputs: StravaPayloadAnalyticsInputs = {},
  contentDirectory = 'content',
): LoadedStravaData {
  const manualKey = JSON.stringify({
    activities: manualTracking?.activities ?? [],
    fueling: manualTracking?.fueling ?? [],
    strength: manualTracking?.strength ?? [],
    moves: manualTracking?.moves ?? [],
    sauna: manualTracking?.sauna ?? [],
    analytics: analyticsInputs,
  })
  const pairs = readRouvyWahooPairs(contentDirectory)
  const pairEntries = pairs.map(pair => ({
    activityId: pair.canonicalId,
    garminActivityId: null,
    virtual: true,
    wahooFitPath: pair.fitPath,
  }))
  const stamps = `${stamp(stravaCachePath)}:${stamp(ouraCachePath)}:${stamp(garminCachePath)}:${stamp(wahooCachePath)}:${stamp(weatherCachePath)}:${stamp(appleCachePath)}:${stamp(coreBodyTemperatureCachePath)}:${stamp(rouvyWahooPairsPath(contentDirectory))}:${wahooTrackingStamp([...pairEntries, ...(manualTracking?.activities ?? [])], contentDirectory)}:${windArchiveStamp(contentDirectory)}:${swimProvenanceStamp(contentDirectory)}`
  const sourceSalt = `${payloadSourceSalt}:${stamp(TRAININGPEAKS_CALENDAR_CACHE)}`
  if (payloadMemo.stamps !== stamps || payloadMemo.sourceSalt !== sourceSalt) {
    payloadMemo.entries.clear()
    payloadMemo.stamps = stamps
    payloadMemo.sourceSalt = sourceSalt
  }
  const key = JSON.stringify([resolve(contentDirectory), since ?? ''])
  const cached = payloadMemo.entries.get(key)
  if (cached?.manualKey === manualKey) return cached.data
  const strava = readStravaCacheFileSync(stravaCachePath)
  const oura = readJson<OuraCache>(ouraCachePath)
  const garmin = readJson<GarminCache>(garminCachePath)
  const wahoo = readWahooCache()
  const apple = readJson<AppleCache>(appleCachePath)
  const core = parseCoreBodyTemperatureCache(readJson<unknown>(coreBodyTemperatureCachePath))
  const weather = parseWeatherCache(readJson<unknown>(weatherCachePath))
  const loaded = buildStravaData(
    {
      strava,
      oura,
      garmin,
      wahoo,
      apple,
      core,
      weather,
      trainingPeaks: parseTrainingPeaksCalendar(readJson<unknown>(TRAININGPEAKS_CALENDAR_CACHE)),
    },
    since,
    manualTracking,
    analyticsInputs,
    contentDirectory,
  )
  payloadMemo.entries.set(key, { manualKey, data: loaded })
  return loaded
}

export function buildStravaData(
  sources: StravaDataSources,
  since: string | undefined,
  manualTracking: ManualActivityTracking | null | undefined,
  analyticsInputs: StravaPayloadAnalyticsInputs = {},
  contentDirectory = 'content',
): LoadedStravaData {
  const { garmin, oura, apple, core, weather } = sources
  const pairs = readRouvyWahooPairs(contentDirectory)
  const activityTracking = pairedActivityTracking(
    pairs,
    manualTracking?.activities ?? [],
    sources.strava,
  )
  const swimProvenance = readSwimProvenance(contentDirectory)
  const wahoo = loadTrackedWahooFits(
    sources.wahoo,
    sources.strava,
    activityTracking,
    contentDirectory,
  )
  const strava = withoutPairedWahooDuplicates(sources.strava, pairs, activityTracking, wahoo)
  const generatedAt = latestProviderSync(strava, oura, garmin, sources.wahoo, apple, core, weather)
  const labs = parseVo2Lab(analyticsInputs.vo2labs)
  const latestLab = labs.at(-1)
  const payload = buildPayload(
    strava,
    oura,
    garmin,
    since,
    weather,
    ATHLETE.ftp,
    latestLab ? (hrZoneUppers(latestLab) ?? undefined) : undefined,
    undefined,
    wahoo,
    ATHLETE.hrMax,
    ATHLETE.lt,
    generatedAt,
    activityTracking,
  )
  enrichTrainingPeaksStrength(payload, strava, sources.trainingPeaks)
  applyManualActivityTracking(payload, manualTracking, oura, weather, garmin)
  for (const day of analyticsInputs.weights ?? []) {
    if (day.windKph != null) {
      const health = payload.health[day.date] ?? emptyHealth()
      payload.health[day.date] = {
        ...health,
        windKph: day.windKph,
        windDir: day.windDir ?? health.windDir,
      }
    }
  }
  enrichActivityDevices(payload, apple)
  enrichRouteLessHeartRate(payload, apple)
  for (const provenance of swimProvenance) {
    const detail = payload.details[String(provenance.activityId)]
    if (
      detail?.sport !== 'swim' ||
      detail.garmin?.activityId !== provenance.garminActivityId ||
      Math.abs(Date.parse(detail.start) - Date.parse(provenance.startDate)) > 1000
    )
      continue
    detail.swimProvenance = {
      source: provenance.source,
      note: provenance.note,
      trainingPeaksWorkoutId: provenance.trainingPeaksWorkoutId,
    }
  }
  enrichSwimMetrics(payload, apple, garmin)
  enrichRunDynamics(payload, apple)
  enrichCoreBodyTemperature(payload, core)
  enrichMyWindsockRoutes(payload, contentDirectory)
  for (const detail of Object.values(payload.details)) {
    applyHeartRatePhysiology(detail, ATHLETE.hrMax)
    applySwimPhysiology(detail, ATHLETE.hrMax)
  }
  const trackedCache = applyActivityTracking(strava, garmin, activityTracking, wahoo)
  const analytics = buildAnalytics(trackedCache, {
    ...analyticsInputs,
    oura,
    apple,
    core,
    garmin,
    weather,
    ftp: ATHLETE.ftp,
    powerCurve: {
      sixWeeks: payload.powerCurveRef,
      year: payload.powerCurveYearRef,
      yearLabel: payload.powerCurveYear,
      criticalPower: payload.criticalPower,
      criticalPowerYear: payload.criticalPowerYear,
      ftp: ATHLETE.ftp,
      goalFtp: ATHLETE.goalFTP,
    },
    zones: payload.zones,
    activityDetails: payload.details,
    since,
    generatedAt,
  })
  enrichRunPaceZones(payload, analytics.distributions)
  enrichCalculatedIntensityFactors(payload, analytics.activities, ATHLETE.ftp, ATHLETE.lt)
  enrichCalculatedExerciseLoads(payload)
  enrichCalculatedTrainingEffects(payload)
  // Recovery increments read the calculated loads of sessions Garmin never processed.
  applyStaminaLedger(payload.details, ATHLETE.hrMax, garminRecoveryReadings(garmin?.health))
  const ouraDetails = oura?.details ?? {}
  const loaded: LoadedStravaPayload = {
    ...payload,
    dailyAnalytics: buildTriathlonDailyAnalytics(analytics, ouraDetails, payload.details),
  }
  const compactStrava = strava ? { ...strava, geo: undefined, activityDetails: undefined } : null
  const compactTrackedCache = trackedCache
    ? { ...trackedCache, geo: undefined, activityDetails: undefined }
    : null
  const compactSources: StravaDataSources = {
    trainingPeaks: sources.trainingPeaks ?? null,
    strava: compactStrava,
    oura: oura ? { ...oura, heartRate: undefined } : null,
    garmin,
    wahoo: null,
    apple: apple ? { ...apple, workouts: {}, swims: {} } : null,
    core: null,
    weather,
  }
  const compactWahooActivities: WahooData['activities'] = {}
  for (const [id, activity] of Object.entries(wahoo?.activities ?? {})) {
    const nextActivity = { ...activity, laps: undefined } as WahooData['activities'][string] & {
      stream?: unknown
    }
    delete nextActivity.stream
    compactWahooActivities[id] = nextActivity
  }
  return {
    payload: loaded,
    analytics,
    trackedCache: compactTrackedCache,
    sources: compactSources,
    wahooActivities: compactWahooActivities,
    generatedAt,
  }
}
