import type { GarminCyclingDynamics, GarminGearShift } from './garmin'
import { normalizeKind, type ActivityKind, type RawStravaActivity } from './strava'
import { parseWahooCacheValue } from './wahoo/cache'

const START_TOLERANCE_MS = 20 * 60 * 1000
const DISTANCE_TOLERANCE_RATIO = 0.08
const DISTANCE_TOLERANCE_M = 1500
const DURATION_TOLERANCE_RATIO = 0.12
const DURATION_TOLERANCE_S = 10 * 60

export const WAHOO_CACHE_VERSION = 4

export type WahooGearShift = GarminGearShift
export type WahooCyclingDynamics = GarminCyclingDynamics

export interface WahooSummitSegment {
  feature: 'summit-segment' | 'summit-freeride'
  uuid: string
  name: string | null
  startDate: string
  endDate: string
  distanceM: number
  durationS: number
  elevationGainM: number | null
  avgGradePct: number | null
  avgSpeedMps: number | null
  avgHeartRate: number | null
  avgPower: number | null
  avgCadence: number | null
}

const BIKE_WORKOUT_TYPE_IDS = new Set([0, 11, 12, 13, 14, 15, 16, 17, 21, 49, 61, 64, 68, 70])
const RUN_WORKOUT_TYPE_IDS = new Set([1, 3, 4, 5, 19, 67, 71])
const SWIM_WORKOUT_TYPE_IDS = new Set([25, 26])
const WALK_WORKOUT_TYPE_IDS = new Set([6, 7, 8, 56])
const HIKE_WORKOUT_TYPE_IDS = new Set([9, 10])
const STRENGTH_WORKOUT_TYPE_IDS = new Set([2, 18, 42, 43])

export interface WahooMetrics {
  totalCalories: number | null
  avgHeartRate: number | null
  maxHeartRate: number | null
  avgPower: number | null
  normalizedPower: number | null
  maxPower: number | null
  avgCadence: number | null
  totalAscentM: number | null
  totalDescentM: number | null
  totalWorkKJ: number | null
  trainingStressScore: number | null
  intensityFactor: number | null
  avgSpeedMps: number | null
  maxSpeedMps: number | null
  avgTemperatureC: number | null
}

export interface WahooSummary {
  id: number
  name: string | null
  timeZone: string | null
  manual: boolean
  edited: boolean
  fitnessAppId: number
  durationPausedS: number | null
  createdAt: string | null
  updatedAt: string | null
}

export interface WahooFitFile {
  url: string
  sha256: string
  byteLength: number
  profileVersion: string
}

export interface WahooSweatLoss {
  fluidMl: number | null
  sodiumMg: number | null
}

export interface WahooStreams {
  timestamps: string[]
  time: number[]
  latlng: ([number, number] | null)[]
  altitude: (number | null)[]
  distance: (number | null)[]
  watts: (number | null)[]
  rightBalance: (number | null)[]
  heartrate: (number | null)[]
  cadence: (number | null)[]
  speed: (number | null)[]
  temperature: (number | null)[]
  respiration: (number | null)[]
  garminStamina?: { current: (number | null)[]; potential: (number | null)[] }
  muscleOxygenPercent: (number | null)[]
  totalHemoglobinConcentration: (number | null)[]
  heatStrainIndex: (number | null)[]
  coreTemperatureC: (number | null)[]
  skinTemperatureC: (number | null)[]
  minuteVentilation: (number | null)[]
  tidalVolume: (number | null)[]
  fluidLossMl: (number | null)[]
  sodiumLossMg: (number | null)[]
}

export interface WahooActivity {
  id: string
  workoutId: number
  workoutTypeId: number
  workoutUpdatedAt?: string | null
  name: string | null
  sport: ActivityKind | null
  startDate: string
  startDateLocal: string
  distanceM: number | null
  movingTimeS: number | null
  elapsedTimeS: number | null
  sourceDevice: string | null
  sourceFile: WahooFitFile
  sweatLoss: WahooSweatLoss
  metrics: WahooMetrics
  summary: WahooSummary
}

export interface WahooLocalFitActivity extends Omit<
  WahooActivity,
  'workoutId' | 'workoutTypeId' | 'workoutUpdatedAt' | 'summary' | 'sourceFile'
> {
  sourceFile: Omit<WahooFitFile, 'url'> & { path: string }
}

export type WahooActivitySource = WahooActivity | WahooLocalFitActivity

export interface WahooData {
  lastSync: number
  activities: Record<string, WahooActivitySource>
  streams: Record<string, WahooStreams>
  gearShifts: Record<string, WahooGearShift[]>
  cyclingDynamics: Record<string, WahooCyclingDynamics>
  summitSegments: Record<string, WahooSummitSegment[]>
}

export interface WahooCache extends WahooData {
  version: typeof WAHOO_CACHE_VERSION
  activities: Record<string, WahooActivity>
}

export interface WahooActivityMatch {
  activity: WahooActivitySource
  score: number
  startDiffMs: number
  distanceDiffM: number | null
  durationDiffS: number | null
}

export interface WahooTitleSyncOptions {
  since?: string | null
  limit?: number
  ids?: ReadonlySet<string>
  includeEdited?: boolean
}

export interface WahooTitleStravaCache {
  activities: Readonly<Record<string, RawStravaActivity>>
}

export interface WahooTitleUpdate {
  stravaId: number
  wahooId: string
  wahooWorkoutId: number
  from: string
  to: string
  startDate: string
  startDateLocal: string
  score: number
  startDiffS: number
  distanceDiffM: number | null
  durationDiffS: number | null
}

export function emptyWahooMetrics(): WahooMetrics {
  return {
    totalCalories: null,
    avgHeartRate: null,
    maxHeartRate: null,
    avgPower: null,
    normalizedPower: null,
    maxPower: null,
    avgCadence: null,
    totalAscentM: null,
    totalDescentM: null,
    totalWorkKJ: null,
    trainingStressScore: null,
    intensityFactor: null,
    avgSpeedMps: null,
    maxSpeedMps: null,
    avgTemperatureC: null,
  }
}

export function normalizeWahooSport(
  workoutTypeId: number,
  fitSport?: string | null,
): ActivityKind | null {
  if (BIKE_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'bike'
  if (RUN_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'run'
  if (SWIM_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'swim'
  if (WALK_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'walk'
  if (HIKE_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'hike'
  if (STRENGTH_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'strength'
  if (workoutTypeId === 66) return 'yoga'
  const normalized = fitSport?.toLowerCase() ?? ''
  if (normalized.includes('cycl') || normalized.includes('bike')) return 'bike'
  if (normalized.includes('run')) return 'run'
  if (normalized.includes('swim')) return 'swim'
  if (normalized.includes('hik') || normalized.includes('mountaineer')) return 'hike'
  if (normalized.includes('walk')) return 'walk'
  if (normalized.includes('yoga') || normalized.includes('pilates')) return 'yoga'
  if (
    normalized.includes('strength') ||
    normalized.includes('training') ||
    normalized.includes('fitness')
  )
    return 'strength'
  return null
}

function positive(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? value : null
}

function distanceDiffM(stravaDistanceM: number, wahooDistanceM: number | null): number | null {
  const distance = positive(wahooDistanceM)
  if (distance == null || stravaDistanceM <= 0) return null
  return Math.abs(distance - stravaDistanceM)
}

function distanceScore(stravaDistanceM: number, wahooDistanceM: number | null): number | null {
  const diff = distanceDiffM(stravaDistanceM, wahooDistanceM)
  if (diff == null) return null
  const ratio = diff / stravaDistanceM
  if (diff > DISTANCE_TOLERANCE_M && ratio > DISTANCE_TOLERANCE_RATIO) return null
  return ratio * 100
}

function durationDiffS(strava: RawStravaActivity, wahoo: WahooActivitySource): number | null {
  const stravaDurations = [positive(strava.movingTime), positive(strava.elapsedTime)].filter(
    (value): value is number => value != null,
  )
  const candidates = [positive(wahoo.movingTimeS), positive(wahoo.elapsedTimeS)].filter(
    (value): value is number => value != null,
  )
  if (stravaDurations.length === 0 || candidates.length === 0) return null
  return Math.min(
    ...candidates.flatMap(value => stravaDurations.map(duration => Math.abs(value - duration))),
  )
}

function durationScore(strava: RawStravaActivity, wahoo: WahooActivitySource): number | null {
  const diff = durationDiffS(strava, wahoo)
  if (diff == null) return null
  const tolerance = Math.max(DURATION_TOLERANCE_S, strava.elapsedTime * DURATION_TOLERANCE_RATIO)
  return diff > tolerance ? null : diff / 60
}

export function matchWahooActivity(
  strava: RawStravaActivity,
  sport: ActivityKind,
  cache: WahooData | null,
  fitPath?: string | null,
): WahooActivityMatch | null {
  if (!cache) return null
  const stravaStart = Date.parse(strava.startDate)
  if (!Number.isFinite(stravaStart)) return null
  let best: { activity: WahooActivitySource; score: number } | null = null
  for (const activity of Object.values(cache.activities)) {
    if (fitPath && (!('path' in activity.sourceFile) || activity.sourceFile.path !== fitPath))
      continue
    if (activity.sport != null && activity.sport !== sport) continue
    const wahooStart = Date.parse(activity.startDate)
    if (!Number.isFinite(wahooStart)) continue
    const startDiff = Math.abs(wahooStart - stravaStart)
    if (startDiff > START_TOLERANCE_MS) continue
    const stationary =
      (sport === 'strength' || sport === 'yoga' || sport === 'treatment' || sport === 'sauna') &&
      strava.distance === 0 &&
      (activity.distanceM == null || activity.distanceM === 0)
    const dScore = fitPath || stationary ? 0 : distanceScore(strava.distance, activity.distanceM)
    if (dScore == null) continue
    const tScore = durationScore(strava, activity)
    if (tScore == null) continue
    const score = startDiff / 60_000 + dScore + tScore
    if (!best || score < best.score) best = { activity, score }
  }
  if (!best) return null
  return {
    activity: best.activity,
    score: best.score,
    startDiffMs: Math.abs(Date.parse(best.activity.startDate) - stravaStart),
    distanceDiffM: distanceDiffM(strava.distance, best.activity.distanceM),
    durationDiffS: durationDiffS(strava, best.activity),
  }
}

function normalizedTitle(value: string | null | undefined): string {
  return (value ?? '').trim().replace(/\s+/g, ' ')
}

function startValue(activity: RawStravaActivity): string {
  return activity.startDateLocal || activity.startDate
}

export function selectWahooTitleUpdates(
  strava: WahooTitleStravaCache,
  wahoo: WahooCache,
  options: WahooTitleSyncOptions = {},
): WahooTitleUpdate[] {
  const unique = new Map<string, { activity: RawStravaActivity; match: WahooActivityMatch }>()
  const activities = Object.values(strava.activities)
    .filter(activity => !options.ids?.size || options.ids.has(String(activity.id)))
    .filter(activity => !options.since || startValue(activity).slice(0, 10) >= options.since)
    .sort((left, right) => startValue(left).localeCompare(startValue(right)))
  for (const activity of activities) {
    const kind = normalizeKind(activity.sportType)
    if (!kind) continue
    const match = matchWahooActivity(activity, kind, wahoo)
    if (!match) continue
    const previous = unique.get(match.activity.id)
    if (!previous || match.score < previous.match.score)
      unique.set(match.activity.id, { activity, match })
  }
  const updates: WahooTitleUpdate[] = []
  for (const { activity, match } of [...unique.values()].sort((left, right) =>
    startValue(left.activity).localeCompare(startValue(right.activity)),
  )) {
    if (!('summary' in match.activity)) continue
    if (match.activity.summary.edited && !options.includeEdited) continue
    const from = normalizedTitle(match.activity.name)
    const to = normalizedTitle(activity.name)
    if (!to || from === to) continue
    updates.push({
      stravaId: activity.id,
      wahooId: match.activity.id,
      wahooWorkoutId: match.activity.workoutId,
      from,
      to,
      startDate: activity.startDate,
      startDateLocal: activity.startDateLocal,
      score: match.score,
      startDiffS: Math.round(match.startDiffMs / 1000),
      distanceDiffM: match.distanceDiffM,
      durationDiffS: match.durationDiffS,
    })
  }
  return options.limit && options.limit > 0 ? updates.slice(0, options.limit) : updates
}

export function parseWahooCache(value: unknown): WahooCache {
  return parseWahooCacheValue(value, WAHOO_CACHE_VERSION, normalizeWahooSport)
}
