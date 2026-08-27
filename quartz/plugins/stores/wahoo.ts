import type { ActivityKind, RawStravaActivity, Sport } from './strava'
import { isRecord } from '../../util/type-guards'

const START_TOLERANCE_MS = 20 * 60 * 1000
const DISTANCE_TOLERANCE_RATIO = 0.08
const DISTANCE_TOLERANCE_M = 1500
const DURATION_TOLERANCE_RATIO = 0.12
const DURATION_TOLERANCE_S = 10 * 60

export const WAHOO_CACHE_VERSION = 1

const BIKE_WORKOUT_TYPE_IDS = new Set([0, 11, 12, 13, 14, 15, 16, 17, 21, 49, 61, 64, 68, 70])
const RUN_WORKOUT_TYPE_IDS = new Set([1, 3, 4, 5, 19, 67, 71])
const SWIM_WORKOUT_TYPE_IDS = new Set([25, 26])

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
}

export interface WahooActivity {
  id: string
  workoutId: number
  workoutTypeId: number
  workoutUpdatedAt?: string | null
  name: string | null
  sport: Sport | null
  startDate: string
  startDateLocal: string
  distanceM: number | null
  movingTimeS: number | null
  elapsedTimeS: number | null
  sourceDevice: string | null
  sourceFile: WahooFitFile
  metrics: WahooMetrics
  summary: WahooSummary
}

export interface WahooCache {
  version: typeof WAHOO_CACHE_VERSION
  lastSync: number
  activities: Record<string, WahooActivity>
  streams: Record<string, WahooStreams>
}

export interface WahooActivityMatch {
  activity: WahooActivity
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

export function normalizeWahooSport(workoutTypeId: number, fitSport?: string | null): Sport | null {
  if (BIKE_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'bike'
  if (RUN_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'run'
  if (SWIM_WORKOUT_TYPE_IDS.has(workoutTypeId)) return 'swim'
  const normalized = fitSport?.toLowerCase() ?? ''
  if (normalized.includes('cycl') || normalized.includes('bike')) return 'bike'
  if (normalized.includes('run')) return 'run'
  if (normalized.includes('swim')) return 'swim'
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

function durationDiffS(strava: RawStravaActivity, wahoo: WahooActivity): number | null {
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

function durationScore(strava: RawStravaActivity, wahoo: WahooActivity): number | null {
  const diff = durationDiffS(strava, wahoo)
  if (diff == null) return null
  const tolerance = Math.max(DURATION_TOLERANCE_S, strava.elapsedTime * DURATION_TOLERANCE_RATIO)
  return diff > tolerance ? null : diff / 60
}

export function matchWahooActivity(
  strava: RawStravaActivity,
  sport: ActivityKind,
  cache: WahooCache | null,
): WahooActivityMatch | null {
  if (!cache) return null
  const stravaStart = Date.parse(strava.startDate)
  if (!Number.isFinite(stravaStart)) return null
  let best: { activity: WahooActivity; score: number } | null = null
  for (const activity of Object.values(cache.activities)) {
    if (activity.sport != null && activity.sport !== sport) continue
    const wahooStart = Date.parse(activity.startDate)
    if (!Number.isFinite(wahooStart)) continue
    const startDiff = Math.abs(wahooStart - stravaStart)
    if (startDiff > START_TOLERANCE_MS) continue
    const dScore = distanceScore(strava.distance, activity.distanceM)
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
    .filter(activity => {
      const sport = activity.sportType.toLowerCase()
      return sport.includes('ride') || sport.includes('cycling') || sport.includes('bike')
    })
    .filter(activity => !options.ids?.size || options.ids.has(String(activity.id)))
    .filter(activity => !options.since || startValue(activity).slice(0, 10) >= options.since)
    .sort((left, right) => startValue(left).localeCompare(startValue(right)))
  for (const activity of activities) {
    const match = matchWahooActivity(activity, 'bike', wahoo)
    if (!match) continue
    const previous = unique.get(match.activity.id)
    if (!previous || match.score < previous.match.score)
      unique.set(match.activity.id, { activity, match })
  }
  const updates: WahooTitleUpdate[] = []
  for (const { activity, match } of [...unique.values()].sort((left, right) =>
    startValue(left.activity).localeCompare(startValue(right.activity)),
  )) {
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

function finiteNumber(value: unknown, label: string, nullable = false): number | null {
  if (value == null && nullable) return null
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${label} must be finite`)
  return value
}

function integer(value: unknown, label: string): number {
  const parsed = finiteNumber(value, label)
  if (parsed == null || !Number.isInteger(parsed) || parsed < 0)
    throw new Error(`${label} must be a nonnegative integer`)
  return parsed
}

function stringValue(value: unknown, label: string, nullable = false): string | null {
  if (value == null && nullable) return null
  if (typeof value !== 'string') throw new Error(`${label} must be a string`)
  return value
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean`)
  return value
}

function dateValue(value: unknown, label: string): string {
  const parsed = stringValue(value, label)
  if (parsed == null || !Number.isFinite(Date.parse(parsed)))
    throw new Error(`${label} must be a date`)
  return parsed
}

function parseMetrics(value: unknown, label: string): WahooMetrics {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    totalCalories: finiteNumber(value.totalCalories, `${label}.totalCalories`, true),
    avgHeartRate: finiteNumber(value.avgHeartRate, `${label}.avgHeartRate`, true),
    maxHeartRate: finiteNumber(value.maxHeartRate, `${label}.maxHeartRate`, true),
    avgPower: finiteNumber(value.avgPower, `${label}.avgPower`, true),
    normalizedPower: finiteNumber(value.normalizedPower, `${label}.normalizedPower`, true),
    maxPower: finiteNumber(value.maxPower, `${label}.maxPower`, true),
    avgCadence: finiteNumber(value.avgCadence, `${label}.avgCadence`, true),
    totalAscentM: finiteNumber(value.totalAscentM, `${label}.totalAscentM`, true),
    totalDescentM: finiteNumber(value.totalDescentM, `${label}.totalDescentM`, true),
    totalWorkKJ: finiteNumber(value.totalWorkKJ, `${label}.totalWorkKJ`, true),
    trainingStressScore: finiteNumber(
      value.trainingStressScore,
      `${label}.trainingStressScore`,
      true,
    ),
    intensityFactor: finiteNumber(value.intensityFactor, `${label}.intensityFactor`, true),
    avgSpeedMps: finiteNumber(value.avgSpeedMps, `${label}.avgSpeedMps`, true),
    maxSpeedMps: finiteNumber(value.maxSpeedMps, `${label}.maxSpeedMps`, true),
    avgTemperatureC: finiteNumber(value.avgTemperatureC, `${label}.avgTemperatureC`, true),
  }
}

function parseSummary(value: unknown, label: string): WahooSummary {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    id: integer(value.id, `${label}.id`),
    name: stringValue(value.name, `${label}.name`, true),
    timeZone: stringValue(value.timeZone, `${label}.timeZone`, true),
    manual: booleanValue(value.manual, `${label}.manual`),
    edited: booleanValue(value.edited, `${label}.edited`),
    fitnessAppId: integer(value.fitnessAppId, `${label}.fitnessAppId`),
    durationPausedS: finiteNumber(value.durationPausedS, `${label}.durationPausedS`, true),
    createdAt: value.createdAt == null ? null : dateValue(value.createdAt, `${label}.createdAt`),
    updatedAt: value.updatedAt == null ? null : dateValue(value.updatedAt, `${label}.updatedAt`),
  }
}

function parseFitFile(value: unknown, label: string): WahooFitFile {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const url = stringValue(value.url, `${label}.url`)
  const sha256 = stringValue(value.sha256, `${label}.sha256`)
  const profileVersion = stringValue(value.profileVersion, `${label}.profileVersion`)
  if (url == null || sha256 == null || profileVersion == null)
    throw new Error(`${label} is incomplete`)
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error(`${label}.sha256 is invalid`)
  return {
    url,
    sha256,
    byteLength: integer(value.byteLength, `${label}.byteLength`),
    profileVersion,
  }
}

function parseActivity(value: unknown, label: string): WahooActivity {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const id = stringValue(value.id, `${label}.id`)
  const sport = value.sport == null ? null : value.sport
  if (id == null) throw new Error(`${label}.id is missing`)
  if (sport != null && sport !== 'bike' && sport !== 'run' && sport !== 'swim')
    throw new Error(`${label}.sport is invalid`)
  return {
    id,
    workoutId: integer(value.workoutId, `${label}.workoutId`),
    workoutTypeId: integer(value.workoutTypeId, `${label}.workoutTypeId`),
    workoutUpdatedAt:
      value.workoutUpdatedAt == null
        ? null
        : dateValue(value.workoutUpdatedAt, `${label}.workoutUpdatedAt`),
    name: stringValue(value.name, `${label}.name`, true),
    sport,
    startDate: dateValue(value.startDate, `${label}.startDate`),
    startDateLocal: dateValue(value.startDateLocal, `${label}.startDateLocal`),
    distanceM: finiteNumber(value.distanceM, `${label}.distanceM`, true),
    movingTimeS: finiteNumber(value.movingTimeS, `${label}.movingTimeS`, true),
    elapsedTimeS: finiteNumber(value.elapsedTimeS, `${label}.elapsedTimeS`, true),
    sourceDevice: stringValue(value.sourceDevice, `${label}.sourceDevice`, true),
    sourceFile: parseFitFile(value.sourceFile, `${label}.sourceFile`),
    metrics: parseMetrics(value.metrics, `${label}.metrics`),
    summary: parseSummary(value.summary, `${label}.summary`),
  }
}

function nullableNumberArray(value: unknown, label: string): (number | null)[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`)
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`, true))
}

function parseLatLngArray(value: unknown, label: string): ([number, number] | null)[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`)
  return value.map((item, index) => {
    if (item == null) return null
    if (!Array.isArray(item) || item.length !== 2) throw new Error(`${label}[${index}] is invalid`)
    const lat = finiteNumber(item[0], `${label}[${index}][0]`)
    const lng = finiteNumber(item[1], `${label}[${index}][1]`)
    if (lat == null || lng == null || lat < -90 || lat > 90 || lng < -180 || lng > 180)
      throw new Error(`${label}[${index}] is outside coordinate bounds`)
    return [lat, lng]
  })
}

function parseStreams(value: unknown, label: string): WahooStreams {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  if (!Array.isArray(value.timestamps) || !value.timestamps.every(item => typeof item === 'string'))
    throw new Error(`${label}.timestamps must be a string array`)
  if (!Array.isArray(value.time) || !value.time.every(item => typeof item === 'number'))
    throw new Error(`${label}.time must be a number array`)
  const streams: WahooStreams = {
    timestamps: [...value.timestamps],
    time: [...value.time],
    latlng: parseLatLngArray(value.latlng, `${label}.latlng`),
    altitude: nullableNumberArray(value.altitude, `${label}.altitude`),
    distance: nullableNumberArray(value.distance, `${label}.distance`),
    watts: nullableNumberArray(value.watts, `${label}.watts`),
    rightBalance: nullableNumberArray(value.rightBalance, `${label}.rightBalance`),
    heartrate: nullableNumberArray(value.heartrate, `${label}.heartrate`),
    cadence: nullableNumberArray(value.cadence, `${label}.cadence`),
    speed: nullableNumberArray(value.speed, `${label}.speed`),
    temperature: nullableNumberArray(value.temperature, `${label}.temperature`),
  }
  const lengths = Object.values(streams).map(stream => stream.length)
  if (lengths.some(length => length !== streams.time.length))
    throw new Error(`${label} arrays must have equal lengths`)
  return streams
}

function recordValue<T>(
  value: unknown,
  label: string,
  parse: (item: unknown, itemLabel: string) => T,
): Record<string, T> {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const entries: [string, T][] = []
  for (const [key, item] of Object.entries(value))
    entries.push([key, parse(item, `${label}.${key}`)])
  return Object.fromEntries(entries)
}

export function parseWahooCache(value: unknown): WahooCache {
  if (!isRecord(value)) throw new Error('Wahoo cache must be an object')
  const version = integer(value.version, 'Wahoo cache.version')
  if (version !== WAHOO_CACHE_VERSION)
    throw new Error(`Wahoo cache version ${version} is unsupported`)
  const activities = recordValue(value.activities, 'Wahoo cache.activities', parseActivity)
  const streams = recordValue(value.streams, 'Wahoo cache.streams', parseStreams)
  for (const [id, activity] of Object.entries(activities)) {
    if (activity.id !== id)
      throw new Error(`Wahoo cache activity key ${id} does not match activity id`)
    if (!streams[id]) throw new Error(`Wahoo cache activity ${id} is missing streams`)
  }
  return {
    version,
    lastSync: finiteNumber(value.lastSync, 'Wahoo cache.lastSync') ?? 0,
    activities,
    streams,
  }
}
