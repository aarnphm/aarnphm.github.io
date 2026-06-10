import type { RawStravaActivity, Sport } from './strava'

const START_TOLERANCE_MS = 20 * 60 * 1000
const DISTANCE_TOLERANCE_RATIO = 0.08
const DISTANCE_TOLERANCE_M = 1500
const DURATION_TOLERANCE_RATIO = 0.12
const DURATION_TOLERANCE_S = 10 * 60

export interface GarminFueling {
  caloriesConsumed: number | null
  carbsConsumedG: number | null
  fluidMl: number | null
  carbsRecommendedG: number | null
  fluidRecommendedMl: number | null
  sweatLossMl: number | null
  sourceDevice: string | null
}

export interface GarminMetrics {
  totalCalories: number | null
  metabolicCalories: number | null
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
}

export interface GarminActivity {
  id: string
  name: string | null
  sport: Sport | null
  startDate: string
  startDateLocal: string
  distanceM: number | null
  movingTimeS: number | null
  elapsedTimeS: number | null
  sourceDevice: string | null
  sourceFile: string | null
  metrics: GarminMetrics
  fueling: GarminFueling
}

export interface GarminStreams {
  latlng: [number, number][]
  altitude: number[]
  distance: number[]
  watts?: number[]
  heartrate?: number[]
  cadence?: number[]
}

export interface GarminActivityMatch {
  activity: GarminActivity
  score: number
  startDiffMs: number
  distanceDiffM: number | null
  durationDiffS: number | null
}

export interface GarminCache {
  version?: number
  lastSync: number
  activities: Record<string, GarminActivity>
  streams?: Record<string, GarminStreams>
}

export function emptyGarminMetrics(): GarminMetrics {
  return {
    totalCalories: null,
    metabolicCalories: null,
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
  }
}

export function emptyGarminFueling(sourceDevice: string | null = null): GarminFueling {
  return {
    caloriesConsumed: null,
    carbsConsumedG: null,
    fluidMl: null,
    carbsRecommendedG: null,
    fluidRecommendedMl: null,
    sweatLossMl: null,
    sourceDevice,
  }
}

export function hasGarminFueling(fueling: GarminFueling): boolean {
  return (
    fueling.caloriesConsumed != null ||
    fueling.carbsConsumedG != null ||
    fueling.fluidMl != null ||
    fueling.carbsRecommendedG != null ||
    fueling.fluidRecommendedMl != null ||
    fueling.sweatLossMl != null
  )
}

export function hasGarminMetrics(metrics: GarminMetrics): boolean {
  return Object.values(metrics).some(value => value != null)
}

export function normalizeGarminSport(value: string | null | undefined): Sport | null {
  if (!value) return null
  const sport = value.toLowerCase()
  if (sport.includes('swim')) return 'swim'
  if (sport.includes('bike') || sport.includes('cycling') || sport.includes('ride')) return 'bike'
  if (sport.includes('run')) return 'run'
  return null
}

function positive(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? value : null
}

function distanceDiffM(stravaDistanceM: number, garminDistanceM: number | null): number | null {
  const distance = positive(garminDistanceM)
  if (distance == null || stravaDistanceM <= 0) return 0
  return Math.abs(distance - stravaDistanceM)
}

function distanceScore(stravaDistanceM: number, garminDistanceM: number | null): number | null {
  const diff = distanceDiffM(stravaDistanceM, garminDistanceM)
  if (diff == null) return null
  if (diff === 0) return 0
  const ratio = diff / stravaDistanceM
  if (diff > DISTANCE_TOLERANCE_M && ratio > DISTANCE_TOLERANCE_RATIO) return null
  return ratio * 100
}

function durationDiffS(strava: RawStravaActivity, garmin: GarminActivity): number {
  const candidates = [positive(garmin.movingTimeS), positive(garmin.elapsedTimeS)].filter(
    (value): value is number => value != null,
  )
  if (candidates.length === 0) return 0
  return Math.min(
    ...candidates.flatMap(value => [
      Math.abs(value - strava.movingTime),
      Math.abs(value - strava.elapsedTime),
    ]),
  )
}

function durationScore(strava: RawStravaActivity, garmin: GarminActivity): number | null {
  const diff = durationDiffS(strava, garmin)
  const tolerance = Math.max(DURATION_TOLERANCE_S, strava.elapsedTime * DURATION_TOLERANCE_RATIO)
  if (diff > tolerance) return null
  return diff / 60
}

function withDevice(fueling: GarminFueling, sourceDevice: string | null): GarminFueling {
  return { ...fueling, sourceDevice: fueling.sourceDevice ?? sourceDevice }
}

export function matchGarminActivity(
  strava: RawStravaActivity,
  sport: Sport,
  cache: GarminCache | null,
): GarminActivityMatch | null {
  if (!cache) return null
  const stravaStart = Date.parse(strava.startDate)
  if (!Number.isFinite(stravaStart)) return null

  let best: { score: number; activity: GarminActivity } | null = null
  for (const activity of Object.values(cache.activities)) {
    if (activity.sport != null && activity.sport !== sport) continue
    const garminStart = Date.parse(activity.startDate)
    if (!Number.isFinite(garminStart)) continue
    const startDiff = Math.abs(garminStart - stravaStart)
    if (startDiff > START_TOLERANCE_MS) continue

    const dScore = distanceScore(strava.distance, activity.distanceM)
    if (dScore == null) continue
    const tScore = durationScore(strava, activity)
    if (tScore == null) continue

    const score = startDiff / 60_000 + dScore + tScore
    if (!best || score < best.score) best = { score, activity }
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

export function matchGarminFueling(
  strava: RawStravaActivity,
  sport: Sport,
  cache: GarminCache | null,
): GarminFueling | null {
  const match = matchGarminActivity(strava, sport, cache)
  if (!match || !hasGarminFueling(match.activity.fueling)) return null
  return withDevice(match.activity.fueling, match.activity.sourceDevice)
}
