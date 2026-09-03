import type { DetailCtx } from '../../../util/triathlon-card'
import { isActivityKind, type StravaActivityDetail } from '../../../plugins/stores/strava'
import {
  isStravaDetailShardPath,
  STRAVA_DETAIL_INDEX_KIND,
  type StravaDetailIndex,
  type StravaDetailPayload,
  type StravaDetailShard,
} from '../../../util/strava-detail'
import { isTriathlonDailyAnalytics } from '../../../util/triathlon-day-analytics'
import { isRecord } from '../../../util/type-guards'

export type DetailPayload = StravaDetailPayload

const isDetailIndex = (value: unknown): value is StravaDetailIndex =>
  isRecord(value) &&
  value.kind === STRAVA_DETAIL_INDEX_KIND &&
  Array.isArray(value.shards) &&
  value.shards.every(isStravaDetailShardPath) &&
  isRecord(value.health) &&
  (value.dailyAnalytics === undefined || isTriathlonDailyAnalytics(value.dailyAnalytics))

const isStaminaTrace = (value: unknown): boolean => {
  if (value === null) return true
  if (!isRecord(value)) return false
  if (value.source === 'garmin')
    return (
      value.method === 'garmin-native' && value.ftpWatts === null && value.maxHeartRateBpm === null
    )
  return (
    value.source === 'garden-estimate' &&
    value.method === 'garden-stamina-v1' &&
    typeof value.ftpWatts === 'number' &&
    Number.isFinite(value.ftpWatts) &&
    value.ftpWatts > 0 &&
    typeof value.maxHeartRateBpm === 'number' &&
    Number.isFinite(value.maxHeartRateBpm) &&
    value.maxHeartRateBpm > 0
  )
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const nullableFinite = (value: unknown): value is number | null => value === null || finite(value)

const bounded = (value: unknown, minimum: number, maximum: number): boolean =>
  finite(value) && value >= minimum && value <= maximum

const nullableBounded = (value: unknown, minimum: number, maximum: number): boolean =>
  value === null || bounded(value, minimum, maximum)

const PRIVATE_ANALYSIS_KEYS = new Set([
  'description',
  'rawBlock',
  'activityUrl',
  'query',
  'accountId',
  'token',
  'routeFingerprint',
  'lat',
  'lng',
  'latitude',
  'longitude',
  'coordinates',
  'route',
])

const hasPrivateAnalysisData = (value: unknown): boolean => {
  if (Array.isArray(value)) return value.some(hasPrivateAnalysisData)
  if (!isRecord(value)) return false
  return Object.entries(value).some(
    ([key, child]) => PRIVATE_ANALYSIS_KEYS.has(key) || hasPrivateAnalysisData(child),
  )
}

const isProviderProvenance = (
  value: Record<string, unknown>,
  provider: 'pelotan' | 'mywindsock',
  activityId: number,
): boolean =>
  value.source === 'provider-native' &&
  value.provider === provider &&
  value.transport === 'strava-description' &&
  value.schemaVersion === 1 &&
  value.activityId === activityId &&
  finite(value.retrievedAt) &&
  value.retrievedAt >= 0

const UV_SEVERITIES = new Set(['negligible', 'low', 'moderate', 'high', 'serious', 'extreme'])

const isPelotanReport = (value: unknown, activityId: number): boolean => {
  if (value === null) return true
  return (
    isRecord(value) &&
    isProviderProvenance(value, 'pelotan', activityId) &&
    nullableBounded(value.score, 0, 100) &&
    (value.rawBand === null || typeof value.rawBand === 'string') &&
    (value.severity === null ||
      (typeof value.severity === 'string' && UV_SEVERITIES.has(value.severity))) &&
    nullableBounded(value.averageUvIndex, 0, 30) &&
    nullableBounded(value.averageTemperatureC, -90, 70) &&
    nullableBounded(value.averageCloudCoverPct, 0, 100)
  )
}

const isMyWindsockReport = (value: unknown, activityId: number): boolean => {
  if (value === null) return true
  if (!isRecord(value) || !isProviderProvenance(value, 'mywindsock', activityId)) return false
  return (
    nullableFinite(value.weatherImpactPct) &&
    nullableBounded(value.cdaM2, 0, 5) &&
    nullableFinite(value.feelsLikeElevationM) &&
    nullableBounded(value.headwindPct, 0, 100) &&
    nullableBounded(value.headwindMinKph, 0, 1_000) &&
    nullableBounded(value.headwindMaxKph, 0, 1_000) &&
    nullableBounded(value.longestHeadwindS, 0, Number.MAX_SAFE_INTEGER) &&
    nullableBounded(value.airSpeedKph, 0, 1_000) &&
    nullableBounded(value.averageTemperatureC, -90, 70) &&
    nullableBounded(value.precipitationProbabilityPct, 0, 100) &&
    nullableBounded(value.precipitationRateMmPerHour, 0, 1_000)
  )
}

const isGardenProvenance = (value: Record<string, unknown>, formulaId: string): boolean =>
  value.source === 'garden-estimate' &&
  value.formulaId === formulaId &&
  value.formulaVersion === 1 &&
  value.inputVersion === 'weatherkit-route-hour-v1+strava-stream-v1' &&
  value.normalizationVersion === 1 &&
  finite(value.computedAt) &&
  value.computedAt >= 0 &&
  finite(value.inputAsOf) &&
  value.inputAsOf >= 0 &&
  value.temporalSamplingModel === 'weatherkit-hourly-piecewise-constant' &&
  value.spatialSamplingModel === 'route-coordinate-nearest-hour-overlap-midpoint'

const isEnvironmentSample = (value: unknown): boolean =>
  isRecord(value) &&
  bounded(value.elapsedS, 0, Number.MAX_SAFE_INTEGER) &&
  bounded(value.distanceKm, 0, Number.MAX_SAFE_INTEGER) &&
  nullableBounded(value.uvIndex, 0, 30) &&
  nullableBounded(value.cumulativeSed, 0, Number.MAX_SAFE_INTEGER) &&
  nullableBounded(value.cumulativeMovingTelemetrySed, 0, Number.MAX_SAFE_INTEGER) &&
  nullableBounded(value.ambientTemperatureC, -90, 70) &&
  nullableBounded(value.cloudCoverPct, 0, 100) &&
  nullableBounded(value.headwindKph, -1_000, 1_000) &&
  nullableBounded(value.crosswindKph, -1_000, 1_000) &&
  nullableBounded(value.apparentAirSpeedKph, 0, 1_000) &&
  nullableBounded(value.yawDeg, -180, 180)

const isMonotonicSamples = (samples: readonly unknown[], elapsedTimeS: number): boolean => {
  let elapsed = -1
  let distance = -1
  let cumulativeSed = -1
  let cumulativeMovingTelemetrySed = -1
  for (const sample of samples) {
    if (!isEnvironmentSample(sample) || !isRecord(sample)) return false
    if (
      !finite(sample.elapsedS) ||
      !finite(sample.distanceKm) ||
      sample.elapsedS < elapsed ||
      sample.elapsedS > elapsedTimeS + 1 ||
      sample.distanceKm < distance
    )
      return false
    if (finite(sample.cumulativeSed)) {
      if (sample.cumulativeSed < cumulativeSed) return false
      cumulativeSed = sample.cumulativeSed
    }
    if (finite(sample.cumulativeMovingTelemetrySed)) {
      if (sample.cumulativeMovingTelemetrySed < cumulativeMovingTelemetrySed) return false
      cumulativeMovingTelemetrySed = sample.cumulativeMovingTelemetrySed
    }
    elapsed = sample.elapsedS
    distance = sample.distanceKm
  }
  return true
}

const isEnvironmentEstimate = (value: unknown, elapsedTimeS: number): boolean => {
  if (value === null) return true
  if (!isRecord(value) || !isGardenProvenance(value, 'garden-environment-v1')) return false
  const summary = value.summary
  const clocks = value.doseClocks
  const coverage = value.coverage
  if (!isRecord(summary) || !isRecord(clocks) || !isRecord(coverage)) return false
  return (
    nullableBounded(summary.averageUvIndex, 0, 30) &&
    nullableBounded(summary.peakUvIndex, 0, 30) &&
    nullableBounded(summary.uviHours, 0, Number.MAX_SAFE_INTEGER) &&
    nullableBounded(summary.ambientSed, 0, Number.MAX_SAFE_INTEGER) &&
    nullableBounded(summary.averageAmbientTemperatureC, -90, 70) &&
    nullableBounded(summary.averageCloudCoverPct, 0, 100) &&
    bounded(summary.daylightCoveragePct, 0, 100) &&
    bounded(summary.weatherCoveragePct, 0, 100) &&
    bounded(summary.coveredDurationS, 0, elapsedTimeS + 1) &&
    bounded(summary.elapsedDurationS, Math.max(0, elapsedTimeS - 1), elapsedTimeS + 1) &&
    nullableBounded(clocks.elapsedSed, 0, Number.MAX_SAFE_INTEGER) &&
    nullableBounded(clocks.movingTelemetrySed, 0, Number.MAX_SAFE_INTEGER) &&
    ['weatherPct', 'uvPct', 'temperaturePct', 'cloudPct', 'daylightPct'].every(key =>
      bounded(coverage[key], 0, 100),
    ) &&
    Array.isArray(value.samples) &&
    value.samples.length <= 320 &&
    isMonotonicSamples(value.samples, elapsedTimeS) &&
    (value.attribution === null ||
      (isRecord(value.attribution) &&
        typeof value.attribution.serviceName === 'string' &&
        typeof value.attribution.logoLightUrl === 'string' &&
        typeof value.attribution.logoDarkUrl === 'string' &&
        typeof value.attribution.legalPageUrl === 'string'))
  )
}

const isGardenUvScore = (value: unknown): boolean => {
  if (value === null) return true
  return (
    isRecord(value) &&
    isGardenProvenance(value, 'garden-uv-score-v1') &&
    Number.isInteger(value.score) &&
    bounded(value.score, 0, 100) &&
    typeof value.severity === 'string' &&
    UV_SEVERITIES.has(value.severity) &&
    (value.doseClock === 'elapsed' || value.doseClock === 'moving-telemetry') &&
    bounded(value.doseSed, 0, Number.MAX_SAFE_INTEGER) &&
    bounded(value.coefficientSed, Number.MIN_VALUE, Number.MAX_SAFE_INTEGER) &&
    value.calibrationVersion === 1
  )
}

const isGardenWind = (value: unknown): boolean => {
  if (value === null) return true
  if (!isRecord(value) || !isGardenProvenance(value, 'garden-apparent-wind-v1')) return false
  const summary = value.summary
  const coverage = value.coverage
  if (!isRecord(summary) || !isRecord(coverage)) return false
  return (
    bounded(summary.headwindSharePct, 0, 100) &&
    bounded(summary.headwindTimeS, 0, Number.MAX_SAFE_INTEGER) &&
    bounded(summary.tailwindTimeS, 0, Number.MAX_SAFE_INTEGER) &&
    bounded(summary.longestHeadwindS, 0, Number.MAX_SAFE_INTEGER) &&
    bounded(summary.averageHeadwindKph, -1_000, 1_000) &&
    bounded(summary.averageCrosswindKph, -1_000, 1_000) &&
    bounded(summary.maximumHeadwindKph, 0, 1_000) &&
    bounded(summary.maximumCrosswindKph, 0, 1_000) &&
    bounded(summary.averageGroundSpeedKph, 0, 1_000) &&
    bounded(summary.averageApparentAirSpeedKph, 0, 1_000) &&
    bounded(summary.apparentAirRatio, 0, 100) &&
    bounded(summary.averageYawDeg, -180, 180) &&
    bounded(summary.coveragePct, 0, 100) &&
    bounded(coverage.windPct, 0, 100)
  )
}

const isActivityAnalyses = (value: unknown, activityId: number, elapsedTimeS: number): boolean => {
  if (!isRecord(value) || hasPrivateAnalysisData(value)) return false
  const native = value.native
  const derived = value.derived
  if (!isRecord(native) || !isRecord(derived)) return false
  return (
    isPelotanReport(native.pelotan, activityId) &&
    isMyWindsockReport(native.myWindsock, activityId) &&
    isEnvironmentEstimate(derived.environment, elapsedTimeS) &&
    isGardenUvScore(derived.uvScore) &&
    isGardenWind(derived.apparentWind)
  )
}

export const isActivityDetail = (value: unknown): value is StravaActivityDetail => {
  if (
    !isRecord(value) ||
    typeof value.id !== 'number' ||
    !/^\d{4}-\d{2}-\d{2}$/.test(typeof value.date === 'string' ? value.date : '') ||
    !isActivityKind(value.sport) ||
    !isStaminaTrace(value.staminaTrace) ||
    !finite(value.elapsedTimeS) ||
    value.elapsedTimeS < 0 ||
    value.elapsedTimeS > Number.MAX_SAFE_INTEGER ||
    !nullableBounded(value.deviceTemperatureC, -90, 100) ||
    !nullableBounded(value.ambientTemperatureC, -90, 70)
  )
    return false
  return isActivityAnalyses(value.analyses, value.id, value.elapsedTimeS)
}

const isDetailShard = (value: unknown): value is StravaDetailShard =>
  isRecord(value) &&
  isRecord(value.details) &&
  Object.entries(value.details).every(
    ([id, detail]) => /^\d+$/.test(id) && isActivityDetail(detail) && String(detail.id) === id,
  )

export async function readDetailPayload(
  response: Response,
  signal: AbortSignal,
): Promise<DetailPayload> {
  const value: unknown = await response.json()
  if (!isDetailIndex(value)) throw new Error('invalid Strava detail index')
  const shardDetails = await Promise.all(
    value.shards.map(async path => {
      const shardResponse = await fetch(new URL(path, response.url), { signal })
      if (!shardResponse.ok) throw new Error(`${path} returned ${shardResponse.status}`)
      const shard: unknown = await shardResponse.json()
      if (!isDetailShard(shard)) throw new Error(`${path} is not a valid Strava detail shard`)
      return shard.details
    }),
  )
  const details: Record<string, StravaActivityDetail> = {}
  for (const shard of shardDetails)
    for (const [id, detail] of Object.entries(shard)) {
      if (details[id]) throw new Error(`duplicate Strava detail ${id}`)
      details[id] = detail
    }
  return {
    details,
    swimTrend: value.swimTrend,
    health: value.health,
    dailyAnalytics: value.dailyAnalytics,
    zones: value.zones,
    powerCurveRef: value.powerCurveRef,
    powerCurveYearRef: value.powerCurveYearRef,
    powerCurveYear: value.powerCurveYear,
    criticalPower: value.criticalPower,
    criticalPowerYear: value.criticalPowerYear,
    ftp: value.ftp,
    goalFtp: value.goalFtp,
    vt1Hr: value.vt1Hr,
    matchedRuns: value.matchedRuns,
    matchedRides: value.matchedRides,
  }
}

export const detailContextFromPayload = (payload?: DetailPayload | null): DetailCtx => ({
  zones: payload?.zones ?? null,
  curveRef: payload?.powerCurveRef ?? [],
  curveYearRef: payload?.powerCurveYearRef ?? [],
  curveYear: payload?.powerCurveYear ?? null,
  criticalPower: payload?.criticalPower ?? null,
  criticalPowerYear: payload?.criticalPowerYear ?? null,
  ftp: payload?.ftp ?? null,
  goalFtp: payload?.goalFtp ?? null,
  vt1: payload?.vt1Hr ?? null,
})
