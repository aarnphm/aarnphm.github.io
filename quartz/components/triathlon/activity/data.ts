import type { DetailCtx } from '../../../util/triathlon-card'
import {
  GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD,
  isActivityDevice,
  isActivityKind,
  type StravaActivityDetail,
} from '../../../plugins/stores/strava'
import { emptyWahooMetrics } from '../../../plugins/stores/wahoo'
import { CYCLING_POWER_MAX_POINTS } from '../../../util/cycling-power'
import { GARDEN_CYCLING_STAMINA_METHOD } from '../../../util/cycling-stamina'
import { HEART_RATE_PHYSIOLOGY_METHOD } from '../../../util/heart-rate-physiology'
import { isMyWindsockArchiveReference } from '../../../util/mywindsock-graphs'
import {
  isStravaDetailShardPath,
  STRAVA_DETAIL_INDEX_KIND,
  type StravaDetailIndex,
  type StravaDetailPayload,
  type StravaDetailShard,
} from '../../../util/strava-detail'
import { parsePublicSurfaceCurrentEstimate } from '../../../util/surface-current'
import { SWIM_PHYSIOLOGY_METHOD } from '../../../util/swim-physiology'
import { SWIM_POWER_REFERENCE_PACE_S_PER_100M } from '../../../util/swim-power'
import { isTriathlonDailyAnalytics } from '../../../util/triathlon-day-analytics'
import { isRecord } from '../../../util/type-guards'
import { WALK_POWER_METHOD, WALK_POWER_WINDOW_SECONDS } from '../../../util/walk-power'

const isWalkPower = (value: unknown, elapsedTimeS: number, date: unknown): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garden-estimate' ||
    (value.inputSource !== 'strava' && value.inputSource !== 'garmin') ||
    value.method !== WALK_POWER_METHOD ||
    value.kind !== 'net-metabolic' ||
    value.windowSeconds !== WALK_POWER_WINDOW_SECONDS ||
    !isRecord(value.weight) ||
    !finite(value.weight.kg) ||
    value.weight.kg <= 0 ||
    value.weight.date !== date ||
    value.weight.source !== 'garmin' ||
    !finite(value.averageWatts) ||
    value.averageWatts < 0 ||
    !Array.isArray(value.points) ||
    value.points.length < 3
  )
    return false
  let previous = -1
  let distance = 0
  let usable = 0
  const valid = value.points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previous ||
      !finite(point.distanceKm) ||
      point.distanceKm < distance ||
      !(point.watts === null || (finite(point.watts) && point.watts >= 0))
    )
      return false
    previous = point.elapsedS
    distance = point.distanceKm
    if (point.watts != null) usable++
    return true
  })
  return valid && usable >= 2
}

const isHeartRatePhysiology = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garden-estimate' ||
    value.method !== HEART_RATE_PHYSIOLOGY_METHOD ||
    !bounded(value.maxHeartRateBpm, 100, 240) ||
    !bounded(value.baselineHeartRateBpm, 35, 240) ||
    value.windowSeconds !== 60 ||
    !Array.isArray(value.points) ||
    value.points.length < 2
  )
    return false
  return isPhysiologyPoints(value.points, elapsedTimeS)
}

const isPhysiologyPoints = (points: unknown, elapsedTimeS: number): boolean => {
  if (!Array.isArray(points) || points.length < 2) return false
  let previous = -1
  let distance = 0
  return points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previous ||
      !finite(point.distanceKm) ||
      point.distanceKm < distance ||
      !nullableBounded(point.stamina, 0, 100) ||
      !nullableBounded(point.potentialStamina, 0, 100) ||
      !nullableBounded(point.performanceCondition, -20, 20)
    )
      return false
    previous = point.elapsedS
    distance = point.distanceKm
    return true
  })
}

const isSwimPhysiology = (value: unknown, elapsedTimeS: number): boolean =>
  value == null ||
  (isRecord(value) &&
    value.source === 'garden-estimate' &&
    value.method === SWIM_PHYSIOLOGY_METHOD &&
    value.exertionSource === 'heart-rate' &&
    (value.speedBasis === 'ground-speed' || value.speedBasis === 'pool-length') &&
    (value.strokeRateSource === 'stream' ||
      value.strokeRateSource === 'stream-with-average' ||
      value.strokeRateSource === 'activity-average' ||
      value.strokeRateSource === 'unavailable') &&
    bounded(value.maxHeartRateBpm, 100, 240) &&
    bounded(value.baselineHeartRateBpm, 35, 240) &&
    bounded(value.baselineSpeedMps, 100 / 360, 100 / 45) &&
    nullableBounded(value.baselineStrokesPerM, 0.001, 10) &&
    value.baselineSeconds === 180 &&
    value.windowSeconds === 60 &&
    bounded(value.coverage, 0, 1) &&
    isPhysiologyPoints(value.points, elapsedTimeS))

const isSwimPower = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garden-estimate' ||
    !(
      (value.speedBasis === 'pool-length' &&
        value.method === 'freestyle-drag-index-v1' &&
        (value.inputSource === 'garmin' || value.inputSource === 'apple')) ||
      (value.speedBasis === 'ground-speed' &&
        value.method === 'open-water-drag-index-v1' &&
        value.inputSource === 'route')
    ) ||
    value.referencePaceSPer100m !== SWIM_POWER_REFERENCE_PACE_S_PER_100M ||
    !bounded(value.averageIndex, 0.001, 4000) ||
    !finite(value.activeTimeS) ||
    value.activeTimeS <= 0 ||
    !finite(value.distanceM) ||
    value.distanceM <= 0 ||
    !finite(value.validIntervalCount) ||
    !Number.isSafeInteger(value.validIntervalCount) ||
    value.validIntervalCount <= 0 ||
    !finite(value.excludedIntervalCount) ||
    !Number.isSafeInteger(value.excludedIntervalCount) ||
    value.excludedIntervalCount < 0 ||
    !Array.isArray(value.curve) ||
    !Array.isArray(value.histogramS) ||
    value.histogramS.length === 0 ||
    value.histogramS.length > 160
  )
    return false
  let histogramTimeS = 0
  for (const seconds of value.histogramS) {
    if (!finite(seconds) || seconds < 0) return false
    histogramTimeS += seconds
  }
  if (Math.abs(histogramTimeS - value.activeTimeS) > 0.5) return false
  let previous = 0
  return value.curve.every((p: unknown) => {
    if (
      !isRecord(p) ||
      !bounded(p.durationS, 60, elapsedTimeS + 1) ||
      p.durationS <= previous ||
      !bounded(p.index, 0.001, 4000) ||
      !bounded(p.startElapsedS, 0, elapsedTimeS + 1) ||
      !bounded(p.endElapsedS, p.startElapsedS, elapsedTimeS + 1)
    )
      return false
    previous = p.durationS
    return true
  })
}

const isWahooVerification = (value: unknown): boolean => {
  if (value === undefined) return true
  if (!isRecord(value) || !isRecord(value.metrics) || !isRecord(value.summarySources)) return false
  const metrics = value.metrics
  return (
    typeof value.activityId === 'string' &&
    (value.fitPath === null ||
      (typeof value.fitPath === 'string' &&
        value.fitPath.startsWith('triathlon/wahoo/') &&
        value.fitPath.toLowerCase().endsWith('.fit'))) &&
    typeof value.sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test(value.sha256) &&
    (value.sourceDevice === null || typeof value.sourceDevice === 'string') &&
    finite(value.startOffsetS) &&
    (value.distanceM === null || finite(value.distanceM)) &&
    [null, 'strava', 'garmin'].includes(value.streamFallback as string | null) &&
    Object.keys(emptyWahooMetrics()).every(key => metrics[key] === null || finite(metrics[key])) &&
    Object.values(value.summarySources).every(source => source === 'wahoo')
  )
}

export type DetailPayload = StravaDetailPayload

const isActivitySources = (value: unknown): boolean =>
  value === undefined ||
  (Array.isArray(value) &&
    value.every(
      source =>
        isRecord(source) &&
        (source.provider === 'strava' ||
          source.provider === 'garmin' ||
          source.provider === 'wahoo') &&
        typeof source.activityId === 'string' &&
        (source.name === null || typeof source.name === 'string') &&
        (source.fileName === null || typeof source.fileName === 'string'),
    ))

const isActivityMoves = (value: unknown): boolean =>
  value === undefined ||
  (isRecord(value) &&
    value.source === 'manual' &&
    Array.isArray(value.entries) &&
    value.entries.length > 0 &&
    value.entries.every(
      move =>
        isRecord(move) &&
        typeof move.name === 'string' &&
        move.name.trim().length > 0 &&
        Array.isArray(move.sets) &&
        move.sets.length > 0 &&
        move.sets.every(
          set =>
            isRecord(set) &&
            typeof set.repetitions === 'number' &&
            Number.isSafeInteger(set.repetitions) &&
            set.repetitions > 0 &&
            typeof set.perSide === 'boolean',
        ),
    ))

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
    value.method === GARDEN_CYCLING_STAMINA_METHOD &&
    typeof value.ftpWatts === 'number' &&
    Number.isFinite(value.ftpWatts) &&
    value.ftpWatts > 0 &&
    typeof value.maxHeartRateBpm === 'number' &&
    Number.isFinite(value.maxHeartRateBpm) &&
    value.maxHeartRateBpm > 0
  )
}

const isPerformanceConditionTrace = (value: unknown): boolean => {
  if (!isRecord(value)) return false
  if (value.source === 'garmin') return value.method === 'garmin-native'
  return (
    value.source === 'garden-estimate' &&
    value.method === GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD &&
    finite(value.ftpWatts) &&
    value.ftpWatts > 0 &&
    finite(value.lactateThresholdHeartRateBpm) &&
    finite(value.restingHeartRateBpm) &&
    value.restingHeartRateBpm > 0 &&
    value.lactateThresholdHeartRateBpm > value.restingHeartRateBpm &&
    value.windowSeconds === 360
  )
}

const isCyclingTorqueTrace = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'wahoo' ||
    value.method !== 'power-cadence-v1' ||
    !isRecord(value.summary) ||
    value.summary.source !== 'wahoo' ||
    value.summary.method !== 'power-cadence-v1' ||
    !bounded(value.summary.coverage, 0, 1) ||
    !bounded(value.summary.observedSeconds, 0, elapsedTimeS) ||
    !finite(value.summary.averageTorqueNm) ||
    value.summary.averageTorqueNm < 0 ||
    !finite(value.summary.averageCadenceRpm) ||
    value.summary.averageCadenceRpm <= 0 ||
    !Array.isArray(value.points) ||
    value.points.length < 2 ||
    !Array.isArray(value.cells)
  )
    return false
  let previous = -1
  let distance = 0
  return (
    value.points.every((point: unknown) => {
      if (
        !isRecord(point) ||
        !bounded(point.elapsedS, 0, elapsedTimeS) ||
        point.elapsedS <= previous ||
        !finite(point.distanceKm) ||
        point.distanceKm < distance ||
        !finite(point.durationS) ||
        point.durationS <= 0 ||
        !nullableBounded(point.torqueNm, 0, Number.MAX_VALUE) ||
        !nullableBounded(point.cadenceRpm, 0, Number.MAX_VALUE) ||
        !nullableBounded(point.watts, 0, Number.MAX_VALUE)
      )
        return false
      previous = point.elapsedS
      distance = point.distanceKm
      return true
    }) &&
    value.cells.every(
      (cell: unknown) =>
        isRecord(cell) &&
        finite(cell.cadenceRpm) &&
        cell.cadenceRpm >= 0 &&
        finite(cell.torqueNm) &&
        cell.torqueNm >= 0 &&
        finite(cell.seconds) &&
        cell.seconds > 0,
    )
  )
}

const isCyclingIntensityTrace = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'wahoo' ||
    value.method !== 'cumulative-power-30s-v1' ||
    !(finite(value.ftpWatts) && value.ftpWatts > 0) ||
    !(value.ftpSource === 'wahoo-summary' || value.ftpSource === 'athlete') ||
    !Array.isArray(value.points) ||
    value.points.length < 2 ||
    value.points.length > 322
  )
    return false
  let previousElapsed = -1
  let previousDistance = 0
  return value.points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previousElapsed ||
      !finite(point.distanceKm) ||
      point.distanceKm < previousDistance ||
      !nullableBounded(point.intensityFactor, 0, Number.MAX_VALUE)
    )
      return false
    previousElapsed = point.elapsedS
    previousDistance = point.distanceKm
    return true
  })
}

const isCyclingPowerTrace = (value: unknown, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    (value.source !== 'wahoo' && value.source !== 'strava') ||
    (value.terrainSource !== 'wahoo' &&
      value.terrainSource !== 'strava' &&
      value.terrainSource !== 'garmin') ||
    value.method !== 'recorded-power-average-v1' ||
    !Array.isArray(value.points) ||
    value.points.length < 2 ||
    value.points.length > CYCLING_POWER_MAX_POINTS
  )
    return false
  let previousElapsed = -1
  let previousDistance = 0
  return value.points.every((point: unknown) => {
    if (
      !isRecord(point) ||
      !bounded(point.elapsedS, 0, elapsedTimeS) ||
      point.elapsedS <= previousElapsed ||
      !finite(point.distanceKm) ||
      point.distanceKm < previousDistance ||
      !nullableFinite(point.elevationM) ||
      !nullableBounded(point.power30sWatts, 0, Number.MAX_VALUE) ||
      !nullableBounded(point.power5mWatts, 0, Number.MAX_VALUE) ||
      !nullableBounded(point.cumulativePowerWatts, 0, Number.MAX_VALUE)
    )
      return false
    previousElapsed = point.elapsedS
    previousDistance = point.distanceKm
    return true
  })
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const nullableFinite = (value: unknown): value is number | null => value === null || finite(value)

const bounded = (value: unknown, minimum: number, maximum: number): value is number =>
  finite(value) && value >= minimum && value <= maximum

const nullableBounded = (value: unknown, minimum: number, maximum: number): boolean =>
  value === null || bounded(value, minimum, maximum)

const optionalNullableBounded = (value: unknown, minimum: number, maximum: number): boolean =>
  value === undefined || nullableBounded(value, minimum, maximum)

const isThermalSource = (value: unknown): boolean =>
  value === null || value === 'core-app' || value === 'core-fit'

const isThermalPoint = (value: unknown): boolean =>
  isRecord(value) &&
  nullableBounded(value.heatStrainIndex, 0, 20) &&
  isThermalSource(value.heatStrainSource) &&
  (value.heatStrainIndex == null) === (value.heatStrainSource === null) &&
  nullableBounded(value.coreTemperatureC, 25, 45) &&
  isThermalSource(value.coreTemperatureSource) &&
  (value.coreTemperatureC == null) === (value.coreTemperatureSource === null) &&
  nullableBounded(value.skinTemperatureC, 0, 50) &&
  isThermalSource(value.skinTemperatureSource) &&
  (value.skinTemperatureC == null) === (value.skinTemperatureSource === null)

const isThermalTrace = (value: unknown): boolean =>
  Array.isArray(value) && value.every(isThermalPoint)

const isRouteMetricPoint = (value: unknown): boolean =>
  isRecord(value) &&
  optionalNullableBounded(value.performanceCondition, -20, 20) &&
  optionalNullableBounded(value.strideLengthM, 0.2, 3) &&
  optionalNullableBounded(value.verticalRatioPct, 0, 50) &&
  optionalNullableBounded(value.verticalOscillationCm, 1, 30) &&
  optionalNullableBounded(value.groundContactBalanceLeftPct, 0, 100) &&
  optionalNullableBounded(value.groundContactTimeMs, 50, 1_000) &&
  optionalNullableBounded(value.stepSpeedLossMps, 0, 5) &&
  optionalNullableBounded(value.stepSpeedLossPct, 0, 100) &&
  optionalNullableBounded(value.impactLoadFactor, 0, 10)

const isRouteTrace = (value: unknown): boolean =>
  Array.isArray(value) && value.every(point => isThermalPoint(point) && isRouteMetricPoint(point))

const isRunWalk = (value: unknown): boolean => {
  if (value === null) return true
  if (
    !isRecord(value) ||
    value.source !== 'garmin' ||
    !bounded(value.elapsedTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !bounded(value.runTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !bounded(value.walkTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !bounded(value.idleTimeS, 0, Number.MAX_SAFE_INTEGER) ||
    !Array.isArray(value.segments) ||
    value.segments.length === 0
  )
    return false
  let elapsedTimeS = 0
  let runTimeS = 0
  let walkTimeS = 0
  let idleTimeS = 0
  for (const segment of value.segments) {
    if (
      !isRecord(segment) ||
      (segment.state !== 'run' && segment.state !== 'walk' && segment.state !== 'idle') ||
      !finite(segment.startElapsedS) ||
      !finite(segment.endElapsedS) ||
      Math.abs(segment.startElapsedS - elapsedTimeS) > 0.002 ||
      segment.endElapsedS <= segment.startElapsedS
    )
      return false
    const durationS = segment.endElapsedS - segment.startElapsedS
    elapsedTimeS = segment.endElapsedS
    if (segment.state === 'run') runTimeS += durationS
    else if (segment.state === 'walk') walkTimeS += durationS
    else idleTimeS += durationS
  }
  return (
    Math.abs(value.elapsedTimeS - elapsedTimeS) <= 0.002 &&
    Math.abs(value.runTimeS - runTimeS) <= 0.01 &&
    Math.abs(value.walkTimeS - walkTimeS) <= 0.01 &&
    Math.abs(value.idleTimeS - idleTimeS) <= 0.01
  )
}

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

const ROUTE_SAMPLE_KEYS = new Set([
  'elapsedS',
  'distanceKm',
  'providerHeadwindKph',
  'providerCrosswindKph',
  'weatherCostW',
  'movingAirPenaltyKm',
])

const isMyWindsockRoute = (value: unknown, activityId: number, elapsedTimeS: number): boolean => {
  if (value == null) return true
  if (
    !isRecord(value) ||
    value.source !== 'provider-native' ||
    value.transport !== 'browser-runtime' ||
    value.provider !== 'mywindsock' ||
    value.activityId !== activityId ||
    typeof value.capturedAt !== 'string' ||
    !nullableBounded(value.referenceWatts, 0, 10_000) ||
    !Array.isArray(value.relativeWindPct) ||
    value.relativeWindPct.length !== 8 ||
    !value.relativeWindPct.every(share => bounded(share, 0, 100)) ||
    !Array.isArray(value.samples) ||
    value.samples.length < 2 ||
    value.samples.length > 320
  )
    return false
  let elapsed = -1
  return value.samples.every((sample: unknown) => {
    if (
      !isRecord(sample) ||
      !Object.keys(sample).every(key => ROUTE_SAMPLE_KEYS.has(key)) ||
      !bounded(sample.elapsedS, 0, elapsedTimeS + 120) ||
      sample.elapsedS <= elapsed ||
      !bounded(sample.distanceKm, 0, 10_000) ||
      !nullableBounded(sample.providerHeadwindKph, -1_000, 1_000) ||
      !nullableBounded(sample.providerCrosswindKph, -1_000, 1_000) ||
      !nullableBounded(sample.weatherCostW, -10_000, 10_000) ||
      !nullableBounded(sample.movingAirPenaltyKm, -10_000, 10_000)
    )
      return false
    elapsed = sample.elapsedS
    return true
  })
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
  (!('windSpeedKph' in value) || nullableBounded(value.windSpeedKph, 0, 1_000)) &&
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

const isEnvironmentEstimate = (
  value: unknown,
  activityId: number,
  elapsedTimeS: number,
  activityStart: unknown,
): boolean => {
  if (value === null) return true
  if (!isRecord(value) || !isGardenProvenance(value, 'garden-environment-v1')) return false
  const summary = value.summary
  const clocks = value.doseClocks
  const coverage = value.coverage
  if (!isRecord(summary) || !isRecord(clocks) || !isRecord(coverage)) return false
  if (value.surfaceCurrent != null) {
    const current = parsePublicSurfaceCurrentEstimate(value.surfaceCurrent)
    if (
      !current ||
      current.activityId !== activityId ||
      current.summary.elapsedDurationS !== elapsedTimeS
    )
      return false
    if (
      activityStart !== undefined &&
      (typeof activityStart !== 'string' ||
        Date.parse(activityStart) !== Date.parse(current.start) ||
        Date.parse(current.end) - Date.parse(activityStart) !== elapsedTimeS * 1_000)
    )
      return false
  }
  return (
    nullableBounded(summary.averageUvIndex, 0, 30) &&
    nullableBounded(summary.peakUvIndex, 0, 30) &&
    nullableBounded(summary.uviHours, 0, Number.MAX_SAFE_INTEGER) &&
    nullableBounded(summary.ambientSed, 0, Number.MAX_SAFE_INTEGER) &&
    nullableBounded(summary.averageAmbientTemperatureC, -90, 70) &&
    nullableBounded(summary.averageCloudCoverPct, 0, 100) &&
    (!('averageWindSpeedKph' in summary) ||
      nullableBounded(summary.averageWindSpeedKph, 0, 1_000)) &&
    bounded(summary.daylightCoveragePct, 0, 100) &&
    bounded(summary.weatherCoveragePct, 0, 100) &&
    bounded(summary.coveredDurationS, 0, elapsedTimeS + 1) &&
    bounded(summary.elapsedDurationS, Math.max(0, elapsedTimeS - 1), elapsedTimeS + 1) &&
    nullableBounded(clocks.elapsedSed, 0, Number.MAX_SAFE_INTEGER) &&
    nullableBounded(clocks.movingTelemetrySed, 0, Number.MAX_SAFE_INTEGER) &&
    ['weatherPct', 'uvPct', 'temperaturePct', 'cloudPct', 'daylightPct'].every(key =>
      bounded(coverage[key], 0, 100),
    ) &&
    (!('windPct' in coverage) || bounded(coverage.windPct, 0, 100)) &&
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
  if (!isRecord(value) || !isGardenProvenance(value, 'garden-apparent-wind-v3')) return false
  const summary = value.summary
  const coverage = value.coverage
  if (!isRecord(summary) || !isRecord(coverage)) return false
  return (
    bounded(summary.headwindSharePct, 0, 100) &&
    bounded(summary.headwindTimeS, 0, Number.MAX_SAFE_INTEGER) &&
    bounded(summary.tailwindTimeS, 0, Number.MAX_SAFE_INTEGER) &&
    bounded(summary.longestHeadwindS, 0, Number.MAX_SAFE_INTEGER) &&
    bounded(summary.averageHeadwindWhileIntoKph, 0, 1_000) &&
    bounded(summary.averageCrosswindMagnitudeKph, 0, 1_000) &&
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

const isActivityAnalyses = (
  value: unknown,
  activityId: number,
  elapsedTimeS: number,
  activityStart: unknown,
): boolean => {
  if (!isRecord(value) || hasPrivateAnalysisData(value)) return false
  const native = value.native
  const derived = value.derived
  if (!isRecord(native) || !isRecord(derived)) return false
  return (
    isPelotanReport(native.pelotan, activityId) &&
    isMyWindsockReport(native.myWindsock, activityId) &&
    isMyWindsockRoute(native.myWindsockRoute, activityId, elapsedTimeS) &&
    (native.myWindsockArchive == null ||
      isMyWindsockArchiveReference(native.myWindsockArchive, activityId)) &&
    isEnvironmentEstimate(derived.environment, activityId, elapsedTimeS, activityStart) &&
    isGardenUvScore(derived.uvScore) &&
    isGardenWind(derived.apparentWind)
  )
}

const isRunBestEfforts = (value: unknown): boolean => {
  if (value == null) return true
  return (
    isRecord(value) &&
    value.distanceSource === 'calculated' &&
    value.weightKg === null &&
    value.weightDate === null &&
    Array.isArray(value.power) &&
    value.power.length === 0 &&
    Array.isArray(value.climbs) &&
    value.climbs.length === 0 &&
    Array.isArray(value.distance) &&
    value.distance.every(
      (effort: unknown) =>
        isRecord(effort) &&
        typeof effort.label === 'string' &&
        finite(effort.targetDistanceM) &&
        effort.targetDistanceM > 0 &&
        finite(effort.elapsedTimeS) &&
        effort.elapsedTimeS > 0 &&
        finite(effort.averageSpeedKph) &&
        effort.averageSpeedKph > 0 &&
        (effort.averageHeartRate === null ||
          (finite(effort.averageHeartRate) && effort.averageHeartRate > 0)) &&
        nullableFinite(effort.elevationDeltaM),
    )
  )
}

export const isActivityDetail = (value: unknown): value is StravaActivityDetail => {
  if (
    !isRecord(value) ||
    typeof value.id !== 'number' ||
    !/^\d{4}-\d{2}-\d{2}$/.test(typeof value.date === 'string' ? value.date : '') ||
    !isActivityKind(value.sport) ||
    (value.sport === 'run' && !isRunBestEfforts(value.bestEfforts)) ||
    !isWahooVerification(value.wahoo) ||
    !isActivitySources(value.sources) ||
    !(
      value.powerCurveWeight === undefined ||
      (isRecord(value.powerCurveWeight) &&
        finite(value.powerCurveWeight.kg) &&
        value.powerCurveWeight.kg > 0 &&
        typeof value.powerCurveWeight.date === 'string' &&
        /^\d{4}-\d{2}-\d{2}$/.test(value.powerCurveWeight.date) &&
        typeof value.date === 'string' &&
        value.powerCurveWeight.date <= value.date &&
        value.powerCurveWeight.source === 'garmin')
    ) ||
    !(
      value.equipment === undefined ||
      (isRecord(value.equipment) &&
        typeof value.equipment.id === 'string' &&
        value.equipment.id.trim().length > 0 &&
        (value.equipment.name === null ||
          (typeof value.equipment.name === 'string' && value.equipment.name.trim().length > 0)) &&
        value.equipment.source === 'strava')
    ) ||
    !isActivityMoves(value.moves) ||
    !(
      value.computerOverride === undefined ||
      (value.sport === 'bike' &&
        typeof value.computerOverride === 'string' &&
        value.computerOverride.trim().length > 0)
    ) ||
    !(value.device === null || isActivityDevice(value.device)) ||
    !isStaminaTrace(value.staminaTrace) ||
    (isRecord(value.staminaTrace) &&
      ((value.staminaTrace.source === 'garden-estimate' && value.sport !== 'bike') ||
        (value.staminaTrace.source === 'garmin' &&
          value.sport !== 'bike' &&
          value.sport !== 'run'))) ||
    !(
      value.performanceConditionTrace === null ||
      isPerformanceConditionTrace(value.performanceConditionTrace)
    ) ||
    (isRecord(value.performanceConditionTrace) &&
      ((value.performanceConditionTrace.source === 'garden-estimate' && value.sport !== 'bike') ||
        (value.sport !== 'bike' && value.sport !== 'run'))) ||
    !finite(value.elapsedTimeS) ||
    !isSwimPhysiology(value.swimPhysiology, value.elapsedTimeS) ||
    !isSwimPower(value.swimPower, value.elapsedTimeS) ||
    ((value.swimPhysiology != null || value.swimPower != null) && value.sport !== 'swim') ||
    !isHeartRatePhysiology(value.heartRatePhysiology, value.elapsedTimeS) ||
    !isWalkPower(value.walkPower, value.elapsedTimeS, value.date) ||
    (value.walkPower != null && value.sport !== 'walk') ||
    value.elapsedTimeS < 0 ||
    value.elapsedTimeS > Number.MAX_SAFE_INTEGER ||
    !isCyclingTorqueTrace(value.cyclingTorque, value.elapsedTimeS) ||
    (value.cyclingTorque != null && (value.sport !== 'bike' || value.wahoo === undefined)) ||
    !isCyclingIntensityTrace(value.cyclingIntensityTrace, value.elapsedTimeS) ||
    (value.cyclingIntensityTrace != null &&
      (value.sport !== 'bike' || value.wahoo === undefined)) ||
    !isCyclingPowerTrace(value.cyclingPowerTrace, value.elapsedTimeS) ||
    (isRecord(value.cyclingPowerTrace) &&
      (value.sport !== 'bike' ||
        (value.cyclingPowerTrace.source === 'wahoo' && value.wahoo === undefined) ||
        (value.cyclingPowerTrace.source === 'strava' && value.deviceWatts !== true))) ||
    !nullableBounded(value.deviceTemperatureC, -90, 100) ||
    !nullableBounded(value.ambientTemperatureC, -90, 70) ||
    !isRouteTrace(value.route) ||
    !isThermalTrace(value.heartRateTrace) ||
    !isRunWalk(value.runWalk) ||
    (value.runWalk !== null && value.sport !== 'run')
  )
    return false
  return isActivityAnalyses(value.analyses, value.id, value.elapsedTimeS, value.start)
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
    runPowerCurveRef: value.runPowerCurveRef,
    runPowerCurveYearRef: value.runPowerCurveYearRef,
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
  runCurveRef: payload?.runPowerCurveRef ?? [],
  runCurveYearRef: payload?.runPowerCurveYearRef ?? [],
  curveYear: payload?.powerCurveYear ?? null,
  criticalPower: payload?.criticalPower ?? null,
  criticalPowerYear: payload?.criticalPowerYear ?? null,
  ftp: payload?.ftp ?? null,
  goalFtp: payload?.goalFtp ?? null,
  vt1: payload?.vt1Hr ?? null,
})
