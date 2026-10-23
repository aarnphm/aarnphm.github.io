import { isMyWindsockArchiveReference } from '../../../../util/mywindsock-graphs'
import { parsePublicSurfaceCurrentEstimate } from '../../../../util/surface-current'
import { isRecord } from '../../../../util/type-guards'
import { bounded, finite, nullableBounded, nullableFinite } from './predicates'

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

export const isActivityAnalyses = (
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
