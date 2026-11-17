import type { WeatherActivity, WeatherRouteHour } from '../plugins/stores/weather'
import type {
  ActivityEnvironmentInput,
  ActivityEnvironmentResult,
  EnvironmentChartSample,
  GardenEnvironmentCoverage,
  GardenEnvironmentEstimate,
  GardenEnvironmentSample,
  GardenEstimateProvenance,
  GardenUvDoseClocks,
} from './activity-environment/model'
import type { WindInterval } from './activity-environment/wind'
import {
  conditionAt,
  percentage,
  round,
  routeLength,
  validWindSpeedKph,
} from './activity-environment/model'
import {
  cumulativeSedAt,
  movingTelemetryDose,
  sampleIndices,
} from './activity-environment/sampling'
import { summarizeWind, windIntervals } from './activity-environment/wind'
import { parseSurfaceCurrentEstimate, type SurfaceCurrentSample } from './surface-current'

export const surfaceCurrentChartSamples = (
  samples: readonly SurfaceCurrentSample[],
): EnvironmentChartSample[] =>
  samples.map(sample => ({
    elapsedS: sample.elapsedS,
    surfaceCurrentSpeedMps: sample.speedMps,
    surfaceCurrentDirectionDeg: sample.directionDeg,
  }))

interface MetricAggregate {
  coveredS: number
  weightedTotal: number
  peak: number | null
}

const validRouteHours = (weather: WeatherActivity, durationS: number): WeatherRouteHour[] =>
  (weather.routeHours ?? [])
    .filter(
      hour =>
        Number.isFinite(hour.elapsedStartS) &&
        Number.isFinite(hour.elapsedEndS) &&
        hour.elapsedStartS >= 0 &&
        hour.elapsedEndS > hour.elapsedStartS &&
        hour.elapsedEndS <= durationS + 1,
    )
    .slice()
    .sort((left, right) => left.elapsedStartS - right.elapsedStartS)

const metricAggregate = (
  hours: readonly WeatherRouteHour[],
  value: (hour: WeatherRouteHour) => number | null,
): MetricAggregate => {
  let coveredS = 0
  let weightedTotal = 0
  let peak: number | null = null
  let coveredUntilS = 0
  for (const hour of hours) {
    const metric = value(hour)
    if (metric == null || !Number.isFinite(metric)) continue
    const startS = Math.max(coveredUntilS, hour.elapsedStartS)
    const durationS = Math.max(0, hour.elapsedEndS - startS)
    if (durationS <= 0) continue
    coveredUntilS = Math.max(coveredUntilS, hour.elapsedEndS)
    coveredS += durationS
    weightedTotal += metric * durationS
    peak = Math.max(peak ?? metric, metric)
  }
  return { coveredS, weightedTotal, peak }
}

export function buildActivityEnvironment(
  input: ActivityEnvironmentInput,
): ActivityEnvironmentResult {
  if (
    !Number.isFinite(input.elapsedTimeS) ||
    input.elapsedTimeS <= 0 ||
    input.weather.activityId !== input.activityId
  )
    return { environment: null, apparentWind: null }
  const hours = validRouteHours(input.weather, input.elapsedTimeS)
  if (hours.length === 0) return { environment: null, apparentWind: null }
  const weatherAggregate = metricAggregate(hours, () => 1)
  const uv = metricAggregate(hours, hour => hour.uvIndex)
  const temperature = metricAggregate(hours, hour => hour.temperatureC)
  const cloud = metricAggregate(hours, hour => hour.cloudCover)
  const ambientWind = metricAggregate(hours, hour => validWindSpeedKph(hour.windSpeedKph))
  const daylight = metricAggregate(hours, hour =>
    hour.daylight == null ? null : Number(hour.daylight),
  )
  const uvComplete = input.elapsedTimeS - uv.coveredS <= 1
  const elapsedSed = uvComplete ? round(uv.weightedTotal / 4_000, 3) : null
  const movingDose = uvComplete
    ? movingTelemetryDose(input, hours)
    : { totalSed: null, cumulativeSed: Array.from({ length: routeLength(input) }, () => null) }
  const doseClocks: GardenUvDoseClocks = { elapsedSed, movingTelemetrySed: movingDose.totalSed }
  const baseProvenance: GardenEstimateProvenance = {
    source: 'garden-estimate',
    formulaId: 'garden-environment-v1',
    formulaVersion: 1,
    inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1',
    normalizationVersion: 1,
    computedAt: input.computedAt,
    inputAsOf: input.weather.fetchedAt ?? 0,
    temporalSamplingModel: 'weatherkit-hourly-piecewise-constant',
    spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint',
  }
  const wind = windIntervals(input, hours)
  const apparentWind = summarizeWind(input, wind.intervals, wind.validDurationS, baseProvenance)
  const length = routeLength(input)
  const lastRouteElapsedS = input.timeS[length - 1]
  const lastRecordedDistanceM = input.distanceM
    .slice(0, length)
    .findLast(value => Number.isFinite(value) && value >= 0)
  const hasWeatherTail =
    Number.isFinite(lastRouteElapsedS) &&
    lastRouteElapsedS >= 0 &&
    lastRouteElapsedS < input.elapsedTimeS &&
    lastRecordedDistanceM != null
  const indices = sampleIndices(input, hours, wind.intervals, hasWeatherTail ? 319 : 320)
  const sampleAt = (
    elapsedS: number,
    distanceM: number,
    cumulativeMovingTelemetrySed: number | null,
    windInterval: WindInterval | null,
  ): GardenEnvironmentSample => {
    const weather = conditionAt(hours, elapsedS)
    return {
      elapsedS: round(elapsedS, 1),
      distanceKm: round(Math.max(0, distanceM) / 1_000, 3),
      uvIndex: weather?.uvIndex ?? null,
      cumulativeSed: elapsedSed == null ? null : cumulativeSedAt(hours, elapsedS),
      cumulativeMovingTelemetrySed,
      ambientTemperatureC: weather?.temperatureC == null ? null : round(weather.temperatureC, 1),
      cloudCoverPct: weather?.cloudCover == null ? null : round(weather.cloudCover * 100, 1),
      windSpeedKph: validWindSpeedKph(weather?.windSpeedKph),
      headwindKph: windInterval == null ? null : round(windInterval.headwindKph, 1),
      crosswindKph: windInterval == null ? null : round(windInterval.crosswindKph, 1),
      apparentAirSpeedKph: windInterval == null ? null : round(windInterval.apparentAirSpeedKph, 1),
      yawDeg: windInterval == null ? null : round(windInterval.yawDeg, 1),
    }
  }
  const samples = indices.map(index =>
    sampleAt(
      Math.min(input.elapsedTimeS, Math.max(0, input.timeS[index])),
      input.distanceM[index],
      movingDose.cumulativeSed[index] ?? null,
      wind.intervals[index],
    ),
  )
  if (hasWeatherTail && lastRecordedDistanceM != null)
    samples.push(sampleAt(input.elapsedTimeS, lastRecordedDistanceM, movingDose.totalSed, null))
  const coverage: GardenEnvironmentCoverage = {
    weatherPct: percentage(weatherAggregate.coveredS, input.elapsedTimeS),
    uvPct: percentage(uv.coveredS, input.elapsedTimeS),
    temperaturePct: percentage(temperature.coveredS, input.elapsedTimeS),
    cloudPct: percentage(cloud.coveredS, input.elapsedTimeS),
    daylightPct: percentage(daylight.coveredS, input.elapsedTimeS),
    windPct: percentage(ambientWind.coveredS, input.elapsedTimeS),
  }
  const environment: GardenEnvironmentEstimate = {
    ...baseProvenance,
    formulaId: 'garden-environment-v1',
    summary: {
      averageUvIndex: uv.coveredS > 0 ? round(uv.weightedTotal / uv.coveredS, 2) : null,
      peakUvIndex: uv.peak == null ? null : round(uv.peak, 1),
      uviHours: uvComplete ? round(uv.weightedTotal / 3_600, 3) : null,
      ambientSed: elapsedSed,
      averageAmbientTemperatureC:
        temperature.coveredS > 0
          ? round(temperature.weightedTotal / temperature.coveredS, 1)
          : null,
      averageCloudCoverPct:
        cloud.coveredS > 0 ? round((cloud.weightedTotal / cloud.coveredS) * 100, 1) : null,
      averageWindSpeedKph:
        ambientWind.coveredS > 0
          ? round(ambientWind.weightedTotal / ambientWind.coveredS, 1)
          : null,
      daylightCoveragePct: percentage(daylight.weightedTotal, input.elapsedTimeS),
      weatherCoveragePct: coverage.weatherPct,
      coveredDurationS: round(weatherAggregate.coveredS, 1),
      elapsedDurationS: input.elapsedTimeS,
    },
    doseClocks,
    coverage,
    samples,
    attribution: input.attribution,
  }
  const current = parseSurfaceCurrentEstimate(input.weather.surfaceCurrent)
  if (
    current &&
    current.activityId === input.activityId &&
    current.routeFingerprint === input.weather.routeFingerprint &&
    Date.parse(current.start) === Date.parse(input.weather.start) &&
    Date.parse(current.end) === Date.parse(input.weather.end) &&
    current.summary.elapsedDurationS === input.elapsedTimeS &&
    input.weather.durationS === input.elapsedTimeS
  ) {
    const { routeFingerprint: _routeFingerprint, ...surfaceCurrent } = current
    environment.surfaceCurrent = surfaceCurrent
  }
  return { environment, apparentWind }
}
export type {
  ActivityEnvironmentInput,
  ActivityEnvironmentResult,
  EnvironmentChartSample,
  GardenApparentWindEstimate,
  GardenApparentWindSummary,
  GardenEnvironmentCoverage,
  GardenEnvironmentEstimate,
  GardenEnvironmentSample,
  GardenEnvironmentSummary,
  GardenEstimateProvenance,
  GardenUvDoseClocks,
  GardenUvScore,
} from './activity-environment/model'
