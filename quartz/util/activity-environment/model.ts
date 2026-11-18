import type {
  WeatherActivity,
  WeatherAttribution,
  WeatherRouteHour,
} from '../../plugins/stores/weather'
import { type PublicSurfaceCurrentEstimate } from '../surface-current'

export interface GardenEstimateProvenance {
  source: 'garden-estimate'
  formulaId: string
  formulaVersion: 1
  inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1'
  normalizationVersion: 1
  computedAt: number
  inputAsOf: number
  temporalSamplingModel: 'weatherkit-hourly-piecewise-constant'
  spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint'
}

export interface GardenEnvironmentCoverage {
  weatherPct: number
  uvPct: number
  temperaturePct: number
  cloudPct: number
  daylightPct: number
  windPct?: number
}

export interface GardenEnvironmentSummary {
  averageUvIndex: number | null
  peakUvIndex: number | null
  uviHours: number | null
  ambientSed: number | null
  averageAmbientTemperatureC: number | null
  averageCloudCoverPct: number | null
  averageWindSpeedKph?: number | null
  daylightCoveragePct: number
  weatherCoveragePct: number
  coveredDurationS: number
  elapsedDurationS: number
}

export interface GardenUvDoseClocks {
  elapsedSed: number | null
  movingTelemetrySed: number | null
}

export interface GardenEnvironmentSample {
  elapsedS: number
  distanceKm: number
  uvIndex: number | null
  cumulativeSed: number | null
  cumulativeMovingTelemetrySed: number | null
  ambientTemperatureC: number | null
  cloudCoverPct: number | null
  windSpeedKph?: number | null
  headwindKph: number | null
  crosswindKph: number | null
  apparentAirSpeedKph: number | null
  yawDeg: number | null
}

export interface EnvironmentChartSample extends Partial<
  Omit<GardenEnvironmentSample, 'elapsedS' | 'distanceKm'>
> {
  elapsedS: number
  surfaceCurrentSpeedMps?: number | null
  surfaceCurrentDirectionDeg?: number | null
  providerHeadwindKph?: number | null
  providerCrosswindKph?: number | null
  weatherCostW?: number | null
  movingAirPenaltyKm?: number | null
}

export interface GardenEnvironmentEstimate extends GardenEstimateProvenance {
  formulaId: 'garden-environment-v1'
  summary: GardenEnvironmentSummary
  doseClocks: GardenUvDoseClocks
  coverage: GardenEnvironmentCoverage
  samples: GardenEnvironmentSample[]
  attribution: WeatherAttribution | null
  surfaceCurrent?: PublicSurfaceCurrentEstimate
}

export interface GardenApparentWindSummary {
  headwindSharePct: number
  headwindTimeS: number
  tailwindTimeS: number
  longestHeadwindS: number
  // Signed route means cancel on loops (a closed route integrates uniform wind to zero),
  // so the summary keeps the headwind mean over headwind time and the crosswind magnitude.
  averageHeadwindWhileIntoKph: number
  averageCrosswindMagnitudeKph: number
  maximumHeadwindKph: number
  maximumCrosswindKph: number
  averageGroundSpeedKph: number
  averageApparentAirSpeedKph: number
  apparentAirRatio: number
  averageYawDeg: number
  coveragePct: number
}

export interface GardenApparentWindEstimate extends GardenEstimateProvenance {
  formulaId: 'garden-apparent-wind-v3'
  summary: GardenApparentWindSummary
  coverage: { windPct: number }
}

export interface GardenUvScore extends GardenEstimateProvenance {
  formulaId: 'garden-uv-score-v1'
  score: number
  severity: 'negligible' | 'low' | 'moderate' | 'high' | 'serious' | 'extreme'
  doseClock: 'elapsed' | 'moving-telemetry'
  doseSed: number
  coefficientSed: number
  calibrationVersion: 1
}

export interface ActivityEnvironmentInput {
  activityId: string
  elapsedTimeS: number
  movingTimeS: number
  timeS: readonly number[]
  distanceM: readonly number[]
  latlng: readonly [number, number][]
  // GPS swims are open water: lower speeds and a smooth water surface below the swimmer.
  openWater: boolean
  weather: WeatherActivity
  attribution: WeatherAttribution | null
  computedAt: number
}

export interface ActivityEnvironmentResult {
  environment: GardenEnvironmentEstimate | null
  apparentWind: GardenApparentWindEstimate | null
}

export const round = (value: number, digits = 2): number => {
  const factor = 10 ** digits
  const rounded = Math.round(value * factor) / factor
  return Object.is(rounded, -0) ? 0 : rounded
}

export const percentage = (coveredS: number, durationS: number): number =>
  durationS > 0 ? round(Math.min(1, Math.max(0, coveredS / durationS)) * 100, 1) : 0

export const validWindSpeedKph = (value: number | null | undefined): number | null =>
  value != null && Number.isFinite(value) && value >= 0 && value <= 1_000 ? value : null

export const conditionAt = (
  hours: readonly WeatherRouteHour[],
  elapsedS: number,
): WeatherRouteHour | null =>
  hours.find(
    (hour, index) =>
      elapsedS >= hour.elapsedStartS &&
      (elapsedS < hour.elapsedEndS ||
        (index === hours.length - 1 && elapsedS <= hour.elapsedEndS + 1)),
  ) ?? null

export const routeLength = (input: ActivityEnvironmentInput): number =>
  Math.min(input.timeS.length, input.distanceM.length, input.latlng.length)
