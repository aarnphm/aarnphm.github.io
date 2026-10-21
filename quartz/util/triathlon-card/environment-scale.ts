import type { EnvironmentChartSample, GardenUvScore } from '../activity-environment'
import type { TriathlonPresentation } from '../triathlon-presentation'
import { gardenUvScoreFromDose } from '../activity-uv-score'
import { niceStep, niceTicks, type AxisYTick } from './chart-geometry'
import { isImperial, KM_TO_MI, temperatureUnit, temperatureValue } from './metric-text'

export type EnvironmentChartView =
  | 'cumulative'
  | 'uv-index'
  | 'temperature'
  | 'cloud-cover'
  | 'wind'
  | 'current-speed'
  | 'current-direction'
  | 'relative-wind'
  | 'weather-cost'
  | 'air-penalty'

export const ENVIRONMENT_CHART_VIEWS: readonly EnvironmentChartView[] = [
  'cumulative',
  'uv-index',
  'temperature',
  'cloud-cover',
  'wind',
]

export const ENVIRONMENT_CURRENT_CHART_VIEWS: readonly EnvironmentChartView[] = [
  'current-speed',
  'current-direction',
]

// myWindsock route views read the provider series; the rose has no elapsed axis.
export const ENVIRONMENT_PROVIDER_CHART_VIEWS: readonly EnvironmentChartView[] = [
  'weather-cost',
  'air-penalty',
]

export const ALL_ENVIRONMENT_CHART_VIEWS: readonly EnvironmentChartView[] = [
  ...ENVIRONMENT_CHART_VIEWS,
  ...ENVIRONMENT_CURRENT_CHART_VIEWS,
  'relative-wind',
  'weather-cost',
  'air-penalty',
]

export interface EnvironmentChartSeries {
  path: string
  color: string | null
}

export type EnvironmentScoreModel = Pick<GardenUvScore, 'coefficientSed' | 'doseClock'>
type EnvironmentWindMetric = 'headwind' | 'ambient'

interface EnvironmentChartScale {
  minimum: number
  maximum: number
  ticks: readonly number[]
}

export const ENVIRONMENT_CHART_LEFT = 2
export const ENVIRONMENT_CHART_RIGHT = 98
export const ENVIRONMENT_CHART_TOP = 3
export const ENVIRONMENT_CHART_BOTTOM = 27

export const environmentUvColor = (uvIndex: number): string => {
  if (uvIndex < 3) return 'var(--tri-environment-uv-low)'
  if (uvIndex < 6) return 'var(--tri-environment-uv-moderate)'
  if (uvIndex < 8) return 'var(--tri-environment-uv-high)'
  if (uvIndex < 11) return 'var(--tri-environment-uv-very-high)'
  return 'var(--tri-environment-uv-extreme)'
}

export const environmentWindMetric = (
  samples: readonly EnvironmentChartSample[],
): EnvironmentWindMetric =>
  samples.filter(sample => sample.headwindKph != null && Number.isFinite(sample.headwindKph))
    .length >= 2
    ? 'headwind'
    : 'ambient'

export const environmentSampleValue = (
  presentation: TriathlonPresentation,
  sample: EnvironmentChartSample,
  view: EnvironmentChartView,
  scoreModel: EnvironmentScoreModel | null,
  windMetric: EnvironmentWindMetric = 'headwind',
): number | null => {
  if (view === 'uv-index') return sample.uvIndex ?? null
  if (view === 'current-speed') return sample.surfaceCurrentSpeedMps ?? null
  if (view === 'current-direction') return sample.surfaceCurrentDirectionDeg ?? null
  if (view === 'temperature')
    return sample.ambientTemperatureC == null
      ? null
      : temperatureValue(presentation, sample.ambientTemperatureC)
  if (view === 'cloud-cover') return sample.cloudCoverPct ?? null
  if (view === 'wind') {
    const value = windMetric === 'ambient' ? sample.windSpeedKph : sample.headwindKph
    return value == null ? null : value * (isImperial(presentation) ? KM_TO_MI : 1)
  }
  if (view === 'weather-cost') return sample.weatherCostW ?? null
  if (view === 'air-penalty')
    return sample.movingAirPenaltyKm == null
      ? null
      : sample.movingAirPenaltyKm * (isImperial(presentation) ? KM_TO_MI : 1)
  if (view === 'relative-wind') return null
  const doseSed =
    scoreModel?.doseClock === 'moving-telemetry'
      ? sample.cumulativeMovingTelemetrySed
      : sample.cumulativeSed
  if (doseSed == null) return null
  return scoreModel == null ? doseSed : gardenUvScoreFromDose(doseSed, scoreModel.coefficientSed)
}

const environmentNiceCeiling = (value: number): number => {
  if (!Number.isFinite(value) || value <= 0) return 1
  const magnitude = 10 ** Math.floor(Math.log10(value))
  const normalized = value / magnitude
  const ceiling = [1, 2, 2.5, 5, 10].find(candidate => candidate >= normalized) ?? 10
  return ceiling * magnitude
}

export const environmentChartScale = (
  values: readonly number[],
  view: EnvironmentChartView,
  scoreModel: EnvironmentScoreModel | null,
  windMetric: EnvironmentWindMetric = 'headwind',
): EnvironmentChartScale => {
  if (view === 'current-direction') return { minimum: 0, maximum: 360, ticks: [0, 180, 360] }
  if (view === 'current-speed') {
    const maximum = environmentNiceCeiling(Math.max(...values))
    return { minimum: 0, maximum, ticks: [0, maximum / 2, maximum] }
  }
  if (view === 'cloud-cover' || (view === 'cumulative' && scoreModel != null))
    return { minimum: 0, maximum: 100, ticks: [0, 50, 100] }
  if (view === 'uv-index') {
    const maximum = Math.max(11, Math.ceil(Math.max(...values)))
    return { minimum: 0, maximum, ticks: [0, maximum / 2, maximum] }
  }
  if (view === 'wind' || view === 'weather-cost') {
    const maximum = environmentNiceCeiling(Math.max(...values.map(Math.abs)))
    if (windMetric === 'ambient') return { minimum: 0, maximum, ticks: [0, maximum / 2, maximum] }
    return { minimum: -maximum, maximum, ticks: [-maximum, 0, maximum] }
  }
  if (view === 'air-penalty') {
    const maximum = environmentNiceCeiling(Math.max(...values.map(Math.abs)))
    // A tailwind gives less air than ground distance, so the sum can fall below zero.
    if (Math.min(...values) >= 0) return { minimum: 0, maximum, ticks: [0, maximum / 2, maximum] }
    return { minimum: -maximum, maximum, ticks: [-maximum, 0, maximum] }
  }
  if (view === 'cumulative') {
    const maximum = environmentNiceCeiling(Math.max(...values))
    return { minimum: 0, maximum, ticks: [0, maximum / 2, maximum] }
  }
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const pad = Math.max(0.5, (maximum - minimum) * 0.08)
  const lower = minimum - pad
  const upper = maximum + pad
  const step = niceStep(upper - lower, 2)
  const domainMinimum = Math.floor(lower / step) * step
  const domainMaximum = Math.ceil(upper / step) * step
  return {
    minimum: domainMinimum,
    maximum: domainMaximum,
    ticks: niceTicks(domainMinimum, domainMaximum, 3),
  }
}

export const environmentChartY = (value: number, scale: EnvironmentChartScale): number =>
  ENVIRONMENT_CHART_BOTTOM -
  Math.min(1, Math.max(0, (value - scale.minimum) / (scale.maximum - scale.minimum))) *
    (ENVIRONMENT_CHART_BOTTOM - ENVIRONMENT_CHART_TOP)

export const environmentChartValues = (
  presentation: TriathlonPresentation,
  samples: readonly EnvironmentChartSample[],
  view: EnvironmentChartView,
  scoreModel: EnvironmentScoreModel | null,
): number[] => {
  const windMetric = view === 'wind' ? environmentWindMetric(samples) : 'headwind'
  return samples.flatMap(sample => {
    const value = environmentSampleValue(presentation, sample, view, scoreModel, windMetric)
    return value == null || !Number.isFinite(value) ? [] : [value]
  })
}

export const environmentViewLabel = (view: EnvironmentChartView): string => {
  if (view === 'uv-index') return 'UV index'
  if (view === 'cloud-cover') return 'cloud cover'
  if (view === 'current-speed') return 'current speed'
  if (view === 'current-direction') return 'current direction'
  if (view === 'relative-wind') return 'relative wind'
  if (view === 'weather-cost') return 'weather cost'
  if (view === 'air-penalty') return 'air penalty'
  return view
}

export const environmentViewShortLabel = (view: EnvironmentChartView): string => {
  if (view === 'cumulative') return 'cum.'
  if (view === 'uv-index') return 'UVI'
  if (view === 'temperature') return 'temp.'
  if (view === 'wind') return 'wind'
  if (view === 'current-speed') return 'current spd.'
  if (view === 'current-direction') return 'current dir.'
  if (view === 'relative-wind') return 'rel. wind'
  if (view === 'weather-cost') return 'cost'
  if (view === 'air-penalty') return 'air pen.'
  return 'cloud'
}

export const environmentViewHasSamples = (
  presentation: TriathlonPresentation,
  samples: readonly EnvironmentChartSample[],
  view: EnvironmentChartView,
): boolean => environmentChartValues(presentation, samples, view, null).length >= 2

const environmentAxisLabel = (
  presentation: TriathlonPresentation,
  view: EnvironmentChartView,
  scoreModel: EnvironmentScoreModel | null,
  value: number,
  windMetric: EnvironmentWindMetric,
): string => {
  if (view === 'cloud-cover') return `${Math.round(value)}%`
  if (view === 'current-speed') return `${value.toFixed(2)} m/s`
  if (view === 'current-direction') return `${Math.round(value)}°`
  if (view === 'uv-index') return value.toFixed(1)
  if (view === 'temperature') return `${Number(value.toFixed(1))}${temperatureUnit(presentation)}`
  if (view === 'wind')
    return `${windMetric === 'headwind' && value > 0 ? '+' : ''}${Number(value.toFixed(1))} ${isImperial(presentation) ? 'mph' : 'km/h'}`
  if (view === 'weather-cost') return `${value > 0 ? '+' : ''}${Number(value.toFixed(1))} W`
  if (view === 'air-penalty')
    return `${Number(value.toFixed(2))} ${isImperial(presentation) ? 'mi' : 'km'}`
  if (scoreModel != null) return `${Math.round(value)}`
  const locale = presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  return `${value.toLocaleString(locale, { maximumFractionDigits: 2 })} SED`
}

export const environmentAxisTicks = (
  presentation: TriathlonPresentation,
  samples: readonly EnvironmentChartSample[],
  view: EnvironmentChartView,
  scoreModel: EnvironmentScoreModel | null,
  mode?: 'score' | 'sed',
): AxisYTick[] => {
  const values = environmentChartValues(presentation, samples, view, scoreModel)
  if (values.length < 2) return []
  const windMetric = view === 'wind' ? environmentWindMetric(samples) : 'headwind'
  const scale = environmentChartScale(values, view, scoreModel, windMetric)
  return scale.ticks.map(value => ({
    label: environmentAxisLabel(presentation, view, scoreModel, value, windMetric),
    vbY: environmentChartY(value, scale),
    ...(mode
      ? {
          attrs: {
            'data-environment-cumulative-axis': mode,
            ...(mode === 'sed' ? { hidden: '' } : {}),
          },
        }
      : {}),
  }))
}
