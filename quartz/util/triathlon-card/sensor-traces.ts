import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { ActivityGraphDomain } from './chart-geometry'
import type { TriNodeFactory } from './construction-contracts'
import { activityThermalTracePoints, type ActivityThermalTracePoint } from './activity-metric-data'
import {
  formatMuscleOxygenMath,
  formatRespirationRate,
  formatTemperature,
  formatThermalTemperature,
  isImperial,
  temperatureUnit,
  temperatureValue,
} from './metric-text'
import { buildTrace, buildTraceSeries } from './trace-chart'

export const buildTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const imperial = isImperial(f.presentation)
  const temperaturesC = d.route
    .map(point => point.tempC)
    .filter((value): value is number => value != null)
  const averageC =
    d.ambientTemperatureC ??
    temperaturesC.reduce((total, value) => total + value, 0) / temperaturesC.length
  const values = temperaturesC.map(value => temperatureValue(f.presentation, value))
  const step = imperial ? 5 : 2
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    point => (point.tempC == null ? null : temperatureValue(f.presentation, point.tempC)),
    'temperature',
    () => `${formatTemperature(f.presentation, averageC)} avg`,
    value => `${Math.round(value)}${temperatureUnit(f.presentation)}`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const traceResolution = (values: number[], fallback: number): number => {
  const unique = [...new Set(values)].sort((left, right) => left - right)
  let resolution = Infinity
  for (let index = 1; index < unique.length; index++) {
    const delta = unique[index] - unique[index - 1]
    if (delta > 1e-6 && delta < resolution) resolution = delta
  }
  return Number.isFinite(resolution) ? resolution : fallback
}

const thermalTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  pickCelsius: (point: ActivityThermalTracePoint) => number | null,
  title: string,
  fallbackResolutionC: number,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const imperial = isImperial(f.presentation)
  const points = activityThermalTracePoints(d)
  const valuesC = points.map(pickCelsius).filter((value): value is number => value != null)
  if (valuesC.length < 2) return null
  const displayValues = valuesC.map(value => temperatureValue(f.presentation, value))
  const fallbackResolution = imperial ? (fallbackResolutionC * 9) / 5 : fallbackResolutionC
  const resolution = traceResolution(displayValues, fallbackResolution)
  const min = Math.min(...displayValues) - resolution
  const max = Math.max(...displayValues) + resolution
  const averageC = valuesC.reduce((total, value) => total + value, 0) / valuesC.length
  const digits = resolution < 0.1 ? 2 : 1
  return buildTraceSeries(
    f,
    d,
    points,
    point => {
      const celsius = pickCelsius(point)
      return celsius == null ? null : temperatureValue(f.presentation, celsius)
    },
    title,
    () => `${formatThermalTemperature(f.presentation, averageC)} avg`,
    value => `${value.toFixed(digits)}${temperatureUnit(f.presentation)}`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const buildHeatStrainTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const points = activityThermalTracePoints(d)
  const values = points
    .map(point => point.heatStrainIndex)
    .filter((value): value is number => value != null)
  if (values.length < 2) return null
  const resolution = traceResolution(values, 0.1)
  const min = Math.max(0, Math.min(...values) - resolution)
  const max = Math.max(...values) + resolution
  const average = values.reduce((total, value) => total + value, 0) / values.length
  return buildTraceSeries(
    f,
    d,
    points,
    point => point.heatStrainIndex,
    'heat strain index',
    () => `${average.toFixed(1)} avg`,
    value => value.toFixed(1),
    { min, max, intervals: 2 },
    selection,
    graphDomain,
    'dotted',
  )
}

export const buildCoreTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  thermalTemperatureTrace(
    f,
    d,
    point => point.coreTemperatureC,
    'CORE temperature',
    0.01,
    selection,
    graphDomain,
  )

export const buildSkinTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  thermalTemperatureTrace(
    f,
    d,
    point => point.skinTemperatureC,
    'skin temperature',
    0.05,
    selection,
    graphDomain,
  )

export const buildRespirationTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const values = d.route
    .map(point => point.resp)
    .filter((value): value is number => value != null && value > 0)
  const average =
    d.garmin?.runningDynamics?.averageRespirationRate ??
    values.reduce((total, value) => total + value, 0) / values.length
  const step = 5
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    point => point.resp ?? average,
    'respiration',
    () => `${formatRespirationRate(average)} avg`,
    value => `${Math.round(value)}brpm`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const buildMuscleOxygenTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const values = d.route
    .map(point => point.muscleOxygenPct)
    .filter((value): value is number => value != null && value >= 0 && value <= 100)
  if (values.length < 2) return null
  const average = values.reduce((total, value) => total + value, 0) / values.length
  let min = Math.floor(Math.min(...values) / 5) * 5
  let max = Math.ceil(Math.max(...values) / 5) * 5
  if (max <= min) {
    min = Math.max(0, min - 5)
    max = Math.min(100, max + 5)
  }
  return buildTrace(
    f,
    d,
    point => point.muscleOxygenPct ?? average,
    'muscle oxygen',
    () => `${formatMuscleOxygenMath(average)} avg`,
    value => `${value.toFixed(1)}%`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}
