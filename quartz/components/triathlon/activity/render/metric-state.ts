import type { StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import {
  activityCadenceUnit,
  activityPhysiologyTracePoints,
  clock,
  hasHeartRateTrace,
  interpolatePositiveMetricSeries,
  KM_TO_MI,
  performanceConditionValue,
  positiveMetricDomain,
  routeStreamFlags,
  runGroundContactTimeMs,
  runStrideLengthLabel,
  runStrideLengthValue,
  runVerticalOscillationCm,
} from '../../../../util/triathlon-card'
import { createDomFactory } from '../../runtime/dom'
import { analysisRate } from '../analysis'
import { environmentValueAtElapsed } from './scrub-surfaces'

export type ThermalMetric = 'temperature' | 'heat-strain' | 'core-temperature' | 'skin-temperature'

export const metricSpecState = (presentation: TriathlonPresentation, d: StravaActivityDetail) => {
  const domF = createDomFactory(presentation)
  const route = d.route
  const staminaPoints = new Map(
    activityPhysiologyTracePoints(d, 'stamina').map(point => [point.elapsedS, point]),
  )
  const conditionPoints = new Map(
    activityPhysiologyTracePoints(d, 'performanceCondition').map(point => [point.elapsedS, point]),
  )
  const conditionAt = (point: StravaActivityDetail['route'][number]): number | null =>
    performanceConditionValue(conditionPoints.get(point.elapsedS) ?? point)
  const staminaAt = (point: StravaActivityDetail['route'][number]) =>
    staminaPoints.get(point.elapsedS) ?? point
  const imperial = presentation.distance === 'imperial'
  const filterPowerZeros = d.sport === 'bike' && presentation.powerSamples === 'exclude-zero'
  const powerValues = filterPowerZeros
    ? interpolatePositiveMetricSeries(route, point => point.w)
    : route.map(point => point.w)
  const powerDomain = filterPowerZeros ? positiveMetricDomain(powerValues) : undefined
  const pace = route.map(point => point.speedKph)
  const hasPower = d.deviceWatts && route.some(p => p.w > 0)
  const hasHr = hasHeartRateTrace(d)
  const hasCad = route.some(p => p.cad > 0)
  const strideLabel = runStrideLengthLabel(d)
  const hasStride =
    d.sport === 'run' && route.filter(p => runStrideLengthValue(d, p) != null).length >= 2
  const hasGroundContact =
    d.sport === 'run' && route.filter(p => runGroundContactTimeMs(p) != null).length >= 2
  const hasVerticalOscillation =
    d.sport === 'run' && route.filter(p => runVerticalOscillationCm(p) != null).length >= 2
  const hasResp = route.some(p => p.resp != null && p.resp > 0)
  const hasElev = d.maxAlt > d.minAlt
  const flags = routeStreamFlags(d)
  const environmentSamples = d.analyses.derived.environment?.samples ?? []
  const uvValues = route.map(point =>
    environmentValueAtElapsed(environmentSamples, point.elapsedS, sample => sample.uvIndex, true),
  )
  const headwindValues = route.map(point =>
    environmentValueAtElapsed(environmentSamples, point.elapsedS, sample => sample.headwindKph),
  )
  const crosswindValues = route.map(point =>
    environmentValueAtElapsed(environmentSamples, point.elapsedS, sample => sample.crosswindKph),
  )
  const apparentAirValues = route.map(point =>
    environmentValueAtElapsed(
      environmentSamples,
      point.elapsedS,
      sample => sample.apparentAirSpeedKph,
    ),
  )
  const yawValues = route.map(point =>
    environmentValueAtElapsed(environmentSamples, point.elapsedS, sample => sample.yawDeg),
  )
  const hasUv = uvValues.filter(value => value != null).length >= 2
  const hasWind = headwindValues.filter(value => value != null).length >= 2
  const hasMuscleOxygen = flags.muscleOxygen
  const maxDistanceKm = Math.max(route.at(-1)?.d ?? d.distanceKm, 0.001)
  const hasRiderPosition =
    d.sport === 'bike' &&
    d.cyclingDynamics?.positionChanges.some(
      change => change.distanceKm >= 0 && change.distanceKm <= maxDistanceKm,
    ) === true
  const hasShifting = d.sport === 'bike' && d.gearShifts.length > 0
  const thermalMetrics: ThermalMetric[] = []
  if (flags.temp) thermalMetrics.push('temperature')
  if (flags.heatStrain) thermalMetrics.push('heat-strain')
  if (flags.coreTemperature) thermalMetrics.push('core-temperature')
  if (flags.skinTemperature) thermalMetrics.push('skin-temperature')
  const hasThermal = thermalMetrics.length > 0
  const primaryThermal = thermalMetrics[0] ?? 'temperature'
  const thermalValue = (
    point: StravaActivityDetail['route'][number],
    metric: ThermalMetric,
  ): number | null => {
    if (metric === 'temperature') return point.tempC
    if (metric === 'heat-strain') return point.heatStrainIndex
    if (metric === 'core-temperature') return point.coreTemperatureC
    return point.skinTemperatureC
  }
  const primaryThermalValues = route
    .map(point => thermalValue(point, primaryThermal))
    .filter((value): value is number => value != null)
  const primaryThermalAverage =
    primaryThermalValues.reduce((total, value) => total + value, 0) /
    Math.max(1, primaryThermalValues.length)
  const cadUnit = activityCadenceUnit(d.sport)
  const paceFmt = (kmh: number): string => {
    return analysisRate(presentation, d.sport, kmh)
  }
  const paceTick = (kmh: number): string => {
    if (kmh <= 0) return '0'
    if (d.sport === 'bike')
      return imperial ? `${(kmh * KM_TO_MI).toFixed(0)}mph` : `${kmh.toFixed(0)}km/h`
    if (d.sport === 'swim') return clock(3600 / (kmh * 10))
    return clock(3600 / (kmh * (imperial ? KM_TO_MI : 1)))
  }
  return {
    domF,
    route,
    conditionAt,
    staminaAt,
    filterPowerZeros,
    powerValues,
    powerDomain,
    pace,
    hasPower,
    hasHr,
    hasCad,
    strideLabel,
    hasStride,
    hasGroundContact,
    hasVerticalOscillation,
    hasResp,
    hasElev,
    flags,
    uvValues,
    headwindValues,
    crosswindValues,
    apparentAirValues,
    yawValues,
    hasUv,
    hasWind,
    hasMuscleOxygen,
    hasRiderPosition,
    hasShifting,
    thermalMetrics,
    hasThermal,
    primaryThermal,
    thermalValue,
    primaryThermalAverage,
    cadUnit,
    paceFmt,
    paceTick,
  }
}

export type MetricSpecState = ReturnType<typeof metricSpecState>
