import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import { runMetersPerBeat, type RunHeartCostPoint } from '../run-mechanics'
import { excludesZeroPower } from './metric-text'

export const powerViewActivity = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
): StravaActivityDetail => {
  const filtered = activity.powerWithoutZeros
  if (!excludesZeroPower(presentation) || activity.sport !== 'bike' || !filtered) return activity
  return {
    ...activity,
    avgWatts: filtered.avgWatts ?? activity.avgWatts,
    powerZones: filtered.powerZones ?? activity.powerZones,
    powerHist: filtered.powerHist ?? activity.powerHist,
  }
}

export const interpolatePositiveMetricSeries = (
  route: StravaActivityDetail['route'],
  pick: (point: StravaActivityDetail['route'][number], index: number) => number | null | undefined,
): (number | null)[] => {
  const values = route.map((point, index) => {
    const value = pick(point, index)
    return value != null && Number.isFinite(value) && value > 0 ? value : null
  })
  const previous = Array.from({ length: values.length }, () => -1)
  const next = Array.from({ length: values.length }, () => -1)
  let measured = -1
  for (let index = 0; index < values.length; index++) {
    previous[index] = measured
    if (values[index] != null) measured = index
  }
  measured = -1
  for (let index = values.length - 1; index >= 0; index--) {
    next[index] = measured
    if (values[index] != null) measured = index
  }
  return values.map((value, index) => {
    if (value != null) return value
    const left = previous[index]
    const right = next[index]
    if (left < 0) return right < 0 ? null : values[right]
    if (right < 0) return values[left]
    const leftValue = values[left]
    const rightValue = values[right]
    if (leftValue == null || rightValue == null) return leftValue ?? rightValue
    const distanceSpan = route[right].d - route[left].d
    const fraction =
      distanceSpan > 0
        ? (route[index].d - route[left].d) / distanceSpan
        : (index - left) / (right - left)
    return leftValue + (rightValue - leftValue) * Math.max(0, Math.min(1, fraction))
  })
}

export const gradeAt = (route: StravaActivityDetail['route'], i: number): number => {
  const j0 = Math.max(0, i - 2)
  const j1 = Math.min(route.length - 1, i + 2)
  const dKm = route[j1].d - route[j0].d
  return dKm > 0 ? ((route[j1].alt - route[j0].alt) / (dKm * 1000)) * 100 : 0
}

export type ActivityThermalTracePoint = {
  d: number
  elapsedS: number
  heatStrainIndex: number | null
  coreTemperatureC: number | null
  skinTemperatureC: number | null
}

const routeLessTraceDistance = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['heartRateTrace'][number],
): number => (activityTraceUsesElapsedAxis(d) ? point.elapsedS : point.distanceKm)

export const activityTraceUsesElapsedAxis = (d: StravaActivityDetail): boolean =>
  d.route.length < 2 &&
  (d.sport !== 'swim' || !d.heartRateTrace.some(point => point.distanceKm > 0))

export const activityThermalTracePoints = (d: StravaActivityDetail): ActivityThermalTracePoint[] =>
  d.route.length >= 2
    ? d.route
    : d.heartRateTrace.map(point => ({
        d: routeLessTraceDistance(d, point),
        elapsedS: point.elapsedS,
        heatStrainIndex: point.heatStrainIndex,
        coreTemperatureC: point.coreTemperatureC,
        skinTemperatureC: point.skinTemperatureC,
      }))

export const routeStreamFlags = (
  d: StravaActivityDetail,
): {
  power: boolean
  powerBalance: boolean
  hr: boolean
  cad: boolean
  performanceCondition: boolean
  stride: boolean
  verticalRatio: boolean
  groundContact: boolean
  groundContactBalance: boolean
  verticalOscillation: boolean
  stepSpeedLoss: boolean
  stepSpeedLossPercent: boolean
  impactLoadFactor: boolean
  runWalk: boolean
  stamina: boolean
  resp: boolean
  muscleOxygen: boolean
  temp: boolean
  heatStrain: boolean
  coreTemperature: boolean
  skinTemperature: boolean
} => {
  const thermalPoints = activityThermalTracePoints(d)
  return {
    power: d.deviceWatts && d.route.some(p => p.w > 0),
    powerBalance:
      d.sport === 'bike' &&
      d.route.filter(
        point =>
          point.rightPowerPct != null &&
          Number.isFinite(point.rightPowerPct) &&
          point.rightPowerPct >= 0 &&
          point.rightPowerPct <= 100 &&
          point.w > 0,
      ).length >= 2,
    hr: hasHeartRateTrace(d),
    cad: d.route.some(p => p.cad > 0),
    performanceCondition:
      activityPhysiologyTracePoints(d, 'performanceCondition').filter(
        point => performanceConditionValue(point) != null,
      ).length >= 2,
    stride:
      d.sport === 'run' &&
      d.route.filter(point => runStrideLengthValue(d, point) != null).length >= 2,
    groundContact:
      d.sport === 'run' &&
      d.route.filter(point => runGroundContactTimeMs(point) != null).length >= 2,
    groundContactBalance:
      d.sport === 'run' &&
      d.route.filter(point => runGroundContactBalanceLeftPct(point) != null).length >= 2,
    verticalOscillation:
      d.sport === 'run' &&
      d.route.filter(point => runVerticalOscillationCm(point) != null).length >= 2,
    verticalRatio:
      d.sport === 'run' && d.route.filter(point => runVerticalRatioPct(point) != null).length >= 2,
    stepSpeedLoss:
      d.sport === 'run' && d.route.filter(point => runStepSpeedLossMps(point) != null).length >= 2,
    stepSpeedLossPercent:
      d.sport === 'run' && d.route.filter(point => runStepSpeedLossPct(point) != null).length >= 2,
    impactLoadFactor:
      d.sport === 'run' && d.route.filter(point => runImpactLoadFactor(point) != null).length >= 2,
    runWalk: d.sport === 'run' && d.runWalk != null,
    stamina:
      activityPhysiologyTracePoints(d, 'stamina').filter(
        point => point.stamina != null && point.potentialStamina != null,
      ).length >= 2,
    resp: d.route.some(p => p.resp != null && p.resp > 0),
    muscleOxygen:
      d.route.filter(
        point =>
          point.muscleOxygenPct != null &&
          Number.isFinite(point.muscleOxygenPct) &&
          point.muscleOxygenPct >= 0 &&
          point.muscleOxygenPct <= 100,
      ).length >= 2,
    temp: d.route.some(p => p.tempC != null),
    heatStrain: thermalPoints.filter(point => point.heatStrainIndex != null).length >= 2,
    coreTemperature: thermalPoints.filter(point => point.coreTemperatureC != null).length >= 2,
    skinTemperature: thermalPoints.filter(point => point.skinTemperatureC != null).length >= 2,
  }
}

const nativeRunStrideLengthM = (point: StravaActivityDetail['route'][number]): number | null => {
  const meters = point.strideLengthM
  return meters != null && Number.isFinite(meters) && meters >= 0.2 && meters <= 3 ? meters : null
}

const estimatedRunStrideLengthM = (point: StravaActivityDetail['route'][number]): number | null => {
  if (!Number.isFinite(point.speedKph) || !Number.isFinite(point.cad) || point.cad <= 0) return null
  const meters = (point.speedKph * 1000) / 60 / (point.cad * 2)
  return Number.isFinite(meters) && meters >= 0.2 && meters <= 3 ? meters : null
}

export const runStrideLengthM = (point: StravaActivityDetail['route'][number]): number | null =>
  nativeRunStrideLengthM(point) ?? estimatedRunStrideLengthM(point)

export const runStrideLengthLabel = (d: StravaActivityDetail): string =>
  d.route.filter(point => nativeRunStrideLengthM(point) != null).length >= 2
    ? 'stride length'
    : 'estimated stride length'

export const runStrideLengthValue = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
): number | null =>
  runStrideLengthLabel(d) === 'stride length'
    ? nativeRunStrideLengthM(point)
    : estimatedRunStrideLengthM(point)

export const runGroundContactTimeMs = (
  point: StravaActivityDetail['route'][number],
): number | null => {
  const milliseconds = point.groundContactTimeMs
  return milliseconds != null &&
    Number.isFinite(milliseconds) &&
    milliseconds >= 50 &&
    milliseconds <= 1_000
    ? milliseconds
    : null
}

const boundedRunMetric = (
  value: number | null | undefined,
  minimum: number,
  maximum: number,
): number | null =>
  value != null && Number.isFinite(value) && value >= minimum && value <= maximum ? value : null

export const performanceConditionValue = (point: {
  performanceCondition?: number | null
}): number | null => boundedRunMetric(point.performanceCondition, -20, 20)

export const runVerticalRatioPct = (point: StravaActivityDetail['route'][number]): number | null =>
  boundedRunMetric(point.verticalRatioPct, 0, 50)

export const runGroundContactBalanceLeftPct = (
  point: StravaActivityDetail['route'][number],
): number | null => boundedRunMetric(point.groundContactBalanceLeftPct, 0, 100)

export const runStepSpeedLossMps = (point: StravaActivityDetail['route'][number]): number | null =>
  boundedRunMetric(point.stepSpeedLossMps, 0, 5)

export const runStepSpeedLossPct = (point: StravaActivityDetail['route'][number]): number | null =>
  boundedRunMetric(point.stepSpeedLossPct, 0, 100)

export const runImpactLoadFactor = (point: StravaActivityDetail['route'][number]): number | null =>
  boundedRunMetric(point.impactLoadFactor, 0, 10)

export const runVerticalOscillationCm = (
  point: StravaActivityDetail['route'][number],
): number | null => {
  const centimeters = point.verticalOscillationCm
  return centimeters != null &&
    Number.isFinite(centimeters) &&
    centimeters >= 1 &&
    centimeters <= 30
    ? centimeters
    : null
}

export const activityWalkPowerPoints = (d: StravaActivityDetail) =>
  (d.walkPower?.points ?? []).map(point => ({
    ...point,
    d: activityTraceUsesElapsedAxis(d) ? point.elapsedS : point.distanceKm,
  }))

export const activityCyclingIntensityPoints = (d: StravaActivityDetail) =>
  (d.cyclingIntensityTrace?.points ?? []).map(point => ({
    ...point,
    d: activityTraceUsesElapsedAxis(d) ? point.elapsedS : point.distanceKm,
  }))

export const activityCyclingTorquePoints = (d: StravaActivityDetail) => {
  const points = (d.cyclingTorque?.points ?? []).map(point => ({
    ...point,
    d: activityTraceUsesElapsedAxis(d) ? point.elapsedS : point.distanceKm,
  }))
  const last = points.at(-1)
  if (last && last.elapsedS < d.elapsedTimeS)
    points.push({
      ...last,
      elapsedS: d.elapsedTimeS,
      durationS: 0,
      distanceKm: d.route.at(-1)?.d ?? d.distanceKm,
      d: activityTraceUsesElapsedAxis(d) ? d.elapsedTimeS : (d.route.at(-1)?.d ?? d.distanceKm),
    })
  return points
}

export const activityCyclingPowerPoints = (d: StravaActivityDetail) =>
  (d.cyclingPowerTrace?.points ?? []).map(point => ({ ...point, d: point.elapsedS }))

export const activityRunHeartCostPoints = (d: StravaActivityDetail) =>
  (d.runHeartCost?.points ?? []).map(point => ({ ...point, d: point.elapsedS }))

export type RunHeartCostWindow = '30' | '300'

export const runHeartCostValue = (
  point: RunHeartCostPoint,
  window: RunHeartCostWindow,
): number | null =>
  window === '30'
    ? runMetersPerBeat(point.gapSpeed30sMps, point.heartRate30sBpm)
    : runMetersPerBeat(point.gapSpeed5mMps, point.heartRate5mBpm)

export const activityHeartRateTracePoints = (
  d: StravaActivityDetail,
): { d: number; elapsedS: number; heartRate: number | null }[] => {
  const route = d.route.map(point => ({
    d: point.d,
    elapsedS: point.elapsedS,
    heartRate: point.hr > 0 ? point.hr : null,
  }))
  if (route.filter(point => point.heartRate != null).length >= 2) return route
  return d.heartRateTrace.map(point => ({
    d: routeLessTraceDistance(d, point),
    elapsedS: point.elapsedS,
    heartRate: point.heartRate,
  }))
}

export const hasHeartRateTrace = (d: StravaActivityDetail): boolean =>
  activityHeartRateTracePoints(d).filter(point => point.heartRate != null).length >= 2

export interface ActivityPhysiologyTracePoint {
  d: number
  elapsedS: number
  stamina: number | null
  potentialStamina: number | null
  performanceCondition?: number | null
}

export const activityPhysiologyTracePoints = (
  d: StravaActivityDetail,
  metric: 'stamina' | 'performanceCondition',
): ActivityPhysiologyTracePoint[] => {
  if (d.route.filter(point => point[metric] != null).length >= 2) return d.route
  return (d.swimPhysiology?.points ?? d.heartRatePhysiology?.points ?? []).map(point => ({
    ...point,
    d: activityTraceUsesElapsedAxis(d) ? point.elapsedS : point.distanceKm,
  }))
}

export const activityZonePercentages = (values: readonly number[] | null | undefined): number[] => {
  if (!values) return []
  if (values.some(value => !Number.isFinite(value) || value < 0)) return []
  const total = values.reduce((sum, value) => sum + value, 0)
  return total > 0 ? values.map(value => (value / total) * 100) : values.map(() => 0)
}

export const activityPowerDistributionPercentages = (
  values: readonly number[] | null | undefined,
): number[] => (values && values.length >= 2 ? activityZonePercentages(values) : [])
