import type {
  ActivityAnalysisKind,
  ActivityAnalysisRange,
  StravaActivityDetail,
} from '../../plugins/stores/strava'

export type ActivitySelectionSummary = {
  startElapsedS: number
  endElapsedS: number
  startDistanceKm: number
  endDistanceKm: number
  durationS: number
  distanceKm: number
  elevationGainM: number | null
  averageSpeedKph: number | null
  averageHeartRate: number | null
  averageWatts: number | null
  averageCadence: number | null
  averageRespirationRate: number | null
  averageTemperatureC: number | null
}

type WeightedRouteMetric = { total: number; durationS: number }

const addWeightedRouteMetric = (
  metric: WeightedRouteMetric,
  previous: number | null,
  next: number | null,
  durationS: number,
): void => {
  if (previous == null || next == null || durationS <= 0) return
  metric.total += ((previous + next) / 2) * durationS
  metric.durationS += durationS
}

const weightedRouteValue = (metric: WeightedRouteMetric): number | null =>
  metric.durationS > 0 ? metric.total / metric.durationS : null

export const activitySelectionSummary = (
  route: StravaActivityDetail['route'],
  anchorIndex: number,
  focusIndex: number,
): ActivitySelectionSummary | null => {
  if (route.length < 2) return null
  const first = Math.max(0, Math.min(route.length - 1, Math.round(anchorIndex)))
  const last = Math.max(0, Math.min(route.length - 1, Math.round(focusIndex)))
  const startIndex = Math.min(first, last)
  const endIndex = Math.max(first, last)
  if (startIndex === endIndex) return null
  const start = route[startIndex]
  const end = route[endIndex]
  const durationS = end.elapsedS - start.elapsedS
  const distanceKm = end.d - start.d
  if (durationS <= 0 || distanceKm <= 0) return null

  let elevationGainM = 0
  const heartRate: WeightedRouteMetric = { total: 0, durationS: 0 }
  const watts: WeightedRouteMetric = { total: 0, durationS: 0 }
  const cadence: WeightedRouteMetric = { total: 0, durationS: 0 }
  const respiration: WeightedRouteMetric = { total: 0, durationS: 0 }
  const temperature: WeightedRouteMetric = { total: 0, durationS: 0 }
  let hasPower = false
  let hasCadence = false
  for (let index = startIndex; index <= endIndex && (!hasPower || !hasCadence); index++) {
    hasPower ||= route[index].w > 0
    hasCadence ||= route[index].cad > 0
  }

  for (let index = startIndex + 1; index <= endIndex; index++) {
    const previous = route[index - 1]
    const next = route[index]
    const elapsedS = next.elapsedS - previous.elapsedS
    if (elapsedS <= 0) continue
    elevationGainM += Math.max(0, next.alt - previous.alt)
    addWeightedRouteMetric(
      heartRate,
      previous.hr > 0 ? previous.hr : null,
      next.hr > 0 ? next.hr : null,
      elapsedS,
    )
    addWeightedRouteMetric(watts, hasPower ? previous.w : null, hasPower ? next.w : null, elapsedS)
    addWeightedRouteMetric(
      cadence,
      hasCadence ? previous.cad : null,
      hasCadence ? next.cad : null,
      elapsedS,
    )
    addWeightedRouteMetric(respiration, previous.resp, next.resp, elapsedS)
    addWeightedRouteMetric(temperature, previous.tempC, next.tempC, elapsedS)
  }

  return {
    startElapsedS: start.elapsedS,
    endElapsedS: end.elapsedS,
    startDistanceKm: start.d,
    endDistanceKm: end.d,
    durationS,
    distanceKm,
    elevationGainM,
    averageSpeedKph: (distanceKm / durationS) * 3600,
    averageHeartRate: weightedRouteValue(heartRate),
    averageWatts: weightedRouteValue(watts),
    averageCadence: weightedRouteValue(cadence),
    averageRespirationRate: weightedRouteValue(respiration),
    averageTemperatureC: weightedRouteValue(temperature),
  }
}

export const ANALYSIS_KIND_ORDER: ActivityAnalysisKind[] = ['lap', 'segment', 'climb']

export const validAnalysisRanges = (d: StravaActivityDetail): ActivityAnalysisRange[] => {
  const seen = new Set<string>()
  return d.analysisRanges.filter(range => {
    const timedLap = d.sport === 'sauna' && range.kind === 'lap' && range.distanceKm === 0
    const key = `${range.kind}:${range.id}`
    if (
      seen.has(key) ||
      range.id.trim().length === 0 ||
      range.label.trim().length === 0 ||
      !Number.isFinite(range.startElapsedS) ||
      !Number.isFinite(range.endElapsedS) ||
      range.startElapsedS < 0 ||
      range.endElapsedS <= range.startElapsedS ||
      !Number.isFinite(range.startDistanceKm) ||
      !Number.isFinite(range.endDistanceKm) ||
      range.startDistanceKm < 0 ||
      (timedLap
        ? range.endDistanceKm !== range.startDistanceKm
        : range.endDistanceKm <= range.startDistanceKm) ||
      !Number.isFinite(range.durationS) ||
      range.durationS <= 0 ||
      !Number.isFinite(range.distanceKm) ||
      (!timedLap && range.distanceKm <= 0)
    )
      return false
    seen.add(key)
    return true
  })
}

export const hasAnalysisWorkspace = (d: StravaActivityDetail): boolean =>
  d.route.length >= 2 &&
  d.route.every(
    point =>
      Number.isFinite(point.d) &&
      Number.isFinite(point.elapsedS) &&
      Number.isFinite(point.speedKph) &&
      Number.isFinite(point.lat) &&
      Number.isFinite(point.lng),
  )

export const activityAnalysisAvailable = (d: StravaActivityDetail): boolean =>
  d.route.length >= 2 ||
  d.heartRateTrace.filter(
    point =>
      point.heartRate != null ||
      point.coreTemperatureC != null ||
      point.skinTemperatureC != null ||
      point.heatStrainIndex != null,
  ).length >= 2 ||
  (d.sport === 'swim' &&
    d.swimIntervals.some(
      interval =>
        interval.endElapsedS > interval.startElapsedS &&
        interval.distanceM > 0 &&
        (interval.paceSPer100m != null || interval.strokeRateSpm != null),
    ))
