import type { StravaActivityDetail } from '../../plugins/stores/strava'
import { comparisonSwimMetricSegments } from './comparison-metrics'

type ActivityComparisonGeographicPoint = { lat: number; lng: number; d: number }

const comparisonActivityRoutePoints = (
  activity: StravaActivityDetail,
): StravaActivityDetail['route'] =>
  activity.route.filter(point => Number.isFinite(point.d) && point.d >= 0).sort((a, b) => a.d - b.d)

const comparisonSwimDistanceKm = (activity: StravaActivityDetail): number =>
  activity.sport === 'swim'
    ? activity.swimIntervals.reduce(
        (max, interval) =>
          Number.isFinite(interval.cumulativeDistanceM)
            ? Math.max(max, interval.cumulativeDistanceM / 1_000)
            : max,
        Number.isFinite(activity.distanceKm) && activity.distanceKm > 0 ? activity.distanceKm : 0,
      )
    : 0

export const comparisonMaxDistanceKm = (activities: readonly StravaActivityDetail[]): number =>
  activities.reduce((max, activity) => {
    const route = comparisonActivityRoutePoints(activity)
    return Math.max(max, route[route.length - 1]?.d ?? 0, comparisonSwimDistanceKm(activity))
  }, 0)

const comparisonRouteCapable = (activity: StravaActivityDetail): boolean => {
  const route = comparisonActivityRoutePoints(activity)
  return route.length >= 2 && route[route.length - 1].d > route[0].d
}

const splitComparisonGeographicPoints = (
  points: readonly ActivityComparisonGeographicPoint[],
): ActivityComparisonGeographicPoint[][] => {
  const segments: ActivityComparisonGeographicPoint[][] = []
  let segment: ActivityComparisonGeographicPoint[] = []
  const flush = (): void => {
    if (segment.length >= 2) segments.push(segment)
    segment = []
  }
  for (const point of points) {
    if (!Number.isFinite(point.lat) || !Number.isFinite(point.lng) || !Number.isFinite(point.d)) {
      flush()
      continue
    }
    if (segment.length > 0 && point.d < segment[segment.length - 1].d) flush()
    segment.push({ lat: point.lat, lng: point.lng, d: point.d })
  }
  flush()
  return segments
}

export const comparisonGeographicSegments = (
  activity: StravaActivityDetail,
): ActivityComparisonGeographicPoint[][] => {
  const mapSegments = activity.mapRoute.flatMap(segment => splitComparisonGeographicPoints(segment))
  if (mapSegments.length > 0) return mapSegments
  return splitComparisonGeographicPoints(activity.route)
}

export const activityComparisonEligible = (activity: StravaActivityDetail): boolean => {
  const swimTelemetry =
    activity.sport === 'swim' &&
    [
      comparisonSwimMetricSegments(activity, 'swim-pace'),
      comparisonSwimMetricSegments(activity, 'stroke-rate'),
    ]
      .flat()
      .some(segment => segment.length >= 2)
  return (
    swimTelemetry ||
    (comparisonRouteCapable(activity) && comparisonGeographicSegments(activity).length > 0)
  )
}
