import type { RawStravaActivity, StravaStreams } from './strava'
import {
  mapDistanceMeters,
  mapRoutePointAtDistance,
  rawMapRouteSegments,
  type MapRoutePoint,
} from '../../util/triathlon-map-route'

export interface MatchedRouteConfig {
  minimumRouteDistanceM: number
  sampleSpacingM: number
  maximumSampleDistanceM: number
  minimumDistanceRatio: number
  minimumOrderedCoverage: number
}

export interface MatchedRouteMethod {
  source: 'gps'
  sampleSpacingM: number
  maximumSampleDistanceM: number
  minimumDistanceRatio: number
  minimumOrderedCoverage: number
}

export interface RouteCandidate {
  activity: RawStravaActivity
  route: MapRoutePoint[]
}

export interface RouteGroup {
  members: RouteCandidate[]
}

interface RouteMatchScore {
  distanceRatio: number
  orderedCoverage: number
}

export const matchedRouteMethod = (config: MatchedRouteConfig): MatchedRouteMethod => ({
  source: 'gps',
  sampleSpacingM: config.sampleSpacingM,
  maximumSampleDistanceM: config.maximumSampleDistanceM,
  minimumDistanceRatio: config.minimumDistanceRatio,
  minimumOrderedCoverage: config.minimumOrderedCoverage,
})

const routeSamples = (
  streams: StravaStreams | undefined,
  config: MatchedRouteConfig,
): MapRoutePoint[] | null => {
  if (!streams) return null
  const pointCount = Math.min(streams.latlng.length, streams.distance.length)
  if (pointCount < 8) return null
  const elapsedS = streams.time ?? Array.from({ length: pointCount }, (_, index) => index)
  const segments = rawMapRouteSegments(
    streams.latlng,
    streams.distance,
    elapsedS,
    0,
    pointCount - 1,
  )
  const firstPoint = segments[0]?.[0]
  const lastSegment = segments[segments.length - 1]
  const lastPoint = lastSegment?.[lastSegment.length - 1]
  if (!firstPoint || !lastPoint) return null
  const routeDistanceM = (lastPoint.d - firstPoint.d) * 1_000
  if (routeDistanceM < config.minimumRouteDistanceM) return null

  const sampled: MapRoutePoint[] = []
  for (let distanceM = 0; distanceM < routeDistanceM; distanceM += config.sampleSpacingM) {
    const point = mapRoutePointAtDistance(segments, firstPoint.d + distanceM / 1_000)
    if (point) sampled.push(point)
  }
  sampled.push({ ...lastPoint })
  return sampled.length >= 2 ? sampled : null
}

export const buildRouteCandidates = (
  activities: readonly RawStravaActivity[],
  streams: Readonly<Record<string, StravaStreams>>,
  config: MatchedRouteConfig,
): RouteCandidate[] =>
  activities.flatMap(activity => {
    const route = routeSamples(streams[String(activity.id)], config)
    return route ? [{ activity, route }] : []
  })

const orderedRouteMatches = (
  left: MapRoutePoint[],
  right: MapRoutePoint[],
  maximumSampleDistanceM: number,
): number => {
  let previous = new Uint16Array(right.length + 1)
  let current = new Uint16Array(right.length + 1)
  for (let leftIndex = 0; leftIndex < left.length; leftIndex++) {
    for (let rightIndex = 0; rightIndex < right.length; rightIndex++) {
      current[rightIndex + 1] =
        mapDistanceMeters(left[leftIndex], right[rightIndex]) <= maximumSampleDistanceM
          ? previous[rightIndex] + 1
          : Math.max(previous[rightIndex + 1], current[rightIndex])
    }
    const swap = previous
    previous = current
    current = swap
    current.fill(0)
  }
  return previous[right.length]
}

const routeMatchScore = (
  left: RouteCandidate,
  right: RouteCandidate,
  config: MatchedRouteConfig,
): RouteMatchScore | null => {
  const distanceRatio =
    Math.min(left.activity.distance, right.activity.distance) /
    Math.max(left.activity.distance, right.activity.distance)
  if (!Number.isFinite(distanceRatio) || distanceRatio < config.minimumDistanceRatio) return null

  const orderedCoverage =
    orderedRouteMatches(left.route, right.route, config.maximumSampleDistanceM) /
    Math.min(left.route.length, right.route.length)
  return orderedCoverage >= config.minimumOrderedCoverage
    ? { distanceRatio, orderedCoverage }
    : null
}

const completeLinkScore = (
  candidate: RouteCandidate,
  group: RouteGroup,
  config: MatchedRouteConfig,
): number | null => {
  const scores = group.members.map(member => routeMatchScore(candidate, member, config))
  if (!scores.every((score): score is RouteMatchScore => score !== null)) return null
  return Math.max(
    ...scores.map(score => (1 - score.orderedCoverage) * 1_000 + (1 - score.distanceRatio) * 100),
  )
}

export const clusterRouteCandidates = (
  candidates: readonly RouteCandidate[],
  config: MatchedRouteConfig,
): RouteGroup[] => {
  const groups: RouteGroup[] = []
  for (const candidate of candidates) {
    let bestGroup: RouteGroup | null = null
    let bestScore = Infinity
    for (const group of groups) {
      const score = completeLinkScore(candidate, group, config)
      if (score != null && score < bestScore) {
        bestGroup = group
        bestScore = score
      }
    }
    if (bestGroup) bestGroup.members.push(candidate)
    else groups.push({ members: [candidate] })
  }
  return groups
}
