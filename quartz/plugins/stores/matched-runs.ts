import type { RawStravaActivity, StravaStreams } from './strava'
import {
  mapDistanceMeters,
  mapRoutePointAtDistance,
  rawMapRouteSegments,
  type MapRoutePoint,
} from '../../util/triathlon-map-route'

const MIN_ROUTE_DISTANCE_M = 1_000
const ROUTE_SAMPLE_SPACING_M = 50
const MAX_SAMPLE_DISTANCE_M = 50
const MIN_DISTANCE_RATIO = 0.8
const MIN_ORDERED_COVERAGE = 0.85
const RUN_TYPES = new Set(['Run', 'TrailRun', 'VirtualRun'])

export interface MatchedRunEffort {
  id: number
  date: string
  name: string
  distanceKm: number
  movingTimeS: number
  paceSPerKm: number
  relativeEffort: number | null
}

export interface MatchedRunGroup {
  id: string
  routeDistanceKm: number
  averagePaceSPerKm: number
  fastestPaceSPerKm: number
  slowestPaceSPerKm: number
  efforts: MatchedRunEffort[]
}

export interface MatchedRunsBlock {
  candidateRunCount: number
  matchedActivityCount: number
  groups: MatchedRunGroup[]
  method: {
    source: 'gps'
    sampleSpacingM: number
    maximumSampleDistanceM: number
    minimumDistanceRatio: number
    minimumOrderedCoverage: number
  }
}

interface RouteCandidate {
  activity: RawStravaActivity
  route: MapRoutePoint[]
}

interface RouteMatchScore {
  distanceRatio: number
  orderedCoverage: number
}

interface RouteGroup {
  members: RouteCandidate[]
}

const roundTo = (value: number, digits: number): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

const routeSamples = (streams: StravaStreams | undefined): MapRoutePoint[] | null => {
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
  if (routeDistanceM < MIN_ROUTE_DISTANCE_M) return null

  const sampled: MapRoutePoint[] = []
  for (let distanceM = 0; distanceM < routeDistanceM; distanceM += ROUTE_SAMPLE_SPACING_M) {
    const point = mapRoutePointAtDistance(segments, firstPoint.d + distanceM / 1_000)
    if (point) sampled.push(point)
  }
  sampled.push({ ...lastPoint })
  return sampled.length >= 2 ? sampled : null
}

const orderedRouteMatches = (left: MapRoutePoint[], right: MapRoutePoint[]): number => {
  let previous = new Uint16Array(right.length + 1)
  let current = new Uint16Array(right.length + 1)
  for (let leftIndex = 0; leftIndex < left.length; leftIndex++) {
    for (let rightIndex = 0; rightIndex < right.length; rightIndex++) {
      current[rightIndex + 1] =
        mapDistanceMeters(left[leftIndex], right[rightIndex]) <= MAX_SAMPLE_DISTANCE_M
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

const routeMatchScore = (left: RouteCandidate, right: RouteCandidate): RouteMatchScore | null => {
  const distanceRatio =
    Math.min(left.activity.distance, right.activity.distance) /
    Math.max(left.activity.distance, right.activity.distance)
  if (!Number.isFinite(distanceRatio) || distanceRatio < MIN_DISTANCE_RATIO) return null

  const orderedCoverage =
    orderedRouteMatches(left.route, right.route) / Math.min(left.route.length, right.route.length)
  return orderedCoverage >= MIN_ORDERED_COVERAGE ? { distanceRatio, orderedCoverage } : null
}

const completeLinkScore = (candidate: RouteCandidate, group: RouteGroup): number | null => {
  const scores = group.members.map(member => routeMatchScore(candidate, member))
  if (!scores.every((score): score is RouteMatchScore => score !== null)) return null
  return Math.max(
    ...scores.map(score => (1 - score.orderedCoverage) * 1_000 + (1 - score.distanceRatio) * 100),
  )
}

const median = (values: number[]): number => {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

const effortFromCandidate = ({ activity }: RouteCandidate): MatchedRunEffort => ({
  id: activity.id,
  date: activity.startDateLocal.slice(0, 10),
  name: activity.name || 'Run',
  distanceKm: roundTo(activity.distance / 1_000, 2),
  movingTimeS: activity.movingTime,
  paceSPerKm: roundTo(activity.movingTime / (activity.distance / 1_000), 1),
  relativeEffort:
    activity.sufferScore != null && Number.isFinite(activity.sufferScore)
      ? activity.sufferScore
      : null,
})

const matchedRunGroup = (group: RouteGroup): MatchedRunGroup => {
  const efforts = group.members.map(effortFromCandidate)
  const paces = efforts.map(effort => effort.paceSPerKm)
  const exactPaces = group.members.map(
    member => member.activity.movingTime / (member.activity.distance / 1_000),
  )
  return {
    id: String(group.members[0].activity.id),
    routeDistanceKm: roundTo(
      median(group.members.map(member => member.activity.distance / 1_000)),
      2,
    ),
    averagePaceSPerKm: roundTo(
      exactPaces.reduce((total, pace) => total + pace, 0) / exactPaces.length,
      3,
    ),
    fastestPaceSPerKm: Math.min(...paces),
    slowestPaceSPerKm: Math.max(...paces),
    efforts,
  }
}

export const emptyMatchedRuns = (): MatchedRunsBlock => ({
  candidateRunCount: 0,
  matchedActivityCount: 0,
  groups: [],
  method: {
    source: 'gps',
    sampleSpacingM: ROUTE_SAMPLE_SPACING_M,
    maximumSampleDistanceM: MAX_SAMPLE_DISTANCE_M,
    minimumDistanceRatio: MIN_DISTANCE_RATIO,
    minimumOrderedCoverage: MIN_ORDERED_COVERAGE,
  },
})

export const buildMatchedRuns = (
  activities: readonly RawStravaActivity[],
  streams: Readonly<Record<string, StravaStreams>>,
): MatchedRunsBlock => {
  const candidates = activities
    .filter(
      activity =>
        RUN_TYPES.has(activity.sportType) &&
        activity.distance >= MIN_ROUTE_DISTANCE_M &&
        activity.movingTime > 0,
    )
    .flatMap(activity => {
      const route = routeSamples(streams[String(activity.id)])
      return route ? [{ activity, route }] : []
    })
    .sort(
      (left, right) =>
        left.activity.startDateLocal.localeCompare(right.activity.startDateLocal) ||
        left.activity.id - right.activity.id,
    )

  const routeGroups: RouteGroup[] = []
  for (const candidate of candidates) {
    let bestGroup: RouteGroup | null = null
    let bestScore = Infinity
    for (const group of routeGroups) {
      const score = completeLinkScore(candidate, group)
      if (score != null && score < bestScore) {
        bestGroup = group
        bestScore = score
      }
    }
    if (bestGroup) bestGroup.members.push(candidate)
    else routeGroups.push({ members: [candidate] })
  }

  const groups = routeGroups
    .filter(group => group.members.length >= 2)
    .map(matchedRunGroup)
    .sort((left, right) => {
      const latestLeft = left.efforts[left.efforts.length - 1]
      const latestRight = right.efforts[right.efforts.length - 1]
      return (
        latestRight.date.localeCompare(latestLeft.date) ||
        right.efforts.length - left.efforts.length ||
        left.id.localeCompare(right.id)
      )
    })

  return {
    ...emptyMatchedRuns(),
    candidateRunCount: candidates.length,
    matchedActivityCount: groups.reduce((total, group) => total + group.efforts.length, 0),
    groups,
  }
}
