import type { RawStravaActivity, StravaStreams } from './strava'
import {
  buildRouteCandidates,
  clusterRouteCandidates,
  matchedRouteMethod,
  type MatchedRouteConfig,
  type MatchedRouteMethod,
  type RouteCandidate,
  type RouteGroup,
} from './matched-routes'

const MIN_ROUTE_DISTANCE_M = 1_000
const ROUTE_SAMPLE_SPACING_M = 50
const MAX_SAMPLE_DISTANCE_M = 50
const MIN_DISTANCE_RATIO = 0.8
const MIN_ORDERED_COVERAGE = 0.85
const RUN_TYPES = new Set(['Run', 'TrailRun', 'VirtualRun'])
const ROUTE_CONFIG: MatchedRouteConfig = {
  minimumRouteDistanceM: MIN_ROUTE_DISTANCE_M,
  sampleSpacingM: ROUTE_SAMPLE_SPACING_M,
  maximumSampleDistanceM: MAX_SAMPLE_DISTANCE_M,
  minimumDistanceRatio: MIN_DISTANCE_RATIO,
  minimumOrderedCoverage: MIN_ORDERED_COVERAGE,
}

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
  method: MatchedRouteMethod
}

const roundTo = (value: number, digits: number): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
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
  method: matchedRouteMethod(ROUTE_CONFIG),
})

export const buildMatchedRuns = (
  activities: readonly RawStravaActivity[],
  streams: Readonly<Record<string, StravaStreams>>,
): MatchedRunsBlock => {
  const runActivities = activities.filter(
    activity =>
      RUN_TYPES.has(activity.sportType) &&
      activity.distance >= MIN_ROUTE_DISTANCE_M &&
      activity.movingTime > 0,
  )
  const candidates = buildRouteCandidates(runActivities, streams, ROUTE_CONFIG).sort(
    (left, right) =>
      left.activity.startDateLocal.localeCompare(right.activity.startDateLocal) ||
      left.activity.id - right.activity.id,
  )

  const routeGroups = clusterRouteCandidates(candidates, ROUTE_CONFIG)

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
