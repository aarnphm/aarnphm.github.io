import type { RawStravaActivity, StravaStreams } from './strava'
import {
  buildRouteCandidates,
  clusterRouteCandidates,
  matchedRouteMethod,
  type MatchedRouteConfig,
} from './matched-routes'

const MIN_RIDE_DISTANCE_M = 3_000
const RIDE_TYPES = new Set([
  'Ride',
  'VirtualRide',
  'MountainBikeRide',
  'GravelRide',
  'EBikeRide',
  'EMountainBikeRide',
  'Velomobile',
])
const ROUTE_CONFIG: MatchedRouteConfig = {
  minimumRouteDistanceM: MIN_RIDE_DISTANCE_M,
  sampleSpacingM: 200,
  maximumSampleDistanceM: 100,
  minimumDistanceRatio: 0.8,
  minimumOrderedCoverage: 0.82,
}

interface CharacteristicConfig {
  minimumDistanceRatio: number
  minimumElevationGainRatio: number
  minimumClimbingDensityRatio: number
  minimumAveragePowerRatio: number
  minimumNormalizedPowerRatio: number
}

const CHARACTERISTIC_CONFIG: CharacteristicConfig = {
  minimumDistanceRatio: 0.75,
  minimumElevationGainRatio: 0.65,
  minimumClimbingDensityRatio: 0.7,
  minimumAveragePowerRatio: 0.85,
  minimumNormalizedPowerRatio: 0.85,
}

export type MatchedRideKind = 'route' | 'characteristics'
export type MatchedRidePowerMetric = 'normalized' | 'average'
export type MatchedRidePowerSource = 'device' | 'estimate'

export interface MatchedRideEffort {
  id: number
  date: string
  name: string
  distanceKm: number
  movingTimeS: number
  elevationM: number
  climbingMPerKm: number
  averageWatts: number
  normalizedWatts: number | null
  powerSource: MatchedRidePowerSource
}

export interface MatchedRideGroup {
  id: string
  match: MatchedRideKind
  powerMetric: MatchedRidePowerMetric
  averagePowerWatts: number
  highestPowerWatts: number
  lowestPowerWatts: number
  averageDistanceKm: number
  averageElevationM: number
  averageClimbingMPerKm: number
  efforts: MatchedRideEffort[]
}

export interface MatchedRidesBlock {
  candidateRideCount: number
  matchedActivityCount: number
  routeMatchedActivityCount: number
  characteristicMatchedActivityCount: number
  groups: MatchedRideGroup[]
  method: {
    route: ReturnType<typeof matchedRouteMethod>
    characteristics: CharacteristicConfig & {
      source: 'activity-summary'
      powerSourceMustMatch: true
    }
  }
}

interface CharacteristicCandidate {
  activity: RawStravaActivity
  elevationM: number
  climbingMPerKm: number
  averageWatts: number
  normalizedWatts: number | null
  powerSource: MatchedRidePowerSource
}

interface CharacteristicGroup {
  members: CharacteristicCandidate[]
}

interface CharacteristicMatchScore {
  distanceRatio: number
  elevationGainRatio: number
  climbingDensityRatio: number
  averagePowerRatio: number
  normalizedPowerRatio: number
}

const roundTo = (value: number, digits: number): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

const average = (values: readonly number[]): number =>
  values.reduce((total, value) => total + value, 0) / values.length

const positive = (value: number | undefined): number | null =>
  value != null && Number.isFinite(value) && value > 0 ? value : null

const ratio = (left: number, right: number, floor: number): number => {
  const boundedLeft = Math.max(floor, left)
  const boundedRight = Math.max(floor, right)
  return Math.min(boundedLeft, boundedRight) / Math.max(boundedLeft, boundedRight)
}

const rideActivities = (activities: readonly RawStravaActivity[]): RawStravaActivity[] =>
  activities
    .filter(
      activity =>
        RIDE_TYPES.has(activity.sportType) &&
        activity.distance >= MIN_RIDE_DISTANCE_M &&
        activity.movingTime > 0 &&
        positive(activity.averageWatts) != null,
    )
    .sort(
      (left, right) =>
        left.startDateLocal.localeCompare(right.startDateLocal) || left.id - right.id,
    )

const characteristicCandidate = (activity: RawStravaActivity): CharacteristicCandidate | null => {
  const averageWatts = positive(activity.averageWatts)
  if (averageWatts == null) return null
  const distanceKm = activity.distance / 1_000
  const elevationM = Math.max(0, activity.totalElevationGain)
  return {
    activity,
    elevationM,
    climbingMPerKm: elevationM / distanceKm,
    averageWatts,
    normalizedWatts: positive(activity.weightedAverageWatts),
    powerSource: activity.deviceWatts === true ? 'device' : 'estimate',
  }
}

const characteristicMatchScore = (
  left: CharacteristicCandidate,
  right: CharacteristicCandidate,
): CharacteristicMatchScore | null => {
  if (left.powerSource !== right.powerSource) return null
  const distanceRatio = ratio(left.activity.distance, right.activity.distance, MIN_RIDE_DISTANCE_M)
  const elevationGainRatio = ratio(left.elevationM, right.elevationM, 25)
  const climbingDensityRatio = ratio(left.climbingMPerKm, right.climbingMPerKm, 2)
  const averagePowerRatio = ratio(left.averageWatts, right.averageWatts, 50)
  const normalizedPowerRatio =
    left.normalizedWatts == null && right.normalizedWatts == null
      ? 1
      : left.normalizedWatts != null && right.normalizedWatts != null
        ? ratio(left.normalizedWatts, right.normalizedWatts, 50)
        : 0
  if (
    distanceRatio < CHARACTERISTIC_CONFIG.minimumDistanceRatio ||
    elevationGainRatio < CHARACTERISTIC_CONFIG.minimumElevationGainRatio ||
    climbingDensityRatio < CHARACTERISTIC_CONFIG.minimumClimbingDensityRatio ||
    averagePowerRatio < CHARACTERISTIC_CONFIG.minimumAveragePowerRatio ||
    normalizedPowerRatio < CHARACTERISTIC_CONFIG.minimumNormalizedPowerRatio
  )
    return null
  return {
    distanceRatio,
    elevationGainRatio,
    climbingDensityRatio,
    averagePowerRatio,
    normalizedPowerRatio,
  }
}

const characteristicScore = (score: CharacteristicMatchScore): number =>
  (1 - score.distanceRatio) * 3 +
  (1 - score.elevationGainRatio) * 2 +
  (1 - score.climbingDensityRatio) * 2 +
  (1 - score.averagePowerRatio) * 3 +
  (1 - score.normalizedPowerRatio) * 2

const completeLinkCharacteristicScore = (
  candidate: CharacteristicCandidate,
  group: CharacteristicGroup,
): number | null => {
  const scores = group.members.map(member => characteristicMatchScore(candidate, member))
  if (!scores.every((score): score is CharacteristicMatchScore => score !== null)) return null
  return Math.max(...scores.map(characteristicScore))
}

const clusterCharacteristics = (
  candidates: readonly CharacteristicCandidate[],
): CharacteristicGroup[] => {
  const groups: CharacteristicGroup[] = []
  for (const candidate of candidates) {
    let bestGroup: CharacteristicGroup | null = null
    let bestScore = Infinity
    for (const group of groups) {
      const score = completeLinkCharacteristicScore(candidate, group)
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

const effortFromActivity = (activity: RawStravaActivity): MatchedRideEffort => {
  const distanceKm = activity.distance / 1_000
  const elevationM = Math.max(0, activity.totalElevationGain)
  return {
    id: activity.id,
    date: activity.startDateLocal.slice(0, 10),
    name: activity.name || 'Ride',
    distanceKm: roundTo(distanceKm, 2),
    movingTimeS: activity.movingTime,
    elevationM: roundTo(elevationM, 1),
    climbingMPerKm: roundTo(elevationM / distanceKm, 2),
    averageWatts: roundTo(positive(activity.averageWatts) ?? 0, 1),
    normalizedWatts:
      activity.weightedAverageWatts == null ? null : roundTo(activity.weightedAverageWatts, 1),
    powerSource: activity.deviceWatts === true ? 'device' : 'estimate',
  }
}

const matchedRideGroup = (
  match: MatchedRideKind,
  activities: readonly RawStravaActivity[],
): MatchedRideGroup => {
  const efforts = activities.map(effortFromActivity)
  const powerMetric: MatchedRidePowerMetric = efforts.every(
    effort => effort.normalizedWatts != null,
  )
    ? 'normalized'
    : 'average'
  const powers = efforts.map(effort =>
    powerMetric === 'normalized'
      ? (effort.normalizedWatts ?? effort.averageWatts)
      : effort.averageWatts,
  )
  return {
    id: `${match}:${efforts[0].id}`,
    match,
    powerMetric,
    averagePowerWatts: roundTo(average(powers), 3),
    highestPowerWatts: Math.max(...powers),
    lowestPowerWatts: Math.min(...powers),
    averageDistanceKm: roundTo(average(efforts.map(effort => effort.distanceKm)), 2),
    averageElevationM: roundTo(average(efforts.map(effort => effort.elevationM)), 1),
    averageClimbingMPerKm: roundTo(average(efforts.map(effort => effort.climbingMPerKm)), 2),
    efforts,
  }
}

const sortGroups = (left: MatchedRideGroup, right: MatchedRideGroup): number => {
  const latestLeft = left.efforts[left.efforts.length - 1]
  const latestRight = right.efforts[right.efforts.length - 1]
  return (
    latestRight.date.localeCompare(latestLeft.date) ||
    right.efforts.length - left.efforts.length ||
    left.id.localeCompare(right.id)
  )
}

export const emptyMatchedRides = (): MatchedRidesBlock => ({
  candidateRideCount: 0,
  matchedActivityCount: 0,
  routeMatchedActivityCount: 0,
  characteristicMatchedActivityCount: 0,
  groups: [],
  method: {
    route: matchedRouteMethod(ROUTE_CONFIG),
    characteristics: {
      source: 'activity-summary',
      ...CHARACTERISTIC_CONFIG,
      powerSourceMustMatch: true,
    },
  },
})

export const buildMatchedRides = (
  activities: readonly RawStravaActivity[],
  streams: Readonly<Record<string, StravaStreams>>,
): MatchedRidesBlock => {
  const rides = rideActivities(activities)
  const routeCandidates = buildRouteCandidates(rides, streams, ROUTE_CONFIG)
  const routeGroups = clusterRouteCandidates(routeCandidates, ROUTE_CONFIG).filter(
    group => group.members.length >= 2,
  )
  const routeMatchedIds = new Set(
    routeGroups.flatMap(group => group.members.map(member => member.activity.id)),
  )
  const characteristicGroups = clusterCharacteristics(
    rides
      .filter(activity => !routeMatchedIds.has(activity.id))
      .flatMap(activity => {
        const candidate = characteristicCandidate(activity)
        return candidate ? [candidate] : []
      }),
  ).filter(group => group.members.length >= 2)

  const routeMatches = routeGroups.map(group =>
    matchedRideGroup(
      'route',
      group.members.map(member => member.activity),
    ),
  )
  const characteristicMatches = characteristicGroups.map(group =>
    matchedRideGroup(
      'characteristics',
      group.members.map(member => member.activity),
    ),
  )
  const groups = [...routeMatches, ...characteristicMatches].sort(sortGroups)

  return {
    ...emptyMatchedRides(),
    candidateRideCount: rides.length,
    matchedActivityCount: groups.reduce((total, group) => total + group.efforts.length, 0),
    routeMatchedActivityCount: routeMatches.reduce(
      (total, group) => total + group.efforts.length,
      0,
    ),
    characteristicMatchedActivityCount: characteristicMatches.reduce(
      (total, group) => total + group.efforts.length,
      0,
    ),
    groups,
  }
}
