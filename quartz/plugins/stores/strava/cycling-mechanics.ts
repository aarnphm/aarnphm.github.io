import type {
  GarminActivityMatch,
  GarminCache,
  GarminCyclingDynamics,
  GarminGearShift,
  GarminRiderPosition,
  GarminStreams,
} from '../garmin'
import type { WahooActivityMatch, WahooCyclingDynamics, WahooData, WahooStreams } from '../wahoo'
import { DEFAULT_GEAR_CASSETTE } from '../../../util/triathlon-gear-ratio'
import {
  WAHOO_TELEMETRY_MAX_DISTANCE_S,
  type RawStravaActivity,
  type StravaStreams,
} from './raw-and-kinds'
import {
  nearestElapsedIndex,
  round,
  timedStreamAlignment,
  type TimedStreamAlignment,
} from './stream-primitives'

export interface ActivityGearShift {
  elapsedS: number
  distanceKm: number
  frontGearNum: number
  frontTeeth: number
  rearGearNum: number
  rearTeeth: number
  // Present when the head unit's teeth were wrong and teeth come from the gear number instead.
  teethSource?: 'gear-number'
}

export interface ActivityRiderPositionChange {
  elapsedS: number
  distanceKm: number
  position: GarminRiderPosition
}

export interface ActivityCyclingDynamics {
  elapsedS: number[]
  distanceKm: number[]
  leftPedalSmoothness: (number | null)[]
  rightPedalSmoothness: (number | null)[]
  leftTorqueEffectiveness: (number | null)[]
  rightTorqueEffectiveness: (number | null)[]
  leftPowerPhaseStart: (number | null)[]
  leftPowerPhaseEnd: (number | null)[]
  rightPowerPhaseStart: (number | null)[]
  rightPowerPhaseEnd: (number | null)[]
  positionChanges: ActivityRiderPositionChange[]
  seatedTimeS: number | null
  standingTimeS: number | null
}

function wahooTimedStreamAlignment(streams: WahooStreams | undefined): TimedStreamAlignment | null {
  if (!streams || streams.time.length < 2 || streams.distance.length !== streams.time.length)
    return null
  const distance: number[] = []
  let previousDistance = 0
  for (const value of streams.distance) {
    if (typeof value === 'number' && Number.isFinite(value))
      previousDistance = Math.max(previousDistance, value)
    distance.push(previousDistance)
  }
  const normalized: GarminStreams = { time: [...streams.time], latlng: [], altitude: [], distance }
  return timedStreamAlignment(normalized)
}

function alignedDistanceAt(alignment: TimedStreamAlignment, elapsedS: number): number {
  const { time, distance } = alignment
  if (elapsedS <= time[0]) return distance[0]
  if (elapsedS >= time[time.length - 1]) return distance[distance.length - 1]
  let low = 0
  let high = time.length - 1
  while (low + 1 < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (time[middle] <= elapsedS) low = middle
    else high = middle
  }
  const span = time[high] - time[low]
  if (span <= 0) return distance[high]
  const fraction = (elapsedS - time[low]) / span
  return distance[low] + (distance[high] - distance[low]) * fraction
}

interface CyclingPowerTerrain {
  source: 'garmin' | 'strava'
  alignment: TimedStreamAlignment
  startOffsetS: number
}

export function cyclingPowerTerrain(
  activity: RawStravaActivity,
  strava: StravaStreams | undefined,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
): CyclingPowerTerrain | null {
  const fromStrava = timedStreamAlignment(strava)
  const fromGarmin = timedStreamAlignment(
    garminMatch ? garmin?.streams?.[garminMatch.activity.id] : undefined,
  )
  if (fromStrava && (!activity.sportType.startsWith('Virtual') || !fromGarmin))
    return { source: 'strava', alignment: fromStrava, startOffsetS: 0 }
  if (fromGarmin && garminMatch) {
    const startOffsetS =
      (Date.parse(garminMatch.activity.startDate) - Date.parse(activity.startDate)) / 1_000
    if (Number.isFinite(startOffsetS))
      return { source: 'garmin', alignment: fromGarmin, startOffsetS }
  }
  return fromStrava ? { source: 'strava', alignment: fromStrava, startOffsetS: 0 } : null
}

export function cyclingPowerStreamsOnTerrain(
  streams: Pick<WahooStreams, 'time' | 'watts' | 'distance' | 'altitude'> | undefined,
  startOffsetS: number,
  terrain: CyclingPowerTerrain | null,
): Pick<WahooStreams, 'time' | 'watts' | 'distance' | 'altitude'> | undefined {
  if (!streams || !terrain) return streams
  const { time, distance, streams: course } = terrain.alignment
  const projectedDistance: number[] = []
  const altitude: (number | null)[] = []
  let index = 0
  for (const sourceTime of streams.time) {
    const elapsedS = sourceTime + startOffsetS - terrain.startOffsetS
    while (index + 1 < time.length && time[index + 1] <= elapsedS) index++
    const next = Math.min(index + 1, time.length - 1)
    const fraction =
      time[next] > time[index]
        ? Math.max(0, Math.min(1, (elapsedS - time[index]) / (time[next] - time[index])))
        : 0
    projectedDistance.push(distance[index] + (distance[next] - distance[index]) * fraction)
    const left = course.altitude[index]
    const right = course.altitude[next]
    altitude.push(
      Number.isFinite(left) && Number.isFinite(right) ? left + (right - left) * fraction : null,
    )
  }
  return { time: streams.time, watts: streams.watts, distance: projectedDistance, altitude }
}

function projectedGearShifts(
  activity: RawStravaActivity,
  sourceStartDate: string,
  shifts: readonly GarminGearShift[],
  alignment: TimedStreamAlignment | null,
): ActivityGearShift[] {
  const activityStartMs = Date.parse(activity.startDate)
  const sourceStartMs = Date.parse(sourceStartDate)
  if (!Number.isFinite(activityStartMs) || !Number.isFinite(sourceStartMs)) return []
  const durationS = Math.max(activity.elapsedTime, activity.movingTime, 1)
  return shifts.flatMap(shift => {
    const timestampMs = Date.parse(shift.timestamp)
    if (!Number.isFinite(timestampMs)) return []
    const elapsedS = (timestampMs - activityStartMs) / 1000
    const sourceElapsedS = (timestampMs - sourceStartMs) / 1000
    if (sourceElapsedS < -60 || sourceElapsedS > durationS + 60) return []
    const distanceM = alignment
      ? alignedDistanceAt(alignment, sourceElapsedS)
      : (Math.min(durationS, Math.max(0, elapsedS)) / durationS) * activity.distance
    return [
      {
        elapsedS: round(Math.min(durationS, Math.max(0, elapsedS)), 3),
        distanceKm: round(Math.min(activity.distance, Math.max(0, distanceM)) / 1_000, 3),
        frontGearNum: shift.frontGearNum,
        frontTeeth: shift.frontTeeth,
        rearGearNum: shift.rearGearNum,
        rearTeeth: shift.rearTeeth,
      },
    ]
  })
}

// Until the Edge bike profile was set (first correct ride 2026-07-22), the Garmin wrote its
// default Di2 teeth while the bike ran 52/36 with an Ultegra 11-34. Di2 gear numbers stayed
// right (front 1 = small ring, rear 1 = largest cog), so teeth are read back from them.
const GARMIN_DEFAULT_PROFILE_UNTIL = Date.parse('2026-07-22T00:00:00Z')
const GARMIN_DEFAULT_PROFILES = [
  { front: [39, 53], rear: [23, 21, 19, 18, 17, 16, 15, 14, 13, 12, 11] },
  { front: [39, 53], rear: [26, 23, 21, 19, 17, 16, 15, 14, 13, 12, 11, 10] },
] as const
const FITTED_DRIVETRAIN = {
  front: [36, 52],
  rear: [...DEFAULT_GEAR_CASSETTE.cogs].reverse(),
} as const

function garminGearShiftsWithFittedTeeth(
  activity: RawStravaActivity,
  shifts: ActivityGearShift[],
): ActivityGearShift[] {
  if (!(Date.parse(activity.startDate) < GARMIN_DEFAULT_PROFILE_UNTIL)) return shifts
  const recordedDefault = GARMIN_DEFAULT_PROFILES.some(profile =>
    shifts.every(
      shift =>
        profile.front[shift.frontGearNum - 1] === shift.frontTeeth &&
        profile.rear[shift.rearGearNum - 1] === shift.rearTeeth,
    ),
  )
  if (!recordedDefault) return shifts
  return shifts.map(shift => {
    const frontTeeth = FITTED_DRIVETRAIN.front[shift.frontGearNum - 1]
    const rearTeeth = FITTED_DRIVETRAIN.rear[shift.rearGearNum - 1]
    return frontTeeth == null || rearTeeth == null
      ? shift
      : { ...shift, frontTeeth, rearTeeth, teethSource: 'gear-number' }
  })
}

export function activityGearShifts(
  activity: RawStravaActivity,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooData | null,
): ActivityGearShift[] {
  if (wahooMatch) {
    const shifts = projectedGearShifts(
      activity,
      wahooMatch.activity.startDate,
      wahoo?.gearShifts[wahooMatch.activity.id] ?? [],
      wahooTimedStreamAlignment(wahoo?.streams[wahooMatch.activity.id]),
    )
    if (shifts.length > 0) return shifts
  }
  if (garminMatch) {
    const projected = projectedGearShifts(
      activity,
      garminMatch.activity.startDate,
      garmin?.gearShifts?.[garminMatch.activity.id] ?? [],
      timedStreamAlignment(garmin?.streams?.[garminMatch.activity.id]),
    )
    if (projected.length > 0) return garminGearShiftsWithFittedTeeth(activity, projected)
  }
  return []
}

const cyclingDynamicsMetricKeys = [
  'leftPedalSmoothness',
  'rightPedalSmoothness',
  'leftTorqueEffectiveness',
  'rightTorqueEffectiveness',
  'leftPowerPhaseStart',
  'leftPowerPhaseEnd',
  'rightPowerPhaseStart',
  'rightPowerPhaseEnd',
] as const

function validCyclingDynamicsSamples(
  dynamics: GarminCyclingDynamics | WahooCyclingDynamics,
): boolean {
  const length = dynamics.time.length
  return (
    length >= 2 &&
    dynamics.distance.length === length &&
    cyclingDynamicsMetricKeys.every(key => dynamics[key].length === length)
  )
}

function projectCyclingDynamics(
  activity: RawStravaActivity,
  sourceStartDate: string,
  dynamics: GarminCyclingDynamics | WahooCyclingDynamics,
): ActivityCyclingDynamics | null {
  const startOffsetS = (Date.parse(sourceStartDate) - Date.parse(activity.startDate)) / 1000
  if (!Number.isFinite(startOffsetS)) return null
  const hasSamples = validCyclingDynamicsSamples(dynamics)
  const maxDistanceM = Math.max(activity.distance, 0)
  const metric = (key: (typeof cyclingDynamicsMetricKeys)[number]): (number | null)[] =>
    hasSamples
      ? dynamics[key].map(value =>
          value == null || !Number.isFinite(value) ? null : round(value, 1),
        )
      : []
  const positionChanges = dynamics.positionChanges
    .map(change => ({
      elapsedS: round(Math.max(0, change.elapsedS + startOffsetS), 3),
      distanceKm: round(Math.min(maxDistanceM, Math.max(0, change.distanceM)) / 1_000, 3),
      position: change.position,
    }))
    .filter(
      (change, index, changes) => index === 0 || changes[index - 1].position !== change.position,
    )
  if (!hasSamples && positionChanges.length === 0) return null
  return {
    elapsedS: hasSamples
      ? dynamics.time.map(value => round(Math.max(0, value + startOffsetS), 3))
      : [],
    distanceKm: hasSamples
      ? dynamics.distance.map(value => round(Math.min(maxDistanceM, Math.max(0, value)) / 1_000, 3))
      : [],
    leftPedalSmoothness: metric('leftPedalSmoothness'),
    rightPedalSmoothness: metric('rightPedalSmoothness'),
    leftTorqueEffectiveness: metric('leftTorqueEffectiveness'),
    rightTorqueEffectiveness: metric('rightTorqueEffectiveness'),
    leftPowerPhaseStart: metric('leftPowerPhaseStart'),
    leftPowerPhaseEnd: metric('leftPowerPhaseEnd'),
    rightPowerPhaseStart: metric('rightPowerPhaseStart'),
    rightPowerPhaseEnd: metric('rightPowerPhaseEnd'),
    positionChanges,
    seatedTimeS:
      dynamics.seatedTimeS == null || !Number.isFinite(dynamics.seatedTimeS)
        ? null
        : round(dynamics.seatedTimeS, 3),
    standingTimeS:
      dynamics.standingTimeS == null || !Number.isFinite(dynamics.standingTimeS)
        ? null
        : round(dynamics.standingTimeS, 3),
  }
}

function cyclingDynamicsScore(dynamics: ActivityCyclingDynamics | null): number {
  if (!dynamics) return 0
  return (
    cyclingDynamicsMetricKeys.reduce(
      (total, key) => total + dynamics[key].filter(value => value != null).length,
      0,
    ) + dynamics.positionChanges.length
  )
}

export function activityCyclingDynamics(
  activity: RawStravaActivity,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooData | null,
): ActivityCyclingDynamics | null {
  const garminDynamics = garminMatch
    ? garmin?.cyclingDynamics?.[garminMatch.activity.id]
    : undefined
  const wahooDynamics = wahooMatch ? wahoo?.cyclingDynamics[wahooMatch.activity.id] : undefined
  const projectedGarmin = garminDynamics
    ? projectCyclingDynamics(activity, garminMatch?.activity.startDate ?? '', garminDynamics)
    : null
  const projectedWahoo = wahooDynamics
    ? projectCyclingDynamics(activity, wahooMatch?.activity.startDate ?? '', wahooDynamics)
    : null
  if (projectedWahoo && cyclingDynamicsScore(projectedWahoo) > 0) {
    if (projectedGarmin?.elapsedS.length && projectedWahoo.elapsedS.length)
      for (const key of cyclingDynamicsMetricKeys)
        projectedWahoo[key] = projectedWahoo[key].map((value, index) => {
          if (value != null) return value
          const elapsedS = projectedWahoo.elapsedS[index]
          const nearest = nearestElapsedIndex(projectedGarmin.elapsedS, elapsedS)
          return Math.abs(projectedGarmin.elapsedS[nearest] - elapsedS) <=
            WAHOO_TELEMETRY_MAX_DISTANCE_S
            ? projectedGarmin[key][nearest]
            : null
        })
    return projectedWahoo
  }
  return projectedGarmin
}
