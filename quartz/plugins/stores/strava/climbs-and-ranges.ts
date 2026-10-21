import type { GarminActivityMatch, GarminCache, GarminClimbSegment, GarminStreams } from '../garmin'
import type { WahooActivityMatch, WahooData, WahooSummitSegment } from '../wahoo'
import type { ActivityClimbSource, CyclingClimbEffort } from './efforts-and-curves'
import type {
  ActivityKind,
  RawStravaActivity,
  RawStravaActivityDetail,
  RawStravaAnalysisRange,
  RawStravaRunSplit,
  StravaStreams,
} from './raw-and-kinds'
import {
  nearestElapsedIndex,
  nullableRound,
  round,
  timedStreamAlignment,
  validStreamIndex,
  type TimedStreamAlignment,
} from './stream-primitives'

export type ActivityAnalysisKind = 'lap' | 'segment' | 'climb'

interface ActivityAnalysisRangeBase {
  id: string
  label: string
  startElapsedS: number
  endElapsedS: number
  startDistanceKm: number
  endDistanceKm: number
  durationS: number
  movingTimeS?: number
  distanceKm: number
  elevationGainM: number | null
  averageGradePct?: number | null
  averageSpeedKph: number | null
  averageHeartRate: number | null
  averageWatts: number | null
  averageCadence: number | null
  heartRateChange?: { source: 'strava'; startBpm: number; endBpm: number }
}

export type ActivityAnalysisRange = ActivityAnalysisRangeBase &
  ({ kind: 'lap' | 'segment' } | { kind: 'climb'; source: ActivityClimbSource })

export interface ActivityRunSplit {
  split: number
  distanceKm: number
  elapsedTimeS: number
  movingTimeS: number
  averageSpeedKph: number
  averageHeartRate?: number | null
  elevationDifferenceM: number | null
  paceZone: number | null
}

export interface ActivityClimbSegment {
  source: ActivityClimbSource
  name: string
  startDate: string
  endDate: string
  distanceM: number
  durationS: number
  elevationGainM: number | null
  avgGradePct: number | null
  avgSpeedMps: number | null
  avgHeartRate: number | null
  avgPower: number | null
  avgCadence: number | null
}

export function cyclingClimbEfforts(
  segments: ActivityClimbSegment[],
  weightKg: number | null,
): CyclingClimbEffort[] {
  return segments.flatMap(segment => {
    const durationS = segment.durationS
    const elevationGainM = segment.elevationGainM ?? 0
    if (durationS <= 0 || segment.distanceM <= 0 || elevationGainM <= 0) return []
    const averageWatts = segment.avgPower
    return [
      {
        source: segment.source,
        name: segment.name,
        durationS: round(durationS, 1),
        distanceM: round(segment.distanceM, 1),
        elevationGainM: round(elevationGainM, 1),
        averageGradePct:
          segment.avgGradePct ?? round((elevationGainM / segment.distanceM) * 100, 1),
        averageSpeedKph:
          segment.avgSpeedMps != null
            ? round(segment.avgSpeedMps * 3.6, 1)
            : round((segment.distanceM / durationS) * 3.6, 1),
        averageHeartRate: segment.avgHeartRate,
        averageWatts,
        wattsPerKg:
          averageWatts != null && weightKg != null && weightKg > 0
            ? round(averageWatts / weightKg, 2)
            : null,
        vamMPerHour: Math.round((elevationGainM / durationS) * 3600),
      },
    ]
  })
}

function garminClimbSegments(segments: GarminClimbSegment[]): ActivityClimbSegment[] {
  return segments.map((segment, index) => ({
    source: 'garmin-climbpro',
    name: `Climb ${index + 1}`,
    startDate: segment.startDate,
    endDate: segment.endDate,
    distanceM: segment.distanceM,
    durationS: segment.durationS,
    elevationGainM: segment.elevationGainM,
    avgGradePct: segment.avgGradePct,
    avgSpeedMps: segment.avgSpeedMps,
    avgHeartRate: segment.avgHeartRate,
    avgPower: segment.avgPower,
    avgCadence: segment.avgCadence,
  }))
}

function wahooClimbSegments(segments: WahooSummitSegment[]): ActivityClimbSegment[] {
  return segments.map((segment, index) => ({
    source: segment.feature === 'summit-segment' ? 'wahoo-summit-segment' : 'wahoo-summit-freeride',
    name: segment.name ? `Summit ${segment.name}` : `Summit ${index + 1}`,
    startDate: segment.startDate,
    endDate: segment.endDate,
    distanceM: segment.distanceM,
    durationS: segment.durationS,
    elevationGainM: segment.elevationGainM,
    avgGradePct: segment.avgGradePct,
    avgSpeedMps: segment.avgSpeedMps,
    avgHeartRate: segment.avgHeartRate,
    avgPower: segment.avgPower,
    avgCadence: segment.avgCadence,
  }))
}

export function activityClimbSegments(
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooData | null,
): ActivityClimbSegment[] {
  if (sport !== 'bike') return []
  const wahooClimbs = wahooMatch
    ? wahooClimbSegments(wahoo?.summitSegments[wahooMatch.activity.id] ?? [])
    : []
  if (wahooClimbs.length > 0) return wahooClimbs
  if (garminMatch) return garminClimbSegments(garmin?.climbs?.[garminMatch.activity.id] ?? [])
  return []
}

interface ProjectedAnalysis {
  ranges: ActivityAnalysisRange[]
  boundaryIndices: number[]
}

function rangeIndices(
  raw: RawStravaAnalysisRange,
  alignment: TimedStreamAlignment,
  activityStartMs: number,
  fallbackStartElapsedS: number | null,
  hasStravaStreams: boolean,
): [number, number] | null {
  const startMs = raw.startDate ? Date.parse(raw.startDate) : NaN
  // Lap indices can refer to the original recording rather than the exported stream.
  // Strava timestamps survive resampling and provider timeline replacement.
  if (Number.isFinite(startMs)) {
    const startElapsedS = (startMs - activityStartMs) / 1000
    return [
      nearestElapsedIndex(alignment.time, startElapsedS),
      nearestElapsedIndex(alignment.time, startElapsedS + raw.elapsedTime),
    ]
  }
  if (
    hasStravaStreams &&
    validStreamIndex(raw.startIndex, alignment.time.length) &&
    validStreamIndex(raw.endIndex, alignment.time.length) &&
    raw.endIndex >= raw.startIndex
  )
    return [raw.startIndex, raw.endIndex]

  const startElapsedS = fallbackStartElapsedS
  if (startElapsedS == null || !Number.isFinite(startElapsedS)) return null
  const endElapsedS = startElapsedS + raw.elapsedTime
  return [
    nearestElapsedIndex(alignment.time, startElapsedS),
    nearestElapsedIndex(alignment.time, endElapsedS),
  ]
}

function projectStravaAnalysisRange(
  kind: 'lap' | 'segment',
  raw: RawStravaAnalysisRange,
  index: number,
  bounds: [number, number],
  alignment: TimedStreamAlignment,
): ActivityAnalysisRange {
  const [startIndex, endIndex] = bounds
  return {
    kind,
    id: `${kind}:${raw.id}`,
    label: raw.name || `${kind === 'lap' ? 'Lap' : 'Segment'} ${index + 1}`,
    startElapsedS: round(alignment.time[startIndex], 3),
    endElapsedS: round(alignment.time[endIndex], 3),
    startDistanceKm: round(alignment.distance[startIndex] / 1000, 3),
    endDistanceKm: round(alignment.distance[endIndex] / 1000, 3),
    durationS: raw.elapsedTime,
    movingTimeS: raw.movingTime,
    distanceKm: round(raw.distance / 1000, 3),
    elevationGainM: nullableRound(raw.totalElevationGain),
    averageSpeedKph: nullableRound(raw.averageSpeed == null ? null : raw.averageSpeed * 3.6, 2),
    averageHeartRate: nullableRound(raw.averageHeartrate),
    averageWatts: nullableRound(raw.averageWatts),
    averageCadence: nullableRound(raw.averageCadence),
  }
}

export function projectRunSplits(splits: RawStravaRunSplit[] | undefined): ActivityRunSplit[] {
  return (splits ?? []).map(split => ({
    split: split.split,
    distanceKm: round(split.distance / 1000, 3),
    elapsedTimeS: split.elapsedTime,
    movingTimeS: split.movingTime,
    averageSpeedKph: round(split.averageSpeed * 3.6, 2),
    averageHeartRate: split.averageHeartrate ?? null,
    elevationDifferenceM: nullableRound(split.elevationDifference),
    paceZone: nullableRound(split.paceZone, 0),
  }))
}

export function projectAnalysisRanges(
  a: RawStravaActivity,
  detail: RawStravaActivityDetail | undefined,
  streams: StravaStreams | GarminStreams | undefined,
  climbs: ActivityClimbSegment[],
  hasStravaStreams: boolean,
): ProjectedAnalysis {
  const stationary = a.distance === 0 && streams?.distance.length === 0
  const alignment = timedStreamAlignment(
    stationary && streams?.time ? { ...streams, distance: streams.time.map(() => 0) } : streams,
  )
  if (!alignment) return { ranges: [], boundaryIndices: [] }
  const activityStartMs = Date.parse(a.startDate)
  if (!Number.isFinite(activityStartMs)) return { ranges: [], boundaryIndices: [] }
  const ranges: ActivityAnalysisRange[] = []
  const boundaryIndices: number[] = []
  let lapElapsedS = 0

  for (const [index, raw] of (detail?.laps ?? []).entries()) {
    const bounds = rangeIndices(raw, alignment, activityStartMs, lapElapsedS, hasStravaStreams)
    lapElapsedS += raw.elapsedTime
    if (!bounds || bounds[1] < bounds[0]) continue
    const range = projectStravaAnalysisRange('lap', raw, index, bounds, alignment)
    if (a.distance === 0 && raw.distance === 0) {
      range.elevationGainM = null
      range.averageSpeedKph = null
      const heartRate = streams?.heartrate
      const startBpm = heartRate?.[bounds[0]]
      const endBpm = heartRate?.[bounds[1]]
      if (
        hasStravaStreams &&
        heartRate?.length === alignment.time.length &&
        startBpm != null &&
        Number.isFinite(startBpm) &&
        startBpm > 0 &&
        endBpm != null &&
        Number.isFinite(endBpm) &&
        endBpm > 0
      )
        range.heartRateChange = { source: 'strava', startBpm, endBpm }
    }
    ranges.push(range)
    boundaryIndices.push(...bounds)
  }
  for (const [index, raw] of (detail?.segmentEfforts ?? []).entries()) {
    const bounds = rangeIndices(raw, alignment, activityStartMs, null, hasStravaStreams)
    if (!bounds || bounds[1] < bounds[0]) continue
    ranges.push(projectStravaAnalysisRange('segment', raw, index, bounds, alignment))
    boundaryIndices.push(...bounds)
  }

  const activityEndElapsedS = alignment.time[alignment.time.length - 1]
  for (const [index, climb] of climbs.entries()) {
    const startMs = Date.parse(climb.startDate)
    const endMs = Date.parse(climb.endDate)
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) continue
    const unclampedStartS = (startMs - activityStartMs) / 1000
    const unclampedEndS = (endMs - activityStartMs) / 1000
    if (unclampedEndS < 0 || unclampedStartS > activityEndElapsedS) continue
    const startIndex = nearestElapsedIndex(
      alignment.time,
      Math.max(0, Math.min(activityEndElapsedS, unclampedStartS)),
    )
    const endIndex = nearestElapsedIndex(
      alignment.time,
      Math.max(0, Math.min(activityEndElapsedS, unclampedEndS)),
    )
    if (endIndex < startIndex) continue
    ranges.push({
      kind: 'climb',
      source: climb.source,
      id: `${climb.source}:${index + 1}:${climb.startDate}`,
      label: climb.name,
      startElapsedS: round(alignment.time[startIndex], 3),
      endElapsedS: round(alignment.time[endIndex], 3),
      startDistanceKm: round(alignment.distance[startIndex] / 1000, 3),
      endDistanceKm: round(alignment.distance[endIndex] / 1000, 3),
      durationS: climb.durationS,
      distanceKm: round(climb.distanceM / 1000, 3),
      elevationGainM: nullableRound(climb.elevationGainM),
      averageGradePct: nullableRound(climb.avgGradePct),
      averageSpeedKph: nullableRound(climb.avgSpeedMps == null ? null : climb.avgSpeedMps * 3.6, 2),
      averageHeartRate: nullableRound(climb.avgHeartRate),
      averageWatts: nullableRound(climb.avgPower),
      averageCadence: nullableRound(climb.avgCadence),
    })
    boundaryIndices.push(startIndex, endIndex)
  }

  const order: Record<ActivityAnalysisKind, number> = { lap: 0, segment: 1, climb: 2 }
  if (a.sportType.startsWith('Virtual'))
    for (const range of ranges) {
      if (range.kind === 'climb') continue
      range.distanceKm = round(Math.max(0, range.endDistanceKm - range.startDistanceKm), 3)
      const durationS = range.movingTimeS ?? range.durationS
      range.averageSpeedKph =
        durationS > 0 ? round((range.distanceKm * 3_600) / durationS, 2) : null
      range.elevationGainM = null
    }
  ranges.sort(
    (left, right) =>
      left.startElapsedS - right.startElapsedS ||
      order[left.kind] - order[right.kind] ||
      left.id.localeCompare(right.id),
  )
  return { ranges, boundaryIndices }
}
