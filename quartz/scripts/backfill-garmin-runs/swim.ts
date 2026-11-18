import type { AppleSwim, SwimStroke } from '../../plugins/stores/apple'
import { type RawStravaActivity } from '../../plugins/stores/strava'
import {
  type GarminPoolSwimLength,
  type GarminSwimFitInput,
  type GarminSwimSample,
  type GarminSwimStroke,
} from '../../util/garmin-fit'
import { type TimedStravaStreams } from '../../util/strava-tcx'

const APPLE_SWIM_DEFER_MS = 3 * 24 * 60 * 60 * 1000

interface ApplePoolSwimProjection {
  distanceMeters: number
  elapsedTimeSeconds: number
  timerTimeSeconds: number
  poolLengthMeters: number
  lengths: GarminPoolSwimLength[]
}

function finite(value: number | undefined): number | undefined {
  return value != null && Number.isFinite(value) ? value : undefined
}

function positiveValue(value: number | undefined): number | undefined {
  return value != null && Number.isFinite(value) && value > 0 ? value : undefined
}

function garminSwimStroke(stroke: SwimStroke | null): GarminSwimStroke | undefined {
  if (stroke === 'kickboard') return 'drill'
  return stroke ?? undefined
}

function intervalElapsedSeconds(
  absolute: string,
  stored: number | null | undefined,
  activityStartMs: number,
): number | null {
  const time = Date.parse(absolute)
  const elapsed = Number.isFinite(time) ? (time - activityStartMs) / 1000 : stored
  if (elapsed == null || !Number.isFinite(elapsed) || elapsed < -1) return null
  return Math.max(0, elapsed)
}

function uniformPoolLength(intervals: readonly GarminPoolSwimLength[]): number | null {
  const first = intervals[0]?.distanceMeters
  if (first == null || first <= 0) return null
  return intervals.every(
    interval => Math.abs(interval.distanceMeters - first) <= Math.max(0.5, first * 0.02),
  )
    ? first
    : null
}

export function applePoolSwimProjection(
  activity: RawStravaActivity,
  swim: AppleSwim | null | undefined,
): ApplePoolSwimProjection | null {
  if (!swim?.intervals?.length) return null
  const activityStartMs = Date.parse(activity.startDate)
  if (!Number.isFinite(activityStartMs)) return null
  const lengths: GarminPoolSwimLength[] = []
  let previousEnd = 0
  let measuredStrokes = 0
  for (const interval of swim.intervals) {
    const start = intervalElapsedSeconds(interval.start, interval.startElapsedS, activityStartMs)
    const end = intervalElapsedSeconds(interval.end, interval.endElapsedS, activityStartMs)
    if (
      start == null ||
      end == null ||
      start < previousEnd ||
      end <= start ||
      !Number.isFinite(interval.distanceM) ||
      interval.distanceM <= 0
    )
      return null
    const hasMeasuredStroke =
      interval.strokeCount != null &&
      Number.isFinite(interval.strokeCount) &&
      interval.strokeCount >= 0 &&
      interval.strokeTimeS != null &&
      Number.isFinite(interval.strokeTimeS) &&
      interval.strokeTimeS > 0
    if (interval.stroke !== 'kickboard' && !hasMeasuredStroke) return null
    const strokeTime = hasMeasuredStroke
      ? Math.min(interval.strokeTimeS ?? end - start, end - start)
      : undefined
    measuredStrokes += hasMeasuredStroke ? (interval.strokeCount ?? 0) : 0
    const swimStroke = garminSwimStroke(interval.stroke)
    lengths.push({
      startElapsedSeconds: start,
      endElapsedSeconds: end,
      distanceMeters: interval.distanceM,
      ...(strokeTime != null
        ? { totalStrokes: interval.strokeCount ?? 0, strokeTimeSeconds: strokeTime }
        : {}),
      ...(swimStroke ? { swimStroke } : {}),
    })
    previousEnd = end
  }
  const poolLengthMeters = uniformPoolLength(lengths)
  if (poolLengthMeters == null) return null
  const distanceMeters = lengths.reduce((sum, length) => sum + length.distanceMeters, 0)
  if (Math.abs(distanceMeters - swim.totalM) > Math.max(1, swim.totalM * 0.01)) return null
  if (
    swim.strokeCount != null &&
    Math.abs(measuredStrokes - swim.strokeCount) > Math.max(2, swim.strokeCount * 0.05)
  )
    return null
  const swimEndMs = swim.end ? Date.parse(swim.end) : NaN
  const swimElapsed = Number.isFinite(swimEndMs) ? (swimEndMs - activityStartMs) / 1000 : 0
  const elapsedTimeSeconds = Math.max(activity.elapsedTime, previousEnd, swimElapsed)
  const measuredTime = lengths.reduce(
    (sum, length) => sum + (length.endElapsedSeconds - length.startElapsedSeconds),
    0,
  )
  const timerTimeSeconds = Math.min(elapsedTimeSeconds, measuredTime)
  if (timerTimeSeconds <= 0) return null
  return { distanceMeters, elapsedTimeSeconds, timerTimeSeconds, poolLengthMeters, lengths }
}

function poolDistanceAtElapsed(
  lengths: readonly GarminPoolSwimLength[],
  elapsedSeconds: number,
): number {
  let distance = 0
  for (const length of lengths) {
    if (elapsedSeconds <= length.startElapsedSeconds) return distance
    if (elapsedSeconds < length.endElapsedSeconds) {
      const progress =
        (elapsedSeconds - length.startElapsedSeconds) /
        (length.endElapsedSeconds - length.startElapsedSeconds)
      return distance + length.distanceMeters * progress
    }
    distance += length.distanceMeters
  }
  return distance
}

export function shouldDeferPoolSwimForApple(
  activity: RawStravaActivity,
  input: GarminSwimFitInput,
  now = Date.now(),
): boolean {
  if (input.kind !== 'pool' || input.lengths?.length) return false
  const start = Date.parse(activity.startDate)
  if (!Number.isFinite(start)) return false
  const end = start + activity.elapsedTime * 1000
  return now - end >= 0 && now - end < APPLE_SWIM_DEFER_MS
}

function swimSample(streams: TimedStravaStreams, index: number): GarminSwimSample {
  const coordinate =
    streams.latlng.length === streams.time.length ? streams.latlng[index] : undefined
  const altitudeMeters = finite(streams.altitude[index])
  const heartRateBpm = positiveValue(streams.heartrate[index])
  const cadenceRpm = positiveValue(streams.cadence[index])
  const powerWatts = positiveValue(streams.watts[index])
  const temperatureCelsius = finite(streams.temp[index])
  return {
    elapsedSeconds: streams.time[index],
    distanceMeters: finite(streams.distance[index]) ?? 0,
    ...(coordinate ? { latitudeDegrees: coordinate[0], longitudeDegrees: coordinate[1] } : {}),
    ...(altitudeMeters != null ? { altitudeMeters } : {}),
    ...(heartRateBpm != null ? { heartRateBpm } : {}),
    ...(cadenceRpm != null ? { cadenceRpm } : {}),
    ...(powerWatts != null ? { powerWatts } : {}),
    ...(temperatureCelsius != null ? { temperatureCelsius } : {}),
  }
}

function strictlyTimedSwimSamples(streams: TimedStravaStreams): GarminSwimSample[] {
  const samples: GarminSwimSample[] = []
  for (let index = 0; index < streams.time.length; index++) {
    const sample = swimSample(streams, index)
    const previous = samples[samples.length - 1]
    if (!previous) {
      samples.push(sample)
      continue
    }
    if (sample.elapsedSeconds < previous.elapsedSeconds)
      throw new Error('Strava swim stream times must be nondecreasing')
    if (sample.elapsedSeconds > previous.elapsedSeconds) {
      samples.push(sample)
      continue
    }
    samples[samples.length - 1] = {
      ...previous,
      ...sample,
      distanceMeters:
        sample.elapsedSeconds === 0 ? 0 : Math.max(previous.distanceMeters, sample.distanceMeters),
    }
  }
  return samples
}

export function garminSwimFitInput(
  activity: RawStravaActivity,
  streams: TimedStravaStreams,
  appleSwim?: AppleSwim | null,
): GarminSwimFitInput {
  const startTime = new Date(activity.startDate)
  if (!Number.isFinite(startTime.getTime()))
    throw new Error(`Strava activity ${activity.id} has an invalid start time`)
  if (streams.time.length < 2)
    throw new Error(`Strava activity ${activity.id} has fewer than two timed samples`)
  let samples = strictlyTimedSwimSamples(streams)
  if (samples.length < 2)
    throw new Error(`Strava activity ${activity.id} has fewer than two distinct timed samples`)
  let elapsedTimeSeconds = Math.max(
    activity.elapsedTime,
    streams.time[streams.time.length - 1] ?? 0,
  )
  const calories = positiveValue(activity.calories ?? undefined)
  const base = {
    sourceId: activity.id,
    name: activity.name,
    startTime,
    distanceMeters: activity.distance,
    elapsedTimeSeconds,
    timerTimeSeconds: Math.min(activity.movingTime, elapsedTimeSeconds),
    samples,
    ...(calories != null ? { calories: Math.round(calories) } : {}),
  }
  if (streams.latlng.length === streams.time.length && streams.latlng.length >= 2)
    return { ...base, kind: 'openWater' }
  const apple = applePoolSwimProjection(activity, appleSwim)
  if (!apple) return { ...base, kind: 'pool', poolLengthMeters: 25 }
  elapsedTimeSeconds = apple.elapsedTimeSeconds
  samples = samples.map(sample => ({
    ...sample,
    distanceMeters: poolDistanceAtElapsed(apple.lengths, sample.elapsedSeconds),
  }))
  return {
    ...base,
    distanceMeters: apple.distanceMeters,
    elapsedTimeSeconds,
    timerTimeSeconds: apple.timerTimeSeconds,
    samples,
    kind: 'pool',
    poolLengthMeters: apple.poolLengthMeters,
    lengths: apple.lengths,
  }
}
