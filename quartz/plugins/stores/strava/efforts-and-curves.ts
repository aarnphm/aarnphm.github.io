import type { CyclingMechanics } from '../../../util/cycling-torque'
import type { GarminStreams } from '../garmin'
import { DAY_MS, type StravaStreams } from './raw-and-kinds'
import { MAX_ACCEL_MPS2, MAX_SPEED_WINDOW_S, round } from './stream-primitives'

export interface PowerCurvePoint {
  s: number
  w: number
  activityId?: number
  activityDate?: string
}

export interface ActivityDistanceEffort {
  label: string
  targetDistanceM: number
  elapsedTimeS: number
  averageSpeedKph: number
  averageHeartRate: number | null
  elevationDeltaM: number | null
}

export interface CyclingPowerEffort {
  startElapsedS?: number
  endElapsedS?: number
  mechanics?: CyclingMechanics | null
  durationS: number
  averageWatts: number
  wattsPerKg: number | null
  averageHeartRate: number | null
  elevationDeltaM: number
}

export type ActivityClimbSource =
  | 'garmin-climbpro'
  | 'wahoo-summit-segment'
  | 'wahoo-summit-freeride'

export interface CyclingClimbEffort {
  source: ActivityClimbSource
  name: string
  durationS: number
  distanceM: number
  elevationGainM: number
  averageGradePct: number
  averageSpeedKph: number
  averageHeartRate: number | null
  averageWatts: number | null
  wattsPerKg: number | null
  vamMPerHour: number
}

export interface ActivityBestEfforts {
  distanceSource?: 'calculated'
  weightKg: number | null
  weightDate: string | null
  distance: ActivityDistanceEffort[]
  power: CyclingPowerEffort[]
  climbs: CyclingClimbEffort[]
}

const MILE_M = 1609.344
const MAX_EFFORT_TIMELINE_S = (7 * DAY_MS) / 1000
const POWER_EFFORT_SECS = [
  3, 5, 6, 15, 30, 45, 60, 120, 180, 300, 360, 420, 480, 600, 720, 900, 1200, 1800, 2700, 3600,
  5400, 7200,
]
const DISTANCE_EFFORTS = [
  ['5 mile', 5 * MILE_M],
  ['10K', 10_000],
  ['10 mile', 10 * MILE_M],
  ['20K', 20_000],
  ['30K', 30_000],
  ['40K', 40_000],
  ['50K', 50_000],
  ['80K', 80_000],
  ['50 mile', 50 * MILE_M],
  ['90K', 90_000],
  ['100K', 100_000],
  ['100 mile', 100 * MILE_M],
  ['180K', 180_000],
] as const

interface EffortTimeline {
  distanceM: Float64Array
  altitudeM: Float64Array
  watts: Float64Array
  wattsObserved: Uint8Array
  heartRate: Float64Array
}

interface BestPowerWindow {
  durationS: number
  start: number
  end: number
  averageWatts: number
}

export function effortTimeline(
  streams: StravaStreams | GarminStreams | undefined,
  movingTimeS: number,
): EffortTimeline | null {
  if (!streams) return null
  const sampleCount = Math.max(
    streams.distance.length,
    streams.altitude.length,
    streams.watts?.length ?? 0,
    streams.heartrate?.length ?? 0,
  )
  if (sampleCount < 2) return null
  const rawTime = 'time' in streams ? streams.time : undefined
  if ('time' in streams && rawTime?.length !== sampleCount) return null
  if (!rawTime && Math.abs(sampleCount - movingTimeS) / Math.max(1, movingTimeS) > 0.15) return null
  const sampleSeconds = new Int32Array(sampleCount)
  let previousSecond = 0
  for (let i = 0; i < sampleCount; i++) {
    const raw = rawTime ? rawTime[i] : i
    const second = Number.isFinite(raw) ? Math.max(previousSecond, Math.round(raw)) : previousSecond
    sampleSeconds[i] = second
    previousSecond = second
  }
  if (previousSecond < 1 || previousSecond > MAX_EFFORT_TIMELINE_S) return null

  const length = previousSecond + 1
  const distanceM = new Float64Array(length)
  const altitudeM = new Float64Array(length)
  const watts = new Float64Array(length)
  const wattsObserved = new Uint8Array(length)
  const heartRate = new Float64Array(length)
  const distanceSet = new Uint8Array(length)
  const altitudeSet = new Uint8Array(length)
  const wattCount = new Uint16Array(length)
  const heartRateCount = new Uint16Array(length)
  const initialAltitude = streams.altitude.find(Number.isFinite) ?? 0

  for (let i = 0; i < sampleCount; i++) {
    const second = sampleSeconds[i]
    const distance = streams.distance[i]
    if (Number.isFinite(distance)) {
      distanceM[second] = Math.max(0, distance)
      distanceSet[second] = 1
    }
    const altitude = streams.altitude[i]
    if (Number.isFinite(altitude)) {
      altitudeM[second] = altitude
      altitudeSet[second] = 1
    }
    const power = streams.watts?.[i]
    if (Number.isFinite(power)) {
      watts[second] += Math.max(0, power ?? 0)
      wattCount[second]++
      wattsObserved[second] = 1
    }
    const hr = streams.heartrate?.[i]
    if (Number.isFinite(hr) && (hr ?? 0) > 0) {
      heartRate[second] += hr ?? 0
      heartRateCount[second]++
    }
  }

  let distance = 0
  let altitude = initialAltitude
  for (let second = 0; second < length; second++) {
    if (distanceSet[second]) distance = Math.max(distance, distanceM[second])
    distanceM[second] = distance
    if (altitudeSet[second]) altitude = altitudeM[second]
    altitudeM[second] = altitude
    if (wattCount[second]) watts[second] /= wattCount[second]
    if (heartRateCount[second]) heartRate[second] /= heartRateCount[second]
  }
  return { distanceM, altitudeM, watts, wattsObserved, heartRate }
}

export function sumPrefix(values: Float64Array): Float64Array {
  const prefix = new Float64Array(values.length + 1)
  for (let i = 0; i < values.length; i++) prefix[i + 1] = prefix[i] + values[i]
  return prefix
}

function positivePrefixes(values: Float64Array): [Float64Array, Uint32Array] {
  const sum = new Float64Array(values.length + 1)
  const count = new Uint32Array(values.length + 1)
  for (let i = 0; i < values.length; i++) {
    const value = values[i]
    sum[i + 1] = sum[i] + (value > 0 ? value : 0)
    count[i + 1] = count[i] + (value > 0 ? 1 : 0)
  }
  return [sum, count]
}

function averagePositive(
  sum: Float64Array,
  count: Uint32Array,
  start: number,
  end: number,
): number | null {
  const n = count[end] - count[start]
  return n > 0 ? Math.round((sum[end] - sum[start]) / n) : null
}

function bestPowerWindows(
  timeline: EffortTimeline,
  durations: readonly number[],
): BestPowerWindow[] {
  const prefix = sumPrefix(timeline.watts)
  const windows: BestPowerWindow[] = []
  for (const durationS of durations) {
    if (durationS > timeline.watts.length) break
    let bestSum = -1
    let bestStart = 0
    for (let start = 0; start + durationS <= timeline.watts.length; start++) {
      const sum = prefix[start + durationS] - prefix[start]
      if (sum > bestSum) {
        bestSum = sum
        bestStart = start
      }
    }
    windows.push({
      durationS,
      start: bestStart,
      end: bestStart + durationS,
      averageWatts: Math.floor(Math.max(0, bestSum / durationS)),
    })
  }
  return windows
}

export function bestObservedPowerWindows(
  timeline: EffortTimeline,
  durations: readonly number[],
): BestPowerWindow[] {
  const powerPrefix = sumPrefix(timeline.watts)
  const observedPrefix = new Uint32Array(timeline.wattsObserved.length + 1)
  for (let index = 0; index < timeline.wattsObserved.length; index++)
    observedPrefix[index + 1] = observedPrefix[index] + timeline.wattsObserved[index]

  const windows: BestPowerWindow[] = []
  for (const durationS of durations) {
    if (durationS > timeline.watts.length) break
    let bestSum = -1
    let bestStart = 0
    for (let start = 0; start + durationS <= timeline.watts.length; start++) {
      const end = start + durationS
      if (observedPrefix[end] - observedPrefix[start] !== durationS) continue
      const sum = powerPrefix[end] - powerPrefix[start]
      if (sum > bestSum) {
        bestSum = sum
        bestStart = start
      }
    }
    if (bestSum < 0) continue
    windows.push({
      durationS,
      start: bestStart,
      end: bestStart + durationS,
      averageWatts: round(bestSum / durationS, 2),
    })
  }
  return windows
}

export function meanMaxCurve(timeline: EffortTimeline | null): PowerCurvePoint[] {
  if (!timeline) return []
  const durations = Array.from({ length: timeline.watts.length }, (_, index) => index + 1)
  return bestPowerWindows(timeline, durations).map(window => ({
    s: window.durationS,
    w: window.averageWatts,
  }))
}

export function maxSpeedKph(timeline: EffortTimeline | null): number | null {
  if (!timeline) return null
  const distanceM = timeline.distanceM
  if (distanceM.length < 2) return null
  const speeds = new Float64Array(distanceM.length - 1)
  let allowed = 0
  for (let second = 0; second < speeds.length; second++) {
    allowed = Math.min(distanceM[second + 1] - distanceM[second], allowed + MAX_ACCEL_MPS2)
    speeds[second] = allowed
  }
  const windowS = Math.min(MAX_SPEED_WINDOW_S, speeds.length)
  let bestM = 0
  let sumM = 0
  for (let second = 0; second < speeds.length; second++) {
    sumM += speeds[second]
    if (second >= windowS) sumM -= speeds[second - windowS]
    if (second >= windowS - 1) bestM = Math.max(bestM, sumM)
  }
  return bestM > 0 ? round((bestM / windowS) * 3.6, 1) : null
}

export function distanceBestEfforts(timeline: EffortTimeline): ActivityDistanceEffort[] {
  const [hrSum, hrCount] = positivePrefixes(timeline.heartRate)
  const efforts: ActivityDistanceEffort[] = []
  for (const [label, targetDistanceM] of DISTANCE_EFFORTS) {
    if (timeline.distanceM[timeline.distanceM.length - 1] - timeline.distanceM[0] < targetDistanceM)
      break
    let best: ActivityDistanceEffort | null = null
    let bestElapsed = Infinity
    let end = 1
    for (let start = 0; start < timeline.distanceM.length - 1; start++) {
      if (end <= start) end = start + 1
      const target = timeline.distanceM[start] + targetDistanceM
      while (end < timeline.distanceM.length && timeline.distanceM[end] < target) end++
      if (end >= timeline.distanceM.length) break
      const previous = Math.max(start, end - 1)
      const spanM = timeline.distanceM[end] - timeline.distanceM[previous]
      const fraction = spanM > 0 ? (target - timeline.distanceM[previous]) / spanM : 1
      const elapsed = previous - start + Math.min(1, Math.max(0, fraction))
      if (elapsed <= 0 || elapsed >= bestElapsed) continue
      const endAltitude =
        timeline.altitudeM[previous] +
        (timeline.altitudeM[end] - timeline.altitudeM[previous]) * fraction
      bestElapsed = elapsed
      best = {
        label,
        targetDistanceM,
        elapsedTimeS: Math.ceil(elapsed),
        averageSpeedKph: round((targetDistanceM / 1000 / elapsed) * 3600, 3),
        averageHeartRate: averagePositive(hrSum, hrCount, start, end + 1),
        elevationDeltaM: round(endAltitude - timeline.altitudeM[start], 1),
      }
    }
    if (best) efforts.push(best)
  }
  return efforts
}

export function powerBestEfforts(
  timeline: EffortTimeline,
  weightKg: number | null,
): CyclingPowerEffort[] {
  const [hrSum, hrCount] = positivePrefixes(timeline.heartRate)
  return bestPowerWindows(timeline, POWER_EFFORT_SECS).map(window => ({
    startElapsedS: window.start,
    endElapsedS: window.end,
    durationS: window.durationS,
    averageWatts: window.averageWatts,
    wattsPerKg: weightKg != null && weightKg > 0 ? round(window.averageWatts / weightKg, 2) : null,
    averageHeartRate: averagePositive(hrSum, hrCount, window.start, window.end),
    elevationDeltaM: round(
      timeline.altitudeM[Math.max(window.start, window.end - 1)] - timeline.altitudeM[window.start],
      1,
    ),
  }))
}

export interface PowerCurveSource {
  activityId: number
  activityDate: string
  curve: PowerCurvePoint[]
}

export function mergeMaxCurves(sources: PowerCurveSource[]): PowerCurvePoint[] {
  const best = new Map<number, PowerCurvePoint>()
  for (const source of sources)
    for (const point of source.curve) {
      const current = best.get(point.s)
      if (current && current.w >= point.w) continue
      best.set(point.s, {
        s: point.s,
        w: point.w,
        activityId: source.activityId,
        activityDate: source.activityDate,
      })
    }
  return [...best.values()].sort((left, right) => left.s - right.s)
}
