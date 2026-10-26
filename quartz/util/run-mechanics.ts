import { RECORDING_PAUSE_MIN_S, type ElapsedPause } from './elapsed-axis'
import { runGradeFactors } from './run-grade'

interface RunStreams {
  time?: readonly number[]
  distance: readonly number[]
  altitude?: readonly number[]
  heartrate?: readonly number[]
  cadence?: readonly number[]
}

export interface RunStrideCell {
  cadenceSpm: number
  strideM: number
  seconds: number
}

export interface RunStrideCentroid {
  cadenceSpm: number
  strideM: number
  seconds: number
}

export interface RunStrideCadenceTrace {
  source: 'strava'
  method: 'speed-cadence-v1'
  seconds: number
  cells: RunStrideCell[]
  early: RunStrideCentroid | null
  late: RunStrideCentroid | null
}

export interface RunHeartCostPoint {
  elapsedS: number
  distanceKm: number
  elevationM: number | null
  gapSpeed30sMps: number | null
  heartRate30sBpm: number | null
  gapSpeed5mMps: number | null
  heartRate5mBpm: number | null
  cumulativeMetersPerBeat: number | null
}

export interface RunHeartCostTrace {
  source: 'strava'
  method: 'gap-per-beat-v1'
  heartRateLagS: number
  points: RunHeartCostPoint[]
  /** Recording gaps of at least RECORDING_PAUSE_MIN_S; the chart gives them no width. */
  pauses: ElapsedPause[]
}

export const RUN_STRIDE_CADENCE_BIN_SPM = 2
export const RUN_STRIDE_BIN_M = 0.04
export const RUN_HEART_COST_MAX_POINTS = 1_000
// Each pause adds up to two points at its edges; only the longest pauses get them.
export const RUN_HEART_COST_MAX_PAUSES = 100
// Heart rate follows a pace change with a time constant near 30 s, so each speed window pairs
// with the heart-rate window that ends 30 s later.
export const RUN_HEART_RATE_LAG_S = 30
// Straps read near zero while they settle and GPS jumps make impossible speeds; samples outside
// these bounds are dropped, which also bounds every emitted average for the client validator.
export const RUN_HEART_RATE_MIN_BPM = 40
export const RUN_HEART_RATE_MAX_BPM = 250
export const RUN_GAP_MAX_MPS = 15

const GAIT_SMOOTH_S = 10
// Walking stays under ~130 spm and ~1.6 m/s; those samples describe another gait.
const GAIT_MIN_SPM = 130
const GAIT_MAX_SPM = 240
const GAIT_MIN_MPS = 1.6
const GAIT_MIN_STRIDE_M = 0.3
const GAIT_MAX_STRIDE_M = 2.5
const THIRDS_MIN_S = 600
const MOVING_MIN_MPS = 1609.344 / (30 * 60)
const WINDOW_COVERAGE = 0.8
// Averages hold through dropouts and brief pauses; a longer stop breaks the line.
const STOP_BREAK_S = 30

const round = (value: number, digits: number): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

const finite = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value)

const streamTiming = (
  time: readonly number[] | undefined,
  length: number,
): { nominalS: number; continuityS: number } | null => {
  if (!time || time.length !== length || length < 2) return null
  for (let i = 0; i < length; i++)
    if (!Number.isFinite(time[i]) || (i > 0 && time[i] <= time[i - 1])) return null
  const intervals = time.slice(1).map((value, i) => value - time[i])
  intervals.sort((a, b) => a - b)
  const nominalS = intervals[Math.floor((intervals.length - 1) / 2)]
  // Smart recording spaces samples by a few seconds; a longer jump is a pause.
  return { nominalS, continuityS: Math.max(2, nominalS * 2.5) }
}

export function buildRunStrideCadenceTrace(
  streams: RunStreams | undefined,
): RunStrideCadenceTrace | null {
  const time = streams?.time
  const cadence = streams?.cadence
  const distance = streams?.distance
  if (!streams || !time || !cadence || !distance) return null
  const n = time.length
  const timing = streamTiming(time, n)
  if (!timing || cadence.length !== n || distance.length !== n) return null
  const cells = new Map<string, RunStrideCell>()
  const included: { seconds: number; logCadence: number; logStride: number }[] = []
  let start = 0
  let end = 0
  for (let i = 0; i < n; i++) {
    while (time[start] < time[i] - GAIT_SMOOTH_S / 2) start++
    while (end + 1 < n && time[end + 1] <= time[i] + GAIT_SMOOTH_S / 2) end++
    const spanS = time[end] - time[start]
    if (spanS < GAIT_SMOOTH_S * 0.6 || spanS > GAIT_SMOOTH_S + timing.continuityS) continue
    let paused = false
    let cadenceSum = 0
    let cadenceCount = 0
    for (let j = start; j <= end; j++) {
      if (j > start && time[j] - time[j - 1] > timing.continuityS) paused = true
      if (finite(cadence[j]) && cadence[j] > 0) {
        cadenceSum += cadence[j]
        cadenceCount++
      }
    }
    if (paused || cadenceCount < (end - start + 1) * WINDOW_COVERAGE) continue
    const meters = distance[end] - distance[start]
    if (!finite(meters)) continue
    const speedMps = meters / spanS
    // Strava stores running cadence per leg.
    const cadenceSpm = (cadenceSum / cadenceCount) * 2
    const strideM = (speedMps * 60) / cadenceSpm
    if (
      speedMps < GAIT_MIN_MPS ||
      cadenceSpm < GAIT_MIN_SPM ||
      cadenceSpm > GAIT_MAX_SPM ||
      strideM < GAIT_MIN_STRIDE_M ||
      strideM > GAIT_MAX_STRIDE_M
    )
      continue
    const deltaS = (time[i + 1] ?? time[i] + timing.nominalS) - time[i]
    const seconds = deltaS <= timing.continuityS ? deltaS : timing.nominalS
    const cellCadence =
      Math.floor(cadenceSpm / RUN_STRIDE_CADENCE_BIN_SPM) * RUN_STRIDE_CADENCE_BIN_SPM
    const cellStride = round(Math.floor(strideM / RUN_STRIDE_BIN_M) * RUN_STRIDE_BIN_M, 2)
    const key = `${cellCadence}:${cellStride}`
    const cell = cells.get(key)
    if (cell) cell.seconds += seconds
    else cells.set(key, { cadenceSpm: cellCadence, strideM: cellStride, seconds })
    included.push({ seconds, logCadence: Math.log(cadenceSpm), logStride: Math.log(strideM) })
  }
  const totalS = included.reduce((total, sample) => total + sample.seconds, 0)
  if (totalS < 60) return null
  const centroid = (from: number, to: number): RunStrideCentroid | null => {
    let elapsed = 0
    let seconds = 0
    let logCadence = 0
    let logStride = 0
    for (const sample of included) {
      const middle = elapsed + sample.seconds / 2
      elapsed += sample.seconds
      if (middle < from || middle >= to) continue
      seconds += sample.seconds
      logCadence += sample.logCadence * sample.seconds
      logStride += sample.logStride * sample.seconds
    }
    if (seconds === 0) return null
    // Geometric means keep the centroid's pace equal to the mean log-speed of its samples.
    return {
      cadenceSpm: round(Math.exp(logCadence / seconds), 2),
      strideM: round(Math.exp(logStride / seconds), 4),
      seconds: round(seconds, 1),
    }
  }
  const thirds = totalS >= THIRDS_MIN_S
  return {
    source: 'strava',
    method: 'speed-cadence-v1',
    seconds: round(totalS, 1),
    cells: Array.from(cells.values())
      .map(cell => ({ ...cell, seconds: round(cell.seconds, 1) }))
      .filter(cell => cell.seconds > 0)
      .sort((a, b) => a.cadenceSpm - b.cadenceSpm || a.strideM - b.strideM),
    early: thirds ? centroid(0, totalS / 3) : null,
    late: thirds ? centroid((2 * totalS) / 3, Infinity) : null,
  }
}

export const runMetersPerBeat = (
  speedMps: number | null | undefined,
  heartRateBpm: number | null | undefined,
): number | null =>
  finite(speedMps) && finite(heartRateBpm) && heartRateBpm > 0 && speedMps >= 0
    ? (speedMps * 60) / heartRateBpm
    : null

/** Grade-adjusted metres per heartbeat; speed leads heart rate by the lag, as HR answers late. */
export function buildRunHeartCostTrace(
  streams: RunStreams | undefined,
  elapsedTimeS: number,
): RunHeartCostTrace | null {
  const time = streams?.time
  const heartrate = streams?.heartrate
  const distance = streams?.distance
  if (!streams || !time || !heartrate || !distance) return null
  const n = time.length
  const timing = streamTiming(time, n)
  if (
    !timing ||
    heartrate.length !== n ||
    distance.length !== n ||
    !Number.isFinite(elapsedTimeS) ||
    elapsedTimeS <= 0 ||
    elapsedTimeS > 48 * 3_600
  )
    return null
  const factors =
    streams.altitude?.length === n
      ? runGradeFactors({ altitude: streams.altitude, distance })
      : null
  const totalS = Math.ceil(Math.min(elapsedTimeS, time[n - 1] + 1))
  // One-second grid; seconds without a continuous moving sample stay empty.
  const gap = new Float64Array(totalS)
  const heartRate = new Float64Array(totalS)
  const moving = new Uint8Array(totalS)
  const elevation = new Float64Array(totalS).fill(Number.NaN)
  const distanceKm = new Float64Array(totalS)
  let lastDistanceKm = 0
  let cursor = 0
  for (let i = 1; i < n; i++) {
    const deltaS = time[i] - time[i - 1]
    const from = Math.max(0, Math.floor(time[i - 1]))
    const to = Math.min(totalS, Math.floor(time[i]))
    const segment = distance[i] - distance[i - 1]
    if (finite(distance[i]) && distance[i] >= 0)
      lastDistanceKm = Math.max(lastDistanceKm, distance[i] / 1_000)
    const altitude = streams.altitude?.[i]
    for (; cursor < to; cursor++) {
      distanceKm[cursor] = lastDistanceKm
      if (finite(altitude)) elevation[cursor] = altitude
    }
    if (deltaS > timing.continuityS || !finite(segment) || segment < 0 || to <= from) continue
    if (segment / deltaS < MOVING_MIN_MPS) continue
    const gapPerS = (segment * (factors?.[i] ?? 1)) / deltaS
    if (gapPerS > RUN_GAP_MAX_MPS) continue
    const bpm =
      finite(heartrate[i]) &&
      heartrate[i] >= RUN_HEART_RATE_MIN_BPM &&
      heartrate[i] <= RUN_HEART_RATE_MAX_BPM
        ? heartrate[i]
        : 0
    for (let s = from; s < to; s++) {
      gap[s] = gapPerS
      heartRate[s] = bpm
      moving[s] = 1
    }
  }
  for (; cursor < totalS; cursor++) distanceKm[cursor] = lastDistanceKm
  const prefix = (pick: (s: number) => number): Float64Array => {
    const sums = new Float64Array(totalS + 1)
    for (let s = 0; s < totalS; s++) sums[s + 1] = sums[s] + pick(s)
    return sums
  }
  const movingBefore = prefix(s => moving[s])
  const heartSum = prefix(s => (moving[s] && heartRate[s] > 0 ? heartRate[s] : 0))
  const jointGap = prefix(s => (moving[s] && heartRate[s] > 0 ? gap[s] : 0))
  const jointSeconds = prefix(s => (moving[s] && heartRate[s] > 0 ? 1 : 0))
  // Windows run in moving time, so a stop pauses the averages instead of emptying them for the
  // next five minutes; the lag also counts moving seconds.
  const movingS = movingBefore[totalS]
  const movingGap = new Float64Array(movingS + 1)
  const movingHeart = new Float64Array(movingS + 1)
  const movingHeartCount = new Float64Array(movingS + 1)
  for (let s = 0, k = 0; s < totalS; s++) {
    if (!moving[s]) continue
    movingGap[k + 1] = movingGap[k] + gap[s]
    movingHeart[k + 1] = movingHeart[k] + heartRate[s]
    movingHeartCount[k + 1] = movingHeartCount[k] + (heartRate[s] > 0 ? 1 : 0)
    k++
  }
  // Wall-clock end of the latest moving second at or before each grid end.
  const lastMovingEnd = new Int32Array(totalS + 1).fill(-STOP_BREAK_S - 1)
  for (let s = 0; s < totalS; s++) lastMovingEnd[s + 1] = moving[s] ? s + 1 : lastMovingEnd[s]
  const window = (
    end: number,
    windowS: number,
  ): { gapSpeedMps: number; heartRateBpm: number } | null => {
    if (end < 1 || end - lastMovingEnd[end] > STOP_BREAK_S) return null
    const rank = movingBefore[end]
    const heartFrom = rank - windowS
    const gapTo = rank - RUN_HEART_RATE_LAG_S
    const gapFrom = gapTo - windowS
    if (gapFrom < 0) return null
    const heartSeconds = movingHeartCount[rank] - movingHeartCount[heartFrom]
    if (heartSeconds < windowS * WINDOW_COVERAGE) return null
    return {
      gapSpeedMps: round((movingGap[gapTo] - movingGap[gapFrom]) / windowS, 4),
      heartRateBpm: round((movingHeart[rank] - movingHeart[heartFrom]) / heartSeconds, 2),
    }
  }
  // A recording gap spans the grid seconds from its last sample to the next one.
  const pauses = time
    .slice(1)
    .flatMap((at, index) => {
      const startS = Math.max(0, Math.floor(time[index]))
      const endS = Math.min(totalS, Math.floor(at))
      return endS - startS >= RECORDING_PAUSE_MIN_S ? [{ startS, endS }] : []
    })
    .sort((left, right) => right.endS - right.startS - (left.endS - left.startS))
    .slice(0, RUN_HEART_COST_MAX_PAUSES)
    .sort((left, right) => left.startS - right.startS)
  const step = Math.max(5, Math.ceil(totalS / (RUN_HEART_COST_MAX_POINTS - 2 - 2 * pauses.length)))
  // Points inside a pause would all land on its join, so the pause edges replace them.
  const ends: number[] = []
  for (let end = 0; end < totalS; end += step)
    if (!pauses.some(pause => end > pause.startS && end < pause.endS)) ends.push(end)
  ends.push(totalS)
  for (const pause of pauses) ends.push(pause.startS, pause.endS)
  ends.sort((left, right) => left - right)
  const points: RunHeartCostPoint[] = []
  for (const [position, end] of ends.entries()) {
    if (end === ends[position - 1]) continue
    const short = window(end, 30)
    const long = window(end, 300)
    const beats = heartSum[end] / 60
    const index = Math.min(totalS - 1, Math.max(0, end - 1))
    points.push({
      elapsedS: end,
      distanceKm: round(distanceKm[index], 4),
      elevationM: Number.isFinite(elevation[index]) ? round(elevation[index], 1) : null,
      gapSpeed30sMps: short?.gapSpeedMps ?? null,
      heartRate30sBpm: short?.heartRateBpm ?? null,
      gapSpeed5mMps: long?.gapSpeedMps ?? null,
      heartRate5mBpm: long?.heartRateBpm ?? null,
      cumulativeMetersPerBeat: jointSeconds[end] >= 60 ? round(jointGap[end] / beats, 4) : null,
    })
  }
  if (points.filter(point => point.gapSpeed30sMps != null).length < 2) return null
  return {
    source: 'strava',
    method: 'gap-per-beat-v1',
    heartRateLagS: RUN_HEART_RATE_LAG_S,
    points,
    pauses,
  }
}
