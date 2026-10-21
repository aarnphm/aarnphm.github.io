import type { GarminLactateThresholdValue } from '../garmin'
import { runGradeFactors } from '../../../util/run-grade'
import { round, type RawStravaActivity, type Sport, type StravaStreams } from '../strava'
import {
  clamp,
  DAY_MS,
  dayMs,
  mean,
  median,
  mmss,
  olsSlope,
  RUN_MOVING_MIN_MPS,
  sd,
  SPORT_PRIOR,
  SPRINT_CAP_MS,
  weightedPercentile,
  type Act,
  type Conf,
  type RunHeartRateAnchor,
  type ThresholdEstimate,
} from './primitives'

export type LactateThresholdHeartRate =
  | { value: number; unit: 'bpm'; source: 'declared' }
  | { value: number; unit: 'bpm'; source: 'garmin'; sport: 'run'; date: string }

/** Garmin's running threshold pace, checked against the garden estimate before use. */
export interface LactateThresholdDeviceEstimate {
  value: number
  date: string
  /** Positive when Garmin's pace is faster than the garden estimate. */
  deltaPct: number | null
  accepted: boolean
}

export interface LactateThresholdSportEstimate {
  sport: Sport
  unit: 's/100m' | 's/km' | 'W'
  value: number
  conf: Conf | null
  source:
    | ThresholdEstimate['method']
    | 'garmin'
    | 'critical-power'
    | 'critical-power-year'
    | 'twenty-minute-power'
    | 'declared-ftp'
  date: string | null
  sampleSize: number
  heartRateAnchor: RunHeartRateAnchor | null
  device: LactateThresholdDeviceEstimate | null
}

export interface LactateThresholdBlock {
  heartRate: LactateThresholdHeartRate | null
  heartRateSeries: GarminLactateThresholdValue[]
  sports: LactateThresholdSportEstimate[]
}

const RUN_LT_WINDOW_S = 8 * 60
const RUN_LT_STEP_S = 60
const RUN_LT_CHUNK_S = 60
// Bootstrapping runs: 56 days with these tolerances keeps the 80% range near ±12 s/km. Tighter
// tolerances leave too few windows and the slope swings by a minute per km.
const RUN_LT_LOOKBACK_DAYS = 56
const RUN_LT_MAX_PACE_CV = 0.2
const RUN_LT_MAX_HR_SD = 10
const RUN_LT_MIN_WINDOWS = 20
const RUN_LT_MIN_RUNS = 4
const RUN_LT_MIN_HR_SPAN = 8
const RUN_LT_MAX_EXTRAPOLATION_BPM = 20
const RUN_LT_FIRM_EXTRAPOLATION_BPM = 5

const isTreadmillRun = (a: RawStravaActivity): boolean =>
  a.trainer === true || /\btreadmill\b/i.test(a.name)

/** Steady 8-minute windows of one run: mean heart rate and grade-adjusted speed. */
function steadyRunWindows(stream: StravaStreams): { heartRate: number; gapSpeed: number }[] {
  const { time, distance, heartrate } = stream
  if (!time || !heartrate || time.length !== distance.length || heartrate.length !== time.length)
    return []
  const factors = runGradeFactors(stream)
  const n = time.length
  const windows: { heartRate: number; gapSpeed: number }[] = []
  let end = 0
  let nextStartS = -Infinity
  for (let start = 0; start < n; start++) {
    if (time[start] < nextStartS) continue
    nextStartS = time[start] + RUN_LT_STEP_S
    while (end < n && time[end] - time[start] < RUN_LT_WINDOW_S) end++
    if (end >= n) break
    const elapsedS = time[end] - time[start]
    // A pause inside the window leaves heart rate and pace out of step.
    if (elapsedS > RUN_LT_WINDOW_S * 1.1) continue

    const beats: number[] = []
    for (let i = start; i < end; i++) if (heartrate[i] > 0) beats.push(heartrate[i])
    if (beats.length < 0.9 * (end - start) || sd(beats) > RUN_LT_MAX_HR_SD) continue

    const chunkSpeeds: number[] = []
    let chunkStart = start
    for (let i = start + 1; i <= end; i++) {
      if (time[i] - time[chunkStart] < RUN_LT_CHUNK_S) continue
      if (time[i] > time[chunkStart])
        chunkSpeeds.push((distance[i] - distance[chunkStart]) / (time[i] - time[chunkStart]))
      chunkStart = i
    }
    const chunkMean = mean(chunkSpeeds)
    if (
      chunkSpeeds.length < 3 ||
      !(chunkMean > 0) ||
      sd(chunkSpeeds) / chunkMean > RUN_LT_MAX_PACE_CV
    )
      continue

    let weighted = 0
    let length = 0
    for (let i = start + 1; i <= end; i++) {
      const segment = distance[i] - distance[i - 1]
      if (!(segment > 0)) continue
      weighted += (factors?.[i] ?? 1) * segment
      length += segment
    }
    if (!(length > 0)) continue
    const gapSpeed = (length / elapsedS) * (weighted / length)
    if (gapSpeed < RUN_MOVING_MIN_MPS || gapSpeed > SPRINT_CAP_MS.run) continue
    windows.push({ heartRate: mean(beats), gapSpeed })
  }
  return windows
}

/**
 * Grade-adjusted speed at the lactate-threshold heart rate, from a line through steady outdoor
 * windows of recent runs. Heart rate and speed are close to linear below threshold.
 */
function runSpeedAtThresholdHeartRate(
  runs: readonly Act[],
  streams: Readonly<Record<string, StravaStreams>> | undefined,
  heartRateBpm: number,
  today: number,
): { vThr: number; conf: Conf; anchor: RunHeartRateAnchor } | null {
  const cutoff = today - RUN_LT_LOOKBACK_DAYS * DAY_MS
  const heartRates: number[] = []
  const speeds: number[] = []
  let runCount = 0
  for (const run of runs) {
    if (dayMs(run.day) < cutoff || isTreadmillRun(run.a)) continue
    const stream = streams?.[String(run.a.id)]
    if (!stream) continue
    const windows = steadyRunWindows(stream)
    if (!windows.length) continue
    runCount++
    for (const window of windows) {
      heartRates.push(window.heartRate)
      speeds.push(window.gapSpeed)
    }
  }
  if (speeds.length < RUN_LT_MIN_WINDOWS || runCount < RUN_LT_MIN_RUNS) return null
  const maxHeartRate = Math.max(...heartRates)
  if (
    maxHeartRate - Math.min(...heartRates) < RUN_LT_MIN_HR_SPAN ||
    heartRateBpm - maxHeartRate > RUN_LT_MAX_EXTRAPOLATION_BPM
  )
    return null
  const slope = olsSlope(heartRates, speeds)
  if (slope == null || !(slope > 0)) return null
  const vThr = mean(speeds) + slope * (heartRateBpm - mean(heartRates))
  if (!(vThr >= RUN_MOVING_MIN_MPS) || vThr > SPRINT_CAP_MS.run) return null
  return {
    vThr,
    conf: heartRateBpm - maxHeartRate <= RUN_LT_FIRM_EXTRAPOLATION_BPM ? 'firm' : 'low',
    anchor: {
      heartRateBpm,
      windows: speeds.length,
      runs: runCount,
      maxWindowHeartRateBpm: Math.round(maxHeartRate),
      lookbackDays: RUN_LT_LOOKBACK_DAYS,
    },
  }
}

export function estimateThreshold(
  acts: Act[],
  sport: Sport,
  today: number,
  runAnchor?: {
    streams: Readonly<Record<string, StravaStreams>> | undefined
    heartRateBpm: number | null
  },
): ThresholdEstimate {
  const mine = acts.filter(x => x.sport === sport)
  const lastDay = mine.length ? mine[mine.length - 1].day : null
  const staleDays = lastDay ? Math.round((today - dayMs(lastDay)) / DAY_MS) : 0
  const n = mine.length
  const anchored =
    sport === 'run' && runAnchor?.heartRateBpm != null
      ? runSpeedAtThresholdHeartRate(mine, runAnchor.streams, runAnchor.heartRateBpm, today)
      : null
  let vThr: number
  let conf: Conf
  let method: ThresholdEstimate['method']
  if (anchored) {
    vThr = anchored.vThr
    conf = anchored.conf
    method = 'heart-rate-anchored'
  } else if (n >= 4) {
    const values = mine.map(x => x.vGap)
    const weights = mine.map(x => Math.floor(x.a.movingTime / 600) + 1)
    vThr = weightedPercentile(values, weights, 0.9)
    conf = 'firm'
    method = 'pace-p90'
  } else if (n >= 2) {
    vThr = 0.97 * Math.max(...mine.map(x => x.vGap))
    conf = 'low'
    method = 'pace-max'
  } else {
    vThr = SPORT_PRIOR[sport]
    conf = 'prior'
    method = 'prior'
  }
  if (sport === 'run' && staleDays > 45) conf = 'stale'

  let paceLabel: string
  let unit: string
  if (sport === 'swim') {
    unit = 's/100m'
    paceLabel = mmss(100 / vThr)
  } else if (sport === 'run') {
    unit = 's/km'
    paceLabel = mmss(1000 / vThr)
  } else {
    unit = 'km/h'
    paceLabel = String(round(vThr * 3.6, 0))
  }
  return {
    sport,
    vThr: round(vThr, 4),
    paceLabel,
    unit,
    conf,
    method,
    sampleSize: n,
    staleDays,
    heartRateAnchor: anchored?.anchor ?? null,
  }
}

const TREND_MIN_SESSIONS = 6
export const TREND_MIN_SPAN_DAYS = 21
// Within-session slopes outside 0.5–2 % per beat come from too few or too similar windows.
const TREND_HR_SLOPE_MIN = 0.005
const TREND_HR_SLOPE_MAX = 0.02

const RIDE_STEADY_MIN_WATTS = 50
const RIDE_STEADY_MAX_POWER_CV = 0.25
const RIDE_STEADY_MAX_COAST = 0.1

/** Steady 8-minute windows of one ride with a power meter: mean heart rate and mean power. */
function steadyRideWindows(stream: StravaStreams): { heartRate: number; watts: number }[] {
  const { time, heartrate, watts } = stream
  if (
    !time ||
    !heartrate ||
    !watts ||
    heartrate.length !== time.length ||
    watts.length !== time.length
  )
    return []
  const n = time.length
  const windows: { heartRate: number; watts: number }[] = []
  let end = 0
  let nextStartS = -Infinity
  for (let start = 0; start < n; start++) {
    if (time[start] < nextStartS) continue
    nextStartS = time[start] + RUN_LT_STEP_S
    while (end < n && time[end] - time[start] < RUN_LT_WINDOW_S) end++
    if (end >= n) break
    if (time[end] - time[start] > RUN_LT_WINDOW_S * 1.1) continue

    const beats: number[] = []
    let coasting = 0
    for (let i = start; i < end; i++) {
      if (heartrate[i] > 0) beats.push(heartrate[i])
      if (!(watts[i] > 0)) coasting++
    }
    if (beats.length < 0.9 * (end - start) || sd(beats) > RUN_LT_MAX_HR_SD) continue
    if (coasting > RIDE_STEADY_MAX_COAST * (end - start)) continue

    const chunks: number[] = []
    let chunkStart = start
    let chunkSum = 0
    let total = 0
    for (let i = start; i < end; i++) {
      const w = watts[i] > 0 ? watts[i] : 0
      chunkSum += w
      total += w
      if (time[i] - time[chunkStart] < RUN_LT_CHUNK_S) continue
      chunks.push(chunkSum / (i - chunkStart + 1))
      chunkStart = i + 1
      chunkSum = 0
    }
    const chunkMean = mean(chunks)
    if (
      chunks.length < 3 ||
      !(chunkMean > RIDE_STEADY_MIN_WATTS) ||
      sd(chunks) / chunkMean > RIDE_STEADY_MAX_POWER_CV
    )
      continue
    windows.push({ heartRate: mean(beats), watts: total / (end - start) })
  }
  return windows
}

export interface TrendObservation {
  day: string
  /** Log speed or log power; at the reference heart rate when the trend is heart-rate anchored. */
  y: number
  /** Sampling variance of y from the steady windows. */
  windowVar: number
  /** Steady-window mean before the heart-rate correction, in log units. */
  steady: number | null
  heartRate: number | null
  windows: number
}

/**
 * Sessions as speed or power at one reference heart rate. A pooled within-session slope through the
 * steady windows removes how hard each session was, so the series follows fitness instead of the mix
 * of easy and hard days. The reference is the median window heart rate, which keeps extrapolation short.
 */
export function heartRateTrendObservations(
  mine: readonly Act[],
  sport: Sport,
  streams: Readonly<Record<string, StravaStreams>> | undefined,
): { observations: TrendObservation[]; referenceHeartRate: number; slope: number } | null {
  if (sport === 'swim' || !streams) return null
  const sessions: { day: string; heartRates: number[]; logs: number[] }[] = []
  for (const act of mine) {
    // Strava's estimated watts follow speed and grade, so only power-meter rides carry power.
    if (sport === 'run' ? isTreadmillRun(act.a) : act.a.deviceWatts !== true) continue
    const stream = streams[String(act.a.id)]
    if (!stream) continue
    const windows =
      sport === 'run'
        ? steadyRunWindows(stream).map(w => ({ heartRate: w.heartRate, value: w.gapSpeed }))
        : steadyRideWindows(stream).map(w => ({ heartRate: w.heartRate, value: w.watts }))
    if (!windows.length) continue
    sessions.push({
      day: act.day,
      heartRates: windows.map(w => w.heartRate),
      logs: windows.map(w => Math.log(w.value)),
    })
  }
  if (sessions.length < TREND_MIN_SESSIONS) return null
  const span = (dayMs(sessions[sessions.length - 1].day) - dayMs(sessions[0].day)) / DAY_MS
  if (span < TREND_MIN_SPAN_DAYS) return null

  const centred = sessions.map(s => ({ ...s, hr: mean(s.heartRates), log: mean(s.logs) }))
  let sxx = 0
  let sxy = 0
  let count = 0
  for (const s of centred) {
    for (let i = 0; i < s.logs.length; i++) {
      sxx += (s.heartRates[i] - s.hr) ** 2
      sxy += (s.heartRates[i] - s.hr) * (s.logs[i] - s.log)
    }
    count += s.logs.length
  }
  if (!(sxx > 0)) return null
  const slope = clamp(sxy / sxx, TREND_HR_SLOPE_MIN, TREND_HR_SLOPE_MAX)
  let rss = 0
  for (const s of centred)
    for (let i = 0; i < s.logs.length; i++)
      rss += (s.logs[i] - s.log - slope * (s.heartRates[i] - s.hr)) ** 2
  const windowVar = rss / Math.max(1, count - centred.length - 1)
  const referenceHeartRate = 5 * Math.round(median(centred.flatMap(s => s.heartRates)) / 5)
  return {
    observations: centred.map(s => ({
      day: s.day,
      y: s.log + slope * (referenceHeartRate - s.hr),
      windowVar: windowVar / s.logs.length,
      steady: s.log,
      heartRate: s.hr,
      windows: s.logs.length,
    })),
    referenceHeartRate,
    slope,
  }
}
