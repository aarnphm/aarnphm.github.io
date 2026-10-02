import { shiftIsoDay } from './local-date'
import { swimPaceSeconds } from './swim-metrics'

export const SWIM_POWER_REFERENCE_PACE_S_PER_100M = 150
export const SWIM_POWER_BIN_SIZE = 25
const MIN_CURVE_DURATION_S = 60
const TIMESTAMP_TOLERANCE_S = 1
const CURVE_DURATIONS_S = [
  60, 90, 120, 180, 240, 300, 360, 480, 600, 720, 900, 1200, 1500, 1800, 2700, 3600, 5400, 7200,
]

export interface SwimPowerIntervalInput {
  startElapsedS: number
  endElapsedS: number
  durationS: number
  distanceM: number
  stroke: string | null
}

export interface SwimPowerCurvePoint {
  durationS: number
  index: number
  startElapsedS: number
  endElapsedS: number
}

export interface SwimPowerEstimate {
  source: 'garden-estimate'
  inputSource: 'garmin' | 'apple' | 'route'
  method: 'freestyle-drag-index-v1' | 'open-water-drag-index-v1'
  speedBasis: 'pool-length' | 'ground-speed'
  referencePaceSPer100m: number
  averageIndex: number
  activeTimeS: number
  distanceM: number
  validIntervalCount: number
  excludedIntervalCount: number
  histogramS: number[]
  curve: SwimPowerCurvePoint[]
}

export interface SwimPowerBestPoint extends SwimPowerCurvePoint {
  activityId: number
  activityDate: string
  inputSource: SwimPowerEstimate['inputSource']
}

export interface SwimPowerCurveBlock {
  referencePaceSPer100m: number
  yearLabel: number
  activityCount: number
  sixWeeks: SwimPowerBestPoint[]
  year: SwimPowerBestPoint[]
}

const rounded = (value: number): number => Math.round(value * 1000) / 1000

interface Segment {
  interval: SwimPowerIntervalInput
  index: number
}

// Each observed interval supplies a constant estimate. Missing intervals must
// never join two sustained efforts.
function bestWindows(run: readonly Segment[]): SwimPowerCurvePoint[] {
  const times = [0]
  const integrals = [0]
  for (const segment of run) {
    times.push(times[times.length - 1] + segment.interval.durationS)
    integrals.push(integrals[integrals.length - 1] + segment.index * segment.interval.durationS)
  }
  const duration = times[times.length - 1]
  if (duration < MIN_CURVE_DURATION_S) return []
  const integralAt = (time: number): number => {
    let lo = 0
    let hi = times.length - 1
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2)
      if (times[mid] <= time) lo = mid
      else hi = mid - 1
    }
    return integrals[lo] + (time - times[lo]) * (run[lo]?.index ?? 0)
  }
  const elapsedAt = (time: number): number => {
    const index = times.findIndex((end, i) => i > 0 && end >= time) - 1
    const segment = run[Math.max(0, index)]
    return segment.interval.startElapsedS + time - times[Math.max(0, index)]
  }
  const points: SwimPowerCurvePoint[] = []
  for (const windowS of CURVE_DURATIONS_S) {
    if (windowS > duration) break
    // An extremum of a piecewise-linear moving integral occurs when either
    // window edge meets a length boundary, including fractional timestamps.
    const candidates = new Set([0, duration - windowS])
    for (const boundary of times) {
      if (boundary <= duration - windowS) candidates.add(boundary)
      if (boundary >= windowS) candidates.add(boundary - windowS)
    }
    let best = -Infinity
    let start = 0
    for (const candidate of [...candidates].sort((a, b) => a - b)) {
      const value = (integralAt(candidate + windowS) - integralAt(candidate)) / windowS
      if (value > best + 1e-9) {
        best = value
        start = candidate
      }
    }
    points.push({
      durationS: windowS,
      index: rounded(best),
      startElapsedS: rounded(elapsedAt(start)),
      endElapsedS: rounded(elapsedAt(start + windowS)),
    })
  }
  return points
}

export function buildSwimPowerEstimate(
  inputSource: SwimPowerEstimate['inputSource'],
  intervals: readonly SwimPowerIntervalInput[],
  speedBasis: SwimPowerEstimate['speedBasis'] = 'pool-length',
): SwimPowerEstimate | null {
  const best = new Map<number, SwimPowerCurvePoint>()
  let run: Segment[] = []
  let activeTimeS = 0
  let distanceM = 0
  let weightedIndex = 0
  let validIntervalCount = 0
  const histogramS: number[] = []
  const finishRun = (): void => {
    for (const point of bestWindows(run))
      if (point.index > (best.get(point.durationS)?.index ?? -Infinity))
        best.set(point.durationS, point)
    run = []
  }
  for (const interval of intervals) {
    const pace = swimPaceSeconds(interval.distanceM, interval.durationS)
    if (
      (speedBasis === 'pool-length' && interval.stroke !== 'freestyle') ||
      pace == null ||
      !Number.isFinite(interval.startElapsedS) ||
      interval.startElapsedS < 0 ||
      !Number.isFinite(interval.endElapsedS) ||
      Math.abs(interval.endElapsedS - interval.startElapsedS - interval.durationS) >
        TIMESTAMP_TOLERANCE_S
    ) {
      finishRun()
      continue
    }
    const previous = run.at(-1)?.interval
    if (previous && Math.abs(interval.startElapsedS - previous.endElapsedS) > TIMESTAMP_TOLERANCE_S)
      finishRun()
    const index =
      100 *
      (interval.distanceM / interval.durationS / (100 / SWIM_POWER_REFERENCE_PACE_S_PER_100M)) ** 3
    run.push({ interval, index })
    const bin = Math.floor(index / SWIM_POWER_BIN_SIZE)
    while (histogramS.length <= bin) histogramS.push(0)
    histogramS[bin] += interval.durationS
    activeTimeS += interval.durationS
    distanceM += interval.distanceM
    weightedIndex += index * interval.durationS
    validIntervalCount++
  }
  finishRun()
  if (validIntervalCount === 0) return null
  return {
    source: 'garden-estimate',
    inputSource,
    method: speedBasis === 'ground-speed' ? 'open-water-drag-index-v1' : 'freestyle-drag-index-v1',
    speedBasis,
    referencePaceSPer100m: SWIM_POWER_REFERENCE_PACE_S_PER_100M,
    averageIndex: rounded(weightedIndex / activeTimeS),
    activeTimeS: rounded(activeTimeS),
    distanceM: rounded(distanceM),
    validIntervalCount,
    excludedIntervalCount: intervals.length - validIntervalCount,
    histogramS: histogramS.map(rounded),
    curve: [...best.values()].sort((a, b) => a.durationS - b.durationS),
  }
}

export function openWaterPowerIntervals(
  route: readonly { elapsedS: number; d: number }[],
): SwimPowerIntervalInput[] {
  const intervals: SwimPowerIntervalInput[] = []
  let start = 0,
    seconds = 0,
    metres = 0
  const finish = (): void => {
    if (seconds > 0)
      intervals.push({
        startElapsedS: start,
        endElapsedS: start + seconds,
        durationS: seconds,
        distanceM: metres,
        stroke: null,
      })
    seconds = 0
    metres = 0
  }
  for (let i = 1; i < route.length; i++) {
    const previous = route[i - 1],
      point = route[i]
    const duration = point.elapsedS - previous.elapsedS
    const distance = (point.d - previous.d) * 1000
    if (duration > 120 || swimPaceSeconds(distance, duration) == null) {
      finish()
      intervals.push({
        startElapsedS: previous.elapsedS,
        endElapsedS: point.elapsedS,
        durationS: duration,
        distanceM: 0,
        stroke: null,
      })
      continue
    }
    // Average GPS speed before cubing it so position noise cannot inflate drag demand.
    let elapsed = previous.elapsedS
    while (elapsed < point.elapsedS) {
      if (seconds === 0) start = elapsed
      const portion = Math.min(60 - seconds, point.elapsedS - elapsed)
      seconds += portion
      metres += (distance * portion) / duration
      elapsed += portion
      if (seconds >= 60) finish()
    }
  }
  finish()
  return intervals
}

export function buildSwimPowerCurveBlock(
  activities: readonly { id: number; date: string; swimPower?: SwimPowerEstimate | null }[],
  today: string,
): SwimPowerCurveBlock {
  const yearLabel = Number(today.slice(0, 4))
  const yearFrom = `${yearLabel}-01-01`
  const recentFrom = shiftIsoDay(today, -41)
  const sixWeeks = new Map<number, SwimPowerBestPoint>()
  const year = new Map<number, SwimPowerBestPoint>()
  let activityCount = 0
  for (const activity of [...activities].sort(
    (a, b) => b.date.localeCompare(a.date) || b.id - a.id,
  )) {
    if (
      activity.date > today ||
      (activity.date < yearFrom && activity.date < recentFrom) ||
      !activity.swimPower
    )
      continue
    if (activity.swimPower.curve.length > 0) activityCount++
    for (const point of activity.swimPower.curve) {
      const candidate = {
        ...point,
        activityId: activity.id,
        activityDate: activity.date,
        inputSource: activity.swimPower.inputSource,
      }
      const targets = [
        ...(activity.date >= recentFrom ? [sixWeeks] : []),
        ...(activity.date >= yearFrom ? [year] : []),
      ]
      for (const target of targets)
        if (candidate.index > (target.get(point.durationS)?.index ?? -Infinity))
          target.set(point.durationS, candidate)
    }
  }
  return {
    referencePaceSPer100m: SWIM_POWER_REFERENCE_PACE_S_PER_100M,
    yearLabel,
    activityCount,
    sixWeeks: [...sixWeeks.values()].sort((a, b) => a.durationS - b.durationS),
    year: [...year.values()].sort((a, b) => a.durationS - b.durationS),
  }
}
