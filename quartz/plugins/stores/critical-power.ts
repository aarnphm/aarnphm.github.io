export const CRITICAL_POWER_DURATIONS_S = [180, 420, 720] as const

export type CriticalPowerWindow = 'six-weeks' | 'calendar-year'
export type CriticalPowerConfidence = 'medium' | 'provisional'

export interface CriticalPowerAnchor {
  durationS: number
  meanPowerWatts: number
  activityId: number
  activityDate: string
  startElapsedS: number
  endElapsedS: number
}

export interface CriticalPowerEstimate {
  criticalPowerWatts: number
  wPrimeJoules: number
  method: 'two-parameter-power-space'
  window: CriticalPowerWindow
  windowFrom: string
  windowTo: string
  anchors: CriticalPowerAnchor[]
  independentEffortCount: number
  rmseWatts: number
  normalizedRmse: number
  confidence: CriticalPowerConfidence
}

export interface CriticalPowerCurvePoint {
  s: number
  w: number
}

const MAX_NORMALIZED_RMSE = 0.05

const round = (value: number, digits: number): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

const selectAnchors = (
  candidates: readonly CriticalPowerAnchor[],
): CriticalPowerAnchor[] | null => {
  const anchors: CriticalPowerAnchor[] = []
  for (const durationS of CRITICAL_POWER_DURATIONS_S) {
    let best: CriticalPowerAnchor | null = null
    for (const candidate of candidates) {
      if (candidate.durationS !== durationS) continue
      if (!Number.isFinite(candidate.meanPowerWatts) || candidate.meanPowerWatts <= 0) continue
      if (candidate.endElapsedS - candidate.startElapsedS !== durationS) continue
      if (
        !best ||
        candidate.meanPowerWatts > best.meanPowerWatts ||
        (candidate.meanPowerWatts === best.meanPowerWatts &&
          (candidate.activityDate > best.activityDate ||
            (candidate.activityDate === best.activityDate &&
              candidate.activityId > best.activityId)))
      )
        best = candidate
    }
    if (!best) return null
    anchors.push(best)
  }
  return anchors
}

const independentEffortCount = (anchors: readonly CriticalPowerAnchor[]): number => {
  const byActivity = new Map<number, CriticalPowerAnchor[]>()
  for (const anchor of anchors) {
    const activity = byActivity.get(anchor.activityId) ?? []
    activity.push(anchor)
    byActivity.set(anchor.activityId, activity)
  }

  let count = 0
  for (const activity of byActivity.values()) {
    activity.sort(
      (left, right) =>
        left.endElapsedS - right.endElapsedS || left.startElapsedS - right.startElapsedS,
    )
    let end = -1
    for (const anchor of activity) {
      if (anchor.startElapsedS >= end) {
        count++
        end = anchor.endElapsedS
      }
    }
  }
  return count
}

export const fitCriticalPower = (
  candidates: readonly CriticalPowerAnchor[],
  window: CriticalPowerWindow,
  windowFrom: string,
  windowTo: string,
): CriticalPowerEstimate | null => {
  const anchors = selectAnchors(candidates)
  if (!anchors) return null

  const x = anchors.map(anchor => 1 / anchor.durationS)
  const y = anchors.map(anchor => anchor.meanPowerWatts)
  const xMean = x.reduce((sum, value) => sum + value, 0) / x.length
  const yMean = y.reduce((sum, value) => sum + value, 0) / y.length
  let covariance = 0
  let variance = 0
  for (let index = 0; index < anchors.length; index++) {
    covariance += (x[index] - xMean) * (y[index] - yMean)
    variance += (x[index] - xMean) ** 2
  }
  if (variance <= 0) return null

  const wPrimeJoules = covariance / variance
  const criticalPowerWatts = yMean - wPrimeJoules * xMean
  if (
    !Number.isFinite(criticalPowerWatts) ||
    !Number.isFinite(wPrimeJoules) ||
    criticalPowerWatts <= 0 ||
    wPrimeJoules <= 0 ||
    criticalPowerWatts >= Math.min(...y)
  )
    return null

  const squaredError = anchors.reduce((sum, anchor) => {
    const predicted = criticalPowerWatts + wPrimeJoules / anchor.durationS
    return sum + (anchor.meanPowerWatts - predicted) ** 2
  }, 0)
  const rmseWatts = Math.sqrt(squaredError / anchors.length)
  const normalizedRmse = rmseWatts / yMean
  if (!Number.isFinite(normalizedRmse) || normalizedRmse > MAX_NORMALIZED_RMSE) return null

  const independent = independentEffortCount(anchors)
  return {
    criticalPowerWatts: round(criticalPowerWatts, 1),
    wPrimeJoules: Math.round(wPrimeJoules),
    method: 'two-parameter-power-space',
    window,
    windowFrom,
    windowTo,
    anchors,
    independentEffortCount: independent,
    rmseWatts: round(rmseWatts, 2),
    normalizedRmse: round(normalizedRmse, 4),
    confidence: independent >= anchors.length ? 'medium' : 'provisional',
  }
}

export const criticalPowerAtDuration = (
  estimate: CriticalPowerEstimate,
  durationS: number,
): number => estimate.criticalPowerWatts + estimate.wPrimeJoules / durationS

export const criticalPowerCurve = (
  estimate: CriticalPowerEstimate,
  minDurationS: number,
  maxDurationS: number,
  samples = 32,
): CriticalPowerCurvePoint[] => {
  const min = Math.max(CRITICAL_POWER_DURATIONS_S[0], minDurationS)
  const max = Math.min(CRITICAL_POWER_DURATIONS_S.at(-1) ?? min, maxDurationS)
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return []
  const count = Math.max(2, Math.round(samples))
  const logMin = Math.log(min)
  const logSpan = Math.log(max) - logMin
  return Array.from({ length: count }, (_, index) => {
    const fraction = index / (count - 1)
    const s = index === 0 ? min : index === count - 1 ? max : Math.exp(logMin + fraction * logSpan)
    return { s, w: criticalPowerAtDuration(estimate, s) }
  })
}
