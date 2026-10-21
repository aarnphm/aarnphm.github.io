import type { SwimActivityInterval } from '../../plugins/stores/strava'
import {
  swimLengthMetrics,
  swimPaceSeconds,
  swimStrokeRate,
  type SwimChartMetric,
} from '../swim-metrics'
import { niceStep } from './chart-geometry'
import { clock, positiveMetric, swimDistanceLabel, swimTrendNumber } from './metric-text'

export type SwimActivityObservation = {
  interval: SwimActivityInterval
  index: number
  strokesPerLength: number | null
  swolf: number | null
}

export type SwimActivityBlock = SwimActivityInterval & {
  strokesPerLength: number | null
  swolf: number | null
}

export type SwimActivityMetric = { observation: SwimActivityObservation; value: number }

export type SwimTrendMode = 'lengths' | '100m'

export type SwimTrendChartPoint = {
  elapsedS: number
  cumulativeDistanceM: number
  value: number
  xPct: number
  yPct: number
  windowStartDistanceM?: number
}

export type SwimTrendHover = SwimTrendChartPoint & { index: number }

export const swimTrendHoverAt = (
  points: SwimTrendChartPoint[],
  fraction: number,
): SwimTrendHover | null => {
  if (points.length === 0) return null
  const xPct = Math.max(0, Math.min(100, (Number.isFinite(fraction) ? fraction : 0) * 100))
  let index = 0
  let distance = Math.abs(points[0].xPct - xPct)
  for (let candidate = 1; candidate < points.length; candidate++) {
    const candidateDistance = Math.abs(points[candidate].xPct - xPct)
    if (candidateDistance < distance) {
      index = candidate
      distance = candidateDistance
    }
  }
  return { ...points[index], index }
}

const swimRoundTenth = (value: number): number => Math.round(value * 10) / 10

export const swimActivityBlocks = (
  intervals: SwimActivityInterval[],
  blockDistanceM = 100,
): SwimActivityBlock[] => {
  if (!Number.isFinite(blockDistanceM) || blockDistanceM <= 0) return []
  const blocks: SwimActivityBlock[] = []
  let cumulativeDistanceM = 0
  let distanceM = 0
  let durationS = 0
  let strokeCount = 0
  let strokeTimeS = 0
  let strokesPerLength = 0
  let swolf = 0
  let measuredLengthWeight = 0
  let startElapsedS = 0
  let endElapsedS = 0
  let strokeComplete = true
  const flush = (): void => {
    if (distanceM <= 0) return
    const paceSPer100m = swimPaceSeconds(distanceM, durationS)
    const strokeRateSpm = strokeComplete ? swimStrokeRate(strokeCount, strokeTimeS) : null
    blocks.push({
      startElapsedS: swimRoundTenth(startElapsedS),
      endElapsedS: swimRoundTenth(endElapsedS),
      distanceM: swimRoundTenth(distanceM),
      durationS: swimRoundTenth(durationS),
      cumulativeDistanceM: swimRoundTenth(cumulativeDistanceM),
      paceSPer100m,
      strokeCount: strokeRateSpm == null ? null : swimRoundTenth(strokeCount),
      strokeTimeS: strokeRateSpm == null ? null : swimRoundTenth(strokeTimeS),
      strokeRateSpm,
      stroke: null,
      strokesPerLength:
        measuredLengthWeight > 0 ? swimRoundTenth(strokesPerLength / measuredLengthWeight) : null,
      swolf: measuredLengthWeight > 0 ? swimRoundTenth(swolf / measuredLengthWeight) : null,
    })
    distanceM = 0
    durationS = 0
    strokeCount = 0
    strokeTimeS = 0
    strokesPerLength = 0
    swolf = 0
    measuredLengthWeight = 0
    strokeComplete = true
  }
  for (const interval of intervals) {
    if (
      !positiveMetric(interval.distanceM) ||
      !positiveMetric(interval.durationS) ||
      interval.endElapsedS <= interval.startElapsedS
    )
      continue
    let consumedDistanceM = 0
    while (consumedDistanceM < interval.distanceM - 0.0001) {
      const remainingDistanceM = interval.distanceM - consumedDistanceM
      const availableDistanceM = blockDistanceM - distanceM
      const contributionDistanceM = Math.min(remainingDistanceM, availableDistanceM)
      const startFraction = consumedDistanceM / interval.distanceM
      const endFraction = (consumedDistanceM + contributionDistanceM) / interval.distanceM
      const contributionDurationS =
        interval.durationS * (contributionDistanceM / interval.distanceM)
      if (distanceM === 0)
        startElapsedS =
          startFraction === 0
            ? interval.startElapsedS
            : interval.startElapsedS + interval.durationS * startFraction
      endElapsedS =
        endFraction >= 1
          ? interval.endElapsedS
          : interval.startElapsedS + interval.durationS * endFraction
      distanceM += contributionDistanceM
      durationS += contributionDurationS
      cumulativeDistanceM += contributionDistanceM
      const fraction = contributionDistanceM / interval.distanceM
      const lengthMetrics = swimLengthMetrics(interval)
      if (lengthMetrics) {
        strokesPerLength += lengthMetrics.strokesPerLength * fraction
        swolf += lengthMetrics.swolf * fraction
        measuredLengthWeight += fraction
      }
      if (interval.stroke !== 'kickboard') {
        if (positiveMetric(interval.strokeCount) && positiveMetric(interval.strokeTimeS)) {
          strokeCount += interval.strokeCount * fraction
          strokeTimeS += interval.strokeTimeS * fraction
        } else {
          strokeComplete = false
        }
      }
      consumedDistanceM += contributionDistanceM
      if (distanceM >= blockDistanceM - 0.0001) flush()
    }
  }
  flush()
  return blocks
}

export const swimTrendDisplayValue = (kind: SwimChartMetric, value: number): string =>
  kind === 'pace'
    ? `${clock(value)} /100m`
    : kind === 'cadence'
      ? `${swimTrendNumber(value)} str/length`
      : kind === 'rate'
        ? `${swimTrendNumber(value)} spm`
        : `${Math.round(value)} SWOLF`

export const swimTrendHeaderValue = (kind: SwimChartMetric, value: number): string =>
  kind === 'pace'
    ? clock(value)
    : kind === 'cadence' || kind === 'rate'
      ? swimTrendNumber(value)
      : Math.round(value).toLocaleString('en-US')

export const swimTrendTitle = (kind: SwimChartMetric): string =>
  kind === 'pace'
    ? 'pace /100m'
    : kind === 'cadence'
      ? 'cadence str/length'
      : kind === 'rate'
        ? 'stroke rate spm'
        : 'SWOLF'

export const SWIM_TREND_GLOSS: Partial<Record<SwimChartMetric, string>> = {
  rate: 'strokerate',
  cadence: 'swimcadence',
  swolf: 'swolf',
}

export const swimTrendLabel = (kind: SwimChartMetric): string =>
  kind === 'pace'
    ? 'pace'
    : kind === 'cadence'
      ? 'cadence'
      : kind === 'rate'
        ? 'stroke rate'
        : 'SWOLF'

export const swimActivityPointLabel = (
  point: Pick<SwimTrendChartPoint, 'elapsedS' | 'cumulativeDistanceM' | 'windowStartDistanceM'>,
): string => {
  const distance =
    point.windowStartDistanceM == null
      ? swimDistanceLabel(point.cumulativeDistanceM)
      : `${Math.round(point.windowStartDistanceM).toLocaleString('en-US')}–${Math.round(point.cumulativeDistanceM).toLocaleString('en-US')} m`
  return `${distance} · ${clock(point.elapsedS)} elapsed`
}

export const swimTrendAriaValue = (
  kind: SwimChartMetric,
  point: Pick<
    SwimTrendChartPoint,
    'elapsedS' | 'cumulativeDistanceM' | 'value' | 'windowStartDistanceM'
  >,
): string => {
  const position =
    point.windowStartDistanceM == null
      ? `${Math.round(point.cumulativeDistanceM)} metres, ${clock(point.elapsedS)} elapsed`
      : `${Math.round(point.cumulativeDistanceM - point.windowStartDistanceM)} metre block from ${Math.round(point.windowStartDistanceM)} to ${Math.round(point.cumulativeDistanceM)} metres, ${clock(point.elapsedS)} elapsed`
  if (kind === 'pace') return `${position}, swim pace ${clock(point.value)} per 100 metres`
  if (kind === 'cadence')
    return `${position}, swim cadence ${swimTrendNumber(point.value)} strokes per length`
  if (kind === 'rate')
    return `${position}, stroke rate ${swimTrendNumber(point.value)} strokes per minute`
  return `${position}, SWOLF score ${Math.round(point.value)}`
}

export const swimTrendDomain = (
  values: number[],
  kind: SwimChartMetric,
): { min: number; max: number } => {
  const observedMin = Math.min(...values)
  const observedMax = Math.max(...values)
  if (kind === 'pace') {
    const step = Math.max(1, niceStep(observedMax, 3))
    return { min: 0, max: Math.max(step, Math.ceil(observedMax / step) * step) }
  }
  const span = observedMax - observedMin
  const step = niceStep(span > 0 ? span : Math.max(1, observedMax * 0.1), 3)
  const min = Math.max(0, Math.floor(observedMin / step) * step)
  const max = Math.ceil(observedMax / step) * step
  return { min, max: max > min ? max : min + step }
}
