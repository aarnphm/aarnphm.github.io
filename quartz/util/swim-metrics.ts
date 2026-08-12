export const SWIM_PACE_MIN_S_PER_100M = 45
export const SWIM_PACE_MAX_S_PER_100M = 360
export const SWIM_STROKE_RATE_MIN_SPM = 5
export const SWIM_STROKE_RATE_MAX_SPM = 100

export type SwimChartMetric = 'pace' | 'cadence' | 'swolf'

export interface SwimLengthInput {
  durationS: number
  strokeCount: number | null
  stroke?: string | null
}

export interface SwimLengthMetrics {
  strokesPerLength: number
  swolf: number
}

const finitePositive = (value: number): boolean => Number.isFinite(value) && value > 0

export const swimPaceSeconds = (distanceM: number, activeTimeS: number): number | null => {
  if (!finitePositive(distanceM) || !finitePositive(activeTimeS)) return null
  const pace = (activeTimeS / distanceM) * 100
  if (pace < SWIM_PACE_MIN_S_PER_100M || pace > SWIM_PACE_MAX_S_PER_100M) return null
  return Math.round(pace * 10) / 10
}

export const swimStrokeRate = (strokeCount: number, strokeTimeS: number): number | null => {
  if (!finitePositive(strokeCount) || !finitePositive(strokeTimeS)) return null
  const rate = (strokeCount / strokeTimeS) * 60
  if (rate < SWIM_STROKE_RATE_MIN_SPM || rate > SWIM_STROKE_RATE_MAX_SPM) return null
  return Math.round(rate * 10) / 10
}

export const swimChartMetric = (value: string | undefined): SwimChartMetric => {
  if (value === 'cadence' || value === 'swolf') return value
  return 'pace'
}

export const swimLengthMetrics = (length: SwimLengthInput): SwimLengthMetrics | null => {
  if (
    length.stroke === 'kickboard' ||
    !finitePositive(length.durationS) ||
    length.strokeCount == null ||
    !finitePositive(length.strokeCount)
  )
    return null
  return {
    strokesPerLength: length.strokeCount,
    swolf: Math.round(length.durationS + length.strokeCount),
  }
}

export const swimLengthAverages = (
  lengths: readonly SwimLengthInput[],
): SwimLengthMetrics | null => {
  let strokes = 0
  let swolf = 0
  let count = 0
  for (const length of lengths) {
    const metrics = swimLengthMetrics(length)
    if (!metrics) continue
    strokes += metrics.strokesPerLength
    swolf += metrics.swolf
    count++
  }
  if (count === 0) return null
  return {
    strokesPerLength: Math.round((strokes / count) * 10) / 10,
    swolf: Math.round((swolf / count) * 10) / 10,
  }
}
