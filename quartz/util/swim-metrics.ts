export const SWIM_PACE_MIN_S_PER_100M = 45
export const SWIM_PACE_MAX_S_PER_100M = 360
export const SWIM_STROKE_RATE_MIN_SPM = 5
export const SWIM_STROKE_RATE_MAX_SPM = 100

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
