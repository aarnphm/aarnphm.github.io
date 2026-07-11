export type WeeklyTargetRange = [number, number]

export interface WeeklyTargetObservation {
  value: number
  complete: boolean
  observed: boolean
}

const TARGET_WEEKS = 3
const TARGET_LOW = 0.6
const TARGET_HIGH = 1.2

export function weeklyTargetRanges(
  observations: readonly WeeklyTargetObservation[],
): (WeeklyTargetRange | null)[] {
  const history: number[] = []
  let warmup = 0
  return observations.map(observation => {
    const recent = history.slice(-TARGET_WEEKS)
    const target =
      recent.length === TARGET_WEEKS
        ? recent.reduce((sum, value) => sum + value, 0) / TARGET_WEEKS
        : warmup
    const range: WeeklyTargetRange = [target * TARGET_LOW, target * TARGET_HIGH]
    if (observation.complete && observation.observed) {
      warmup += 0.5 * (observation.value - warmup)
      history.push(observation.value)
    }
    return range
  })
}

export function weeklyChartX(index: number, length: number): number {
  if (length <= 1) return 0.5
  return index / (length - 1)
}

export function weeklyChartIndex(fraction: number, length: number): number {
  if (length <= 1) return 0
  return Math.round(Math.min(1, Math.max(0, fraction)) * (length - 1))
}
