import { parseClockSeconds } from './duration'

export const RUN_PACE_ZONE_NAMES = [
  'recovery',
  'endurance',
  'tempo',
  'threshold',
  'VO2max',
  'anaerobic',
] as const

const TEN_KM_PACE_ZONE_REFERENCE_TIME_S = 50 * 60
const TEN_KM_PACE_ZONE_REFERENCE_BOUNDS_S_PER_MILE = [623, 537, 482, 451, 424] as const
const KILOMETRES_PER_MILE = 1.609344

export interface RunPaceZoneDistribution {
  zoneSeconds: number[]
  boundsSPerKm: number[]
  tenKmRaceTimeS: number
}

export interface RunPaceZoneRange {
  fastestSPerKm: number | null
  slowestSPerKm: number | null
}

export const runPaceZoneReference = (
  tenKmRaceTime: string,
): { paceZoneBoundsSPerKm: number[]; tenKmRaceTimeS: number | null } => {
  const tenKmRaceTimeS = parseClockSeconds(tenKmRaceTime)
  if (!(tenKmRaceTimeS > 0)) return { paceZoneBoundsSPerKm: [], tenKmRaceTimeS: null }
  const scale = tenKmRaceTimeS / TEN_KM_PACE_ZONE_REFERENCE_TIME_S
  return {
    paceZoneBoundsSPerKm: TEN_KM_PACE_ZONE_REFERENCE_BOUNDS_S_PER_MILE.map(
      seconds => Math.round((seconds / KILOMETRES_PER_MILE) * scale * 1_000) / 1_000,
    ),
    tenKmRaceTimeS,
  }
}

export const runPaceZoneRange = (
  boundsSPerKm: readonly number[],
  index: number,
): RunPaceZoneRange | null => {
  if (
    boundsSPerKm.length === 0 ||
    !Number.isInteger(index) ||
    index < 0 ||
    index > boundsSPerKm.length
  )
    return null
  if (index === 0) return { fastestSPerKm: boundsSPerKm[0], slowestSPerKm: null }
  if (index === boundsSPerKm.length)
    return { fastestSPerKm: null, slowestSPerKm: boundsSPerKm[index - 1] }
  return { fastestSPerKm: boundsSPerKm[index], slowestSPerKm: boundsSPerKm[index - 1] }
}
