import type { RawStravaActivity, Sport } from '../strava'
import { WORLD_TOUR_POWER_REFERENCE } from '../../../util/cycling-ability-reference'
import { SWIM_PACE_MAX_S_PER_100M, SWIM_PACE_MIN_S_PER_100M } from '../../../util/swim-metrics'
import { clamp, DAY_MS, dayMs, mean, SPRINT_CAP_MS, type Act, type DailyPoint } from './primitives'
import { norm01 } from './schema'

export const COGGAN_FTP_WKG: [number, number] = [1.5, 6.4]
export const CTL_ANCHOR: [number, number] = [0, 100]
const RUN_SPM_TARGET = 180
const BIKE_RPM_TARGET = 90
const SWIM_STROKE_TARGET = 30
export const RUN_SPRINT_MS: [number, number] = [3, 8]
export const RUN_THR_MS: [number, number] = [2.5, 5.5]
export const SWIM_SPEED_MS: [number, number] = [
  100 / SWIM_PACE_MAX_S_PER_100M,
  100 / SWIM_PACE_MIN_S_PER_100M,
]

export const PROJ_WINDOW_D = 28
export const PROJ_HORIZON_D = 28
export const PROJ_MIN_POINTS = 8
const RUN_FORM_WINDOW_D = 42
const CAD_TARGET: Record<Sport, number> = {
  swim: SWIM_STROKE_TARGET,
  bike: BIKE_RPM_TARGET,
  run: RUN_SPM_TARGET,
}
export const CAD_UNIT: Record<Sport, string> = { swim: 'str/min', bike: 'rpm', run: 'spm' }

export const bikePowerScore = (key: 'sprint' | 'climb', wattsPerKg: number | null): number | null =>
  wattsPerKg == null
    ? null
    : Math.round(norm01(wattsPerKg, 0, WORLD_TOUR_POWER_REFERENCE[key].wattsPerKg) * 100)
export const plausibleVgap = (sport: 'swim' | 'run', x: Act): number | null =>
  x.vGap > 0 && x.vGap <= SPRINT_CAP_MS[sport] ? x.vGap : null
export const cadValOf = (sport: Sport, a: RawStravaActivity): number | null =>
  a.averageCadence == null ? null : sport === 'run' ? a.averageCadence * 2 : a.averageCadence
export const cadScoreOf = (sport: Sport, v: number): number =>
  Math.round(clamp(100 - (Math.abs(v - CAD_TARGET[sport]) / CAD_TARGET[sport]) * 200, 0, 100))
export const swimPaceScoreOf = (paceSPer100m: number): number =>
  Math.round((1 - norm01(paceSPer100m, SWIM_PACE_MIN_S_PER_100M, SWIM_PACE_MAX_S_PER_100M)) * 100)
export const swimSpeedScoreOf = (speedMps: number): number => swimPaceScoreOf(100 / speedMps)
export const sportCtlOf = (d: DailyPoint, sport: Sport): number =>
  sport === 'swim' ? d.swimCtl : sport === 'bike' ? d.bikeCtl : d.runCtl

export const rollingMetricMean = (
  points: readonly { day: string; value: number }[],
  cutoff: string,
): number | null => {
  const cutoffMs = dayMs(cutoff)
  const earliest = cutoffMs - (RUN_FORM_WINDOW_D - 1) * DAY_MS
  const values = points
    .filter(point => {
      const time = dayMs(point.day)
      return time >= earliest && time <= cutoffMs
    })
    .map(point => point.value)
  return values.length ? mean(values) : null
}
