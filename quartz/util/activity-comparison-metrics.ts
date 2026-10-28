import type { ActivityHealth, StravaActivityDetail } from '../plugins/stores/strava'

export type ActivityComparisonStatKey =
  | 'tss'
  | 'intensity-factor'
  | 'aerobic-effect'
  | 'anaerobic-effect'
  | 'exercise-load'
  | 'relative-effort'
  | 'moving-time'
  | 'elapsed-time'
  | 'distance'
  | 'elevation-gain'
  | 'grade'
  | 'work'
  | 'calories'
  | 'average-power'
  | 'normalized-power'
  | 'watts-per-kg'
  | 'variability-index'
  | 'max-power'
  | 'critical-power'
  | 'w-prime'
  | 'power-balance'
  | 'peak-5s'
  | 'peak-30s'
  | 'peak-1m'
  | 'peak-5m'
  | 'peak-20m'
  | 'peak-60m'
  | 'average-hr'
  | 'max-hr'
  | 'efficiency-factor'
  | 'decoupling'
  | 'speed'
  | 'max-speed'
  | 'cadence'
  | 'stride-length'
  | 'ground-contact-time'
  | 'vertical-oscillation'
  | 'vertical-ratio'
  | 'stroke-rate'
  | 'strokes-per-length'
  | 'torque-effectiveness'
  | 'pedal-smoothness'
  | 'average-torque'
  | 'standing-time'
  | 'temperature'
  | 'humidity'
  | 'wind'
  | 'water-temperature'
  | 'readiness'
  | 'hrv'
  | 'resting-hr'
  | 'sleep-score'
  | 'sleep-duration'
  | 'body-weight'
  | 'carbs'
  | 'carbs-per-hour'
  | 'fluid'
  | 'sweat-loss'

/**
 * `garden` marks values this site derives. `estimate` separates modelled or sparsely sampled
 * values from recorded provider measurements and exact ratios of them.
 */
export type ActivityComparisonStatSource =
  | 'wahoo'
  | 'garmin'
  | 'strava'
  | 'oura'
  | 'apple'
  | 'weather'
  | 'manual'
  | 'garden'

export interface ActivityComparisonStat {
  value: number
  /** Right-leg value for paired left/right metrics; `value` holds the left leg. */
  right?: number
  source: ActivityComparisonStatSource
  estimate: boolean
}

export type ActivityComparisonStats = Partial<
  Record<ActivityComparisonStatKey, ActivityComparisonStat>
>

export interface ActivityComparisonWindow {
  startKm: number
  endKm: number
}

export interface ActivityComparisonStatsOptions {
  window?: ActivityComparisonWindow | null
  health?: ActivityHealth | null
  athleteFtpWatts?: number | null
  /** Display cadence for the whole activity, already scaled to rpm or spm. */
  averageCadence: number | null
  /** Multiplier from recorded route cadence to display cadence. */
  cadenceScale: number
  excludeZeroPower: boolean
}

import { wholeActivityStats } from './activity-comparison-metrics/whole-activity'
import { windowStats } from './activity-comparison-metrics/window-stats'

export const activityComparisonStats = (
  activity: StravaActivityDetail,
  options: ActivityComparisonStatsOptions,
): ActivityComparisonStats =>
  options.window
    ? windowStats(activity, options.window, options)
    : wholeActivityStats(activity, options)
