import type { PowerCurveBlock } from '../../../../../plugins/stores/analytics'
import type { CriticalPowerEstimate } from '../../../../../plugins/stores/critical-power'
import type { PowerSkill } from '../../../../../plugins/stores/power-rank'
import type { PowerCurvePoint } from '../../../../../plugins/stores/strava'
import type { PowerCurveFtpEstimate } from '../../../../../util/cycling-ftp'
import type { TriathlonFormatter } from '../../../runtime/formatter'

export type BestPowerSeriesKey = 'six-weeks' | 'year'

export const powerSkillAtSeconds = (seconds: number): PowerSkill =>
  seconds <= 60 ? 'sprint' : seconds <= 600 ? 'attack' : 'climb'

export const bestPowerSeriesLabel = (
  formatter: TriathlonFormatter,
  power: PowerCurveBlock,
  key: BestPowerSeriesKey,
): string =>
  key === 'six-weeks'
    ? formatter.text('last 6 weeks')
    : power.yearLabel == null
      ? formatter.text('calendar year')
      : `${formatter.text('all of')} ${power.yearLabel}`

export const bestPowerSeries = (
  power: PowerCurveBlock,
): Array<{ key: BestPowerSeriesKey; curve: PowerCurvePoint[] }> => [
  { key: 'six-weeks', curve: power.sixWeeks },
  { key: 'year', curve: power.year },
]

export const criticalPowerForSeries = (
  power: PowerCurveBlock,
  key: BestPowerSeriesKey,
): CriticalPowerEstimate | null =>
  key === 'six-weeks' ? power.criticalPower : power.criticalPowerYear

export const estimatedFtpForSeries = (
  power: PowerCurveBlock,
  key: BestPowerSeriesKey,
): PowerCurveFtpEstimate | null =>
  key === 'six-weeks' ? power.estimatedFtp : power.estimatedFtpYear
