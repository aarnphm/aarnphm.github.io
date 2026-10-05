import type { Analytics } from '../../../plugins/stores/analytics'
import { shiftIsoDay } from '../../../util/local-date'

export type AnalyticsRange = '60d' | 'all'

export const defaultAnalyticsRange = (view?: string): AnalyticsRange =>
  view === 'analytics' ? 'all' : '60d'

export const analyticsForRange = (data: Analytics, range: AnalyticsRange): Analytics => {
  if (range === 'all') return data

  const cutoff = shiftIsoDay(data.meta.today, -59)
  const from = data.meta.windowFrom > cutoff ? data.meta.windowFrom : cutoff
  const contains = (date: string): boolean => date >= from && date <= data.meta.today
  const dated = <T extends { date: string }>(points: T[]): T[] =>
    points.filter(point => contains(point.date))
  const activities = dated(data.activities)

  // Keep estimates calculated from their complete history; limit the displayed observations.
  return {
    ...data,
    meta: { ...data.meta, windowFrom: from, activityCount: activities.length },
    daily: dated(data.daily),
    weekly: data.weekly.filter(point => contains(point.weekStart)),
    activities,
    bests: data.bests.map(sport => ({ ...sport, bestToDate: dated(sport.bestToDate) })),
    calibration: {
      ...data.calibration,
      paces: data.calibration.paces.map(sport => ({ ...sport, points: dated(sport.points) })),
    },
    body: {
      ...data.body,
      series: dated(data.body.series),
      bmrSeries: dated(data.body.bmrSeries),
      ffmiSeries: dated(data.body.ffmiSeries),
    },
    recovery: { ...data.recovery, series: dated(data.recovery.series) },
    powerCurve: {
      ...data.powerCurve,
      powerToWeight: {
        ...data.powerCurve.powerToWeight,
        points: dated(data.powerCurve.powerToWeight.points),
      },
    },
    heat: {
      ...data.heat,
      series: dated(data.heat.series),
      activities: dated(data.heat.activities),
    },
    distributions: { ...data.distributions, activities: dated(data.distributions.activities) },
    engine: {
      ...data.engine,
      vo2max: {
        ...data.engine.vo2max,
        trend: data.engine.vo2max.trend.filter(point => contains(point.weekStart)),
      },
      abilities: {
        ...data.engine.abilities,
        sports: data.engine.abilities.sports.map(sport => ({
          ...sport,
          history: dated(sport.history),
        })),
      },
      cardio: {
        ...data.engine.cardio,
        rhrSeries: dated(data.engine.cardio.rhrSeries),
        hrvSeries: dated(data.engine.cardio.hrvSeries),
        efSeries: dated(data.engine.cardio.efSeries),
        decouplingSeries: dated(data.engine.cardio.decouplingSeries),
      },
    },
  }
}
