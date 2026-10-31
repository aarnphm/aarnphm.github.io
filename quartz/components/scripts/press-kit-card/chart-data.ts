import type { ActivityCardChart, ActivityCardData } from '../../../util/press-kit-activity'
import type { CardModel } from './spec'

export function trainingCharts(
  activity: ActivityCardData | undefined,
  model: CardModel,
): ActivityCardChart[] {
  return (activity?.charts ?? []).flatMap(chart => {
    if (!model.show[chart.key]) return []
    if (chart.key === 'powerCurve' && model.powerCurveUnit === 'relative' && chart.weight) {
      const { kg, date } = chart.weight
      const locale = model.locale ?? 'en'
      const number = new Intl.NumberFormat(locale, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })
      return [
        {
          ...chart,
          axis: 'W/kg',
          note: `${number.format(kg)} kg · Garmin · ${date}`,
          points: chart.points.map(point => ({
            ...point,
            y: point.y == null ? null : point.y / kg,
            value: point.y == null ? '—' : `${number.format(point.y / kg)} W/kg`,
          })),
        },
      ]
    }
    if (chart.key !== 'cyclingPower' || !chart.series) return [chart]
    const series = chart.series.find(series => series.seconds === (model.powerWindow ?? 30))
    return series ? [{ ...chart, points: series.points, window: series.seconds }] : []
  })
}
