import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { axisFrame, distanceXTicks, niceTicks, swimActivityXTicks } from './chart-geometry'
import {
  activityComparisonMetricsForSport,
  comparisonMetricSeries,
  type ActivityComparisonMetricSpec,
} from './comparison-metrics'
import {
  ACTIVITY_COMPARISON_HEIGHT,
  ACTIVITY_COMPARISON_WIDTH,
  activityCompareColor,
  addComparisonSelectionRegion,
  comparisonChartHead,
  comparisonGraphAttrs,
  comparisonNumericDomain,
  comparisonReadout,
  comparisonSeriesPath,
} from './comparison-plot'
import { excludesZeroPower, scrubDist } from './metric-text'

export const buildComparisonMetricChart = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  spec: ActivityComparisonMetricSpec,
  maxDistanceKm: number,
): N => {
  const series = activities.map((activity, index) =>
    comparisonMetricSeries(f.presentation, activity, index, spec),
  )
  const available = series.filter(item => item.values.length > 0).length
  const values = series.flatMap(item => item.values)
  const domain = spec.domain
    ? spec.domain(values)
    : comparisonNumericDomain(
        values,
        spec.metric === 'power' && excludesZeroPower(f.presentation) ? false : spec.includeZero,
      )
  const selectionClipId = `tri-compare-${spec.metric}-selection-clip`
  const sport = activities[0]?.sport ?? 'bike'
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      spec.metric,
      spec.title,
      0,
      maxDistanceKm,
      0,
      scrubDist(f.presentation, 0, sport),
      available,
    ),
    class: 'tri-compare-graph tri-compare-distance-graph',
    'data-domain-x-min': 0,
    'data-domain-x-max': maxDistanceKm,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-available': available,
    'data-selected': activities.length,
  })
  addComparisonSelectionRegion(f, graph, selectionClipId)
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: value === 0 ? '0' : spec.tick(value, sport),
    vbY:
      ACTIVITY_COMPARISON_HEIGHT -
      ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-grid',
        x1: 0,
        y1: tick.vbY,
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: tick.vbY,
      }),
    )
  for (const item of series) {
    if (item.values.length === 0) continue
    const path = comparisonSeriesPath(item.segments, maxDistanceKm, domain)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': item.activity.id,
        'data-activity-index': item.index,
        style: `--tri-compare-color:${activityCompareColor(item.index)}`,
      }),
      f.svg('path', {
        class: 'tri-compare-selection-line',
        d: path,
        'data-activity-id': item.activity.id,
        'data-activity-index': item.index,
        style: `--tri-compare-color:${activityCompareColor(item.index)}`,
        'clip-path': `url(#${selectionClipId})`,
        'aria-hidden': 'true',
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: 0,
      y1: 0,
      x2: 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': spec.metric,
    'data-available': `${available}`,
    'data-selected': `${activities.length}`,
  })
  const head = comparisonChartHead(f, spec.title)
  if (spec.metric === activityComparisonMetricsForSport(sport)[0])
    f.add(head, comparisonReadout(f, activities))
  f.add(
    chart,
    head,
    axisFrame(
      f,
      graph,
      yTicks,
      ACTIVITY_COMPARISON_HEIGHT,
      sport === 'swim'
        ? swimActivityXTicks(maxDistanceKm * 1_000)
        : distanceXTicks(f.presentation, 0, maxDistanceKm),
      true,
      { top: 0, bottom: ACTIVITY_COMPARISON_HEIGHT },
    ),
  )
  return chart
}
