import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import {
  activityPowerDistributionPercentages,
  activityZonePercentages,
} from './activity-metric-data'
import { axisFrame, niceTicks } from './chart-geometry'
import {
  ACTIVITY_COMPARISON_HEIGHT,
  ACTIVITY_COMPARISON_WIDTH,
  activityCompareColor,
  comparisonChartHead,
  comparisonGraphAttrs,
  comparisonNumericDomain,
  comparisonPowerDistributionTicks,
} from './comparison-plot'
import { activityGearRatioDistribution, sampledGearTicks } from './shifting-data'

export const buildComparisonPowerDistribution = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
): N => {
  const distributions = activities.map((activity, index) => ({
    activity,
    index,
    percentages: activityPowerDistributionPercentages(activity.powerHist),
  }))
  const available = distributions.filter(distribution =>
    distribution.percentages.some(value => value > 0),
  )
  const binCount = Math.max(
    2,
    ...distributions.map(distribution => distribution.percentages.length),
  )
  const maxWatts = (binCount - 1) * 25
  const domain = comparisonNumericDomain(
    available.flatMap(distribution => distribution.percentages),
    true,
  )
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      'power-distribution',
      '25W power distribution',
      0,
      maxWatts,
      0,
      '0–24 W',
      available.length,
    ),
    class: 'tri-compare-graph tri-compare-distribution-graph',
    'data-compare-chart': 'power-distribution',
    'data-domain-x-min': 0,
    'data-domain-x-max': maxWatts,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-bin-count': binCount,
    'data-bin-width-watts': 25,
    'data-available': available.length,
    'data-selected': activities.length,
  })
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: `${Math.round(value)}%`,
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
  for (const distribution of available) {
    const path = Array.from(
      { length: binCount },
      (_, index) => distribution.percentages[index] ?? 0,
    )
      .map((value, index) => {
        const x = (index / (binCount - 1)) * ACTIVITY_COMPARISON_WIDTH
        const y =
          ACTIVITY_COMPARISON_HEIGHT -
          ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1)
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
      })
      .join(' ')
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': distribution.activity.id,
        'data-activity-index': distribution.index,
        style: `--tri-compare-color:${activityCompareColor(distribution.index)}`,
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
    'data-compare-chart': 'power-distribution',
    'data-available': `${available.length}`,
    'data-selected': `${activities.length}`,
  })
  f.add(
    chart,
    comparisonChartHead(f, '25W power distribution'),
    axisFrame(
      f,
      graph,
      yTicks,
      ACTIVITY_COMPARISON_HEIGHT,
      comparisonPowerDistributionTicks(binCount),
      true,
      { top: 0, bottom: ACTIVITY_COMPARISON_HEIGHT },
    ),
  )
  return chart
}

export const buildComparisonGearRatioDistribution = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
): N => {
  const distributions = activities.map((activity, index) => ({
    activity,
    index,
    points: activityGearRatioDistribution(activity),
  }))
  const available = distributions.filter(distribution => distribution.points.length > 0)
  const ratios = [
    ...new Set(
      distributions.flatMap(distribution => distribution.points.map(point => point.ratio)),
    ),
  ].sort((left, right) => left - right)
  const ratioCount = ratios.length
  const domain = comparisonNumericDomain(
    available.flatMap(distribution => distribution.points.map(point => point.percentage)),
    true,
  )
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      'gear-ratio-distribution',
      'gear ratio distribution',
      0,
      Math.max(1, ratioCount - 1),
      0,
      ratios[0] == null ? 'no data' : `${ratios[0].toFixed(2)}×`,
      available.length,
    ),
    class: 'tri-compare-graph tri-compare-distribution-graph',
    'data-compare-chart': 'gear-ratio-distribution',
    'data-domain-x-min': 0,
    'data-domain-x-max': Math.max(1, ratioCount - 1),
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-ratio-count': ratioCount,
    'data-available': available.length,
    'data-selected': activities.length,
  })
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: `${Math.round(value)}%`,
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
  for (const distribution of available) {
    const percentages = new Map(
      distribution.points.map(point => [point.ratio, point.percentage] as const),
    )
    const path = ratios
      .map((ratio, index) => {
        const value = percentages.get(ratio) ?? 0
        const x =
          ratioCount <= 1
            ? ACTIVITY_COMPARISON_WIDTH / 2
            : (index / (ratioCount - 1)) * ACTIVITY_COMPARISON_WIDTH
        const y =
          ACTIVITY_COMPARISON_HEIGHT -
          ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1)
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
      })
      .join(' ')
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': distribution.activity.id,
        'data-activity-index': distribution.index,
        style: `--tri-compare-color:${activityCompareColor(distribution.index)}`,
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: ratioCount <= 1 ? ACTIVITY_COMPARISON_WIDTH / 2 : 0,
      y1: 0,
      x2: ratioCount <= 1 ? ACTIVITY_COMPARISON_WIDTH / 2 : 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const sampledRatios = sampledGearTicks(ratios, 6)
  const xTicks = sampledRatios.map(ratio => {
    const index = ratios.indexOf(ratio)
    return {
      label: `${ratio.toFixed(2)}×`,
      pct: ratioCount <= 1 ? 50 : (index / (ratioCount - 1)) * 100,
      cls:
        index === 0
          ? 'tri-cax-xt--first'
          : index === ratioCount - 1
            ? 'tri-cax-xt--last'
            : undefined,
    }
  })
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': 'gear-ratio-distribution',
    'data-available': `${available.length}`,
    'data-selected': `${activities.length}`,
  })
  f.add(
    chart,
    comparisonChartHead(f, 'gear ratio distribution'),
    axisFrame(f, graph, yTicks, ACTIVITY_COMPARISON_HEIGHT, xTicks, true, {
      top: 0,
      bottom: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  return chart
}

export const buildComparisonZones = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  kind: 'hr-zones' | 'power-zones',
): N => {
  const values = activities.map((activity, index) => ({
    activity,
    index,
    percentages: activityZonePercentages(
      kind === 'hr-zones' ? activity.hrZones : activity.powerZones,
    ),
  }))
  const available = values.filter(item => item.percentages.some(value => value > 0))
  const defaultZoneCount = kind === 'hr-zones' ? 5 : 7
  const observedZoneCount = values.reduce((max, item) => Math.max(max, item.percentages.length), 0)
  const zoneCount = observedZoneCount > 0 ? observedZoneCount : defaultZoneCount
  const domain = { min: 0, max: 100 }
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      kind,
      kind === 'hr-zones' ? 'heart rate zones' : 'power zones',
      1,
      zoneCount,
      1,
      'Z1',
      available.length,
    ),
    class: 'tri-compare-graph tri-compare-zone-graph',
    'data-domain-x-min': 1,
    'data-domain-x-max': zoneCount,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-zone-unit': 'percent',
    'data-available': available.length,
    'data-selected': activities.length,
  })
  const yTicks = [0, 25, 50, 75, 100].map(value => ({
    label: `${value}%`,
    vbY: ACTIVITY_COMPARISON_HEIGHT - (value / domain.max) * (ACTIVITY_COMPARISON_HEIGHT - 1),
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
  for (const item of available) {
    const path = item.percentages
      .map((value, index) => {
        const x =
          zoneCount === 1
            ? ACTIVITY_COMPARISON_WIDTH / 2
            : (index / (zoneCount - 1)) * ACTIVITY_COMPARISON_WIDTH
        const y =
          ACTIVITY_COMPARISON_HEIGHT - (value / domain.max) * (ACTIVITY_COMPARISON_HEIGHT - 1)
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
      })
      .join(' ')
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': item.activity.id,
        'data-activity-index': item.index,
        style: `--tri-compare-color:${activityCompareColor(item.index)}`,
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
  const xTicks = Array.from({ length: zoneCount }, (_, index) => ({
    label: `Z${index + 1}`,
    pct: zoneCount === 1 ? 50 : (index / (zoneCount - 1)) * 100,
    cls:
      index === 0 ? 'tri-cax-xt--first' : index === zoneCount - 1 ? 'tri-cax-xt--last' : undefined,
  }))
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': kind,
    'data-available': `${available.length}`,
    'data-selected': `${activities.length}`,
  })
  f.add(
    chart,
    comparisonChartHead(f, kind === 'hr-zones' ? 'heart rate zones' : 'power zones'),
    axisFrame(f, graph, yTicks, ACTIVITY_COMPARISON_HEIGHT, xTicks, true, {
      top: 0,
      bottom: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  return chart
}
