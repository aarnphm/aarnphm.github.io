import type { ActivityHealth, StravaActivityDetail } from '../../plugins/stores/strava'
import type { DetailCtx, TriNodeFactory } from './construction-contracts'
import { activityShareData } from './activity-share'
import {
  buildComparisonGearRatioDistribution,
  buildComparisonPowerDistribution,
  buildComparisonZones,
} from './comparison-distributions'
import {
  activityComparisonEligible,
  comparisonGeographicSegments,
  comparisonMaxDistanceKm,
} from './comparison-geography'
import { buildComparisonMetricChart } from './comparison-metric-chart'
import {
  activityComparisonMetricsForSport,
  activityComparisonMetricSpecs,
} from './comparison-metrics'
import { activityCompareColor, comparisonChartHead } from './comparison-plot'
import { buildComparisonPowerCurve } from './comparison-power-curve'
import { buildActivityComparisonStats } from './comparison-stats'
import { shortDate } from './metric-text'

const buildComparisonLegend = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  removable: boolean,
): N => {
  const legend = f.el(
    'div',
    `tri-compare-legend${removable ? '' : ' tri-compare-legend--static'}`,
    undefined,
    {
      role: 'list',
      'aria-label': 'selected activities',
      'data-i18n-aria-label': 'selected activities',
    },
  )
  for (const [index, activity] of activities.entries()) {
    const item = f.el('div', 'tri-compare-legend-item', undefined, {
      role: 'listitem',
      'data-activity-id': `${activity.id}`,
      'data-activity-index': `${index}`,
      style: `--tri-compare-color:${activityCompareColor(index)}`,
    })
    const toggle = f.el('button', 'tri-compare-legend-toggle', undefined, {
      type: 'button',
      'data-compare-activity-toggle': `${activity.id}`,
      'aria-pressed': 'true',
      'aria-label': 'toggle activity',
      'data-i18n-aria-label': 'toggle activity',
      title: activity.name,
    })
    f.add(
      toggle,
      f.el('span', 'tri-compare-legend-swatch', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-compare-legend-date', shortDate(activity.date)),
      f.el('span', 'tri-compare-legend-name', activity.name, { 'data-site-cursor-bracket': '' }),
    )
    f.add(item, toggle)
    if (removable) {
      const removeAttrs: Record<string, string> = {
        type: 'button',
        'data-compare-activity-remove': `${activity.id}`,
        'aria-label': 'remove activity',
        'data-i18n-aria-label': 'remove activity',
      }
      if (activities.length <= 2) removeAttrs.disabled = ''
      const remove = f.el('button', 'tri-compare-legend-remove', undefined, removeAttrs)
      const removeIcon = f.svg('svg', {
        class: 'tri-compare-legend-remove-icon',
        viewBox: '0 0 16 16',
        fill: 'none',
        'aria-hidden': 'true',
      })
      f.add(
        removeIcon,
        f.svg('path', {
          d: 'M4 4l8 8M12 4 4 12',
          stroke: 'currentColor',
          'stroke-width': 1.25,
          'stroke-linecap': 'round',
        }),
      )
      f.add(remove, removeIcon)
      f.add(item, remove)
    }
    f.add(legend, item)
  }
  return legend
}

const buildComparisonMap = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  available: number,
): N => {
  const panel = f.el('section', 'tri-compare-map-panel', undefined, {
    'data-compare-chart': 'route',
    'data-available': `${available}`,
    'data-selected': `${activities.length}`,
    'aria-label': 'route overlay',
    'data-i18n-aria-label': 'route overlay',
  })
  const stage = f.el('div', 'tri-compare-map-stage')
  const canvas = f.el('div', 'tri-compare-map', undefined, {
    'data-compare-map': '',
    'data-available': `${available}`,
    'data-domain-x-max': `${comparisonMaxDistanceKm(activities)}`,
  })
  f.add(stage, canvas)
  f.add(panel, comparisonChartHead(f, 'route overlay'), stage)
  return panel
}

const comparisonState = (
  activities: readonly StravaActivityDetail[],
): 'empty' | 'insufficient' | 'mixed-sport' | 'route-unavailable' | 'ready' => {
  if (activities.length === 0) return 'empty'
  if (activities.length < 2) return 'insufficient'
  const sport = activities[0].sport
  if (activities.some(activity => activity.sport !== sport)) return 'mixed-sport'
  if (activities.some(activity => !activityComparisonEligible(activity))) return 'route-unavailable'
  return 'ready'
}

const comparisonStateText = (
  state: 'empty' | 'insufficient' | 'mixed-sport' | 'route-unavailable',
): string => {
  if (state === 'mixed-sport') return 'Choose activities from one exact sport.'
  if (state === 'route-unavailable') return 'Route telemetry is unavailable for this comparison.'
  return 'Choose at least two activities.'
}

export const buildActivityComparison = <N>(
  f: TriNodeFactory<N>,
  activities: StravaActivityDetail[],
  ctx?: DetailCtx,
  options: { removable?: boolean; health?: Readonly<Record<string, ActivityHealth>> | null } = {},
): N => {
  const state = comparisonState(activities)
  const rootAttrs: Record<string, string> = {
    'data-compare-state': state,
    'data-selected': `${activities.length}`,
    'data-press-kit-activities': encodeURIComponent(
      JSON.stringify(
        activities.flatMap(activity => {
          const data = activityShareData(f.presentation, activity)
          return data ? [data] : []
        }),
      ),
    ),
  }
  if (activities.length > 0 && activities.every(activity => activity.sport === activities[0].sport))
    rootAttrs['data-sport'] = activities[0].sport
  const root = f.el('section', `tri-compare tri-compare--${state}`, undefined, rootAttrs)
  f.add(root, buildComparisonLegend(f, activities, options.removable !== false))
  if (state !== 'ready') {
    f.add(
      root,
      f.el('p', 'tri-compare-empty', comparisonStateText(state), { 'data-compare-empty': state }),
    )
    return root
  }
  const maxDistanceKm = Math.max(comparisonMaxDistanceKm(activities), 1e-6)
  const content = f.el('div', 'tri-compare-content')
  const body = f.el('div', 'tri-compare-body')
  const routed = activities.filter(
    activity => comparisonGeographicSegments(activity).length > 0,
  ).length
  if (routed > 0) f.add(content, buildComparisonMap(f, activities, routed))
  f.add(body, buildActivityComparisonStats(f, activities, ctx, { health: options.health }))
  const chartViewport = f.el('div', 'tri-compare-charts-viewport')
  const charts = f.el('div', 'tri-compare-charts', undefined, { 'data-keyboard-scroll': '' })
  const sport = activities[0].sport
  const metricSpecs = activityComparisonMetricSpecs(f.presentation)
  for (const metric of activityComparisonMetricsForSport(sport))
    f.add(charts, buildComparisonMetricChart(f, activities, metricSpecs[metric], maxDistanceKm))
  if (sport === 'bike') f.add(charts, buildComparisonGearRatioDistribution(f, activities))
  if (sport !== 'swim')
    f.add(
      charts,
      buildComparisonPowerDistribution(f, activities),
      buildComparisonPowerCurve(f, activities, ctx),
    )
  f.add(charts, buildComparisonZones(f, activities, 'hr-zones'))
  if (sport !== 'swim') f.add(charts, buildComparisonZones(f, activities, 'power-zones'))
  f.add(chartViewport, charts)
  f.add(body, chartViewport)
  f.add(content, body)
  f.add(root, content)
  return root
}
