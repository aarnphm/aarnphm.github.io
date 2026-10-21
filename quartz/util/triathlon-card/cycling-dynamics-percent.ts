import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { triathlonTraceName } from '../triathlon-trace-settings'
import {
  axisFrame,
  distanceXTicks,
  graphViewForDistance,
  type ActivityGraphDomain,
} from './chart-geometry'
import { buildAnalysisSelection } from './chart-selection'
import {
  buildCyclingDynamicsLegend,
  buildCyclingDynamicsTitle,
  cyclingDynamicsAverage,
  cyclingDynamicsDomain,
  cyclingDynamicsPaths,
  cyclingDynamicsSampleCount,
  cyclingDynamicsValues,
  cyclingDynamicsWattsSamples,
  validCyclingDynamicsValue,
  type CyclingDynamicsPercentMetric,
} from './cycling-dynamics-data'
import { buildWattsHeatmap, type CyclingModeTitle } from './watts-heatmap'

const buildCyclingDynamicsPercentChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  metric: CyclingDynamicsPercentMetric,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const dynamics = d.cyclingDynamics
  if (!dynamics || cyclingDynamicsSampleCount(dynamics) < 2) return null
  const leftKey = metric === 'pedalSmoothness' ? 'leftPedalSmoothness' : 'leftTorqueEffectiveness'
  const rightKey =
    metric === 'pedalSmoothness' ? 'rightPedalSmoothness' : 'rightTorqueEffectiveness'
  const left = cyclingDynamicsValues(dynamics, leftKey)
  const right = cyclingDynamicsValues(dynamics, rightKey)
  const observed = [...left, ...right].filter(validCyclingDynamicsValue)
  if (observed.length < 2) return null
  const [domainMin, domainMax] = cyclingDynamicsDomain(observed)
  const height = 30
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const py = (value: number): number =>
    height -
    ((Math.min(domainMax, Math.max(domainMin, value)) - domainMin) / (domainMax - domainMin)) *
      (height - 1)
  const middle = Math.round((domainMin + domainMax) / 2)
  const yTicks = [domainMin, middle, domainMax].map(value => ({
    label: `${value}%`,
    vbY: py(value),
  }))
  const view = graphViewForDistance(maxDistanceKm, graphDomain)
  const title: CyclingModeTitle =
    metric === 'pedalSmoothness' ? 'pedal smoothness' : 'torque effectiveness'
  const className = metric === 'pedalSmoothness' ? 'pedal-smoothness' : 'torque-effectiveness'
  const leftHeat = cyclingDynamicsWattsSamples(d, dynamics, left, graphDomain)
  const rightHeat = cyclingDynamicsWattsSamples(d, dynamics, right, graphDomain)
  const heatAvailable = leftHeat.length + rightHeat.length > 0
  const heatmap = heatAvailable
    ? buildWattsHeatmap(
        f,
        d,
        title,
        `tri-${className}-heatmap`,
        className,
        [
          { side: 'left', samples: leftHeat },
          { side: 'right', samples: rightHeat },
        ],
        domainMin,
        domainMax,
        height,
      )
    : null
  const svgEl = f.svg('svg', {
    class: `tri-elev tri-cycling-dynamics-svg tri-${className}-svg`,
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, title),
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: 100, y2: tick.vbY }),
    )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  for (const [side, values] of [
    ['left', left],
    ['right', right],
  ] as const) {
    const paths = cyclingDynamicsPaths(dynamics, values, maxDistanceKm, py)
    if (paths.bridges)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.bridges,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side} tri-cycling-dynamics-line--missing`,
        }),
      )
    if (paths.measured)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.measured,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side}`,
        }),
      )
  }
  f.add(svgEl, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }))
  const wrap = f.el(
    'div',
    `tri-zone tri-elev-wrap tri-cycling-dynamics-chart tri-${className}-chart${heatmap ? ' tri-cycling-mode-chart' : ''}`,
    undefined,
    {
      'data-tri-trace': triathlonTraceName(title),
      ...(heatmap ? { 'data-cycling-chart-mode': 'distance' } : {}),
    },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-cycling-dynamics-summary')
  const leftAverage = cyclingDynamicsAverage(left)
  const rightAverage = cyclingDynamicsAverage(right)
  if (leftAverage != null && rightAverage != null)
    f.add(
      summary,
      f.el(
        'span',
        'tri-elev-range',
        `L ${leftAverage.toFixed(1)}% / R ${rightAverage.toFixed(1)}% avg`,
      ),
    )
  if (!embedded)
    f.add(summary, buildCyclingDynamicsLegend(f, 'left'), buildCyclingDynamicsLegend(f, 'right'))
  f.add(cap, buildCyclingDynamicsTitle(f, title), summary)
  const distanceFrame = axisFrame(
    f,
    svgEl,
    yTicks,
    height,
    distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
    true,
    { top: 0, bottom: height },
  )
  if (!heatmap) {
    f.add(wrap, cap, distanceFrame)
    return wrap
  }
  f.add(cap, heatmap.readout)
  const distancePane = f.el(
    'div',
    `tri-cycling-chart-pane tri-cycling-chart-pane--distance tri-${className}-pane--distance`,
    undefined,
    { 'data-cycling-chart-mode': 'distance', 'aria-hidden': 'false' },
  )
  f.add(distancePane, distanceFrame)
  const powerPane = f.el(
    'div',
    `tri-cycling-chart-pane tri-cycling-chart-pane--power tri-${className}-pane--power`,
    undefined,
    { 'data-cycling-chart-mode': 'power', 'aria-hidden': 'true', hidden: '' },
  )
  f.add(
    powerPane,
    axisFrame(f, heatmap.svg, yTicks, height, heatmap.wattsTicks, true, { top: 0, bottom: height }),
  )
  f.add(wrap, cap, distancePane, powerPane)
  return wrap
}

export const buildTorqueEffectivenessChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  buildCyclingDynamicsPercentChart(f, d, 'torqueEffectiveness', selection, embedded, graphDomain)

export const buildPedalSmoothnessChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  buildCyclingDynamicsPercentChart(f, d, 'pedalSmoothness', selection, embedded, graphDomain)
