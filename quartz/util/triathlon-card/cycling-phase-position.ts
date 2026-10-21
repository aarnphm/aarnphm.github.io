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
  cyclingDynamicsPaths,
  cyclingDynamicsSampleCount,
  cyclingDynamicsValues,
  validCyclingDynamicsValue,
} from './cycling-dynamics-data'
import { zoneClock } from './metric-text'

export const buildPowerPhaseChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const dynamics = d.cyclingDynamics
  if (!dynamics || cyclingDynamicsSampleCount(dynamics) < 2) return null
  const series = [
    ['left', 'start', cyclingDynamicsValues(dynamics, 'leftPowerPhaseStart')],
    ['left', 'end', cyclingDynamicsValues(dynamics, 'leftPowerPhaseEnd')],
    ['right', 'start', cyclingDynamicsValues(dynamics, 'rightPowerPhaseStart')],
    ['right', 'end', cyclingDynamicsValues(dynamics, 'rightPowerPhaseEnd')],
  ] as const
  if (series.flatMap(([, , values]) => values).filter(validCyclingDynamicsValue).length < 2)
    return null
  const height = 36
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const py = (value: number): number => height - (Math.min(360, Math.max(0, value)) / 360) * height
  const yTicks = [0, 90, 180, 270, 360].map(value => ({ label: `${value}°`, vbY: py(value) }))
  const view = graphViewForDistance(maxDistanceKm, graphDomain)
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-cycling-dynamics-svg tri-power-phase-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, 'power phase'),
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: 100, y2: tick.vbY }),
    )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  for (const [side, boundary, values] of series) {
    const paths = cyclingDynamicsPaths(dynamics, values, maxDistanceKm, py, true)
    if (paths.bridges)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.bridges,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side} tri-power-phase-line--${boundary} tri-cycling-dynamics-line--missing`,
        }),
      )
    if (paths.measured)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.measured,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side} tri-power-phase-line--${boundary}`,
        }),
      )
  }
  f.add(svgEl, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }))
  const wrap = f.el(
    'div',
    'tri-zone tri-elev-wrap tri-cycling-dynamics-chart tri-power-phase-chart',
    undefined,
    { 'data-tri-trace': triathlonTraceName('power phase') },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-cycling-dynamics-summary tri-power-phase-summary')
  for (const side of ['left', 'right'] as const)
    for (const boundary of ['start', 'end'] as const)
      f.add(summary, buildCyclingDynamicsLegend(f, side, boundary))
  f.add(cap, buildCyclingDynamicsTitle(f, 'power phase'), summary)
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      svgEl,
      yTicks,
      height,
      distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: height },
    ),
  )
  return wrap
}

export const buildRiderPositionChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const dynamics = d.cyclingDynamics
  if (!dynamics || dynamics.positionChanges.length === 0) return null
  const height = 24
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const px = (distanceKm: number): number =>
    (Math.min(maxDistanceKm, Math.max(0, distanceKm)) / maxDistanceKm) * 100
  const py = (position: 'seated' | 'standing'): number => (position === 'standing' ? 4 : 20)
  const yTicks = [
    { label: triText(f.presentation.locale, 'standing'), vbY: py('standing') },
    { label: triText(f.presentation.locale, 'seated'), vbY: py('seated') },
  ]
  const changes = dynamics.positionChanges
    .filter(change => change.distanceKm >= 0 && change.distanceKm <= maxDistanceKm)
    .sort((left, right) => left.distanceKm - right.distanceKm)
  if (changes.length === 0) return null
  const view = graphViewForDistance(maxDistanceKm, graphDomain)
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-cycling-dynamics-svg tri-rider-position-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, 'rider position'),
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: 100, y2: tick.vbY }),
    )
  for (let index = 0; index < changes.length; index++) {
    if (changes[index].position !== 'standing') continue
    const start = px(changes[index].distanceKm)
    const end = px(changes[index + 1]?.distanceKm ?? maxDistanceKm)
    if (end <= start) continue
    f.add(
      svgEl,
      f.svg('rect', {
        class: 'tri-rider-position-standing',
        x: start.toFixed(2),
        y: 0,
        width: (end - start).toFixed(2),
        height,
      }),
    )
  }
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  let path = `M ${px(changes[0].distanceKm).toFixed(2)} ${py(changes[0].position).toFixed(2)} `
  for (let index = 1; index < changes.length; index++) {
    const x = px(changes[index].distanceKm)
    path += `L ${x.toFixed(2)} ${py(changes[index - 1].position).toFixed(2)} L ${x.toFixed(2)} ${py(changes[index].position).toFixed(2)} `
  }
  path += `L 100 ${py(changes[changes.length - 1].position).toFixed(2)}`
  f.add(
    svgEl,
    f.svg('path', { d: path, class: 'tri-rider-position-line' }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const wrap = f.el(
    'div',
    'tri-zone tri-elev-wrap tri-cycling-dynamics-chart tri-rider-position-chart',
    undefined,
    { 'data-tri-trace': triathlonTraceName('rider position') },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const standing = dynamics.standingTimeS
  const seated = dynamics.seatedTimeS
  const total = (standing ?? 0) + (seated ?? 0)
  const summary =
    standing == null
      ? null
      : `${triText(f.presentation.locale, 'standing')} ${zoneClock(standing)}${total > 0 ? ` · ${((standing / total) * 100).toFixed(1)}%` : ''}`
  f.add(
    cap,
    f.el('span', 'tri-elev-d', triText(f.presentation.locale, 'rider position')),
    ...(summary ? [f.el('span', 'tri-elev-range', summary)] : []),
  )
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      svgEl,
      yTicks,
      height,
      distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: height },
    ),
  )
  return wrap
}
