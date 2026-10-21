import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { triathlonTraceName } from '../triathlon-trace-settings'
import { axisFrame, distanceXTicks, graphView, type ActivityGraphDomain } from './chart-geometry'
import { buildAnalysisSelection } from './chart-selection'
import { powerBalanceText } from './metric-text'
import { buildCyclingChartModes, buildWattsHeatmap } from './watts-heatmap'

const routeRightPowerPct = (point: StravaActivityDetail['route'][number]): number | null => {
  const value = point.rightPowerPct
  return value != null && Number.isFinite(value) && value >= 0 && value <= 100 && point.w > 0
    ? value
    : null
}

type PowerBalanceSample = { watts: number; rightPowerPct: number; distanceKm: number }

const powerBalanceSamples = (
  d: StravaActivityDetail,
  graphDomain?: ActivityGraphDomain | null,
): PowerBalanceSample[] => {
  const measured = d.route.flatMap(point => {
    const rightPowerPct = routeRightPowerPct(point)
    return rightPowerPct == null ? [] : [{ watts: point.w, rightPowerPct, distanceKm: point.d }]
  })
  if (!graphDomain) return measured
  const selected = measured.filter(
    sample =>
      sample.distanceKm >= graphDomain.startDistanceKm &&
      sample.distanceKm <= graphDomain.endDistanceKm,
  )
  return selected.length > 0 ? selected : measured
}

const activityRightPowerPct = (d: StravaActivityDetail): number | null => {
  let work = 0
  let rightWork = 0
  for (let index = 1; index < d.route.length; index++) {
    const previous = d.route[index - 1]
    const next = d.route[index]
    const previousRight = routeRightPowerPct(previous)
    const nextRight = routeRightPowerPct(next)
    const elapsedS = next.elapsedS - previous.elapsedS
    if (previousRight == null || nextRight == null || elapsedS <= 0) continue
    work += ((previous.w + next.w) / 2) * elapsedS
    rightWork += ((previous.w * previousRight + next.w * nextRight) / 2) * elapsedS
  }
  return work > 0 ? rightWork / work : null
}

interface PowerBalancePaths {
  measured: string
  bridges: string
}

const powerBalancePaths = (
  d: StravaActivityDetail,
  pick: (rightPowerPct: number) => number,
  domainMin: number,
  domainMax: number,
  width: number,
  height: number,
): PowerBalancePaths => {
  const maxDistanceKm = d.route.at(-1)?.d || 1
  const px = (distanceKm: number): number => (distanceKm / maxDistanceKm) * width
  const py = (value: number): number => {
    const bounded = Math.min(domainMax, Math.max(domainMin, value))
    return height - ((bounded - domainMin) / (domainMax - domainMin)) * (height - 1)
  }
  let measured = ''
  let bridges = ''
  let segmentStart = -1
  let previousValid = -1
  const closeSegment = (start: number, end: number): void => {
    for (let index = start; index <= end; index++) {
      const rightPowerPct = routeRightPowerPct(d.route[index])
      if (rightPowerPct == null) continue
      measured += `${index === start ? 'M' : 'L'} ${px(d.route[index].d).toFixed(2)} ${py(pick(rightPowerPct)).toFixed(2)} `
    }
  }
  d.route.forEach((point, index) => {
    const rightPowerPct = routeRightPowerPct(point)
    const valid = rightPowerPct != null
    if (valid && previousValid >= 0 && index > previousValid + 1) {
      const previousRightPowerPct = routeRightPowerPct(d.route[previousValid])
      if (previousRightPowerPct != null)
        bridges += `M ${px(d.route[previousValid].d).toFixed(2)} ${py(pick(previousRightPowerPct)).toFixed(2)} L ${px(point.d).toFixed(2)} ${py(pick(rightPowerPct)).toFixed(2)} `
    }
    if (valid) previousValid = index
    if (valid && segmentStart < 0) segmentStart = index
    if (segmentStart >= 0 && (!valid || index === d.route.length - 1)) {
      closeSegment(segmentStart, valid ? index : index - 1)
      segmentStart = -1
    }
  })
  return { measured, bridges }
}

export const buildPowerBalanceChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const samples = powerBalanceSamples(d, graphDomain)
  const rightValues = d.route.flatMap(point => {
    const rightPowerPct = routeRightPowerPct(point)
    return rightPowerPct == null ? [] : [rightPowerPct]
  })
  if (d.sport !== 'bike' || rightValues.length < 2) return null
  const width = 100
  const height = 30
  const deviations = rightValues.map(value => Math.abs(value - 50)).sort((a, b) => a - b)
  const observed = deviations[Math.round((deviations.length - 1) * 0.9)]
  const span = Math.min(25, Math.max(5, Math.ceil(observed / 2) * 2))
  const domainMin = 50 - span
  const domainMax = 50 + span
  const py = (value: number): number =>
    height - ((value - domainMin) / (domainMax - domainMin)) * (height - 1)
  const yTicks = [domainMin, 50, domainMax].map(value => ({ label: `${value}%`, vbY: py(value) }))
  const view = graphView(d, graphDomain)
  const leftPaths = powerBalancePaths(
    d,
    rightPowerPct => 100 - rightPowerPct,
    domainMin,
    domainMax,
    width,
    height,
  )
  const rightPaths = powerBalancePaths(
    d,
    rightPowerPct => rightPowerPct,
    domainMin,
    domainMax,
    width,
    height,
  )
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-power-balance-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, 'power balance'),
    'data-i18n-aria-label': 'power balance',
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: width, y2: tick.vbY }),
    )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  if (leftPaths.bridges)
    f.add(
      svgEl,
      f.svg('path', {
        d: leftPaths.bridges,
        class:
          'tri-power-balance-line tri-power-balance-line--left tri-power-balance-line--missing',
      }),
    )
  if (rightPaths.bridges)
    f.add(
      svgEl,
      f.svg('path', {
        d: rightPaths.bridges,
        class:
          'tri-power-balance-line tri-power-balance-line--right tri-power-balance-line--missing',
      }),
    )
  f.add(
    svgEl,
    f.svg('path', {
      d: leftPaths.measured,
      class: 'tri-power-balance-line tri-power-balance-line--left',
    }),
    f.svg('path', {
      d: rightPaths.measured,
      class: 'tri-power-balance-line tri-power-balance-line--right',
    }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const heatmap = buildWattsHeatmap(
    f,
    d,
    'power balance',
    'tri-power-balance-heatmap',
    'power-balance',
    [
      {
        side: 'single',
        samples: samples.map(sample => ({ watts: sample.watts, value: sample.rightPowerPct })),
      },
    ],
    0,
    100,
    height,
    false,
  )
  const heatSvg = heatmap.svg
  f.add(
    heatSvg,
    f.svg('line', {
      class: 'tri-power-balance-reference',
      x1: 0,
      y1: height / 2,
      x2: width,
      y2: height / 2,
    }),
  )
  const wrap = f.el(
    'div',
    'tri-zone tri-elev-wrap tri-cycling-mode-chart tri-power-balance-chart',
    undefined,
    {
      'data-tri-trace': triathlonTraceName('power balance'),
      'data-cycling-chart-mode': 'distance',
    },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-power-balance-summary')
  const rightAverage = activityRightPowerPct(d)
  if (rightAverage != null)
    f.add(
      summary,
      f.el('span', 'tri-elev-range', `${powerBalanceText(rightAverage)}${embedded ? '' : ' avg'}`),
    )
  if (!embedded) {
    for (const side of ['left', 'right'] as const) {
      const item = f.el(
        'span',
        `tri-power-balance-legend-item tri-power-balance-legend-item--${side}`,
      )
      f.add(
        item,
        f.el('span', 'tri-power-balance-legend-line', undefined, { 'aria-hidden': 'true' }),
        f.el('span', 'tri-power-balance-legend-label', side, { 'data-i18n': side }),
      )
      f.add(summary, item)
    }
  }
  f.add(
    cap,
    f.el('span', 'tri-elev-d', 'power balance', { 'data-i18n': 'power balance' }),
    summary,
    buildCyclingChartModes(f),
    heatmap.readout,
  )
  const distancePane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--distance tri-power-balance-pane--distance',
    undefined,
    { 'data-cycling-chart-mode': 'distance', 'aria-hidden': 'false' },
  )
  f.add(
    distancePane,
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
  const powerPane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--power tri-power-balance-pane--power',
    undefined,
    { 'data-cycling-chart-mode': 'power', 'aria-hidden': 'true', hidden: '' },
  )
  f.add(
    powerPane,
    axisFrame(
      f,
      heatSvg,
      [
        { label: '100% L', vbY: 0 },
        { label: '50/50', vbY: height / 2 },
        { label: '100% R', vbY: height },
      ],
      height,
      heatmap.wattsTicks,
      true,
      { top: 0, bottom: height },
    ),
  )
  f.add(wrap, cap, distancePane, powerPane)
  return wrap
}
