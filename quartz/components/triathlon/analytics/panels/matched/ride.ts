import type { AxisXTick } from '../../../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../../../util/triathlon-presentation'
import type { MatchedRideGroup } from './shared'
import { axisFrame, dist, formatAltitude, niceStep } from '../../../../../util/triathlon-card'
import { triathlonDayHrefFromReference } from '../../../../../util/triathlon-date-route'
import { triathlonTraceName } from '../../../../../util/triathlon-trace-settings'
import { createDomFactory, el, svg } from '../../../runtime/dom'
import { createTriathlonFormatter } from '../../../runtime/formatter'
import { anaTitle, buildVerticalGrid, hms, markGlossDefinition } from '../../shared'
import {
  matchedActivityDate,
  matchedElapsed,
  matchedRideClimbing,
  matchedRideEffortPower,
  matchedRidePower,
  matchedRidePowerDelta,
  matchedRidePowerText,
  matchedSmoothPath,
} from './shared'

export const buildMatchedRideGroup = (
  presentation: TriathlonPresentation,
  group: MatchedRideGroup,
  currentActivityId: string,
  dayRouteHref?: string,
): HTMLElement => {
  const formatter = createTriathlonFormatter(presentation)
  const text = (key: string): string => formatter.text(key)
  const wrap = el('section', 'tri-matched tri-matched-group tri-matched--ride', undefined, {
    'data-matched-group': group.id,
    'data-tri-trace': triathlonTraceName('matched rides'),
  })
  const efforts = group.efforts
  const powers = efforts.map(effort => matchedRidePower(effort, group.powerMetric))
  const highestIndex = powers.reduce(
    (best, power, index) => (power > powers[best] ? index : best),
    0,
  )
  const currentIndex = Math.max(
    0,
    efforts.findIndex(effort => effort.id === currentActivityId),
  )
  wrap.dataset.matchedCurrentIndex = String(currentIndex)
  const W = 100
  const H = 100
  const PLOT_END = 87.64
  const AXIS_END = 86.52
  const EFFORT_SLOTS = Math.max(10, efforts.length)
  const X_DENOMINATOR = EFFORT_SLOTS * 4 - 2
  const trendingAverage = powers.map((_, index) => {
    const window = powers.slice(0, index + 1)
    return window.reduce((total, power) => total + power, 0) / window.length
  })
  const powerCenter = (group.highestPowerWatts + group.lowestPowerWatts) / 2
  const powerSpan = Math.max(10, group.highestPowerWatts - group.lowestPowerWatts)
  const chartHighest = powerCenter + powerSpan / 2
  const chartLowest = powerCenter - powerSpan / 2
  const HIGHEST_Y = 35
  const LOWEST_Y = 90
  const powerAtY = (y: number): number =>
    chartHighest + ((y - HIGHEST_Y) / (LOWEST_Y - HIGHEST_Y)) * (chartLowest - chartHighest)
  const axisFirst = powerAtY(0)
  const axisLast = powerAtY(H)
  const powerMin = Math.min(axisFirst, axisLast)
  const powerMax = Math.max(axisFirst, axisLast)
  const X = (index: number): number => (AXIS_END * (index * 4 + 1)) / X_DENOMINATOR
  const Y = (power: number): number =>
    HIGHEST_Y + ((power - chartHighest) / (chartLowest - chartHighest)) * (LOWEST_Y - HIGHEST_Y)
  const chartPath = (values: number[]): string =>
    values
      .map(
        (value, index) =>
          `${index === 0 ? 'M' : 'L'} ${X(index).toFixed(2)} ${Y(value).toFixed(2)}`,
      )
      .join(' ')

  const description = text(
    group.match === 'route'
      ? 'repeated ride routes grouped from private GPS traces'
      : 'rides grouped by similar distance, elevation, climbing density, and power provenance',
  )
  const head = el('div', 'tri-matched-head')
  head.append(
    markGlossDefinition(anaTitle(formatter, 'matched rides'), description),
    el(
      'span',
      'tri-matched-method',
      text(group.match === 'route' ? 'route match' : 'characteristics match'),
    ),
  )
  wrap.appendChild(head)

  const chart = el('div', 'tri-matched-chart')
  const graph = svg('svg', {
    class: 'tri-ana-svg tri-matched-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': text('matched rides power over time'),
  })
  const tickStep = niceStep(powerMax - powerMin, 5)
  const yTicks: { label: string; vbY: number }[] = []
  for (
    let value = Math.ceil(powerMin / tickStep) * tickStep;
    value <= powerMax + tickStep * 1e-6;
    value += tickStep
  )
    yTicks.push({ label: `${Math.round(value)}W`, vbY: Y(value) })
  graph.append(
    buildVerticalGrid(
      efforts.map((_, index) => X(index)),
      0,
      H,
    ),
    svg('line', {
      class: 'tri-matched-boundary',
      x1: 0,
      y1: Y(group.highestPowerWatts).toFixed(2),
      x2: PLOT_END,
      y2: Y(group.highestPowerWatts).toFixed(2),
      'aria-hidden': 'true',
    }),
    svg('line', {
      class: 'tri-matched-average',
      x1: 0,
      y1: Y(group.averagePowerWatts).toFixed(2),
      x2: PLOT_END,
      y2: Y(group.averagePowerWatts).toFixed(2),
      'aria-hidden': 'true',
    }),
    svg('line', {
      class: 'tri-matched-boundary',
      x1: 0,
      y1: Y(group.lowestPowerWatts).toFixed(2),
      x2: PLOT_END,
      y2: Y(group.lowestPowerWatts).toFixed(2),
      'aria-hidden': 'true',
    }),
    svg('path', { class: 'tri-matched-effort-line', d: chartPath(powers), 'aria-hidden': 'true' }),
    svg('path', {
      class: 'tri-matched-trend-line',
      d: matchedSmoothPath(trendingAverage, X, Y),
      'aria-hidden': 'true',
    }),
    svg('line', {
      class: 'tri-matched-cursor',
      x1: X(currentIndex).toFixed(2),
      y1: 0,
      x2: X(currentIndex).toFixed(2),
      y2: H,
      'aria-hidden': 'true',
    }),
  )
  const axisLabel = group.powerMetric === 'normalized' ? 'normalized power' : 'average power'
  const overlays: Array<HTMLElement | SVGElement> = [
    el('span', 'tri-matched-axis-label', text(axisLabel)),
  ]
  const annotations: {
    kind: 'highest' | 'average' | 'lowest'
    label: 'matched highest power' | 'group avg' | 'matched lowest power'
    value: number
  }[] = [
    { kind: 'highest', label: 'matched highest power', value: group.highestPowerWatts },
    { kind: 'average', label: 'group avg', value: group.averagePowerWatts },
    { kind: 'lowest', label: 'matched lowest power', value: group.lowestPowerWatts },
  ]
  for (const annotation of annotations) {
    const item = el(
      'span',
      `tri-matched-annotation tri-matched-annotation--${annotation.kind}`,
      undefined,
      { style: `left:${(PLOT_END + 1.1).toFixed(2)}%;top:${Y(annotation.value).toFixed(2)}%` },
    )
    item.append(
      el('span', 'tri-matched-annotation-label', text(annotation.label)),
      el('strong', 'tri-matched-annotation-value', matchedRidePowerText(annotation.value)),
    )
    overlays.push(item)
  }
  for (const [index, effort] of efforts.entries()) {
    const power = powers[index]
    const point = el(
      'button',
      `tri-matched-point${index === highestIndex ? ' tri-matched-point--highest' : ''}${index === currentIndex ? ' tri-matched-point--current' : ''}`,
      undefined,
      {
        type: 'button',
        'data-matched-index': String(index),
        'data-matched-x': X(index).toFixed(2),
        'data-matched-title':
          index === currentIndex
            ? text('this ride')
            : matchedActivityDate(presentation, effort.date),
        'data-matched-value': matchedRidePowerText(power),
        'data-matched-delta': matchedRidePowerDelta(power, group.averagePowerWatts),
        'data-matched-direction': 'equal',
        'data-selected': String(index === currentIndex),
        'aria-pressed': String(index === currentIndex),
        'aria-label': `${formatter.longDate(effort.date)} · ${matchedRidePowerText(power)}`,
        style: `left:${X(index).toFixed(2)}%;top:${((Y(power) / H) * 100).toFixed(2)}%`,
      },
    )
    overlays.push(point)
  }
  const current = efforts[currentIndex]
  const currentPower = powers[currentIndex]
  const readout = el('div', 'tri-matched-readout', undefined, {
    'aria-live': 'polite',
    'data-direction': 'equal',
    style: `left:${X(currentIndex).toFixed(2)}%`,
  })
  readout.append(
    el('span', 'tri-matched-readout-title', text('this ride')),
    el('strong', 'tri-matched-readout-value', matchedRidePowerText(currentPower)),
    el(
      'span',
      'tri-matched-readout-delta',
      matchedRidePowerDelta(currentPower, group.averagePowerWatts),
    ),
  )
  overlays.push(readout)
  const xTicks: AxisXTick[] = [
    { label: matchedActivityDate(presentation, current.date), pct: X(currentIndex) },
  ]
  chart.appendChild(
    axisFrame(
      createDomFactory(presentation),
      graph,
      yTicks,
      H,
      xTicks,
      true,
      { top: 0, bottom: H },
      overlays,
    ),
  )
  const legend = el('div', 'tri-matched-legend')
  const count = el('span', 'tri-matched-legend-count')
  count.append(el('strong', undefined, String(efforts.length)), ` ${text('rides')}`)
  const trend = el('span', 'tri-matched-legend-item')
  trend.append(
    el('span', 'tri-matched-legend-line', undefined, { 'aria-hidden': 'true' }),
    el('span', undefined, text('trending average')),
  )
  const characteristics = el(
    'span',
    'tri-matched-legend-item',
    `${dist(presentation, group.averageDistanceKm, 'bike')} · +${formatAltitude(presentation, group.averageElevationM)} · ${matchedRideClimbing(presentation, group.averageClimbingMPerKm)}`,
  )
  legend.append(count, trend, characteristics)
  chart.appendChild(legend)
  wrap.appendChild(chart)

  const viewport = el('div', 'tri-effort-viewport tri-matched-viewport')
  const scroll = el('div', 'tri-effort-scroll')
  const table = el(
    'table',
    'tri-effort-table tri-matched-table tri-matched-table--ride',
    undefined,
    { 'aria-label': text('matched rides history') },
  )
  const thead = document.createElement('thead')
  const headRow = document.createElement('tr')
  for (const label of [
    'date',
    'activity',
    'distance',
    'elevation',
    'climbing density',
    'average power',
    'normalized power',
    'moving time',
    'elapsed time',
  ])
    headRow.appendChild(el('th', undefined, text(label), { scope: 'col' }))
  thead.appendChild(headRow)
  const tbody = document.createElement('tbody')
  for (let index = efforts.length - 1; index >= 0; index--) {
    const effort = efforts[index]
    const row = el('tr', undefined, undefined, {
      'data-matched-index': String(index),
      'data-selected': String(index === currentIndex),
      'data-current': String(index === currentIndex),
    })
    const activityCell = document.createElement('td')
    const activityLane = el('span', 'tri-matched-activity-lane')
    if (index === highestIndex)
      activityLane.appendChild(el('span', 'tri-matched-highest', text('highest')))
    const dayHref = triathlonDayHrefFromReference(effort.date, dayRouteHref)
    activityLane.appendChild(
      dayHref
        ? el('a', 'tri-matched-activity internal', effort.name, {
            href: dayHref,
            ...(index === currentIndex ? { 'aria-current': 'true' } : {}),
          })
        : el('span', 'tri-matched-activity', effort.name),
    )
    activityCell.appendChild(activityLane)
    row.append(
      el('th', undefined, formatter.shortDate(effort.date), { scope: 'row' }),
      activityCell,
      el('td', undefined, dist(presentation, effort.distanceKm, 'bike')),
      el('td', undefined, `+${formatAltitude(presentation, effort.elevationM)}`),
      el('td', undefined, matchedRideClimbing(presentation, effort.climbingMPerKm)),
      el('td', undefined, matchedRideEffortPower(presentation, effort, effort.averageWatts)),
      el('td', undefined, matchedRideEffortPower(presentation, effort, effort.normalizedWatts)),
      el('td', undefined, hms(effort.movingTimeS)),
      el('td', undefined, matchedElapsed(effort)),
    )
    tbody.appendChild(row)
  }
  table.append(thead, tbody)
  scroll.appendChild(table)
  viewport.appendChild(scroll)
  wrap.appendChild(viewport)
  return wrap
}
