import type { AxisXTick } from '../../../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../../../util/triathlon-presentation'
import type { MatchedRunGroup } from './shared'
import { axisFrame, clock, niceStep } from '../../../../../util/triathlon-card'
import { triathlonDayHrefFromReference } from '../../../../../util/triathlon-date-route'
import { triathlonTraceName } from '../../../../../util/triathlon-trace-settings'
import { createDomFactory, el, svg } from '../../../runtime/dom'
import { createTriathlonFormatter } from '../../../runtime/formatter'
import { anaTitle, buildVerticalGrid, hms, markGlossDefinition } from '../../shared'
import {
  matchedActivityDate,
  matchedElapsed,
  matchedRunDelta,
  matchedRunDirection,
  matchedRunDisplayPace,
  matchedRunPace,
  matchedRunTrendingAverage,
  matchedSmoothPath,
} from './shared'

export const buildMatchedRunGroup = (
  presentation: TriathlonPresentation,
  group: MatchedRunGroup,
  currentActivityId: string,
  dayRouteHref?: string,
): HTMLElement => {
  const formatter = createTriathlonFormatter(presentation)
  const text = (key: string): string => formatter.text(key)
  const wrap = el('section', 'tri-matched tri-matched-group', undefined, {
    'data-matched-group': group.id,
    'data-tri-trace': triathlonTraceName('matched runs'),
  })
  const efforts = group.efforts
  const fastestIndex = efforts.reduce(
    (best, effort, index) => (effort.paceSPerKm < efforts[best].paceSPerKm ? index : best),
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
  const paces = efforts.map(effort => effort.paceSPerKm)
  const trendingAverage = matchedRunTrendingAverage(group)
  const fastestPace = matchedRunDisplayPace(presentation, group.fastestPaceSPerKm)
  const slowestPace = matchedRunDisplayPace(presentation, group.slowestPaceSPerKm)
  const paceCenter = (fastestPace + slowestPace) / 2
  const paceSpan = Math.max(5, slowestPace - fastestPace)
  const paceFastest = paceCenter - paceSpan / 2
  const paceSlowest = paceCenter + paceSpan / 2
  const FASTEST_Y = 35
  const SLOWEST_Y = 90
  const paceMin = paceFastest - (paceSpan * FASTEST_Y) / (SLOWEST_Y - FASTEST_Y)
  const paceMax = paceSlowest + (paceSpan * (H - SLOWEST_Y)) / (SLOWEST_Y - FASTEST_Y)
  const X = (index: number): number => (AXIS_END * (index * 4 + 1)) / X_DENOMINATOR
  const displayY = (pace: number): number =>
    FASTEST_Y + ((pace - paceFastest) / paceSpan) * (SLOWEST_Y - FASTEST_Y)
  const Y = (pace: number): number => displayY(matchedRunDisplayPace(presentation, pace))
  const chartPath = (values: number[]): string =>
    values
      .map(
        (value, index) =>
          `${index === 0 ? 'M' : 'L'} ${X(index).toFixed(2)} ${Y(value).toFixed(2)}`,
      )
      .join(' ')

  const head = el('div', 'tri-matched-head')
  head.appendChild(
    markGlossDefinition(
      anaTitle(formatter, 'matched runs'),
      text('repeated routes grouped from private GPS traces'),
    ),
  )
  wrap.appendChild(head)

  const chart = el('div', 'tri-matched-chart')
  const graph = svg('svg', {
    class: 'tri-ana-svg tri-matched-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': text('matched runs pace over time'),
  })
  const tickStep = niceStep(paceMax - paceMin, 5)
  const yTicks: { label: string; vbY: number }[] = []
  for (
    let value = Math.ceil(paceMin / tickStep) * tickStep;
    value <= paceMax + tickStep * 1e-6;
    value += tickStep
  )
    yTicks.push({
      label: `${clock(value)}${presentation.distance === 'imperial' ? '/mi' : '/km'}`,
      vbY: displayY(value),
    })
  graph.append(
    buildVerticalGrid(
      efforts.map((_, index) => X(index)),
      0,
      H,
    ),
    svg('line', {
      class: 'tri-matched-boundary',
      x1: 0,
      y1: Y(group.fastestPaceSPerKm).toFixed(2),
      x2: PLOT_END,
      y2: Y(group.fastestPaceSPerKm).toFixed(2),
      'aria-hidden': 'true',
    }),
    svg('line', {
      class: 'tri-matched-average',
      x1: 0,
      y1: Y(group.averagePaceSPerKm).toFixed(2),
      x2: PLOT_END,
      y2: Y(group.averagePaceSPerKm).toFixed(2),
      'aria-hidden': 'true',
    }),
    svg('line', {
      class: 'tri-matched-boundary',
      x1: 0,
      y1: Y(group.slowestPaceSPerKm).toFixed(2),
      x2: PLOT_END,
      y2: Y(group.slowestPaceSPerKm).toFixed(2),
      'aria-hidden': 'true',
    }),
    svg('path', { class: 'tri-matched-effort-line', d: chartPath(paces), 'aria-hidden': 'true' }),
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
  const overlays: Array<HTMLElement | SVGElement> = [
    el('span', 'tri-matched-axis-label', text('pace')),
  ]
  const annotations: {
    kind: 'fastest' | 'average' | 'slowest'
    label: 'matched fastest' | 'all-time avg' | 'matched slowest'
    value: number
  }[] = [
    { kind: 'fastest', label: 'matched fastest', value: group.fastestPaceSPerKm },
    { kind: 'average', label: 'all-time avg', value: group.averagePaceSPerKm },
    { kind: 'slowest', label: 'matched slowest', value: group.slowestPaceSPerKm },
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
      el('strong', 'tri-matched-annotation-value', matchedRunPace(presentation, annotation.value)),
    )
    overlays.push(item)
  }
  for (const [index, effort] of efforts.entries()) {
    const point = el(
      'button',
      `tri-matched-point${index === fastestIndex ? ' tri-matched-point--fastest' : ''}${index === currentIndex ? ' tri-matched-point--current' : ''}`,
      undefined,
      {
        type: 'button',
        'data-matched-index': String(index),
        'data-matched-x': X(index).toFixed(2),
        'data-matched-title':
          index === currentIndex
            ? text('this run')
            : matchedActivityDate(presentation, effort.date),
        'data-matched-value': matchedRunPace(presentation, effort.paceSPerKm),
        'data-matched-delta': matchedRunDelta(
          presentation,
          effort.paceSPerKm,
          group.averagePaceSPerKm,
        ),
        'data-matched-direction': matchedRunDirection(effort.paceSPerKm, group.averagePaceSPerKm),
        'data-selected': String(index === currentIndex),
        'aria-pressed': String(index === currentIndex),
        'aria-label': `${formatter.longDate(effort.date)} · ${matchedRunPace(presentation, effort.paceSPerKm)}`,
        style: `left:${X(index).toFixed(2)}%;top:${((Y(effort.paceSPerKm) / H) * 100).toFixed(2)}%`,
      },
    )
    overlays.push(point)
  }
  const current = efforts[currentIndex]
  const readout = el('div', 'tri-matched-readout', undefined, {
    'aria-live': 'polite',
    'data-direction': matchedRunDirection(current.paceSPerKm, group.averagePaceSPerKm),
    style: `left:${X(currentIndex).toFixed(2)}%`,
  })
  readout.append(
    el('span', 'tri-matched-readout-title', text('this run')),
    el('strong', 'tri-matched-readout-value', matchedRunPace(presentation, current.paceSPerKm)),
    el(
      'span',
      'tri-matched-readout-delta',
      matchedRunDelta(presentation, current.paceSPerKm, group.averagePaceSPerKm),
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
  count.append(el('strong', undefined, String(efforts.length)), ` ${text('runs')}`)
  const trend = el('span', 'tri-matched-legend-item')
  trend.append(
    el('span', 'tri-matched-legend-line', undefined, { 'aria-hidden': 'true' }),
    el('span', undefined, text('trending average')),
  )
  legend.append(count, trend)
  chart.appendChild(legend)
  wrap.appendChild(chart)

  const viewport = el('div', 'tri-effort-viewport tri-matched-viewport')
  const scroll = el('div', 'tri-effort-scroll')
  const table = el('table', 'tri-effort-table tri-matched-table', undefined, {
    'aria-label': text('matched runs history'),
  })
  const thead = document.createElement('thead')
  const headRow = document.createElement('tr')
  for (const label of [
    'date',
    'activity',
    'pace',
    'vs route avg',
    'moving time',
    'elapsed time',
    'relative effort',
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
    if (index === fastestIndex)
      activityLane.appendChild(el('span', 'tri-matched-fastest', text('fastest')))
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
      el('td', undefined, matchedRunPace(presentation, effort.paceSPerKm)),
      el(
        'td',
        undefined,
        matchedRunDelta(presentation, effort.paceSPerKm, group.averagePaceSPerKm),
      ),
      el('td', undefined, hms(effort.movingTimeS)),
      el('td', undefined, matchedElapsed(effort)),
      el('td', undefined, effort.relativeEffort == null ? '—' : String(effort.relativeEffort)),
    )
    tbody.appendChild(row)
  }
  table.append(thead, tbody)
  scroll.appendChild(table)
  viewport.appendChild(scroll)
  wrap.appendChild(viewport)
  return wrap
}
