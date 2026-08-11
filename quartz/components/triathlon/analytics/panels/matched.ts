import type { MatchedRidesBlock } from '../../../../plugins/stores/matched-rides'
import type { MatchedRunsBlock } from '../../../../plugins/stores/matched-runs'
import type { AxisXTick } from '../../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import { axisFrame } from '../../../../util/triathlon-card'
import { clock } from '../../../../util/triathlon-card'
import { dist } from '../../../../util/triathlon-card'
import { formatAltitude } from '../../../../util/triathlon-card'
import { KM_TO_MI } from '../../../../util/triathlon-card'
import { M_TO_FT } from '../../../../util/triathlon-card'
import { niceStep } from '../../../../util/triathlon-card'
import { triathlonDayHrefFromReference } from '../../../../util/triathlon-date-route'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { createTriathlonFormatter } from '../../runtime/formatter'
import { anaTitle } from '../shared'
import { hms } from '../shared'
import { markGlossDefinition } from '../shared'

export type MatchedRunGroup = MatchedRunsBlock['groups'][number]

export type MatchedRideGroup = MatchedRidesBlock['groups'][number]

export const matchedRunUnitScale = (presentation: TriathlonPresentation): number =>
  presentation.distance === 'imperial' ? KM_TO_MI : 1

export const matchedRunDisplayPace = (
  presentation: TriathlonPresentation,
  paceSPerKm: number,
): number => paceSPerKm / matchedRunUnitScale(presentation)

export const matchedRunPace = (presentation: TriathlonPresentation, paceSPerKm: number): string =>
  `${clock(matchedRunDisplayPace(presentation, paceSPerKm))}${presentation.distance === 'imperial' ? '/mi' : '/km'}`

export const matchedRunDelta = (
  presentation: TriathlonPresentation,
  paceSPerKm: number,
  averagePaceSPerKm: number,
): string => {
  const delta = Math.round((paceSPerKm - averagePaceSPerKm) / matchedRunUnitScale(presentation))
  const sign = delta > 0 ? '+' : delta < 0 ? '-' : ''
  return `${sign}${Math.abs(delta)}s${presentation.distance === 'imperial' ? '/mi' : '/km'}`
}

export const matchedRunDirection = (
  paceSPerKm: number,
  averagePaceSPerKm: number,
): 'faster' | 'slower' | 'equal' =>
  paceSPerKm < averagePaceSPerKm ? 'faster' : paceSPerKm > averagePaceSPerKm ? 'slower' : 'equal'

export const matchedActivityDate = (presentation: TriathlonPresentation, iso: string): string => {
  const date = new Date(`${iso}T12:00:00Z`)
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
        year: '2-digit',
        month: 'numeric',
        day: 'numeric',
        timeZone: 'UTC',
      })
}

export const matchedRidePower = (
  effort: MatchedRideGroup['efforts'][number],
  metric: MatchedRideGroup['powerMetric'],
): number =>
  metric === 'normalized' ? (effort.normalizedWatts ?? effort.averageWatts) : effort.averageWatts

export const matchedRidePowerText = (watts: number): string => `${Math.round(watts)} W`

export const matchedRidePowerDelta = (watts: number, averageWatts: number): string => {
  const delta = Math.round(watts - averageWatts)
  return `${delta > 0 ? '+' : delta < 0 ? '-' : ''}${Math.abs(delta)} W`
}

export const matchedRideClimbing = (
  presentation: TriathlonPresentation,
  metersPerKm: number,
): string =>
  presentation.distance === 'imperial'
    ? `${Math.round((metersPerKm * M_TO_FT) / KM_TO_MI)} ft/mi`
    : `${Math.round(metersPerKm)} m/km`

export const matchedRideEffortPower = (
  presentation: TriathlonPresentation,
  effort: MatchedRideGroup['efforts'][number],
  value: number | null,
): string =>
  value == null
    ? '—'
    : `${matchedRidePowerText(value)}${effort.powerSource === 'estimate' ? ` ${createTriathlonFormatter(presentation).text('estimated')}` : ''}`

export const matchedRunTrendingAverage = (group: MatchedRunGroup): number[] =>
  group.efforts.map((_, index) => {
    const window = group.efforts.slice(0, index + 1)
    return window.reduce((total, effort) => total + effort.paceSPerKm, 0) / window.length
  })

export const matchedSmoothPath = (
  values: number[],
  xOf: (index: number) => number,
  yOf: (value: number) => number,
): string => {
  const points = values.map((value, index) => ({ x: xOf(index), y: yOf(value) }))
  if (points.length < 2) return ''
  if (points.length === 2)
    return `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)} L ${points[1].x.toFixed(2)} ${points[1].y.toFixed(2)}`
  let path = `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`
  for (let index = 0; index < points.length - 1; index++) {
    const previous = points[Math.max(0, index - 1)]
    const start = points[index]
    const end = points[index + 1]
    const next = points[Math.min(points.length - 1, index + 2)]
    const firstX = start.x + (end.x - previous.x) * 0.15
    const firstY = start.y + (end.y - previous.y) * 0.15
    const secondX = end.x - (next.x - start.x) * 0.15
    const secondY = end.y - (next.y - start.y) * 0.15
    path += ` C ${firstX.toFixed(2)} ${firstY.toFixed(2)} ${secondX.toFixed(2)} ${secondY.toFixed(2)} ${end.x.toFixed(2)} ${end.y.toFixed(2)}`
  }
  return path
}

export const buildMatchedRunGroup = (
  presentation: TriathlonPresentation,
  group: MatchedRunGroup,
  currentActivityId: number,
  dayRouteHref?: string,
): HTMLElement => {
  const formatter = createTriathlonFormatter(presentation)
  const text = (key: string): string => formatter.text(key)
  const wrap = el('section', 'tri-matched tri-matched-group', undefined, {
    'data-matched-group': group.id,
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

export const buildMatchedRideGroup = (
  presentation: TriathlonPresentation,
  group: MatchedRideGroup,
  currentActivityId: number,
  dayRouteHref?: string,
): HTMLElement => {
  const formatter = createTriathlonFormatter(presentation)
  const text = (key: string): string => formatter.text(key)
  const wrap = el('section', 'tri-matched tri-matched-group tri-matched--ride', undefined, {
    'data-matched-group': group.id,
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
    )
    tbody.appendChild(row)
  }
  table.append(thead, tbody)
  scroll.appendChild(table)
  viewport.appendChild(scroll)
  wrap.appendChild(viewport)
  return wrap
}
