import type { Analytics } from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import { clock } from '../../../../util/triathlon-card'
import { formatThermalTemperature } from '../../../../util/triathlon-card'
import { KM_TO_MI } from '../../../../util/triathlon-card'
import { zoneClock } from '../../../../util/triathlon-card'
import { isRecord } from '../../../../util/type-guards'
import { buildIcon } from '../../activity/primitives'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { nextMapMetricShortcutIndex } from '../../shell/command-palette'
import { buildDatePicker } from '../../tools/date-picker'
import { parsePredDate } from '../../tools/date-picker'
import { anaTitle } from '../shared'
import { buildTrendGlyph } from '../shared'
import { clampN } from '../shared'
import { polyD } from '../shared'
import { missingBridges } from './body'
import { segRuns } from './body'
import {
  DISTRIBUTION_RANGES,
  distributionMetricForSport,
  distributionMetrics,
  initialDistributionModel,
  telemetryTrend,
  telemetryWeightedAverage,
  updateDistributions,
  type DistributionRange,
  type DistributionMetric,
  type DistributionSport,
} from './distributions-model'

export type ActivityDistributionPoint = Analytics['distributions']['activities'][number]

export const TRI_DISTRIBUTION_SELECTION_KEY = 'tri-distribution-selection'

export const DISTRIBUTION_ZONE_NAMES: Record<DistributionMetric, readonly string[]> = {
  'heart-rate': ['endurance', 'moderate', 'tempo', 'threshold', 'anaerobic'],
  power: ['recovery', 'moderate', 'tempo', 'threshold', 'VO2max', 'anaerobic', 'neuromuscular'],
  pace: ['recovery', 'endurance', 'tempo', 'threshold', 'VO2max', 'anaerobic'],
}

export const distributionZoneRange = (bounds: readonly number[], index: number): string => {
  if (bounds.length === 0) return ''
  if (index === 0) return `≤${bounds[0]} bpm`
  if (index >= bounds.length) return `${bounds[bounds.length - 1] + 1}+ bpm`
  return `${bounds[index - 1] + 1}–${bounds[index]} bpm`
}

export const distributionPowerZoneRange = (bounds: readonly number[], index: number): string => {
  if (bounds.length === 0) return ''
  if (index === 0) return `0–${bounds[0]} W`
  if (index >= bounds.length) return `${bounds[bounds.length - 1] + 1}+ W`
  return `${bounds[index - 1] + 1}–${bounds[index]} W`
}

const distributionPaceZoneRange = (
  context: TriathlonContext,
  bounds: readonly number[],
  index: number,
): string => {
  if (bounds.length === 0) return '—'
  const imperial = context.presentation.distance === 'imperial'
  const scale = imperial ? KM_TO_MI : 1
  const unit = imperial ? '/mi' : '/km'
  if (index === 0) return `>${clock(bounds[0] / scale)}${unit}`
  if (index >= bounds.length) return `<${clock(bounds[bounds.length - 1] / scale)}${unit}`
  return `${clock(bounds[index] / scale)}–${clock(bounds[index - 1] / scale)}${unit}`
}

interface DistributionZoneSeries {
  metric: DistributionMetric
  seconds: number[]
  ranges: string[]
  observedActivities: number
}

const buildZoneMetricIcon = (metric: DistributionMetric): SVGElement => {
  const icon = svg('svg', {
    class: 'tri-zone-metric-icon',
    viewBox: '0 0 24 24',
    'aria-hidden': 'true',
    focusable: 'false',
  })
  icon.appendChild(
    svg('path', {
      d:
        metric === 'heart-rate'
          ? 'M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78a5.5 5.5 0 0 0 0-7.78Z'
          : metric === 'power'
            ? 'M13 2 4 14h8l-1 8 9-12h-8l1-8Z'
            : 'M9 2h6M12 5a8 8 0 1 0 8 8M12 9v4l3 2',
      fill: 'none',
      stroke: 'currentColor',
      'stroke-width': 1.8,
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    }),
  )
  return icon
}

const sumDistributionZones = (
  points: readonly ActivityDistributionPoint[],
  count: number,
  values: (point: ActivityDistributionPoint) => readonly number[] | null,
): { seconds: number[]; observedActivities: number } => {
  const seconds = Array.from({ length: count }, () => 0)
  let observedActivities = 0
  for (const point of points) {
    const zones = values(point)
    if (!zones?.some(value => value > 0)) continue
    observedActivities += 1
    zones.forEach((value, index) => {
      if (index < seconds.length && Number.isFinite(value) && value > 0) seconds[index] += value
    })
  }
  return { seconds, observedActivities }
}

export const buildDistributions = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount?: () => () => void } => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-training-distribution')
  const { activities, heartRateZoneBounds, powerZoneBounds, paceZoneBoundsSPerKm, tenKmRaceTimeS } =
    data.distributions
  if (activities.length === 0) {
    block.append(anaTitle(context.formatter, 'training zone distributions', 'hrzones'))
    block.appendChild(el('div', 'tri-ana-empty', text('no activity distribution data')))
    return { element: block }
  }

  const minimumDate = data.meta.windowFrom
  const maximumDate = data.meta.windowTo
  const bounds = {
    minimumDate,
    maximumDate,
    sports: [...new Set(activities.map(activity => activity.sport))],
  }
  let model = initialDistributionModel(bounds)
  let { sport, metric, range, startDate } = model
  const applyModel = (next: typeof model): void => {
    model = next
    sport = model.sport
    metric = model.metric
    range = model.range
    startDate = model.startDate
  }

  const persist = (): void => {
    try {
      localStorage.setItem(
        TRI_DISTRIBUTION_SELECTION_KEY,
        JSON.stringify({ sport, metric, range, startDate }),
      )
    } catch {}
  }

  const head = el('div', 'tri-dist-head')
  head.appendChild(anaTitle(context.formatter, 'training zone distributions', 'hrzones'))
  const controls = el('div', 'tri-dist-controls')
  const metricControls = el('div', 'tri-map-tablist tri-zone-metric-tabs', undefined, {
    role: 'tablist',
    'aria-label': text('training zone metric'),
  })
  const sportControls = el('div', 'tri-dist-sports', undefined, {
    role: 'group',
    'aria-label': text('distribution sport'),
  })
  const sportButtons = new Map<DistributionSport, HTMLButtonElement>()
  for (const option of ['swim', 'bike', 'run'] as DistributionSport[]) {
    const button = el(
      'button',
      `tri-radar-sport tri-dist-sport tri-radar-sport--${option}`,
      undefined,
      {
        type: 'button',
        'aria-label': text(option),
        'aria-pressed': String(option === sport),
        title: text(option),
        'data-sport': option,
      },
    ) as HTMLButtonElement
    button.appendChild(buildIcon(context.presentation, option))
    sportButtons.set(option, button)
    sportControls.appendChild(button)
  }

  const rangeControls = el('div', 'tri-dist-ranges', undefined, {
    role: 'group',
    'aria-label': text('date range'),
  })
  const rangeButtons = new Map<DistributionRange, HTMLButtonElement>()
  for (const option of DISTRIBUTION_RANGES) {
    const button = el('button', 'tri-dist-range', text(option.label), {
      type: 'button',
      'aria-pressed': String(option.key === range),
      'data-range': option.key,
    }) as HTMLButtonElement
    rangeButtons.set(option.key, button)
    rangeControls.appendChild(button)
  }

  const startPicker = buildDatePicker({
    id: 'tri-distribution-start-date',
    formatter: context.formatter,
    label: text('range start'),
    selected: () => startDate,
    min: () => minimumDate,
    max: () => maximumDate,
    onOpen: () => {
      applyModel(updateDistributions(model, { type: 'select-range', range: 'custom' }, bounds))
      persist()
      render()
    },
    onSelect: date => {
      applyModel(updateDistributions(model, { type: 'select-date', date }, bounds))
      persist()
      render()
    },
    onClear: () => {
      applyModel(updateDistributions(model, { type: 'clear-date' }, bounds))
      persist()
      render()
    },
  })
  startPicker.wrap.classList.add('tri-dist-date')
  controls.append(metricControls, sportControls, rangeControls, startPicker.wrap)
  head.appendChild(controls)
  block.appendChild(head)

  const zonePanel = el('section', 'tri-training-zones')
  const telemetryPanel = el('section', 'tri-activity-telemetry')
  block.append(zonePanel, telemetryPanel)

  const selectedActivities = (): ActivityDistributionPoint[] =>
    activities.filter(
      point => point.sport === sport && point.date >= startDate && point.date <= maximumDate,
    )

  const zoneSeries = (
    points: readonly ActivityDistributionPoint[],
    selectedMetric: DistributionMetric,
  ): DistributionZoneSeries => {
    if (selectedMetric === 'heart-rate') {
      const count = heartRateZoneBounds.length > 0 ? heartRateZoneBounds.length + 1 : 5
      return {
        metric: selectedMetric,
        ...sumDistributionZones(points, count, point => point.heartRateZoneSeconds),
        ranges: Array.from({ length: count }, (_, index) =>
          distributionZoneRange(heartRateZoneBounds, index),
        ),
      }
    }
    if (selectedMetric === 'power') {
      const count = powerZoneBounds.length > 0 ? powerZoneBounds.length + 1 : 7
      return {
        metric: selectedMetric,
        ...sumDistributionZones(points, count, point => point.powerZoneSeconds),
        ranges: Array.from({ length: count }, (_, index) =>
          distributionPowerZoneRange(powerZoneBounds, index),
        ),
      }
    }
    const count = paceZoneBoundsSPerKm.length > 0 ? paceZoneBoundsSPerKm.length + 1 : 6
    const totals = sumDistributionZones(points, count, point => point.paceZoneSeconds)
    const ranges = Array.from({ length: count }, (_, index) =>
      distributionPaceZoneRange(context, paceZoneBoundsSPerKm, index),
    )
    return { metric: selectedMetric, ...totals, ranges }
  }

  const zoneMetricLabel = (selectedMetric: DistributionMetric): string =>
    selectedMetric === 'heart-rate' ? text('heart rate') : text(selectedMetric)

  const buildZoneView = (
    points: readonly ActivityDistributionPoint[],
    series: DistributionZoneSeries,
  ): HTMLElement => {
    const total = series.seconds.reduce((sum, value) => sum + value, 0)
    const view = el('section', 'tri-training-zone-view', undefined, {
      id: `tri-training-zones-${series.metric}`,
      'data-metric': series.metric,
      'aria-label': zoneMetricLabel(series.metric),
    })
    if (total <= 0) {
      view.appendChild(el('div', 'tri-ana-empty tri-training-zone-empty', text('no zone data')))
      return view
    }
    let majority = 0
    for (let index = 1; index < series.seconds.length; index++)
      if (series.seconds[index] > series.seconds[majority]) majority = index
    const maximum = Math.max(...series.seconds, 1)
    const majorityPct = (series.seconds[majority] / total) * 100
    const summary = el('div', 'tri-training-zone-summary', undefined, { 'aria-live': 'polite' })
    summary.append(
      el(
        'strong',
        'tri-training-zone-summary-value',
        `${Math.round(majorityPct)}% ${text('in zone')} ${majority + 1}`,
      ),
      el('span', 'tri-training-zone-summary-time', zoneClock(total)),
    )
    const grid = el(
      'div',
      `tri-training-zone-grid tri-training-zone-grid--${series.metric}`,
      undefined,
      {
        role: 'list',
        'aria-label': `${zoneMetricLabel(series.metric)} ${text('zone distribution')}`,
        style: `--tri-zone-count:${series.seconds.length}`,
      },
    )
    for (let index = series.seconds.length - 1; index >= 0; index--) {
      const value = series.seconds[index]
      const percentage = (value / total) * 100
      const range = series.ranges[index] || '—'
      const name = text(DISTRIBUTION_ZONE_NAMES[series.metric][index] ?? '')
      const details = `${zoneClock(value)} · ${percentage.toFixed(1)}% · ${range}`
      const row = el(
        'div',
        `tri-training-zone-row${index === majority ? ' tri-training-zone-row--majority' : ''}`,
        undefined,
        {
          role: 'listitem',
          tabindex: '0',
          'aria-label': `Z${index + 1} ${name} · ${details}`,
          'data-tip-h': `Z${index + 1} · ${name}`,
          'data-tip-d': details,
        },
      )
      const bar = el('span', 'tri-training-zone-bar', undefined, { 'aria-hidden': 'true' })
      const fill = el(
        'span',
        `tri-training-zone-fill tri-training-zone-fill--${series.metric} tri-training-zone-fill--${index + 1}`,
      )
      fill.style.setProperty('--tri-zone-share', `${(value / maximum) * 100}%`)
      bar.appendChild(fill)
      const visual = el('span', 'tri-training-zone-visual')
      visual.append(
        bar,
        el('span', 'tri-training-zone-pct', `${percentage.toFixed(1)}%`, { 'aria-hidden': 'true' }),
      )
      row.append(
        el('span', 'tri-training-zone-name', `Z${index + 1}`, { 'aria-hidden': 'true' }),
        visual,
        el('span', 'tri-training-zone-range', range, { 'aria-hidden': 'true' }),
      )
      grid.appendChild(row)
    }
    const coverage = el(
      'div',
      'tri-dist-cap',
      `${series.observedActivities}/${points.length} ${text('activities')} · ${zoneClock(total)} ${text('training time')} · ${context.formatter.longDate(startDate)}–${context.formatter.longDate(maximumDate)}`,
    )
    view.append(summary, grid)
    if (series.metric === 'pace' && tenKmRaceTimeS != null)
      view.appendChild(
        el(
          'div',
          'tri-dist-cap tri-training-zone-source',
          `${text('based on 10 km race time')} ${clock(tenKmRaceTimeS)}`,
        ),
      )
    view.appendChild(coverage)
    return view
  }

  const applyZoneMetric = (): void => {
    const tabs = Array.from(
      metricControls.querySelectorAll<HTMLButtonElement>('.tri-zone-metric-tab'),
    )
    const views = Array.from(zonePanel.querySelectorAll<HTMLElement>('.tri-training-zone-view'))
    for (const tab of tabs) {
      const selected = tab.dataset.metric === metric
      tab.setAttribute('aria-selected', String(selected))
      tab.tabIndex = selected ? 0 : -1
    }
    for (const view of views) {
      const selected = view.dataset.metric === metric
      view.dataset.active = String(selected)
      view.hidden = !selected
      view.setAttribute('aria-hidden', String(!selected))
      view.inert = !selected
    }
  }

  const renderZones = (points: ActivityDistributionPoint[]): void => {
    metricControls.replaceChildren()
    const stage = el('div', 'tri-training-zone-stage')
    const options = distributionMetrics(sport)
    for (const option of options) {
      const label = zoneMetricLabel(option)
      const tab = el('button', 'tri-map-tab tri-zone-metric-tab', undefined, {
        type: 'button',
        role: 'tab',
        'aria-controls': `tri-training-zones-${option}`,
        'aria-label': label,
        title: label,
        'data-metric': option,
      }) as HTMLButtonElement
      tab.appendChild(buildZoneMetricIcon(option))
      metricControls.appendChild(tab)
      stage.appendChild(buildZoneView(points, zoneSeries(points, option)))
    }
    zonePanel.replaceChildren(stage)
    applyZoneMetric()
  }

  interface TelemetryMetric {
    key: 'power' | 'cadence' | 'skin' | 'hsi'
    label: string
    value: (point: ActivityDistributionPoint) => number | null
    observedSeconds: (point: ActivityDistributionPoint) => number
    text: (value: number, point: ActivityDistributionPoint) => string
  }
  const metrics: TelemetryMetric[] = [
    {
      key: 'power',
      label: 'average power',
      value: point => point.averagePowerWatts,
      observedSeconds: point => point.movingTimeS,
      text: value => `${Math.round(value)} W`,
    },
    {
      key: 'cadence',
      label: 'cadence',
      value: point => point.cadence,
      observedSeconds: point => point.movingTimeS,
      text: (value, point) => `${Math.round(value)} ${point.cadenceUnit}`,
    },
    {
      key: 'skin',
      label: 'skin temperature',
      value: point => point.skinTemperatureC,
      observedSeconds: point => point.skinObservedSeconds,
      text: value => formatThermalTemperature(context.presentation, value),
    },
    {
      key: 'hsi',
      label: 'heat strain index',
      value: point => point.heatStrainIndex,
      observedSeconds: point => point.heatStrainObservedSeconds,
      text: value => `HSI ${value.toFixed(1)}`,
    },
  ]
  let mounted = false
  let telemetryCleanup: (() => void) | null = null
  let mountTelemetry: (() => () => void) | null = null

  const renderTelemetry = (points: ActivityDistributionPoint[]): void => {
    telemetryCleanup?.()
    telemetryCleanup = null
    mountTelemetry = null
    telemetryPanel.replaceChildren()
    const title = anaTitle(context.formatter, 'telemetry', 'activitytelemetry')
    telemetryPanel.appendChild(title)
    const available = points.filter(point => metrics.some(metric => metric.value(point) != null))
    if (available.length === 0) {
      telemetryPanel.appendChild(el('div', 'tri-ana-empty', text('no telemetry data')))
      return
    }

    const rangeStartMs = Date.parse(`${startDate}T00:00:00Z`)
    const rangeEndMs = Date.parse(`${maximumDate}T23:59:59Z`)
    const rangeSpanMs = Math.max(1, rangeEndMs - rangeStartMs)
    const pointX = (point: ActivityDistributionPoint): number =>
      clampN(((Date.parse(point.startedAt) - rangeStartMs) / rangeSpanMs) * 100, 0, 100)
    const plots = el('div', 'tri-dist-plots', undefined, {
      role: 'slider',
      tabindex: '0',
      'aria-label': text('activity telemetry scrubber'),
      'aria-orientation': 'horizontal',
      'aria-valuemin': '1',
      'aria-valuemax': String(available.length),
      'aria-valuenow': String(available.length),
    })
    const readout = el(
      'div',
      'tri-chart-readout tri-dist-readout',
      `${available.length}/${points.length} ${text('activities with telemetry')}`,
    )
    const cursorLines: SVGElement[] = []
    const graphs: SVGElement[] = []

    for (const metric of metrics) {
      const values = points
        .map(point => metric.value(point))
        .filter((value): value is number => value != null && Number.isFinite(value))
      if (values.length === 0) continue
      const rawMin = Math.min(...values)
      const rawMax = Math.max(...values)
      const padding = Math.max((rawMax - rawMin) * 0.12, metric.key === 'hsi' ? 0.25 : 1)
      const domainMin = rawMin - padding
      const domainMax = rawMax + padding
      const domainText = (value: number): string =>
        metric.key === 'skin'
          ? formatThermalTemperature(context.presentation, value)
          : value.toFixed(metric.key === 'hsi' ? 1 : 0)
      const y = (value: number): number => 30 - ((value - domainMin) / (domainMax - domainMin)) * 26
      const row = el('div', `tri-dist-metric tri-dist-metric--${metric.key}`)
      const meta = el('div', 'tri-dist-metric-meta')
      const summaryPoint = points.find(point => metric.value(point) != null)
      const summaryValue = telemetryWeightedAverage(
        points.map(point => ({
          value: metric.value(point),
          observedSeconds: metric.observedSeconds(point),
        })),
      )
      const trend = telemetryTrend(points.map(point => metric.value(point)))
      const status = el('div', 'tri-dist-metric-status')
      status.appendChild(
        summaryPoint && summaryValue != null
          ? el('span', 'tri-dist-metric-latest', metric.text(summaryValue, summaryPoint))
          : el('span', 'tri-dist-metric-latest', '—'),
      )
      if (trend)
        status.appendChild(
          buildTrendGlyph(
            trend,
            text(
              trend === 'up'
                ? 'higher than previous activity'
                : trend === 'down'
                  ? 'lower than previous activity'
                  : 'unchanged from previous activity',
            ),
            'tri-dist-metric-trend',
          ),
        )
      meta.append(el('span', 'tri-dist-metric-name', text(metric.label)), status)
      const graph = svg('svg', {
        class: 'tri-dist-metric-svg',
        viewBox: '0 0 100 34',
        preserveAspectRatio: 'none',
        role: 'img',
        'aria-label': `${text(metric.label)} · ${domainText(rawMin)}–${domainText(rawMax)}`,
      })
      graphs.push(graph)
      graph.append(
        svg('line', { class: 'tri-dist-grid', x1: 0, y1: 4, x2: 100, y2: 4 }),
        svg('line', { class: 'tri-dist-grid', x1: 0, y1: 17, x2: 100, y2: 17 }),
        svg('line', { class: 'tri-dist-grid', x1: 0, y1: 30, x2: 100, y2: 30 }),
        svg('line', { class: 'tri-dist-axis', x1: 0, y1: 4, x2: 0, y2: 30 }),
        svg('line', { class: 'tri-dist-axis', x1: 0, y1: 30, x2: 100, y2: 30 }),
      )
      for (const bridge of missingBridges(
        points,
        point => metric.value(point),
        index => pointX(points[index]),
        y,
      ))
        graph.appendChild(
          svg('path', {
            class: `tri-dist-line tri-dist-line--${metric.key} tri-dist-line--missing`,
            d: polyD(bridge),
          }),
        )
      for (const segment of segRuns(
        points,
        point => metric.value(point),
        index => pointX(points[index]),
        y,
      ))
        graph.appendChild(
          svg('path', { class: `tri-dist-line tri-dist-line--${metric.key}`, d: polyD(segment) }),
        )
      for (const point of points) {
        const value = metric.value(point)
        if (value == null) continue
        const pointY = y(value)
        graph.appendChild(
          svg('line', {
            class: `tri-dist-point tri-dist-point--${metric.key}${metric.key === 'power' && point.powerSource === 'estimated' ? ' tri-dist-point--estimated' : ''}`,
            x1: pointX(point),
            x2: pointX(point),
            y1: pointY - 0.85,
            y2: pointY + 0.85,
          }),
        )
      }
      const cursor = svg('line', { class: 'tri-ana-cursor', x1: 0, y1: 3, x2: 0, y2: 31 })
      graph.appendChild(cursor)
      cursorLines.push(cursor)
      const domain = el('div', 'tri-dist-domain')
      domain.append(
        el('span', undefined, domainText(rawMax)),
        el('span', undefined, domainText(rawMin)),
      )
      row.append(domain, graph, meta)
      plots.appendChild(row)
    }

    const axis = el('div', 'tri-dist-time-axis', undefined, { 'aria-hidden': 'true' })
    const timeScale = el('div', 'tri-dist-time-scale')
    timeScale.append(
      el('span', undefined, context.formatter.shortDate(startDate)),
      el('span', undefined, context.formatter.shortDate(maximumDate)),
    )
    axis.appendChild(timeScale)
    const resetReadout = (): void => {
      readout.textContent = `${available.length}/${points.length} ${text('activities with telemetry')} · ${context.formatter.longDate(startDate)}–${context.formatter.longDate(maximumDate)}`
      telemetryPanel.classList.remove('tri-chart--hover')
    }
    let activePointIndex = available.length - 1
    const graphAnchorX = (fraction: number): number => {
      const plotsRect = plots.getBoundingClientRect()
      const graphRect = graphs[0]?.getBoundingClientRect()
      return graphRect
        ? graphRect.left - plotsRect.left + clampN(fraction, 0, 1) * graphRect.width
        : clampN(fraction, 0, 1) * plots.clientWidth
    }
    const positionReadout = (anchorX: number, anchorY: number): void => {
      const plotWidth = plots.clientWidth
      const plotHeight = plots.clientHeight
      const readoutWidth = readout.offsetWidth
      const readoutHeight = readout.offsetHeight
      const gap = 10
      const inset = 4
      const rightSpace = plotWidth - anchorX
      const leftSpace = anchorX
      const opensRight = rightSpace >= readoutWidth + gap || rightSpace >= leftSpace
      const left = clampN(
        opensRight ? anchorX + gap : anchorX - readoutWidth - gap,
        inset,
        Math.max(inset, plotWidth - readoutWidth - inset),
      )
      const top = clampN(
        anchorY,
        readoutHeight / 2 + inset,
        Math.max(readoutHeight / 2 + inset, plotHeight - readoutHeight / 2 - inset),
      )
      readout.dataset.side = left < anchorX ? 'left' : 'right'
      readout.style.setProperty('--tri-dist-readout-x', `${left}px`)
      readout.style.setProperty('--tri-dist-readout-y', `${top}px`)
    }
    const showPoint = (
      point: ActivityDistributionPoint,
      anchorX = graphAnchorX(pointX(point) / 100),
      anchorY = plots.clientHeight / 2,
    ): void => {
      const date = context.formatter.shortDate(point.date)
      const table = el('table', 'tri-dist-readout-table', undefined, {
        'aria-label': text('activity telemetry'),
      })
      const body = el('tbody')
      const ariaValues = [date]
      for (const metric of metrics) {
        const value = metric.value(point)
        if (value == null) continue
        const pointIndex = points.indexOf(point)
        const previous = points
          .slice(0, Math.max(0, pointIndex))
          .reverse()
          .find(candidate => metric.value(candidate) != null)
        const previousValue = previous ? metric.value(previous) : null
        const delta =
          previousValue == null
            ? ''
            : ` (${value >= previousValue ? '+' : ''}${(value - previousValue).toFixed(
                metric.key === 'skin' || metric.key === 'hsi' ? 1 : 0,
              )})`
        const formattedValue = metric.text(value, point)
        const row = el('tr')
        row.append(
          el('th', 'tri-dist-readout-metric', text(metric.label), { scope: 'row' }),
          el('td', 'tri-dist-readout-value', formattedValue),
          el('td', 'tri-dist-readout-delta', delta.trim()),
        )
        body.appendChild(row)
        ariaValues.push(`${text(metric.label)} ${formattedValue}${delta}`)
      }
      table.appendChild(body)
      readout.replaceChildren(el('span', 'tri-dist-readout-head', date), table)
      const availableIndex = available.indexOf(point)
      if (availableIndex >= 0) {
        activePointIndex = availableIndex
        plots.setAttribute('aria-valuenow', String(availableIndex + 1))
        plots.setAttribute('aria-valuetext', ariaValues.join(' · '))
      }
      const x = pointX(point).toFixed(2)
      for (const cursor of cursorLines) {
        cursor.setAttribute('x1', x)
        cursor.setAttribute('x2', x)
      }
      telemetryPanel.classList.add('tri-chart--hover')
      positionReadout(anchorX, anchorY)
    }
    const showPointAt = (index: number): void => {
      showPoint(available[clampN(index, 0, available.length - 1)])
    }
    const onMove = (event: PointerEvent): void => {
      const plotsRect = plots.getBoundingClientRect()
      const graphRect = graphs[0]?.getBoundingClientRect() ?? plotsRect
      const fraction = clampN((event.clientX - graphRect.left) / graphRect.width, 0, 1)
      const targetMs = rangeStartMs + fraction * rangeSpanMs
      let nearest = available[0]
      for (const point of available)
        if (
          Math.abs(Date.parse(point.startedAt) - targetMs) <
          Math.abs(Date.parse(nearest.startedAt) - targetMs)
        )
          nearest = point
      showPoint(
        nearest,
        clampN(event.clientX - plotsRect.left, 0, plotsRect.width),
        event.clientY - plotsRect.top,
      )
    }
    const onLeave = (): void => {
      if (document.activeElement !== plots) resetReadout()
    }
    const onFocus = (): void => showPointAt(activePointIndex)
    const onKeydown = (event: KeyboardEvent): void => {
      const nextIndex =
        event.key === 'ArrowLeft'
          ? activePointIndex - 1
          : event.key === 'ArrowRight'
            ? activePointIndex + 1
            : event.key === 'Home'
              ? 0
              : event.key === 'End'
                ? available.length - 1
                : null
      if (nextIndex == null) return
      event.preventDefault()
      showPointAt(nextIndex)
    }
    mountTelemetry = () => {
      plots.addEventListener('pointermove', onMove)
      plots.addEventListener('pointerleave', onLeave)
      plots.addEventListener('focus', onFocus)
      plots.addEventListener('blur', resetReadout)
      plots.addEventListener('keydown', onKeydown)
      return () => {
        plots.removeEventListener('pointermove', onMove)
        plots.removeEventListener('pointerleave', onLeave)
        plots.removeEventListener('focus', onFocus)
        plots.removeEventListener('blur', resetReadout)
        plots.removeEventListener('keydown', onKeydown)
      }
    }
    plots.appendChild(readout)
    telemetryPanel.append(plots, axis)
    showPointAt(activePointIndex)
    resetReadout()
    if (mounted) telemetryCleanup = mountTelemetry()
  }

  function render(): void {
    block.dataset.sport = sport
    block.dataset.range = range
    block.dataset.rangeStart = startDate
    for (const [option, button] of sportButtons)
      button.setAttribute('aria-pressed', String(option === sport))
    for (const [option, button] of rangeButtons) {
      const selected = option === range
      button.classList.toggle('tri-dist-range--on', selected)
      button.setAttribute('aria-pressed', String(selected))
    }
    const dateText = startPicker.trigger.querySelector<HTMLElement>('.tri-pred-date-text')
    if (dateText) dateText.textContent = context.formatter.shortDate(startDate)
    startPicker.trigger.dataset.value = startDate
    if (startPicker.panel.matches(':popover-open')) startPicker.render()
    const points = selectedActivities()
    renderZones(points)
    renderTelemetry(points)
  }

  const restoreSelection = (): void => {
    try {
      const stored: unknown = JSON.parse(
        localStorage.getItem(TRI_DISTRIBUTION_SELECTION_KEY) ?? 'null',
      )
      if (!isRecord(stored)) return
      const storedSport = stored.sport
      const restoredSport =
        storedSport === 'swim' || storedSport === 'bike' || storedSport === 'run'
          ? storedSport
          : model.sport
      const storedMetric = stored.metric
      const restoredMetric = distributionMetricForSport(
        restoredSport,
        storedMetric === 'heart-rate' || storedMetric === 'power' || storedMetric === 'pace'
          ? storedMetric
          : undefined,
      )
      const storedRange = stored.range
      const restoredRange =
        storedRange === '7' ||
        storedRange === '14' ||
        storedRange === '30' ||
        storedRange === '60' ||
        storedRange === 'custom'
          ? storedRange
          : model.range
      const restoredStartDate =
        typeof stored.startDate === 'string' &&
        parsePredDate(stored.startDate) &&
        stored.startDate >= minimumDate &&
        stored.startDate <= maximumDate
          ? stored.startDate
          : model.startDate
      applyModel(
        updateDistributions(
          model,
          {
            type: 'restore',
            model: {
              sport: restoredSport,
              metric: restoredMetric,
              range: restoredRange,
              startDate: restoredStartDate,
            },
          },
          bounds,
        ),
      )
    } catch {}
  }
  const onSportClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const button = target.closest<HTMLButtonElement>('.tri-dist-sport[data-sport]')
    if (!button || !sportControls.contains(button)) return
    const selected = button.dataset.sport
    if (selected !== 'swim' && selected !== 'bike' && selected !== 'run') return
    applyModel(updateDistributions(model, { type: 'select-sport', sport: selected }, bounds))
    persist()
    render()
  }
  const onRangeClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const button = target.closest<HTMLButtonElement>('.tri-dist-range[data-range]')
    if (!button || !rangeControls.contains(button)) return
    const selected = button.dataset.range
    if (
      selected !== '7' &&
      selected !== '14' &&
      selected !== '30' &&
      selected !== '60' &&
      selected !== 'custom'
    )
      return
    applyModel(updateDistributions(model, { type: 'select-range', range: selected }, bounds))
    persist()
    render()
    if (range === 'custom') queueMicrotask(() => startPicker.trigger.click())
  }

  const selectZoneMetric = (selected: string | undefined): void => {
    if (selected !== 'heart-rate' && selected !== 'power' && selected !== 'pace') return
    if (!distributionMetrics(sport).includes(selected)) return
    applyModel(updateDistributions(model, { type: 'select-metric', metric: selected }, bounds))
    persist()
    applyZoneMetric()
  }
  const onZoneMetricClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const tab = target.closest<HTMLButtonElement>('.tri-zone-metric-tab[data-metric]')
    if (!tab || !metricControls.contains(tab)) return
    selectZoneMetric(tab.dataset.metric)
  }
  const onZoneMetricKeydown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    const target = event.target
    if (!(target instanceof Element) || !target.closest('.tri-zone-metric-tab')) return
    const tabs = Array.from(
      metricControls.querySelectorAll<HTMLButtonElement>('.tri-zone-metric-tab'),
    )
    const active = tabs.findIndex(tab => tab.dataset.metric === metric)
    let next = -1
    if (event.key === 'ArrowLeft') next = (active - 1 + tabs.length) % tabs.length
    else if (event.key === 'ArrowRight') next = (active + 1) % tabs.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = tabs.length - 1
    else
      next = nextMapMetricShortcutIndex(
        tabs.map(tab => tab.dataset.shortcut),
        active,
        event.key,
      )
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    selectZoneMetric(tabs[next]?.dataset.metric)
    tabs[next]?.focus()
  }

  let zoneTip: HTMLElement | null = null
  const hideZoneTip = (): void => zoneTip?.classList.remove('tri-gloss--on')
  const showZoneTip = (row: HTMLElement, clientX: number, clientY: number): void => {
    if (!zoneTip) return
    zoneTip.replaceChildren(
      el('span', 'tri-gloss-h', row.dataset.tipH ?? ''),
      el('span', 'tri-gloss-def', row.dataset.tipD ?? ''),
    )
    zoneTip.classList.add('tri-gloss--on')
    const rect = zoneTip.getBoundingClientRect()
    const gap = 12
    const inset = 8
    const left = clampN(
      clientX + gap + rect.width <= window.innerWidth - inset
        ? clientX + gap
        : clientX - gap - rect.width,
      inset,
      Math.max(inset, window.innerWidth - rect.width - inset),
    )
    const top = clampN(
      clientY - rect.height / 2,
      inset,
      Math.max(inset, window.innerHeight - rect.height - inset),
    )
    zoneTip.style.left = `${left}px`
    zoneTip.style.top = `${top}px`
  }
  const onZonePointerMove = (event: PointerEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const row = target.closest<HTMLElement>('.tri-training-zone-row')
    if (!row || !zonePanel.contains(row)) {
      hideZoneTip()
      return
    }
    showZoneTip(row, event.clientX, event.clientY)
  }
  const onZoneFocusIn = (event: FocusEvent): void => {
    const target = event.target
    if (!(target instanceof HTMLElement) || !target.matches('.tri-training-zone-row')) return
    const rect = target.getBoundingClientRect()
    showZoneTip(target, rect.left + rect.width / 2, rect.top + rect.height / 2)
  }
  const onZoneFocusOut = (): void => {
    window.queueMicrotask(() => {
      if (!zonePanel.contains(document.activeElement)) hideZoneTip()
    })
  }

  render()
  return {
    element: block,
    mount: () => {
      mounted = true
      restoreSelection()
      sportControls.addEventListener('click', onSportClick)
      rangeControls.addEventListener('click', onRangeClick)
      metricControls.addEventListener('click', onZoneMetricClick)
      metricControls.addEventListener('keydown', onZoneMetricKeydown)
      zonePanel.addEventListener('pointermove', onZonePointerMove)
      zonePanel.addEventListener('pointerleave', hideZoneTip)
      zonePanel.addEventListener('focusin', onZoneFocusIn)
      zonePanel.addEventListener('focusout', onZoneFocusOut)
      zoneTip = el('div', 'tri-gloss tri-zone-distribution-tip', undefined, { role: 'tooltip' })
      document.body.appendChild(zoneTip)
      const datePickerCleanup = startPicker.mount()
      render()
      return () => {
        mounted = false
        sportControls.removeEventListener('click', onSportClick)
        rangeControls.removeEventListener('click', onRangeClick)
        metricControls.removeEventListener('click', onZoneMetricClick)
        metricControls.removeEventListener('keydown', onZoneMetricKeydown)
        zonePanel.removeEventListener('pointermove', onZonePointerMove)
        zonePanel.removeEventListener('pointerleave', hideZoneTip)
        zonePanel.removeEventListener('focusin', onZoneFocusIn)
        zonePanel.removeEventListener('focusout', onZoneFocusOut)
        zoneTip?.remove()
        zoneTip = null
        datePickerCleanup()
        telemetryCleanup?.()
        telemetryCleanup = null
      }
    },
  }
}
