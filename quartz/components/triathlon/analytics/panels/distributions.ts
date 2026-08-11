import type { Analytics } from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import { formatThermalTemperature } from '../../../../util/triathlon-card'
import { zoneClock } from '../../../../util/triathlon-card'
import { isRecord } from '../../../../util/type-guards'
import { buildIcon } from '../../activity/primitives'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { buildDatePicker } from '../../tools/date-picker'
import { parsePredDate } from '../../tools/date-picker'
import { anaTitle } from '../shared'
import { clampN } from '../shared'
import { polyD } from '../shared'
import { missingBridges } from './body'
import { segRuns } from './body'
import {
  DISTRIBUTION_RANGES,
  initialDistributionModel,
  updateDistributions,
  type DistributionRange,
  type DistributionSport,
} from './distributions-model'

export type ActivityDistributionPoint = Analytics['distributions']['activities'][number]

export const TRI_DISTRIBUTION_SELECTION_KEY = 'tri-distribution-selection'

export const DISTRIBUTION_ZONE_NAMES = ['recovery', 'endurance', 'tempo', 'threshold', 'anaerobic']

export const distributionZoneRange = (bounds: readonly number[], index: number): string => {
  if (bounds.length === 0) return ''
  if (index === 0) return `≤${bounds[0]} bpm`
  if (index >= bounds.length) return `${bounds[bounds.length - 1] + 1}+ bpm`
  return `${bounds[index - 1] + 1}–${bounds[index]} bpm`
}

export const buildDistributions = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount?: () => () => void } => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-training-distribution')
  const { activities, heartRateZoneBounds } = data.distributions
  if (activities.length === 0) {
    block.append(anaTitle(context.formatter, 'training distributions', 'hrzones'))
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
  let { sport, range, startDate } = model
  const applyModel = (next: typeof model): void => {
    model = next
    sport = model.sport
    range = model.range
    startDate = model.startDate
  }

  const persist = (): void => {
    try {
      localStorage.setItem(
        TRI_DISTRIBUTION_SELECTION_KEY,
        JSON.stringify({ sport, range, startDate }),
      )
    } catch {}
  }

  const head = el('div', 'tri-dist-head')
  head.appendChild(anaTitle(context.formatter, 'heart rate zone distribution', 'hrzones'))
  const controls = el('div', 'tri-dist-controls')
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
  controls.append(sportControls, rangeControls, startPicker.wrap)
  head.appendChild(controls)
  block.appendChild(head)

  const zonePanel = el('section', 'tri-hr-distribution')
  const telemetryPanel = el('section', 'tri-activity-telemetry')
  block.append(zonePanel, telemetryPanel)

  const selectedActivities = (): ActivityDistributionPoint[] =>
    activities.filter(
      point => point.sport === sport && point.date >= startDate && point.date <= maximumDate,
    )

  const renderZones = (points: ActivityDistributionPoint[]): void => {
    const zoneCount = heartRateZoneBounds.length > 0 ? heartRateZoneBounds.length + 1 : 0
    const seconds = Array.from({ length: zoneCount }, () => 0)
    let observedActivities = 0
    for (const point of points) {
      if (!point.heartRateZoneSeconds?.some(value => value > 0)) continue
      observedActivities += 1
      point.heartRateZoneSeconds.forEach((value, index) => {
        if (index < seconds.length && Number.isFinite(value) && value > 0) seconds[index] += value
      })
    }
    const total = seconds.reduce((sum, value) => sum + value, 0)
    zonePanel.replaceChildren()
    if (total <= 0) {
      zonePanel.appendChild(el('div', 'tri-ana-empty', text('no heart rate zone data')))
      return
    }
    let majority = 0
    for (let index = 1; index < seconds.length; index++)
      if (seconds[index] > seconds[majority]) majority = index
    const majorityPct = (seconds[majority] / total) * 100
    const summary = el(
      'div',
      'tri-hr-majority',
      `${text('majority zone')} · Z${majority + 1} ${text(DISTRIBUTION_ZONE_NAMES[majority] ?? '')} · ${majorityPct.toFixed(1)}%`,
      { 'aria-live': 'polite' },
    )
    const stack = el('div', 'tri-hr-stack', undefined, {
      role: 'img',
      'aria-label': `${text('heart rate zone distribution')} · ${seconds
        .map((value, index) => `Z${index + 1} ${((value / total) * 100).toFixed(1)}%`)
        .join(' · ')}`,
    })
    seconds.forEach((value, index) => {
      if (value <= 0) return
      const segment = el('span', `tri-hr-segment tri-hr-segment--${index + 1}`)
      segment.style.width = `${(value / total) * 100}%`
      segment.title = `Z${index + 1} · ${zoneClock(value)} · ${((value / total) * 100).toFixed(1)}%`
      stack.appendChild(segment)
    })
    const legend = el('div', 'tri-hr-legend')
    seconds.forEach((value, index) => {
      const row = el('div', `tri-hr-zone${index === majority ? ' tri-hr-zone--majority' : ''}`)
      row.append(
        el('span', `tri-hr-swatch tri-hr-swatch--${index + 1}`),
        el(
          'span',
          'tri-hr-zone-name',
          `Z${index + 1} ${text(DISTRIBUTION_ZONE_NAMES[index] ?? '')}`,
        ),
        el('span', 'tri-hr-zone-range', distributionZoneRange(heartRateZoneBounds, index)),
        el('span', 'tri-hr-zone-time', zoneClock(value)),
        el('span', 'tri-hr-zone-pct', `${((value / total) * 100).toFixed(1)}%`),
      )
      legend.appendChild(row)
    })
    const coverage = el(
      'div',
      'tri-dist-cap',
      `${observedActivities}/${points.length} ${text('activities')} · ${zoneClock(total)} ${text('training time')} · ${context.formatter.longDate(startDate)}–${context.formatter.longDate(maximumDate)}`,
    )
    zonePanel.append(summary, stack, legend, coverage)
  }

  interface DistributionMetric {
    key: 'power' | 'cadence' | 'skin' | 'hsi'
    label: string
    value: (point: ActivityDistributionPoint) => number | null
    text: (value: number, point: ActivityDistributionPoint) => string
  }
  const metrics: DistributionMetric[] = [
    {
      key: 'power',
      label: 'average power',
      value: point => point.averagePowerWatts,
      text: value => `${Math.round(value)} W`,
    },
    {
      key: 'cadence',
      label: 'cadence',
      value: point => point.cadence,
      text: (value, point) => `${Math.round(value)} ${point.cadenceUnit}`,
    },
    {
      key: 'skin',
      label: 'skin temperature',
      value: point => point.skinTemperatureC,
      text: value => formatThermalTemperature(context.presentation, value),
    },
    {
      key: 'hsi',
      label: 'heat strain index',
      value: point => point.heatStrainIndex,
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
    const title = anaTitle(context.formatter, 'activity telemetry over time', 'activitytelemetry')
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
      const latest = [...points].reverse().find(point => metric.value(point) != null)
      const latestValue = latest ? metric.value(latest) : null
      meta.append(
        el('span', 'tri-dist-metric-name', text(metric.label)),
        latest && latestValue != null
          ? el('span', 'tri-dist-metric-latest', metric.text(latestValue, latest))
          : el('span', 'tri-dist-metric-latest', '—'),
      )
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
      row.append(meta, graph, domain)
      plots.appendChild(row)
    }

    const axis = el('div', 'tri-dist-time-axis')
    axis.append(
      el('span', undefined, context.formatter.shortDate(startDate)),
      el('span', undefined, context.formatter.shortDate(maximumDate)),
    )
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
            model: { sport: restoredSport, range: restoredRange, startDate: restoredStartDate },
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

  render()
  return {
    element: block,
    mount: () => {
      mounted = true
      restoreSelection()
      sportControls.addEventListener('click', onSportClick)
      rangeControls.addEventListener('click', onRangeClick)
      const datePickerCleanup = startPicker.mount()
      render()
      return () => {
        mounted = false
        sportControls.removeEventListener('click', onSportClick)
        rangeControls.removeEventListener('click', onRangeClick)
        datePickerCleanup()
        telemetryCleanup?.()
        telemetryCleanup = null
      }
    },
  }
}
