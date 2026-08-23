import type {
  Analytics,
  PowerToWeightDurationS,
  PowerToWeightEffort,
  PowerToWeightReference,
  PowerToWeightTrendPoint,
} from '../../../../plugins/stores/analytics'
import type { PowerCurvePoint } from '../../../../plugins/stores/strava'
import type { TriathlonContext } from '../../runtime/context'
import { POWER_TO_WEIGHT_DURATIONS } from '../../../../plugins/stores/analytics'
import { axisFrame, axisNumber, niceStep, type AxisXTick } from '../../../../util/triathlon-card'
import { syncPowerCurveActivityLink } from '../../activity/power-links'
import { createDomFactory, el, svg } from '../../runtime/dom'
import { anaTitle, clampN, markGlossDefinition, monthTicks } from '../shared'

const W = 100
const H = 34

export const powerToWeightDurationLabel = (durationS: PowerToWeightDurationS): string =>
  durationS === 5 ? '5s' : durationS === 60 ? '1m' : durationS === 300 ? '5m' : '20m'

export const deconflictPowerToWeightMonthTicks = (ticks: AxisXTick[]): AxisXTick[] => {
  if (ticks.length < 2 || ticks[1].pct - ticks[0].pct >= 5) return ticks
  return ticks
    .slice(1)
    .map((tick, index) => (index === 0 ? { ...tick, cls: 'tri-cax-xt--first' } : tick))
}

const effortAt = (
  point: PowerToWeightTrendPoint,
  durationS: PowerToWeightDurationS,
): PowerToWeightEffort | null => point.efforts[durationS]

const effortPowerPoint = (effort: PowerToWeightEffort | null): PowerCurvePoint | null =>
  effort == null
    ? null
    : {
        s: effort.durationS,
        w: effort.watts,
        activityId: effort.activityId,
        activityDate: effort.activityDate,
      }

const durationFrom = (value: string | undefined): PowerToWeightDurationS | null =>
  POWER_TO_WEIGHT_DURATIONS.find(durationS => String(durationS) === value) ?? null

const seriesPath = (
  points: readonly PowerToWeightTrendPoint[],
  durationS: PowerToWeightDurationS,
  y: (value: number) => number,
): string => {
  let drawing = false
  const commands: string[] = []
  for (let index = 0; index < points.length; index++) {
    const effort = effortAt(points[index], durationS)
    if (effort == null) {
      drawing = false
      continue
    }
    const x = points.length === 1 ? W / 2 : (index / (points.length - 1)) * W
    commands.push(`${drawing ? 'L' : 'M'} ${x.toFixed(2)} ${y(effort.wattsPerKg).toFixed(2)}`)
    drawing = true
  }
  return commands.join(' ')
}

const observedMax = (
  points: readonly PowerToWeightTrendPoint[],
  active: ReadonlySet<PowerToWeightDurationS>,
  references: readonly PowerToWeightReference[],
): number => {
  let maximum = 1
  for (const point of points)
    for (const durationS of active) {
      const effort = effortAt(point, durationS)
      if (effort != null) maximum = Math.max(maximum, effort.wattsPerKg)
    }
  for (const reference of references)
    if (active.has(reference.durationS)) maximum = Math.max(maximum, reference.average)
  return maximum
}

const axisScale = (
  points: readonly PowerToWeightTrendPoint[],
  active: ReadonlySet<PowerToWeightDurationS>,
  references: readonly PowerToWeightReference[],
): { domainMax: number; step: number; values: number[] } => {
  const maximum = observedMax(points, active, references)
  const step = niceStep(maximum, 4)
  const domainMax = Math.ceil(maximum / step) * step
  const values = Array.from(
    { length: Math.round(domainMax / step) + 1 },
    (_, index) => index * step,
  )
  return { domainMax, step, values }
}

const buildReferenceTable = (data: Analytics, context: TriathlonContext): HTMLElement | null => {
  const trend = data.powerCurve.powerToWeight
  if (trend.references.length === 0 || trend.ageGroup == null) return null
  const wrap = el('div', 'tri-power-weight-reference')
  const table = el('table', 'tri-power-weight-reference-table')
  const caption = el('caption')
  caption.append(
    el('span', undefined, `circa Zwift (${trend.ageGroup.replace('–', '-')}) — `),
    el('a', undefined, 'video', { href: trend.source.url, rel: 'noreferrer' }),
  )
  const head = el('thead')
  const header = el('tr')
  for (const label of [
    context.formatter.text('duration'),
    'P10',
    context.formatter.text('average'),
    'P90',
  ])
    header.appendChild(el('th', undefined, label, { scope: 'col' }))
  head.appendChild(header)
  const body = el('tbody')
  for (const reference of trend.references) {
    const row = el('tr')
    row.append(
      el('th', undefined, powerToWeightDurationLabel(reference.durationS), { scope: 'row' }),
      el('td', undefined, reference.p10.toFixed(2)),
      el('td', undefined, reference.average.toFixed(2)),
      el('td', undefined, reference.p90.toFixed(2)),
    )
    body.appendChild(row)
  }
  table.append(caption, head, body)
  wrap.appendChild(table)
  return wrap
}

interface PowerToWeightView {
  element: HTMLElement
  mount?: () => () => void
}

export const buildPowerToWeightTrend = (
  data: Analytics,
  context: TriathlonContext,
): PowerToWeightView => {
  const trend = data.powerCurve.powerToWeight
  const block = el('section', 'tri-power-weight')
  const head = el('div', 'tri-power-weight-head')
  const heading = el('div', 'tri-power-weight-heading')
  const sourceRideIds = new Set<number>()
  for (const point of trend.points)
    for (const durationS of POWER_TO_WEIGHT_DURATIONS) {
      const effort = effortAt(point, durationS)
      if (effort != null) sourceRideIds.add(effort.activityId)
    }
  heading.appendChild(
    markGlossDefinition(
      anaTitle(context.formatter, 'power-to-weight trend'),
      `${context.formatter.text('your ride history')} · ${trend.windowDays}${context.formatter.text('d')} ${context.formatter.text('rolling best for each duration')}. ${context.formatter.text('sample')}: ${trend.points.length} ${context.formatter.text('calendar days')} · ${sourceRideIds.size} ${context.formatter.text('source rides')}.`,
    ),
  )
  const available = new Set<PowerToWeightDurationS>()
  for (const durationS of POWER_TO_WEIGHT_DURATIONS)
    if (trend.points.some(point => effortAt(point, durationS) != null)) available.add(durationS)
  const controls = el('div', 'tri-power-weight-controls', undefined, {
    role: 'group',
    'aria-label': context.formatter.text('power-to-weight durations'),
  })
  for (const durationS of POWER_TO_WEIGHT_DURATIONS) {
    const attrs: Record<string, string> = {
      type: 'button',
      'data-power-weight-duration': String(durationS),
      'aria-pressed': String(available.has(durationS)),
    }
    if (!available.has(durationS)) attrs.disabled = ''
    const button = el('button', 'tri-power-weight-toggle', undefined, attrs)
    button.append(
      el('span', 'tri-power-weight-marker', undefined, {
        'aria-hidden': 'true',
        'data-power-weight-duration': String(durationS),
      }),
      el('span', undefined, powerToWeightDurationLabel(durationS)),
    )
    controls.appendChild(button)
  }
  head.append(heading, controls)
  block.appendChild(head)

  if (trend.points.length < 2 || available.size === 0) {
    block.appendChild(
      el('div', 'tri-ana-empty', context.formatter.text('no historical power-to-weight data')),
    )
    const reference = buildReferenceTable(data, context)
    if (reference) block.appendChild(reference)
    return { element: block }
  }

  const active = new Set(available)
  let scale = axisScale(trend.points, active, trend.references)
  const y = (value: number): number => H - (value / scale.domainMax) * (H - 1)
  const initialIndex = trend.points.length - 1
  const graph = svg('svg', {
    class: 'tri-power-weight-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': context.formatter.text('power-to-weight trend'),
    'aria-orientation': 'horizontal',
    'aria-valuemin': 0,
    'aria-valuemax': trend.points.length - 1,
    'aria-valuenow': initialIndex,
    'data-power-weight-index': initialIndex,
  })
  const grids = svg('g', { class: 'tri-power-weight-grids', 'aria-hidden': 'true' })
  for (const value of scale.values)
    grids.appendChild(
      svg('line', {
        class: 'tri-power-weight-grid',
        x1: 0,
        x2: W,
        y1: y(value).toFixed(2),
        y2: y(value).toFixed(2),
      }),
    )
  graph.appendChild(grids)
  const referenceLines = svg('g', {
    class: 'tri-power-weight-reference-lines',
    'aria-hidden': 'true',
  })
  for (const reference of trend.references)
    referenceLines.appendChild(
      svg('line', {
        class: 'tri-power-weight-reference-line',
        x1: 0,
        x2: W,
        y1: y(reference.average).toFixed(2),
        y2: y(reference.average).toFixed(2),
        'data-power-weight-duration': String(reference.durationS),
      }),
    )
  graph.appendChild(referenceLines)
  for (const durationS of POWER_TO_WEIGHT_DURATIONS)
    graph.appendChild(
      svg('path', {
        class: 'tri-power-weight-line',
        d: seriesPath(trend.points, durationS, y),
        'data-power-weight-duration': String(durationS),
        'aria-hidden': 'true',
      }),
    )
  const cursor = svg('line', {
    class: 'tri-power-weight-cursor',
    x1: W,
    x2: W,
    y1: 0,
    y2: H,
    'aria-hidden': 'true',
  })
  graph.appendChild(cursor)

  const markers = POWER_TO_WEIGHT_DURATIONS.map(durationS =>
    el('span', 'tri-power-weight-point', undefined, {
      'aria-hidden': 'true',
      'data-power-weight-duration': String(durationS),
    }),
  )
  const dates = trend.points.map(point => point.date)
  const ticks = deconflictPowerToWeightMonthTicks(
    monthTicks(context.formatter, dates, index => (index / Math.max(1, dates.length - 1)) * W),
  )
  const initialYTicks = scale.values.map(value => ({
    label: axisNumber(value, scale.step),
    vbY: y(value),
  }))
  const frame = axisFrame(
    createDomFactory(context.presentation),
    graph,
    initialYTicks,
    H,
    ticks,
    true,
    undefined,
    markers,
  )
  const plot = el('div', 'tri-power-weight-plot')
  const readout = el('div', 'tri-power-weight-readout')
  readout.appendChild(el('span', 'tri-power-weight-date'))
  for (const durationS of POWER_TO_WEIGHT_DURATIONS) {
    const row = el('a', 'tri-power-weight-readout-row', undefined, {
      'data-power-weight-duration': String(durationS),
    })
    row.append(
      el('span', 'tri-power-weight-marker', undefined, {
        'aria-hidden': 'true',
        'data-power-weight-duration': String(durationS),
      }),
      el('span', 'tri-power-weight-readout-duration', powerToWeightDurationLabel(durationS)),
      el('strong', 'tri-power-weight-readout-value'),
    )
    readout.appendChild(row)
  }
  plot.append(frame, readout)
  block.appendChild(plot)
  const reference = buildReferenceTable(data, context)
  if (reference) block.appendChild(reference)

  return {
    element: block,
    mount: () => {
      const yAxis = frame.querySelector<HTMLElement>('.tri-cax-yax')
      const paths = Array.from(graph.querySelectorAll<SVGPathElement>('.tri-power-weight-line'))
      const averageLines = Array.from(
        graph.querySelectorAll<SVGLineElement>('.tri-power-weight-reference-line'),
      )
      let selectedIndex = initialIndex

      const referenceFor = (durationS: PowerToWeightDurationS): PowerToWeightReference | null =>
        trend.references.find(reference => reference.durationS === durationS) ?? null
      const setLink = (link: HTMLAnchorElement, effort: PowerToWeightEffort | null): void => {
        syncPowerCurveActivityLink(link, effortPowerPoint(effort))
      }
      const updateScale = (): void => {
        scale = axisScale(trend.points, active, trend.references)
        const scaleY = (value: number): number => H - (value / scale.domainMax) * (H - 1)
        grids.replaceChildren(
          ...scale.values.map(value =>
            svg('line', {
              class: 'tri-power-weight-grid',
              x1: 0,
              x2: W,
              y1: scaleY(value).toFixed(2),
              y2: scaleY(value).toFixed(2),
            }),
          ),
        )
        if (yAxis)
          yAxis.replaceChildren(
            ...scale.values.map(value =>
              el('span', 'tri-cax-yt', axisNumber(value, scale.step), {
                style: `top:${((scaleY(value) / H) * 100).toFixed(2)}%`,
              }),
            ),
          )
        for (const path of paths) {
          const durationS = durationFrom(path.dataset.powerWeightDuration)
          if (durationS != null) path.setAttribute('d', seriesPath(trend.points, durationS, scaleY))
        }
        for (const line of averageLines) {
          const durationS = durationFrom(line.dataset.powerWeightDuration)
          const reference = durationS == null ? null : referenceFor(durationS)
          if (reference == null) continue
          line.setAttribute('y1', scaleY(reference.average).toFixed(2))
          line.setAttribute('y2', scaleY(reference.average).toFixed(2))
        }
      }
      const showIndex = (requestedIndex: number, commit: boolean): void => {
        const index = clampN(Math.round(requestedIndex), 0, trend.points.length - 1)
        const point = trend.points[index]
        if (commit) {
          selectedIndex = index
          graph.dataset.powerWeightIndex = String(index)
        }
        const xPct = (index / Math.max(1, trend.points.length - 1)) * 100
        cursor.setAttribute('x1', xPct.toFixed(2))
        cursor.setAttribute('x2', xPct.toFixed(2))
        const date = readout.querySelector<HTMLElement>('.tri-power-weight-date')
        if (date) date.textContent = context.formatter.longDate(point.date)
        const valueText: string[] = []
        for (const durationS of POWER_TO_WEIGHT_DURATIONS) {
          const enabled = active.has(durationS)
          const effort = enabled ? effortAt(point, durationS) : null
          const marker = markers.find(
            item => durationFrom(item.dataset.powerWeightDuration) === durationS,
          )
          if (marker) {
            marker.hidden = effort == null
            if (effort != null) {
              marker.style.left = `${xPct.toFixed(2)}%`
              marker.style.top = `${((y(effort.wattsPerKg) / H) * 100).toFixed(2)}%`
            }
          }
          const row = readout.querySelector<HTMLAnchorElement>(
            `.tri-power-weight-readout-row[data-power-weight-duration="${durationS}"]`,
          )
          if (!row) continue
          row.hidden = !enabled
          setLink(row, effort)
          const value = row.querySelector<HTMLElement>('.tri-power-weight-readout-value')
          if (value)
            value.textContent = effort == null ? '—' : `${effort.wattsPerKg.toFixed(2)} W/kg`
          if (effort != null)
            valueText.push(
              `${powerToWeightDurationLabel(durationS)} ${effort.wattsPerKg.toFixed(2)} watts per kilogram`,
            )
        }
        graph.setAttribute('aria-valuenow', String(index))
        graph.setAttribute(
          'aria-valuetext',
          `${context.formatter.longDate(point.date)}; ${valueText.join('; ')}`,
        )
      }
      const showPointer = (event: PointerEvent, commit: boolean): void => {
        const rect = graph.getBoundingClientRect()
        if (rect.width <= 0) return
        showIndex(((event.clientX - rect.left) / rect.width) * (trend.points.length - 1), commit)
      }
      const onMove = (event: PointerEvent): void => showPointer(event, false)
      const onDown = (event: PointerEvent): void => {
        showPointer(event, true)
        graph.focus({ preventScroll: true })
      }
      const onLeave = (): void => showIndex(selectedIndex, false)
      const onKey = (event: KeyboardEvent): void => {
        let next: number | null = null
        if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = selectedIndex - 1
        else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = selectedIndex + 1
        else if (event.key === 'Home') next = 0
        else if (event.key === 'End') next = trend.points.length - 1
        else if (event.key === 'Escape') {
          event.preventDefault()
          event.stopPropagation()
          graph.blur()
          return
        }
        if (next == null) return
        event.preventDefault()
        event.stopPropagation()
        showIndex(next, true)
      }
      const onToggle = (event: MouseEvent): void => {
        if (!(event.target instanceof Element)) return
        const button = event.target.closest<HTMLButtonElement>('.tri-power-weight-toggle')
        const durationS = durationFrom(button?.dataset.powerWeightDuration)
        if (!button || durationS == null || button.disabled) return
        const enabled = active.has(durationS)
        if (enabled && active.size === 1) return
        if (enabled) active.delete(durationS)
        else active.add(durationS)
        button.setAttribute('aria-pressed', String(!enabled))
        for (const item of block.querySelectorAll<HTMLElement | SVGElement>(
          `[data-power-weight-duration="${durationS}"]`,
        ))
          if (!controls.contains(item)) item.toggleAttribute('hidden', enabled)
        updateScale()
        showIndex(selectedIndex, false)
      }
      updateScale()
      showIndex(initialIndex, true)
      graph.addEventListener('pointermove', onMove)
      graph.addEventListener('pointerdown', onDown)
      graph.addEventListener('pointerleave', onLeave)
      graph.addEventListener('pointercancel', onLeave)
      graph.addEventListener('keydown', onKey)
      controls.addEventListener('click', onToggle)
      return () => {
        graph.removeEventListener('pointermove', onMove)
        graph.removeEventListener('pointerdown', onDown)
        graph.removeEventListener('pointerleave', onLeave)
        graph.removeEventListener('pointercancel', onLeave)
        graph.removeEventListener('keydown', onKey)
        controls.removeEventListener('click', onToggle)
      }
    },
  }
}
