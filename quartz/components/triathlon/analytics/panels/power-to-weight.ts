import type {
  Analytics,
  PowerToWeightDurationS,
  PowerToWeightEffort,
  PowerToWeightReference,
} from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import { POWER_TO_WEIGHT_DURATIONS } from '../../../../plugins/stores/analytics'
import { axisFrame, axisNumber } from '../../../../util/triathlon-card'
import { syncPowerCurveActivityLink } from '../../activity/power-links'
import { createDomFactory, el, setText, svg } from '../../runtime/dom'
import { rafEvent, rectOf } from '../scrub-primitives'
import {
  anaTitle,
  buildVerticalGrid,
  clampN,
  dateGridPositions,
  markGlossDefinition,
  monthTicks,
} from '../shared'
import {
  axisScale,
  deconflictPowerToWeightMonthTicks,
  durationFrom,
  effortAt,
  effortPowerPoint,
  H,
  powerToWeightDurationLabel,
  restorePowerToWeightDurations,
  seriesPath,
  W,
} from './power-to-weight/model'
import { buildReferenceTable } from './power-to-weight/reference'

const SELECTION_KEY = 'tri-power-weight-durations'

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
  const sourceRideIds = new Set<string>()
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
  const dates = trend.points.map(point => point.date)
  const ticks = deconflictPowerToWeightMonthTicks(
    monthTicks(context.formatter, dates, index => (index / Math.max(1, dates.length - 1)) * W),
  )
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
  graph.append(
    grids,
    buildVerticalGrid(
      ticks.length >= 3
        ? ticks.map(tick => tick.pct)
        : dateGridPositions(dates, index => (index / Math.max(1, dates.length - 1)) * W),
      1,
      H,
    ),
  )
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
      const dateEl = readout.querySelector<HTMLElement>('.tri-power-weight-date')
      const markerByDuration = new Map<PowerToWeightDurationS, HTMLElement>()
      for (const marker of markers) {
        const d = durationFrom(marker.dataset.powerWeightDuration)
        if (d != null) markerByDuration.set(d, marker)
      }
      const rowByDuration = new Map<
        PowerToWeightDurationS,
        { row: HTMLAnchorElement; value: HTMLElement | null }
      >()
      for (const durationS of POWER_TO_WEIGHT_DURATIONS) {
        const row = readout.querySelector<HTMLAnchorElement>(
          `.tri-power-weight-readout-row[data-power-weight-duration="${durationS}"]`,
        )
        if (row)
          rowByDuration.set(durationS, {
            row,
            value: row.querySelector<HTMLElement>('.tri-power-weight-readout-value'),
          })
      }
      let selectedIndex = initialIndex
      let shownIndex = -1

      const referenceFor = (durationS: PowerToWeightDurationS): PowerToWeightReference | null =>
        trend.references.find(reference => reference.durationS === durationS) ?? null
      const setLink = (link: HTMLAnchorElement, effort: PowerToWeightEffort | null): void => {
        syncPowerCurveActivityLink(link, effortPowerPoint(effort))
      }
      const updateScale = (): void => {
        shownIndex = -1
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
        if (index === shownIndex) return
        shownIndex = index
        const xPct = (index / Math.max(1, trend.points.length - 1)) * 100
        cursor.setAttribute('x1', xPct.toFixed(2))
        cursor.setAttribute('x2', xPct.toFixed(2))
        if (dateEl) setText(dateEl, context.formatter.longDate(point.date))
        const valueText: string[] = []
        for (const durationS of POWER_TO_WEIGHT_DURATIONS) {
          const enabled = active.has(durationS)
          const effort = enabled ? effortAt(point, durationS) : null
          const marker = markerByDuration.get(durationS)
          if (marker) {
            marker.style.display = effort == null ? 'none' : ''
            if (effort != null) {
              if (marker.hidden) marker.hidden = false
              marker.style.left = `${xPct.toFixed(2)}%`
              marker.style.top = `${((y(effort.wattsPerKg) / H) * 100).toFixed(2)}%`
            }
          }
          const slot = rowByDuration.get(durationS)
          if (!slot) continue
          if (slot.row.hidden !== !enabled) slot.row.hidden = !enabled
          setLink(slot.row, effort)
          if (slot.value)
            setText(slot.value, effort == null ? '—' : `${effort.wattsPerKg.toFixed(2)} W/kg`)
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
        const rect = rectOf(graph)
        if (rect.width <= 0) return
        showIndex(((event.clientX - rect.left) / rect.width) * (trend.points.length - 1), commit)
      }
      const rafMove = rafEvent((event: PointerEvent): void => showPointer(event, false))
      const onDown = (event: PointerEvent): void => {
        rafMove.cancel()
        showPointer(event, true)
        graph.focus({ preventScroll: true })
      }
      const onLeave = (): void => {
        rafMove.cancel()
        showIndex(selectedIndex, false)
      }
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
        try {
          localStorage.setItem(SELECTION_KEY, JSON.stringify([...active]))
        } catch {}
        syncSelection()
        updateScale()
        showIndex(selectedIndex, false)
      }
      const syncSelection = (): void => {
        for (const button of controls.querySelectorAll<HTMLButtonElement>(
          '.tri-power-weight-toggle',
        )) {
          const durationS = durationFrom(button.dataset.powerWeightDuration)
          button.setAttribute('aria-pressed', String(durationS != null && active.has(durationS)))
        }
        for (const item of block.querySelectorAll<HTMLElement | SVGElement>(
          '[data-power-weight-duration]',
        )) {
          if (controls.contains(item)) continue
          const durationS = durationFrom(item.dataset.powerWeightDuration)
          item.toggleAttribute('hidden', durationS == null || !active.has(durationS))
        }
      }
      try {
        const stored: unknown = JSON.parse(localStorage.getItem(SELECTION_KEY) ?? 'null')
        const restored = restorePowerToWeightDurations(stored, available)
        active.clear()
        for (const durationS of restored) active.add(durationS)
      } catch {}
      syncSelection()
      updateScale()
      showIndex(initialIndex, true)
      graph.addEventListener('pointermove', rafMove.onEvent)
      graph.addEventListener('pointerdown', onDown)
      graph.addEventListener('pointerleave', onLeave)
      graph.addEventListener('pointercancel', onLeave)
      graph.addEventListener('keydown', onKey)
      controls.addEventListener('click', onToggle)
      return () => {
        rafMove.cancel()
        graph.removeEventListener('pointermove', rafMove.onEvent)
        graph.removeEventListener('pointerdown', onDown)
        graph.removeEventListener('pointerleave', onLeave)
        graph.removeEventListener('pointercancel', onLeave)
        graph.removeEventListener('keydown', onKey)
        controls.removeEventListener('click', onToggle)
      }
    },
  }
}
export {
  deconflictPowerToWeightMonthTicks,
  powerToWeightDurationLabel,
  restorePowerToWeightDurations,
} from './power-to-weight/model'
