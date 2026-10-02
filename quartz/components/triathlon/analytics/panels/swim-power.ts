import type { Analytics } from '../../../../plugins/stores/analytics'
import type { SwimPowerBestPoint } from '../../../../util/swim-power'
import type { AxisXTick } from '../../../../util/triathlon-card'
import type { TriathlonContext } from '../../runtime/context'
import {
  axisFrame,
  axisNumber,
  dlabel,
  niceStep,
  powerCurveDurationTicks,
  powerCurveFraction,
} from '../../../../util/triathlon-card'
import { powerCurveActivityLinkAttributes } from '../../../../util/triathlon-power-activity'
import { createDomFactory, el, svg } from '../../runtime/dom'
import { setupPowerCurveTicks } from '../../shell/power-curve-ticks'
import { anaTitle, markGlossDefinition } from '../shared'

const W = 100
const H = 34

export const buildSwimPowerCurve = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount: () => () => void } => {
  const block = el('section', 'tri-best-power tri-swim-power')
  const power = data.swimPowerCurve
  const series = [
    { key: 'six-weeks', label: context.formatter.text('last 6 weeks'), curve: power.sixWeeks },
    {
      key: 'year',
      label: `${context.formatter.text('all of')} ${power.yearLabel}`,
      curve: power.year,
    },
  ]
  const available = series.filter(item => item.curve.length > 0)
  const head = el('div', 'tri-best-power-head')
  head.appendChild(
    markGlossDefinition(
      anaTitle(context.formatter, 'swim · modeled drag power curve'),
      'Relative drag demand from speed cubed: 2:30/100 m = 100 idx; 200 idx means twice that modeled demand. Assumes a constant drag coefficient. Pool times include turns and push-offs; outdoor GPS speed includes currents. Rest and invalid telemetry break continuous efforts.',
    ),
  )
  const controls = el('div', 'tri-best-power-controls', undefined, {
    role: 'group',
    'aria-label': 'swim power curve periods',
  })
  const active = new Set(available.map(item => item.key))
  const buttons = series.map(item => {
    const button = el('button', 'tri-best-power-toggle', undefined, {
      type: 'button',
      'data-swim-power-series': item.key,
      'aria-pressed': String(active.has(item.key)),
      ...(item.curve.length === 0 ? { disabled: '' } : {}),
    })
    button.append(
      el('span', `tri-best-power-swatch tri-best-power-swatch--${item.key}`, undefined, {
        'aria-hidden': 'true',
      }),
      el('span', undefined, item.label),
    )
    controls.appendChild(button)
    return { button, key: item.key }
  })
  head.appendChild(controls)
  block.appendChild(head)
  if (available.length === 0) {
    block.appendChild(
      el(
        'div',
        'tri-ana-empty',
        'No continuous swim effort of at least 1 minute with valid timing and distance.',
      ),
    )
    return { element: block, mount: () => () => {} }
  }

  const durations = [
    ...new Set(available.flatMap(item => item.curve.map(point => point.durationS))),
  ].sort((a, b) => a - b)
  const minSeconds = durations[0]
  const maxSeconds = durations[durations.length - 1]
  const maximum = Math.max(100, ...available.flatMap(item => item.curve.map(point => point.index)))
  const step = niceStep(maximum, 4)
  const domainMax = Math.ceil(maximum / step) * step
  const X = (seconds: number): number => powerCurveFraction(seconds, minSeconds, maxSeconds) * W
  const Y = (value: number): number => H - (value / domainMax) * (H - 1)
  const graph = svg('svg', {
    class: 'tri-best-power-svg tri-swim-power-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': 'best sustained swim drag-power index by duration',
    'aria-orientation': 'horizontal',
    'aria-valuemin': minSeconds,
    'aria-valuemax': maxSeconds,
  })
  const yTicks = Array.from({ length: Math.round(domainMax / step) + 1 }, (_, i) => ({
    label: axisNumber(i * step, step),
    vbY: Y(i * step),
  }))
  for (const tick of yTicks)
    graph.appendChild(
      svg('line', {
        class: 'tri-best-power-grid',
        x1: 0,
        x2: W,
        y1: tick.vbY,
        y2: tick.vbY,
        'aria-hidden': 'true',
      }),
    )
  graph.appendChild(
    svg('line', {
      class: 'tri-best-power-ftp',
      x1: 0,
      x2: W,
      y1: Y(100),
      y2: Y(100),
      'aria-hidden': 'true',
    }),
  )
  const lines = available.map(item => {
    const line = svg('path', {
      class: `tri-best-power-line tri-swim-power-line tri-best-power-line--${item.key}`,
      d: item.curve
        .map(
          (point, i) =>
            `${i === 0 ? 'M' : 'L'} ${X(point.durationS).toFixed(2)} ${Y(point.index).toFixed(2)}`,
        )
        .join(' '),
      'data-swim-power-series': item.key,
      'aria-hidden': 'true',
    })
    graph.appendChild(line)
    return { line, key: item.key }
  })
  const cursor = svg('line', {
    class: 'tri-best-power-cursor',
    x1: 0,
    x2: 0,
    y1: 0,
    y2: H,
    'aria-hidden': 'true',
  })
  graph.appendChild(cursor)
  const readout = el('div', 'tri-best-power-readout')
  const duration = el('span', 'tri-best-power-duration')
  readout.appendChild(duration)
  const overlays: HTMLElement[] = [readout]
  const rows = available.map(item => {
    const row = el('a', 'tri-best-power-readout-row', undefined, {
      'data-swim-power-series': item.key,
    })
    const value = el('strong', 'tri-best-power-value')
    row.append(
      el('span', `tri-best-power-swatch tri-best-power-swatch--${item.key}`, undefined, {
        'aria-hidden': 'true',
      }),
      value,
    )
    readout.appendChild(row)
    const marker = el('span', `tri-best-power-point tri-best-power-point--${item.key}`, undefined, {
      'aria-hidden': 'true',
    })
    overlays.push(marker)
    return { row, value, marker, ...item }
  })
  let selectedSeconds = durations.reduce(
    (nearest, seconds) => (Math.abs(seconds - 300) < Math.abs(nearest - 300) ? seconds : nearest),
    minSeconds,
  )
  const ticks: AxisXTick[] = powerCurveDurationTicks(
    minSeconds,
    maxSeconds,
    [60, 300, 600, 1200, 1800, 2700, 3600, 5400, 7200],
  ).map((seconds, i, all) => ({
    label: dlabel(seconds),
    pct: X(seconds),
    cls: `tri-best-power-tick${i === 0 ? ' tri-cax-xt--first' : i === all.length - 1 ? ' tri-cax-xt--last' : ''}`,
    tag: 'button',
    attrs: { type: 'button', 'data-power-seconds': String(seconds) },
  }))
  block.appendChild(
    axisFrame(
      createDomFactory(context.presentation),
      graph,
      yTicks,
      H,
      ticks,
      true,
      undefined,
      overlays,
    ),
  )
  const visibleDurations = (): number[] =>
    [
      ...new Set(
        available
          .filter(item => active.has(item.key))
          .flatMap(item => item.curve.map(point => point.durationS)),
      ),
    ].sort((a, b) => a - b)
  const select = (requested: number): void => {
    const observed = visibleDurations()
    selectedSeconds = observed.reduce(
      (nearest, seconds) =>
        Math.abs(Math.log(seconds / requested)) < Math.abs(Math.log(nearest / requested))
          ? seconds
          : nearest,
      observed[0],
    )
    cursor.setAttribute('x1', String(X(selectedSeconds)))
    cursor.setAttribute('x2', String(X(selectedSeconds)))
    duration.textContent = dlabel(selectedSeconds)
    const labels = [dlabel(selectedSeconds)]
    for (const item of rows) {
      const point: SwimPowerBestPoint | undefined = item.curve.find(
        point => point.durationS === selectedSeconds,
      )
      item.row.hidden = !active.has(item.key)
      item.marker.hidden = item.row.hidden || point == null
      item.value.textContent = `${item.label} · ${point ? point.index.toFixed(1) : '—'} idx`
      for (const name of [
        'href',
        'data-power-activity-id',
        'data-power-activity-date',
        'aria-disabled',
      ])
        item.row.removeAttribute(name)
      for (const [name, value] of Object.entries(powerCurveActivityLinkAttributes(point)))
        item.row.setAttribute(name, value)
      if (point) {
        item.row.title = `${point.activityDate} · ${point.inputSource === 'route' ? 'GPS ground speed' : `${point.inputSource} lengths`} · ${dlabel(point.startElapsedS)}–${dlabel(point.endElapsedS)}`
        item.marker.style.left = `${X(selectedSeconds)}%`
        item.marker.style.top = `${(Y(point.index) / H) * 100}%`
      } else item.row.removeAttribute('title')
      if (!item.row.hidden) labels.push(item.value.textContent)
    }
    graph.setAttribute('aria-valuenow', String(selectedSeconds))
    graph.setAttribute('aria-valuetext', labels.join(' · '))
    for (const tick of block.querySelectorAll<HTMLButtonElement>('[data-power-seconds]'))
      tick.setAttribute(
        'aria-pressed',
        String(Number(tick.dataset.powerSeconds) === selectedSeconds),
      )
  }
  select(selectedSeconds)
  return {
    element: block,
    mount: () => {
      const move = (event: PointerEvent): void => {
        const rect = graph.getBoundingClientRect()
        if (rect.width <= 0) return
        const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
        select(minSeconds * (maxSeconds / minSeconds) ** fraction)
      }
      const key = (event: KeyboardEvent): void => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
        event.preventDefault()
        event.stopPropagation()
        const observed = visibleDurations()
        const index = observed.indexOf(selectedSeconds)
        select(
          event.key === 'Home'
            ? observed[0]
            : event.key === 'End'
              ? observed[observed.length - 1]
              : observed[
                  Math.max(
                    0,
                    Math.min(observed.length - 1, index + (event.key === 'ArrowLeft' ? -1 : 1)),
                  )
                ],
        )
      }
      const click = (event: MouseEvent): void => {
        if (!(event.target instanceof Element)) return
        const target = event.target.closest<HTMLButtonElement>('button')
        const toggle = buttons.find(item => item.button === target)
        if (toggle) {
          if (active.has(toggle.key) && active.size === 1) return
          if (active.has(toggle.key)) active.delete(toggle.key)
          else active.add(toggle.key)
          for (const item of buttons)
            item.button.setAttribute('aria-pressed', String(active.has(item.key)))
          for (const item of lines) item.line.toggleAttribute('hidden', !active.has(item.key))
          select(selectedSeconds)
        } else if (target?.dataset.powerSeconds) select(Number(target.dataset.powerSeconds))
      }
      graph.addEventListener('pointermove', move)
      graph.addEventListener('keydown', key)
      block.addEventListener('click', click)
      const cleanupTicks = setupPowerCurveTicks(block)
      return () => {
        graph.removeEventListener('pointermove', move)
        graph.removeEventListener('keydown', key)
        block.removeEventListener('click', click)
        cleanupTicks()
      }
    },
  }
}
