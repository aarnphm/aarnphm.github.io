import type { Analytics } from '../../../../plugins/stores/analytics'
import type { PowerCurveBlock } from '../../../../plugins/stores/analytics'
import type { PowerCurvePoint } from '../../../../plugins/stores/strava'
import type { AxisXTick } from '../../../../util/triathlon-card'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { axisFrame } from '../../../../util/triathlon-card'
import { axisNumber } from '../../../../util/triathlon-card'
import { dlabel } from '../../../../util/triathlon-card'
import { nearestPowerCurveValue } from '../../../../util/triathlon-card'
import { niceStep } from '../../../../util/triathlon-card'
import { powerCurveDurationTicks } from '../../../../util/triathlon-card'
import { powerCurveFraction } from '../../../../util/triathlon-card'
import { powerCurveHoverAt } from '../../../../util/triathlon-card'
import { powerCurvePathPoints } from '../../../../util/triathlon-card'
import { zoneClock } from '../../../../util/triathlon-card'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { anaTitle } from '../shared'

export type BestPowerSeriesKey = 'six-weeks' | 'year'

export const bestPowerSeriesLabel = (
  formatter: TriathlonFormatter,
  power: PowerCurveBlock,
  key: BestPowerSeriesKey,
): string =>
  key === 'six-weeks'
    ? formatter.text('last 6 weeks')
    : power.yearLabel == null
      ? formatter.text('calendar year')
      : `${formatter.text('all of')} ${power.yearLabel}`

export const bestPowerSeries = (
  power: PowerCurveBlock,
): Array<{ key: BestPowerSeriesKey; curve: PowerCurvePoint[] }> => [
  { key: 'six-weeks', curve: power.sixWeeks },
  { key: 'year', curve: power.year },
]

export const buildBestPowerCurve = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('section', 'tri-best-power')
  const power = data.powerCurve
  const head = el('div', 'tri-best-power-head')
  head.appendChild(anaTitle(context.formatter, 'best efforts · power curve'))
  const controls = el('div', 'tri-best-power-controls', undefined, {
    role: 'group',
    'aria-label': context.formatter.text('power curve periods'),
  })
  const series = bestPowerSeries(power)
  for (const { key, curve } of series) {
    const available = curve.length >= 2
    const attrs: Record<string, string> = {
      type: 'button',
      'data-power-series': key,
      'aria-pressed': String(available),
    }
    if (!available) attrs.disabled = ''
    const button = el('button', 'tri-best-power-toggle', undefined, attrs)
    button.append(
      el('span', `tri-best-power-swatch tri-best-power-swatch--${key}`, undefined, {
        'aria-hidden': 'true',
      }),
      el('span', undefined, bestPowerSeriesLabel(context.formatter, power, key)),
    )
    controls.appendChild(button)
  }
  head.appendChild(controls)
  block.appendChild(head)

  const available = series.filter(({ curve }) => curve.length >= 2)
  if (available.length === 0) {
    block.appendChild(el('div', 'tri-ana-empty', context.formatter.text('no cycling power data')))
    return block
  }

  block.appendChild(
    el(
      'p',
      'tri-best-power-note',
      context.formatter.text('maximal average power sustained for each duration'),
    ),
  )
  const minSeconds = Math.min(...available.map(({ curve }) => curve[0].s))
  const maxSeconds = Math.max(...available.map(({ curve }) => curve[curve.length - 1].s))
  const W = 100
  const H = 34
  const observedMax = Math.max(
    1,
    ...available.flatMap(({ curve }) => curve.map(point => point.w)),
    power.ftp ?? 0,
    power.goalFtp ?? 0,
  )
  const step = niceStep(observedMax, 4)
  const domainMax = Math.ceil(observedMax / step) * step
  const X = (seconds: number): number => powerCurveFraction(seconds, minSeconds, maxSeconds) * W
  const Y = (watts: number): number => H - (watts / domainMax) * (H - 1)
  const yTicks = Array.from(
    { length: Math.round(domainMax / step) + 1 },
    (_, index) => index * step,
  ).map(value => ({ label: value === 0 ? '0' : `${axisNumber(value, step)}w`, vbY: Y(value) }))
  const path = (curve: PowerCurvePoint[]): string =>
    powerCurvePathPoints(curve)
      .map(
        (point, index) =>
          `${index === 0 ? 'M' : 'L'} ${X(point.s).toFixed(2)} ${Y(point.w).toFixed(2)}`,
      )
      .join(' ')
  const anchor = available[0].curve
  const initial = powerCurveHoverAt(
    anchor,
    [],
    powerCurveFraction(300, anchor[0].s, anchor[anchor.length - 1].s),
  )
  const selectedSeconds = initial?.durationS ?? anchor[0].s
  const graph = svg('svg', {
    class: 'tri-best-power-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': context.formatter.text('best efforts power curve'),
    'aria-orientation': 'horizontal',
    'aria-valuemin': minSeconds,
    'aria-valuemax': maxSeconds,
    'aria-valuenow': selectedSeconds,
    'data-power-selected-seconds': selectedSeconds,
    'data-power-domain-max': domainMax,
  })
  for (const tick of yTicks)
    graph.appendChild(
      svg('line', {
        class: 'tri-best-power-grid',
        x1: 0,
        y1: tick.vbY.toFixed(2),
        x2: W,
        y2: tick.vbY.toFixed(2),
        'aria-hidden': 'true',
      }),
    )
  for (const { key, curve } of available)
    graph.appendChild(
      svg('path', {
        class: `tri-best-power-line tri-best-power-line--${key}`,
        d: path(curve),
        'data-power-series': key,
        'aria-hidden': 'true',
      }),
    )
  if (power.ftp != null)
    graph.appendChild(
      svg('line', {
        class: 'tri-best-power-ftp',
        x1: 0,
        y1: Y(power.ftp).toFixed(2),
        x2: W,
        y2: Y(power.ftp).toFixed(2),
      }),
    )
  if (power.goalFtp != null)
    graph.appendChild(
      svg('line', {
        class: 'tri-best-power-goal',
        x1: 0,
        y1: Y(power.goalFtp).toFixed(2),
        x2: W,
        y2: Y(power.goalFtp).toFixed(2),
      }),
    )
  graph.appendChild(
    svg('line', {
      class: 'tri-best-power-cursor',
      x1: X(selectedSeconds).toFixed(2),
      y1: 0,
      x2: X(selectedSeconds).toFixed(2),
      y2: H,
    }),
  )

  const overlays: Array<HTMLElement | SVGElement> = []
  const readout = el('div', 'tri-best-power-readout')
  readout.appendChild(el('span', 'tri-best-power-duration', zoneClock(selectedSeconds)))
  for (const { key, curve } of available) {
    const watts = nearestPowerCurveValue(curve, selectedSeconds)
    const point = el('span', `tri-best-power-point tri-best-power-point--${key}`, undefined, {
      'data-power-series': key,
      'aria-hidden': 'true',
    })
    if (watts != null)
      point.setAttribute(
        'style',
        `left:${X(selectedSeconds).toFixed(2)}%;top:${((Y(watts) / H) * 100).toFixed(2)}%`,
      )
    const row = el('span', 'tri-best-power-readout-row', undefined, { 'data-power-series': key })
    row.append(
      el('span', `tri-best-power-swatch tri-best-power-swatch--${key}`, undefined, {
        'aria-hidden': 'true',
      }),
      el('strong', 'tri-best-power-value', watts == null ? '—' : `${watts.toLocaleString()} W`),
      el('span', 'tri-best-power-label', bestPowerSeriesLabel(context.formatter, power, key)),
    )
    overlays.push(point)
    readout.appendChild(row)
  }
  overlays.push(readout)

  const durationTicks: AxisXTick[] = powerCurveDurationTicks(
    minSeconds,
    maxSeconds,
    [1, 15, 60, 300, 600, 1_200, 1_800, 2_700, 3_600, 5_400, 7_200, 10_800, 14_400, 18_000],
  ).map((seconds, index, ticks) => ({
    label: dlabel(seconds),
    pct: X(seconds),
    cls: `tri-best-power-tick${index === 0 ? ' tri-cax-xt--first' : index === ticks.length - 1 ? ' tri-cax-xt--last' : ''}`,
    tag: 'button',
    attrs: {
      type: 'button',
      'data-power-seconds': String(seconds),
      'aria-pressed': String(seconds === selectedSeconds),
    },
  }))
  block.appendChild(
    axisFrame(
      createDomFactory(context.presentation),
      graph,
      yTicks,
      H,
      durationTicks,
      true,
      undefined,
      overlays,
    ),
  )

  const cap = el('div', 'tri-best-power-cap')
  if (power.ftp != null) cap.appendChild(el('span', undefined, `FTP ${power.ftp}W`))
  if (power.goalFtp != null)
    cap.appendChild(
      el('span', 'tri-best-power-cap-goal', `${context.formatter.text('goal')} ${power.goalFtp}W`),
    )
  block.appendChild(cap)
  return block
}
