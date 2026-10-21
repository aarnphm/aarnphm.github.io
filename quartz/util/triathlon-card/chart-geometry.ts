import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { isImperial, KM_TO_MI, swimDistanceLabel, zoneClock } from './metric-text'

export const niceStep = (span: number, intervals: number): number => {
  if (!Number.isFinite(span) || span <= 0) return 1
  const raw = span / Math.max(1, intervals)
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const fraction = raw / magnitude
  const nice = fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10
  return nice * magnitude
}

export const niceTicks = (min: number, max: number, intervals: number): number[] => {
  const step = niceStep(max - min, intervals)
  const first = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let value = first; value <= max + step * 1e-6; value += step)
    ticks.push(Math.round(value * 1e6) / 1e6)
  if (ticks.length >= 2) return ticks
  return [min, max]
}

export const axisNumber = (value: number, step: number): string => {
  const decimals = step >= 1 ? 0 : Math.min(2, Math.ceil(-Math.log10(step)))
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export type ActivityGraphDomain = { startDistanceKm: number; endDistanceKm: number }
export type ActivityTraceDomain = { min: number; max: number; intervals?: number }
export type ActivityTraceReference = { value: number; label: string }
export type ActivityTraceRenderOptions<N> = {
  traceName?: string
  wrapClass?: string
  wrapAttrs?: Record<string, string>
  capExtra?: N[]
  areaBaseline?: number
}

export const positiveMetricDomain = (
  values: readonly (number | null)[],
  intervals = 3,
): ActivityTraceDomain | undefined => {
  const measured = values.filter(
    (value): value is number => value != null && Number.isFinite(value) && value > 0,
  )
  if (measured.length === 0) return undefined
  const min = Math.min(...measured)
  const max = Math.max(...measured)
  if (max > min) return { min, max, intervals }
  const pad = Math.max(1, max * 0.05)
  return { min: Math.max(Number.EPSILON, min - pad), max: max + pad, intervals }
}

export const distanceXTicks = (
  presentation: TriathlonPresentation,
  startDistanceKm: number,
  endDistanceKm: number,
): AxisXTick[] => {
  const imperial = isImperial(presentation)
  const scale = imperial ? KM_TO_MI : 1
  const displayStart = startDistanceKm * scale
  const displayEnd = endDistanceKm * scale
  const displaySpan = displayEnd - displayStart
  const step = niceStep(displaySpan, 4)
  const first = Math.ceil(displayStart / step) * step
  const ticks: AxisXTick[] = []
  for (let value = first; value < displayEnd - step * 1e-6; value += step) {
    if (value <= displayStart + step * 1e-6) continue
    ticks.push({
      label: `${axisNumber(value, step)} ${imperial ? 'mi' : 'km'}`,
      pct: ((value - displayStart) / displaySpan) * 100,
    })
  }
  return ticks
}

export const elapsedActivityXTicks = (totalElapsedS: number): AxisXTick[] => [
  { label: zoneClock(0), pct: 0, cls: 'tri-cax-xt--first' },
  { label: zoneClock(totalElapsedS / 2), pct: 50 },
  { label: zoneClock(totalElapsedS), pct: 100, cls: 'tri-cax-xt--last' },
]

export const graphView = (
  d: StravaActivityDetail,
  domain?: ActivityGraphDomain | null,
): ActivityGraphDomain & { start: number; width: number } =>
  graphViewForDistance(d.route[d.route.length - 1].d || 1, domain)

export const graphViewForDistance = (
  maxDistanceKm: number,
  domain?: ActivityGraphDomain | null,
): ActivityGraphDomain & { start: number; width: number } => {
  const resolved = domain
    ? {
        startDistanceKm: Math.max(0, Math.min(maxDistanceKm, domain.startDistanceKm)),
        endDistanceKm: Math.max(0, Math.min(maxDistanceKm, domain.endDistanceKm)),
      }
    : { startDistanceKm: 0, endDistanceKm: maxDistanceKm }
  const valid =
    resolved.endDistanceKm > resolved.startDistanceKm
      ? resolved
      : { startDistanceKm: 0, endDistanceKm: maxDistanceKm }
  return {
    ...valid,
    start: (valid.startDistanceKm / maxDistanceKm) * 100,
    width: ((valid.endDistanceKm - valid.startDistanceKm) / maxDistanceKm) * 100,
  }
}

export const swimActivityXTicks = (endDistanceM: number, startDistanceM = 0): AxisXTick[] => {
  return [
    { label: swimDistanceLabel(startDistanceM), pct: 0, cls: 'tri-cax-xt--first' },
    { label: swimDistanceLabel((startDistanceM + endDistanceM) / 2), pct: 50 },
    { label: swimDistanceLabel(endDistanceM), pct: 100, cls: 'tri-cax-xt--last' },
  ]
}

export type AxisXTick = {
  label: string
  pct: number
  cls?: string
  tag?: 'span' | 'button'
  attrs?: Record<string, string>
}

export type AxisYTick = { label: string; vbY: number; cls?: string; attrs?: Record<string, string> }

export const axisFrame = <N>(
  f: TriNodeFactory<N>,
  svgEl: N,
  yTicks: AxisYTick[],
  vbH: number,
  xTicks: AxisXTick[],
  axes = true,
  axisRange?: { top: number; bottom: number },
  stageOverlays: N[] = [],
  rightYTicks: AxisYTick[] = [],
): N => {
  const frame = f.el('div', `tri-cax-frame${rightYTicks.length > 0 ? ' tri-cax-frame--dual' : ''}`)
  const yax = f.el('div', 'tri-cax-yax')
  for (const t of yTicks)
    f.add(
      yax,
      f.el('span', `tri-cax-yt${t.cls ? ` ${t.cls}` : ''}`, t.label, {
        ...t.attrs,
        style: `top:${((t.vbY / vbH) * 100).toFixed(2)}%`,
      }),
    )
  const stage = f.el('div', 'tri-cax-stage')
  if (axes && (yTicks.length >= 2 || axisRange)) {
    const pcts = yTicks.map(t => (t.vbY / vbH) * 100)
    const top = axisRange ? (axisRange.top / vbH) * 100 : Math.min(...pcts)
    const base = axisRange ? (axisRange.bottom / vbH) * 100 : Math.max(...pcts)
    f.add(
      stage,
      f.el('span', 'tri-cax-ax tri-cax-ax--y', undefined, {
        style: `top:${top.toFixed(2)}%;height:${(base - top).toFixed(2)}%`,
      }),
      f.el('span', 'tri-cax-ax tri-cax-ax--x', undefined, { style: `top:${base.toFixed(2)}%` }),
    )
  }
  f.add(stage, svgEl, ...stageOverlays)
  const xax = f.el('div', 'tri-cax-xax')
  for (const t of xTicks)
    f.add(
      xax,
      f.el(t.tag ?? 'span', `tri-cax-xt${t.cls ? ` ${t.cls}` : ''}`, t.label, {
        ...t.attrs,
        style: `left:${t.pct.toFixed(2)}%`,
      }),
    )
  if (rightYTicks.length > 0) {
    const rightYax = f.el('div', 'tri-cax-yax tri-cax-yax--right')
    for (const t of rightYTicks)
      f.add(
        rightYax,
        f.el('span', `tri-cax-yt tri-cax-yt--right${t.cls ? ` ${t.cls}` : ''}`, t.label, {
          ...t.attrs,
          style: `top:${((t.vbY / vbH) * 100).toFixed(2)}%`,
        }),
      )
    f.add(frame, yax, stage, rightYax, xax)
  } else f.add(frame, yax, stage, xax)
  return frame
}

export const buildPowerCurveGrid = <N>(
  f: TriNodeFactory<N>,
  axis: 'power' | 'duration',
  positions: readonly number[],
  height: number,
): N => {
  const grid = f.svg('g', {
    class: `tri-power-curve-grid tri-power-curve-grid--${axis}`,
    'aria-hidden': 'true',
  })
  for (const position of positions) {
    const value = position.toFixed(2)
    f.add(
      grid,
      f.svg(
        'line',
        axis === 'power'
          ? { x1: 0, x2: 100, y1: value, y2: value }
          : { x1: value, x2: value, y1: 1, y2: height, 'data-curve-grid-pct': value },
      ),
    )
  }
  return grid
}
