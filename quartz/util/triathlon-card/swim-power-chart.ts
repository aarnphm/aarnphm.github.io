import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { SwimPowerEstimate } from '../swim-power'
import type { TriNodeFactory } from './construction-contracts'
import { axisFrame, axisNumber, niceStep } from './chart-geometry'
import { dlabel } from './metric-text'
import {
  embeddedPowerCurveDurationTicks,
  powerCurveDurationTicks,
  powerCurveFraction,
} from './power-curve-data'

export const swimDragPowerDefinition = (speedBasis: SwimPowerEstimate['speedBasis']): string =>
  speedBasis === 'ground-speed'
    ? 'Relative drag demand: 2:30/100m = 100 idx; 200 idx means twice that modeled demand. GPS ground speed is averaged over 60 s intervals, then cubed, assuming a constant drag coefficient. Current, waves and GPS error affect the estimate.'
    : 'Relative freestyle drag demand: 2:30/100m = 100 idx; 200 idx means twice that modeled demand. Recorded length speed is cubed, assuming a constant drag coefficient. Length times include turns and push-offs. Rest and excluded lengths break continuous efforts.'

export const buildSwimPowerCurve = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  embedded = false,
): N | null => {
  const estimate = d.swimPower
  if (d.sport !== 'swim' || !estimate?.curve.length) return null
  const curve = estimate.curve
  const width = 100,
    height = 34
  const min = curve[0].durationS,
    max = curve[curve.length - 1].durationS
  const step = niceStep(Math.max(100, ...curve.map(p => p.index)), 4)
  const domainMax = Math.ceil(Math.max(100, ...curve.map(p => p.index)) / step) * step
  const X = (seconds: number) => powerCurveFraction(seconds, min, max) * width
  const Y = (index: number) => height - (index / domainMax) * (height - 1)
  const selected = curve[0]
  const wrap = f.el('div', 'tri-zone tri-swim-drag-chart', undefined, {
    'data-tri-trace': 'swim-drag-power',
    'data-swim-power-source': estimate.source,
  })
  const head = f.el('div', 'tri-curve-head')
  f.add(
    head,
    f.el('div', 'tri-zone-title', 'modeled drag power', {
      'data-gloss': '',
      'data-gloss-def': swimDragPowerDefinition(estimate.speedBasis),
      tabindex: '0',
    }),
  )
  f.add(wrap, head)
  const graph = f.svg('svg', {
    class: 'tri-swim-drag-svg',
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: 'none',
    'data-swim-drag-curve': JSON.stringify(curve),
    'data-swim-drag-domain-max': domainMax,
    role: 'slider',
    tabindex: 0,
    'aria-label': 'modeled swim drag power curve',
    'aria-orientation': 'horizontal',
    'aria-valuemin': min,
    'aria-valuemax': max,
    'aria-valuenow': min,
    'aria-valuetext': `${dlabel(min)} · ${selected.index.toFixed(1)} idx`,
  })
  const yTicks = Array.from({ length: Math.round(domainMax / step) + 1 }, (_, i) => ({
    label: axisNumber(i * step, step),
    vbY: Y(i * step),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-best-power-grid',
        x1: 0,
        x2: width,
        y1: tick.vbY,
        y2: tick.vbY,
        'aria-hidden': 'true',
      }),
    )
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-curve-ftp',
      x1: 0,
      x2: width,
      y1: Y(100),
      y2: Y(100),
      'aria-hidden': 'true',
    }),
    f.svg('path', {
      class: 'tri-curve-line',
      d: curve
        .map((p, i) => `${i ? 'L' : 'M'} ${X(p.durationS).toFixed(2)} ${Y(p.index).toFixed(2)}`)
        .join(' '),
      'aria-hidden': 'true',
    }),
    f.svg('line', {
      class: 'tri-chart-cursor',
      x1: 0,
      x2: 0,
      y1: 0,
      y2: height,
      'aria-hidden': 'true',
    }),
  )
  const marker = f.el('span', 'tri-curve-point tri-curve-point--ride', undefined, {
    'aria-hidden': 'true',
    style: `left:0%;top:${(Y(selected.index) / height) * 100}%`,
  })
  const readout = f.el('div', 'tri-chart-readout tri-curve-readout')
  f.add(
    readout,
    f.el('span', 'tri-curve-readout-duration', dlabel(min)),
    f.el('strong', 'tri-swim-drag-value', `${selected.index.toFixed(1)} idx`),
  )
  const tickDurations = embedded
    ? embeddedPowerCurveDurationTicks(min, max, [60, 300, 600, 1200, 1800, 3600])
    : powerCurveDurationTicks(min, max, [60, 300, 600, 1200, 1800, 3600])
  f.add(
    wrap,
    axisFrame(
      f,
      graph,
      yTicks,
      height,
      tickDurations.map(sec => ({
        label: dlabel(sec),
        pct: X(sec),
        cls: 'tri-swim-drag-tick',
        tag: 'button',
        attrs: {
          type: 'button',
          'data-swim-drag-seconds': String(sec),
          'aria-pressed': String(sec === min),
        },
      })),
      true,
      undefined,
      [marker, readout],
    ),
  )
  return wrap
}
