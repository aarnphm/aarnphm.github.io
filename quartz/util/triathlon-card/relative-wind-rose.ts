import type { MyWindsockRouteSample } from '../mywindsock-route'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { relativeWindReadout } from './environment-readout'
import { isImperial, KM_TO_MI } from './metric-text'

const RELATIVE_WIND_RADIUS = 40

// Rider-frame wind cloud: each moving bin sits at (crosswind, headwind), so angle reads as
// time share and distance from the rider reads as strength, which sector shares alone hide.
export const buildRelativeWindRose = <N>(
  f: TriNodeFactory<N>,
  shares: readonly number[],
  samples: readonly MyWindsockRouteSample[],
): N => {
  const rose = f.el('div', 'tri-environment-rose')
  const scale = isImperial(f.presentation) ? KM_TO_MI : 1
  // index is the position in the provider series, which the client cursor shares with the wind tab.
  const points = samples.flatMap((sample, index) =>
    sample.providerHeadwindKph == null || sample.providerCrosswindKph == null
      ? []
      : [
          {
            index,
            elapsedS: sample.elapsedS,
            head: sample.providerHeadwindKph * scale,
            cross: sample.providerCrosswindKph * scale,
          },
        ],
  )
  const magnitudes = points.map(point => Math.hypot(point.head, point.cross)).sort((a, b) => a - b)
  // The 95th percentile sets the rim so one gusty bin cannot shrink the cloud; outliers pin to it.
  const reach = magnitudes[Math.floor(magnitudes.length * 0.95)] ?? 0
  const step = reach > 30 ? 10 : reach > 12 ? 5 : reach > 4 ? 2 : 1
  const rim = Math.max(step, Math.ceil(reach / step) * step)
  const unitRadius = RELATIVE_WIND_RADIUS / rim
  const meanHead = points.reduce((total, point) => total + point.head, 0) / (points.length || 1)
  const meanCross = points.reduce((total, point) => total + point.cross, 0) / (points.length || 1)
  const unit = isImperial(f.presentation) ? 'mph' : 'km/h'
  const signed = (value: number) => `${value > 0 ? '+' : ''}${value.toFixed(1)} ${unit}`
  const mean =
    points.length > 0
      ? `${triText(f.presentation.locale, 'average')} ${triText(f.presentation.locale, 'head')} ${signed(meanHead)} · ${triText(f.presentation.locale, 'crosswind')} ${signed(meanCross)}`
      : null
  const summary = [
    `${triText(f.presentation.locale, 'relative wind')}: ${relativeWindReadout(f.presentation, shares)}`,
    ...(mean ? [mean] : []),
  ].join(' · ')
  const svg = f.svg('svg', {
    class: 'tri-environment-rose-plot',
    viewBox: `${-RELATIVE_WIND_RADIUS - 2} ${-RELATIVE_WIND_RADIUS - 2} ${RELATIVE_WIND_RADIUS * 2 + 4} ${RELATIVE_WIND_RADIUS * 2 + 4}`,
    'aria-label': summary,
    ...(points.length > 0
      ? {
          role: 'slider',
          tabindex: 0,
          'aria-valuemin': 0,
          'aria-valuemax': Math.round(points.at(-1)?.elapsedS ?? 0),
          'aria-valuenow': Math.round(points.at(-1)?.elapsedS ?? 0),
          'aria-valuetext': summary,
          'data-environment-rose': '',
        }
      : { role: 'img' }),
  })
  const ringLabels: N[] = []
  const span = RELATIVE_WIND_RADIUS * 2 + 4
  for (let ring = step; ring <= rim; ring += step) {
    const radius = ring * unitRadius
    f.add(svg, f.svg('circle', { class: 'tri-environment-gridline', r: radius }))
    // SVG text would scale with the plot; an HTML label at the ring's 45° point keeps type size.
    const offset = (radius * Math.SQRT1_2 + RELATIVE_WIND_RADIUS + 2) / span
    ringLabels.push(
      f.el(
        'span',
        'tri-environment-rose-ring-label',
        ring === rim ? `${ring} ${unit}` : `${ring}`,
        { style: `left: ${(offset * 100).toFixed(2)}%; bottom: ${(offset * 100).toFixed(2)}%` },
      ),
    )
  }
  f.add(
    svg,
    f.svg('line', {
      class: 'tri-environment-gridline',
      x1: 0,
      x2: 0,
      y1: -RELATIVE_WIND_RADIUS,
      y2: RELATIVE_WIND_RADIUS,
    }),
    f.svg('line', {
      class: 'tri-environment-gridline',
      x1: -RELATIVE_WIND_RADIUS,
      x2: RELATIVE_WIND_RADIUS,
      y1: 0,
      y2: 0,
    }),
  )
  const cloud = f.svg('g', { class: 'tri-environment-rose-cloud' })
  for (const point of points) {
    const magnitude = Math.hypot(point.head, point.cross)
    const clamp = magnitude > rim ? rim / magnitude : 1
    // Same split as the readout: head side is within 67.5° of the nose, tail side of the tail.
    const angle = (Math.atan2(point.cross, point.head) * 180) / Math.PI
    const tone = Math.abs(angle) <= 67.5 ? 'head' : Math.abs(angle) >= 112.5 ? 'tail' : 'cross'
    f.add(
      cloud,
      f.svg('circle', {
        class: `tri-environment-rose-dot tri-environment-rose-dot--${tone}`,
        cx: (point.cross * clamp * unitRadius).toFixed(3),
        cy: (-point.head * clamp * unitRadius).toFixed(3),
        r: 0.9,
        'data-environment-sample-index': point.index,
      }),
    )
  }
  f.add(svg, cloud)
  if (points.length > 0)
    f.add(
      svg,
      f.svg('line', {
        class: 'tri-environment-rose-mean',
        x1: 0,
        y1: 0,
        x2: (meanCross * unitRadius).toFixed(3),
        y2: (-meanHead * unitRadius).toFixed(3),
      }),
      f.svg('circle', {
        class: 'tri-environment-rose-mean-tip',
        cx: (meanCross * unitRadius).toFixed(3),
        cy: (-meanHead * unitRadius).toFixed(3),
        r: 1.3,
      }),
    )
  if (points.length > 0)
    f.add(
      svg,
      f.svg('circle', {
        class: 'tri-environment-rose-cursor',
        cx: 0,
        cy: 0,
        r: 2.2,
        visibility: 'hidden',
      }),
    )
  const labels = [
    ['head', [7, 0, 1], 'top'],
    ['right', [2], 'right'],
    ['tail', [3, 4, 5], 'bottom'],
    ['left', [6], 'left'],
  ] as const
  const frame = f.el('div', 'tri-environment-rose-frame')
  f.add(frame, svg, ...ringLabels)
  for (const [label, indices, side] of labels)
    f.add(
      frame,
      f.el(
        'span',
        `tri-environment-rose-label tri-environment-rose-label--${side}`,
        `${triText(f.presentation.locale, label)} ${Math.round(indices.reduce<number>((total, index) => total + (shares[index] ?? 0), 0))}%`,
      ),
    )
  f.add(rose, frame)
  if (mean) f.add(rose, f.el('span', 'tri-environment-rose-caption', mean))
  return rose
}
