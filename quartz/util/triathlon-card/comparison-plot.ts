import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { niceStep, type AxisXTick } from './chart-geometry'
import { dlabel } from './metric-text'
import { powerCurveDurationTicks, powerCurveFraction } from './power-curve-data'

export type ActivityComparisonMetricPoint = { distanceKm: number; value: number }

export type ActivityComparisonSeries = {
  activity: StravaActivityDetail
  index: number
  segments: ActivityComparisonMetricPoint[][]
  values: number[]
}

export type ActivityComparisonDomain = { min: number; max: number }

export const ACTIVITY_COMPARISON_WIDTH = 100
export const ACTIVITY_COMPARISON_HEIGHT = 34

// Series hues mirror the theme tokens (--tertiary peach, --secondary sage), then Flexoki
// blue-200 and magenta-200. They stay hex because the route overlay is a live mapbox map,
// which cannot parse oklch or CSS variables.
const COMPARE_SERIES = ['#fcc192', '#cdd597', '#abcfe2', '#f4a4c2'] as const
// Light paper needs darker ink for 3:1 marks; comparison-charts.scss uses the same oklab mix.
export const ACTIVITY_COMPARE_LIGHT_INK = 0.58
const LIGHT_INK_HEX = '#100f0f'

type Rgb = [number, number, number]

const hexLinear = (hex: string): Rgb => {
  const channel = (offset: number): number => {
    const encoded = Number.parseInt(hex.slice(offset, offset + 2), 16) / 255
    return encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4
  }
  return [channel(1), channel(3), channel(5)]
}

const linearOklab = ([r, g, b]: Rgb): Rgb => {
  const long = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const medium = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const short = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * long + 0.793617785 * medium - 0.0040720468 * short,
    1.9779984951 * long - 2.428592205 * medium + 0.4505937099 * short,
    0.0259040371 * long + 0.7827717662 * medium - 0.808675766 * short,
  ]
}

const oklabLinear = ([lightness, a, b]: Rgb): Rgb => {
  const long = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const medium = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const short = (lightness - 0.0894841775 * a - 1.291485548 * b) ** 3
  return [
    4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short,
    -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short,
    -0.0041960863 * long - 0.7034186147 * medium + 1.707614701 * short,
  ]
}

const srgbHex = (channels: readonly number[]): string =>
  `#${channels
    .map(channel => {
      const clamped = Math.min(1, Math.max(0, channel))
      const encoded = clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * clamped ** (1 / 2.4) - 0.055
      return Math.round(encoded * 255)
        .toString(16)
        .padStart(2, '0')
    })
    .join('')}`

/** Series colour for the activity at `index`; `light` returns the ink drawn on light paper. */
export const activityCompareColor = (index: number, theme: 'light' | 'dark' = 'dark'): string => {
  const position = Number.isFinite(index) ? Math.max(0, Math.trunc(index)) : 0
  const hex = COMPARE_SERIES[position % COMPARE_SERIES.length]
  if (theme === 'dark') return hex
  const series = linearOklab(hexLinear(hex))
  const ink = linearOklab(hexLinear(LIGHT_INK_HEX))
  const mixed = series.map(
    (value, axis) =>
      value * ACTIVITY_COMPARE_LIGHT_INK + ink[axis] * (1 - ACTIVITY_COMPARE_LIGHT_INK),
  ) as Rgb
  return srgbHex(oklabLinear(mixed))
}

export const comparisonNumericDomain = (
  values: number[],
  includeZero: boolean,
): ActivityComparisonDomain => {
  if (values.length === 0) return { min: 0, max: 1 }
  let observedMin = values[0]
  let observedMax = values[0]
  for (let index = 1; index < values.length; index++) {
    observedMin = Math.min(observedMin, values[index])
    observedMax = Math.max(observedMax, values[index])
  }
  const lower = includeZero ? Math.min(0, observedMin) : observedMin
  const span = observedMax - lower
  const step = niceStep(span > 0 ? span : Math.max(1, Math.abs(observedMax)), 3)
  const min = Math.floor(lower / step) * step
  let max = Math.ceil(observedMax / step) * step
  if (max <= min) max = min + step
  return { min, max }
}

export const comparisonSeriesPath = (
  segments: { distanceKm: number; value: number }[][],
  maxDistanceKm: number,
  domain: ActivityComparisonDomain,
): string => {
  const distanceSpan = Math.max(1e-6, maxDistanceKm)
  const valueSpan = Math.max(1e-6, domain.max - domain.min)
  return segments
    .map(segment =>
      segment
        .map((point, index) => {
          const x =
            Math.min(1, Math.max(0, point.distanceKm / distanceSpan)) * ACTIVITY_COMPARISON_WIDTH
          const y =
            ACTIVITY_COMPARISON_HEIGHT -
            ((point.value - domain.min) / valueSpan) * (ACTIVITY_COMPARISON_HEIGHT - 1)
          return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
        })
        .join(' '),
    )
    .filter(path => path.length > 0)
    .join(' ')
}

export const comparisonReadout = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
): N => {
  const readout = f.el('div', 'tri-compare-readout', undefined, {
    'data-compare-readout': '',
    'data-visible': 'false',
    'aria-hidden': 'true',
  })
  f.add(
    readout,
    f.el('span', 'tri-compare-readout-context', undefined, {
      'data-compare-readout-context': '',
      hidden: '',
    }),
  )
  for (const [index, activity] of activities.entries()) {
    const row = f.el('div', 'tri-compare-readout-row', undefined, {
      'data-activity-id': `${activity.id}`,
      'data-activity-index': `${index}`,
      style: `--tri-compare-color:${activityCompareColor(index)}`,
    })
    f.add(
      row,
      f.el('span', 'tri-compare-readout-swatch', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-compare-readout-name', activity.name || activity.sport, {
        title: activity.name || activity.sport,
      }),
      f.el('strong', 'tri-compare-readout-value', undefined, { 'data-compare-readout-value': '' }),
    )
    f.add(readout, row)
  }
  return readout
}

export const comparisonChartHead = <N>(
  f: TriNodeFactory<N>,
  title: string,
  controls?: N,
  readout = controls === undefined,
): N => {
  // A readout head holds the shared hover readout on its title row; with controls, the
  // readout covers them while it shows.
  const head = f.el(
    'div',
    [
      'tri-compare-chart-head',
      ...(readout ? ['tri-compare-chart-head--readout'] : []),
      ...(readout && controls !== undefined ? ['tri-compare-chart-head--controls'] : []),
    ].join(' '),
    undefined,
    readout && controls !== undefined
      ? { style: `--tri-compare-title-width:${title.length}ch` }
      : {},
  )
  f.add(
    head,
    f.el('div', 'tri-compare-title', title, { 'data-i18n': title }),
    ...(controls === undefined ? [] : [controls]),
  )
  return head
}

export const comparisonGraphAttrs = (
  chart: string,
  label: string,
  valueMin: number,
  valueMax: number,
  valueNow: number,
  valueText: string,
  available: number,
): Record<string, string | number> => {
  const attrs: Record<string, string | number> = {
    class: 'tri-compare-graph',
    viewBox: `0 0 ${ACTIVITY_COMPARISON_WIDTH} ${ACTIVITY_COMPARISON_HEIGHT}`,
    preserveAspectRatio: 'none',
    'data-compare-chart': chart,
    'aria-label': label,
    'data-i18n-aria-label': label,
  }
  if (available <= 0) {
    attrs.role = 'img'
    attrs['aria-disabled'] = 'true'
    return attrs
  }
  attrs.role = 'slider'
  attrs.tabindex = 0
  attrs['aria-orientation'] = 'horizontal'
  attrs['aria-valuemin'] = valueMin
  attrs['aria-valuemax'] = valueMax
  attrs['aria-valuenow'] = valueNow
  attrs['aria-valuetext'] = valueText
  return attrs
}

export const addComparisonSelectionRegion = <N>(
  f: TriNodeFactory<N>,
  graph: N,
  clipId: string,
): void => {
  const clip = f.svg('clipPath', { id: clipId })
  f.add(
    clip,
    f.svg('rect', {
      class: 'tri-compare-selection-clip',
      x: 0,
      y: 0,
      width: 0,
      height: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const defs = f.svg('defs', {})
  f.add(defs, clip)
  f.add(
    graph,
    defs,
    f.svg('rect', {
      class: 'tri-compare-selection-region',
      x: 0,
      y: 0,
      width: 0,
      height: ACTIVITY_COMPARISON_HEIGHT,
      'aria-hidden': 'true',
    }),
  )
}

export const comparisonDurationTicks = (
  minSeconds: number,
  maxSeconds: number,
  selectedSeconds: number,
): AxisXTick[] => {
  const durations = powerCurveDurationTicks(minSeconds, maxSeconds, [
    minSeconds,
    1,
    60,
    300,
    1_200,
    3_600,
    10_800,
  ])
  return durations.map((seconds, index) => ({
    label: dlabel(seconds),
    pct: powerCurveFraction(seconds, minSeconds, maxSeconds) * 100,
    cls: `tri-curve-tick${index === 0 ? ' tri-cax-xt--first' : index === durations.length - 1 ? ' tri-cax-xt--last' : ''}`,
    tag: 'button',
    attrs: {
      type: 'button',
      'data-curve-seconds': String(seconds),
      'aria-pressed': String(seconds === selectedSeconds),
    },
  }))
}

export const comparisonPowerDistributionTicks = (binCount: number): AxisXTick[] => {
  const maxWatts = Math.max(25, (binCount - 1) * 25)
  const stepWatts = maxWatts <= 300 ? 100 : maxWatts <= 700 ? 200 : 300
  const ticks: AxisXTick[] = []
  for (let watts = 0; watts <= maxWatts; watts += stepWatts)
    ticks.push({
      label: `${watts} W`,
      pct: (watts / maxWatts) * 100,
      cls: watts === 0 ? 'tri-cax-xt--first' : watts === maxWatts ? 'tri-cax-xt--last' : undefined,
    })
  return ticks
}
