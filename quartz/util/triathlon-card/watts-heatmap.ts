import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { axisNumber, niceStep, type AxisXTick } from './chart-geometry'
import { powerBalanceText } from './metric-text'

// numberLabel is the gear-number form of label, for the shifting chart's 'show gears as' setting.
export type WattsHeatSample = { watts: number; value: number; label?: string; numberLabel?: string }
export interface CyclingWattsCell {
  count: number
  minWatts: number
  maxWatts: number
  value: number
  label: string
  side: 'left' | 'right' | 'single'
}
type WattsHeatCell = Omit<CyclingWattsCell, 'side'> & {
  numberLabel: string
  x: number
  y: number
  width: number
  height: number
}
export type CyclingModeTitle =
  | 'power balance'
  | 'torque effectiveness'
  | 'pedal smoothness'
  | 'electronic shifting'

const WATTS_HEAT_X_BINS = 64
const WATTS_HEAT_Y_BINS = 24

export const cyclingWattsCellText = (
  cell: CyclingWattsCell,
  powerBalance: boolean,
  locale: TriathlonPresentation['locale'],
): string => {
  const watts = (value: number): string => Number(value.toFixed(1)).toLocaleString(locale)
  const metric =
    cell.label || (powerBalance ? powerBalanceText(cell.value) : `${cell.value.toFixed(1)}%`)
  const side = cell.side === 'single' ? '' : `${triText(locale, cell.side)} `
  const average = cell.count > 1 && !cell.label ? ` ${triText(locale, 'avg')}` : ''
  return `${watts(cell.minWatts)}–${watts(cell.maxWatts)} W · ${side}${metric}${average} · ${cell.count} ${triText(locale, cell.count === 1 ? 'sample' : 'samples')}`
}

const wattsAxis = (
  d: StravaActivityDetail,
  samples: readonly { watts: number }[],
): { max: number; ticks: AxisXTick[] } => {
  const observedMax = Math.max(d.maxWatts ?? 0, ...samples.map(sample => sample.watts))
  const step = niceStep(observedMax, 5)
  const max = Math.max(step, Math.ceil(observedMax / step) * step)
  const ticks: AxisXTick[] = []
  for (let watts = 0; watts <= max + step * 1e-6; watts += step) {
    ticks.push({
      label: `${axisNumber(watts, step)} W`,
      pct: (watts / max) * 100,
      ...(watts === 0
        ? { cls: 'tri-cax-xt--first' }
        : watts >= max
          ? { cls: 'tri-cax-xt--last' }
          : {}),
    })
  }
  return { max, ticks }
}

const wattsHeatCells = (
  samples: readonly WattsHeatSample[],
  maxWatts: number,
  domainMin: number,
  domainMax: number,
  height: number,
  invertY: boolean,
): WattsHeatCell[] => {
  const counts = new Map<
    number,
    { count: number; totalValue: number; labels: Set<string>; numberLabels: Set<string> }
  >()
  for (const sample of samples) {
    const xBin = Math.min(
      WATTS_HEAT_X_BINS - 1,
      Math.max(0, Math.floor((sample.watts / maxWatts) * WATTS_HEAT_X_BINS)),
    )
    const normalized = (sample.value - domainMin) / Math.max(1e-6, domainMax - domainMin)
    const valueBin = Math.min(
      WATTS_HEAT_Y_BINS - 1,
      Math.max(0, Math.floor(normalized * WATTS_HEAT_Y_BINS)),
    )
    const yBin = invertY ? WATTS_HEAT_Y_BINS - 1 - valueBin : valueBin
    const key = yBin * WATTS_HEAT_X_BINS + xBin
    const cell = counts.get(key) ?? {
      count: 0,
      totalValue: 0,
      labels: new Set<string>(),
      numberLabels: new Set<string>(),
    }
    cell.count++
    cell.totalValue += sample.value
    if (sample.label) cell.labels.add(sample.label)
    if (sample.numberLabel) cell.numberLabels.add(sample.numberLabel)
    counts.set(key, cell)
  }
  const xSize = 100 / WATTS_HEAT_X_BINS
  const ySize = height / WATTS_HEAT_Y_BINS
  return Array.from(counts, ([key, cell]) => {
    const xBin = key % WATTS_HEAT_X_BINS
    const yBin = Math.floor(key / WATTS_HEAT_X_BINS)
    return {
      count: cell.count,
      minWatts: (xBin / WATTS_HEAT_X_BINS) * maxWatts,
      maxWatts: ((xBin + 1) / WATTS_HEAT_X_BINS) * maxWatts,
      value: cell.totalValue / cell.count,
      label: [...cell.labels].join(' / '),
      numberLabel: [...cell.numberLabels].join(' / '),
      x: xBin * xSize,
      y: yBin * ySize,
      width: xSize,
      height: ySize,
    }
  }).sort((left, right) => left.y - right.y || left.x - right.x)
}

export const buildWattsHeatmap = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  title: CyclingModeTitle,
  className: string,
  dataName: string,
  series: readonly { side: 'left' | 'right' | 'single'; samples: readonly WattsHeatSample[] }[],
  domainMin: number,
  domainMax: number,
  height: number,
  invertY = true,
): { svg: N; wattsTicks: AxisXTick[]; readout: N } => {
  const allSamples = series.flatMap(item => item.samples)
  const axis = wattsAxis(d, allSamples)
  const layers = series.map(item => ({
    side: item.side,
    cells: wattsHeatCells(item.samples, axis.max, domainMin, domainMax, height, invertY),
  }))
  const cellCount = layers.reduce((total, layer) => total + layer.cells.length, 0)
  const svg = f.svg('svg', {
    class: `tri-cycling-watts-heatmap ${className}`,
    viewBox: `0 0 100 ${height}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: '0',
    'aria-valuemin': 0,
    'aria-valuemax': Math.max(0, cellCount - 1),
    'aria-valuenow': 0,
    'aria-label': triText(f.presentation.locale, `${title} by watts`),
    'data-i18n-aria-label': `${title} by watts`,
    [`data-${dataName}-samples`]: allSamples.length,
    [`data-${dataName}-max-watts`]: axis.max,
  })
  const splitSides =
    series.some(item => item.side === 'left') && series.some(item => item.side === 'right')
  for (const { side, cells } of layers) {
    const maxCellCount = Math.max(1, ...cells.map(cell => cell.count))
    for (const cell of cells) {
      const density = 0.16 + 0.84 * Math.sqrt(cell.count / maxCellCount)
      const width = splitSides ? cell.width / 2 : cell.width
      const x = cell.x + (splitSides && side === 'right' ? width : 0)
      f.add(
        svg,
        f.svg('rect', {
          class: `tri-cycling-watts-heat-cell tri-cycling-watts-heat-cell--${side}`,
          x: x.toFixed(3),
          y: cell.y.toFixed(3),
          width: width.toFixed(3),
          height: cell.height.toFixed(3),
          style: `--tri-cycling-watts-density:${density.toFixed(3)}`,
          'data-samples': cell.count,
          'data-watts-min': cell.minWatts,
          'data-watts-max': cell.maxWatts,
          'data-value': cell.value,
          'data-value-label': cell.label,
          ...(cell.numberLabel ? { 'data-value-label-number': cell.numberLabel } : {}),
          'data-side': side,
          'aria-label': cyclingWattsCellText(
            { ...cell, side },
            title === 'power balance',
            f.presentation.locale,
          ),
        }),
      )
    }
  }
  return {
    svg,
    wattsTicks: axis.ticks,
    readout: f.el('span', 'tri-chart-readout tri-cycling-watts-readout', undefined, { hidden: '' }),
  }
}

export const buildCyclingChartModes = <N>(f: TriNodeFactory<N>): N => {
  const modes = f.el('span', 'tri-cycling-chart-modes', undefined, {
    role: 'group',
    'aria-label': triText(f.presentation.locale, 'cycling charts view'),
    'data-i18n-aria-label': 'cycling charts view',
  })
  for (const [mode, label] of [
    ['distance', 'distance'],
    ['power', 'watts'],
  ] as const)
    f.add(
      modes,
      f.el('button', 'tri-cycling-chart-mode tri-curve-range', label, {
        type: 'button',
        'data-cycling-chart-mode': mode,
        'aria-pressed': String(mode === 'distance'),
        'data-i18n': label,
      }),
    )
  return modes
}
