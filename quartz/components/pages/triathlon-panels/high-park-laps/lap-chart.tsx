import type { HighParkLap } from '../../../../util/high-park-loops'
import { PICK_SLOTS } from '../../../../util/high-park-lap-compare'
import { clock, lapSpeed, paceOf, r1, type HighParkUnits, type LapSport } from './fields'

type Value = (lap: HighParkLap) => number | null
export interface LapMetric {
  key: string
  label: string
  unit: string
  /** Second series, drawn dashed (the right leg). */
  values: [Value] | [Value, Value]
  format?: (value: number) => string
  /** Lower is better: the axis flips so better is always up. */
  lowerIsBetter?: boolean
  /** The caption's range reads high to low (pace: the larger speed is the smaller pace). */
  descending?: boolean
}

// Lap-by-lap panels, in reading order. A panel needs two laps with a value.
export const lapMetrics = (units: HighParkUnits, sport: LapSport): LapMetric[] => [
  { key: 'time', label: 'time', unit: '', values: [l => l.s], format: clock, lowerIsBetter: true },
  sport === 'run'
    ? {
        key: 'speed',
        label: 'pace',
        unit: `/${units.distance}`,
        values: [l => lapSpeed(l, units)],
        format: paceOf,
        descending: true,
      }
    : {
        key: 'speed',
        label: 'speed',
        unit: units.speed,
        values: [l => lapSpeed(l, units)],
        format: v => v.toFixed(1),
      },
  { key: 'power', label: 'power', unit: 'W', values: [l => l.w] },
  { key: 'hr', label: 'heart rate', unit: 'bpm', values: [l => l.hr] },
  { key: 'cad', label: 'cadence', unit: sport === 'run' ? 'spm' : 'rpm', values: [l => l.cad] },
  {
    key: 'torque',
    label: 'torque',
    unit: units.torque,
    values: [l => l.nm && units.nm(l.nm)],
    format: v => v.toFixed(1),
  },
  { key: 'te', label: 'torque effectiveness L/R', unit: '%', values: [l => l.teL, l => l.teR] },
  { key: 'ps', label: 'pedal smoothness L/R', unit: '%', values: [l => l.psL, l => l.psR] },
  { key: 'bal', label: 'balance, left', unit: '%', values: [l => l.balL] },
  { key: 'shifts', label: 'shifts', unit: '', values: [l => l.shifts] },
  {
    key: 'resp',
    label: 'breathing',
    unit: 'br/min',
    values: [l => l.resp],
    format: v => v.toFixed(1),
  },
  {
    key: 'core',
    label: 'core temp',
    unit: units.temp,
    values: [l => l.core && units.celsius(l.core)],
    format: v => v.toFixed(2),
  },
]

/** Lap-by-lap chart viewBox: width, height, inset. */
export const LAP_CHART = { w: 200, h: 54, pad: 6 } as const

export const lapChartX = (lap: number, laps: number): number =>
  r1(LAP_CHART.pad + (lap / Math.max(1, laps - 1)) * (LAP_CHART.w - 2 * LAP_CHART.pad))

/** A panel's series and scale; null when fewer than two laps carry the metric. */
export const lapChartScale = (metric: LapMetric, laps: readonly HighParkLap[]) => {
  const series = metric.values.map(value => laps.map(value))
  const known = series.flat().filter((v): v is number => v !== null)
  if (series[0].filter(v => v !== null).length < 2) return null
  const [lo, hi] = [Math.min(...known), Math.max(...known)]
  const span = hi - lo || 1
  const format = metric.format ?? ((v: number) => String(Math.round(v)))
  const y = (v: number) => {
    const t = (v - lo) / span
    return r1(
      LAP_CHART.pad + (metric.lowerIsBetter ? t : 1 - t) * (LAP_CHART.h - 2 * LAP_CHART.pad),
    )
  }
  /**
   * One lap's point per series, in percent of the plot box. Pins are HTML boxes: WebKit misplaces
   * zero-length non-scaling strokes whose `d` is written after first paint.
   */
  const pointsAt = (lap: number): ([number, number] | null)[] =>
    series.map(values =>
      values[lap] == null
        ? null
        : [
            r1((lapChartX(lap, laps.length) / LAP_CHART.w) * 100),
            r1((y(values[lap]) / LAP_CHART.h) * 100),
          ],
    )
  /** One lap's value as the caption reads it: 235 W, 71/70 %. */
  const read = (lap: number): string => {
    const values = series.map(row => row[lap])
    if (values.every(v => v == null)) return '–'
    const text = values.map(v => (v == null ? '–' : format(v))).join('/')
    return metric.unit ? `${text} ${metric.unit}` : text
  }
  return { series, known, lo, hi, y, format, pointsAt, read }
}

export const LapChart = ({
  metric,
  laps,
  best,
}: {
  metric: LapMetric
  laps: HighParkLap[]
  best: number
}) => {
  const { w: CW, h: CH, pad: CPAD } = LAP_CHART
  const scale = lapChartScale(metric, laps)
  if (!scale) return null
  const { series, known, lo, hi, y, format } = scale
  const x = (i: number) => lapChartX(i, laps.length)
  const line = (values: (number | null)[]) => {
    let d = ''
    let open = false
    values.forEach((v, i) => {
      if (v === null) return void (open = false)
      d += `${open ? 'L' : 'M'}${x(i)} ${y(v)}`
      open = true
    })
    return d
  }
  const marks = (values: (number | null)[]) =>
    values.map((v, i) => (v === null || i === best ? '' : `M${x(i)} ${y(v)}h0`)).join('')
  const mean = known.reduce((a, b) => a + b, 0) / known.length
  const bestValue = series[0][best]
  const [from, to] = metric.descending ? [hi, lo] : [lo, hi]
  const range = `${format(from)}–${format(to)}${metric.unit && ` ${metric.unit}`}`
  return (
    <figure class="tri-hp-lapchart" data-metric={metric.key}>
      <figcaption>
        <span>{metric.label}</span>
        {/* The client swaps this for the hovered lap's value. */}
        <span class="tri-hp-range" data-idle={range}>
          {range}
        </span>
      </figcaption>
      <div class="tri-hp-plot">
        <svg viewBox={`0 0 ${CW} ${CH}`} preserveAspectRatio="none" aria-hidden="true">
          <path class="tri-hp-lapchart-mean" d={`M${CPAD} ${y(mean)}H${CW - CPAD}`} />
          {series.map((values, s) => (
            <g key={s} class={s ? 'is-right' : undefined}>
              <path class="tri-hp-lapchart-line" d={line(values)} />
              <path class="tri-hp-lapchart-marks" d={marks(values)} />
            </g>
          ))}
          {bestValue !== null && (
            <path class="tri-hp-lapchart-best" d={`M${x(best)} ${y(bestValue)}h0`} />
          )}
          <path class="tri-hp-cursor" d="" />
        </svg>
        {Array.from({ length: PICK_SLOTS }, (_, slot) =>
          series.map((_, s) => (
            <i key={`${slot}-${s}`} class="tri-hp-pin" data-who={slot} style="display:none" />
          )),
        )}
      </div>
    </figure>
  )
}
