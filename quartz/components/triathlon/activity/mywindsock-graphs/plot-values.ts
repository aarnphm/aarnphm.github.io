import type { MyWindsockGraphSeries, MyWindsockGraphPlot } from '../../../../util/mywindsock-graphs'
export const number = (value: number, maximumFractionDigits = 3): string =>
  value.toLocaleString('en-US', { maximumFractionDigits }).replace('-', '\u2212')

export const extent = (values: number[]): [number, number] => {
  let lo = Infinity
  let hi = -Infinity
  for (const value of values) {
    lo = Math.min(lo, value)
    hi = Math.max(hi, value)
  }
  return Number.isFinite(lo) ? [lo, hi === lo ? lo + 1 : hi] : [0, 1]
}

export const clock = (milliseconds: number): string => {
  const seconds = Math.max(0, Math.round(milliseconds / 1000))
  return `${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

export const yDomain = (
  plot: MyWindsockGraphPlot,
  series: MyWindsockGraphSeries[],
): [number, number] => {
  const axis = plot.axes[series[0]?.axis ?? 'scale-y']
  const values = series.flatMap(series =>
    series.points.flatMap(point =>
      [point.y, point.upper].flatMap(value => (value == null ? [] : [value])),
    ),
  )
  const [low, high] = extent(values)
  // Native axis limits only widen the domain: Feels Like elevation drops below the provider floor.
  const minimum = Math.min(
    low,
    axis?.minimum ?? (series.some(series => series.kind === 'bar') ? Math.min(0, low) : low),
  )
  const maximum = Math.max(high, axis?.maximum ?? high)
  return maximum > minimum ? [minimum, maximum] : [minimum, minimum + 1]
}
