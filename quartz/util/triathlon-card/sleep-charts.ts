import type { SleepMetrics } from '../sleep-metrics'
import type { TriNodeFactory } from './construction-contracts'
import { sleepSeriesFromSamples, type SleepSeries } from '../triathlon-day-sleep'
import { healthTooltip } from '../triathlon-health'
import { triText } from '../triathlon-i18n'
import { axisFrame, buildPowerCurveGrid } from './chart-geometry'
import { dayAnalyticsNumber } from './day-analytics-metrics'
import {
  DAY_ANALYTICS_SLEEP_HEIGHT,
  DAY_ANALYTICS_SLEEP_WIDTH,
  dayAnalyticsHourTicks,
  dayAnalyticsSeriesPaths,
  dayAnalyticsWallClock,
  dayAnalyticsWallMinute,
  daySleepNativeTicks,
  daySleepSampleClock,
  SLEEP_RESPIRATION_MAX_GAP_S,
} from './sleep-chart-data'

export const dayAnalyticsSleepSeries = <N>(
  f: TriNodeFactory<N>,
  date: string,
  key: 'hrv' | 'heart-rate' | 'respiration',
  title: string,
  series: SleepSeries | null,
  source?: string,
): N | null => {
  if (!series) return null
  const offsetsS = 'offsetsS' in series ? series.offsetsS : undefined
  const values = series.items.filter((value): value is number => value != null)
  if (values.length < (offsetsS ? 1 : 2)) return null
  const low = Math.min(...values)
  const high = Math.max(...values)
  const padding = Math.max((high - low) * 0.1, 1)
  const minimum = low - padding
  const maximum = high + padding
  const preciseClock = offsetsS && key !== 'respiration'
  const elapsedS = offsetsS?.at(-1) ?? 0
  const x = (index: number): number =>
    (offsetsS
      ? offsetsS[index] / Math.max(1, elapsedS)
      : index / Math.max(1, series.items.length - 1)) * DAY_ANALYTICS_SLEEP_WIDTH
  const y = (value: number): number =>
    DAY_ANALYTICS_SLEEP_HEIGHT -
    2 -
    ((value - minimum) / Math.max(1, maximum - minimum)) * (DAY_ANALYTICS_SLEEP_HEIGHT - 4)
  const average = values.reduce((sum, value) => sum + value, 0) / values.length
  const unit = key === 'hrv' ? 'ms' : key === 'respiration' ? 'brpm' : 'bpm'
  const readoutId = `tri-day-${date}-sleep-${key}-readout`
  const initialIndex = series.items.findLastIndex(value => value != null)
  const startMinute =
    (dayAnalyticsWallMinute(series.startTs) ?? 0) +
    (offsetsS ? Number(series.startTs.slice(17, 19)) / 60 : 0)
  const initialMinute =
    startMinute +
    ('offsetsS' in series
      ? series.offsetsS[Math.max(0, initialIndex)]
      : Math.max(0, initialIndex) * series.intervalS) /
      60
  const initialTime = preciseClock
    ? daySleepSampleClock(initialMinute, 1)
    : dayAnalyticsWallClock(offsetsS ? Math.floor(initialMinute) : initialMinute)
  const initialValue = initialIndex >= 0 ? series.items[initialIndex] : null
  const initialReadout = `${initialTime} · ${initialValue == null ? '—' : Math.round(initialValue)} ${unit}`
  const chart = f.el(
    'section',
    `tri-day-sleep-chart tri-day-sleep-series tri-day-sleep-series--${key}`,
    undefined,
    {
      'data-day-sleep-series': key,
      'data-day-sleep-values': series.items.map(value => value ?? '').join(','),
      'data-day-sleep-start': series.startTs,
      ...('offsetsS' in series
        ? { 'data-day-sleep-times': series.offsetsS.join(',') }
        : { 'data-day-sleep-interval': series.intervalS.toString() }),
      ...(preciseClock ? { 'data-day-sleep-precise': '' } : {}),
      ...(source ? { 'data-day-sleep-source': source } : {}),
      'data-day-sleep-unit': unit,
      'data-day-sleep-width': DAY_ANALYTICS_SLEEP_WIDTH.toString(),
    },
  )
  const heading = f.el(
    'h4',
    source ? 'tri-ana-block-title tri-health-tooltip-trigger' : 'tri-ana-block-title',
    undefined,
    source ? { tabindex: '0', 'aria-describedby': `${readoutId}-recording` } : undefined,
  )
  f.add(
    heading,
    f.el('span', undefined, triText(f.presentation.locale, title), { 'data-i18n': title }),
  )
  if (source) f.add(heading, healthTooltip(f, `${readoutId}-recording`, source))
  f.add(chart, heading)
  const svg = f.svg('svg', {
    class: 'tri-ana-svg tri-day-sleep-line-svg',
    viewBox: `0 0 ${DAY_ANALYTICS_SLEEP_WIDTH} ${DAY_ANALYTICS_SLEEP_HEIGHT}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': triText(f.presentation.locale, title),
    'aria-valuemin': 0,
    'aria-valuemax': series.items.length - 1,
    'aria-valuenow': Math.max(0, initialIndex),
    'aria-valuetext': initialReadout,
    'aria-describedby': readoutId,
  })
  const xTicks =
    'offsetsS' in series
      ? daySleepNativeTicks(startMinute, elapsedS)
      : dayAnalyticsHourTicks(
          series.startTs,
          series.intervalS,
          series.items.length,
          series.items.length - 1,
        )
  f.add(
    svg,
    buildPowerCurveGrid(
      f,
      'power',
      high === low ? [y(high)] : [y(high), y(low)],
      DAY_ANALYTICS_SLEEP_HEIGHT,
    ),
    buildPowerCurveGrid(
      f,
      'duration',
      xTicks.map(tick => tick.pct),
      DAY_ANALYTICS_SLEEP_HEIGHT,
    ),
    f.svg('line', {
      x1: 0,
      y1: y(average),
      x2: DAY_ANALYTICS_SLEEP_WIDTH,
      y2: y(average),
      class: 'tri-rec-target',
    }),
  )
  for (const path of dayAnalyticsSeriesPaths(series.items, x, y, offsetsS))
    f.add(svg, f.svg('path', { d: path, class: `tri-day-sleep-line tri-day-sleep-line--${key}` }))
  if (offsetsS)
    for (const [index, value] of series.items.entries()) {
      const previousConnected =
        index > 0 &&
        series.items[index - 1] != null &&
        offsetsS[index] - offsetsS[index - 1] <= SLEEP_RESPIRATION_MAX_GAP_S
      const nextConnected =
        index + 1 < series.items.length &&
        series.items[index + 1] != null &&
        offsetsS[index + 1] - offsetsS[index] <= SLEEP_RESPIRATION_MAX_GAP_S
      if (value != null && !previousConnected && !nextConnected)
        f.add(
          svg,
          f.svg('path', {
            d: `M${x(index).toFixed(3)} ${y(value).toFixed(3)}l0 0`,
            class: `tri-day-sleep-line tri-day-sleep-line--${key}`,
            'stroke-linecap': 'round',
          }),
        )
    }
  f.add(
    svg,
    f.svg('line', {
      x1: x(Math.max(0, initialIndex)),
      y1: 0,
      x2: x(Math.max(0, initialIndex)),
      y2: DAY_ANALYTICS_SLEEP_HEIGHT,
      class: 'tri-ana-cursor',
    }),
  )
  f.add(
    chart,
    axisFrame(
      f,
      svg,
      (high === low ? [high] : [high, low]).map(value => ({
        label: dayAnalyticsNumber(f.presentation, value),
        vbY: y(value),
      })),
      DAY_ANALYTICS_SLEEP_HEIGHT,
      xTicks,
    ),
    f.el('div', 'tri-chart-readout', initialReadout, { id: readoutId }),
  )
  return chart
}

export const buildSleepRespirationChart = <N>(
  f: TriNodeFactory<N>,
  date: string,
  sleep: SleepMetrics | null | undefined,
): N | null => {
  const garmin = sleep?.garmin
  if (!garmin) return null
  return dayAnalyticsSleepSeries(
    f,
    date,
    'respiration',
    'respiration',
    sleepSeriesFromSamples(
      garmin.respiration,
      sample => sample.breathsPerMinute,
      garmin.utcOffsetMinutes,
    ),
    garmin.utcOffsetMinutes == null ? 'Garmin (UTC)' : 'Garmin',
  )
}
