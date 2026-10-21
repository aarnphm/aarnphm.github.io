import type { SleepMetrics } from '../sleep-metrics'
import type { TriathlonDayAnalytics } from '../triathlon-day-analytics'
import type { TriNodeFactory } from './construction-contracts'
import { healthTooltip } from '../triathlon-health'
import { triText } from '../triathlon-i18n'
import { axisFrame, buildPowerCurveGrid, type AxisXTick } from './chart-geometry'
import { dayAnalyticsDuration, dayAnalyticsNumber } from './day-analytics-metrics'
import {
  DAY_ANALYTICS_MOVEMENT_LANES,
  DAY_ANALYTICS_SLEEP_HEIGHT,
  DAY_ANALYTICS_SLEEP_LANE_BY_CODE,
  DAY_ANALYTICS_SLEEP_LANES,
  DAY_ANALYTICS_SLEEP_WIDTH,
  dayAnalyticsHourTicks,
  dayAnalyticsSeriesPaths,
  dayAnalyticsWallClock,
  dayAnalyticsWallMinute,
  daySleepMovementLabel,
  daySleepSampleClock,
  daySleepStageLabel,
  SLEEP_RESPIRATION_MAX_GAP_S,
  type SleepChartSeries,
} from './sleep-chart-data'

export const dayAnalyticsSleepStages = <N>(
  f: TriNodeFactory<N>,
  date: string,
  sleep: NonNullable<TriathlonDayAnalytics['sleep']>,
  kind: 'stages' | 'movement' = 'stages',
): N | null => {
  const movement = kind === 'movement'
  const title = movement ? 'sleep movement' : 'sleep stages'
  const labels = movement ? DAY_ANALYTICS_MOVEMENT_LANES : DAY_ANALYTICS_SLEEP_LANES
  const labelAt = movement ? daySleepMovementLabel : daySleepStageLabel
  const finePhase = sleep.phase30Sec && /[1-4]/.test(sleep.phase30Sec) ? sleep.phase30Sec : null
  const phase = movement ? sleep.movement30Sec : (finePhase ?? sleep.phase5Min)
  const interval = movement || finePhase ? 30 : 300
  const durations: readonly (readonly [string, number | null])[] = movement
    ? DAY_ANALYTICS_MOVEMENT_LANES.map((label, lane) => [
        label,
        phase ? Array.from(phase).filter(code => Number(code) === lane + 1).length * 30 : null,
      ])
    : [
        ['deep', sleep.deepS],
        ['light', sleep.lightS],
        ['rem', sleep.remS],
        ['awake', sleep.awakeS],
      ]
  if (!phase && durations.every(([, seconds]) => seconds == null)) return null
  const bedtimeStart = sleep.bedtimeStart
  const wallMinute = bedtimeStart ? dayAnalyticsWallMinute(bedtimeStart) : null
  const startMinute =
    wallMinute == null
      ? null
      : wallMinute + (interval === 30 ? Number(bedtimeStart?.slice(17, 19) || 0) / 60 : 0)
  const hypnogram =
    phase && phase.length >= 2 && bedtimeStart && startMinute != null
      ? {
          bedtimeStart,
          startMinute,
          lanes: Array.from(phase, code =>
            movement
              ? /[1-4]/.test(code)
                ? Number(code) - 1
                : null
              : (DAY_ANALYTICS_SLEEP_LANE_BY_CODE[code] ?? null),
          ),
        }
      : null
  const readoutId = `tri-day-${date}-sleep-${kind}-readout`
  const chart = f.el(
    'section',
    `tri-day-sleep-chart tri-day-sleep-${kind}`,
    undefined,
    hypnogram
      ? {
          'data-day-sleep-series': kind,
          'data-day-sleep-values': hypnogram.lanes.map(lane => lane ?? '').join(','),
          'data-day-sleep-start': hypnogram.bedtimeStart,
          'data-day-sleep-interval': interval.toString(),
          'data-day-sleep-width': hypnogram.lanes.length.toString(),
        }
      : undefined,
  )
  const heading = f.el('h4', 'tri-ana-block-title tri-health-tooltip-trigger', undefined, {
    tabindex: '0',
    'aria-describedby': `${readoutId}-recording`,
  })
  f.add(
    heading,
    f.el('span', undefined, triText(f.presentation.locale, title), { 'data-i18n': title }),
    healthTooltip(
      f,
      `${readoutId}-recording`,
      [
        `Oura · ${interval}s`,
        sleep.lowBatteryAlert
          ? triText(f.presentation.locale, 'sleep low battery description')
          : '',
        sleep.sleepAlgorithmVersion
          ? `${triText(f.presentation.locale, 'algorithm')} ${sleep.sleepAlgorithmVersion}`
          : '',
      ]
        .filter(Boolean)
        .join('\n'),
    ),
  )
  f.add(chart, heading)
  if (hypnogram) {
    const { lanes, startMinute: bedMinute } = hypnogram
    const height = 16
    const measured = lanes.findLastIndex(lane => lane != null)
    const cursorIndex = Math.max(0, measured)
    const readout = `${daySleepSampleClock(bedMinute + (cursorIndex * interval) / 60, interval)} · ${labelAt(f.presentation.locale, measured >= 0 ? lanes[measured] : null)}`
    const svg = f.svg('svg', {
      class: 'tri-ana-svg tri-day-sleep-stage-svg',
      viewBox: `0 0 ${lanes.length} ${height}`,
      preserveAspectRatio: 'none',
      role: 'slider',
      tabindex: 0,
      'aria-label': triText(f.presentation.locale, title),
      'aria-valuemin': 0,
      'aria-valuemax': lanes.length - 1,
      'aria-valuenow': cursorIndex,
      'aria-valuetext': readout,
      'aria-describedby': readoutId,
    })
    let start = 0
    while (start < lanes.length) {
      const lane = lanes[start]
      let end = start + 1
      while (end < lanes.length && lanes[end] === lane) end++
      if (lane != null)
        f.add(
          svg,
          f.svg('rect', {
            x: start,
            y: lane * 4 + 0.3,
            width: end - start,
            height: 3.4,
            class: movement
              ? `tri-movement--${lane}`
              : `tri-hyp--${DAY_ANALYTICS_SLEEP_LANES[lane]}`,
          }),
        )
      start = end
    }
    f.add(
      svg,
      f.svg('line', {
        x1: cursorIndex + 0.5,
        y1: 0,
        x2: cursorIndex + 0.5,
        y2: height,
        class: 'tri-ana-cursor',
      }),
    )
    f.add(
      chart,
      axisFrame(
        f,
        svg,
        labels.map((lane, index) => ({
          label: triText(f.presentation.locale, movement ? `${lane} short` : lane),
          vbY: index * 4 + 2,
          ...(movement ? { attrs: { title: triText(f.presentation.locale, lane) } } : {}),
        })),
        height,
        dayAnalyticsHourTicks(hypnogram.bedtimeStart, interval, lanes.length, lanes.length),
        false,
      ),
      f.el('div', 'tri-chart-readout', readout, { id: readoutId }),
    )
  }
  const summary = f.el('div', 'tri-day-sleep-stage-summary')
  for (const [label, seconds] of durations)
    if (seconds != null)
      f.add(
        summary,
        f.el(
          'span',
          'tri-ana-k',
          `${triText(f.presentation.locale, label)} ${dayAnalyticsDuration(seconds)}`,
        ),
      )
  f.add(chart, summary)
  return chart
}

export const dayAnalyticsSleepSeries = <N>(
  f: TriNodeFactory<N>,
  date: string,
  key: 'hrv' | 'heart-rate' | 'respiration',
  title: string,
  series: SleepChartSeries | null,
  source?: string,
): N | null => {
  if (!series || series.items.length < 2) return null
  const values = series.items.filter((value): value is number => value != null)
  if (values.length < 2) return null
  const low = Math.min(...values)
  const high = Math.max(...values)
  const padding = Math.max((high - low) * 0.1, 1)
  const minimum = low - padding
  const maximum = high + padding
  const offsetsS = 'offsetsS' in series ? series.offsetsS : undefined
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
  const initialTime = dayAnalyticsWallClock(offsetsS ? Math.floor(initialMinute) : initialMinute)
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
      ...(source ? { 'data-day-sleep-source': source } : {}),
      'data-day-sleep-unit': unit,
      'data-day-sleep-width': DAY_ANALYTICS_SLEEP_WIDTH.toString(),
    },
  )
  f.add(
    chart,
    f.el('h4', 'tri-ana-block-title', triText(f.presentation.locale, title), {
      'data-i18n': title,
    }),
  )
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
  const ticks: AxisXTick[] = []
  if (offsetsS)
    for (
      let minute = (Math.floor(startMinute / 120) + 1) * 120;
      minute <= startMinute + elapsedS / 60;
      minute += 120
    )
      ticks.push({
        label: dayAnalyticsWallClock(minute),
        pct: ((minute - startMinute) * 60 * 100) / elapsedS,
      })
  const xTicks =
    'offsetsS' in series
      ? ticks
      : dayAnalyticsHourTicks(
          series.startTs,
          series.intervalS,
          series.items.length,
          series.items.length - 1,
        )
  f.add(
    svg,
    buildPowerCurveGrid(f, 'power', [y(high), y(low)], DAY_ANALYTICS_SLEEP_HEIGHT),
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
      [
        { label: dayAnalyticsNumber(f.presentation, high), vbY: y(high) },
        { label: dayAnalyticsNumber(f.presentation, low), vbY: y(low) },
      ],
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
  const samples = garmin?.respiration
  if (!samples || samples.length < 2) return null
  const start = samples[0].timestamp
  const offset = garmin.utcOffsetMinutes ?? 0
  const absoluteOffset = Math.abs(offset)
  const offsetHours = Math.floor(absoluteOffset / 60)
    .toString()
    .padStart(2, '0')
  const offsetMinutes = (absoluteOffset % 60).toString().padStart(2, '0')
  const localStart = new Date(start + offset * 60_000).toISOString().slice(0, -1)
  const startTs = `${localStart}${offset < 0 ? '-' : '+'}${offsetHours}:${offsetMinutes}`
  return dayAnalyticsSleepSeries(
    f,
    date,
    'respiration',
    'respiration',
    {
      startTs,
      offsetsS: samples.map(sample => (sample.timestamp - start) / 1000),
      items: samples.map(sample => sample.breathsPerMinute),
    },
    garmin.utcOffsetMinutes == null ? 'Garmin (UTC)' : 'Garmin',
  )
}
