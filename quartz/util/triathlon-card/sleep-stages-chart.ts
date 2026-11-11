import type { TriathlonDaySleep } from '../triathlon-day-sleep'
import type { TriNodeFactory } from './construction-contracts'
import { sleepTimestampWithOffset } from '../triathlon-day-sleep'
import { healthTooltip } from '../triathlon-health'
import { triText } from '../triathlon-i18n'
import { axisFrame } from './chart-geometry'
import { dayAnalyticsDuration } from './day-analytics-metrics'
import {
  DAY_ANALYTICS_MOVEMENT_LANES,
  DAY_ANALYTICS_SLEEP_LANE_BY_CODE,
  DAY_ANALYTICS_SLEEP_LANES,
  dayAnalyticsHourTicks,
  dayAnalyticsWallMinute,
  daySleepMovementLabel,
  daySleepNativeTicks,
  daySleepSampleClock,
  daySleepStageLabel,
} from './sleep-chart-data'

type SleepBands = { startTs: string; lanes: (number | null)[] } & (
  | { intervalS: number }
  | { boundariesS: number[] }
)

const nativeBands = (sleep: TriathlonDaySleep, offset: number | undefined): SleepBands | null => {
  const first = sleep.stages?.[0]
  if (!sleep.stages || !first) return null
  const bedtime = Date.parse(sleep.bedtimeStart ?? '')
  const wakeup = Date.parse(sleep.bedtimeEnd ?? '')
  const start = Number.isFinite(bedtime) ? Math.min(first.startTime, bedtime) : first.startTime
  const lanes: (number | null)[] = []
  const boundariesS = [0]
  let last = start
  const push = (lane: number | null, until: number): void => {
    lanes.push(lane)
    boundariesS.push((until - start) / 1000)
    last = until
  }
  for (const stage of sleep.stages) {
    if (stage.startTime > last) push(null, stage.startTime)
    push(DAY_ANALYTICS_SLEEP_LANES.indexOf(stage.stage), stage.endTime)
  }
  if (wakeup > last) push(null, wakeup)
  return {
    startTs:
      sleep.nightSource === 'apple' && sleep.bedtimeStart && start === bedtime
        ? sleep.bedtimeStart
        : sleepTimestampWithOffset(start, offset),
    lanes,
    boundariesS,
  }
}

export const dayAnalyticsSleepStages = <N>(
  f: TriNodeFactory<N>,
  date: string,
  sleep: TriathlonDaySleep,
  kind: 'stages' | 'movement' = 'stages',
  garminOffset?: number,
): N | null => {
  const movement = kind === 'movement'
  const garmin = !movement && sleep.nightSource === 'garmin'
  const apple = !movement && sleep.nightSource === 'apple'
  const native = garmin || apple
  const title = movement ? 'sleep movement' : 'sleep stages'
  const labels = movement
    ? DAY_ANALYTICS_MOVEMENT_LANES
    : native &&
        ((sleep.unmeasurableS ?? 0) > 0 || sleep.stages?.some(row => row.stage === 'unmeasurable'))
      ? DAY_ANALYTICS_SLEEP_LANES
      : DAY_ANALYTICS_SLEEP_LANES.slice(0, 4)
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
        ['unmeasurable', sleep.unmeasurableS ?? null],
      ]
  const bedtimeStart = sleep.bedtimeStart
  const bands: SleepBands | null = native
    ? nativeBands(sleep, garminOffset)
    : phase && phase.length >= 2 && bedtimeStart
      ? {
          startTs: bedtimeStart,
          intervalS: interval,
          lanes: Array.from(phase, code =>
            movement
              ? /[1-4]/.test(code)
                ? Number(code) - 1
                : null
              : (DAY_ANALYTICS_SLEEP_LANE_BY_CODE[code] ?? null),
          ),
        }
      : null
  if (!phase && !bands && durations.every(([, seconds]) => seconds == null)) return null
  const boundaries = bands && 'boundariesS' in bands ? bands.boundariesS : undefined
  const width = boundaries?.at(-1) ?? bands?.lanes.length ?? 0
  const source = apple
    ? 'Apple Health'
    : garmin
      ? garminOffset == null
        ? 'Garmin (UTC)'
        : 'Garmin'
      : 'Oura'
  const readoutId = `tri-day-${date}-sleep-${kind}-readout`
  const chart = f.el(
    'section',
    `tri-day-sleep-chart tri-day-sleep-${kind}`,
    undefined,
    bands
      ? {
          'data-day-sleep-series': kind,
          'data-day-sleep-values': bands.lanes.map(lane => lane ?? '').join(','),
          'data-day-sleep-start': bands.startTs,
          ...('boundariesS' in bands
            ? { 'data-day-sleep-boundaries': bands.boundariesS.join(',') }
            : { 'data-day-sleep-interval': bands.intervalS.toString() }),
          'data-day-sleep-width': width.toString(),
          'data-day-sleep-source': source,
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
        garmin ? source : `${source} · ${interval}s`,
        !garmin && sleep.lowBatteryAlert
          ? triText(f.presentation.locale, 'sleep low battery description')
          : '',
        !garmin && sleep.sleepAlgorithmVersion
          ? `${triText(f.presentation.locale, 'algorithm')} ${sleep.sleepAlgorithmVersion}`
          : '',
      ]
        .filter(Boolean)
        .join('\n'),
    ),
  )
  f.add(chart, heading)
  if (bands) {
    const { lanes, startTs } = bands
    const readoutInterval = boundaries ? 1 : interval
    const bedMinute =
      (dayAnalyticsWallMinute(startTs) ?? 0) +
      (readoutInterval < 60 ? Number(startTs.slice(17, 19)) / 60 : 0)
    const height = labels.length * 4
    const measured = lanes.findLastIndex(lane => lane != null)
    const cursorIndex = Math.max(0, measured)
    const edge = (index: number): number => boundaries?.[index] ?? index
    const cursorX = (edge(cursorIndex) + edge(cursorIndex + 1)) / 2
    const elapsedS = boundaries?.[cursorIndex] ?? cursorIndex * interval
    const readout = `${daySleepSampleClock(bedMinute + elapsedS / 60, readoutInterval)} · ${labelAt(f.presentation.locale, measured >= 0 ? lanes[measured] : null)}`
    const svg = f.svg('svg', {
      class: 'tri-ana-svg tri-day-sleep-stage-svg',
      viewBox: `0 0 ${width} ${height}`,
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
            x: edge(start),
            y: lane * 4 + 0.3,
            width: edge(end) - edge(start),
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
      f.svg('line', { x1: cursorX, y1: 0, x2: cursorX, y2: height, class: 'tri-ana-cursor' }),
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
        boundaries
          ? daySleepNativeTicks(bedMinute, width)
          : dayAnalyticsHourTicks(startTs, interval, lanes.length, lanes.length),
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
