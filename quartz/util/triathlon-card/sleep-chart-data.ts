import type { Locale } from '../triathlon-presentation'
import type { AxisXTick } from './chart-geometry'
import { triText } from '../triathlon-i18n'

export const DAY_ANALYTICS_SLEEP_WIDTH = 100
export const DAY_ANALYTICS_SLEEP_HEIGHT = 24
// Garmin's two-minute samples can skip an epoch; longer gaps stay disconnected.
export const SLEEP_RESPIRATION_MAX_GAP_S = 300

export const dayAnalyticsWallMinute = (iso: string): number | null => {
  const match = /T(\d{2}):(\d{2})/.exec(iso)
  if (!match) return null
  return Number(match[1]) * 60 + Number(match[2])
}

export const dayAnalyticsWallClock = (minutes: number): string => {
  const value = ((Math.round(minutes) % 1440) + 1440) % 1440
  return `${Math.floor(value / 60)
    .toString()
    .padStart(2, '0')}:${(value % 60).toString().padStart(2, '0')}`
}

export const dayAnalyticsHourTicks = (
  startIso: string,
  intervalS: number,
  count: number,
  denominator: number,
): AxisXTick[] => {
  const startMinute = dayAnalyticsWallMinute(startIso)
  if (startMinute == null || count < 2 || denominator <= 0) return []
  if (intervalS * denominator < 7200)
    return [
      { label: dayAnalyticsWallClock(startMinute), pct: 0, cls: 'tri-cax-xt--first' },
      {
        label: dayAnalyticsWallClock(startMinute + (intervalS * (count - 1)) / 60),
        pct: ((count - 1) / denominator) * 100,
        cls: 'tri-cax-xt--last',
      },
    ]
  const ticks: AxisXTick[] = []
  const startS = startMinute * 60
  let bucket = Math.floor(startS / 7200)
  for (let index = 1; index < count; index++) {
    const nextBucket = Math.floor((startS + index * intervalS) / 7200)
    if (nextBucket === bucket) continue
    bucket = nextBucket
    ticks.push({
      label: dayAnalyticsWallClock((nextBucket * 7200) / 60),
      pct: (index / denominator) * 100,
    })
  }
  return ticks
}

export const DAY_ANALYTICS_SLEEP_LANES = ['awake', 'rem', 'light', 'deep'] as const

export const DAY_ANALYTICS_SLEEP_LANE_BY_CODE: Record<string, number> = {
  '4': 0,
  '3': 1,
  '2': 2,
  '1': 3,
}

export const DAY_ANALYTICS_MOVEMENT_LANES = [
  'quiet',
  'restless',
  'tossing and turning',
  'active',
] as const

export const daySleepSampleClock = (minute: number, intervalSeconds: number): string => {
  if (intervalSeconds >= 60 || intervalSeconds <= 0) return dayAnalyticsWallClock(minute)
  const seconds = ((Math.round(minute * 60) % 86400) + 86400) % 86400
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map(part => String(part).padStart(2, '0'))
    .join(':')
}

export const daySleepMovementLabel = (locale: Locale, lane: number | null): string =>
  lane == null || DAY_ANALYTICS_MOVEMENT_LANES[lane] == null
    ? '—'
    : triText(locale, DAY_ANALYTICS_MOVEMENT_LANES[lane])

export const daySleepStageLabel = (locale: Locale, lane: number | null): string =>
  lane == null || DAY_ANALYTICS_SLEEP_LANES[lane] == null
    ? '—'
    : triText(locale, DAY_ANALYTICS_SLEEP_LANES[lane])

export const dayAnalyticsSeriesPaths = (
  items: readonly (number | null)[],
  x: (index: number) => number,
  y: (value: number) => number,
  offsetsS?: readonly number[],
): string[] => {
  const paths: string[] = []
  let path = ''
  let points = 0
  const flush = (): void => {
    if (points >= 2) paths.push(path)
    path = ''
    points = 0
  }
  for (const [index, value] of items.entries()) {
    if (
      offsetsS &&
      index > 0 &&
      offsetsS[index] - offsetsS[index - 1] > SLEEP_RESPIRATION_MAX_GAP_S
    )
      flush()
    if (value == null) {
      flush()
      continue
    }
    const point = `${x(index).toFixed(3)} ${y(value).toFixed(3)}`
    path += `${points === 0 ? 'M' : 'L'}${point}`
    points++
  }
  flush()
  return paths
}

export type SleepChartSeries = { startTs: string; items: (number | null)[] } & (
  | { intervalS: number }
  | { offsetsS: number[] }
)
