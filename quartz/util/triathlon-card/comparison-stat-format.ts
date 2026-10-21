import type { ActivityHealth, ActivityKind } from '../../plugins/stores/strava'
import type {
  ActivityComparisonStat,
  ActivityComparisonStatKey,
  ActivityComparisonStatSource,
  ActivityComparisonWindow,
} from '../activity-comparison-metrics'
import type { TriNodeFactory } from './construction-contracts'

type ComparisonStatGroup =
  | 'load'
  | 'session'
  | 'power'
  | 'peak power'
  | 'heart rate'
  | 'technique'
  | 'conditions'
  | 'athlete'
  | 'fueling'

export const COMPARISON_STAT_GROUPS: readonly ComparisonStatGroup[] = [
  'load',
  'session',
  'power',
  'peak power',
  'heart rate',
  'technique',
  'conditions',
  'athlete',
  'fueling',
]

// Day and session context stays visible when a distance selection rescopes the table.
export const COMPARISON_CONTEXT_STATS: readonly ActivityComparisonStatKey[] = [
  'temperature',
  'humidity',
  'wind',
  'water-temperature',
  'body-weight',
  'readiness',
  'hrv',
  'resting-hr',
  'sleep-score',
  'sleep-duration',
]

export const COMPARISON_STAT_SOURCE: Record<ActivityComparisonStatSource, string> = {
  wahoo: 'Wahoo',
  garmin: 'Garmin',
  strava: 'Strava',
  oura: 'Oura',
  apple: 'Apple Watch',
  weather: 'weather',
  manual: 'manual',
  garden: 'calculated',
}

// A selection whose distance falls short of the window by more than this is partial.
export const COMPARISON_STAT_FULL_COVERAGE = 0.98

type ComparisonStatDirection = 'up' | 'down' | 'balance'

export const activityComparisonMathText = (text: string): string =>
  text
    .replace(/W′/g, '$W^{\\prime}$')
    .replace(/≈/g, '$\\approx$')
    .replace(/±/g, '$\\pm$')
    .replace(/−/g, '$-$')
    .replace(/\+/g, '$+$')
    .replace(/≠/g, '$\\ne$')
    .replace(/×/g, '$\\times$')
    .replace(/°/g, '$^{\\circ}$')
    .replace(/·/g, '$\\cdot$')

export const comparisonMathNode = <N>(
  f: TriNodeFactory<N>,
  className: string,
  value: string,
  attrs: Record<string, string> = {},
): N => {
  const math = activityComparisonMathText(value)
  if (math === value) return f.el('span', className, value, attrs)
  const node = f.el('span', className, undefined, { role: 'img', 'aria-label': value, ...attrs })
  f.add(node, f.math('tri-compare-math', math))
  return node
}

export type ComparisonStatSpec = {
  key: ActivityComparisonStatKey
  group: ComparisonStatGroup
  label: (sport: ActivityKind) => string
  gloss?: string
  format: (stat: ActivityComparisonStat, sport: ActivityKind) => string
  delta: (stat: ActivityComparisonStat, base: ActivityComparisonStat, sport: ActivityKind) => string
  /** Quantity behind the better or worse direction. */
  scalar: (stat: ActivityComparisonStat) => number
  /** Quantity behind the relative change, matching the sign of the displayed delta. */
  relativeScalar?: (stat: ActivityComparisonStat, sport: ActivityKind) => number
  better?: ComparisonStatDirection | ((selection: boolean) => ComparisonStatDirection | undefined)
  relative: boolean
  /** Firstbeat training effect and load differ in kind from the garden proxy, so deltas need one source. */
  sameSourceDelta?: boolean
}

export const comparisonFixed = (value: number, digits: number): string => {
  const rounded = Number(value.toFixed(digits))
  const text = Math.abs(rounded).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
  return rounded < 0 ? `−${text}` : text
}

export const comparisonSigned = (value: number, digits: number): string => {
  const rounded = Number(value.toFixed(digits))
  if (rounded === 0) return '±0'
  return `${rounded > 0 ? '+' : '−'}${comparisonFixed(Math.abs(rounded), digits)}`
}

const comparisonClock = (seconds: number): string => {
  const total = Math.round(Math.abs(seconds))
  const hours = Math.floor(total / 3_600)
  const minutes = Math.floor((total % 3_600) / 60)
  const rest = (total % 60).toString().padStart(2, '0')
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, '0')}:${rest}`
    : `${minutes}:${rest}`
}

export const comparisonSignedClock = (seconds: number): string =>
  Math.round(seconds) === 0 ? '±0:00' : `${seconds > 0 ? '+' : '−'}${comparisonClock(seconds)}`

type ComparisonUnit = string | ((sport: ActivityKind) => string)

const comparisonUnit = (unit: ComparisonUnit | undefined, sport: ActivityKind): string =>
  typeof unit === 'function' ? unit(sport) : (unit ?? '')

export const numericStat = (
  key: ActivityComparisonStatKey,
  group: ComparisonStatGroup,
  label: string | ((sport: ActivityKind) => string),
  options: {
    unit?: ComparisonUnit
    deltaUnit?: ComparisonUnit
    digits?: number
    display?: (value: number) => number
    better?: ComparisonStatSpec['better']
    relative?: boolean
    gloss?: string
    sameSourceDelta?: boolean
  } = {},
): ComparisonStatSpec => {
  const digits = options.digits ?? 0
  const display = options.display ?? ((value: number) => value)
  return {
    key,
    group,
    label: typeof label === 'string' ? () => label : label,
    gloss: options.gloss,
    format: (stat, sport) =>
      `${comparisonFixed(display(stat.value), digits)}${comparisonUnit(options.unit, sport)}`,
    delta: (stat, base, sport) =>
      `${comparisonSigned(display(stat.value) - display(base.value), digits)}${comparisonUnit(options.deltaUnit ?? options.unit, sport)}`,
    scalar: stat => stat.value,
    better: options.better,
    relative: options.relative ?? true,
    sameSourceDelta: options.sameSourceDelta,
  }
}

export const durationStat = (
  key: ActivityComparisonStatKey,
  group: ComparisonStatGroup,
  label: string,
  better?: ComparisonStatSpec['better'],
): ComparisonStatSpec => ({
  key,
  group,
  label: () => label,
  format: stat => comparisonClock(stat.value),
  delta: (stat, base) => comparisonSignedClock(stat.value - base.value),
  scalar: stat => stat.value,
  better,
  relative: true,
})

export const pairedSideStat = (
  key: ActivityComparisonStatKey,
  label: string,
  gloss: string,
): ComparisonStatSpec => {
  const mean = (stat: ActivityComparisonStat): number =>
    stat.right == null ? stat.value : (stat.value + stat.right) / 2
  return {
    key,
    group: 'technique',
    label: () => label,
    gloss,
    format: stat =>
      stat.right == null
        ? `${Math.round(stat.value)}%`
        : `${Math.round(stat.value)} / ${Math.round(stat.right)}%`,
    delta: (stat, base) => `${comparisonSigned(mean(stat) - mean(base), 1)} pp`,
    scalar: mean,
    relative: false,
  }
}

export const peakStat = (key: ActivityComparisonStatKey, label: string): ComparisonStatSpec =>
  numericStat(key, 'peak power', label, {
    unit: ' W',
    better: 'up',
    gloss:
      'Best rolling mean power over the duration, from the power curve. A selection reads the 30 s and 5 min rolling traces inside it.',
  })

export const comparisonStatTrend = (
  direction: ComparisonStatDirection,
  value: number,
  base: number,
): 'better' | 'worse' | undefined => {
  const distance =
    direction === 'balance' ? Math.abs(base - 50) - Math.abs(value - 50) : value - base
  if (distance === 0) return undefined
  return (direction === 'down' ? distance < 0 : distance > 0) ? 'better' : 'worse'
}

export const comparisonRelativeChange = (value: number, base: number): string | null => {
  if (!Number.isFinite(value) || !Number.isFinite(base) || base === 0) return null
  const percent = ((value - base) / Math.abs(base)) * 100
  const text = comparisonSigned(percent, Math.abs(percent) < 10 ? 1 : 0)
  return text === '±0' ? null : `${text}%`
}

export type ActivityComparisonStatsScope = 'whole' | 'selection'

export type ActivityComparisonStatsView = {
  hiddenIds?: ReadonlySet<string>
  /** Distance window of the chart selection, available to the selection scope. */
  window?: ActivityComparisonWindow | null
  scope?: ActivityComparisonStatsScope
  health?: Readonly<Record<string, ActivityHealth>> | null
}
