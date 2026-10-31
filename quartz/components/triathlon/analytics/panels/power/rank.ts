import type { PowerCurveBlock } from '../../../../../plugins/stores/analytics'
import type {
  PowerRankEffort,
  PowerRankInterval,
  PowerRankLevelName,
  PowerRankReference,
} from '../../../../../plugins/stores/power-rank'
import type { TriathlonContext } from '../../../runtime/context'
import { POWER_RANK_LEVELS } from '../../../../../plugins/stores/power-rank'
import { dlabel } from '../../../../../util/triathlon-card'
import { el, svg } from '../../../runtime/dom'
import { bestPowerSeriesLabel, type BestPowerSeriesKey } from './series'

export const powerRankLevelLabel = (name: PowerRankLevelName | null): string =>
  name == null
    ? 'below aspiring'
    : name
        .split(' ')
        .map(word =>
          word
            .split('-')
            .map(part => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
            .join('-'),
        )
        .join(' ')

export const powerRankCohortLabel = (reference: PowerRankReference): string =>
  `men · age ${reference.ageMin}–${reference.ageMax}`

export const powerRankEffortLabel = (effort: PowerRankEffort | null): string =>
  effort == null
    ? 'no data'
    : `${effort.watts.toLocaleString()} W · ${effort.wattsPerKg.toFixed(2)} W/kg`

export const powerRankProgressLabel = (effort: PowerRankEffort | null): string =>
  effort == null ? 'no data' : `${powerRankLevelLabel(effort.levelName)} · ${effort.percentile}%`

export const powerRankProgressNextLabel = (effort: PowerRankEffort | null): string =>
  effort == null
    ? ''
    : effort.wattsToNext == null
      ? 'top level'
      : `${effort.wattsToNext.toLocaleString()} W to ${powerRankLevelLabel(effort.nextLevelName)}`

export interface PowerRankRangeRow {
  level: number
  label: string
  percentile: number
  thresholdWatts: number
  bestWatts: number | null
  gapWatts: number | null
  current: boolean
}

export const powerRankRangeRows = (
  interval: PowerRankInterval,
  key: BestPowerSeriesKey,
): PowerRankRangeRow[] => {
  const effort = interval.efforts[key]
  return [...interval.thresholds].reverse().map(threshold => {
    const current = effort?.level === threshold.level
    return {
      level: threshold.level,
      label: powerRankLevelLabel(threshold.name),
      percentile: threshold.percentile,
      thresholdWatts: threshold.watts,
      bestWatts: current ? effort.watts : null,
      gapWatts:
        effort != null && threshold.level > effort.level
          ? Math.max(0, threshold.watts - effort.watts)
          : null,
      current,
    }
  })
}

const radarAngle = (index: number, count: number): number =>
  ((-90 + (360 / count) * index) * Math.PI) / 180

const radarPoint = (
  index: number,
  score: number,
  count: number,
  radius: number,
): [number, number] => {
  const angle = radarAngle(index, count)
  const distance = (radius * score) / 100
  return [50 + distance * Math.cos(angle), 50 + distance * Math.sin(angle)]
}

const radarPolygon = (
  intervals: readonly PowerRankInterval[],
  scoreAt: (interval: PowerRankInterval) => number,
  radius: number,
): string =>
  `${intervals
    .map((interval, index) => {
      const [x, y] = radarPoint(index, scoreAt(interval), intervals.length, radius)
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')} Z`

const radarArc = (from: number, to: number, count: number, radius: number): string => {
  const startAngle = radarAngle(from - 0.43, count)
  const endAngle = radarAngle(to + 0.43, count)
  const startX = 50 + radius * Math.cos(startAngle)
  const startY = 50 + radius * Math.sin(startAngle)
  const endX = 50 + radius * Math.cos(endAngle)
  const endY = 50 + radius * Math.sin(endAngle)
  return `M ${startX.toFixed(2)} ${startY.toFixed(2)} A ${radius} ${radius} 0 0 1 ${endX.toFixed(2)} ${endY.toFixed(2)}`
}

export const buildPowerRankRadar = (
  power: PowerCurveBlock,
  context: TriathlonContext,
  selectedSeconds: number,
  availableSeries: readonly BestPowerSeriesKey[],
): HTMLElement => {
  const rank = power.ranking
  const wrap = el('section', 'tri-power-radar')
  const head = el('div', 'tri-power-radar-head')
  head.append(
    el('span', 'tri-power-radar-title', 'weight-adjusted rank'),
    el(
      'span',
      'tri-power-radar-mass',
      rank.massKg == null ? context.formatter.text('no data') : `${rank.massKg.toFixed(1)} kg`,
    ),
  )
  wrap.appendChild(head)
  if (rank.intervals.length === 0) {
    wrap.appendChild(
      el(
        'div',
        'tri-ana-empty',
        rank.massKg == null
          ? 'body weight is required for power ranking'
          : `ranking is unavailable outside the ${powerRankCohortLabel(rank.reference)} reference`,
      ),
    )
    return wrap
  }

  const intervals = rank.intervals
  const radius = 31
  const initialIndex = intervals.reduce(
    (best, interval, index) =>
      Math.abs(Math.log(interval.durationS) - Math.log(selectedSeconds)) <
      Math.abs(Math.log(intervals[best].durationS) - Math.log(selectedSeconds))
        ? index
        : best,
    0,
  )
  const radar = svg('svg', {
    class: 'tri-power-radar-svg',
    viewBox: '-5 -5 110 110',
    role: 'slider',
    tabindex: 0,
    'aria-label': 'weight-adjusted power ranking',
    'aria-orientation': 'horizontal',
    'aria-valuemin': intervals[0].durationS,
    'aria-valuemax': intervals[intervals.length - 1].durationS,
    'aria-valuenow': intervals[initialIndex].durationS,
    'data-power-rank-index': initialIndex,
  })
  for (const score of [25, 50, 75, 100])
    radar.appendChild(
      svg('path', {
        class: 'tri-power-radar-grid',
        d: radarPolygon(intervals, () => score, radius),
        'aria-hidden': 'true',
      }),
    )
  intervals.forEach((_, index) => {
    const [x, y] = radarPoint(index, 100, intervals.length, radius)
    radar.appendChild(
      svg('line', {
        class: 'tri-power-radar-spoke',
        x1: 50,
        y1: 50,
        x2: x.toFixed(2),
        y2: y.toFixed(2),
        'aria-hidden': 'true',
      }),
    )
  })
  for (const [skill, from, to] of [
    ['sprint', 0, 2],
    ['attack', 3, 6],
    ['climb', 7, 11],
  ])
    radar.appendChild(
      svg('path', {
        class: `tri-power-radar-arc tri-power-radar-arc--${skill}`,
        d: radarArc(Number(from), Number(to), intervals.length, 37),
        'aria-hidden': 'true',
      }),
    )
  for (const key of [...availableSeries].reverse()) {
    radar.appendChild(
      svg('path', {
        class: `tri-power-radar-fill tri-power-radar-fill--${key}`,
        d: radarPolygon(intervals, interval => interval.efforts[key]?.percentile ?? 0, radius),
        'data-power-radar-series': key,
        'aria-hidden': 'true',
      }),
    )
  }
  intervals.forEach((interval, index) => {
    const angle = radarAngle(index, intervals.length)
    const x = 50 + 43 * Math.cos(angle)
    const y = 50 + 43 * Math.sin(angle) + 1.25
    const label = svg('text', {
      class: `tri-power-radar-label${index === initialIndex ? ' tri-power-radar-label--active' : ''}`,
      x: x.toFixed(2),
      y: y.toFixed(2),
      'text-anchor':
        Math.abs(Math.cos(angle)) < 0.25 ? 'middle' : Math.cos(angle) > 0 ? 'start' : 'end',
      'data-power-rank-index': index,
    })
    label.textContent = dlabel(interval.durationS)
    radar.appendChild(label)
  })
  wrap.appendChild(radar)

  const selected = intervals[initialIndex]
  const readout = el('div', 'tri-power-radar-readout')
  readout.appendChild(
    el('span', 'tri-power-radar-duration', `${dlabel(selected.durationS)} · ${selected.skill}`),
  )
  for (const key of availableSeries) {
    const row = el('div', 'tri-power-radar-readout-row', undefined, {
      'data-power-rank-series': key,
    })
    row.append(
      el('span', `tri-best-power-swatch tri-best-power-swatch--${key}`, undefined, {
        'aria-hidden': 'true',
        'data-power-skill': selected.skill,
      }),
      el('span', 'tri-power-radar-readout-value', powerRankEffortLabel(selected.efforts[key])),
    )
    readout.appendChild(row)
  }
  wrap.appendChild(readout)
  return wrap
}

export const buildPowerRankProgress = (
  power: PowerCurveBlock,
  context: TriathlonContext,
  selectedSeconds: number,
  availableSeries: readonly BestPowerSeriesKey[],
): HTMLElement | null => {
  if (power.ranking.intervals.length === 0 || availableSeries.length === 0) return null
  const index = power.ranking.intervals.reduce(
    (best, interval, intervalIndex) =>
      Math.abs(Math.log(interval.durationS) - Math.log(selectedSeconds)) <
      Math.abs(Math.log(power.ranking.intervals[best].durationS) - Math.log(selectedSeconds))
        ? intervalIndex
        : best,
    0,
  )
  const interval = power.ranking.intervals[index]
  const primaryKey = availableSeries[0]
  const primaryEffort = interval.efforts[primaryKey]
  const progress = el('div', 'tri-power-rank-progress')
  const label = el('span', 'tri-power-rank-progress-label', powerRankProgressLabel(primaryEffort), {
    'data-power-rank-progress-label': '',
  })
  const track = el('span', 'tri-power-rank-progress-track', undefined, {
    'data-power-rank-progress-track': '',
    'data-power-skill': interval.skill,
  })
  for (const level of POWER_RANK_LEVELS)
    track.appendChild(
      el('span', 'tri-power-rank-progress-step', undefined, {
        'aria-hidden': 'true',
        style: `--tri-power-rank-step:${level.percentile}%`,
      }),
    )
  track.appendChild(
    el('span', 'tri-power-rank-progress-fill', undefined, {
      role: 'progressbar',
      'aria-label': `${bestPowerSeriesLabel(context.formatter, power, primaryKey)} power rank`,
      'aria-valuemin': '0',
      'aria-valuemax': '100',
      'aria-valuenow': String(primaryEffort?.percentile ?? 0),
      'aria-valuetext': powerRankProgressLabel(primaryEffort),
      'data-power-rank-progress-fill': '',
      style: `--tri-power-rank-progress:${primaryEffort?.percentile ?? 0}%`,
    }),
  )
  const next = el(
    'span',
    'tri-power-rank-progress-next',
    powerRankProgressNextLabel(primaryEffort),
    { 'data-power-rank-progress-next': '' },
  )
  progress.append(label, track, next)
  return progress
}
