import type { Analytics } from '../../../../plugins/stores/analytics'
import type { PowerCurveBlock } from '../../../../plugins/stores/analytics'
import type { CriticalPowerEstimate } from '../../../../plugins/stores/critical-power'
import type {
  PowerRankEffort,
  PowerRankInterval,
  PowerRankLevelName,
  PowerRankReference,
  PowerSkill,
} from '../../../../plugins/stores/power-rank'
import type { PowerCurvePoint } from '../../../../plugins/stores/strava'
import type { AxisXTick } from '../../../../util/triathlon-card'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { criticalPowerAtDuration } from '../../../../plugins/stores/critical-power'
import { criticalPowerCurve } from '../../../../plugins/stores/critical-power'
import { POWER_RANK_LEVELS } from '../../../../plugins/stores/power-rank'
import { axisFrame } from '../../../../util/triathlon-card'
import { axisNumber } from '../../../../util/triathlon-card'
import { dlabel } from '../../../../util/triathlon-card'
import { nearestPowerCurvePoint } from '../../../../util/triathlon-card'
import { niceStep } from '../../../../util/triathlon-card'
import { powerCurveDurationTicks } from '../../../../util/triathlon-card'
import { powerCurveFraction } from '../../../../util/triathlon-card'
import { powerCurveHoverAt } from '../../../../util/triathlon-card'
import { powerCurvePathPoints } from '../../../../util/triathlon-card'
import { zoneClock } from '../../../../util/triathlon-card'
import { criticalPowerSummaryText } from '../../../../util/triathlon-i18n'
import { powerCurveActivityLinkAttributes } from '../../../../util/triathlon-power-activity'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { anaTitle } from '../shared'

export type BestPowerSeriesKey = 'six-weeks' | 'year'

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

export const powerSkillAtSeconds = (seconds: number): PowerSkill =>
  seconds <= 60 ? 'sprint' : seconds <= 600 ? 'attack' : 'climb'

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

const buildPowerRankRadar = (
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

const buildPowerRankProgress = (
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

export const bestPowerSeriesLabel = (
  formatter: TriathlonFormatter,
  power: PowerCurveBlock,
  key: BestPowerSeriesKey,
): string =>
  key === 'six-weeks'
    ? formatter.text('last 6 weeks')
    : power.yearLabel == null
      ? formatter.text('calendar year')
      : `${formatter.text('all of')} ${power.yearLabel}`

export const bestPowerSeries = (
  power: PowerCurveBlock,
): Array<{ key: BestPowerSeriesKey; curve: PowerCurvePoint[] }> => [
  { key: 'six-weeks', curve: power.sixWeeks },
  { key: 'year', curve: power.year },
]

export const criticalPowerForSeries = (
  power: PowerCurveBlock,
  key: BestPowerSeriesKey,
): CriticalPowerEstimate | null =>
  key === 'six-weeks' ? power.criticalPower : power.criticalPowerYear

export const buildBestPowerCurve = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('section', 'tri-best-power')
  const power = data.powerCurve
  const head = el('div', 'tri-best-power-head')
  head.appendChild(anaTitle(context.formatter, 'best efforts · power curve'))
  const controls = el('div', 'tri-best-power-controls', undefined, {
    role: 'group',
    'aria-label': context.formatter.text('power curve periods'),
  })
  const series = bestPowerSeries(power)
  for (const { key, curve } of series) {
    const available = curve.length >= 2
    const attrs: Record<string, string> = {
      type: 'button',
      'data-power-series': key,
      'aria-pressed': String(available),
    }
    if (!available) attrs.disabled = ''
    const button = el('button', 'tri-best-power-toggle', undefined, attrs)
    button.append(
      el('span', `tri-best-power-swatch tri-best-power-swatch--${key}`, undefined, {
        'aria-hidden': 'true',
      }),
      el('span', undefined, bestPowerSeriesLabel(context.formatter, power, key)),
    )
    controls.appendChild(button)
  }
  head.appendChild(controls)
  block.appendChild(head)

  const available = series.filter(({ curve }) => curve.length >= 2)
  if (available.length === 0) {
    block.appendChild(el('div', 'tri-ana-empty', context.formatter.text('no cycling power data')))
    return block
  }

  const minSeconds = Math.min(...available.map(({ curve }) => curve[0].s))
  const maxSeconds = Math.max(...available.map(({ curve }) => curve[curve.length - 1].s))
  const W = 100
  const H = 34
  const observedMax = Math.max(
    1,
    ...available.flatMap(({ curve }) => curve.map(point => point.w)),
    power.ftp ?? 0,
    power.goalFtp ?? 0,
    ...series.flatMap(({ key }) => {
      const estimate = criticalPowerForSeries(power, key)
      return estimate ? [estimate.criticalPowerWatts + estimate.wPrimeJoules / 180] : []
    }),
  )
  const step = niceStep(observedMax, 4)
  const domainMax = Math.ceil(observedMax / step) * step
  const X = (seconds: number): number => powerCurveFraction(seconds, minSeconds, maxSeconds) * W
  const Y = (watts: number): number => H - (watts / domainMax) * (H - 1)
  const yTicks = Array.from(
    { length: Math.round(domainMax / step) + 1 },
    (_, index) => index * step,
  ).map(value => ({ label: value === 0 ? '0' : `${axisNumber(value, step)}w`, vbY: Y(value) }))
  const path = (curve: PowerCurvePoint[]): string =>
    powerCurvePathPoints(curve)
      .map(
        (point, index) =>
          `${index === 0 ? 'M' : 'L'} ${X(point.s).toFixed(2)} ${Y(point.w).toFixed(2)}`,
      )
      .join(' ')
  const skillPaths = (curve: PowerCurvePoint[]): Array<{ skill: PowerSkill; d: string }> => {
    const points = powerCurvePathPoints(curve)
    const bySkill = new Map<PowerSkill, string[]>([
      ['sprint', []],
      ['attack', []],
      ['climb', []],
    ])
    for (let index = 1; index < points.length; index++) {
      const from = points[index - 1]
      const to = points[index]
      const skill = powerSkillAtSeconds(Math.sqrt(from.s * to.s))
      bySkill
        .get(skill)
        ?.push(
          `M ${X(from.s).toFixed(2)} ${Y(from.w).toFixed(2)} L ${X(to.s).toFixed(2)} ${Y(to.w).toFixed(2)}`,
        )
    }
    return [...bySkill].flatMap(([skill, paths]) =>
      paths.length === 0 ? [] : [{ skill, d: paths.join(' ') }],
    )
  }
  const anchor = available[0].curve
  const initial = powerCurveHoverAt(
    anchor,
    [],
    powerCurveFraction(300, anchor[0].s, anchor[anchor.length - 1].s),
  )
  const selectedSeconds = initial?.durationS ?? anchor[0].s
  const graph = svg('svg', {
    class: 'tri-best-power-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': context.formatter.text('best efforts power curve'),
    'aria-orientation': 'horizontal',
    'aria-valuemin': minSeconds,
    'aria-valuemax': maxSeconds,
    'aria-valuenow': selectedSeconds,
    'data-power-selected-seconds': selectedSeconds,
    'data-power-domain-max': domainMax,
  })
  for (const tick of yTicks)
    graph.appendChild(
      svg('line', {
        class: 'tri-best-power-grid',
        x1: 0,
        y1: tick.vbY.toFixed(2),
        x2: W,
        y2: tick.vbY.toFixed(2),
        'aria-hidden': 'true',
      }),
    )
  for (const { key, curve } of available)
    for (const { skill, d } of skillPaths(curve))
      graph.appendChild(
        svg('path', {
          class: `tri-best-power-line tri-best-power-line--${key} tri-best-power-line--${skill}`,
          d,
          'data-power-series': key,
          'data-power-skill': skill,
          'aria-hidden': 'true',
        }),
      )
  for (const { key } of available) {
    const estimate = criticalPowerForSeries(power, key)
    if (!estimate) continue
    const model = criticalPowerCurve(estimate, minSeconds, maxSeconds)
    if (model.length >= 2)
      graph.appendChild(
        svg('path', {
          class: `tri-best-power-model-line tri-best-power-model-line--${key}`,
          d: path(model),
          'data-power-model-series': key,
          'aria-hidden': 'true',
        }),
      )
    graph.appendChild(
      svg('line', {
        class: `tri-best-power-cp tri-best-power-cp--${key}`,
        x1: 0,
        y1: Y(estimate.criticalPowerWatts).toFixed(2),
        x2: W,
        y2: Y(estimate.criticalPowerWatts).toFixed(2),
        'data-power-model-series': key,
        'aria-hidden': 'true',
      }),
    )
  }
  if (power.ftp != null)
    graph.appendChild(
      svg('line', {
        class: 'tri-best-power-ftp',
        x1: 0,
        y1: Y(power.ftp).toFixed(2),
        x2: W,
        y2: Y(power.ftp).toFixed(2),
      }),
    )
  if (power.goalFtp != null)
    graph.appendChild(
      svg('line', {
        class: 'tri-best-power-goal',
        x1: 0,
        y1: Y(power.goalFtp).toFixed(2),
        x2: W,
        y2: Y(power.goalFtp).toFixed(2),
      }),
    )
  graph.appendChild(
    svg('line', {
      class: 'tri-best-power-cursor',
      x1: X(selectedSeconds).toFixed(2),
      y1: 0,
      x2: X(selectedSeconds).toFixed(2),
      y2: H,
    }),
  )

  const overlays: Array<HTMLElement | SVGElement> = []
  const readout = el('div', 'tri-best-power-readout')
  readout.appendChild(el('span', 'tri-best-power-duration', zoneClock(selectedSeconds)))
  for (const { key, curve } of available) {
    const selectedPoint = nearestPowerCurvePoint(curve, selectedSeconds)
    const watts = selectedPoint?.w ?? null
    const point = el('span', `tri-best-power-point tri-best-power-point--${key}`, undefined, {
      'data-power-series': key,
      'data-power-skill': powerSkillAtSeconds(selectedSeconds),
      'aria-hidden': 'true',
    })
    if (watts != null)
      point.setAttribute(
        'style',
        `left:${X(selectedSeconds).toFixed(2)}%;top:${((Y(watts) / H) * 100).toFixed(2)}%`,
      )
    const row = el('a', 'tri-best-power-readout-row', undefined, {
      'data-power-series': key,
      ...powerCurveActivityLinkAttributes(selectedPoint),
    })
    row.append(
      el('span', `tri-best-power-swatch tri-best-power-swatch--${key}`, undefined, {
        'aria-hidden': 'true',
        'data-power-skill': powerSkillAtSeconds(selectedSeconds),
      }),
      el('strong', 'tri-best-power-value', watts == null ? '—' : `${watts.toLocaleString()} W`),
    )
    overlays.push(point)
    readout.appendChild(row)
  }
  for (const { key } of available) {
    const estimate = criticalPowerForSeries(power, key)
    if (!estimate) continue
    const visible = selectedSeconds >= 180 && selectedSeconds <= 720
    const watts = visible ? Math.round(criticalPowerAtDuration(estimate, selectedSeconds)) : null
    const row = el(
      'span',
      'tri-best-power-readout-row tri-best-power-readout-row--model',
      undefined,
      { 'data-power-model-series': key },
    )
    row.hidden = !visible
    row.append(
      el(
        'span',
        `tri-best-power-swatch tri-best-power-swatch--model tri-best-power-swatch--model-${key}`,
        undefined,
        { 'aria-hidden': 'true' },
      ),
      el('strong', 'tri-best-power-value', watts == null ? '—' : `${watts.toLocaleString()} W`),
    )
    readout.appendChild(row)
  }
  overlays.push(readout)

  const durationTicks: AxisXTick[] = powerCurveDurationTicks(
    minSeconds,
    maxSeconds,
    [1, 15, 60, 300, 600, 1_200, 1_800, 2_700, 3_600, 5_400, 7_200, 10_800, 14_400, 18_000],
  ).map((seconds, index, ticks) => ({
    label: dlabel(seconds),
    pct: X(seconds),
    cls: `tri-best-power-tick${index === 0 ? ' tri-cax-xt--first' : index === ticks.length - 1 ? ' tri-cax-xt--last' : ''}`,
    tag: 'button',
    attrs: {
      type: 'button',
      'data-power-seconds': String(seconds),
      'aria-pressed': String(seconds === selectedSeconds),
    },
  }))
  const curvePane = el('div', 'tri-best-power-curve-pane')
  curvePane.appendChild(
    axisFrame(
      createDomFactory(context.presentation),
      graph,
      yTicks,
      H,
      durationTicks,
      true,
      undefined,
      overlays,
    ),
  )

  const criticalPowerCaptions = [
    ['six-weeks', power.criticalPower],
    ['year', power.criticalPowerYear],
  ] as const
  if (
    power.ftp != null ||
    criticalPowerCaptions.some(([, estimate]) => estimate != null) ||
    power.goalFtp != null
  ) {
    const cap = el('div', 'tri-best-power-cap')
    if (power.ftp != null)
      cap.appendChild(el('span', 'tri-best-power-cap-ftp', `FTP ${power.ftp}W`))
    let hasCriticalPowerCaption = false
    for (const [key, estimate] of criticalPowerCaptions) {
      if (!estimate) continue
      cap.appendChild(
        el(
          'span',
          'tri-best-power-cap-cp',
          criticalPowerSummaryText(context.presentation.locale, estimate),
          { 'data-power-cap-series': key, ...(hasCriticalPowerCaption ? { hidden: '' } : {}) },
        ),
      )
      hasCriticalPowerCaption = true
    }
    if (power.goalFtp != null)
      cap.appendChild(
        el(
          'span',
          'tri-best-power-cap-goal',
          `${context.formatter.text('goal')} ${power.goalFtp}W`,
        ),
      )
    curvePane.appendChild(cap)
  }
  const rankProgress = buildPowerRankProgress(
    power,
    context,
    selectedSeconds,
    available.map(item => item.key),
  )
  if (rankProgress) curvePane.appendChild(rankProgress)
  const visuals = el('div', 'tri-best-power-visuals')
  visuals.append(
    curvePane,
    buildPowerRankRadar(
      power,
      context,
      selectedSeconds,
      available.map(item => item.key),
    ),
  )
  block.appendChild(visuals)
  return block
}
