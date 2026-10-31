import type { Analytics } from '../../../../../plugins/stores/analytics'
import type { PowerSkill } from '../../../../../plugins/stores/power-rank'
import type { PowerCurvePoint } from '../../../../../plugins/stores/strava'
import type { AxisXTick } from '../../../../../util/triathlon-card'
import type { TriathlonContext } from '../../../runtime/context'
import { criticalPowerAtDuration } from '../../../../../plugins/stores/critical-power'
import { criticalPowerCurve } from '../../../../../plugins/stores/critical-power'
import { axisFrame } from '../../../../../util/triathlon-card'
import { buildPowerCurveGrid } from '../../../../../util/triathlon-card'
import { dlabel } from '../../../../../util/triathlon-card'
import { nearestPowerCurvePoint } from '../../../../../util/triathlon-card'
import { POWER_CURVE_WATT_STEP } from '../../../../../util/triathlon-card'
import { powerCurveAxisTicks } from '../../../../../util/triathlon-card'
import { powerCurveDurationTicks } from '../../../../../util/triathlon-card'
import { powerCurveFraction } from '../../../../../util/triathlon-card'
import { powerCurveHoverAt } from '../../../../../util/triathlon-card'
import { powerCurvePathPoints } from '../../../../../util/triathlon-card'
import { zoneClock } from '../../../../../util/triathlon-card'
import { powerCurveActivityLinkAttributes } from '../../../../../util/triathlon-power-activity'
import { createDomFactory } from '../../../runtime/dom'
import { el } from '../../../runtime/dom'
import { svg } from '../../../runtime/dom'
import { anaTitle } from '../../shared'
import { buildPowerRankProgress, buildPowerRankRadar } from './rank'
import {
  bestPowerSeries,
  bestPowerSeriesLabel,
  criticalPowerForSeries,
  estimatedFtpForSeries,
  powerSkillAtSeconds,
} from './series'
import { buildPowerUnits, buildPowerCaptions } from './units'

export const buildBestPowerCurve = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('section', 'tri-best-power', undefined, { 'data-power-unit': 'watts' })
  const power = data.powerCurve
  const head = el('div', 'tri-best-power-head')
  head.appendChild(anaTitle(context.formatter, 'best efforts · power curve'))
  const controls = el('div', 'tri-best-power-controls', undefined, {
    role: 'group',
    'aria-label': context.formatter.text('power curve controls'),
  })
  const series = bestPowerSeries(power)
  const available = series.filter(({ curve }) => curve.length >= 2)
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
  const hasEstimatedFtp = series.some(
    ({ key, curve }) => curve.length >= 2 && estimatedFtpForSeries(power, key) != null,
  )
  const ftpToggle = el('button', 'tri-best-power-toggle tri-best-power-eftp-toggle', 'eFTP', {
    type: 'button',
    'aria-label': context.formatter.text('estimated FTP'),
    'aria-pressed': String(hasEstimatedFtp),
    title: context.formatter.text(
      hasEstimatedFtp
        ? 'eFTP = 95% of best recorded 20-minute power. Provisional: effort may be submaximal.'
        : 'A recorded 20-minute power effort is required to estimate FTP.',
    ),
    ...(hasEstimatedFtp ? {} : { disabled: '' }),
  })
  controls.appendChild(ftpToggle)
  if (available.length > 0) controls.appendChild(buildPowerUnits(power, context))
  head.appendChild(controls)
  block.appendChild(head)

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
  const domainMax = Math.ceil(observedMax / POWER_CURVE_WATT_STEP) * POWER_CURVE_WATT_STEP
  const X = (seconds: number): number => powerCurveFraction(seconds, minSeconds, maxSeconds) * W
  const Y = (watts: number): number => H - (watts / domainMax) * (H - 1)
  const yTicks = powerCurveAxisTicks(domainMax, null, context.presentation.locale).map(tick => ({
    label: tick.label,
    vbY: Y(tick.watts),
  }))
  const durations = powerCurveDurationTicks(
    minSeconds,
    maxSeconds,
    [1, 15, 60, 300, 600, 1_200, 1_800, 2_700, 3_600, 5_400, 7_200, 10_800, 14_400, 18_000],
  )
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
  const factory = createDomFactory(context.presentation)
  graph.append(
    buildPowerCurveGrid(
      factory,
      'power',
      yTicks.map(tick => tick.vbY),
      H,
    ),
    buildPowerCurveGrid(factory, 'duration', durations.map(X), H),
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
  for (const { key } of available) {
    const estimate = estimatedFtpForSeries(power, key)
    if (!estimate) continue
    graph.appendChild(
      svg('line', {
        class: `tri-best-power-eftp tri-best-power-eftp--${key}`,
        x1: 0,
        y1: Y(estimate.watts).toFixed(2),
        x2: W,
        y2: Y(estimate.watts).toFixed(2),
        'data-power-eftp-series': key,
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

  const durationTicks: AxisXTick[] = durations.map((seconds, index, ticks) => ({
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

  curvePane.appendChild(buildPowerCaptions(power, context))
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
