import type { PowerCurvePoint, StravaActivityDetail } from '../../plugins/stores/strava'
import type { DetailCtx, TriNodeFactory } from './construction-contracts'
import { criticalPowerCurve } from '../../plugins/stores/critical-power'
import { encodePowerCurves } from '../power-curve'
import { axisFrame, niceTicks } from './chart-geometry'
import {
  ACTIVITY_COMPARISON_HEIGHT,
  ACTIVITY_COMPARISON_WIDTH,
  activityCompareColor,
  comparisonChartHead,
  comparisonDurationTicks,
  comparisonGraphAttrs,
  comparisonNumericDomain,
} from './comparison-plot'
import { dlabel } from './metric-text'
import { addPowerCurveThresholdCaption, buildPowerCurveRanges } from './power-curve-captions'
import {
  normalizePowerCurvePoints,
  powerCurveFraction,
  powerCurvePathPoints,
} from './power-curve-data'

export const buildComparisonPowerCurve = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  ctx?: DetailCtx,
): N => {
  const isBike = activities.length > 0 && activities.every(activity => activity.sport === 'bike')
  const curves = activities.map((activity, index) => ({
    activity,
    index,
    points: normalizePowerCurvePoints(activity.powerCurve),
  }))
  const availableCurves = curves.filter(curve => curve.points.length >= 2)
  let minSeconds = 1
  let maxSeconds = 2
  if (availableCurves.length > 0) {
    minSeconds = availableCurves[0].points[0].s
    maxSeconds = availableCurves[0].points[availableCurves[0].points.length - 1].s
    for (let index = 1; index < availableCurves.length; index++) {
      const points = availableCurves[index].points
      minSeconds = Math.min(minSeconds, points[0].s)
      maxSeconds = Math.max(maxSeconds, points[points.length - 1].s)
    }
  }
  const sixWeekReference =
    isBike && availableCurves.length > 0
      ? normalizePowerCurvePoints(ctx?.curveRef ?? null).filter(
          point => point.s >= minSeconds && point.s <= maxSeconds,
        )
      : []
  const yearReference =
    isBike && availableCurves.length > 0
      ? normalizePowerCurvePoints(ctx?.curveYearRef ?? null).filter(
          point => point.s >= minSeconds && point.s <= maxSeconds,
        )
      : []
  const sixWeekModel =
    isBike && availableCurves.length > 0 && ctx?.criticalPower
      ? criticalPowerCurve(ctx.criticalPower, minSeconds, maxSeconds)
      : []
  const yearModel =
    isBike && availableCurves.length > 0 && ctx?.criticalPowerYear
      ? criticalPowerCurve(ctx.criticalPowerYear, minSeconds, maxSeconds)
      : []
  const defaultRange = sixWeekReference.length >= 2 ? 'six-weeks' : 'year'
  const reference = defaultRange === 'six-weeks' ? sixWeekReference : yearReference
  const ftp = isBike && availableCurves.length > 0 ? (ctx?.ftp ?? null) : null
  const goalFtp = isBike && availableCurves.length > 0 ? (ctx?.goalFtp ?? null) : null
  const domain = comparisonNumericDomain(
    [
      ...availableCurves.flatMap(curve => curve.points.map(point => point.w)),
      ...sixWeekReference.map(point => point.w),
      ...yearReference.map(point => point.w),
      ...sixWeekModel.map(point => point.w),
      ...yearModel.map(point => point.w),
      ...(!isBike || ctx?.criticalPower == null ? [] : [ctx.criticalPower.criticalPowerWatts]),
      ...(!isBike || ctx?.criticalPowerYear == null
        ? []
        : [ctx.criticalPowerYear.criticalPowerWatts]),
      ...(ftp == null ? [] : [ftp]),
      ...(goalFtp == null ? [] : [goalFtp]),
    ],
    true,
  )
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      'power-curve',
      'power curve',
      minSeconds,
      maxSeconds,
      minSeconds,
      dlabel(minSeconds),
      availableCurves.length,
    ),
    class: 'tri-compare-graph tri-compare-curve-graph',
    'data-compare-chart': 'power-curve',
    'data-domain-x-scale': 'log',
    'data-domain-x-min': minSeconds,
    'data-domain-x-max': maxSeconds,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-available': availableCurves.length,
    'data-selected': activities.length,
    'data-power-curves': encodePowerCurves([sixWeekReference, yearReference]),
    'data-curve-range': defaultRange,
    'data-curve-year': ctx?.curveYear ?? '',
  })
  const X = (seconds: number): number =>
    powerCurveFraction(seconds, minSeconds, maxSeconds) * ACTIVITY_COMPARISON_WIDTH
  const Y = (watts: number): number =>
    ACTIVITY_COMPARISON_HEIGHT -
    ((watts - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1)
  const toPath = (points: PowerCurvePoint[]): string =>
    powerCurvePathPoints(points)
      .map(
        (point, index) =>
          `${index === 0 ? 'M' : 'L'} ${X(point.s).toFixed(2)} ${Y(point.w).toFixed(2)}`,
      )
      .join(' ')
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: value === 0 ? '0' : `${Math.round(value)} W`,
    vbY: Y(value),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-grid',
        x1: 0,
        y1: tick.vbY,
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: tick.vbY,
      }),
    )
  if (sixWeekReference.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-ref',
        d: toPath(sixWeekReference),
        'data-curve-range': 'six-weeks',
        ...(defaultRange === 'six-weeks' ? {} : { hidden: '' }),
      }),
    )
  if (yearReference.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-ref',
        d: toPath(yearReference),
        'data-curve-range': 'year',
        ...(defaultRange === 'year' ? {} : { hidden: '' }),
      }),
    )
  if (sixWeekModel.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-model',
        d: toPath(sixWeekModel),
        'data-critical-power-range': 'six-weeks',
        ...(defaultRange === 'six-weeks' ? {} : { hidden: '' }),
      }),
    )
  if (yearModel.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-model',
        d: toPath(yearModel),
        'data-critical-power-range': 'year',
        ...(defaultRange === 'year' ? {} : { hidden: '' }),
      }),
    )
  for (const [range, estimate] of [
    ['six-weeks', isBike ? (ctx?.criticalPower ?? null) : null],
    ['year', isBike ? (ctx?.criticalPowerYear ?? null) : null],
  ] as const) {
    if (!estimate) continue
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-curve-cp',
        x1: 0,
        y1: Y(estimate.criticalPowerWatts).toFixed(2),
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: Y(estimate.criticalPowerWatts).toFixed(2),
        'data-critical-power-range': range,
        ...(defaultRange === range ? {} : { hidden: '' }),
      }),
    )
  }
  if (ftp != null)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-curve-ftp',
        x1: 0,
        y1: Y(ftp).toFixed(2),
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: Y(ftp).toFixed(2),
      }),
    )
  if (goalFtp != null)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-curve-goal',
        x1: 0,
        y1: Y(goalFtp).toFixed(2),
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: Y(goalFtp).toFixed(2),
      }),
    )
  for (const curve of availableCurves) {
    const path = toPath(curve.points)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': curve.activity.id,
        'data-activity-index': curve.index,
        style: `--tri-compare-color:${activityCompareColor(curve.index)}`,
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: 0,
      y1: 0,
      x2: 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': 'power-curve',
    'data-available': `${availableCurves.length}`,
    'data-selected': `${activities.length}`,
  })
  const ranges = buildPowerCurveRanges(
    f,
    defaultRange,
    ctx?.curveYear ?? null,
    sixWeekReference.length >= 2,
    yearReference.length >= 2,
  )
  f.add(
    chart,
    comparisonChartHead(f, 'power curve', ranges ?? undefined),
    axisFrame(
      f,
      graph,
      yTicks,
      ACTIVITY_COMPARISON_HEIGHT,
      comparisonDurationTicks(minSeconds, maxSeconds, minSeconds),
      true,
      { top: 0, bottom: ACTIVITY_COMPARISON_HEIGHT },
    ),
  )
  if (
    sixWeekReference.length >= 2 ||
    yearReference.length >= 2 ||
    (isBike && ctx?.criticalPower != null) ||
    (isBike && ctx?.criticalPowerYear != null) ||
    ftp != null ||
    goalFtp != null
  ) {
    const cap = f.el('div', 'tri-elev-cap')
    if (reference.length >= 2)
      f.add(
        cap,
        f.el(
          'span',
          'tri-ana-k tri-compare-curve-reference-label',
          defaultRange === 'year' && ctx?.curveYear != null
            ? `${ctx.curveYear} best`
            : '6-week best',
          defaultRange === 'six-weeks' ? { 'data-i18n': '6-week best' } : undefined,
        ),
      )
    addPowerCurveThresholdCaption(
      f,
      cap,
      [
        ['six-weeks', isBike ? (ctx?.criticalPower ?? null) : null],
        ['year', isBike ? (ctx?.criticalPowerYear ?? null) : null],
      ],
      defaultRange,
      ftp,
      goalFtp,
    )
    f.add(chart, cap)
  }
  return chart
}
