import type { PowerCurvePoint, StravaActivityDetail } from '../../plugins/stores/strava'
import type { DetailCtx, TriNodeFactory } from './construction-contracts'
import { criticalPowerCurve, type CriticalPowerEstimate } from '../../plugins/stores/critical-power'
import { encodePowerCurves } from '../power-curve'
import { triText } from '../triathlon-i18n'
import { powerCurveActivityLinkAttributes } from '../triathlon-power-activity'
import { triathlonTraceName } from '../triathlon-trace-settings'
import { axisFrame, buildPowerCurveGrid } from './chart-geometry'
import { dlabel, zoneClock } from './metric-text'
import {
  addActivityCriticalPowerCaption,
  buildPowerCurveRanges,
  powerCurveValueNode,
} from './power-curve-captions'
import {
  embeddedPowerCurveDurationTicks,
  nearestPowerCurvePoint,
  POWER_CURVE_WATT_STEP,
  powerCurveAxisTicks,
  powerCurveDurationTicks,
  powerCurveFraction,
  powerCurvePathPoints,
  powerCurveWeight,
} from './power-curve-data'

export const buildPowerCurve = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  ctx: DetailCtx,
  embedded = false,
): N | null => {
  const curve = d.powerCurve
  if (!curve || curve.length < 2) return null
  const weight = powerCurveWeight(d)
  const isBike = d.sport === 'bike'
  const sixWeekRef = isBike ? ctx.curveRef : d.sport === 'run' ? ctx.runCurveRef : []
  const yearRef = isBike ? ctx.curveYearRef : d.sport === 'run' ? ctx.runCurveYearRef : []
  const activityLabel = d.sport === 'run' ? 'this run' : 'this ride'
  const activityCriticalPower = isBike ? d.activityCriticalPower : null
  const activityModel =
    activityCriticalPower != null
      ? criticalPowerCurve(activityCriticalPower, curve[0].s, curve[curve.length - 1].s)
      : []
  const ftpRef = isBike ? ctx.ftp : null
  const goalRef = isBike ? ctx.goalFtp : null
  const wrap = f.el('div', 'tri-zone tri-curve-chart', undefined, {
    'data-tri-trace': triathlonTraceName('power curve'),
    'data-curve-unit': 'watts',
  })
  const W = 100
  const H = 34
  const secs = curve.map(c => c.s)
  const visibleSixWeekRef = sixWeekRef.filter(c => c.s >= secs[0] && c.s <= secs[secs.length - 1])
  const visibleYearRef = yearRef.filter(c => c.s >= secs[0] && c.s <= secs[secs.length - 1])
  const defaultRange = visibleSixWeekRef.length > 0 ? 'six-weeks' : 'year'
  const visibleRef = defaultRange === 'six-weeks' ? visibleSixWeekRef : visibleYearRef
  const head = f.el('div', 'tri-curve-head')
  f.add(head, f.el('div', 'tri-zone-title', 'power curve', { 'data-i18n': 'power curve' }))
  const controls = f.el('div', 'tri-curve-controls')
  const units = f.el('div', 'tri-curve-units', undefined, {
    role: 'group',
    'aria-label': triText(f.presentation.locale, 'power curve units'),
    'data-i18n-aria-label': 'power curve units',
  })
  for (const [unit, label] of [
    ['watts', 'W'],
    ['watts-per-kg', 'W/kg'],
  ])
    f.add(
      units,
      f.el('button', 'tri-curve-unit', label, {
        type: 'button',
        'data-curve-unit': unit,
        'aria-pressed': String(unit === 'watts'),
        ...(unit === 'watts-per-kg' && !weight
          ? {
              disabled: '',
              title: triText(
                f.presentation.locale,
                'W/kg unavailable: no weight recorded on or before this activity',
              ),
            }
          : {}),
      }),
    )
  const ranges = buildPowerCurveRanges(
    f,
    defaultRange,
    ctx.curveYear,
    visibleSixWeekRef.length > 0,
    visibleYearRef.length > 0,
  )
  if (ranges) f.add(controls, ranges)
  f.add(controls, units)
  f.add(head, controls)
  f.add(wrap, head)
  let observedMaxW = Math.max(
    1,
    activityCriticalPower?.criticalPowerWatts ?? 0,
    ftpRef ?? 0,
    goalRef ?? 0,
  )
  for (const points of [curve, visibleSixWeekRef, visibleYearRef, activityModel])
    for (const point of points) observedMaxW = Math.max(observedMaxW, point.w)
  const curveMax = Math.ceil(observedMaxW / POWER_CURVE_WATT_STEP) * POWER_CURVE_WATT_STEP
  const X = (sec: number): number => powerCurveFraction(sec, secs[0], secs[secs.length - 1]) * W
  const Y = (w: number): number => H - (w / curveMax) * (H - 1)
  const powerTicks = powerCurveAxisTicks(curveMax, null, f.presentation.locale).map(tick => ({
    label: tick.label,
    vbY: Y(tick.watts),
  }))
  const durationMarkers = [1, 60, 300, 1200, 3600, 10_800]
  const curveDurTicks = embedded
    ? embeddedPowerCurveDurationTicks(secs[0], secs[secs.length - 1], durationMarkers)
    : powerCurveDurationTicks(secs[0], secs[secs.length - 1], durationMarkers)
  const toPath = (pts: PowerCurvePoint[]): string =>
    powerCurvePathPoints(pts)
      .map((c, i) => `${i ? 'L' : 'M'} ${X(c.s).toFixed(2)} ${Y(c.w).toFixed(2)}`)
      .join(' ')
  const initialValueText = `${zoneClock(curve[0].s)} · ${curve[0].w.toLocaleString('en-US')} W`
  const s = f.svg('svg', {
    class: 'tri-curve-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    'data-power-curves': encodePowerCurves([curve, visibleSixWeekRef, visibleYearRef]),
    'data-curve-sport': d.sport,
    'data-curve-range': defaultRange,
    'data-curve-year': ctx.curveYear ?? '',
    'data-curve-domain-max': curveMax,
    'data-curve-weight-kg': weight?.kg ?? '',
    'data-curve-selected-index': 0,
    'data-i18n-aria-label': 'power curve',
    role: 'slider',
    tabindex: 0,
    'aria-label': 'power curve',
    'aria-orientation': 'horizontal',
    'aria-valuemin': curve[0].s,
    'aria-valuemax': curve[curve.length - 1].s,
    'aria-valuenow': curve[0].s,
    'aria-valuetext': initialValueText,
  })
  f.add(
    s,
    buildPowerCurveGrid(
      f,
      'power',
      powerTicks.map(tick => tick.vbY),
      H,
    ),
    buildPowerCurveGrid(f, 'duration', curveDurTicks.map(X), H),
  )
  if (visibleSixWeekRef.length >= 2)
    f.add(
      s,
      f.svg('path', {
        d: toPath(visibleSixWeekRef),
        class: 'tri-curve-ref',
        'data-curve-range': 'six-weeks',
        ...(defaultRange === 'six-weeks' ? {} : { hidden: '' }),
      }),
    )
  if (visibleYearRef.length >= 2)
    f.add(
      s,
      f.svg('path', {
        d: toPath(visibleYearRef),
        class: 'tri-curve-ref',
        'data-curve-range': 'year',
        ...(defaultRange === 'year' ? {} : { hidden: '' }),
      }),
    )
  if (activityModel.length >= 2)
    f.add(
      s,
      f.svg('path', {
        d: toPath(activityModel),
        class: 'tri-curve-model tri-curve-model--ride',
        'aria-hidden': 'true',
      }),
    )
  if (activityCriticalPower)
    f.add(
      s,
      f.svg('line', {
        x1: 0,
        y1: Y(activityCriticalPower.criticalPowerWatts).toFixed(2),
        x2: W,
        y2: Y(activityCriticalPower.criticalPowerWatts).toFixed(2),
        class: 'tri-curve-cp tri-curve-cp--ride',
        'aria-hidden': 'true',
      }),
    )
  if (ftpRef != null)
    f.add(
      s,
      f.svg('line', {
        x1: 0,
        y1: Y(ftpRef).toFixed(2),
        x2: W,
        y2: Y(ftpRef).toFixed(2),
        class: 'tri-curve-ftp',
      }),
    )
  if (goalRef != null)
    f.add(
      s,
      f.svg('line', {
        x1: 0,
        y1: Y(goalRef).toFixed(2),
        x2: W,
        y2: Y(goalRef).toFixed(2),
        class: 'tri-curve-goal',
      }),
    )
  f.add(s, f.svg('path', { d: toPath(curve), class: 'tri-curve-line' }))
  f.add(s, f.svg('line', { class: 'tri-chart-cursor', x1: 0, y1: 0, x2: 0, y2: H }))
  const pointMarkers: N[] = []
  if (visibleRef.length > 0) {
    const initialRef = visibleRef.find(point => point.s === curve[0].s)
    const attrs: Record<string, string> = {
      'aria-hidden': 'true',
      style: `left:${X(curve[0].s).toFixed(2)}%;top:${((Y(initialRef?.w ?? 0) / H) * 100).toFixed(2)}%`,
    }
    if (!initialRef) attrs.hidden = ''
    pointMarkers.push(f.el('span', 'tri-curve-point tri-curve-point--ref', undefined, attrs))
  }
  pointMarkers.push(
    f.el('span', 'tri-curve-point tri-curve-point--ride', undefined, {
      'aria-hidden': 'true',
      style: `left:${X(curve[0].s).toFixed(2)}%;top:${((Y(curve[0].w) / H) * 100).toFixed(2)}%`,
    }),
  )
  const readout = f.el('div', 'tri-chart-readout tri-curve-readout')
  f.add(readout, f.el('span', 'tri-curve-readout-duration'))
  const rideRow = f.el('span', 'tri-curve-readout-row')
  f.add(
    rideRow,
    f.el('span', 'tri-curve-readout-swatch tri-curve-readout-swatch--ride', undefined, {
      'aria-hidden': 'true',
    }),
    f.el('strong', 'tri-curve-readout-value tri-curve-readout-value--ride'),
    f.el('span', 'tri-curve-readout-label', activityLabel, { 'data-i18n': activityLabel }),
  )
  f.add(readout, rideRow)
  if (visibleRef.length > 0) {
    const referenceRow = f.el(
      'a',
      'tri-curve-readout-row tri-curve-readout-row--ref',
      undefined,
      powerCurveActivityLinkAttributes(nearestPowerCurvePoint(visibleRef, curve[0].s), d.id),
    )
    f.add(
      referenceRow,
      f.el('span', 'tri-curve-readout-swatch tri-curve-readout-swatch--ref', undefined, {
        'aria-hidden': 'true',
      }),
      f.el('strong', 'tri-curve-readout-value tri-curve-readout-value--ref'),
      f.el(
        'span',
        'tri-curve-readout-label tri-curve-readout-label--ref',
        defaultRange === 'year' && ctx.curveYear != null ? `${ctx.curveYear} best` : '6-week best',
        defaultRange === 'six-weeks' ? { 'data-i18n': '6-week best' } : undefined,
      ),
    )
    f.add(readout, referenceRow)
  }
  const addCriticalPowerReadout = (estimate: CriticalPowerEstimate, label: string): void => {
    const durations = estimate.anchors.map(anchor => anchor.durationS)
    const minDuration = Math.min(...durations)
    const maxDuration = Math.max(...durations)
    const attrs: Record<string, string> = {
      'data-curve-critical-power': String(estimate.criticalPowerWatts),
      'data-curve-w-prime': String(estimate.wPrimeJoules),
      'data-curve-model-min-seconds': String(minDuration),
      'data-curve-model-max-seconds': String(maxDuration),
      hidden: '',
    }
    const row = f.el(
      'span',
      'tri-curve-readout-row tri-curve-readout-row--model tri-curve-readout-row--model-ride',
      undefined,
      attrs,
    )
    f.add(
      row,
      f.el(
        'span',
        'tri-curve-readout-swatch tri-curve-readout-swatch--model tri-curve-readout-swatch--model-ride',
        undefined,
        { 'aria-hidden': 'true' },
      ),
      f.el('strong', 'tri-curve-readout-value tri-curve-readout-value--model'),
      f.el('span', 'tri-curve-readout-label tri-curve-readout-label--model', label),
    )
    f.add(readout, row)
  }
  if (activityCriticalPower)
    addCriticalPowerReadout(
      activityCriticalPower,
      triText(f.presentation.locale, 'this ride eCP model'),
    )
  f.add(
    wrap,
    axisFrame(
      f,
      s,
      powerTicks,
      H,
      curveDurTicks.map((sec, idx) => ({
        label: dlabel(sec),
        pct: X(sec),
        cls: `tri-curve-tick${idx === 0 ? ' tri-cax-xt--first' : sec === secs[secs.length - 1] ? ' tri-cax-xt--last' : ''}`,
        tag: 'button',
        attrs: {
          type: 'button',
          'data-curve-seconds': String(sec),
          'aria-pressed': String(sec === curve[0].s),
        },
      })),
      true,
      undefined,
      [...pointMarkers, readout],
    ),
  )
  const cap = f.el('div', 'tri-elev-cap')
  const addPowerCaption = (label: string, watts: number, cls = ''): void => {
    const item = f.el('span', `tri-ana-k${cls ? ` ${cls}` : ''}`)
    f.add(item, f.el('span', undefined, `${label} `), powerCurveValueNode(f, watts))
    f.add(cap, item)
  }
  for (const sec of [5, 60, 300, 1200]) {
    const p = curve.find(c => c.s === sec)
    if (p) addPowerCaption(dlabel(sec), p.w)
  }
  if (!embedded && ftpRef != null) addPowerCaption('FTP', ftpRef, 'tri-curve-ftp-k')
  if (!embedded && goalRef != null) addPowerCaption('goal', goalRef, 'tri-curve-goal-k')
  const criticalPowerCaption =
    !embedded && activityCriticalPower
      ? addActivityCriticalPowerCaption(f, cap, activityCriticalPower)
      : undefined
  if (weight)
    f.add(
      criticalPowerCaption ?? cap,
      f.el(
        'span',
        'tri-curve-weight-note',
        `${criticalPowerCaption ? ' · ' : ''}${weight.kg} kg${weight.date < d.date ? ` · ${weight.date}` : ''}`,
        { hidden: '', title: `Garmin · ${weight.date}` },
      ),
    )
  f.add(wrap, cap)
  return wrap
}
