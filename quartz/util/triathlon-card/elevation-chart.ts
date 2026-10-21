import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { gradeAt } from './activity-metric-data'
import {
  axisFrame,
  axisNumber,
  distanceXTicks,
  graphView,
  niceStep,
  niceTicks,
  type ActivityGraphDomain,
} from './chart-geometry'
import { buildAnalysisSelection } from './chart-selection'
import { elevationValue, formatAltitude, formatElevationGain, isImperial } from './metric-text'

export type ClimbGradeBand = 'gentle' | 'moderate' | 'hard' | 'steep' | 'wall'

export const climbGradeBand = (gradePct: number): ClimbGradeBand | null => {
  if (!Number.isFinite(gradePct) || gradePct <= 0) return null
  if (gradePct < 4) return 'gentle'
  if (gradePct < 8) return 'moderate'
  if (gradePct < 12) return 'hard'
  if (gradePct < 20) return 'steep'
  return 'wall'
}

type ElevationGradePoint = { x: number; y: number }

type ElevationGradeRun = { band: ClimbGradeBand; points: ElevationGradePoint[] }

const elevationGradePaths = (
  route: StravaActivityDetail['route'],
  px: (distanceKm: number) => number,
  py: (altitudeM: number) => number,
  baseline: number,
): { band: ClimbGradeBand; path: string }[] => {
  const runs: ElevationGradeRun[] = []
  let activeRunIndex = -1
  for (let index = 1; index < route.length; index += 1) {
    const previous = route[index - 1]
    const current = route[index]
    const band =
      current.d > previous.d && current.alt > previous.alt
        ? climbGradeBand(gradeAt(route, index))
        : null
    if (band == null) {
      activeRunIndex = -1
      continue
    }
    const currentPoint = { x: px(current.d), y: py(current.alt) }
    const activeRun = runs[activeRunIndex]
    if (activeRun?.band === band) {
      activeRun.points.push(currentPoint)
      continue
    }
    runs.push({ band, points: [{ x: px(previous.d), y: py(previous.alt) }, currentPoint] })
    activeRunIndex = runs.length - 1
  }
  return runs.map(run => {
    const first = run.points[0]
    const last = run.points.at(-1) ?? first
    return {
      band: run.band,
      path: `M ${first.x.toFixed(2)} ${baseline} ${run.points.map(point => `L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')} L ${last.x.toFixed(2)} ${baseline} Z`,
    }
  })
}

export const buildElevation = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  domain?: ActivityGraphDomain | null,
): N => {
  const w = 100
  const h = 30
  const maxD = d.route[d.route.length - 1].d || 1
  const imperial = isImperial(f.presentation)
  const displayMinAlt = elevationValue(f.presentation, d.minAlt)
  const displayMaxAlt = elevationValue(f.presentation, d.maxAlt)
  const altPad = displayMinAlt === displayMaxAlt ? 1 : 0
  const minAlt = displayMinAlt - altPad
  const maxAlt = displayMaxAlt + altPad
  const altSpan = Math.max(1e-6, maxAlt - minAlt)
  const px = (km: number): number => (km / maxD) * w
  const py = (alt: number): number =>
    h - ((elevationValue(f.presentation, alt) - minAlt) / altSpan) * h
  const yValues = niceTicks(minAlt, maxAlt, 4)
  const yStep = niceStep(maxAlt - minAlt, 4)
  const yTicks = yValues.map(value => ({
    label: `${axisNumber(value, yStep)} ${imperial ? 'ft' : 'm'}`,
    vbY: h - ((value - minAlt) / altSpan) * h,
  }))
  const view = graphView(d, domain)
  const xTicks = distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm)
  const gradePaths = elevationGradePaths(d.route, px, py, h)
  let area = `M 0 ${h} `
  let line = ''
  d.route.forEach((p, i) => {
    area += `L ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
    line += `${i ? 'L' : 'M'} ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
  })
  area += `L ${w} ${h} Z`
  const fig = f.svg('svg', {
    class: 'tri-elev',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${h}`,
    preserveAspectRatio: 'none',
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(fig, f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: w, y2: tick.vbY }))
  f.add(fig, f.svg('path', { d: area, class: 'tri-elev-area' }))
  if (gradePaths.length > 0) {
    const grades = f.svg('g', { class: 'tri-elev-grades', 'aria-hidden': 'true' })
    for (const grade of gradePaths)
      f.add(
        grades,
        f.svg('path', {
          class: `tri-elev-grade tri-elev-grade--${grade.band}`,
          d: grade.path,
          'data-grade-band': grade.band,
        }),
      )
    f.add(fig, grades)
  }
  if (selection !== undefined) f.add(fig, buildAnalysisSelection(f, d, h, selection))
  f.add(fig, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(fig, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el('div', 'tri-elev-wrap')
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  f.add(
    cap,
    f.el('span', 'tri-elev-d', `+${formatElevationGain(f.presentation, d.elevationM)}`),
    f.el('span', 'tri-elev-d', `−${formatElevationGain(f.presentation, d.descentM)}`),
    f.el(
      'span',
      'tri-elev-range',
      `${formatAltitude(f.presentation, d.minAlt)}–${formatAltitude(f.presentation, d.maxAlt)}`,
    ),
  )
  const frame = axisFrame(f, fig, yTicks, h, xTicks, true, { top: 0, bottom: h })
  f.add(wrap, frame, cap)
  return wrap
}

export const buildUnavailableElevation = <N>(f: TriNodeFactory<N>): N => {
  const wrap = f.el('div', 'tri-elev-wrap tri-elev-wrap--unavailable')
  const unavailable = f.el('div', 'tri-elev-unavailable', 'no data available', {
    'data-i18n': 'no data available',
  })
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--unavailable', undefined, {
    'aria-hidden': 'true',
  })
  f.add(wrap, axisFrame(f, unavailable, [], 30, [], false), cap)
  return wrap
}
