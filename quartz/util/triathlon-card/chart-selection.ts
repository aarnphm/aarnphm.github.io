import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { elapsedAxis, type ElapsedPause } from '../elapsed-axis'
import { activityTraceUsesElapsedAxis } from './activity-metric-data'

const routeDistanceAtElapsed = (d: StravaActivityDetail, elapsedS: number): number => {
  const route = d.route
  if (elapsedS <= route[0].elapsedS) return route[0].d
  for (let index = 1; index < route.length; index++) {
    const previous = route[index - 1]
    const next = route[index]
    if (elapsedS > next.elapsedS) continue
    const elapsedSpan = next.elapsedS - previous.elapsedS
    const fraction = elapsedSpan > 0 ? (elapsedS - previous.elapsedS) / elapsedSpan : 1
    return previous.d + (next.d - previous.d) * Math.max(0, Math.min(1, fraction))
  }
  return route[route.length - 1].d
}

export type AnalysisInterval = Pick<
  ActivityAnalysisRange,
  'startElapsedS' | 'endElapsedS' | 'startDistanceKm' | 'endDistanceKm'
>

export type ActivityAnalysisChartDomain =
  | (Pick<AnalysisInterval, 'startElapsedS' | 'endElapsedS'> & { pauses?: readonly ElapsedPause[] })
  | Pick<AnalysisInterval, 'startDistanceKm' | 'endDistanceKm'>

/** Domain position in [0, 1]; an elapsed domain gives its pauses no width. */
export const analysisDomainFraction = (
  domain: ActivityAnalysisChartDomain,
  value: number,
): number => {
  if ('startElapsedS' in domain)
    return elapsedAxis(domain.startElapsedS, domain.endElapsedS, domain.pauses).fraction(value)
  const span = domain.endDistanceKm - domain.startDistanceKm
  return span > 0 ? Math.max(0, Math.min(1, (value - domain.startDistanceKm) / span)) : 0
}

/** Inverse of analysisDomainFraction: elapsed seconds or kilometres at a fraction. */
export const analysisDomainValueAt = (
  domain: ActivityAnalysisChartDomain,
  fraction: number,
): number =>
  'startElapsedS' in domain
    ? elapsedAxis(domain.startElapsedS, domain.endElapsedS, domain.pauses).elapsedAt(fraction)
    : domain.startDistanceKm + fraction * (domain.endDistanceKm - domain.startDistanceKm)

export const analysisChartSelectionBounds = (
  range: AnalysisInterval,
  domain: ActivityAnalysisChartDomain,
  plot = { x: 0, width: 100 },
): { x: number; width: number } => {
  const [domainStart, domainEnd, rangeStart, rangeEnd] =
    'startElapsedS' in domain
      ? [domain.startElapsedS, domain.endElapsedS, range.startElapsedS, range.endElapsedS]
      : [domain.startDistanceKm, domain.endDistanceKm, range.startDistanceKm, range.endDistanceKm]
  if (!(domainEnd - domainStart > 0) || !(plot.width > 0)) return { x: plot.x, width: 0 }
  const start = analysisDomainFraction(domain, rangeStart)
  const end = Math.max(start, analysisDomainFraction(domain, rangeEnd))
  return { x: plot.x + start * plot.width, width: (end - start) * plot.width }
}

export const analysisSelectionBounds = (
  d: StravaActivityDetail,
  range: AnalysisInterval,
): { x: number; width: number } => {
  if (activityTraceUsesElapsedAxis(d)) {
    const maxElapsedS = d.heartRateTrace.at(-1)?.elapsedS || d.elapsedTimeS || 1
    return analysisChartSelectionBounds(range, { startElapsedS: 0, endElapsedS: maxElapsedS })
  }
  const hasRoute = d.route.length >= 2
  const maxD = hasRoute
    ? d.route[d.route.length - 1].d || 1
    : Math.max(d.distanceKm, range.endDistanceKm) || 1
  const startDistanceKm = hasRoute
    ? routeDistanceAtElapsed(d, range.startElapsedS)
    : range.startDistanceKm
  const endDistanceKm = hasRoute
    ? routeDistanceAtElapsed(d, range.endElapsedS)
    : range.endDistanceKm
  return analysisChartSelectionBounds(
    { ...range, startDistanceKm, endDistanceKm },
    { startDistanceKm: 0, endDistanceKm: maxD },
  )
}

export const buildAnalysisSelectionRect = <N>(
  f: TriNodeFactory<N>,
  height: number,
  bounds = { x: 0, width: 0 },
  y = 0,
): N =>
  f.svg('rect', {
    class: 'tri-analysis-selection',
    x: bounds.x.toFixed(2),
    y,
    width: bounds.width.toFixed(2),
    height,
  })

export const buildAnalysisSelection = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  height: number,
  range: ActivityAnalysisRange | null,
): N => {
  const bounds = range ? analysisSelectionBounds(d, range) : { x: 0, width: 0 }
  return buildAnalysisSelectionRect(f, height, bounds)
}
