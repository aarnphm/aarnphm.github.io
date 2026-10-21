import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { triathlonTraceName } from '../triathlon-trace-settings'
import { activityTraceUsesElapsedAxis } from './activity-metric-data'
import {
  axisFrame,
  distanceXTicks,
  elapsedActivityXTicks,
  graphViewForDistance,
  niceTicks,
  swimActivityXTicks,
  type ActivityGraphDomain,
  type ActivityTraceDomain,
  type ActivityTraceReference,
  type ActivityTraceRenderOptions,
} from './chart-geometry'
import { buildAnalysisSelection } from './chart-selection'

export const buildTraceSeries = <N, P extends { d: number; elapsedS: number }>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  points: readonly P[],
  pick: (point: P, index: number) => number | null,
  title: string,
  cap: (max: number) => string,
  tick: (value: number) => string,
  domain?: ActivityTraceDomain,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  missing?: 'dotted',
  reference?: ActivityTraceReference | null,
  options?: ActivityTraceRenderOptions<N>,
): N => {
  const w = 100
  const h = 30
  const maxD = points.at(-1)?.d || d.distanceKm || 1
  let peak = 1
  const values = points.map(pick)
  values.forEach(value => {
    if (value != null && Number.isFinite(value) && value > peak) peak = value
  })
  const domainMin = Math.min(domain?.min ?? 0, reference?.value ?? Number.POSITIVE_INFINITY)
  const candidateMax = Math.max(domain?.max ?? peak, reference?.value ?? Number.NEGATIVE_INFINITY)
  const domainMax = candidateMax > domainMin ? candidateMax : domainMin + 1
  const px = (km: number): number => (km / maxD) * w
  const py = (v: number): number =>
    Math.min(h, Math.max(0, h - ((v - domainMin) / (domainMax - domainMin)) * (h - 1)))
  const areaBaselineY = options?.areaBaseline == null ? h : py(options.areaBaseline)
  let area = ''
  let line = ''
  let missingLine = ''
  let segmentStart = -1
  const closeSegment = (start: number, end: number): void => {
    const first = values[start]
    if (first == null) return
    const startX = start === 0 ? 0 : px(points[start].d)
    const firstX = px(points[start].d)
    const firstY = py(first).toFixed(2)
    area += `M ${startX.toFixed(2).replace('.00', '')} ${areaBaselineY.toFixed(2).replace('.00', '')} L ${startX.toFixed(2).replace('.00', '')} ${firstY} `
    line += `M ${startX.toFixed(2).replace('.00', '')} ${firstY} `
    if (firstX !== startX) {
      area += `L ${firstX.toFixed(2)} ${firstY} `
      line += `L ${firstX.toFixed(2)} ${firstY} `
    }
    for (let i = start + 1; i <= end; i++) {
      const value = values[i]
      if (value == null) continue
      const x = px(points[i].d).toFixed(2)
      const y = py(value).toFixed(2)
      area += `L ${x} ${y} `
      line += `L ${x} ${y} `
    }
    area += `L ${px(points[end].d).toFixed(2)} ${areaBaselineY.toFixed(2).replace('.00', '')} Z `
  }
  values.forEach((value, index) => {
    const valid = value != null && Number.isFinite(value)
    if (valid && segmentStart < 0) segmentStart = index
    if (segmentStart >= 0 && (!valid || index === values.length - 1)) {
      closeSegment(segmentStart, valid ? index : index - 1)
      segmentStart = -1
    }
  })
  if (missing === 'dotted') {
    let previous = -1
    values.forEach((value, index) => {
      if (value == null || !Number.isFinite(value)) return
      const x = px(points[index].d).toFixed(2)
      const y = py(value).toFixed(2)
      if (previous < 0 && index > 0) missingLine += `M 0 ${y} L ${x} ${y} `
      else if (previous >= 0 && index > previous + 1) {
        const previousValue = values[previous]
        if (previousValue != null) {
          missingLine += `M ${px(points[previous].d).toFixed(2)} ${py(previousValue).toFixed(2)} L ${x} ${y} `
        }
      }
      previous = index
    })
    if (previous >= 0 && previous < values.length - 1) {
      const value = values[previous]
      if (value != null)
        missingLine += `M ${px(points[previous].d).toFixed(2)} ${py(value).toFixed(2)} L ${w} ${py(value).toFixed(2)} `
    }
  }
  const yTicks = niceTicks(domainMin, domainMax, domain?.intervals ?? 3).map(value => ({
    label: value === 0 ? '0' : tick(value),
    vbY: py(value),
  }))
  const view = graphViewForDistance(maxD, graphDomain)
  const usesElapsedAxis = activityTraceUsesElapsedAxis(d)
  const s = f.svg('svg', {
    class: 'tri-elev',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${h}`,
    preserveAspectRatio: 'none',
    ...(usesElapsedAxis
      ? {
          'data-domain-start-elapsed-s': view.startDistanceKm,
          'data-domain-end-elapsed-s': view.endDistanceKm,
        }
      : {
          'data-domain-start-distance-km': view.startDistanceKm,
          'data-domain-end-distance-km': view.endDistanceKm,
        }),
  })
  for (const t of yTicks)
    f.add(s, f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: t.vbY, x2: w, y2: t.vbY }))
  f.add(s, f.svg('path', { d: area, class: 'tri-elev-area' }))
  if (selection !== undefined) f.add(s, buildAnalysisSelection(f, d, h, selection))
  if (missingLine)
    f.add(s, f.svg('path', { d: missingLine, class: 'tri-elev-line tri-elev-line--missing' }))
  if (reference)
    f.add(
      s,
      f.svg('line', {
        class: 'tri-trace-reference',
        x1: 0,
        y1: py(reference.value).toFixed(2),
        x2: w,
        y2: py(reference.value).toFixed(2),
      }),
    )
  f.add(s, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(s, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el(
    'div',
    `tri-elev-wrap${options?.wrapClass ? ` ${options.wrapClass}` : ''}`,
    undefined,
    { 'data-tri-trace': triathlonTraceName(options?.traceName ?? title), ...options?.wrapAttrs },
  )
  const capEl = f.el('div', 'tri-elev-cap')
  const capText = cap(peak)
  f.add(
    capEl,
    f.el('span', 'tri-elev-d', triText(f.presentation.locale, title)),
    capText.includes('$')
      ? f.math('tri-elev-range', capText)
      : f.el('span', 'tri-elev-range', capText),
  )
  if (reference) f.add(capEl, f.el('span', 'tri-elev-range tri-trace-reference-k', reference.label))
  if (options?.capExtra) f.add(capEl, ...options.capExtra)
  f.add(
    wrap,
    capEl,
    axisFrame(
      f,
      s,
      yTicks,
      h,
      usesElapsedAxis
        ? elapsedActivityXTicks(maxD)
        : d.sport === 'swim'
          ? swimActivityXTicks(view.endDistanceKm * 1000, view.startDistanceKm * 1000)
          : distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: h },
    ),
  )
  return wrap
}

export const buildTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  pick: (p: StravaActivityDetail['route'][number], i: number) => number | null,
  title: string,
  cap: (max: number) => string,
  tick: (value: number) => string,
  domain?: ActivityTraceDomain,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  missing?: 'dotted',
  reference?: ActivityTraceReference | null,
  options?: ActivityTraceRenderOptions<N>,
): N =>
  buildTraceSeries(
    f,
    d,
    d.route,
    pick,
    title,
    cap,
    tick,
    domain,
    d.route.length >= 2 ? selection : undefined,
    graphDomain,
    missing,
    reference,
    options,
  )
