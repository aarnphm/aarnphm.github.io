import type { GarminRunWalkSegment } from '../../plugins/stores/garmin'
import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { triathlonTraceName } from '../triathlon-trace-settings'
import { axisFrame, elapsedActivityXTicks } from './chart-geometry'
import { buildAnalysisSelectionRect } from './chart-selection'
import { clock } from './metric-text'

const formatRunWalkTime = (seconds: number): string => {
  if (seconds >= 60) return clock(Math.round(seconds))
  return `0:${seconds.toFixed(1).padStart(4, '0')}`
}

export const runWalkSegmentAt = (
  segments: readonly GarminRunWalkSegment[],
  elapsedS: number,
): GarminRunWalkSegment | null => {
  if (!Number.isFinite(elapsedS)) return null
  return (
    segments.find(
      (segment, index) =>
        elapsedS >= segment.startElapsedS &&
        (elapsedS < segment.endElapsedS ||
          (index === segments.length - 1 && elapsedS === segment.endElapsedS)),
    ) ?? null
  )
}

export const buildRunWalkTrace = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const runWalk = d.runWalk
  if (d.sport !== 'run' || !runWalk || runWalk.segments.length === 0 || runWalk.elapsedTimeS <= 0)
    return null
  const width = 100
  const height = 30
  const rowY = (state: 'run' | 'walk' | 'idle'): number =>
    state === 'idle' ? 1 : state === 'walk' ? 11 : 21
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-run-walk',
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: 'none',
    'data-domain-start-elapsed-s': 0,
    'data-domain-end-elapsed-s': runWalk.elapsedTimeS,
    tabindex: 0,
    role: 'slider',
    'aria-label': triText(f.presentation.locale, 'run/walk'),
    'aria-valuemin': 0,
    'aria-valuemax': runWalk.elapsedTimeS,
    'aria-valuenow': 0,
    'aria-valuetext': `0:00 · ${triText(f.presentation.locale, runWalkSegmentAt(runWalk.segments, 0)?.state ?? 'unavailable')}`,
  })
  for (const segment of runWalk.segments) {
    const x = (segment.startElapsedS / runWalk.elapsedTimeS) * width
    const segmentWidth =
      ((segment.endElapsedS - segment.startElapsedS) / runWalk.elapsedTimeS) * width
    f.add(
      svgEl,
      f.svg('rect', {
        class: `tri-run-walk-segment tri-run-walk-segment--${segment.state}`,
        x: x.toFixed(4),
        y: rowY(segment.state),
        width: segmentWidth.toFixed(4),
        height: 8,
        'data-run-walk-state': segment.state,
        'data-start-elapsed-s': segment.startElapsedS,
        'data-end-elapsed-s': segment.endElapsedS,
      }),
    )
  }
  f.add(
    svgEl,
    buildAnalysisSelectionRect(f, height),
    f.svg('line', { class: 'tri-chart-cursor tri-elev-cursor', x1: 0, x2: 0, y1: 0, y2: height }),
  )
  const wrap = f.el('div', 'tri-elev-wrap tri-run-walk-chart', undefined, {
    'data-tri-trace': triathlonTraceName('run/walk'),
    'data-run-walk-source': runWalk.source,
    'aria-label': `run ${formatRunWalkTime(runWalk.runTimeS)}, walk ${formatRunWalkTime(runWalk.walkTimeS)}, idle ${formatRunWalkTime(runWalk.idleTimeS)}`,
  })
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-run-walk-summary')
  const totals: readonly { state: 'run' | 'walk' | 'idle'; seconds: number }[] = [
    { state: 'run', seconds: runWalk.runTimeS },
    { state: 'walk', seconds: runWalk.walkTimeS },
    { state: 'idle', seconds: runWalk.idleTimeS },
  ]
  for (const { state, seconds } of totals) {
    const total = f.el('span', `tri-run-walk-total tri-run-walk-total--${state}`)
    f.add(
      total,
      f.el('span', 'tri-run-walk-label', triText(f.presentation.locale, state), {
        'data-i18n': state,
      }),
      f.el('span', 'tri-run-walk-time', formatRunWalkTime(seconds)),
    )
    f.add(summary, total)
  }
  f.add(
    cap,
    f.el('span', 'tri-elev-d', triText(f.presentation.locale, 'run/walk')),
    summary,
    f.el('span', 'tri-fig-readout tri-chart-readout'),
  )
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      svgEl,
      [
        { label: triText(f.presentation.locale, 'idle'), vbY: 5 },
        { label: triText(f.presentation.locale, 'walk'), vbY: 15 },
        { label: triText(f.presentation.locale, 'run'), vbY: 25 },
      ],
      height,
      elapsedActivityXTicks(runWalk.elapsedTimeS),
      true,
      { top: 0, bottom: height },
    ),
  )
  return wrap
}
