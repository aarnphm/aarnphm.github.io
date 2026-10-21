import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { activityHeartRateTracePoints, hasHeartRateTrace } from './activity-metric-data'
import { validAnalysisRanges } from './analysis-data'
import { analysisRangeAttrs, analysisRangeMetrics } from './analysis-presentation'
import { analysisSelectionBounds } from './chart-selection'
import { buildHeartRateTrace } from './effort-traces'
import { clock } from './metric-text'
import { buildSaunaHeatTrainingLoad } from './training-effect-details'

const buildSaunaLapAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  layout: 'timeline' | 'legend' = 'timeline',
): N | null => {
  if (d.sport !== 'sauna') return null
  const laps = validAnalysisRanges(d)
    .filter(range => range.kind === 'lap')
    .sort((left, right) => left.startElapsedS - right.startElapsedS)
  if (laps.length < (layout === 'legend' ? 1 : 2)) return null
  const lapPhases = d.sauna?.lapPhases?.length === laps.length ? d.sauna.lapPhases : undefined
  const text = (value: string): string => triText(f.presentation.locale, value)
  const wrap = f.el('div', `tri-sauna-laps tri-sauna-laps--${layout}`, undefined, {
    'data-sauna-phase-source': lapPhases ? 'manual' : 'lap-order',
    'aria-label': text('sauna phases'),
  })
  const head = f.el('div', 'tri-sauna-laps-head')
  const legend = f.el('div', 'tri-sauna-laps-legend')
  for (const phase of ['hot sauna', 'cold plunge', 'break'])
    f.add(legend, f.el('span', 'tri-sauna-phase', text(phase), { 'data-sauna-phase': phase }))
  // The readout precedes the legend so CSS can hide the legend with a sibling combinator. A
  // `:has([data-visible])` rule made every `data-visible` change on the page cost ~50 ms of style.
  if (layout === 'timeline') {
    const readout = f.el('div', 'tri-analysis-readout', undefined, {
      'data-tri-analysis-readout': '',
      'data-visible': 'false',
      'aria-hidden': 'true',
    })
    f.add(
      readout,
      f.el('span', 'tri-analysis-readout-label'),
      f.el('span', 'tri-analysis-readout-metrics'),
    )
    f.add(head, readout)
  }
  f.add(head, legend)
  const timeline = f.el(
    'div',
    layout === 'legend' ? 'tri-sauna-lap-list' : 'tri-sauna-laps-timeline',
    undefined,
    layout === 'timeline' ? { 'data-site-cursor-line': '' } : undefined,
  )
  laps.forEach((range, index) => {
    const phase =
      lapPhases?.[index] ??
      (index < 2 || index === 3
        ? 'hot sauna'
        : index === 2 && d.sauna?.cooldown !== 'natural'
          ? 'cold plunge'
          : 'break')
    const label = `${text('lap')} ${index + 1} · ${text(phase)}`
    const metrics = analysisRangeMetrics(f.presentation, d, range)
    const bounds = analysisSelectionBounds(d, range)
    const attrs = analysisRangeAttrs(range)
    attrs['data-range-label'] = label
    attrs['data-sauna-phase'] = phase
    attrs['aria-pressed'] = 'false'
    attrs['aria-label'] = `${label}, ${metrics.join(', ')}`
    if (layout === 'timeline')
      attrs.style = `--tri-sauna-lap-start:${bounds.x.toFixed(3)}%;--tri-sauna-lap-width:${bounds.width.toFixed(3)}%`
    const button = f.el(
      'button',
      layout === 'legend' ? 'tri-sauna-lap-legend' : 'tri-sauna-lap',
      undefined,
      attrs,
    )
    if (layout === 'legend')
      f.add(
        button,
        f.el('span', 'tri-sauna-phase', `${text('lap')} ${index + 1}`),
        f.el('span', 'tri-sauna-lap-duration', clock(range.durationS)),
      )
    else f.add(button, f.el('span', undefined, String(index + 1), { 'aria-hidden': 'true' }))
    f.add(timeline, button)
  })
  f.add(wrap, head, timeline)
  return wrap
}

export const buildHeartRateWorkoutAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const saunaLaps = d.sauna?.location ? null : buildSaunaLapAnalysis(f, d)
  if (!hasHeartRateTrace(d)) return saunaLaps
  const points = activityHeartRateTracePoints(d)
  const values = points.flatMap(point => (point.heartRate == null ? [] : [point.heartRate]))
  let total = 0
  let seconds = 0
  for (let index = 1; index < points.length; index++) {
    const previous = points[index - 1]
    const point = points[index]
    const duration = point.elapsedS - previous.elapsedS
    if (previous.heartRate == null || point.heartRate == null || duration <= 0 || duration > 120)
      continue
    total += ((previous.heartRate + point.heartRate) / 2) * duration
    seconds += duration
  }
  const average =
    seconds > 0 ? total / seconds : values.reduce((sum, value) => sum + value, 0) / values.length
  const wrap = f.el(
    'section',
    `tri-workout tri-hr-workout${d.sport === 'sauna' ? ' tri-sauna-workout' : ''}`,
    undefined,
    {
      'aria-label': d.sport === 'sauna' ? 'Sauna workout analysis' : 'Heart rate workout analysis',
    },
  )
  const stats = f.el('div', 'tri-workout-stats')
  f.add(
    stats,
    f.el('span', undefined, `highest ${Math.round(Math.max(...values))} bpm`),
    f.el('span', undefined, `avg ${Math.round(average)} bpm`),
    f.el('span', undefined, `lowest ${Math.round(Math.min(...values))} bpm`),
  )
  f.add(wrap, stats, buildHeartRateTrace(f, d, null))
  if (saunaLaps) f.add(wrap, saunaLaps)
  return wrap
}

export const buildSaunaSummary = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const location = d.sport === 'sauna' ? d.sauna?.location : null
  if (!location) return null
  const { name, latitude, longitude } = location
  const figure = f.el('figure', 'tri-sauna-location')
  if (latitude != null && longitude != null) {
    const map = f.el('a', 'tri-sauna-location-map', undefined, {
      href: `https://www.openstreetmap.org/?mlat=${latitude}&mlon=${longitude}#map=16/${latitude}/${longitude}`,
      'aria-label': `${name} (${latitude}, ${longitude})`,
      'data-no-popover': 'true',
    })
    for (const theme of ['light', 'dark']) {
      f.add(
        map,
        f.el('img', `tri-sauna-location-image tri-sauna-location-image--${theme}`, undefined, {
          src: `/api/sauna-map?location=${encodeURIComponent(name)}&theme=${theme}`,
          alt: '',
          'data-no-popover': 'true',
          loading: 'lazy',
          decoding: 'async',
          width: '240',
          height: '240',
        }),
      )
    }
    const pin = f.svg('svg', {
      class: 'tri-sauna-location-pin',
      viewBox: '0 0 24 32',
      'aria-hidden': 'true',
    })
    f.add(
      pin,
      f.svg('path', { d: 'M12 30C10 27 2 19 2 12a10 10 0 0 1 20 0c0 7-8 15-10 18Z' }),
      f.svg('circle', { cx: 12, cy: 12, r: 3.5 }),
    )
    f.add(map, pin)
    f.add(figure, map)
  }
  const caption = f.el('figcaption', 'tri-sauna-location-caption')
  f.add(caption, f.el('span', 'tri-sauna-location-name', name))
  if (latitude != null && longitude != null) {
    f.add(
      caption,
      f.el(
        'span',
        'tri-sauna-location-coordinates',
        `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`,
      ),
      f.el('a', 'tri-sauna-location-attribution', '© OpenStreetMap contributors', {
        href: 'https://www.openstreetmap.org/copyright',
        'data-no-popover': 'true',
      }),
    )
  }
  f.add(figure, caption)
  const summary = f.el('div', 'tri-sauna-summary')
  const details = f.el('div', 'tri-sauna-summary-details')
  const laps = buildSaunaLapAnalysis(f, d, 'legend')
  const htl = buildSaunaHeatTrainingLoad(f, d)
  if (laps) f.add(details, laps)
  if (htl) f.add(details, htl)
  f.add(summary, figure, details)
  return summary
}
