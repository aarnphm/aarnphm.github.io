import type {
  ActivityAnalysisKind,
  ActivityAnalysisRange,
  ActivityClimbSource,
  ActivityKind,
  StravaActivityDetail,
} from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { ActivityGraphDomain } from './chart-geometry'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import {
  activityAnalysisAvailable,
  ANALYSIS_KIND_ORDER,
  hasAnalysisWorkspace,
  validAnalysisRanges,
} from './analysis-data'
import { analysisSelectionBounds } from './chart-selection'
import {
  activityCadenceScale,
  activityCadenceUnit,
  clock,
  formatElevationGain,
  isImperial,
  KM_TO_MI,
  scrubDist,
  speedKph,
} from './metric-text'
import { buildTrace } from './trace-chart'

export const buildActivityAnalyzeButton = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N => {
  const label = triText(f.presentation.locale, 'analyze')
  const available = activityAnalysisAvailable(d)
  const button = f.el('button', 'tri-activity-analyze', undefined, {
    type: 'button',
    'data-activity-analyze': `${d.id}`,
    'aria-label': label,
    'data-i18n-aria-label': 'analyze',
    'aria-expanded': 'false',
    title: available
      ? label
      : triText(f.presentation.locale, 'No recorded telemetry for this activity.'),
    ...(!available ? { disabled: '' } : {}),
    'data-site-cursor-action': '',
  })
  const icon = f.svg('svg', {
    viewBox: '0 0 24 24',
    fill: 'none',
    'aria-hidden': 'true',
    'data-site-cursor-icon': '',
  })
  f.add(icon, f.svg('path', { d: 'M4 4v16h16M7 14l4-5 4 3 5-7' }))
  f.add(button, icon)
  return button
}

export const analysisRangeAttrs = (range: ActivityAnalysisRange): Record<string, string> => {
  const displayDurationS =
    range.distanceKm === 0 ? range.durationS : (range.movingTimeS ?? range.durationS)
  const attrs: Record<string, string> = {
    type: 'button',
    'data-analysis-range': '',
    'data-range-kind': range.kind,
    'data-range-id': range.id,
    'data-range-label': range.label,
    'data-start-elapsed-s': `${range.startElapsedS}`,
    'data-end-elapsed-s': `${range.endElapsedS}`,
    'data-start-distance-km': `${range.startDistanceKm}`,
    'data-end-distance-km': `${range.endDistanceKm}`,
    'data-duration-s': `${displayDurationS}`,
    'data-distance-km': `${range.distanceKm}`,
  }
  if (range.elevationGainM != null) attrs['data-elevation-gain-m'] = `${range.elevationGainM}`
  if (range.averageGradePct != null && Number.isFinite(range.averageGradePct))
    attrs['data-average-grade-pct'] = `${range.averageGradePct}`
  if (range.averageSpeedKph != null) attrs['data-average-speed-kph'] = `${range.averageSpeedKph}`
  if (range.averageHeartRate != null) attrs['data-average-heart-rate'] = `${range.averageHeartRate}`
  if (range.averageWatts != null) attrs['data-average-watts'] = `${range.averageWatts}`
  if (range.averageCadence != null) attrs['data-average-cadence'] = `${range.averageCadence}`
  if (range.heartRateChange) {
    attrs['data-heart-rate-change-source'] = range.heartRateChange.source
    attrs['data-start-heart-rate'] = `${range.heartRateChange.startBpm}`
    attrs['data-end-heart-rate'] = `${range.heartRateChange.endBpm}`
  }
  return attrs
}

export const analysisHeartRateChangeText = (
  change: ActivityAnalysisRange['heartRateChange'],
): string | null => {
  if (!change) return null
  const start = Math.round(change.startBpm)
  const end = Math.round(change.endBpm)
  const delta = end - start
  return `${start} → ${end} bpm (${delta > 0 ? '+' : ''}${delta})`
}

const analysisRangeRate = (
  presentation: TriathlonPresentation,
  sport: ActivityKind,
  speedKphValue: number,
): string => {
  const imperial = isImperial(presentation)
  if (sport === 'bike') return speedKph(presentation, speedKphValue)
  if (sport === 'swim') return `${clock(360 / speedKphValue)} /100m`
  return `${clock(3600 / (speedKphValue * (imperial ? KM_TO_MI : 1)))} /${imperial ? 'mi' : 'km'}`
}

export const buildPaceTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const speeds = d.route
    .map(point => point.speedKph)
    .filter(speed => Number.isFinite(speed) && speed > 0)
  if (speeds.length < 2) return null
  const averageSpeedKph =
    d.distanceKm > 0 && d.movingTimeS > 0
      ? d.distanceKm / (d.movingTimeS / 3_600)
      : speeds.reduce((total, speed) => total + speed, 0) / speeds.length
  return buildTrace(
    f,
    d,
    point => (Number.isFinite(point.speedKph) && point.speedKph > 0 ? point.speedKph : null),
    'pace',
    () => `${analysisRangeRate(f.presentation, d.sport, averageSpeedKph)} avg`,
    value => analysisRangeRate(f.presentation, d.sport, value),
    undefined,
    selection,
    graphDomain,
  )
}

export const buildSpeedTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
): N | null => {
  if (d.sport !== 'bike') return null
  const speeds = d.route
    .map(point => point.speedKph)
    .filter(speed => Number.isFinite(speed) && speed >= 0)
  if (speeds.length < 2) return null
  const peak = Math.max(...speeds)
  return buildTrace(
    f,
    d,
    point => (Number.isFinite(point.speedKph) && point.speedKph >= 0 ? point.speedKph : null),
    'speed',
    () => `${speedKph(f.presentation, peak)} peak`,
    value => speedKph(f.presentation, value),
    undefined,
    selection,
    undefined,
    undefined,
    undefined,
    {
      wrapAttrs: {
        'data-speed-source': 'distance-time',
        title: 'Speed calculated from recorded distance and elapsed time.',
      },
    },
  )
}

export const analysisRangeMetrics = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  range: ActivityAnalysisRange,
): string[] => {
  if (d.sport === 'sauna') {
    const values = [clock(range.durationS)]
    if (range.averageHeartRate != null) values.push(`${Math.round(range.averageHeartRate)} bpm avg`)
    const change = analysisHeartRateChangeText(range.heartRateChange)
    if (change) values.push(change)
    return values
  }
  const cadenceUnit = activityCadenceUnit(d.sport)
  const cadenceScale = activityCadenceScale(d.sport)
  const values = [scrubDist(presentation, range.distanceKm, d.sport)]
  if (range.elevationGainM != null)
    values.push(`+${formatElevationGain(presentation, range.elevationGainM)}`)
  if (range.averageGradePct != null && Number.isFinite(range.averageGradePct))
    values.push(`${range.averageGradePct.toFixed(1)}% avg grade`)
  values.push(clock(range.movingTimeS ?? range.durationS))
  if (range.averageSpeedKph != null)
    values.push(analysisRangeRate(presentation, d.sport, range.averageSpeedKph))
  if (range.averageWatts != null) values.push(`${Math.round(range.averageWatts)} W`)
  if (range.averageHeartRate != null) values.push(`${Math.round(range.averageHeartRate)} bpm`)
  if (range.averageCadence != null)
    values.push(`${Math.round(range.averageCadence * cadenceScale)} ${cadenceUnit}`)
  return values
}

type PositionedAnalysisRange = { range: ActivityAnalysisRange; lane: number }

export const climbSourceLabel = (sources: readonly ActivityClimbSource[]): string => {
  const unique = new Set(sources)
  if (unique.size === 1) {
    const source = sources[0]
    if (source === 'garmin-climbpro') return 'ClimbPro'
    if (source === 'wahoo-summit-segment') return 'Summit Segments'
    return 'Summit Freeride'
  }
  if (
    sources.length > 0 &&
    sources.every(source => source === 'wahoo-summit-segment' || source === 'wahoo-summit-freeride')
  )
    return 'Summit Climbs'
  return 'Climbs'
}

const analysisGroupLabel = (
  kind: ActivityAnalysisKind,
  ranges: readonly ActivityAnalysisRange[],
): string => {
  if (kind === 'lap') return 'Laps'
  if (kind === 'segment') return 'Segments'
  return climbSourceLabel(ranges.flatMap(range => (range.kind === 'climb' ? [range.source] : [])))
}

const positionAnalysisRanges = (
  ranges: ActivityAnalysisRange[],
  laneLimit: number,
): PositionedAnalysisRange[] => {
  const laneEnds = Array.from({ length: laneLimit }, () => Number.NEGATIVE_INFINITY)
  return ranges
    .slice()
    .sort((a, b) => a.startDistanceKm - b.startDistanceKm || a.endDistanceKm - b.endDistanceKm)
    .map(range => {
      let lane = laneEnds.findIndex(end => end <= range.startDistanceKm)
      if (lane < 0) {
        lane = 0
        for (let index = 1; index < laneEnds.length; index++)
          if (laneEnds[index] < laneEnds[lane]) lane = index
      }
      laneEnds[lane] = Math.max(laneEnds[lane], range.endDistanceKm)
      return { range, lane }
    })
}

export const buildAnalysisBar = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const ranges = validAnalysisRanges(d)
  const timedLaps = d.sport === 'sauna' && ranges.some(range => range.kind === 'lap')
  if (!hasAnalysisWorkspace(d) && !timedLaps) return null
  const wrapAttrs: Record<string, string> = {
    'data-tri-analysis': '',
    'data-activity-id': `${d.id}`,
  }
  if (ranges.length === 0) wrapAttrs['aria-hidden'] = 'true'
  else wrapAttrs['aria-label'] = 'Activity analysis'
  const wrap = f.el('section', 'tri-analysis', undefined, wrapAttrs)
  const readout = f.el('div', 'tri-analysis-readout', undefined, {
    'data-tri-analysis-readout': '',
    'data-visible': 'false',
    'aria-hidden': 'true',
    'aria-live': 'polite',
  })
  f.add(
    readout,
    f.el('span', 'tri-analysis-readout-label'),
    f.el('span', 'tri-analysis-readout-metrics'),
  )
  f.add(wrap, readout)

  const rangeBands = f.el('div', 'tri-analysis-ranges')
  for (const kind of ANALYSIS_KIND_ORDER) {
    if (timedLaps && kind !== 'lap') continue
    const groupRanges = ranges.filter(range => range.kind === kind)
    const label = analysisGroupLabel(kind, groupRanges)
    const visibleLabel = label === 'Summit Segments' ? 'Summit' : label
    const empty = groupRanges.length === 0
    const laneLimit = kind === 'lap' || kind === 'climb' ? 1 : 4
    const positioned = positionAnalysisRanges(groupRanges, laneLimit)
    const bandAttrs: Record<string, string> = { 'data-analysis-kind': kind }
    if (empty) bandAttrs['aria-hidden'] = 'true'
    else {
      bandAttrs.role = 'group'
      bandAttrs['aria-label'] = label
    }
    const band = f.el('div', 'tri-analysis-band', undefined, bandAttrs)
    const items = f.el('div', 'tri-analysis-band-items', undefined, {
      style: `--tri-analysis-lanes:${laneLimit}`,
      ...(timedLaps ? { 'data-site-cursor-line': '' } : {}),
    })
    for (const { range, lane } of positioned) {
      const metrics = analysisRangeMetrics(f.presentation, d, range)
      const bounds = analysisSelectionBounds(d, range)
      const attrs = analysisRangeAttrs(range)
      attrs['aria-pressed'] = 'false'
      attrs['aria-label'] = `${range.label}, ${metrics.join(', ')}`
      attrs.style = `--tri-analysis-start:${bounds.x.toFixed(3)}%;--tri-analysis-width:${Math.max(0.18, bounds.width).toFixed(3)}%;--tri-analysis-lane:${lane}`
      f.add(items, f.el('button', 'tri-analysis-range', undefined, attrs))
    }
    f.add(band, f.el('span', 'tri-analysis-band-label', empty ? undefined : visibleLabel), items)
    f.add(rangeBands, band)
  }
  f.add(wrap, rangeBands)
  return wrap
}
