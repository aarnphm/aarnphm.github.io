import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { STAMINA_LEDGER_OFFSET_METHOD } from '../stamina-ledger'
import { triText } from '../triathlon-i18n'
import { triathlonTraceName } from '../triathlon-trace-settings'
import {
  activityPhysiologyTracePoints,
  activityTraceUsesElapsedAxis,
  performanceConditionValue,
  type ActivityPhysiologyTracePoint,
} from './activity-metric-data'
import {
  axisFrame,
  distanceXTicks,
  elapsedActivityXTicks,
  graphViewForDistance,
  swimActivityXTicks,
  type ActivityGraphDomain,
} from './chart-geometry'
import { buildAnalysisSelection } from './chart-selection'
import { formatPerformanceCondition } from './metric-text'
import { buildTraceSeries } from './trace-chart'

const staminaLedgerDescription = (d: StravaActivityDetail): string | null => {
  const ledger = d.staminaLedger
  if (!ledger) return null
  const recovery =
    ledger.recoveryHours < 1
      ? ''
      : ledger.recoveryFloor > 0
        ? ` · ${ledger.recoveryHours.toFixed(0)} h recovery time left holds a ${ledger.recoveryFloor.toFixed(1)}-point floor`
        : ` · ${ledger.recoveryHours.toFixed(0)} h recovery time left (no floor before Garmin recovery readings)`
  return `starts ${Math.max(ledger.fatigueDeficit, ledger.recoveryFloor).toFixed(1)} points down: ${ledger.fatigueDeficit.toFixed(1)} carried from earlier sessions (5.6 h recovery)${recovery}`
}

const swimPhysiologyDescription = (d: StravaActivityDetail, metric: string): string => {
  const model = d.swimPhysiology
  if (!model) return ''
  const strokes =
    model.strokeRateSource === 'activity-average'
      ? 'activity-average stroke rate held constant'
      : model.strokeRateSource === 'stream-with-average'
        ? 'recorded stroke rate with activity-average fallback for missing strokes'
        : model.strokeRateSource === 'stream'
          ? 'recorded stroke rate'
          : 'stroke rate unavailable'
  const speed =
    model.speedBasis === 'pool-length'
      ? 'recorded pool length distance / active time'
      : 'ground speed'
  const limitation =
    model.speedBasis === 'pool-length'
      ? 'length times include turns and push-offs; rests and missing HR remain gaps'
      : 'currents and GPS error affect speed'
  return `Garden swim ${metric} estimate · HR / max HR ${model.maxHeartRateBpm} bpm, ${speed}, ${strokes} · first minute excluded from the three-minute opening baseline · 60 s smoothing · ${(model.coverage * 100).toFixed(0)}% telemetry coverage · ${metric === 'stamina' ? `${staminaLedgerDescription(d) ?? 'starts from the stamina carried from earlier sessions'}; depletion integrates observed intervals only; HR load increases with relative drag demand and strokes per metre; current and potential coincide` : 'speed per HR relative to the opening baseline, penalized by extra strokes per metre'} · experimental session proxy; ${limitation}`
}

const staminaSeriesPath = (
  points: ActivityPhysiologyTracePoint[],
  pick: (point: ActivityPhysiologyTracePoint) => number | null,
  closeArea: boolean,
): string => {
  const width = 100
  const height = 30
  const maxDistanceKm = points.at(-1)?.d || 1
  const px = (distanceKm: number): number => (distanceKm / maxDistanceKm) * width
  const py = (value: number): number => height - (Math.min(100, Math.max(0, value)) / 100) * height
  let path = ''
  let segmentStart = -1
  const closeSegment = (start: number, end: number): void => {
    const first = pick(points[start])
    if (first == null) return
    const firstX = px(points[start].d)
    if (closeArea)
      path += `M ${firstX.toFixed(2)} ${height} L ${firstX.toFixed(2)} ${py(first).toFixed(2)} `
    else path += `M ${firstX.toFixed(2)} ${py(first).toFixed(2)} `
    for (let index = start + 1; index <= end; index++) {
      const value = pick(points[index])
      if (value == null) continue
      path += `L ${px(points[index].d).toFixed(2)} ${py(value).toFixed(2)} `
    }
    if (closeArea) path += `L ${px(points[end].d).toFixed(2)} ${height} Z `
  }
  points.forEach((point, index) => {
    const value = pick(point)
    const valid = value != null && Number.isFinite(value) && value >= 0 && value <= 100
    if (valid && segmentStart < 0) segmentStart = index
    if (segmentStart >= 0 && (!valid || index === points.length - 1)) {
      closeSegment(segmentStart, valid ? index : index - 1)
      segmentStart = -1
    }
  })
  return path
}

export const buildStaminaChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const tracePoints = activityPhysiologyTracePoints(d, 'stamina')
  if (
    tracePoints.filter(point => point.stamina != null && point.potentialStamina != null).length < 2
  )
    return null
  const heartRateEstimate =
    d.route.filter(point => point.stamina != null).length >= 2 ? null : d.heartRatePhysiology
  const swimEstimate =
    d.route.filter(point => point.stamina != null).length >= 2 ? null : d.swimPhysiology
  const width = 100
  const height = 30
  const view = graphViewForDistance(tracePoints.at(-1)?.d || 1, graphDomain)
  const yTicks = [0, 25, 50, 75, 100].map(value => ({
    label: `${value}%`,
    vbY: height - (value / 100) * height,
  }))
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-stamina-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    ...(activityTraceUsesElapsedAxis(d)
      ? {
          'data-domain-start-elapsed-s': view.startDistanceKm,
          'data-domain-end-elapsed-s': view.endDistanceKm,
        }
      : {
          'data-domain-start-distance-km': view.startDistanceKm,
          'data-domain-end-distance-km': view.endDistanceKm,
        }),
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: width, y2: tick.vbY }),
    )
  f.add(
    svgEl,
    f.svg('path', {
      d: staminaSeriesPath(tracePoints, point => point.stamina, true),
      class: 'tri-stamina-area',
    }),
  )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  f.add(
    svgEl,
    f.svg('path', {
      d: staminaSeriesPath(tracePoints, point => point.stamina, false),
      class: 'tri-stamina-line tri-stamina-line--current',
    }),
    f.svg('path', {
      d: staminaSeriesPath(tracePoints, point => point.potentialStamina, false),
      class: 'tri-stamina-line tri-stamina-line--potential',
    }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const wrap = f.el('div', 'tri-zone tri-elev-wrap tri-stamina-chart', undefined, {
    'data-tri-trace': triathlonTraceName('stamina'),
    'data-stamina-source':
      swimEstimate?.source ?? heartRateEstimate?.source ?? d.staminaTrace?.source ?? 'garmin',
  })
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const estimatedTrace = d.staminaTrace?.source === 'garden-estimate' ? d.staminaTrace : null
  f.add(
    cap,
    f.el(
      'span',
      'tri-elev-d',
      triText(f.presentation.locale, 'stamina'),
      swimEstimate
        ? {
            'data-gloss': '',
            'data-gloss-def': swimPhysiologyDescription(d, 'stamina'),
            tabindex: '0',
          }
        : heartRateEstimate
          ? {
              'data-gloss': '',
              'data-gloss-def': `Garden HR estimate · ${staminaLedgerDescription(d) ?? 'starts from the stamina carried from earlier sessions'} · observed HR / max HR ${heartRateEstimate.maxHeartRateBpm} bpm · ${d.sport === 'run' ? 'run' : 'ride'} curve fit to Garmin native traces${d.sport === 'run' || d.sport === 'bike' ? ', scaled to recent native sessions' : ''} · current and potential coincide`,
              tabindex: '0',
            }
          : estimatedTrace?.method === STAMINA_LEDGER_OFFSET_METHOD
            ? {
                'data-gloss': '',
                'data-gloss-def': `Garmin Connect trace lowered ${estimatedTrace.garminOffset.toFixed(1)} points for sessions Garmin did not record${staminaLedgerDescription(d) ? ` · ${staminaLedgerDescription(d)}` : ''}`,
                tabindex: '0',
              }
            : estimatedTrace
              ? {
                  'data-gloss': '',
                  'data-gloss-def': `${triText(f.presentation.locale, 'estimate')} · FTP ${estimatedTrace.ftpWatts} W · ${triText(f.presentation.locale, 'max hr')} ${estimatedTrace.maxHeartRateBpm} bpm${staminaLedgerDescription(d) ? ` · ${staminaLedgerDescription(d)}` : ''}`,
                  tabindex: '0',
                }
              : d.staminaTrace?.source === 'garmin'
                ? { 'data-gloss': '', 'data-gloss-def': 'Garmin Connect', tabindex: '0' }
                : undefined,
    ),
  )
  if (swimEstimate)
    f.add(
      cap,
      f.el('span', 'tri-elev-range', 'calculated', {
        'data-gloss': '',
        'data-gloss-def': swimPhysiologyDescription(d, 'stamina'),
        tabindex: '0',
      }),
    )
  for (const kind of ['current', 'potential'] as const) {
    const item = f.el('span', `tri-stamina-legend-item tri-stamina-legend-item--${kind}`)
    f.add(
      item,
      f.el('span', 'tri-stamina-legend-line', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-stamina-legend-label', triText(f.presentation.locale, kind)),
    )
    f.add(cap, item)
  }
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      svgEl,
      yTicks,
      height,
      activityTraceUsesElapsedAxis(d)
        ? elapsedActivityXTicks(view.endDistanceKm)
        : d.sport === 'swim'
          ? swimActivityXTicks(view.endDistanceKm * 1000, view.startDistanceKm * 1000)
          : distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: height },
    ),
  )
  return wrap
}

export const buildPerformanceConditionTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const points = activityPhysiologyTracePoints(d, 'performanceCondition')
  const heartRateEstimate =
    d.route.filter(point => point.performanceCondition != null).length >= 2
      ? null
      : d.heartRatePhysiology
  const swimEstimate =
    d.route.filter(point => point.performanceCondition != null).length >= 2
      ? null
      : d.swimPhysiology
  const values = points
    .map(performanceConditionValue)
    .filter((value): value is number => value != null)
  if (values.length < 2) {
    const wrap = f.el('div', 'tri-elev-wrap tri-elev-wrap--unavailable', undefined, {
      'data-tri-unavailable': 'performance-condition',
    })
    const cap = f.el('div', 'tri-elev-cap')
    f.add(cap, f.el('span', 'tri-elev-d', triText(f.presentation.locale, 'performance condition')))
    f.add(
      wrap,
      cap,
      axisFrame(
        f,
        f.el('div', 'tri-elev-unavailable', triText(f.presentation.locale, 'no data available'), {
          'data-i18n': 'no data available',
        }),
        [],
        30,
        [],
        false,
      ),
    )
    return wrap
  }
  const average = values.reduce((total, value) => total + value, 0) / values.length
  const magnitude = Math.max(2, Math.ceil(Math.max(...values.map(Math.abs)) / 2) * 2)
  const min = -magnitude
  const max = magnitude
  const estimatedTrace =
    d.performanceConditionTrace?.source === 'garden-estimate' ? d.performanceConditionTrace : null
  const estimateLabel = swimEstimate
    ? f.el('span', 'tri-elev-range tri-performance-condition-source', 'calculated', {
        'data-gloss': '',
        'data-gloss-def': swimPhysiologyDescription(d, 'performance condition'),
        tabindex: '0',
      })
    : heartRateEstimate
      ? f.el(
          'span',
          'tri-elev-range tri-performance-condition-source',
          triText(f.presentation.locale, 'calculated'),
          {
            'data-gloss': '',
            'data-gloss-def': `Garden HR change proxy · 100 × (opening HR ${heartRateEstimate.baselineHeartRateBpm.toFixed(1)} bpm − trailing 60 s HR) / max HR ${heartRateEstimate.maxHeartRateBpm} bpm · first 60 observed seconds establish baseline · each window requires 80% HR coverage; gaps stay unavailable · positive means lower HR · experimental, not a Garmin fitness assessment`,
            tabindex: '0',
          },
        )
      : estimatedTrace
        ? f.el(
            'span',
            'tri-elev-range tri-performance-condition-source',
            triText(f.presentation.locale, 'calculated'),
            {
              'data-gloss': '',
              'data-gloss-def': `${triText(f.presentation.locale, 'Garden estimate')} · ${estimatedTrace.windowSeconds / 60} min (NP / FTP) ÷ ((HR − RHR) / (LTHR − RHR)) · FTP ${estimatedTrace.ftpWatts} W · LTHR ${estimatedTrace.lactateThresholdHeartRateBpm} bpm · RHR ${estimatedTrace.restingHeartRateBpm} bpm`,
              tabindex: '0',
            },
          )
        : null
  return buildTraceSeries(
    f,
    d,
    points,
    performanceConditionValue,
    'performance condition',
    () => `${formatPerformanceCondition(average)} avg`,
    formatPerformanceCondition,
    { min, max, intervals: 4 },
    selection,
    graphDomain,
    undefined,
    { value: 0, label: 'baseline' },
    {
      wrapAttrs: {
        'data-performance-condition-source':
          swimEstimate?.source ??
          heartRateEstimate?.source ??
          d.performanceConditionTrace?.source ??
          'garmin',
      },
      capExtra: estimateLabel ? [estimateLabel] : undefined,
      areaBaseline: 0,
    },
  )
}
