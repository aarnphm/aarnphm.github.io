import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { swimLengthAverages, swimLengthMetrics, type SwimChartMetric } from '../swim-metrics'
import {
  triathlonTraceEnabled,
  triathlonTraceName,
  type TriathlonTraceSettings,
} from '../triathlon-trace-settings'
import { axisFrame, axisNumber, niceStep, niceTicks, swimActivityXTicks } from './chart-geometry'
import { buildAnalysisSelectionRect } from './chart-selection'
import { clock, positiveMetric } from './metric-text'
import {
  SWIM_TREND_GLOSS,
  swimActivityBlocks,
  swimActivityPointLabel,
  swimTrendAriaValue,
  swimTrendDisplayValue,
  swimTrendDomain,
  swimTrendHeaderValue,
  swimTrendLabel,
  swimTrendTitle,
  type SwimActivityMetric,
  type SwimActivityObservation,
  type SwimTrendChartPoint,
  type SwimTrendMode,
} from './swim-trend-data'

const buildSwimModeToggle = <N>(f: TriNodeFactory<N>): N => {
  const toggle = f.el('div', 'tri-swim-mode-toggle', undefined, {
    role: 'group',
    'aria-label': 'swim chart aggregation',
    'data-i18n-aria-label': 'swim chart aggregation',
    'data-swim-mode': 'lengths',
  })
  f.add(
    toggle,
    f.el('button', 'tri-swim-mode', 'lengths', {
      type: 'button',
      'data-swim-mode': 'lengths',
      'aria-pressed': 'true',
      'data-i18n': 'lengths',
    }),
    f.el('button', 'tri-swim-mode', '100 m', {
      type: 'button',
      'data-swim-mode': '100m',
      'aria-pressed': 'false',
      'data-i18n': '100 m',
    }),
  )
  return toggle
}

const buildSwimTrendChart = <N>(
  f: TriNodeFactory<N>,
  observations: SwimActivityObservation[],
  hundredMetreObservations: SwimActivityObservation[],
  totalDistanceM: number,
  kind: SwimChartMetric,
  average: number | null,
  pick: (observation: SwimActivityObservation) => number | null,
  modeToggle?: N,
): N | null => {
  const metricSeries = (source: SwimActivityObservation[]): SwimActivityMetric[] => {
    const metrics: SwimActivityMetric[] = []
    for (const observation of source) {
      const value = pick(observation)
      if (positiveMetric(value)) metrics.push({ observation, value })
    }
    return metrics
  }
  const series = metricSeries(observations)
  const hundredMetreSeries = metricSeries(hundredMetreObservations)
  if (series.length < 2) return null
  const currentIndex = series.length - 1
  const activityAverage = positiveMetric(average)
    ? average
    : series.reduce((sum, metric) => sum + metric.value, 0) / series.length
  const title = swimTrendTitle(kind)
  const value = swimTrendHeaderValue(kind, activityAverage)
  const wrap = f.el('article', `tri-zone tri-swim-trend tri-swim-trend--${kind}`, undefined, {
    'data-tri-trace': triathlonTraceName(swimTrendLabel(kind)),
  })
  const head = f.el(
    'div',
    `tri-swim-trend-head${modeToggle === undefined ? '' : ' tri-swim-trend-head--with-mode'}`,
  )
  const glossKey = SWIM_TREND_GLOSS[kind]
  if (modeToggle === undefined)
    f.add(
      head,
      f.el('span', 'tri-swim-trend-title', title, {
        'data-i18n': title,
        ...(glossKey ? { 'data-gloss': glossKey, tabindex: '0' } : {}),
      }),
    )
  else f.add(head, modeToggle)
  f.add(
    head,
    f.el('strong', 'tri-swim-trend-value', value, {
      'data-swim-average-kind': kind,
      'data-swim-average-value': activityAverage.toString(),
    }),
  )
  const W = 100
  const H = 30
  const X = (observation: SwimActivityObservation): number =>
    (observation.interval.cumulativeDistanceM / totalDistanceM) * W
  const domain = swimTrendDomain(
    [...series, ...hundredMetreSeries].map(metric => metric.value),
    kind,
  )
  const domainSpan = domain.max - domain.min
  const Y =
    kind === 'pace'
      ? (metric: number): number => ((metric - domain.min) / domainSpan) * H
      : (metric: number): number => H - ((metric - domain.min) / domainSpan) * H
  const ticks = niceTicks(domain.min, domain.max, 3)
  const tickStep = niceStep(domainSpan, 3)
  const yTicks = ticks.map(tick => ({
    label: kind === 'pace' ? clock(tick) : axisNumber(tick, tickStep),
    vbY: Y(tick),
  }))
  const chartSeries = (
    metrics: SwimActivityMetric[],
    mode: SwimTrendMode,
  ): { points: SwimTrendChartPoint[]; linePath: string; areaPath: string } => {
    const startX = (metric: SwimActivityMetric): number =>
      ((metric.observation.interval.cumulativeDistanceM - metric.observation.interval.distanceM) /
        totalDistanceM) *
      W
    const points = metrics.map(metric => ({
      elapsedS: metric.observation.interval.endElapsedS,
      cumulativeDistanceM: metric.observation.interval.cumulativeDistanceM,
      value: metric.value,
      xPct: X(metric.observation),
      yPct: (Y(metric.value) / H) * 100,
      ...(mode === '100m'
        ? {
            windowStartDistanceM:
              metric.observation.interval.cumulativeDistanceM -
              metric.observation.interval.distanceM,
          }
        : {}),
    }))
    const runs: SwimActivityMetric[][] = []
    for (const metric of metrics) {
      const run = runs.at(-1)
      const prior = run?.at(-1)
      if (run && prior && metric.observation.index === prior.observation.index + 1) run.push(metric)
      else runs.push([metric])
    }
    const linePath = runs
      .map(run =>
        run
          .map(
            (metric, index) =>
              `${index === 0 ? 'M' : 'L'} ${startX(metric).toFixed(2)} ${Y(metric.value).toFixed(2)} L ${X(metric.observation).toFixed(2)} ${Y(metric.value).toFixed(2)}`,
          )
          .join(' '),
      )
      .join(' ')
    const areaPath = runs
      .map(run => {
        const first = run[0]
        const last = run[run.length - 1]
        const values = run
          .map(
            metric =>
              `L ${startX(metric).toFixed(2)} ${Y(metric.value).toFixed(2)} L ${X(metric.observation).toFixed(2)} ${Y(metric.value).toFixed(2)}`,
          )
          .join(' ')
        return `M ${startX(first).toFixed(2)} ${H} ${values} L ${X(last.observation).toFixed(2)} ${H} Z`
      })
      .join(' ')
    return { points, linePath, areaPath }
  }
  const lengthsChart = chartSeries(series, 'lengths')
  const hundredMetreChart = chartSeries(hundredMetreSeries, '100m')
  const currentChartPoint = lengthsChart.points[currentIndex]
  const svg = f.svg('svg', {
    class: `tri-swim-trend-svg tri-swim-trend-svg--${kind}`,
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': `Swim ${swimTrendLabel(kind)} by length`,
    'aria-orientation': 'horizontal',
    'aria-valuemin': 0,
    'aria-valuemax': Math.round(totalDistanceM),
    'aria-valuenow': Math.round(currentChartPoint.cumulativeDistanceM),
    'aria-valuetext': `${swimTrendAriaValue(kind, currentChartPoint)}. Activity average ${swimTrendDisplayValue(kind, activityAverage)}.`,
    'data-swim-series-lengths': JSON.stringify(lengthsChart.points),
    'data-swim-series-hundred': JSON.stringify(hundredMetreChart.points),
    'data-swim-mode': 'lengths',
    'data-swim-kind': kind,
    'data-swim-index': currentIndex,
    'data-domain-start-distance-km': 0,
    'data-domain-end-distance-km': totalDistanceM / 1000,
  })
  for (const tick of yTicks)
    f.add(
      svg,
      f.svg('line', { class: 'tri-swim-trend-grid', x1: 0, y1: tick.vbY, x2: W, y2: tick.vbY }),
    )
  const addLayer = (
    mode: SwimTrendMode,
    chart: { linePath: string; areaPath: string },
    active: boolean,
  ): void => {
    const layer = f.svg('g', {
      class: `tri-swim-series tri-swim-series--${mode}${active ? ' tri-swim-series--active' : ''}`,
      'data-swim-mode': mode,
      'aria-hidden': String(!active),
    })
    f.add(
      layer,
      f.svg('path', {
        class: `tri-swim-trend-area tri-swim-trend-area--${mode}`,
        d: chart.areaPath,
      }),
      f.svg('path', {
        class: `tri-swim-trend-line tri-swim-trend-line--${mode}`,
        d: chart.linePath,
      }),
    )
    f.add(svg, layer)
  }
  addLayer('lengths', lengthsChart, true)
  if (hundredMetreChart.points.length >= 2) addLayer('100m', hundredMetreChart, false)
  f.add(
    svg,
    buildAnalysisSelectionRect(f, H),
    f.svg('line', {
      class: 'tri-chart-cursor',
      x1: currentChartPoint.xPct.toFixed(2),
      y1: 0,
      x2: currentChartPoint.xPct.toFixed(2),
      y2: H,
    }),
  )
  const hoverMarker = f.el('span', 'tri-swim-trend-hover', undefined, {
    'aria-hidden': 'true',
    hidden: '',
    style: `left:${currentChartPoint.xPct.toFixed(2)}%;top:${currentChartPoint.yPct.toFixed(2)}%`,
  })
  const readout = f.el('div', 'tri-chart-readout tri-swim-trend-readout', undefined, {
    'aria-hidden': 'true',
  })
  f.add(
    readout,
    f.el('span', 'tri-swim-trend-readout-position', swimActivityPointLabel(currentChartPoint)),
    f.el(
      'strong',
      'tri-swim-trend-readout-value',
      swimTrendDisplayValue(kind, currentChartPoint.value),
    ),
  )
  f.add(
    wrap,
    head,
    axisFrame(f, svg, yTicks, H, swimActivityXTicks(totalDistanceM), true, { top: 0, bottom: H }, [
      hoverMarker,
      readout,
    ]),
  )
  return wrap
}

export const buildSwimTrends = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  traceSettings?: TriathlonTraceSettings,
): N | null => {
  if (d.sport !== 'swim') return null
  const observations = d.swimIntervals
    .filter(
      interval => interval.endElapsedS > interval.startElapsedS && interval.cumulativeDistanceM > 0,
    )
    .map((interval, index) => {
      const length = d.swimLocation === 'pool' ? swimLengthMetrics(interval) : null
      return {
        interval,
        index,
        strokesPerLength: length?.strokesPerLength ?? null,
        swolf: length?.swolf ?? null,
      }
    })
  const totalDistanceM = observations.at(-1)?.interval.cumulativeDistanceM ?? 0
  if (observations.length < 2 || totalDistanceM <= 0) return null
  const hundredMetreObservations = swimActivityBlocks(
    observations.map(observation => observation.interval),
  ).map((interval, index) => ({
    interval,
    index,
    strokesPerLength: d.swimLocation === 'pool' ? interval.strokesPerLength : null,
    swolf: d.swimLocation === 'pool' ? interval.swolf : null,
  }))
  const hasSeries = (
    source: SwimActivityObservation[],
    pick: (observation: SwimActivityObservation) => number | null,
  ): boolean => source.filter(observation => positiveMetric(pick(observation))).length >= 2
  const paceVisible =
    triathlonTraceEnabled(traceSettings, 'pace') &&
    hasSeries(observations, observation => observation.interval.paceSPer100m)
  const rateVisible =
    triathlonTraceEnabled(traceSettings, 'stroke-rate') &&
    hasSeries(observations, observation => observation.interval.strokeRateSpm)
  const cadenceVisible =
    triathlonTraceEnabled(traceSettings, 'cadence') &&
    hasSeries(observations, observation => observation.strokesPerLength)
  const swolfVisible =
    triathlonTraceEnabled(traceSettings, 'swolf') &&
    hasSeries(observations, observation => observation.swolf)
  const canToggle =
    hundredMetreObservations.length >= 2 &&
    (!paceVisible ||
      hasSeries(hundredMetreObservations, observation => observation.interval.paceSPer100m)) &&
    (!rateVisible ||
      hasSeries(hundredMetreObservations, observation => observation.interval.strokeRateSpm)) &&
    (!cadenceVisible ||
      hasSeries(hundredMetreObservations, observation => observation.strokesPerLength)) &&
    (!swolfVisible || hasSeries(hundredMetreObservations, observation => observation.swolf))
  const normalizedObservations = canToggle ? hundredMetreObservations : []
  const modeToggle = canToggle ? buildSwimModeToggle(f) : undefined
  const paceAverage = positiveMetric(d.swimPaceSPer100m) ? d.swimPaceSPer100m : null
  const lengthAverages = d.swimLocation === 'pool' ? swimLengthAverages(d.swimIntervals) : null
  const pace = paceVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'pace',
        paceAverage,
        observation => observation.interval.paceSPer100m,
        modeToggle,
      )
    : null
  const rate = rateVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'rate',
        positiveMetric(d.strokeRateSpm) ? d.strokeRateSpm : null,
        observation => observation.interval.strokeRateSpm,
        paceVisible ? undefined : modeToggle,
      )
    : null
  const cadence = cadenceVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'cadence',
        lengthAverages?.strokesPerLength ?? null,
        observation => observation.strokesPerLength,
        paceVisible || rateVisible ? undefined : modeToggle,
      )
    : null
  const swolf = swolfVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'swolf',
        lengthAverages?.swolf ?? null,
        observation => observation.swolf,
        paceVisible || rateVisible || cadenceVisible ? undefined : modeToggle,
      )
    : null
  const trends = f.el('div', 'tri-swim-chart-grid')
  let chartCount = 0
  for (const chart of [pace, rate, cadence, swolf])
    if (chart) {
      f.add(trends, chart)
      chartCount++
    }
  if (chartCount === 0) return null
  const wrap = f.el('section', 'tri-swim-trends', undefined, {
    'aria-label': 'Swim activity analysis',
    'data-i18n-aria-label': 'swim activity analysis',
  })
  f.add(wrap, trends)
  return wrap
}
