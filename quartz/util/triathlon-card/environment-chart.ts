import type { EnvironmentChartSample } from '../activity-environment'
import type { ElapsedPause } from '../elapsed-axis'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { elapsedAxis, formatElapsedPauses } from '../elapsed-axis'
import { triText } from '../triathlon-i18n'
import { axisFrame, type AxisYTick } from './chart-geometry'
import { buildAnalysisSelectionRect } from './chart-selection'
import { addElapsedBreaks } from './elapsed-chart'
import { environmentChartReadout, environmentElapsedClock } from './environment-readout'
import {
  ENVIRONMENT_CHART_BOTTOM,
  ENVIRONMENT_CHART_LEFT,
  ENVIRONMENT_CHART_RIGHT,
  ENVIRONMENT_CHART_TOP,
  environmentAxisTicks,
  environmentChartScale,
  environmentChartValues,
  environmentChartY,
  environmentSampleValue,
  environmentUvColor,
  environmentViewHasSamples,
  environmentViewLabel,
  environmentWindMetric,
  type EnvironmentChartSeries,
  type EnvironmentChartView,
  type EnvironmentScoreModel,
} from './environment-scale'

export const environmentChartSeries = (
  presentation: TriathlonPresentation,
  samples: readonly EnvironmentChartSample[],
  elapsedTimeS: number,
  view: EnvironmentChartView,
  scoreModel: EnvironmentScoreModel | null = null,
  pauses: readonly ElapsedPause[] = [],
): EnvironmentChartSeries[] => {
  const values = environmentChartValues(presentation, samples, view, scoreModel)
  if (values.length < 2 || elapsedTimeS <= 0) return []
  const windMetric = view === 'wind' ? environmentWindMetric(samples) : 'headwind'
  const scale = environmentChartScale(values, view, scoreModel, windMetric)
  // A recording pause takes no width; samples inside it are dropped and lines lift at its join.
  const axis = elapsedAxis(0, elapsedTimeS, pauses)
  const insidePause = (elapsedS: number): boolean =>
    axis.pauses.some(pause => elapsedS > pause.startS && elapsedS < pause.endS)
  const x = (elapsedS: number): number =>
    ENVIRONMENT_CHART_LEFT +
    axis.fraction(elapsedS) * (ENVIRONMENT_CHART_RIGHT - ENVIRONMENT_CHART_LEFT)
  const y = (value: number): number => environmentChartY(value, scale)
  const stepped =
    view === 'uv-index' || view === 'cloud-cover' || (view === 'wind' && windMetric === 'ambient')
  if (view === 'uv-index') {
    const segments: EnvironmentChartSeries[] = []
    for (let index = 1; index < samples.length; index += 1) {
      const previous = environmentSampleValue(presentation, samples[index - 1], view, scoreModel)
      const current = environmentSampleValue(presentation, samples[index], view, scoreModel)
      if (previous == null || current == null) continue
      if (
        insidePause(samples[index - 1].elapsedS) ||
        insidePause(samples[index].elapsedS) ||
        axis.crossesPause(samples[index - 1].elapsedS, samples[index].elapsedS)
      )
        continue
      segments.push({
        path: `M${x(samples[index - 1].elapsedS).toFixed(3)},${y(previous).toFixed(3)}H${x(samples[index].elapsedS).toFixed(3)}V${y(current).toFixed(3)}`,
        color: environmentUvColor(previous),
      })
    }
    return segments
  }
  const series: EnvironmentChartSeries[] = []
  let path = ''
  let previous: number | null = null
  let previousS = 0
  for (const sample of samples) {
    const value = environmentSampleValue(presentation, sample, view, scoreModel, windMetric)
    const gap = value == null || !Number.isFinite(value) || insidePause(sample.elapsedS)
    if (gap || (path && axis.crossesPause(previousS, sample.elapsedS))) {
      if (path) series.push({ path, color: null })
      path = ''
      previous = null
      if (gap) continue
    }
    const px = x(sample.elapsedS).toFixed(3)
    const py = y(value).toFixed(3)
    if (view === 'current-direction' && previous != null && Math.abs(value - previous) > 180) {
      if (path) series.push({ path, color: null })
      path = ''
    }
    if (!path) path = `M${px},${py}`
    else path += stepped && previous != null ? `H${px}V${py}` : `L${px},${py}`
    previous = value
    previousS = sample.elapsedS
  }
  if (path) series.push({ path, color: null })
  return series
}

export const buildEnvironmentChart = <N>(
  f: TriNodeFactory<N>,
  samples: readonly EnvironmentChartSample[],
  elapsedTimeS: number,
  view: EnvironmentChartView,
  scoreModel: EnvironmentScoreModel | null,
  currentModel: { formulaId: string; formulaVersion: number } | null = null,
  provider: 'mywindsock' | 'garden' | null = null,
  pauses: readonly ElapsedPause[] = [],
): N => {
  const axis = elapsedAxis(0, elapsedTimeS, pauses)
  const chart = f.el('div', 'tri-environment-chart')
  const hasSamples = elapsedTimeS > 0 && environmentViewHasSamples(f.presentation, samples, view)
  const windMetric = view === 'wind' ? environmentWindMetric(samples) : 'headwind'
  const svg = f.svg('svg', {
    class: 'tri-environment-plot',
    viewBox: '0 0 100 32',
    preserveAspectRatio: 'none',
    role: hasSamples ? 'slider' : 'img',
    'aria-label': triText(
      f.presentation.locale,
      view === 'wind' && windMetric === 'ambient'
        ? 'ambient wind speed'
        : environmentViewLabel(view),
    ),
    ...(view === 'wind' ? { 'data-environment-wind-metric': windMetric } : {}),
    ...(hasSamples && currentModel
      ? {
          'data-analysis-source': 'provider-modeled',
          'data-analysis-provider': 'noaa-loofs',
          'data-analysis-formula': currentModel.formulaId,
          'data-analysis-formula-version': currentModel.formulaVersion,
        }
      : {}),
    ...(hasSamples && provider
      ? {
          'data-analysis-source': provider === 'mywindsock' ? 'provider-native' : 'garden-estimate',
          'data-analysis-provider': provider,
        }
      : {}),
    ...(hasSamples
      ? {
          tabindex: 0,
          'aria-valuemin': 0,
          'aria-valuemax': Math.round(elapsedTimeS),
          'aria-valuenow': Math.round(elapsedTimeS),
          'data-environment-chart': view,
          ...(view === 'current-speed' || view === 'current-direction'
            ? {
                'aria-valuetext': environmentChartReadout(
                  f.presentation,
                  samples.at(-1) ?? { elapsedS: elapsedTimeS },
                  scoreModel,
                  view,
                ),
              }
            : {}),
        }
      : {}),
    'data-domain-start-elapsed-s': 0,
    'data-domain-end-elapsed-s': elapsedTimeS,
    ...(axis.pauses.length ? { 'data-domain-pauses': formatElapsedPauses(axis.pauses) } : {}),
    'data-domain-start-x': ENVIRONMENT_CHART_LEFT,
    'data-domain-end-x': ENVIRONMENT_CHART_RIGHT,
  })
  const yTicks: AxisYTick[] = hasSamples
    ? view === 'cumulative' && scoreModel != null
      ? [
          ...environmentAxisTicks(f.presentation, samples, view, scoreModel, 'score'),
          ...environmentAxisTicks(f.presentation, samples, view, null, 'sed'),
        ]
      : environmentAxisTicks(f.presentation, samples, view, scoreModel)
    : [
        ENVIRONMENT_CHART_TOP,
        (ENVIRONMENT_CHART_TOP + ENVIRONMENT_CHART_BOTTOM) / 2,
        ENVIRONMENT_CHART_BOTTOM,
      ].map(vbY => ({ label: '—', vbY }))
  for (const tick of yTicks.filter(tick => tick.attrs?.hidden === undefined))
    f.add(
      svg,
      f.svg('line', {
        class: `tri-environment-gridline${((view === 'wind' && windMetric === 'headwind') || view === 'weather-cost') && hasSamples && tick.vbY === (ENVIRONMENT_CHART_TOP + ENVIRONMENT_CHART_BOTTOM) / 2 ? ' tri-environment-gridline--zero' : ''}`,
        x1: ENVIRONMENT_CHART_LEFT,
        x2: ENVIRONMENT_CHART_RIGHT,
        y1: tick.vbY,
        y2: tick.vbY,
      }),
    )
  const addSeries = (model: EnvironmentScoreModel | null, mode?: 'score' | 'sed'): void => {
    const group = f.svg('g', {
      class: 'tri-environment-series',
      ...(mode ? { 'data-environment-cumulative-series': mode } : {}),
      ...(mode === 'sed' && scoreModel != null ? { hidden: '' } : {}),
    })
    for (const segment of environmentChartSeries(
      f.presentation,
      samples,
      elapsedTimeS,
      view,
      model,
      axis.pauses,
    ))
      f.add(
        group,
        f.svg('path', {
          class: 'tri-environment-line',
          d: segment.path,
          ...(segment.color ? { style: `stroke:${segment.color}` } : {}),
        }),
      )
    f.add(svg, group)
  }
  if (view === 'cumulative' && scoreModel != null) {
    addSeries(scoreModel, 'score')
    addSeries(null, 'sed')
  } else {
    addSeries(null)
  }
  if (hasSamples)
    addElapsedBreaks(
      f,
      svg,
      axis,
      ENVIRONMENT_CHART_BOTTOM,
      fraction =>
        ENVIRONMENT_CHART_LEFT + fraction * (ENVIRONMENT_CHART_RIGHT - ENVIRONMENT_CHART_LEFT),
      ENVIRONMENT_CHART_TOP,
    )
  if (hasSamples)
    f.add(
      svg,
      buildAnalysisSelectionRect(
        f,
        ENVIRONMENT_CHART_BOTTOM - ENVIRONMENT_CHART_TOP,
        undefined,
        ENVIRONMENT_CHART_TOP,
      ),
      f.svg('rect', {
        class: 'tri-environment-selection',
        x: -10,
        y: ENVIRONMENT_CHART_TOP,
        width: 0,
        height: ENVIRONMENT_CHART_BOTTOM - ENVIRONMENT_CHART_TOP,
      }),
      f.svg('line', {
        class: 'tri-environment-cursor',
        x1: ENVIRONMENT_CHART_RIGHT,
        x2: ENVIRONMENT_CHART_RIGHT,
        y1: ENVIRONMENT_CHART_TOP,
        y2: ENVIRONMENT_CHART_BOTTOM,
      }),
    )
  f.add(
    chart,
    axisFrame(
      f,
      svg,
      yTicks,
      32,
      [
        {
          label: environmentElapsedClock(0),
          pct: ENVIRONMENT_CHART_LEFT,
          cls: 'tri-cax-xt--first',
        },
        // Labels keep elapsed clock time, so the middle tick includes stopped time before it.
        {
          label: environmentElapsedClock(
            axis.pauses.length ? axis.elapsedAt(0.5) : elapsedTimeS / 2,
          ),
          pct: 50,
        },
        {
          label: environmentElapsedClock(elapsedTimeS),
          pct: ENVIRONMENT_CHART_RIGHT,
          cls: 'tri-cax-xt--last',
        },
      ],
      true,
      { top: ENVIRONMENT_CHART_TOP, bottom: ENVIRONMENT_CHART_BOTTOM },
      hasSamples
        ? []
        : [
            f.el(
              'span',
              'tri-environment-empty',
              triText(f.presentation.locale, 'no data available'),
              { 'data-i18n': 'no data available' },
            ),
          ],
    ),
  )
  return chart
}
