import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type {
  ActivityGraphDomain,
  ActivityTraceDomain,
  ActivityTraceRenderOptions,
} from './chart-geometry'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import {
  runGroundContactBalanceLeftPct,
  runGroundContactTimeMs,
  runImpactLoadFactor,
  runStepSpeedLossMps,
  runStepSpeedLossPct,
  runStrideLengthLabel,
  runStrideLengthValue,
  runVerticalOscillationCm,
  runVerticalRatioPct,
} from './activity-metric-data'
import {
  formatGroundContactBalance,
  formatGroundContactTimeAverage,
  formatImpactLoadFactor,
  formatStepSpeedLoss,
  formatStepSpeedLossPercent,
  formatStrideLength,
  formatVerticalOscillation,
  formatVerticalRatio,
  isImperial,
  M_TO_FT,
} from './metric-text'
import { buildTrace } from './trace-chart'

export const buildRunStrideTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  options?: ActivityTraceRenderOptions<N>,
): N | null => {
  const imperial = isImperial(f.presentation)
  const label = runStrideLengthLabel(d)
  const valuesM = d.route
    .map(point => runStrideLengthValue(d, point))
    .filter((value): value is number => value != null)
  if (d.sport !== 'run' || valuesM.length < 2) return null
  const displayValue = (meters: number): number => (imperial ? meters * M_TO_FT : meters)
  const values = valuesM.map(displayValue)
  const averageM =
    d.garmin?.runningDynamics?.averageStrideLengthCm != null
      ? d.garmin.runningDynamics.averageStrideLengthCm / 100
      : valuesM.reduce((total, value) => total + value, 0) / valuesM.length
  const step = imperial ? 0.5 : 0.25
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  const unit = imperial ? 'ft' : 'm'
  return buildTrace(
    f,
    d,
    point => {
      const meters = runStrideLengthValue(d, point)
      return meters == null ? null : displayValue(meters)
    },
    label,
    () => `${formatStrideLength(f.presentation, averageM)} avg`,
    value => `${value.toFixed(1)}${unit}`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
    undefined,
    undefined,
    options,
  )
}

export const buildRunGroundContactTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  options?: ActivityTraceRenderOptions<N>,
): N | null => {
  const values = d.route
    .map(runGroundContactTimeMs)
    .filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average =
    d.garmin?.runningDynamics?.averageGroundContactTimeMs ??
    values.reduce((total, value) => total + value, 0) / values.length
  const step = 25
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    runGroundContactTimeMs,
    'ground contact time',
    () => `${formatGroundContactTimeAverage(average)} avg`,
    value => `${Math.round(value)}ms`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
    undefined,
    undefined,
    options,
  )
}

export const buildRunVerticalOscillationTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  options?: ActivityTraceRenderOptions<N>,
): N | null => {
  const values = d.route
    .map(runVerticalOscillationCm)
    .filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average =
    d.garmin?.runningDynamics?.averageVerticalOscillationCm ??
    values.reduce((total, value) => total + value, 0) / values.length
  const step = 1
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    runVerticalOscillationCm,
    'vertical oscillation',
    () => `${formatVerticalOscillation(f.presentation, average)} avg`,
    value => `${value.toFixed(1)}cm`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
    undefined,
    undefined,
    options,
  )
}

const runMetricDomain = (values: number[], step: number): ActivityTraceDomain => {
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return { min, max, intervals: 2 }
}

export const buildRunVerticalRatioTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  options?: ActivityTraceRenderOptions<N>,
): N | null => {
  const values = d.route.map(runVerticalRatioPct).filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average =
    d.garmin?.runningDynamics?.averageVerticalRatioPct ??
    values.reduce((total, value) => total + value, 0) / values.length
  return buildTrace(
    f,
    d,
    runVerticalRatioPct,
    'vertical ratio',
    () => `${formatVerticalRatio(average)} avg`,
    formatVerticalRatio,
    runMetricDomain(values, 1),
    selection,
    graphDomain,
    undefined,
    undefined,
    options,
  )
}

export const buildRunGroundContactBalanceTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  options?: ActivityTraceRenderOptions<N>,
): N | null => {
  const values = d.route
    .map(runGroundContactBalanceLeftPct)
    .filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average =
    d.garmin?.runningDynamics?.averageGroundContactBalanceLeftPct ??
    values.reduce((total, value) => total + value, 0) / values.length
  return buildTrace(
    f,
    d,
    runGroundContactBalanceLeftPct,
    'ground contact balance',
    () => `${formatGroundContactBalance(average)} avg`,
    value => `${value.toFixed(1)}% L`,
    runMetricDomain(values, 1),
    selection,
    graphDomain,
    undefined,
    { value: 50, label: '50 / 50' },
    options,
  )
}

export const buildRunStepSpeedLossTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  options?: ActivityTraceRenderOptions<N>,
): N | null => {
  const values = d.route.map(runStepSpeedLossMps).filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const centimetersPerSecond = values.map(value => value * 100)
  const average =
    d.garmin?.runningDynamics?.averageStepSpeedLossMps ??
    values.reduce((total, value) => total + value, 0) / values.length
  return buildTrace(
    f,
    d,
    point => {
      const value = runStepSpeedLossMps(point)
      return value == null ? null : value * 100
    },
    'step speed loss',
    () => `${formatStepSpeedLoss(average)} avg`,
    value => `${value.toFixed(1)}cm/s`,
    runMetricDomain(centimetersPerSecond, 1),
    selection,
    graphDomain,
    undefined,
    undefined,
    options,
  )
}

export const buildRunStepSpeedLossPercentTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  options?: ActivityTraceRenderOptions<N>,
): N | null => {
  const values = d.route.map(runStepSpeedLossPct).filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average =
    d.garmin?.runningDynamics?.averageStepSpeedLossPct ??
    values.reduce((total, value) => total + value, 0) / values.length
  return buildTrace(
    f,
    d,
    runStepSpeedLossPct,
    'step speed loss percent',
    () => `${formatStepSpeedLossPercent(average)} avg`,
    value => `${value.toFixed(1)}%`,
    runMetricDomain(values, 1),
    selection,
    graphDomain,
    undefined,
    undefined,
    options,
  )
}

export const buildImpactLoadFactorTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const values = d.route.map(runImpactLoadFactor).filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average = values.reduce((total, value) => total + value, 0) / values.length
  return buildTrace(
    f,
    d,
    runImpactLoadFactor,
    'impact load factor',
    () => `${formatImpactLoadFactor(average)} avg`,
    value => value.toFixed(2),
    runMetricDomain(values, 0.25),
    selection,
    graphDomain,
  )
}

type RunMetricMode = { mode: string; label: string }

const buildRunMetricControls = <N>(
  f: TriNodeFactory<N>,
  label: string,
  activeMode: string,
  modes: readonly RunMetricMode[],
): N => {
  const controls = f.el('div', 'tri-run-metric-modes', undefined, {
    role: 'group',
    'aria-label': label,
  })
  for (const mode of modes)
    f.add(
      controls,
      f.el(
        'button',
        'tri-run-metric-mode tri-curve-range',
        triText(f.presentation.locale, mode.label),
        {
          type: 'button',
          'data-run-metric-mode': mode.mode,
          'data-i18n': mode.label,
          'aria-pressed': String(mode.mode === activeMode),
        },
      ),
    )
  return controls
}

const runMetricTraceOptions = <N>(
  f: TriNodeFactory<N>,
  label: string,
  activeMode: string,
  mode: string,
  modes: readonly RunMetricMode[],
): ActivityTraceRenderOptions<N> => ({
  wrapClass: 'tri-run-metric-pane',
  wrapAttrs: {
    'data-run-metric-pane': mode,
    'aria-hidden': String(mode !== activeMode),
    ...(mode === activeMode ? {} : { hidden: '' }),
  },
  capExtra: [buildRunMetricControls(f, label, activeMode, modes)],
})

const buildRunMetricPair = <N>(
  f: TriNodeFactory<N>,
  group: string,
  activeMode: string,
  first: N,
  second: N,
): N => {
  const chart = f.el('div', 'tri-run-metric-chart', undefined, {
    'data-run-metric-group': group,
    'data-run-metric-mode': activeMode,
  })
  f.add(chart, first, second)
  return chart
}

export const buildRunVerticalMetricChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
): N | null => {
  const modes = [
    { mode: 'vertical-oscillation', label: 'vertical oscillation' },
    { mode: 'vertical-ratio', label: 'vertical ratio' },
  ]
  const activeMode = 'vertical-oscillation'
  const oscillation = buildRunVerticalOscillationTrace(
    f,
    d,
    selection,
    undefined,
    runMetricTraceOptions(f, 'vertical dynamics', activeMode, modes[0].mode, modes),
  )
  const ratio = buildRunVerticalRatioTrace(
    f,
    d,
    selection,
    undefined,
    runMetricTraceOptions(f, 'vertical dynamics', activeMode, modes[1].mode, modes),
  )
  return oscillation && ratio
    ? buildRunMetricPair(f, 'vertical', activeMode, oscillation, ratio)
    : oscillation
      ? buildRunVerticalOscillationTrace(f, d, selection)
      : buildRunVerticalRatioTrace(f, d, selection)
}

export const buildRunGroundContactMetricChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
): N | null => {
  const modes = [
    { mode: 'ground-contact-time', label: 'ground contact time' },
    { mode: 'ground-contact-balance', label: 'ground contact balance' },
  ]
  const activeMode = 'ground-contact-time'
  const time = buildRunGroundContactTrace(
    f,
    d,
    selection,
    undefined,
    runMetricTraceOptions(f, 'ground contact dynamics', activeMode, modes[0].mode, modes),
  )
  const balance = buildRunGroundContactBalanceTrace(
    f,
    d,
    selection,
    undefined,
    runMetricTraceOptions(f, 'ground contact dynamics', activeMode, modes[1].mode, modes),
  )
  return time && balance
    ? buildRunMetricPair(f, 'ground-contact', activeMode, time, balance)
    : time
      ? buildRunGroundContactTrace(f, d, selection)
      : buildRunGroundContactBalanceTrace(f, d, selection)
}

export const buildRunStepSpeedLossMetricChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
): N | null => {
  const modes = [
    { mode: 'step-speed-loss', label: 'step speed loss' },
    { mode: 'step-speed-loss-percent', label: 'step speed loss percent' },
  ]
  const activeMode = 'step-speed-loss'
  const speed = buildRunStepSpeedLossTrace(
    f,
    d,
    selection,
    undefined,
    runMetricTraceOptions(f, 'step speed loss', activeMode, modes[0].mode, modes),
  )
  const percent = buildRunStepSpeedLossPercentTrace(
    f,
    d,
    selection,
    undefined,
    runMetricTraceOptions(f, 'step speed loss', activeMode, modes[1].mode, modes),
  )
  return speed && percent
    ? buildRunMetricPair(f, 'step-speed-loss', activeMode, speed, percent)
    : speed
      ? buildRunStepSpeedLossTrace(f, d, selection)
      : buildRunStepSpeedLossPercentTrace(f, d, selection)
}
