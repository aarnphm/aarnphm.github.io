import type { StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { MetricSpecState } from './metric-state'
import {
  activityCadenceScale,
  buildImpactLoadFactorTrace as buildImpactLoadFactorTraceNode,
  buildPedalSmoothnessChart as buildPedalSmoothnessChartNode,
  buildPerformanceConditionTrace as buildPerformanceConditionTraceNode,
  buildPowerBalanceChart as buildPowerBalanceChartNode,
  buildPowerPhaseChart as buildPowerPhaseChartNode,
  buildRespirationTrace as buildRespirationTraceNode,
  buildRunGroundContactBalanceTrace as buildRunGroundContactBalanceTraceNode,
  buildRunGroundContactTrace as buildRunGroundContactTraceNode,
  buildRunStepSpeedLossPercentTrace as buildRunStepSpeedLossPercentTraceNode,
  buildRunStepSpeedLossTrace as buildRunStepSpeedLossTraceNode,
  buildRunStrideTrace as buildRunStrideTraceNode,
  buildRunVerticalOscillationTrace as buildRunVerticalOscillationTraceNode,
  buildRunVerticalRatioTrace as buildRunVerticalRatioTraceNode,
  buildTorqueEffectivenessChart as buildTorqueEffectivenessChartNode,
  buildTrainingEffectDetails as buildTrainingEffectDetailsNode,
  formatGroundContactBalance,
  formatGroundContactTime,
  formatImpactLoadFactor,
  formatPerformanceCondition,
  formatRespirationRate,
  formatStepSpeedLoss,
  formatStepSpeedLossPercent,
  formatStrideLength,
  formatVerticalOscillation,
  formatVerticalRatio,
  interpolatePositiveMetricSeries,
  positiveMetricDomain,
  powerBalanceText,
  runGroundContactBalanceLeftPct,
  runGroundContactTimeMs,
  runImpactLoadFactor,
  runStepSpeedLossMps,
  runStepSpeedLossPct,
  runStrideLengthValue,
  runVerticalOscillationCm,
  runVerticalRatioPct,
  scrubDist,
  type DetailCtx,
} from '../../../../util/triathlon-card'
import { CAD_RAMP, HEAT_RAMP, HR_RAMP, RESP_RAMP, SPD_RAMP, STRIDE_RAMP } from '../../maps/palette'
import {
  buildHeartRateTrace,
  buildHrZones,
  buildPowerCurve,
  buildPowerHist,
  buildPowerZones,
  buildTrace,
} from '../primitives'
import {
  cyclingDynamicsPairAtDistance,
  powerPhaseAtDistance,
  requiredMapProfile,
  type MapMetric,
  type MapTraceSurface,
} from './scrub-surfaces'

export const primaryMapMetrics = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  detailContext: DetailCtx,
  state: MetricSpecState,
) => {
  const {
    domF,
    route,
    conditionAt,
    filterPowerZeros,
    powerValues,
    powerDomain,
    pace,
    strideLabel,
    cadUnit,
    paceFmt,
    paceTick,
  } = state
  const paceSpec: MapMetric = {
    label: d.sport === 'bike' ? 'speed' : 'pace',
    shortLabel: d.sport === 'bike' ? 'S' : 'P',
    ramp: SPD_RAMP,
    pick: (_p, i) => pace[i],
    fmt: paceFmt,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (_p, i) => pace[i],
        d.sport === 'bike' ? 'speed' : 'pace',
        () => '',
        paceTick,
        graphDomain,
      ),
    readout: (p, i) => `${scrubDist(presentation, p.d, d.sport)} · ${paceFmt(pace[i])}`,
  }
  const powerSpec: MapMetric = {
    label: 'power',
    shortLabel: 'W',
    ramp: HEAT_RAMP,
    pick: (p, i) => powerValues[i] ?? p.w,
    fmt: v => `${Math.round(v)} W`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (p, i) => powerValues[i] ?? p.w,
        'power',
        m => `${m} W peak`,
        v => `${Math.round(v)}w`,
        graphDomain,
        powerDomain,
        d.sport === 'bike' && detailContext.criticalPower
          ? {
              value: detailContext.criticalPower.criticalPowerWatts,
              label: `eCP ${detailContext.criticalPower.criticalPowerWatts} W`,
            }
          : null,
      ),
    readout: (p, i) =>
      `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(powerValues[i] ?? p.w)} W`,
    traces: graphDomain => {
      const traces: MapTraceSurface[] = []
      const add = (wrap: Element | null, fmt: MapTraceSurface['fmt']): void => {
        if (wrap instanceof HTMLElement) traces.push({ wrap, fmt })
      }
      add(
        buildPowerBalanceChartNode(domF, d, null, false, graphDomain),
        p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.rightPowerPct == null || p.w <= 0 ? '—' : powerBalanceText(p.rightPowerPct)}`,
      )
      add(
        buildTorqueEffectivenessChartNode(domF, d, null, false, graphDomain),
        p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${cyclingDynamicsPairAtDistance(d, p, d.cyclingDynamics?.leftTorqueEffectiveness ?? [], d.cyclingDynamics?.rightTorqueEffectiveness ?? [], '%')}`,
      )
      add(
        buildPedalSmoothnessChartNode(domF, d, null, false, graphDomain),
        p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${cyclingDynamicsPairAtDistance(d, p, d.cyclingDynamics?.leftPedalSmoothness ?? [], d.cyclingDynamics?.rightPedalSmoothness ?? [], '%')}`,
      )
      add(
        buildPowerPhaseChartNode(domF, d, null, graphDomain),
        p => `${scrubDist(presentation, p.d, d.sport)} · ${powerPhaseAtDistance(d, p)}`,
      )
      return traces
    },
    extra: () => [
      buildPowerCurve(presentation, d, detailContext),
      buildPowerHist(presentation, d),
      buildPowerZones(presentation, d, detailContext),
    ],
  }
  const hrSpec: MapMetric = {
    label: 'heart rate',
    shortLabel: 'HR',
    ramp: HR_RAMP,
    pick: p => p.hr,
    fmt: v => `${Math.round(v)} bpm`,
    profile: graphDomain => buildHeartRateTrace(presentation, d, graphDomain),
    readout: p => `${scrubDist(presentation, p.d, d.sport)} · ${p.hr} bpm`,
    extra: () => [
      buildTrainingEffectDetailsNode(domF, d) as HTMLElement | null,
      buildHrZones(presentation, d, detailContext),
    ],
  }
  const cadScale = activityCadenceScale(d.sport)
  const cadenceValues = filterPowerZeros
    ? interpolatePositiveMetricSeries(route, point => point.cad * cadScale)
    : route.map(point => point.cad * cadScale)
  const cadenceDomain = filterPowerZeros ? positiveMetricDomain(cadenceValues) : undefined
  const cadSpec: MapMetric = {
    label: 'cadence',
    shortLabel: 'C',
    ramp: CAD_RAMP,
    zeroGap: !filterPowerZeros,
    pick: (p, i) => cadenceValues[i] ?? p.cad * cadScale,
    fmt: v => `${Math.round(v)} ${cadUnit}`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (p, i) => cadenceValues[i] ?? p.cad * cadScale,
        'cadence',
        m => `${m} ${cadUnit} peak`,
        v => `${Math.round(v)}${cadUnit}`,
        graphDomain,
        cadenceDomain,
      ),
    readout: (p, i) =>
      `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(cadenceValues[i] ?? p.cad * cadScale)} ${cadUnit}`,
  }
  const strideSpec: MapMetric = {
    label: strideLabel,
    shortLabel: 'SL',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runStrideLengthValue(d, p) ?? 0,
    fmt: value => formatStrideLength(presentation, value),
    profile: graphDomain => buildRunStrideTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const stride = runStrideLengthValue(d, p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${stride == null ? '—' : formatStrideLength(presentation, stride)}`
    },
  }
  const groundContactSpec: MapMetric = {
    label: 'ground contact time',
    shortLabel: 'GCT',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runGroundContactTimeMs(p) ?? 0,
    fmt: formatGroundContactTime,
    profile: graphDomain =>
      buildRunGroundContactTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const groundContact = runGroundContactTimeMs(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${groundContact == null ? '—' : formatGroundContactTime(groundContact)}`
    },
  }
  const verticalOscillationSpec: MapMetric = {
    label: 'vertical oscillation',
    shortLabel: 'VO',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runVerticalOscillationCm(p) ?? 0,
    fmt: value => formatVerticalOscillation(presentation, value),
    profile: graphDomain =>
      buildRunVerticalOscillationTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const verticalOscillation = runVerticalOscillationCm(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${verticalOscillation == null ? '—' : formatVerticalOscillation(presentation, verticalOscillation)}`
    },
  }
  const performanceConditionSpec: MapMetric = {
    label: 'performance condition',
    shortLabel: 'PC',
    ramp: STRIDE_RAMP,
    valid: p => conditionAt(p) != null,
    pick: p => conditionAt(p) ?? 0,
    fmt: formatPerformanceCondition,
    profile: graphDomain =>
      requiredMapProfile(
        buildPerformanceConditionTraceNode(domF, d, null, graphDomain),
        'performance condition',
      ),
    readout: p => {
      const value = conditionAt(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatPerformanceCondition(value)}`
    },
  }
  const verticalRatioSpec: MapMetric = {
    label: 'vertical ratio',
    shortLabel: 'VR',
    ramp: STRIDE_RAMP,
    valid: p => runVerticalRatioPct(p) != null,
    pick: p => runVerticalRatioPct(p) ?? 0,
    fmt: formatVerticalRatio,
    profile: graphDomain =>
      requiredMapProfile(
        buildRunVerticalRatioTraceNode(domF, d, null, graphDomain),
        'vertical ratio',
      ),
    readout: p => {
      const value = runVerticalRatioPct(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatVerticalRatio(value)}`
    },
  }
  const groundContactBalanceSpec: MapMetric = {
    label: 'ground contact balance',
    shortLabel: 'GCB',
    ramp: STRIDE_RAMP,
    valid: p => runGroundContactBalanceLeftPct(p) != null,
    pick: p => runGroundContactBalanceLeftPct(p) ?? 0,
    fmt: formatGroundContactBalance,
    profile: graphDomain =>
      requiredMapProfile(
        buildRunGroundContactBalanceTraceNode(domF, d, null, graphDomain),
        'ground contact balance',
      ),
    readout: p => {
      const value = runGroundContactBalanceLeftPct(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatGroundContactBalance(value)}`
    },
  }
  const stepSpeedLossSpec: MapMetric = {
    label: 'step speed loss',
    shortLabel: 'SSL',
    ramp: STRIDE_RAMP,
    valid: p => runStepSpeedLossMps(p) != null,
    pick: p => runStepSpeedLossMps(p) ?? 0,
    fmt: formatStepSpeedLoss,
    profile: graphDomain =>
      requiredMapProfile(
        buildRunStepSpeedLossTraceNode(domF, d, null, graphDomain),
        'step speed loss',
      ),
    readout: p => {
      const value = runStepSpeedLossMps(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatStepSpeedLoss(value)}`
    },
  }
  const stepSpeedLossPercentSpec: MapMetric = {
    label: 'step speed loss percent',
    shortLabel: 'SSL%',
    ramp: STRIDE_RAMP,
    valid: p => runStepSpeedLossPct(p) != null,
    pick: p => runStepSpeedLossPct(p) ?? 0,
    fmt: formatStepSpeedLossPercent,
    profile: graphDomain =>
      requiredMapProfile(
        buildRunStepSpeedLossPercentTraceNode(domF, d, null, graphDomain),
        'step speed loss percent',
      ),
    readout: p => {
      const value = runStepSpeedLossPct(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatStepSpeedLossPercent(value)}`
    },
  }
  const impactLoadFactorSpec: MapMetric = {
    label: 'impact load factor',
    shortLabel: 'ILF',
    ramp: STRIDE_RAMP,
    valid: p => runImpactLoadFactor(p) != null,
    pick: p => runImpactLoadFactor(p) ?? 0,
    fmt: formatImpactLoadFactor,
    profile: graphDomain =>
      requiredMapProfile(
        buildImpactLoadFactorTraceNode(domF, d, null, graphDomain),
        'impact load factor',
      ),
    readout: p => {
      const value = runImpactLoadFactor(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatImpactLoadFactor(value)}`
    },
  }
  const respirationSpec: MapMetric = {
    label: 'respiration',
    shortLabel: 'R',
    ramp: RESP_RAMP,
    pick: p => p.resp ?? 0,
    fmt: formatRespirationRate,
    profile: graphDomain => buildRespirationTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p =>
      `${scrubDist(presentation, p.d, d.sport)} · ${p.resp == null ? '—' : formatRespirationRate(p.resp)}`,
  }
  return {
    paceSpec,
    powerSpec,
    hrSpec,
    cadSpec,
    strideSpec,
    groundContactSpec,
    verticalOscillationSpec,
    performanceConditionSpec,
    verticalRatioSpec,
    groundContactBalanceSpec,
    stepSpeedLossSpec,
    stepSpeedLossPercentSpec,
    impactLoadFactorSpec,
    respirationSpec,
  }
}
