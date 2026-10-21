import type { StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import {
  buildWalkPowerTrace as buildWalkPowerTraceNode,
  scrubDist,
  type DetailCtx,
} from '../../../../util/triathlon-card'
import { triText } from '../../../../util/triathlon-i18n'
import { walkPowerAt } from '../../../../util/walk-power'
import { HEAT_RAMP } from '../../maps/palette'
import { metricSpecState } from './metric-state'
import { primaryMapMetrics } from './primary-metrics'
import { requiredMapProfile, type MapMetric } from './scrub-surfaces'
import { secondaryMapMetrics } from './secondary-metrics'

const metricSpecsCache = new WeakMap<StravaActivityDetail, { key: string; specs: MapMetric[] }>()

export const metricSpecs = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  detailContext: DetailCtx,
): MapMetric[] => {
  const cacheKey = `${presentation.locale}:${presentation.distance}:${presentation.powerSamples}:${detailContext.criticalPower?.criticalPowerWatts ?? ''}:${detailContext.ftp ?? ''}:${detailContext.vt1 ?? ''}:${detailContext.curveRef.length}:${detailContext.runCurveRef.length}`
  const cached = metricSpecsCache.get(d)
  if (cached && cached.key === cacheKey) return cached.specs
  const state = metricSpecState(presentation, d)
  const primary = primaryMapMetrics(presentation, d, detailContext, state)
  const secondary = secondaryMapMetrics(presentation, d, state)
  const {
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
  } = primary
  const {
    muscleOxygenSpec,
    riderPositionSpec,
    staminaSpec,
    shiftingSpec,
    temperatureSpec,
    elevSpec,
    uvSpec,
    windSpec,
  } = secondary
  const {
    domF,
    hasPower,
    hasHr,
    hasCad,
    hasStride,
    hasGroundContact,
    hasVerticalOscillation,
    hasResp,
    hasElev,
    flags,
    hasUv,
    hasWind,
    hasMuscleOxygen,
    hasRiderPosition,
    hasShifting,
    hasThermal,
  } = state
  const specs: MapMetric[] = []
  if (d.sport === 'bike') {
    if (hasPower) specs.push(powerSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasCad) specs.push(cadSpec)
    if (flags.performanceCondition) specs.push(performanceConditionSpec)
    if (hasResp) specs.push(respirationSpec)
    if (hasMuscleOxygen) specs.push(muscleOxygenSpec)
    specs.push(paceSpec)
    if (hasElev) specs.push(elevSpec)
    if (hasRiderPosition) specs.push(riderPositionSpec)
    if (flags.stamina) specs.push(staminaSpec)
    if (hasShifting) specs.push(shiftingSpec)
    if (hasThermal) specs.push(temperatureSpec)
  } else if (d.sport === 'run') {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
    if (flags.stamina) specs.push(staminaSpec)
    if (hasCad) specs.push(cadSpec)
    if (flags.performanceCondition) specs.push(performanceConditionSpec)
    if (hasStride) specs.push(strideSpec)
    if (flags.verticalRatio) specs.push(verticalRatioSpec)
    if (hasGroundContact) specs.push(groundContactSpec)
    if (flags.groundContactBalance) specs.push(groundContactBalanceSpec)
    if (hasVerticalOscillation) specs.push(verticalOscillationSpec)
    if (flags.stepSpeedLoss) specs.push(stepSpeedLossSpec)
    if (flags.stepSpeedLossPercent) specs.push(stepSpeedLossPercentSpec)
    if (hasResp) specs.push(respirationSpec)
    if (flags.impactLoadFactor) specs.push(impactLoadFactorSpec)
    if (hasElev) specs.push(elevSpec)
    if (hasPower) specs.push(powerSpec)
    if (hasThermal) specs.push(temperatureSpec)
  } else if (d.sport === 'walk') {
    specs.push(paceSpec)
    if (hasPower) specs.push(powerSpec)
    else if (!d.deviceWatts && d.walkPower)
      specs.push({
        label: 'walking power',
        shortLabel: 'W',
        ramp: HEAT_RAMP,
        valid: point => walkPowerAt(d.walkPower, point.elapsedS) != null,
        pick: point => walkPowerAt(d.walkPower, point.elapsedS) ?? 0,
        fmt: value => `${Math.round(value)} W · ${triText(presentation.locale, 'estimated')}`,
        profile: graphDomain =>
          requiredMapProfile(buildWalkPowerTraceNode(domF, d, null, graphDomain), 'walking power'),
        readout: point => {
          const watts = walkPowerAt(d.walkPower, point.elapsedS)
          return `${scrubDist(presentation, point.d, d.sport)} · ${watts == null ? '—' : `${Math.round(watts)} W`} · ${triText(presentation.locale, 'estimated')}`
        },
      })
    if (hasHr) specs.push(hrSpec)
    if (flags.stamina) specs.push(staminaSpec)
    if (hasCad) specs.push(cadSpec)
    if (flags.performanceCondition) specs.push(performanceConditionSpec)
    if (hasResp) specs.push(respirationSpec)
    if (hasElev) specs.push(elevSpec)
    if (hasThermal) specs.push(temperatureSpec)
  } else {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
    if (flags.stamina) specs.push(staminaSpec)
    if (flags.performanceCondition) specs.push(performanceConditionSpec)
    if (hasResp) specs.push(respirationSpec)
    if (hasThermal) specs.push(temperatureSpec)
  }
  if (hasUv) specs.push(uvSpec)
  if (hasWind) specs.push(windSpec)
  metricSpecsCache.set(d, { key: cacheKey, specs })
  return specs
}
