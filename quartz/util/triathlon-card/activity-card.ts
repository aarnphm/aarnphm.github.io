import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { DetailCtx, TriNodeFactory } from './construction-contracts'
import { triathlonActivityAnchor } from '../triathlon-date-route'
import { triText } from '../triathlon-i18n'
import { buildMyWindsockGraphs } from '../triathlon-mywindsock'
import { triathlonTraceEnabled, type TriathlonTraceSettings } from '../triathlon-trace-settings'
import { buildActivityIcon } from './activity-icons'
import { interpolatePositiveMetricSeries, routeStreamFlags } from './activity-metric-data'
import { buildActivityShareButton } from './activity-share'
import {
  activityAnalysisStatAttrs,
  activityEquipmentStatAttrs,
  activitySourceStatAttrs,
  relativeHumidityStatAttrs,
} from './activity-stat-details'
import { activityStatRows, activityTableRows } from './activity-stat-rows'
import {
  buildActivityAnalyzeButton,
  buildAnalysisBar,
  buildPaceTrace,
  buildSpeedTrace,
} from './analysis-presentation'
import { buildBestEfforts } from './best-efforts'
import { buildFueling, buildReservedFueling, statsTable } from './card-tables'
import { positiveMetricDomain } from './chart-geometry'
import {
  buildPedalSmoothnessChart,
  buildTorqueEffectivenessChart,
} from './cycling-dynamics-percent'
import { buildPowerPhaseChart, buildRiderPositionChart } from './cycling-phase-position'
import { buildCyclingPowerChart } from './cycling-power-chart'
import {
  buildCrankTorqueChart,
  buildCyclingImpactLoadTrace,
  buildHeartRateTrace,
  buildIntensityFactorChart,
  buildWalkPowerTrace,
} from './effort-traces'
import { buildElevation, buildUnavailableElevation } from './elevation-chart'
import { buildEnvironmentAnalysis } from './environment-analysis'
import { buildSaunaSummary } from './heart-rate-workout'
import { activityCadenceScale, activityCadenceUnit, excludesZeroPower } from './metric-text'
import { buildPerformanceConditionTrace, buildStaminaChart } from './physiology-charts'
import { buildPool, buildSwimStrokes } from './pool-summary'
import { buildPowerBalanceChart } from './power-balance-chart'
import { buildPowerCurve } from './power-curve-chart'
import { buildRoute } from './route-chart'
import { buildRunHeartCostChart } from './run-heart-cost-chart'
import {
  buildImpactLoadFactorTrace,
  buildRunGroundContactMetricChart,
  buildRunStepSpeedLossMetricChart,
  buildRunVerticalMetricChart,
} from './run-mechanics-traces'
import { buildRunStrideCadenceChart } from './run-stride-cadence-chart'
import { buildRunWalkTrace } from './run-walk-chart'
import {
  buildCoreTemperatureTrace,
  buildHeatStrainTrace,
  buildMuscleOxygenTrace,
  buildRespirationTrace,
  buildSkinTemperatureTrace,
  buildTemperatureTrace,
} from './sensor-traces'
import { buildShiftingChart } from './shifting-chart'
import { buildActivityMoves, buildStrengthExercises } from './strength-moves'
import { buildSwimPowerCurve } from './swim-power-chart'
import { buildSwimTrends } from './swim-trend-charts'
import { buildTrace } from './trace-chart'
import { activityTrainingEffectLabel, dominantTrainingEffectGroup } from './training-effect-data'
import { buildSaunaHeatTrainingLoad, buildTrainingEffectDetails } from './training-effect-details'
import { buildWorkoutAnalysisSection } from './workout-analysis'
import { buildHrZones, buildPowerHist, buildPowerZones, zoneDuo } from './zone-distributions'

export const buildActivity = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  expanded = false,
  ctx?: DetailCtx,
  fillMissingRunPower = false,
  embedded = false,
  traceSettings?: TriathlonTraceSettings,
  reserveFueling = false,
): N => {
  const normalizeBikeMetrics = excludesZeroPower(f.presentation) && d.sport === 'bike'
  const normalizedPower = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.w)
    : null
  const normalizedCadence = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.cad)
    : null
  const activityAnchor = triathlonActivityAnchor(d.id)
  const summaryTrainingEffectGroup = dominantTrainingEffectGroup(activityTrainingEffectLabel(d))
  const primaryStatCount = activityStatRows(f.presentation, d, false).length
  const wrap = f.el('section', expanded ? 'tri-act tri-act--expanded' : 'tri-act', undefined, {
    ...(activityAnchor ? { id: activityAnchor } : {}),
    'data-activity-id': `${d.id}`,
    'data-activity-title': d.name || d.sport,
    'data-sport': d.sport,
  })
  const head = f.el('div', 'tri-act-head')
  f.add(head, buildActivityIcon(f, d))
  const headActions = f.el('div', 'tri-act-head-actions')
  const share = buildActivityShareButton(f, d, fillMissingRunPower, ctx)
  if (d.commute === true)
    f.add(
      headActions,
      f.el('span', 'tri-activity-tag', triText(f.presentation.locale, 'commute'), {
        title: triText(f.presentation.locale, 'commute'),
        'data-activity-tag': 'commute',
      }),
    )
  if (d.virtual === true)
    f.add(
      headActions,
      f.el('span', 'tri-activity-tag', triText(f.presentation.locale, 'virtual'), {
        title: triText(f.presentation.locale, 'virtual'),
        'data-activity-tag': 'virtual',
      }),
    )
  f.add(headActions, buildActivityAnalyzeButton(f, d))
  if (share) f.add(headActions, share)
  f.add(head, headActions)
  f.add(wrap, head)
  if (d.provenance)
    f.add(
      wrap,
      f.el('p', 'tri-note', d.provenance.note, { 'data-provenance': d.provenance.source }),
    )
  f.add(
    wrap,
    statsTable(f, activityTableRows(f.presentation, d, fillMissingRunPower), (label, index) => {
      if (label === 'source') return activitySourceStatAttrs(d)
      if (label === 'equipment') return activityEquipmentStatAttrs(d)
      if (label === 'training effect')
        return { 'data-training-effect-group': summaryTrainingEffectGroup }
      const analysisAttrs = activityAnalysisStatAttrs(d, label)
      if (analysisAttrs) return analysisAttrs
      if (index >= primaryStatCount && label === 'humidity') return relativeHumidityStatAttrs(d)
      return undefined
    }),
  )
  let hasSummaryVisual = false
  const saunaFigs =
    d.sport === 'sauna' && d.route.length < 2
      ? f.el('div', 'tri-act-figs tri-act-figs--sauna')
      : null
  const saunaSummary = buildSaunaSummary(f, d)
  if (saunaSummary) {
    f.add(saunaFigs ?? wrap, saunaSummary)
    hasSummaryVisual = true
  }
  if (d.moves) {
    const moves = buildActivityMoves(f, d.moves)
    if (moves) {
      f.add(saunaFigs ?? wrap, moves)
      hasSummaryVisual = true
    }
  }
  const strength = d.strength ? buildStrengthExercises(f, d.strength) : null
  if (strength && d.sport !== 'sauna') {
    f.add(wrap, strength)
    hasSummaryVisual = true
  }
  let hasFueling = false
  if (d.fueling) {
    const fueling = buildFueling(f, d.fueling)
    if (fueling) {
      f.add(wrap, fueling)
      hasFueling = true
    }
  }
  if (embedded && reserveFueling && !hasFueling) f.add(wrap, buildReservedFueling(f))
  const analysis = saunaSummary ? null : buildAnalysisBar(f, d)
  const analysisSelection = null
  if (d.route.length >= 2) {
    const secondary =
      d.sport === 'swim'
        ? (buildSwimStrokes(f, d) ?? (embedded ? buildUnavailableElevation(f) : null))
        : buildElevation(f, d, analysisSelection)
    const figs = f.el(
      'div',
      `tri-act-figs tri-act-figs--route${secondary ? ' tri-act-figs--split' : ''}`,
    )
    f.add(figs, buildRoute(f, d.route))
    if (secondary) f.add(figs, secondary)
    if (analysis) f.add(figs, analysis)
    f.add(wrap, figs)
    hasSummaryVisual = true
  } else if (d.sport === 'swim') {
    const figs = f.el('div', 'tri-act-figs tri-act-figs--pool')
    f.add(figs, buildPool(f, d))
    f.add(wrap, figs)
    hasSummaryVisual = true
  }
  if (saunaFigs && (hasSummaryVisual || analysis)) {
    if (analysis) f.add(saunaFigs, analysis)
    f.add(wrap, saunaFigs)
    hasSummaryVisual = true
  }
  if (embedded && !hasSummaryVisual)
    f.add(
      wrap,
      f.el('div', 'tri-act-figs tri-act-figs--empty', undefined, { 'aria-hidden': 'true' }),
    )
  const swimTrends = buildSwimTrends(f, d, traceSettings)
  {
    const moreId = `tri-act-more-${d.id}`
    const more = f.el('div', 'tri-act-more', undefined, { id: moreId })
    const flags = routeStreamFlags(d)
    const workoutAnalysis = buildWorkoutAnalysisSection(f, d, embedded, ctx?.zones, ctx?.vt1)
    if (workoutAnalysis) f.add(more, workoutAnalysis.element)
    const activityGraphs: N[] = []
    const stamina = buildStaminaChart(f, d, analysisSelection)
    if (stamina) activityGraphs.push(stamina)
    if (triathlonTraceEnabled(traceSettings, 'performance-condition'))
      activityGraphs.push(buildPerformanceConditionTrace(f, d, analysisSelection))
    const trainingEffect = buildTrainingEffectDetails(f, d)
    if (trainingEffect) activityGraphs.push(trainingEffect)
    if (ctx) {
      const zones = zoneDuo(
        f,
        triathlonTraceEnabled(traceSettings, 'heart-rate-zones') ? buildHrZones(f, d, ctx) : null,
        triathlonTraceEnabled(traceSettings, 'power-zones') ? buildPowerZones(f, d, ctx) : null,
      )
      if (zones) activityGraphs.push(zones)
    }
    if (flags.heatStrain) {
      const heatStrain = buildHeatStrainTrace(f, d, analysisSelection)
      if (heatStrain) activityGraphs.push(heatStrain)
    }
    if (flags.coreTemperature) {
      const coreTemperature = buildCoreTemperatureTrace(f, d, analysisSelection)
      if (coreTemperature) activityGraphs.push(coreTemperature)
    }
    if (flags.skinTemperature) {
      const skinTemperature = buildSkinTemperatureTrace(f, d, analysisSelection)
      if (skinTemperature) activityGraphs.push(skinTemperature)
    }
    if (flags.hr && workoutAnalysis?.metric !== 'hr')
      activityGraphs.push(buildHeartRateTrace(f, d, analysisSelection))
    if (flags.temp) activityGraphs.push(buildTemperatureTrace(f, d, analysisSelection))
    if (flags.cad) {
      const cadenceScale = activityCadenceScale(d.sport)
      const cadenceUnit = activityCadenceUnit(d.sport)
      activityGraphs.push(
        buildTrace(
          f,
          d,
          (point, index) => (normalizedCadence?.[index] ?? point.cad) * cadenceScale,
          'cadence',
          max => `${max} ${cadenceUnit} peak`,
          value => `${Math.round(value)}${cadenceUnit}`,
          normalizedCadence
            ? positiveMetricDomain(normalizedCadence.map(value => (value ?? 0) * cadenceScale))
            : undefined,
          analysisSelection,
        ),
      )
    }
    if (flags.resp) activityGraphs.push(buildRespirationTrace(f, d, analysisSelection))
    if (flags.power)
      activityGraphs.push(
        buildTrace(
          f,
          d,
          (point, index) => normalizedPower?.[index] ?? point.w,
          'power',
          max => `${max} W peak`,
          value => `${Math.round(value)}w`,
          normalizedPower ? positiveMetricDomain(normalizedPower) : undefined,
          analysisSelection,
          undefined,
          undefined,
          d.sport === 'bike' && d.activityCriticalPower
            ? {
                value: d.activityCriticalPower.criticalPowerWatts,
                label: `eCP ${d.activityCriticalPower.criticalPowerWatts} W`,
              }
            : null,
        ),
      )
    if (d.sport === 'walk') {
      const walkPower = buildWalkPowerTrace(f, d, analysisSelection)
      if (walkPower) activityGraphs.push(walkPower)
      const pace = buildPaceTrace(f, d, analysisSelection)
      if (pace) activityGraphs.push(pace)
    }
    if (triathlonTraceEnabled(traceSettings, 'intensity-factor')) {
      const intensity = buildIntensityFactorChart(f, d, analysisSelection)
      if (intensity) activityGraphs.push(intensity)
    }
    const powerBalance = buildPowerBalanceChart(f, d, analysisSelection, embedded)
    if (powerBalance) activityGraphs.push(powerBalance)
    const torqueEffectiveness = buildTorqueEffectivenessChart(f, d, analysisSelection, embedded)
    if (torqueEffectiveness) activityGraphs.push(torqueEffectiveness)
    const pedalSmoothness = buildPedalSmoothnessChart(f, d, analysisSelection, embedded)
    if (pedalSmoothness) activityGraphs.push(pedalSmoothness)
    const powerPhase = buildPowerPhaseChart(f, d, analysisSelection)
    if (powerPhase) activityGraphs.push(powerPhase)
    const riderPosition = buildRiderPositionChart(f, d, analysisSelection)
    if (riderPosition) activityGraphs.push(riderPosition)
    const shifting = buildShiftingChart(f, d, analysisSelection)
    if (shifting) activityGraphs.push(shifting)
    const speed = triathlonTraceEnabled(traceSettings, 'speed')
      ? buildSpeedTrace(f, d, analysisSelection)
      : null
    if (speed) activityGraphs.push(speed)
    if (triathlonTraceEnabled(traceSettings, 'stride-length')) {
      const stride = buildRunStrideCadenceChart(f, d, analysisSelection)
      if (stride) activityGraphs.push(stride)
    }
    if (flags.verticalOscillation || flags.verticalRatio) {
      const vertical = buildRunVerticalMetricChart(f, d, analysisSelection)
      if (vertical) activityGraphs.push(vertical)
    }
    if (flags.groundContact || flags.groundContactBalance) {
      const groundContact = buildRunGroundContactMetricChart(f, d, analysisSelection)
      if (groundContact) activityGraphs.push(groundContact)
    }
    if (flags.stepSpeedLoss || flags.stepSpeedLossPercent) {
      const stepSpeedLoss = buildRunStepSpeedLossMetricChart(f, d, analysisSelection)
      if (stepSpeedLoss) activityGraphs.push(stepSpeedLoss)
    }
    if (flags.runWalk) {
      const runWalk = buildRunWalkTrace(f, d)
      if (runWalk) activityGraphs.push(runWalk)
    }
    if (flags.impactLoadFactor) {
      const impactLoadFactor = buildImpactLoadFactorTrace(f, d, analysisSelection)
      if (impactLoadFactor) activityGraphs.push(impactLoadFactor)
    }
    if (flags.muscleOxygen) {
      const muscleOxygen = buildMuscleOxygenTrace(f, d, analysisSelection)
      if (muscleOxygen) activityGraphs.push(muscleOxygen)
    }
    if (triathlonTraceEnabled(traceSettings, 'run-heart-cost')) {
      const heartCost = buildRunHeartCostChart(f, d, analysisSelection)
      if (heartCost) activityGraphs.push(heartCost)
    }
    if (triathlonTraceEnabled(traceSettings, 'crank-torque')) {
      const torque = buildCrankTorqueChart(f, d, analysisSelection)
      if (torque) activityGraphs.push(torque)
    }
    if (triathlonTraceEnabled(traceSettings, 'impact-load-factor')) {
      const impactLoad = buildCyclingImpactLoadTrace(f, d, analysisSelection)
      if (impactLoad) activityGraphs.push(impactLoad)
    }
    if (triathlonTraceEnabled(traceSettings, 'cycling-power')) {
      const power = buildCyclingPowerChart(f, d, analysisSelection)
      if (power) activityGraphs.push(power)
    }
    f.add(more, ...activityGraphs)
    if (strength && d.sport === 'sauna') f.add(more, strength)
    const environment = buildEnvironmentAnalysis(f, d)
    if (environment) f.add(more, environment)
    const windGraphs = buildMyWindsockGraphs(f, d)
    if (windGraphs) f.add(more, windGraphs)
    if (swimTrends) f.add(more, swimTrends)
    const swimPowerCurve = buildSwimPowerCurve(f, d, embedded)
    if (swimPowerCurve) f.add(more, swimPowerCurve)
    if (d.sport === 'swim') {
      const distribution = buildPowerHist(f, d)
      if (distribution) f.add(more, distribution)
    }

    const saunaHtl = saunaSummary ? null : buildSaunaHeatTrainingLoad(f, d)
    if (saunaHtl) f.add(more, saunaHtl)
    if (ctx) {
      const charts = [
        triathlonTraceEnabled(traceSettings, 'power-curve')
          ? buildPowerCurve(f, d, ctx, embedded)
          : null,
        d.sport !== 'swim' && triathlonTraceEnabled(traceSettings, '25w-power-distribution')
          ? buildPowerHist(f, d)
          : null,
      ]
      for (const chart of charts) if (chart) f.add(more, chart)
    }
    const bestEfforts = buildBestEfforts(f, d)
    if (bestEfforts) f.add(more, bestEfforts)
    f.add(
      wrap,
      f.el('button', 'tri-act-toggle', expanded ? '− see less' : '+ see more', {
        type: 'button',
        'aria-expanded': String(expanded),
        'aria-controls': moreId,
      }),
      more,
    )
  }
  return wrap
}
