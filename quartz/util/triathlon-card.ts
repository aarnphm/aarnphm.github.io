import type { StravaActivityDetail } from '../plugins/stores/strava'
import { buildActivity } from './triathlon-card/activity-card'
import { buildRestStatus } from './triathlon-card/activity-icons'
import { moreStatRows } from './triathlon-card/activity-stat-details'
import { activityStatRows } from './triathlon-card/activity-stat-rows'
import { buildRecovery, fuelingRows } from './triathlon-card/card-tables'
import {
  dayCardActivitiesExpanded,
  type DayCardExtras,
  type DayCardPayload,
  type DetailCtx,
  type TriNodeFactory,
} from './triathlon-card/construction-contracts'
import { buildDayAnalytics, buildDaySleepSection } from './triathlon-card/day-analytics'
import { buildDayTrackPager } from './triathlon-card/day-track'
import { prettyDate } from './triathlon-card/metric-text'
import { triText } from './triathlon-i18n'
export {
  dayCardActivitiesExpanded,
  parseExcludedActivityIds,
} from './triathlon-card/construction-contracts'
export type {
  DayCardExtras,
  DayCardPayload,
  DetailCtx,
  TriNodeFactory,
} from './triathlon-card/construction-contracts'
export {
  HR_ZONE_NAMES,
  KM_TO_MI,
  M_TO_FT,
  POWER_ZONE_NAMES,
  activityAverageCadence,
  activityCadenceScale,
  activityCadenceUnit,
  clock,
  dist,
  distCombined,
  dlabel,
  dur,
  formatAltitude,
  formatGroundContactBalance,
  formatGroundContactTime,
  formatImpactLoadFactor,
  formatMuscleOxygen,
  formatPerformanceCondition,
  formatRespirationRate,
  formatStepSpeedLoss,
  formatStepSpeedLossPercent,
  formatStrideLength,
  formatTemperature,
  formatThermalTemperature,
  formatVerticalOscillation,
  formatVerticalRatio,
  powerBalanceText,
  raceDistanceValue,
  rate,
  scrubDist,
  speedKph,
  zoneClock,
} from './triathlon-card/metric-text'
export {
  activityCyclingIntensityPoints,
  activityCyclingPowerPoints,
  activityCyclingTorquePoints,
  activityHeartRateTracePoints,
  activityPhysiologyTracePoints,
  activityPowerDistributionPercentages,
  activityRunHeartCostPoints,
  activityThermalTracePoints,
  activityTraceUsesElapsedAxis,
  activityWalkPowerPoints,
  activityZonePercentages,
  gradeAt,
  hasHeartRateTrace,
  interpolatePositiveMetricSeries,
  performanceConditionValue,
  powerViewActivity,
  routeStreamFlags,
  runGroundContactBalanceLeftPct,
  runGroundContactTimeMs,
  runHeartCostValue,
  runImpactLoadFactor,
  runStepSpeedLossMps,
  runStepSpeedLossPct,
  runStrideLengthLabel,
  runStrideLengthM,
  runStrideLengthValue,
  runVerticalOscillationCm,
  runVerticalRatioPct,
} from './triathlon-card/activity-metric-data'
export type { RunHeartCostWindow } from './triathlon-card/activity-metric-data'
export { activityAnalysisAvailable, activitySelectionSummary } from './triathlon-card/analysis-data'
export type { ActivitySelectionSummary } from './triathlon-card/analysis-data'
export {
  LAYERS_ICON,
  buildActivityIcon,
  buildIcon,
  buildLayers,
} from './triathlon-card/activity-icons'
export {
  analysisHeartRateChangeText,
  buildActivityAnalyzeButton,
  buildAnalysisBar,
} from './triathlon-card/analysis-presentation'
export { buildRoute } from './triathlon-card/route-chart'
export {
  axisFrame,
  axisNumber,
  buildPowerCurveGrid,
  niceStep,
  positiveMetricDomain,
} from './triathlon-card/chart-geometry'
export type {
  ActivityTraceDomain,
  ActivityTraceReference,
  AxisXTick,
  AxisYTick,
} from './triathlon-card/chart-geometry'
export {
  analysisChartSelectionBounds,
  analysisDomainFraction,
  analysisDomainValueAt,
} from './triathlon-card/chart-selection'
export type { ActivityAnalysisChartDomain } from './triathlon-card/chart-selection'
export { buildElevation, climbGradeBand } from './triathlon-card/elevation-chart'
export { buildTrace } from './triathlon-card/trace-chart'
export {
  buildCrankTorqueChart,
  buildHeartRateTrace,
  buildIntensityFactorChart,
  buildWalkPowerTrace,
} from './triathlon-card/effort-traces'
export { buildCyclingPowerChart } from './triathlon-card/cycling-power-chart'
export { buildRunStrideCadenceChart } from './triathlon-card/run-stride-cadence-chart'
export { buildRunHeartCostChart } from './triathlon-card/run-heart-cost-chart'
export { cyclingWattsCellText } from './triathlon-card/watts-heatmap'
export { buildPowerBalanceChart } from './triathlon-card/power-balance-chart'
export {
  cyclingDynamicsIndexAtDistance,
  riderPositionAtDistance,
} from './triathlon-card/cycling-dynamics-data'
export {
  buildPedalSmoothnessChart,
  buildTorqueEffectivenessChart,
} from './triathlon-card/cycling-dynamics-percent'
export {
  buildPowerPhaseChart,
  buildRiderPositionChart,
} from './triathlon-card/cycling-phase-position'
export { activityGearRatioDistribution, gearShiftAtFraction } from './triathlon-card/shifting-data'
export { SHIFTING_SETTINGS, buildShiftingChart } from './triathlon-card/shifting-chart'
export type { ShiftingSetting } from './triathlon-card/shifting-chart'
export {
  buildPerformanceConditionTrace,
  buildStaminaChart,
} from './triathlon-card/physiology-charts'
export {
  buildImpactLoadFactorTrace,
  buildRunGroundContactBalanceTrace,
  buildRunGroundContactTrace,
  buildRunStepSpeedLossPercentTrace,
  buildRunStepSpeedLossTrace,
  buildRunStrideTrace,
  buildRunVerticalOscillationTrace,
  buildRunVerticalRatioTrace,
} from './triathlon-card/run-mechanics-traces'
export { runWalkSegmentAt } from './triathlon-card/run-walk-chart'
export {
  buildCoreTemperatureTrace,
  buildHeatStrainTrace,
  buildMuscleOxygenTrace,
  buildRespirationTrace,
  buildSkinTemperatureTrace,
  buildTemperatureTrace,
} from './triathlon-card/sensor-traces'
export {
  cyclingWorkoutLaps,
  cyclingWorkoutPowerSummary,
  swimWorkoutLaps,
} from './triathlon-card/workout-lap-data'
export { buildWorkoutAnalysis } from './triathlon-card/workout-analysis'
export {
  ALL_ENVIRONMENT_CHART_VIEWS,
  ENVIRONMENT_CHART_VIEWS,
  ENVIRONMENT_PROVIDER_CHART_VIEWS,
} from './triathlon-card/environment-scale'
export type { EnvironmentChartView } from './triathlon-card/environment-scale'
export {
  environmentChartReadout,
  environmentElapsedClock,
  relativeWindReadout,
} from './triathlon-card/environment-readout'
export { environmentChartSeries } from './triathlon-card/environment-chart'
export { buildEnvironmentAnalysis } from './triathlon-card/environment-analysis'
export { buildPool } from './triathlon-card/pool-summary'
export {
  swimActivityBlocks,
  swimActivityPointLabel,
  swimTrendHoverAt,
} from './triathlon-card/swim-trend-data'
export type { SwimTrendChartPoint, SwimTrendMode } from './triathlon-card/swim-trend-data'
export { buildSwimTrends } from './triathlon-card/swim-trend-charts'
export { buildRecovery, fuelingRows, statRow } from './triathlon-card/card-tables'
export {
  POWER_CURVE_EXTRA_AXIS_MARKERS,
  POWER_CURVE_WATT_STEP,
  nearestPowerCurvePoint,
  nearestPowerCurveValue,
  normalizePowerCurvePoints,
  powerCurveAxisTicks,
  powerCurveDurationTicks,
  powerCurveFraction,
  powerCurveHoverAt,
  powerCurvePathPoints,
  powerCurveValueText,
  powerCurveWeight,
} from './triathlon-card/power-curve-data'
export { buildBestEfforts } from './triathlon-card/best-efforts'
export {
  buildHrZones,
  buildPowerHist,
  buildPowerZones,
  powerHistogramReadout,
  zoneDuo,
} from './triathlon-card/zone-distributions'
export { buildSwimPowerCurve } from './triathlon-card/swim-power-chart'
export { buildPowerCurve } from './triathlon-card/power-curve-chart'
export {
  activityTrainingEffectLabel,
  dominantTrainingEffectGroup,
  formatTrainingEffectLabel,
  formatTrainingEffectNote,
} from './triathlon-card/training-effect-data'
export {
  buildSaunaHeatTrainingLoad,
  buildTrainingEffectDetails,
} from './triathlon-card/training-effect-details'
export {
  activityAnalysisStatAttrs,
  activityEquipmentStatAttrs,
  activitySourceStatAttrs,
  moreStatRows,
  relativeHumidityStatAttrs,
} from './triathlon-card/activity-stat-details'
export { activityStatRows, activityTableRows } from './triathlon-card/activity-stat-rows'
export { buildActivityShareButton } from './triathlon-card/activity-share'
export { buildActivity } from './triathlon-card/activity-card'
export {
  activityComparisonDisplayValueAtDistance,
  activityComparisonFractionForKey,
  activityComparisonMetricAtDistance,
  activityComparisonMetricsForSport,
} from './triathlon-card/comparison-metrics'
export type { ActivityComparisonMetric } from './triathlon-card/comparison-metrics'
export { activityComparisonEligible } from './triathlon-card/comparison-geography'
export { activityCompareColor } from './triathlon-card/comparison-plot'
export { activityComparisonMathText } from './triathlon-card/comparison-stat-format'
export type { ActivityComparisonStatsScope } from './triathlon-card/comparison-stat-format'
export { buildActivityComparisonStats } from './triathlon-card/comparison-stats'
export { buildActivityComparison } from './triathlon-card/activity-comparison'
export { buildTimelineDayCard } from './triathlon-card/timeline-day-card'
export {
  daySleepMovementLabel,
  daySleepSampleClock,
  daySleepStageLabel,
} from './triathlon-card/sleep-chart-data'
export { buildSleepRespirationChart } from './triathlon-card/sleep-charts'
export {
  buildDayAnalytics,
  buildDaySleepAnalytics,
  sleepSupplementMetrics,
} from './triathlon-card/day-analytics'
export { buildDayPageTabs, dayTrackPages } from './triathlon-card/day-track'

const dayDetailsByMap = new WeakMap<
  DayCardPayload['details'],
  Map<string, StravaActivityDetail[]>
>()
const recentLocationByMap = new WeakMap<DayCardPayload['details'], string | undefined>()

const dayDetails = (payload: DayCardPayload, dateIso: string): StravaActivityDetail[] => {
  let byDate = dayDetailsByMap.get(payload.details)
  if (!byDate) {
    byDate = new Map()
    for (const detail of Object.values(payload.details)) {
      let list = byDate.get(detail.date)
      if (!list) {
        list = []
        byDate.set(detail.date, list)
      }
      list.push(detail)
    }
    for (const list of byDate.values()) {
      list.sort((a, b) => b.distanceKm - a.distanceKm)
    }
    dayDetailsByMap.set(payload.details, byDate)
  }
  return byDate.get(dateIso) ?? []
}

const recentLocation = (payload: DayCardPayload): string | undefined => {
  if (recentLocationByMap.has(payload.details)) {
    return recentLocationByMap.get(payload.details)
  }
  const location =
    Object.values(payload.details)
      .sort((a, b) => b.date.localeCompare(a.date))
      .find(d => d.location)?.location ?? undefined
  recentLocationByMap.set(payload.details, location)
  return location
}

export const buildDayCard = <N>(
  f: TriNodeFactory<N>,
  dateIso: string,
  payload: DayCardPayload | null,
  extras: DayCardExtras = {},
  activity?: (d: StravaActivityDetail, reserveFueling: boolean) => N,
  ctx?: DetailCtx,
): N => {
  const allDay = payload ? dayDetails(payload, dateIso) : []
  const selectedDay = extras.activityId
    ? allDay.filter(d => `${d.id}` === extras.activityId)
    : allDay
  const excludedActivityIds = new Set(extras.excludedActivityIds)
  const visibleDay =
    excludedActivityIds.size > 0
      ? selectedDay.filter(d => !excludedActivityIds.has(`${d.id}`))
      : selectedDay
  const day = visibleDay.filter(
    d =>
      (!extras.sport || d.sport === extras.sport) &&
      (!extras.settings?.focused || d.sport === 'bike' || d.sport === 'run' || d.sport === 'swim'),
  )
  const summaryRows =
    extras.embedded === true && day.length > 1
      ? Math.max(
          ...day.map(
            d =>
              activityStatRows(f.presentation, d).length +
              moreStatRows(f.presentation, d, extras.event != null).length,
          ),
        )
      : null
  const sharedFuelingRows =
    extras.embedded === true && day.length > 1
      ? Math.max(...day.map(d => (d.fueling ? fuelingRows(d.fueling).length : 0)))
      : 0
  const render =
    activity ??
    ((d: StravaActivityDetail) =>
      buildActivity(
        f,
        d,
        dayCardActivitiesExpanded(extras),
        ctx,
        extras.event != null,
        extras.embedded === true,
        extras.settings,
        sharedFuelingRows > 0,
      ))
  const cardStyles = [
    summaryRows == null ? null : `--tri-embedded-summary-rows:${summaryRows}`,
    sharedFuelingRows === 0 ? null : `--tri-embedded-fueling-rows:${sharedFuelingRows}`,
  ].filter(style => style != null)
  const card = f.el(
    'div',
    'tri-pop-card',
    undefined,
    cardStyles.length === 0 ? undefined : { style: cardStyles.join(';') },
  )
  const head = f.el('div', 'tri-pop-head')
  f.add(
    head,
    extras.dateHref
      ? f.el('a', 'tri-pop-date', prettyDate(dateIso), { href: extras.dateHref })
      : f.el('span', 'tri-pop-date', prettyDate(dateIso)),
  )
  if (day.length > 0) {
    f.add(
      head,
      f.el(
        'span',
        'tri-pop-loc',
        day[0].location ?? recentLocation(payload!) ?? extras.location ?? 'Toronto',
      ),
    )
  }
  if (extras.event) {
    const track = f.el('div', 'tri-pop-track')
    f.add(track, f.el('span', 'tri-pop-race', extras.event))
    f.add(head, track)
  }
  f.add(card, head)
  const dailyAnalytics = payload?.dailyAnalytics?.[dateIso]
  if (extras.analytics && dailyAnalytics) {
    f.add(card, buildDayAnalytics(f, dailyAnalytics))
    if (day.length > 0)
      f.add(
        card,
        f.el(
          'h2',
          'tri-ana-block-title tri-day-activities-title',
          triText(f.presentation.locale, 'activities'),
          { 'data-i18n': 'activities' },
        ),
      )
  }
  if (!payload) {
    f.add(card, f.el('div', 'tri-pop-rest', '·'))
  } else if (day.length === 0) {
    if (extras.sport) {
      const rest = f.el('div', 'tri-pop-rest')
      f.add(rest, f.el('span', 'tri-pop-rest-label', `no ${extras.sport}`))
      f.add(card, rest)
    } else {
      f.add(card, buildRestStatus(f))
    }
  } else if (extras.embedded === true && day.length > 1) {
    f.add(
      card,
      buildDayTrackPager(
        f,
        day.map(d => d.sport),
      ),
    )
    const track = f.el('div', 'tri-day-track', undefined, {
      style: `--tri-day-count:${day.length}`,
    })
    for (const d of day) f.add(track, render(d, sharedFuelingRows > 0))
    f.add(card, track)
  } else {
    for (const d of day) f.add(card, render(d, sharedFuelingRows > 0))
  }
  if (!extras.sport && !extras.activityId && !extras.analytics) {
    const dh = payload?.health[dateIso]
    if (dh) {
      const rec = buildRecovery(f, dh)
      if (rec) f.add(card, rec)
    }
    if (dailyAnalytics) {
      const sleep = buildDaySleepSection(f, dailyAnalytics)
      if (sleep) f.add(card, sleep)
    }
  }
  return card
}
