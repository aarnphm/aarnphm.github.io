import type { Element, ElementContent } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h, s } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { CriticalPowerEstimate } from '../plugins/stores/critical-power'
import type { GarminRunWalkSegment, GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDayDetail, OuraDaily } from '../plugins/stores/oura'
import type {
  ActivityAnalyses,
  ActivityAnalysisRange,
  ActivityHeartRateTracePoint,
  ActivityThermalSource,
  StravaActivityDetail,
  SwimActivityInterval,
  SwimTrendPoint,
} from '../plugins/stores/strava'
import type { PublicSurfaceCurrentEstimate } from './surface-current'
import type { TriathlonDayAnalytics } from './triathlon-day-analytics'
import { triathlonDayCard } from '../components/triathlon-day-card'
import { isActivityDetail } from '../components/triathlon/activity/data'
import { metricSpecs } from '../components/triathlon/activity/render'
import {
  workspaceTraces,
  workspaceTracePaths,
  workspaceValueAt,
  workspaceTimeline,
  workspaceLocationAt,
} from '../components/triathlon/activity/workspace-data'
import { createTriathlonFormatter } from '../components/triathlon/runtime/formatter'
import { buildAnalytics } from '../plugins/stores/analytics'
import { calculateActivityExerciseLoad, emptyHealth } from '../plugins/stores/strava'
import { parseTrackingBlock } from '../plugins/stores/tracking'
import { emptyWahooMetrics } from '../plugins/stores/wahoo'
import { applyOuraSleepRows } from '../scripts/sync-oura'
import { buildCyclingIntensityTrace } from './cycling-intensity'
import { buildCyclingPowerTrace } from './cycling-power'
import { buildCyclingTorqueTrace, cyclingTorqueSamples } from './cycling-torque'
import { health as garminHealthFixture } from './fixtures/garmin-health'
import { applyHeartRatePhysiology, estimateHeartRatePhysiology } from './heart-rate-physiology'
import { decodePowerCurves, encodePowerCurves, powerCurvePointAt } from './power-curve'
import { resolveSleepMetrics } from './sleep-metrics'
import { estimateSwimPhysiology } from './swim-physiology'
import { buildSwimPowerEstimate } from './swim-power'
import {
  activityCompareColor,
  activityComparisonDisplayValueAtDistance as displayValueAtDistance,
  activityComparisonEligible,
  activityComparisonFractionForKey,
  activityComparisonMetricAtDistance as metricAtDistance,
  activityComparisonMetricsForSport,
  activityCyclingPowerPoints,
  activityGearRatioDistribution,
  activityPowerDistributionPercentages,
  activitySelectionSummary,
  activityStatRows,
  activityTableRows,
  activityTrainingEffectLabel,
  activityZonePercentages,
  axisFrame,
  buildActivity,
  buildActivityAnalyzeButton,
  buildActivityIcon,
  buildActivityComparison,
  buildBestEfforts,
  buildCyclingPowerChart,
  buildCrankTorqueChart,
  buildIntensityFactorChart,
  buildDayAnalytics,
  buildDayCard,
  buildElevation,
  buildEnvironmentAnalysis,
  buildHeartRateTrace,
  buildIcon,
  buildPowerBalanceChart,
  buildPowerCurve,
  buildSwimPowerCurve,
  buildPowerHist,
  buildSaunaHeatTrainingLoad,
  buildShiftingChart,
  buildSleepRespirationChart,
  buildStaminaChart,
  buildTorqueEffectivenessChart,
  buildTrainingEffectDetails,
  buildTimelineDayCard,
  cyclingDynamicsIndexAtDistance,
  dominantTrainingEffectGroup,
  buildRoute,
  buildRunGroundContactTrace,
  buildRunStrideTrace,
  buildRunVerticalOscillationTrace,
  buildSwimTrends,
  buildTrace,
  buildWorkoutAnalysis,
  climbGradeBand,
  clock,
  dlabel,
  environmentElapsedClock,
  environmentChartReadout,
  environmentChartSeries,
  formatAltitude,
  fuelingRows,
  formatGroundContactTime,
  formatStrideLength,
  formatVerticalOscillation,
  formatTrainingEffectLabel,
  formatTrainingEffectNote,
  gearShiftAtFraction,
  interpolatePositiveMetricSeries,
  nearestPowerCurvePoint,
  nearestPowerCurveValue,
  moreStatRows,
  normalizePowerCurvePoints,
  parseExcludedActivityIds,
  powerCurveDurationTicks,
  powerCurveAxisTicks,
  powerCurveFraction,
  powerCurveHoverAt,
  powerCurveValueText,
  powerCurveWeight,
  powerViewActivity,
  riderPositionAtDistance,
  runWalkSegmentAt,
  runStrideLengthM,
  runStrideLengthValue,
  swimActivityBlocks,
  swimActivityPointLabel,
  swimTrendHoverAt,
  zoneDuo,
  type SwimTrendChartPoint,
  type DetailCtx,
  type TriNodeFactory,
} from './triathlon-card'
import { buildTriathlonDailyAnalytics } from './triathlon-day-analytics'
import {
  criticalPowerEvidenceText,
  criticalPowerSummaryText,
  glossFor,
  swimActivityDistanceText,
  swimActivityDisplayValue,
  swimActivityHeaderValue,
  swimActivityPointText,
  swimActivityValueText,
  triText,
  trendUnavailableText,
  vo2SourceText,
} from './triathlon-i18n'
import {
  DEFAULT_TRIATHLON_PRESENTATION,
  type TriathlonPresentation,
} from './triathlon-presentation'
import { TRIATHLON_TRACE_DISPLAY_SETTINGS } from './triathlon-trace-settings'
import { buildWalkPowerEstimate } from './walk-power'

const METRIC_TRIATHLON_PRESENTATION: TriathlonPresentation = Object.freeze({
  ...DEFAULT_TRIATHLON_PRESENTATION,
  distance: 'metric',
})

const factory: TriNodeFactory<Element> = {
  presentation: METRIC_TRIATHLON_PRESENTATION,
  el: (tag, cls, text, attrs) =>
    h(tag, { ...(cls ? { class: cls } : {}), ...attrs }, text === undefined ? [] : [text]),
  math: (cls, text) =>
    h(
      'span',
      { class: cls },
      text
        .split(/\$([^$]+)\$/)
        .flatMap((part, index) =>
          part ? [index % 2 === 0 ? part : h('span', { class: 'tri-math' }, part)] : [],
        ),
    ),
  svg: (tag, attrs) => s(tag, attrs),
  add: (parent, ...children) => parent.children.push(...children),
}

const decodedPowerCurves = (svg: Element) =>
  decodePowerCurves(String(svg.properties.dataPowerCurves)).map(curve =>
    Array.from({ length: curve.length }, (_, index) => powerCurvePointAt(curve, index)),
  )

const presentation = (overrides: Partial<TriathlonPresentation> = {}): TriathlonPresentation => ({
  ...METRIC_TRIATHLON_PRESENTATION,
  ...overrides,
})

const factoryFor = (value: TriathlonPresentation): TriNodeFactory<Element> => ({
  ...factory,
  presentation: value,
})

const english = createTriathlonFormatter(METRIC_TRIATHLON_PRESENTATION)
const frenchPresentation = presentation({ locale: 'fr' })
const french = createTriathlonFormatter(frenchPresentation)
const imperialPresentation = presentation({ distance: 'imperial' })
const excludeZeroPresentation = presentation({ powerSamples: 'exclude-zero' })

const activityComparisonDisplayValueAtDistance = (
  activity: StravaActivityDetail,
  metric: Parameters<typeof displayValueAtDistance>[2],
  distanceKm: number,
  value: TriathlonPresentation = METRIC_TRIATHLON_PRESENTATION,
): string => displayValueAtDistance(value, activity, metric, distanceKm)

const activityComparisonMetricAtDistance = (
  activity: StravaActivityDetail,
  metric: Parameters<typeof metricAtDistance>[2],
  distanceKm: number,
  value: TriathlonPresentation = METRIC_TRIATHLON_PRESENTATION,
): number | null => metricAtDistance(value, activity, metric, distanceKm)

test('renders manual fueling with its source', () => {
  assert.deepEqual(
    fuelingRows({
      caloriesConsumed: 0,
      carbsConsumedG: null,
      fluidMl: null,
      carbsRecommendedG: null,
      fluidRecommendedMl: null,
      sweatLossMl: null,
      sodiumLossMg: null,
      sourceDevice: null,
      source: 'manual',
    }),
    [
      ['consumed', '0 kcal'],
      ['source', 'manual'],
    ],
  )
  assert.deepEqual(
    fuelingRows({
      caloriesConsumed: 200,
      carbsConsumedG: null,
      fluidMl: null,
      carbsRecommendedG: null,
      fluidRecommendedMl: null,
      sweatLossMl: null,
      sodiumLossMg: null,
      sourceDevice: 'Edge 1050',
      source: 'garmin',
    }),
    [
      ['consumed', '200 kcal'],
      ['source', 'Garmin Edge 1050'],
    ],
  )
})

test('renders Wahoo sweat and sodium loss without treating them as bottle intake', () => {
  assert.deepEqual(
    fuelingRows({
      caloriesConsumed: null,
      carbsConsumedG: null,
      fluidMl: null,
      carbsRecommendedG: null,
      fluidRecommendedMl: null,
      sweatLossMl: 900,
      sodiumLossMg: 740,
      sourceDevice: 'ELEMNT BOLT',
      source: 'wahoo',
    }),
    [
      ['sweat', '900 ml'],
      ['sodium loss', '740 mg'],
      ['source', 'Wahoo ELEMNT BOLT'],
    ],
  )
  assert.deepEqual(
    fuelingRows({
      caloriesConsumed: 260,
      carbsConsumedG: null,
      fluidMl: 1_200,
      carbsRecommendedG: null,
      fluidRecommendedMl: null,
      sweatLossMl: 900,
      sodiumLossMg: 740,
      sourceDevice: 'Edge 1050',
      source: 'garmin+wahoo',
    }),
    [
      ['consumed', '260 kcal'],
      ['fluid', '1.2 L'],
      ['sweat', '900 ml'],
      ['sodium loss', '740 mg'],
      ['source', 'Garmin + Wahoo'],
    ],
  )
})

test('carries rounded pace seconds into the next minute', () => {
  assert.equal(clock(539.6), '9:00')
})

test('formats arbitrary power durations without decimal-hour noise', () => {
  assert.equal(dlabel(90), '1m30s')
  assert.equal(dlabel(5_097), '1h25m')
})

test('localizes swim block readouts and accessible values', () => {
  const point = { elapsed: '2:11', cumulativeDistanceM: 100, windowStartDistanceM: 0 }
  assert.equal(swimActivityPointText('fr', point), '0–100 m · 2:11 écoulé')
  assert.equal(swimActivityDisplayValue('fr', 'cadence', 13.8, '0:14'), '13,8 coups/longueur')
  assert.equal(swimActivityHeaderValue('fr', 'cadence', 13.8, '0:14'), '13,8')
  assert.equal(swimActivityDisplayValue('fr', 'swolf', 46.3, '0:46'), '46 SWOLF')
  assert.equal(swimActivityHeaderValue('fr', 'swolf', 46.3, '0:46'), '46')
  assert.equal(swimActivityDistanceText('fr', 1_000), '1 000 m')
  assert.equal(
    swimActivityValueText('fr', 'pace', point, 107, '1:47'),
    'bloc de 100 mètres, de 0 à 100 mètres, temps écoulé 2:11, allure de nage 1:47 par 100 mètres',
  )
  assert.equal(swimActivityPointText('en', point), '0–100 m · 2:11 elapsed')
  assert.equal(swimActivityHeaderValue('en', 'pace', 107, '1:47'), '1:47')
  assert.equal(swimActivityDistanceText('en', 1_000), '1,000 m')
  assert.equal(
    swimActivityValueText('en', 'cadence', point, 13.8, '0:14'),
    '100 metre block from 0 to 100 metres, 2:11 elapsed, swim cadence 13.8 strokes per length',
  )
  assert.equal(
    swimActivityValueText('en', 'swolf', point, 46.3, '0:46'),
    '100 metre block from 0 to 100 metres, 2:11 elapsed, SWOLF score 46',
  )
})

test('localizes dynamic analytics explanations', () => {
  const bike: NonNullable<Parameters<typeof vo2SourceText>[2]> = {
    ftpW: 230,
    ftpSource: 'athlete',
    mapW: 307,
    weightKg: 88.9,
    labBaseline: { date: '2026-06-25', vo2max: 47.8, ftpW: 230, weightKg: 88.9 },
  }
  assert.equal(vo2SourceText('fr', 'garmin', null), "Garmin Connect ou d'une saisie manuelle.")
  assert.equal(vo2SourceText('fr', 'apple', null), "Cette mesure vient de l'Apple Watch.")
  assert.equal(
    vo2SourceText('fr', 'run', null),
    'Cette estimation utilise la vitesse de course et la fréquence cardiaque.',
  )
  assert.equal(
    vo2SourceText('fr', 'hrratio', null),
    'Cette estimation utilise les fréquences cardiaques maximale et au repos.',
  )
  assert.equal(
    vo2SourceText('fr', 'lab', null),
    "Cette valeur vient d'un test d'effort progressif.",
  )
  assert.equal(
    vo2SourceText('fr', 'none', null),
    'Il manque les données de puissance ou de fréquence cardiaque.',
  )
  assert.equal(
    vo2SourceText('fr', 'bike', bike),
    'FTP 230 W (athlète). La puissance aérobie maximale estimée est de 307 W. Le poids est de 88,9 kg. La référence du laboratoire est une VO₂max de 47,8, une FTP de 230 W et un poids de 88,9 kg.',
  )
  assert.equal(trendUnavailableText('fr', 0, null), "Aucun effort n'a été enregistré.")
  assert.equal(trendUnavailableText('fr', 2, 0), "Le dernier effort date d'aujourd'hui.")
  assert.equal(trendUnavailableText('fr', 2, 1), 'Le dernier effort remonte à 1 jour.')
  assert.equal(trendUnavailableText('fr', 2, 48), 'Le dernier effort remonte à 48 jours.')
  assert.equal(trendUnavailableText('fr', null, null), 'Données insuffisantes.')
  assert.equal(triText('fr', 'reset'), 'réinit.')
  assert.equal(triText('fr', 'no data available'), 'aucune donnée disponible')
  assert.match(
    triText('fr', 'radar stroke rate swim definition'),
    /^La fréquence de nage est la moyenne des fréquences/,
  )
  assert.equal(
    vo2SourceText('en', 'bike', bike),
    'FTP 230 W (athlete). Estimated maximum aerobic power is 307 W. Body weight is 88.9 kg. The lab baseline is VO₂max 47.8, FTP 230 W, and body weight 88.9 kg.',
  )
  assert.equal(trendUnavailableText('en', 2, 1), 'The latest effort was 1 day ago.')
  assert.equal(trendUnavailableText('en', 2, 48), 'The latest effort was 48 days ago.')
  assert.match(
    triText('en', 'radar pace swim definition'),
    /Fewer seconds per 100 metres give a higher score\.$/,
  )
  assert.equal(criticalPowerSummaryText('en', criticalPower()), 'eCP 249 W · eW′ 10.3 kJ')
  assert.equal(criticalPowerSummaryText('fr', criticalPower()), 'eCP 249 W · eW′ 10,3 kJ')
  assert.equal(
    criticalPowerEvidenceText('en', criticalPower()),
    '2 independent efforts · provisional',
  )
  assert.equal(
    criticalPowerEvidenceText('fr', criticalPower()),
    '2 efforts indépendants · provisoire',
  )
})

test('localizes the triathlon page chrome', () => {
  assert.deepEqual(
    [
      'home',
      'swim',
      'bike',
      'run',
      'strength',
      'gear',
      'pace',
      'analytics',
      'map',
      'training',
      'calculator',
      'inspired by rauno',
    ].map(key => triText('fr', key)),
    [
      'accueil',
      'natation',
      'vélo',
      'course',
      'renforcement',
      'matériel',
      'allure',
      'analyses',
      'carte',
      'entraînement',
      'calculateur',
      'inspiré de rauno',
    ],
  )
})

test('localizes analytics dates, numbers, and lab chart labels', () => {
  assert.equal(french.shortDate('2026-05-20'), '20 mai')
  assert.equal(french.longDate('2026-06-25'), '25 juin 2026')
  assert.equal(french.month('2026-07-17'), 'juill.')
  assert.equal(french.monthYear('2026-07-01'), 'juillet 2026')
  assert.deepEqual(
    Array.from({ length: 7 }, (_, day) => french.weekdayNarrow(day)),
    ['D', 'L', 'M', 'M', 'J', 'V', 'S'],
  )
  assert.equal(french.number(27.4, 1, 1), '27,4')
  assert.deepEqual(
    ['wk', 'BMR', 'FFM', 'obese', 'Metabolic', 'Target', 'Avg', 'HR', 'Cool-Down'].map(key =>
      triText('fr', key),
    ),
    ['sem', 'MB', 'MM', 'obésité', 'Métabolique', 'Objectif', 'Moy', 'FC', 'Retour au calme'],
  )
  assert.equal(english.shortDate('2026-05-20'), 'May 20')
  assert.equal(english.longDate('2026-06-25'), 'Jun 25, 2026')
  assert.equal(english.number(27.4, 1, 1), '27.4')
  assert.equal(english.shortDate('invalid'), 'invalid')
})

test('resolves an exact power duration and its six-week reference value', () => {
  const curve = [
    { s: 1, w: 700 },
    { s: 2, w: 660 },
    { s: 3, w: 635 },
    { s: 5, w: 590 },
    { s: 60, w: 350 },
  ]
  const reference = [
    { s: 1, w: 1_060 },
    { s: 2, w: 1_034 },
    { s: 3, w: 1_016 },
    { s: 5, w: 983 },
    { s: 60, w: 396 },
  ]
  const fraction = powerCurveFraction(3, 1, 60)
  assert.deepEqual(powerCurveHoverAt(curve, reference, fraction), {
    index: 2,
    durationS: 3,
    watts: 635,
    referenceWatts: 1_016,
    xPct: fraction * 100,
  })
})

test('keeps a long-duration hover honest when its reference point is missing', () => {
  const curve = [
    { s: 60, w: 350 },
    { s: 2_340, w: 166 },
    { s: 3_600, w: 150 },
  ]
  const reference = [
    { s: 60, w: 400 },
    { s: 3_600, w: 170 },
  ]
  const fraction = powerCurveFraction(2_340, 60, 3_600)
  assert.deepEqual(powerCurveHoverAt(curve, reference, fraction), {
    index: 1,
    durationS: 2_340,
    watts: 166,
    referenceWatts: null,
    xPct: fraction * 100,
  })
})

test('shares one duration axis across dense, sparse, and shorter power curves', () => {
  const curves = [
    [
      { s: 1, w: 700 },
      { s: 2, w: 660 },
      { s: 3, w: 0 },
    ],
    [
      { s: 1, w: 800, activityId: 11, activityDate: '2026-08-01' },
      { s: 3, w: 750, activityId: 12, activityDate: '2026-08-02' },
      { s: 60, w: 350, activityId: 12, activityDate: '2026-08-02' },
    ],
    [{ s: 2, w: 900 }],
    [],
  ]
  const decoded = decodePowerCurves(encodePowerCurves(curves))
  assert.equal(decoded.length, curves.length)
  for (const curve of decoded) assert.equal(curve.durations, decoded[0].durations)
  assert.deepEqual(decoded[0].durations, [1, 2, 3, 60])
  assert.deepEqual(
    decoded.map(curve =>
      Array.from({ length: curve.length }, (_, index) => powerCurvePointAt(curve, index)),
    ),
    curves,
  )
  assert.deepEqual(nearestPowerCurvePoint(decoded[1], 2.7), curves[1][1])
  assert.equal(nearestPowerCurvePoint(decoded[2], 3), null)
  assert.deepEqual(powerCurveHoverAt(decoded[0], decoded[1], powerCurveFraction(2, 1, 3)), {
    index: 1,
    durationS: 2,
    watts: 660,
    referenceWatts: null,
    xPct: powerCurveFraction(2, 1, 3) * 100,
  })
  for (const encoded of [
    undefined,
    '',
    '{',
    JSON.stringify({ durations: 'd|1|3', series: [{ offset: 2, watts: [700, 600] }] }),
    JSON.stringify({ durations: 's|1,3,60', series: [{ offset: 0, watts: [700], indices: [3] }] }),
    JSON.stringify({
      durations: 's|1,3,60',
      series: [{ offset: 0, watts: [700, 600], indices: [1, 0] }],
    }),
    JSON.stringify({ durations: 's|1,1', series: [] }),
    JSON.stringify({ durations: 'd|1|2', series: [{ offset: 0, watts: [700, 'nope'] }] }),
    JSON.stringify({
      durations: 'd|1|2',
      series: [{ offset: 0, watts: [700, 600], activities: '11,2026-08-01,1' }],
    }),
  ])
    assert.deepEqual(decodePowerCurves(encoded), [])
})

test('selects the nearest serialized swim trend point and clamps the scrub range', () => {
  const points: SwimTrendChartPoint[] = [
    { elapsedS: 30, cumulativeDistanceM: 25, value: 112, xPct: 10, yPct: 80 },
    { elapsedS: 120, cumulativeDistanceM: 100, value: 108, xPct: 40, yPct: 40 },
    { elapsedS: 300, cumulativeDistanceM: 250, value: 100, xPct: 100, yPct: 0 },
  ]

  assert.deepEqual(swimTrendHoverAt(points, 0.51), {
    index: 1,
    elapsedS: 120,
    cumulativeDistanceM: 100,
    value: 108,
    xPct: 40,
    yPct: 40,
  })
  assert.equal(swimTrendHoverAt(points, -1)?.index, 0)
  assert.equal(swimTrendHoverAt(points, 2)?.index, 2)
  assert.equal(swimTrendHoverAt(points, Number.NaN)?.index, 0)
  assert.equal(swimTrendHoverAt([], 0.5), null)
})

test('aggregates measured lengths into a weighted 100 metre block without rest time', () => {
  const intervals: SwimActivityInterval[] = [
    {
      startElapsedS: 0,
      endElapsedS: 25,
      distanceM: 25,
      durationS: 25,
      cumulativeDistanceM: 25,
      paceSPer100m: 100,
      strokeCount: 10,
      strokeTimeS: 25,
      strokeRateSpm: 24,
      stroke: 'freestyle',
    },
    {
      startElapsedS: 40,
      endElapsedS: 66,
      distanceM: 25,
      durationS: 26,
      cumulativeDistanceM: 50,
      paceSPer100m: 104,
      strokeCount: 11,
      strokeTimeS: 25.4,
      strokeRateSpm: 26,
      stroke: 'freestyle',
    },
    {
      startElapsedS: 80,
      endElapsedS: 105,
      distanceM: 25,
      durationS: 25,
      cumulativeDistanceM: 75,
      paceSPer100m: 100,
      strokeCount: null,
      strokeTimeS: null,
      strokeRateSpm: null,
      stroke: 'kickboard',
    },
    {
      startElapsedS: 120,
      endElapsedS: 144,
      distanceM: 25,
      durationS: 24,
      cumulativeDistanceM: 100,
      paceSPer100m: 96,
      strokeCount: 12,
      strokeTimeS: 24,
      strokeRateSpm: 30,
      stroke: 'freestyle',
    },
  ]

  assert.deepEqual(swimActivityBlocks(intervals), [
    {
      startElapsedS: 0,
      endElapsedS: 144,
      distanceM: 100,
      durationS: 100,
      cumulativeDistanceM: 100,
      paceSPer100m: 100,
      strokeCount: 33,
      strokeTimeS: 74.4,
      strokeRateSpm: 26.6,
      stroke: null,
      strokesPerLength: 11,
      swolf: 36,
    },
  ])
})

test('normalizes raw swim samples after aggregation instead of trusting filtered length rates', () => {
  const starts = [0, 100, 120, 140]
  const durations = [100, 20, 20, 20]
  const intervals = durations.map(
    (durationS, index): SwimActivityInterval => ({
      startElapsedS: starts[index],
      endElapsedS: starts[index] + durationS,
      distanceM: 25,
      durationS,
      cumulativeDistanceM: (index + 1) * 25,
      paceSPer100m: index === 0 ? null : 80,
      strokeCount: index === 0 ? 50 : 5,
      strokeTimeS: 20,
      strokeRateSpm: index === 0 ? null : 15,
      stroke: 'freestyle',
    }),
  )

  assert.deepEqual(swimActivityBlocks(intervals), [
    {
      startElapsedS: 0,
      endElapsedS: 160,
      distanceM: 100,
      durationS: 160,
      cumulativeDistanceM: 100,
      paceSPer100m: 160,
      strokeCount: 65,
      strokeTimeS: 80,
      strokeRateSpm: 48.8,
      stroke: null,
      strokesPerLength: 16.3,
      swolf: 56.3,
    },
  ])
})

test('splits a length at the 100 metre boundary and keeps the final partial block', () => {
  const intervals: SwimActivityInterval[] = [
    {
      startElapsedS: 0,
      endElapsedS: 60,
      distanceM: 60,
      durationS: 60,
      cumulativeDistanceM: 60,
      paceSPer100m: 100,
      strokeCount: 24,
      strokeTimeS: 60,
      strokeRateSpm: 24,
      stroke: 'freestyle',
    },
    {
      startElapsedS: 100,
      endElapsedS: 190,
      distanceM: 60,
      durationS: 90,
      cumulativeDistanceM: 120,
      paceSPer100m: 150,
      strokeCount: 30,
      strokeTimeS: 60,
      strokeRateSpm: 30,
      stroke: 'breaststroke',
    },
  ]

  assert.deepEqual(swimActivityBlocks(intervals), [
    {
      startElapsedS: 0,
      endElapsedS: 160,
      distanceM: 100,
      durationS: 120,
      cumulativeDistanceM: 100,
      paceSPer100m: 120,
      strokeCount: 44,
      strokeTimeS: 100,
      strokeRateSpm: 26.4,
      stroke: null,
      strokesPerLength: 26.4,
      swolf: 98.4,
    },
    {
      startElapsedS: 160,
      endElapsedS: 190,
      distanceM: 20,
      durationS: 30,
      cumulativeDistanceM: 120,
      paceSPer100m: 150,
      strokeCount: 10,
      strokeTimeS: 20,
      strokeRateSpm: 30,
      stroke: null,
      strokesPerLength: 30,
      swolf: 120,
    },
  ])
  assert.equal(
    swimActivityPointLabel({ elapsedS: 190, cumulativeDistanceM: 120, windowStartDistanceM: 100 }),
    '100–120 m · 3:10 elapsed',
  )
})

test('keeps a normalized stroke block empty when a non-kickboard length lacks stroke samples', () => {
  const block = swimActivityBlocks([
    {
      startElapsedS: 0,
      endElapsedS: 100,
      distanceM: 100,
      durationS: 100,
      cumulativeDistanceM: 100,
      paceSPer100m: 100,
      strokeCount: null,
      strokeTimeS: null,
      strokeRateSpm: null,
      stroke: 'freestyle',
    },
  ])[0]

  assert.ok(block)
  assert.equal(block.strokeCount, null)
  assert.equal(block.strokeTimeS, null)
  assert.equal(block.strokeRateSpm, null)
  assert.equal(block.strokesPerLength, null)
  assert.equal(block.swolf, null)
})

const classNames = (element: Element): string[] => {
  const value = element.properties.className
  if (Array.isArray(value)) return value.map(String)
  return value == null ? [] : [String(value)]
}

const descendants = (root: Element, predicate: (element: Element) => boolean): Element[] => {
  const matches: Element[] = []
  const visit = (children: ElementContent[]): void => {
    for (const child of children) {
      if (child.type !== 'element') continue
      if (predicate(child)) matches.push(child)
      visit(child.children)
    }
  }
  if (predicate(root)) matches.push(root)
  visit(root.children)
  return matches
}

const byClass = (root: Element, cls: string): Element[] =>
  descendants(root, element => classNames(element).includes(cls))

const byTag = (root: Element, tag: string): Element[] =>
  descendants(root, element => element.tagName === tag)

const text = (root: Element): string => {
  let value = ''
  const visit = (children: ElementContent[]): void => {
    for (const child of children) {
      if (child.type === 'text') value += child.value
      else if (child.type === 'element') visit(child.children)
    }
  }
  visit(root.children)
  return value
}

const table = (root: Element, kind: string): Element => {
  const result = byClass(root, `tri-effort-table--${kind}`)[0]
  assert.ok(result)
  return result
}

const headerText = (root: Element): string[] => {
  const head = byTag(root, 'thead')[0]
  assert.ok(head)
  return byTag(head, 'th').map(text)
}

const bodyRows = (root: Element): string[][] => {
  const body = byTag(root, 'tbody')[0]
  assert.ok(body)
  return byTag(body, 'tr').map(row =>
    row.children.filter((child): child is Element => child.type === 'element').map(text),
  )
}

test('renders exact interactive targets on an x axis', () => {
  const graph = factory.svg('svg', {})
  const frame = axisFrame(
    factory,
    graph,
    [],
    100,
    [
      {
        label: '5m',
        pct: 50,
        tag: 'button',
        attrs: { type: 'button', 'data-power-seconds': '300', 'aria-pressed': 'true' },
      },
    ],
    false,
  )
  const tick = byClass(frame, 'tri-cax-xt')[0]

  assert.ok(tick)
  assert.equal(tick.tagName, 'button')
  assert.equal(tick.properties.type, 'button')
  assert.equal(tick.properties.dataPowerSeconds, '300')
  assert.equal(tick.properties.ariaPressed, 'true')
  assert.equal(tick.properties.style, 'left:50.00%')
})

const heartRateTracePoint = (
  distanceKm: number,
  elapsedS: number,
  heartRate: number | null,
  thermal: Partial<
    Pick<
      ActivityHeartRateTracePoint,
      | 'heatStrainIndex'
      | 'heatStrainSource'
      | 'coreTemperatureC'
      | 'coreTemperatureSource'
      | 'skinTemperatureC'
      | 'skinTemperatureSource'
    >
  > = {},
): ActivityHeartRateTracePoint => ({
  distanceKm,
  elapsedS,
  heartRate,
  heatStrainIndex: null,
  heatStrainSource: null,
  coreTemperatureC: null,
  coreTemperatureSource: null,
  skinTemperatureC: null,
  skinTemperatureSource: null,
  ...thermal,
})

const thermalSource = (
  value: number | null,
  source: ActivityThermalSource,
): ActivityThermalSource | null => (value == null ? null : source)

const detail = (overrides: Partial<StravaActivityDetail> = {}): StravaActivityDetail => ({
  id: 101,
  sport: 'bike',
  name: 'Threshold ride',
  date: '2026-07-09',
  start: '2026-07-09T12:00:00Z',
  distanceKm: 30,
  movingTimeS: 4_800,
  elapsedTimeS: 4_800,
  maxSpeedKph: null,
  elevationM: 100,
  avgHr: 148,
  maxHr: 171,
  avgWatts: 188,
  npWatts: 205,
  maxWatts: 565,
  kilojoules: 900,
  deviceWatts: true,
  avgCadence: 88,
  sufferScore: null,
  calories: 960,
  deviceTemperatureC: 24,
  ambientTemperatureC: 22,
  windKph: null,
  windDir: null,
  windDirDeg: null,
  windGustKph: null,
  averageRelativeHumidityPct: null,
  relativeHumidityProvenance: null,
  location: 'Toronto',
  fueling: null,
  strength: null,
  sauna: null,
  garmin: null,
  computer: null,
  device: null,
  staminaTrace: null,
  performanceConditionTrace: null,
  calculatedIntensityFactor: null,
  calculatedExerciseLoad: null,
  anaerobicPowerEstimate: null,
  calculatedTrainingEffect: null,
  gearShifts: [],
  cyclingDynamics: null,
  runWalk: null,
  route: [
    {
      x: 0,
      y: 0,
      d: 0,
      alt: 75,
      w: 160,
      hr: 130,
      cad: 82,
      stamina: null,
      potentialStamina: null,
      resp: 20,
      tempC: 22,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
      lat: 43.6,
      lng: -79.4,
      elapsedS: 0,
      speedKph: 22,
    },
    {
      x: 0.34,
      y: 0.4,
      d: 10,
      alt: 89,
      w: 200,
      hr: 145,
      cad: 86,
      stamina: null,
      potentialStamina: null,
      resp: 24,
      tempC: 23,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
      lat: 43.7,
      lng: -79.3,
      elapsedS: 1_600,
      speedKph: 24,
    },
    {
      x: 0.67,
      y: 0.8,
      d: 20,
      alt: 103,
      w: 215,
      hr: 153,
      cad: 90,
      stamina: null,
      potentialStamina: null,
      resp: 28,
      tempC: 24,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
      lat: 43.8,
      lng: -79.2,
      elapsedS: 3_200,
      speedKph: 25,
    },
    {
      x: 1,
      y: 1,
      d: 30,
      alt: 110,
      w: 175,
      hr: 149,
      cad: 84,
      stamina: null,
      potentialStamina: null,
      resp: 32,
      tempC: 25,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
      lat: 43.9,
      lng: -79.1,
      elapsedS: 4_800,
      speedKph: 23,
    },
  ],
  heartRateTrace: [],
  mapRoute: [],
  analysisRanges: [],
  runSplitsMetric: [],
  runSplitsStandard: [],
  runPaceZones: null,
  minAlt: 75,
  maxAlt: 110,
  descentM: 20,
  hrZones: null,
  powerZones: null,
  powerHist: null,
  powerWithoutZeros: null,
  powerCurve: null,
  activityCriticalPower: null,
  bestEfforts: {
    weightKg: 87.55,
    weightDate: '2026-07-09',
    distance: [
      {
        label: '10K',
        targetDistanceM: 10_000,
        elapsedTimeS: 1_471,
        averageSpeedKph: 24.5,
        averageHeartRate: 151,
        elevationDeltaM: -30,
      },
    ],
    power: [
      {
        durationS: 5,
        averageWatts: 565,
        wattsPerKg: 6.45,
        averageHeartRate: 150,
        elevationDeltaM: 4,
      },
    ],
    climbs: [
      {
        source: 'garmin-climbpro',
        name: 'Snake Road',
        durationS: 480,
        distanceM: 2_500,
        elevationGainM: 120,
        averageGradePct: 4.8,
        averageSpeedKph: 18.8,
        averageHeartRate: 155,
        averageWatts: 240,
        wattsPerKg: 2.74,
        vamMPerHour: 900,
      },
    ],
  },
  strokeCount: null,
  strokeRateSpm: null,
  swimPaceSPer100m: null,
  swimPaceSource: null,
  swimDurationS: null,
  swimIntervals: [],
  swimLocation: null,
  waterTemperatureC: null,
  analyses: {
    native: { myWindsock: null, pelotan: null },
    derived: { environment: null, uvScore: null, apparentWind: null },
  },
  ...overrides,
})

const environmentAnalyses = (): ActivityAnalyses => ({
  native: {
    pelotan: {
      source: 'provider-native',
      provider: 'pelotan',
      transport: 'strava-description',
      schemaVersion: 1,
      activityId: 101,
      retrievedAt: Date.parse('2026-07-09T18:00:00Z'),
      score: 83,
      rawBand: 'High',
      severity: 'high',
      averageUvIndex: 2.3,
      averageTemperatureC: 20,
      averageCloudCoverPct: 42,
    },
    myWindsock: {
      source: 'provider-native',
      provider: 'mywindsock',
      transport: 'strava-description',
      schemaVersion: 1,
      activityId: 101,
      retrievedAt: Date.parse('2026-07-09T18:00:00Z'),
      weatherImpactPct: 0.6,
      cdaM2: 0.324,
      feelsLikeElevationM: 247,
      headwindPct: 56,
      headwindMinKph: 8,
      headwindMaxKph: 24,
      longestHeadwindS: 12_761,
      airSpeedKph: 27,
      averageTemperatureC: 20,
      precipitationProbabilityPct: 20,
      precipitationRateMmPerHour: 0.2,
    },
  },
  derived: {
    environment: {
      source: 'garden-estimate',
      formulaId: 'garden-environment-v1',
      formulaVersion: 1,
      inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1',
      normalizationVersion: 1,
      computedAt: Date.parse('2026-07-09T18:05:00Z'),
      inputAsOf: Date.parse('2026-07-09T17:00:00Z'),
      temporalSamplingModel: 'weatherkit-hourly-piecewise-constant',
      spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint',
      summary: {
        averageUvIndex: 2.4,
        peakUvIndex: 6.4,
        uviHours: 4,
        ambientSed: 3.6,
        averageAmbientTemperatureC: 20,
        averageCloudCoverPct: 42,
        daylightCoveragePct: 100,
        weatherCoveragePct: 100,
        coveredDurationS: 4_800,
        elapsedDurationS: 4_800,
      },
      doseClocks: { elapsedSed: 3.6, movingTelemetrySed: 3.4 },
      coverage: {
        weatherPct: 100,
        uvPct: 100,
        temperaturePct: 100,
        cloudPct: 100,
        daylightPct: 100,
      },
      samples: [
        {
          elapsedS: 0,
          distanceKm: 0,
          uvIndex: 6.4,
          cumulativeSed: 0,
          cumulativeMovingTelemetrySed: 0,
          ambientTemperatureC: 22,
          cloudCoverPct: 91,
          headwindKph: null,
          crosswindKph: null,
          apparentAirSpeedKph: null,
          yawDeg: null,
        },
        {
          elapsedS: 1_600,
          distanceKm: 10,
          uvIndex: 4.2,
          cumulativeSed: 2.1,
          cumulativeMovingTelemetrySed: 1.9,
          ambientTemperatureC: 21,
          cloudCoverPct: 70,
          headwindKph: 8,
          crosswindKph: -3,
          apparentAirSpeedKph: 31,
          yawDeg: -5,
        },
        {
          elapsedS: 3_200,
          distanceKm: 20,
          uvIndex: 2.1,
          cumulativeSed: 3.1,
          cumulativeMovingTelemetrySed: 2.9,
          ambientTemperatureC: 19,
          cloudCoverPct: 45,
          headwindKph: -4,
          crosswindKph: 2,
          apparentAirSpeedKph: 21,
          yawDeg: 4,
        },
        {
          elapsedS: 4_800,
          distanceKm: 30,
          uvIndex: 0,
          cumulativeSed: 3.6,
          cumulativeMovingTelemetrySed: 3.4,
          ambientTemperatureC: 18,
          cloudCoverPct: 0,
          headwindKph: 5,
          crosswindKph: 1,
          apparentAirSpeedKph: 28,
          yawDeg: 2,
        },
      ],
      attribution: {
        serviceName: 'Apple Weather',
        logoLightUrl: 'https://weatherkit.apple.com/assets/light.svg',
        logoDarkUrl: 'https://weatherkit.apple.com/assets/dark.svg',
        legalPageUrl: 'https://weatherkit.apple.com/legal-attribution.html',
      },
    },
    uvScore: {
      source: 'garden-estimate',
      formulaId: 'garden-uv-score-v1',
      formulaVersion: 1,
      inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1',
      normalizationVersion: 1,
      computedAt: Date.parse('2026-07-09T18:05:00Z'),
      inputAsOf: Date.parse('2026-07-09T17:00:00Z'),
      temporalSamplingModel: 'weatherkit-hourly-piecewise-constant',
      spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint',
      score: 30,
      severity: 'low',
      doseClock: 'elapsed',
      doseSed: 3.6,
      coefficientSed: 10,
      calibrationVersion: 1,
    },
    apparentWind: {
      source: 'garden-estimate',
      formulaId: 'garden-apparent-wind-v3',
      formulaVersion: 1,
      inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1',
      normalizationVersion: 1,
      computedAt: Date.parse('2026-07-09T18:05:00Z'),
      inputAsOf: Date.parse('2026-07-09T17:00:00Z'),
      temporalSamplingModel: 'weatherkit-hourly-piecewise-constant',
      spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint',
      summary: {
        headwindSharePct: 56,
        headwindTimeS: 2_000,
        tailwindTimeS: 1_000,
        longestHeadwindS: 800,
        averageHeadwindWhileIntoKph: 3,
        averageCrosswindMagnitudeKph: 1,
        maximumHeadwindKph: 10,
        maximumCrosswindKph: 5,
        averageGroundSpeedKph: 24,
        averageApparentAirSpeedKph: 27,
        apparentAirRatio: 1.125,
        averageYawDeg: -2,
        coveragePct: 75,
      },
      coverage: { windPct: 75 },
    },
  },
})

test('renders one provider-first environment table with explicit Garden estimate provenance', () => {
  const activity = detail({ analyses: environmentAnalyses() })
  const rendered = buildActivity(factory, activity, true)
  const environment = byClass(rendered, 'tri-environment')[0]

  assert.ok(environment)
  assert.deepEqual(activityTableRows(METRIC_TRIATHLON_PRESENTATION, activity).slice(-2), [
    ['weather impact', '0.6%'],
    ['uv load™', '83 · High'],
  ])
  assert.equal(byClass(environment, 'tri-environment-table').length, 1)
  assert.equal(byClass(environment, 'tri-environment-table-group').length, 2)
  const estimates = byClass(environment, 'tri-environment-estimate')
  assert.ok(estimates.length > 0)
  assert.ok(
    estimates.every(
      estimate =>
        estimate.properties.dataAnalysisSource === 'garden-estimate' &&
        estimate.properties.dataAnalysisFormulaVersion === '1' &&
        estimate.properties.dataGloss === '' &&
        estimate.properties.dataGlossDef === 'estimate' &&
        estimate.properties.ariaLabel === `${text(estimate)}, estimate` &&
        estimate.properties.tabIndex === 0,
    ),
  )
  const cdaLabel = descendants(
    environment,
    element =>
      classNames(element).includes('tri-environment-row-label') &&
      element.properties.dataI18n === 'CdA',
  )[0]
  assert.ok(cdaLabel)
  assert.match(
    String(cdaLabel.properties.dataGlossDef),
    /drag coefficient multiplied by frontal area/,
  )
  const cdaRow = byTag(environment, 'tr').find(row =>
    byClass(row, 'tri-environment-row-label').some(label => label.properties.dataI18n === 'CdA'),
  )
  assert.ok(cdaRow)
  assert.equal(text(cdaRow), 'CdA0.324')
  const ratioLabel = byClass(environment, 'tri-environment-row-math')[0]
  assert.ok(ratioLabel)
  assert.equal(text(ratioLabel), 'v_{\\mathrm{air}} / v_{\\mathrm{ground}}')
  assert.equal(byClass(ratioLabel, 'tri-math').length, 1)
  const ratioRow = byTag(environment, 'tr').find(row =>
    byClass(row, 'tri-environment-row-math').some(label => label === ratioLabel),
  )
  assert.ok(ratioRow)
  assert.equal(text(ratioRow), 'v_{\\mathrm{air}} / v_{\\mathrm{ground}}1.125')
  assert.doesNotMatch(text(environment), /2\.4/)
  assert.match(text(environment), /2\.3/)
  assert.deepEqual(byClass(environment, 'tri-environment-tab-full').map(text), [
    'cumulative',
    'UV index',
    'temperature',
    'cloud cover',
    'wind',
  ])
  const tabs = byClass(environment, 'tri-environment-tab')
  assert.deepEqual(byClass(environment, 'tri-environment-tab-short').map(text), [
    'cum.',
    'UVI',
    'temp.',
    'cloud',
    'wind',
  ])
  for (const tab of tabs) {
    assert.equal(tab.properties.ariaLabel, text(byClass(tab, 'tri-environment-tab-full')[0]))
    assert.equal(tab.properties.title, tab.properties.ariaLabel)
  }
  const panels = byClass(environment, 'tri-environment-panel')
  assert.deepEqual(
    tabs.map(tab => [tab.properties.role, tab.properties.tabIndex, tab.properties.ariaSelected]),
    [
      ['tab', 0, 'true'],
      ['tab', -1, 'false'],
      ['tab', -1, 'false'],
      ['tab', -1, 'false'],
      ['tab', -1, 'false'],
    ],
  )
  assert.deepEqual(
    panels.map(panel => [panel.properties.role, panel.properties.hidden]),
    [
      ['tabpanel', undefined],
      ['tabpanel', true],
      ['tabpanel', true],
      ['tabpanel', true],
      ['tabpanel', true],
    ],
  )
  assert.equal(byClass(environment, 'tri-environment-series').length, 6)
  assert.deepEqual(byClass(panels[0], 'tri-cax-xt').map(text), ['0:00', '40:00', '1:20:00'])
  assert.deepEqual(
    byClass(panels[0], 'tri-cax-yt')
      .filter(tick => tick.properties.hidden === undefined)
      .map(text),
    ['0', '50', '100'],
  )
  assert.deepEqual(
    new Set(
      descendants(
        environment,
        element => element.properties.dataAnalysisSource === 'provider-native',
      ).map(element => element.properties.dataAnalysisProvider),
    ),
    new Set(['pelotan', 'mywindsock']),
  )
  const providerLogos = byClass(environment, 'tri-environment-provider-logo')
  assert.equal(providerLogos.length, 2)
  assert.deepEqual(
    providerLogos.map(link => [link.properties.title, link.properties.href]),
    [
      ['Powered by myWindsock', 'https://mywindsock.com/activity/101/'],
      ['Pelotan UV Load™', 'https://pelotan.cc/pages/uv-load'],
    ],
  )
  const attribution = byClass(environment, 'tri-environment-attribution')[0]
  assert.ok(attribution)
  assert.deepEqual(
    byTag(attribution, 'img').map(image => image.properties.src),
    [
      '/static/triathlon/mywindsock.svg',
      '/static/triathlon/pelotan.png',
      '/static/triathlon/apple-weather-light.png',
    ],
  )
  assert.deepEqual(
    byTag(attribution, 'source').map(source => [source.properties.media, source.properties.srcSet]),
    [['(prefers-color-scheme: dark)', '/static/triathlon/apple-weather-dark.png']],
  )
  assert.ok(
    byTag(attribution, 'a').some(
      link => link.properties.href === 'https://weatherkit.apple.com/legal-attribution.html',
    ),
  )
  assert.doesNotMatch(text(environment), /Garden estimate from Apple WeatherKit/)
  assert.ok(
    byTag(environment, 'a').some(
      link =>
        link.properties.title ===
        'Garden estimate from Apple WeatherKit. Apple Weather data modified by Garden calculations.',
    ),
  )
  assert.doesNotMatch(text(environment), /latitude|longitude|routeFingerprint/)
})

test('renders elapsed lap highlights inside every environment plot', () => {
  const activity = detail({ analyses: environmentAnalyses() })
  const rendered = buildEnvironmentAnalysis(factory, activity)
  assert.ok(rendered)
  const graphs = byClass(rendered, 'tri-environment-plot')
  assert.equal(graphs.length, 5)
  for (const graph of graphs) {
    assert.equal(graph.properties.dataDomainStartElapsedS, 0)
    assert.equal(
      graph.properties.dataDomainEndElapsedS,
      activity.analyses.derived.environment?.summary.elapsedDurationS,
    )
    assert.equal(graph.properties.dataDomainStartX, 2)
    assert.equal(graph.properties.dataDomainEndX, 98)
    const selections = byClass(graph, 'tri-analysis-selection')
    assert.equal(selections.length, 1)
    assert.equal(selections[0].properties.width, '0.00')
    assert.equal(selections[0].properties.y, 3)
    assert.equal(selections[0].properties.height, 24)
  }
})

test('keeps unavailable wind and aero rows for open-water swims with UV evidence', () => {
  const analyses = environmentAnalyses()
  analyses.native.myWindsock = null
  analyses.derived.apparentWind = null
  const environment = analyses.derived.environment
  assert.ok(environment)
  environment.summary.averageWindSpeedKph = 0
  environment.samples = environment.samples.map(sample => ({
    ...sample,
    windSpeedKph: 0,
    headwindKph: null,
    crosswindKph: null,
    apparentAirSpeedKph: null,
    yawDeg: null,
  }))
  const activity = detail({ sport: 'swim', swimLocation: 'openWater', analyses })
  const rendered = buildEnvironmentAnalysis(factory, activity)
  assert.ok(rendered)
  assert.deepEqual(byClass(rendered, 'tri-environment-table-group').map(text), [
    'UV exposure',
    'wind and aero',
  ])
  const rows = byTag(rendered, 'tr')
  for (const label of [
    'Weather Impact',
    'air speed',
    'surface current speed',
    'surface current direction',
    'surface current coverage',
  ]) {
    const row = rows.find(row =>
      byClass(row, 'tri-environment-row-label').some(cell => text(cell) === label),
    )
    assert.ok(row, label)
    const value = byTag(row, 'td')[0]
    assert.equal(text(value), '—')
    assert.equal(value.properties.ariaLabel, 'unavailable')
    assert.equal(value.properties.dataAnalysisSource, undefined)
  }
  const ambientWind = rows.find(row =>
    byClass(row, 'tri-environment-row-label').some(label => text(label) === 'ambient wind speed'),
  )
  assert.ok(ambientWind)
  const ambientValue = byTag(ambientWind, 'td')[0]
  assert.equal(text(ambientValue), '0.0 km/h')
  assert.equal(ambientValue.properties.dataAnalysisSource, 'garden-estimate')
  assert.equal(ambientValue.properties.dataAnalysisFormula, environment.formulaId)

  const otherSwimLocations: StravaActivityDetail['swimLocation'][] = ['pool', null]
  for (const swimLocation of otherSwimLocations) {
    const other = buildEnvironmentAnalysis(factory, { ...activity, swimLocation })
    assert.ok(other)
    assert.equal(
      byClass(other, 'tri-environment-row-label').some(
        label => text(label) === 'surface current speed',
      ),
      false,
    )
  }
  const pool = buildEnvironmentAnalysis(factory, { ...activity, swimLocation: 'pool' })
  assert.ok(pool)
  const poolWind = byTag(pool, 'tr').find(row =>
    byClass(row, 'tri-environment-row-label').some(label => text(label) === 'ambient wind speed'),
  )
  assert.ok(poolWind)
  assert.equal(text(byTag(poolWind, 'td')[0]), '—')
  const poolWindPanel = byClass(pool, 'tri-environment-panel').at(-1)
  assert.ok(poolWindPanel)
  assert.equal(byClass(poolWindPanel, 'tri-environment-line').length, 0)
  assert.equal(byClass(poolWindPanel, 'tri-environment-empty').length, 1)
  assert.doesNotMatch(text(byClass(pool, 'tri-environment-readout')[0]), /ambient wind speed/)
})

const modeledSurfaceCurrent = (): PublicSurfaceCurrentEstimate => ({
  source: 'noaa-loofs',
  sourceKind: 'modeled',
  formulaId: 'garden-surface-current-v1',
  formulaVersion: 1,
  activityId: 101,
  start: '2026-07-09T12:00:00Z',
  end: '2026-07-09T13:20:00Z',
  computedAt: Date.parse('2026-07-09T18:05:00Z'),
  layer: 0,
  spatialSamplingModel: 'containing-element',
  temporalSamplingModel: 'hourly-linear-vector',
  summary: {
    averageSpeedMps: 0.14,
    averageDirectionDeg: 237,
    coveragePct: 75,
    coveredDurationS: 3_600,
    elapsedDurationS: 4_800,
  },
  samples: [],
})

test('renders source-backed open-water surface currents as modeled vectors with coverage', () => {
  const analyses = environmentAnalyses()
  const environment = analyses.derived.environment
  assert.ok(environment)
  environment.surfaceCurrent = modeledSurfaceCurrent()
  const activity = detail({ sport: 'swim', swimLocation: 'openWater', analyses })
  const currentRows = (rendered: Element): Element[] =>
    byTag(rendered, 'tr').filter(row =>
      byClass(row, 'tri-environment-row-label').some(label =>
        text(label).startsWith('surface current'),
      ),
    )
  const aeroLabels = ['CdA', 'Feels Like Elevation', 'headwind share']
  const run = buildEnvironmentAnalysis(factory, { ...activity, sport: 'run' })
  assert.ok(run)
  assert.equal(currentRows(run).length, 0)
  assert.ok(
    aeroLabels.every(label =>
      byClass(run, 'tri-environment-row-label').some(cell => text(cell) === label),
    ),
  )
  const units: TriathlonPresentation['distance'][] = ['metric', 'imperial']
  for (const distance of units) {
    const rendered = buildEnvironmentAnalysis(factoryFor(presentation({ distance })), activity)
    assert.ok(rendered)
    assert.equal(byTag(rendered, 'tr').length, byTag(run, 'tr').length)
    assert.ok(
      aeroLabels.every(label =>
        byClass(rendered, 'tri-environment-row-label').every(cell => text(cell) !== label),
      ),
    )
    const windGroup = byTag(rendered, 'tbody')[1]
    assert.deepEqual(byClass(windGroup, 'tri-environment-row-label').slice(1, 4).map(text), [
      'surface current speed',
      'surface current direction',
      'surface current coverage',
    ])
    const rows = currentRows(rendered)
    assert.deepEqual(
      rows.map(row => text(byTag(row, 'td')[0])),
      ['0.14 m/s', '237° toward', '75.0% · 1h / 1h20m'],
    )
    const values = rows.map(row => byTag(row, 'td')[0])
    assert.ok(
      values.every(
        value =>
          value.properties.dataAnalysisSource === 'provider-modeled' &&
          value.properties.dataAnalysisProvider === 'noaa-loofs' &&
          value.properties.dataAnalysisFormula === 'garden-surface-current-v1' &&
          value.properties.dataAnalysisFormulaVersion === '1' &&
          String(value.properties.ariaLabel).includes('NOAA LOOFS') &&
          String(value.properties.dataGlossDef).includes('uppermost'),
      ),
    )
    assert.ok(
      byTag(rendered, 'a').some(
        link =>
          text(link) === 'NOAA LOOFS' &&
          link.properties.href === 'https://tidesandcurrents.noaa.gov/ofs/loofs/loofs_info.html',
      ),
    )
  }

  const surfaceCurrent = environment.surfaceCurrent
  surfaceCurrent.summary.averageSpeedMps = 0
  surfaceCurrent.summary.averageDirectionDeg = null
  const calm = buildEnvironmentAnalysis(factory, activity)
  assert.ok(calm)
  assert.equal(text(byTag(currentRows(calm)[0], 'td')[0]), '0.00 m/s')
  assert.equal(text(byTag(currentRows(calm)[1], 'td')[0]), '—')

  surfaceCurrent.summary.averageSpeedMps = null
  surfaceCurrent.summary.coveragePct = 0
  surfaceCurrent.summary.coveredDurationS = 0
  const missing = buildEnvironmentAnalysis(factory, activity)
  assert.ok(missing)
  const missingSpeed = byTag(currentRows(missing)[0], 'td')[0]
  assert.equal(text(missingSpeed), '—')
  assert.equal(missingSpeed.properties.dataAnalysisSource, undefined)
  assert.equal(text(byTag(currentRows(missing)[2], 'td')[0]), '0.0% · 0s / 1h20m')

  const otherLocations: StravaActivityDetail['swimLocation'][] = ['pool', null]
  for (const swimLocation of otherLocations) {
    const other = buildEnvironmentAnalysis(factory, { ...activity, swimLocation })
    assert.ok(other)
    assert.equal(currentRows(other).length, 0)
    assert.ok(
      aeroLabels.every(label =>
        byClass(other, 'tri-environment-row-label').some(cell => text(cell) === label),
      ),
    )
    assert.ok(byTag(other, 'a').every(link => text(link) !== 'NOAA LOOFS'))
  }
})

test('renders open-water current views in the shared environment graph frame with modeled provenance', () => {
  const analyses = environmentAnalyses()
  const environment = analyses.derived.environment
  assert.ok(environment)
  const current = modeledSurfaceCurrent()
  current.samples = [350, 10, null, 20].map((directionDeg, index) => {
    const speedMps = directionDeg == null ? null : 0.14
    const angle = ((directionDeg ?? 0) * Math.PI) / 180
    return {
      elapsedS: index * 1_600,
      speedMps,
      directionDeg,
      uMps: speedMps == null ? null : speedMps * Math.sin(angle),
      vMps: speedMps == null ? null : speedMps * Math.cos(angle),
      element: speedMps == null ? null : 101,
      validTime:
        speedMps == null
          ? null
          : new Date(Date.parse(current.start) + index * 1_600_000).toISOString(),
      cycleTime: speedMps == null ? null : current.start,
      sourceUrl:
        speedMps == null
          ? null
          : 'https://www.ncei.noaa.gov/thredds/dodsC/model-loofs-files/2026/07/loofs.t12z.20260709.fields.n001.nc.ascii',
    }
  })
  environment.surfaceCurrent = current
  const activity = detail({ sport: 'swim', swimLocation: 'openWater', analyses })
  const units: TriathlonPresentation['distance'][] = ['metric', 'imperial']
  for (const distance of units) {
    const rendered = buildEnvironmentAnalysis(factoryFor(presentation({ distance })), activity)
    assert.ok(rendered)
    assert.deepEqual(
      byClass(rendered, 'tri-environment-tab').map(tab => tab.properties.dataEnvironmentTab),
      [
        'cumulative',
        'uv-index',
        'temperature',
        'cloud-cover',
        'wind',
        'current-speed',
        'current-direction',
      ],
    )
    assert.equal(byClass(rendered, 'tri-environment-stage').length, 1)
    const graphs = byClass(rendered, 'tri-environment-graphs')[0]
    assert.deepEqual(
      JSON.parse(String(graphs.properties.dataEnvironmentSeries)),
      environment.samples,
    )
    assert.deepEqual(
      JSON.parse(String(graphs.properties.dataEnvironmentCurrentSeries)),
      current.samples.map(sample => ({
        elapsedS: sample.elapsedS,
        surfaceCurrentSpeedMps: sample.speedMps,
        surfaceCurrentDirectionDeg: sample.directionDeg,
      })),
    )
    for (const view of ['current-speed', 'current-direction']) {
      const panel: Element | undefined = byClass(rendered, 'tri-environment-panel').find(
        candidate => candidate.properties.dataEnvironmentPanel === view,
      )
      assert.ok(panel)
      const plot: Element = byClass(panel, 'tri-environment-plot')[0]
      assert.equal(plot.properties.viewBox, '0 0 100 32')
      assert.equal(plot.properties.role, 'slider')
      assert.equal(plot.properties.dataDomainEndElapsedS, activity.elapsedTimeS)
      assert.equal(plot.properties.dataAnalysisSource, 'provider-modeled')
      assert.equal(plot.properties.dataAnalysisProvider, 'noaa-loofs')
      assert.equal(plot.properties.dataAnalysisFormula, current.formulaId)
      assert.equal(plot.properties.dataAnalysisFormulaVersion, 1)
      assert.deepEqual(
        byClass(panel, 'tri-cax-yt').map(text),
        view === 'current-direction'
          ? ['0°', '180°', '360°']
          : ['0.00 m/s', '0.10 m/s', '0.20 m/s'],
      )
      assert.equal(
        byClass(panel, 'tri-environment-line').length,
        view === 'current-direction' ? 3 : 2,
      )
    }
  }

  delete environment.surfaceCurrent
  const missing = buildEnvironmentAnalysis(factory, activity)
  assert.ok(missing)
  for (const view of ['current-speed', 'current-direction']) {
    const panel: Element | undefined = byClass(missing, 'tri-environment-panel').find(
      candidate => candidate.properties.dataEnvironmentPanel === view,
    )
    assert.ok(panel)
    assert.equal(byClass(panel, 'tri-environment-empty').length, 1)
    assert.equal(byClass(panel, 'tri-environment-line').length, 0)
    assert.equal(byClass(panel, 'tri-environment-plot')[0].properties.role, 'img')
    assert.equal(byClass(panel, 'tri-environment-plot')[0].properties.dataAnalysisSource, undefined)
  }
  environment.surfaceCurrent = {
    ...current,
    samples: current.samples.map(sample => ({
      ...sample,
      speedMps: null,
      directionDeg: null,
      uMps: null,
      vMps: null,
      element: null,
      validTime: null,
      cycleTime: null,
      sourceUrl: null,
    })),
  }
  const gaps = buildEnvironmentAnalysis(factory, activity)
  assert.ok(gaps)
  for (const view of ['current-speed', 'current-direction']) {
    const panel: Element | undefined = byClass(gaps, 'tri-environment-panel').find(
      candidate => candidate.properties.dataEnvironmentPanel === view,
    )
    assert.ok(panel)
    const plot: Element = byClass(panel, 'tri-environment-plot')[0]
    assert.equal(plot.properties.role, 'img')
    assert.equal(plot.properties.dataAnalysisSource, undefined)
  }
  const otherLocations: StravaActivityDetail['swimLocation'][] = ['pool', null]
  for (const swimLocation of otherLocations) {
    const other = buildEnvironmentAnalysis(factory, { ...activity, swimLocation })
    assert.ok(other)
    assert.equal(byClass(other, 'tri-environment-panel').length, 5)
  }
  const run = buildEnvironmentAnalysis(factory, { ...activity, sport: 'run' })
  assert.ok(run)
  assert.equal(byClass(run, 'tri-environment-panel').length, 5)
})

test('current direction paths avoid north-crossing rotations and keep calm and unavailable readouts', () => {
  const samples = [350, 10, null, 20, 25].map((surfaceCurrentDirectionDeg, index) => ({
    elapsedS: index * 10,
    surfaceCurrentSpeedMps: surfaceCurrentDirectionDeg == null ? null : 0.14,
    surfaceCurrentDirectionDeg,
  }))
  const directions = environmentChartSeries(
    METRIC_TRIATHLON_PRESENTATION,
    samples,
    40,
    'current-direction',
  )
  assert.equal(directions.length, 3)
  assert.ok(directions.slice(0, 2).every(series => !series.path.includes('L')))
  assert.ok(directions[2].path.includes('L'))
  assert.equal(
    environmentChartReadout(
      METRIC_TRIATHLON_PRESENTATION,
      { elapsedS: 0, surfaceCurrentSpeedMps: 0 },
      null,
      'current-speed',
    ),
    '0:00 · 0.00 m/s',
  )
  assert.equal(
    environmentChartReadout(METRIC_TRIATHLON_PRESENTATION, samples[1], null, 'current-direction'),
    '0:10 · 10° toward',
  )
  assert.equal(
    environmentChartReadout(METRIC_TRIATHLON_PRESENTATION, samples[2], null, 'current-direction'),
    '0:20 · —',
  )
})

test('renders empty environment axes with native evidence and translates their no-data state', () => {
  const analyses = environmentAnalyses()
  analyses.derived.environment = null
  analyses.derived.uvScore = null
  analyses.derived.apparentWind = null
  const activity = detail({ analyses })
  const nativeOnly = buildEnvironmentAnalysis(factory, activity)
  assert.ok(nativeOnly)
  const panels = byClass(nativeOnly, 'tri-environment-panel')
  assert.equal(panels.length, 5)
  assert.equal(byClass(nativeOnly, 'tri-environment-tab').length, 5)
  assert.equal(byClass(nativeOnly, 'tri-environment-line').length, 0)
  assert.equal(byClass(nativeOnly, 'tri-environment-cursor').length, 0)
  for (const panel of panels) {
    const plot = byClass(panel, 'tri-environment-plot')[0]
    assert.equal(plot.properties.viewBox, '0 0 100 32')
    assert.equal(plot.properties.role, 'img')
    assert.equal(plot.properties.tabIndex, undefined)
    assert.equal(plot.properties.dataEnvironmentChart, undefined)
    assert.equal(plot.properties.dataDomainEndElapsedS, activity.elapsedTimeS)
    assert.equal(text(byClass(panel, 'tri-environment-empty')[0]), 'no data available')
    assert.equal(byClass(panel, 'tri-cax-ax--x').length, 1)
    assert.equal(byClass(panel, 'tri-cax-ax--y').length, 1)
    assert.deepEqual(byClass(panel, 'tri-cax-yt').map(text), ['—', '—', '—'])
    assert.deepEqual(byClass(panel, 'tri-cax-xt').map(text), [
      '0:00',
      environmentElapsedClock(activity.elapsedTimeS / 2),
      environmentElapsedClock(activity.elapsedTimeS),
    ])
  }
  assert.match(text(nativeOnly), /83 · High/)
  assert.equal(byTag(nativeOnly, 'tr').length, 21)
  const unavailable = byClass(nativeOnly, 'tri-environment-unavailable')
  assert.equal(unavailable.length, 7)
  assert.ok(unavailable.every(cell => text(cell) === '—'))
  assert.ok(
    byClass(nativeOnly, 'tri-environment-row-label').some(label => text(label) === 'air speed'),
  )

  const frenchEnvironment = buildEnvironmentAnalysis(
    factoryFor(frenchPresentation),
    detail({ analyses: environmentAnalyses() }),
  )
  assert.ok(frenchEnvironment)
  assert.match(text(frenchEnvironment), /environnement/)
  assert.match(text(frenchEnvironment), /exposition UV/)
  assert.match(text(frenchEnvironment), /couverture nuageuse/)
  const frenchEmpty = buildEnvironmentAnalysis(factoryFor(frenchPresentation), activity)
  assert.ok(frenchEmpty)
  assert.equal(text(byClass(frenchEmpty, 'tri-environment-empty')[0]), 'aucune donnée disponible')
})

test('keeps unavailable environment views while rendering recorded zero values', () => {
  const analyses = environmentAnalyses()
  const environment = analyses.derived.environment
  assert.ok(environment)
  environment.samples = environment.samples.map(sample => ({
    ...sample,
    cumulativeSed: null,
    cumulativeMovingTelemetrySed: null,
    uvIndex: null,
    ambientTemperatureC: 0,
    cloudCoverPct: null,
    headwindKph: null,
  }))
  const rendered = buildEnvironmentAnalysis(factory, detail({ analyses }))
  assert.ok(rendered)
  const panels = byClass(rendered, 'tri-environment-panel')
  assert.equal(panels.length, 5)
  const temperature = panels.find(panel => panel.properties.dataEnvironmentPanel === 'temperature')
  assert.ok(temperature)
  assert.equal(temperature.properties.hidden, undefined)
  assert.equal(byClass(temperature, 'tri-environment-empty').length, 0)
  assert.equal(byClass(temperature, 'tri-environment-line').length, 1)
  assert.equal(byClass(temperature, 'tri-environment-plot')[0].properties.role, 'slider')
  assert.equal(byClass(rendered, 'tri-environment-empty').length, 4)

  environment.samples = environment.samples.slice(0, 1)
  const sparse = buildEnvironmentAnalysis(factory, detail({ analyses }))
  assert.ok(sparse)
  assert.equal(byClass(sparse, 'tri-environment-empty').length, 5)
  assert.equal(byClass(sparse, 'tri-environment-line').length, 0)
})

test('environment paths retain explicit gaps and step hourly values', () => {
  const samples = environmentAnalyses().derived.environment?.samples ?? []
  const gapped = samples.map((sample, index) =>
    index === 2 ? { ...sample, ambientTemperatureC: null } : sample,
  )
  const temperature = environmentChartSeries(
    METRIC_TRIATHLON_PRESENTATION,
    gapped,
    4_800,
    'temperature',
  )
  const ultraviolet = environmentChartSeries(
    METRIC_TRIATHLON_PRESENTATION,
    samples,
    4_800,
    'uv-index',
  )

  assert.equal(temperature.length, 2)
  assert.equal(ultraviolet.length, 3)
  assert.ok(ultraviolet.every(segment => segment.path.includes('H') && segment.path.includes('V')))
  assert.ok(ultraviolet.every(segment => segment.color != null))
})

test('wind plots signed headwind with centered axes and shares localized cursor values', () => {
  const activity = detail({ analyses: environmentAnalyses() })
  const locales: TriathlonPresentation['locale'][] = ['en', 'fr']
  const units: TriathlonPresentation['distance'][] = ['metric', 'imperial']
  for (const locale of locales) {
    for (const distance of units) {
      const selected = presentation({ locale, distance })
      const rendered = buildEnvironmentAnalysis(factoryFor(selected), activity)
      assert.ok(rendered)
      const panel = byClass(rendered, 'tri-environment-panel').at(-1)
      assert.ok(panel)
      assert.equal(panel.properties.dataEnvironmentPanel, 'wind')
      const plot = byClass(panel, 'tri-environment-plot')[0]
      assert.equal(plot.properties.dataEnvironmentChart, 'wind')
      assert.equal(plot.properties.role, 'slider')
      assert.equal(plot.properties.ariaLabel, locale === 'fr' ? 'vent' : 'wind')
      assert.deepEqual(
        byClass(panel, 'tri-cax-yt').map(text),
        distance === 'imperial'
          ? ['-5 mph', '0 mph', '+5 mph']
          : ['-10 km/h', '0 km/h', '+10 km/h'],
      )
      const baseline = byClass(panel, 'tri-environment-gridline--zero')[0]
      assert.equal(baseline.properties.y1, 15)
      assert.equal(baseline.properties.y2, 15)
      assert.equal(
        byClass(panel, 'tri-environment-line')[0].properties.d,
        distance === 'imperial'
          ? 'M34.000,3.070L66.000,20.965L98.000,7.544'
          : 'M34.000,5.400L66.000,19.800L98.000,9.000',
      )
      const readout = text(byClass(rendered, 'tri-environment-readout')[0])
      const windSpeed = distance === 'imperial' ? '+3.1 mph' : '+5.0 km/h'
      assert.ok(readout.includes(`${locale === 'fr' ? 'vent de face' : 'headwind'} ${windSpeed}`))
      assert.ok(readout.includes(locale === 'fr' ? 'vent latéral' : 'crosswind'))
      assert.ok(readout.includes(distance === 'imperial' ? '17.4 mph' : '28.0 km/h'))
    }
  }
})

test('wind preserves calm zero samples, missing data, and explicit trace gaps', () => {
  const analyses = environmentAnalyses()
  const environment = analyses.derived.environment
  assert.ok(environment)
  const samples = environment.samples
  environment.samples = samples.map(sample => ({ ...sample, headwindKph: 0 }))
  const calm = buildEnvironmentAnalysis(factory, detail({ analyses }))
  assert.ok(calm)
  const calmPanel = byClass(calm, 'tri-environment-panel').at(-1)
  assert.ok(calmPanel)
  assert.equal(byClass(calmPanel, 'tri-environment-empty').length, 0)
  assert.equal(
    byClass(calmPanel, 'tri-environment-line')[0].properties.d,
    'M2.000,15.000L34.000,15.000L66.000,15.000L98.000,15.000',
  )

  environment.samples = samples.map(sample => ({ ...sample, headwindKph: null }))
  const missing = buildEnvironmentAnalysis(factory, detail({ analyses }))
  assert.ok(missing)
  const missingPanel = byClass(missing, 'tri-environment-panel').at(-1)
  assert.ok(missingPanel)
  assert.equal(byClass(missingPanel, 'tri-environment-line').length, 0)
  assert.equal(byClass(missingPanel, 'tri-environment-empty').length, 1)
  assert.equal(byClass(missingPanel, 'tri-environment-plot')[0].properties.role, 'img')

  const gapped = samples.map((sample, index) => ({
    ...sample,
    headwindKph: index === 2 ? null : 8,
  }))
  assert.deepEqual(
    environmentChartSeries(METRIC_TRIATHLON_PRESENTATION, gapped, 4_800, 'wind').map(
      series => series.path,
    ),
    ['M2.000,5.400L34.000,5.400', 'M98.000,5.400'],
  )
})

test('open-water wind plots ambient observations independently of unavailable apparent wind', () => {
  const analyses = environmentAnalyses()
  analyses.native.myWindsock = null
  analyses.derived.apparentWind = null
  const environment = analyses.derived.environment
  assert.ok(environment)
  environment.samples = environment.samples.map((sample, index) => ({
    ...sample,
    windSpeedKph: index === 2 ? null : 0,
    headwindKph: null,
    crosswindKph: null,
    apparentAirSpeedKph: null,
    yawDeg: null,
  }))
  const activity = detail({ sport: 'swim', swimLocation: 'openWater', analyses })
  const units: TriathlonPresentation['distance'][] = ['metric', 'imperial']
  for (const distance of units) {
    const selected = presentation({ distance })
    const rendered = buildEnvironmentAnalysis(factoryFor(selected), activity)
    assert.ok(rendered)
    const panel = byClass(rendered, 'tri-environment-panel').find(
      panel => panel.properties.dataEnvironmentPanel === 'wind',
    )
    assert.ok(panel)
    const plot = byClass(panel, 'tri-environment-plot')[0]
    assert.equal(plot.properties.role, 'slider')
    assert.equal(plot.properties.dataEnvironmentWindMetric, 'ambient')
    assert.equal(plot.properties.ariaLabel, 'ambient wind speed')
    assert.deepEqual(
      byClass(panel, 'tri-cax-yt').map(text),
      distance === 'imperial' ? ['0 mph', '0.5 mph', '1 mph'] : ['0 km/h', '0.5 km/h', '1 km/h'],
    )
    assert.equal(byClass(panel, 'tri-environment-empty').length, 0)
    assert.equal(byClass(panel, 'tri-environment-line').length, 2)
    const readout = text(byClass(rendered, 'tri-environment-readout')[0])
    assert.match(readout, /ambient wind speed 0\.0 (?:km\/h|mph)/)
    assert.doesNotMatch(readout, /headwind|apparent air|crosswind/)
  }

  environment.samples = environment.samples.map(sample => ({ ...sample, windSpeedKph: null }))
  const missing = buildEnvironmentAnalysis(factory, activity)
  assert.ok(missing)
  const missingPanel = byClass(missing, 'tri-environment-panel').find(
    panel => panel.properties.dataEnvironmentPanel === 'wind',
  )
  assert.ok(missingPanel)
  assert.equal(byClass(missingPanel, 'tri-environment-line').length, 0)
  assert.equal(byClass(missingPanel, 'tri-environment-empty').length, 1)
})

test('temperature axes, series, and readouts follow selected units independently of locale', () => {
  const activity = detail({ analyses: environmentAnalyses() })
  const locales: TriathlonPresentation['locale'][] = ['en', 'fr']
  const units: TriathlonPresentation['distance'][] = ['metric', 'imperial']
  for (const locale of locales) {
    for (const distance of units) {
      const selected = presentation({ locale, distance })
      const imperial = distance === 'imperial'
      const rendered = buildActivity(factoryFor(selected), activity, true)
      const ambient = byClass(rendered, 'tri-elev-wrap').find(
        graph => graph.properties.dataTriTrace === 'temperature',
      )
      assert.ok(ambient)
      assert.equal(text(byClass(ambient, 'tri-elev-range')[0]), imperial ? '72°F avg' : '22°C avg')
      assert.deepEqual(
        byClass(ambient, 'tri-cax-yt').map(text),
        imperial ? ['70°F', '75°F', '80°F'] : ['22°C', '24°C', '26°C'],
      )
      const environment = byClass(rendered, 'tri-environment')[0]
      const panel = byClass(environment, 'tri-environment-panel').find(
        panel => panel.properties.dataEnvironmentPanel === 'temperature',
      )
      assert.ok(panel)
      const ticks = byClass(panel, 'tri-cax-yt')
      assert.deepEqual(
        ticks.map(text),
        imperial ? ['60°F', '65°F', '70°F', '75°F'] : ['16°C', '18°C', '20°C', '22°C', '24°C'],
      )
      const gridlines = byClass(panel, 'tri-environment-gridline')
      assert.deepEqual(
        gridlines.map(line => `top:${((Number(line.properties.y1) / 32) * 100).toFixed(2)}%`),
        ticks.map(tick => tick.properties.style),
      )
      assert.equal(
        byClass(panel, 'tri-environment-line')[0].properties.d,
        imperial
          ? 'M2.000,8.440L34.000,11.320L66.000,17.080L98.000,19.960'
          : 'M2.000,9.000L34.000,12.000L66.000,18.000L98.000,21.000',
      )
      assert.ok(
        text(byClass(environment, 'tri-environment-readout')[0]).includes(
          imperial ? '64°F' : '18°C',
        ),
      )
      const temperature = metricSpecs(selected, activity, {
        zones: null,
        curveRef: [],
        curveYearRef: [],
        runCurveRef: [],
        runCurveYearRef: [],
        curveYear: null,
        criticalPower: null,
        criticalPowerYear: null,
        ftp: null,
        goalFtp: null,
        vt1: null,
      }).find(spec => spec.label === 'temperature')
      assert.ok(temperature)
      assert.equal(temperature.fmt(22), imperial ? '72°F' : '22°C')
      assert.ok(temperature.readout(activity.route[0], 0).includes(imperial ? '72°F' : '22°C'))
    }
  }
  assert.deepEqual(
    activity.route.map(point => point.tempC),
    [22, 23, 24, 25],
  )
  assert.deepEqual(
    activity.analyses.derived.environment?.samples.map(sample => sample.ambientTemperatureC),
    [22, 21, 19, 18],
  )
})

test('constant freezing and subzero environment observations retain Fahrenheit values and precision', () => {
  for (const celsius of [0, -5]) {
    const analyses = environmentAnalyses()
    const environment = analyses.derived.environment
    assert.ok(environment)
    environment.samples = environment.samples.map(sample => ({
      ...sample,
      ambientTemperatureC: celsius,
    }))
    const rendered = buildEnvironmentAnalysis(
      factoryFor(imperialPresentation),
      detail({ analyses }),
    )
    assert.ok(rendered)
    const panel = byClass(rendered, 'tri-environment-panel').find(
      panel => panel.properties.dataEnvironmentPanel === 'temperature',
    )
    assert.ok(panel)
    assert.deepEqual(
      byClass(panel, 'tri-cax-yt').map(text),
      celsius === 0 ? ['31.5°F', '32°F', '32.5°F'] : ['22.5°F', '23°F', '23.5°F'],
    )
    assert.equal(
      byClass(panel, 'tri-environment-line')[0].properties.d,
      'M2.000,15.000L34.000,15.000L66.000,15.000L98.000,15.000',
    )
    assert.ok(
      text(byClass(rendered, 'tri-environment-readout')[0]).includes(
        celsius === 0 ? '32°F' : '23°F',
      ),
    )
  }
})

test('environment elapsed clocks switch to hours without changing the minute clock', () => {
  assert.equal(environmentElapsedClock(0), '0:00')
  assert.equal(environmentElapsedClock(1_999), '33:19')
  assert.equal(environmentElapsedClock(3_997), '1:06:37')
  assert.equal(environmentElapsedClock(36_432), '10:07:12')
})

test('calibrated cumulative paths use the selected dose clock', () => {
  const samples = environmentAnalyses().derived.environment?.samples ?? []
  const moving = environmentChartSeries(
    METRIC_TRIATHLON_PRESENTATION,
    samples,
    4_800,
    'cumulative',
    { coefficientSed: 10, doseClock: 'moving-telemetry' },
  )
  const elapsed = environmentChartSeries(
    METRIC_TRIATHLON_PRESENTATION,
    samples,
    4_800,
    'cumulative',
    { coefficientSed: 10, doseClock: 'elapsed' },
  )

  assert.match(moving[0]?.path ?? '', /L98\.000,20\.040$/)
  assert.match(elapsed[0]?.path ?? '', /L98\.000,19\.800$/)
})

test('route map exposes UV and signed wind from Garden samples', () => {
  const activity = detail({ analyses: environmentAnalyses() })
  const specs = metricSpecs(METRIC_TRIATHLON_PRESENTATION, activity, {
    zones: null,
    curveRef: [],
    curveYearRef: [],
    runCurveRef: [],
    runCurveYearRef: [],
    curveYear: null,
    criticalPower: null,
    criticalPowerYear: null,
    ftp: null,
    goalFtp: null,
    vt1: null,
  })
  const ultraviolet = specs.find(spec => spec.label === 'UV')
  const wind = specs.find(spec => spec.label === 'wind')

  assert.ok(ultraviolet)
  assert.ok(wind)
  assert.equal(ultraviolet.pick(activity.route[3], 3), 0)
  assert.equal(ultraviolet.valid?.(activity.route[3], 3), true)
  assert.equal(wind.pick(activity.route[1], 1), 8)
  assert.match(wind.readout(activity.route[1], 1), /apparent air 31\.0 km\/h/)
  assert.match(wind.readout(activity.route[1], 1), /yaw -5\.0°/)
})

const garminVerification = (
  overrides: Partial<NonNullable<StravaActivityDetail['garmin']>> = {},
): NonNullable<StravaActivityDetail['garmin']> => ({
  activityId: 'connect:123',
  name: 'Threshold ride',
  sourceDevice: 'Edge 1050',
  startDate: '2026-07-09T12:00:00Z',
  startDiffS: 0,
  distanceM: 30_000,
  distanceDeltaM: 0,
  distanceDeltaPct: 0,
  movingTimeS: 4_800,
  movingTimeDeltaS: 0,
  elapsedTimeS: 4_800,
  elapsedTimeDeltaS: 0,
  totalCalories: null,
  caloriesDelta: null,
  avgHeartRate: null,
  avgHeartRateDelta: null,
  avgPower: null,
  avgPowerDelta: null,
  avgCadence: null,
  normalizedPower: null,
  maxPower: null,
  totalWorkKJ: null,
  totalWorkDeltaKJ: null,
  trainingStressScore: null,
  intensityFactor: null,
  trainingEffectActivityId: null,
  aerobicTrainingEffect: null,
  anaerobicTrainingEffect: null,
  exerciseLoad: null,
  trainingEffectLabel: null,
  aerobicTrainingEffectMessage: null,
  anaerobicTrainingEffectMessage: null,
  ...overrides,
})

test('links every activity header icon to its Strava activity in a new tab', () => {
  const sports: StravaActivityDetail['sport'][] = [
    'bike',
    'run',
    'swim',
    'walk',
    'strength',
    'yoga',
    'treatment',
    'sauna',
  ]
  for (const sport of sports) {
    const activity = detail({
      sport,
      sources: [
        { provider: 'garmin', activityId: 'connect:777', name: null, fileName: null },
        { provider: 'strava', activityId: '20147774828', name: null, fileName: null },
      ],
    })
    const card = buildActivity(factory, activity)
    const links = byClass(byClass(card, 'tri-act-head')[0], 'tri-act-icon-link')
    assert.equal(links.length, 1)
    const link = links[0]
    assert.equal(link.tagName, 'a')
    assert.equal(link.properties.href, 'https://www.strava.com/activities/20147774828')
    assert.equal(link.properties.target, '_blank')
    assert.deepEqual(link.properties.rel, ['noopener', 'noreferrer'])
    assert.equal(link.properties.ariaLabel, 'Threshold ride · Strava')
    assert.equal(link.properties.dataSiteCursorAction, '')
    assert.deepEqual(link.children, [buildIcon(factory, sport)])
  }
})

test('activity icon links retain legacy Strava IDs and omit records without Strava activities', () => {
  const legacy = buildActivityIcon(factory, detail())
  assert.equal(legacy.properties.href, 'https://www.strava.com/activities/101')
  const providerOnly = detail({
    sources: [{ provider: 'garmin', activityId: 'connect:777', name: null, fileName: null }],
  })
  assert.deepEqual(buildActivityIcon(factory, providerOnly), buildIcon(factory, providerOnly.sport))
  const manual = detail({
    sport: 'sauna',
    sauna: {
      time: '18:30',
      temperatureC: 90,
      humidityPct: 10,
      cooldown: 'cold plunge',
      heatTrainingLoad: null,
      heartRateSource: 'oura',
      source: 'manual',
    },
  })
  assert.deepEqual(buildActivityIcon(factory, manual), buildIcon(factory, 'sauna'))
})

test('renders equipment in shared summary rows with Strava provenance for bikes and shoes', () => {
  for (const sport of ['bike', 'run', 'walk', 'strength'] as const) {
    const equipment = { id: 'b123', name: 'Speedmax <CF> & SLX', source: 'strava' as const }
    const activity = detail({ sport, equipment })
    const rows = activityTableRows(factory.presentation, activity)
    assert.deepEqual(
      rows.filter(([label]) => label === 'equipment'),
      [['equipment', equipment.name]],
    )
    const built = buildActivity(factory, activity)
    const row = byTag(built, 'tr').find(node => node.properties.dataStatKey === 'equipment')
    assert.ok(row)
    assert.equal(text(byClass(row, 'tri-act-stat-v')[0]), equipment.name)
    assert.equal(row.properties.dataEquipmentSource, 'strava')
    assert.equal(row.properties.dataEquipmentId, equipment.id)
  }
  const unnamed = detail({ equipment: { id: 's123', name: null, source: 'strava' } })
  assert.ok(
    activityTableRows(factory.presentation, unnamed).some(
      ([label, value]) => label === 'equipment' && value === 's123',
    ),
  )
  assert.equal(
    activityTableRows(factory.presentation, detail()).some(([label]) => label === 'equipment'),
    false,
  )
  assert.equal(triText('fr', 'equipment'), 'équipement')
})

test('renders bike computers and run, walk, and swim devices as distinct activity rows', () => {
  const garmin = buildActivity(
    factory,
    detail({ computer: 'garmin', windKph: 11, windDir: 'E', windGustKph: 17 }),
  )
  const wahoo = buildActivity(factory, detail({ computer: 'wahoo' }))
  const appleRun = buildActivity(factory, detail({ sport: 'run', device: 'apple-watch-ultra-3' }))
  const garminWalk = buildActivity(
    factory,
    detail({ sport: 'walk', device: 'garmin-forerunner-970' }),
  )
  const appleSwim = buildActivity(factory, detail({ sport: 'swim', device: 'apple-watch-ultra-3' }))
  const bikeWithDevice = buildActivity(
    factory,
    detail({ sport: 'bike', device: 'garmin-forerunner-970' }),
  )
  const absent = buildActivity(factory, detail())
  const computerRow = (node: Element): Element | undefined =>
    byTag(node, 'tr').find(row => row.properties.dataStatKey === 'computer')
  const deviceRow = (node: Element): Element | undefined =>
    byTag(node, 'tr').find(row => row.properties.dataStatKey === 'device')

  assert.equal(text(byClass(computerRow(garmin)!, 'tri-act-stat-v')[0]), 'Edge 1050')
  assert.equal(text(byClass(computerRow(wahoo)!, 'tri-act-stat-v')[0]), 'ELEMNT BOLT 3')
  assert.equal(text(byClass(deviceRow(appleRun)!, 'tri-act-stat-v')[0]), 'Apple Watch Ultra 3')
  assert.equal(text(byClass(deviceRow(garminWalk)!, 'tri-act-stat-v')[0]), 'Garmin Forerunner 970')
  assert.equal(text(byClass(deviceRow(appleSwim)!, 'tri-act-stat-v')[0]), 'Apple Watch Ultra 3')
  assert.equal(triText('fr', 'device'), 'appareil')
  assert.equal(triText('fr', 'device temp'), "température de l'appareil")
  assert.deepEqual(
    byTag(byClass(garmin, 'tri-act-stats')[0], 'tr')
      .map(row => row.properties.dataStatKey)
      .slice(-2),
    ['wind', 'computer'],
  )
  assert.equal(computerRow(appleRun), undefined)
  assert.equal(computerRow(garminWalk), undefined)
  assert.equal(computerRow(appleSwim), undefined)
  assert.equal(deviceRow(garmin), undefined)
  assert.equal(deviceRow(wahoo), undefined)
  assert.equal(deviceRow(bikeWithDevice), undefined)
  assert.equal(computerRow(absent), undefined)
  assert.equal(deviceRow(absent), undefined)
  assert.equal(byClass(garmin, 'tri-act-computer').length, 0)
})

test('prefers authored computer text in shared summary rows and rendered cards', () => {
  const computers: StravaActivityDetail['computer'][] = ['garmin', 'wahoo', null]
  for (const computer of computers) {
    const activity = detail({ computer, computerOverride: 'Wahoo ELEMNT BOLT 3' })
    if (computer === 'wahoo')
      activity.wahoo = {
        activityId: 'wahoo:99',
        fitPath: null,
        sha256: 'a'.repeat(64),
        sourceDevice: 'ELEMNT ROAM',
        startOffsetS: 0,
        distanceM: 28_100,
        metrics: emptyWahooMetrics(),
        summarySources: {},
        streamFallback: 'strava',
      }
    assert.deepEqual(
      activityTableRows(factory.presentation, activity).filter(([key]) => key === 'computer'),
      [['computer', 'Wahoo ELEMNT BOLT 3']],
    )
    const row = byTag(buildActivity(factory, activity), 'tr').find(
      row => row.properties.dataStatKey === 'computer',
    )
    assert.ok(row)
    assert.equal(text(byClass(row, 'tri-act-stat-v')[0]), 'Wahoo ELEMNT BOLT 3')
  }
})

test('renders one metric per row and lists contributing recordings in the source hover', () => {
  const activity = detail({
    computer: 'wahoo',
    virtual: true,
    distanceSource: 'garmin',
    distanceKm: 28.1,
    garmin: garminVerification({
      distanceM: 28_112,
      distanceDeltaM: -38_686.5,
      normalizedPower: 179,
      trainingStressScore: 89.9,
    }),
    sources: [
      { provider: 'strava', activityId: '101', name: 'Morning ride', fileName: null },
      {
        provider: 'garmin',
        activityId: 'connect:55',
        name: 'Virtual course',
        fileName: 'course.fit',
      },
      { provider: 'wahoo', activityId: 'wahoo:99', name: 'Power & cadence', fileName: 'ride.fit' },
    ],
  })
  const rows = activityTableRows(METRIC_TRIATHLON_PRESENTATION, activity)
  const renderedRows = byTag(
    byClass(buildActivity(factory, activity), 'tri-act-stats')[0],
    'tr',
  ).map(row => [row.properties.dataStatKey, text(byClass(row, 'tri-act-stat-v')[0])])
  assert.deepEqual(renderedRows, rows)
  const computerIndex = rows.findIndex(([label]) => label === 'computer')
  assert.ok(computerIndex >= 0)
  assert.deepEqual(rows.slice(computerIndex + 1, computerIndex + 3), [
    ['source', 'Garmin'],
    ['activity', 'virtual'],
  ])
  assert.equal(rows.filter(([label]) => label === 'distance').length, 1)
  assert.equal(rows.filter(([label]) => label === 'NP').length, 1)
  assert.equal(rows.filter(([label]) => label === 'TSS').length, 1)
  assert.equal(
    rows.some(([label]) => /^(Strava|Garmin|Wahoo) |^(distance|telemetry) source$/.test(label)),
    false,
  )
  assert.ok(rows.some(([label, value]) => label === 'TSS' && value === '89.9'))
  assert.ok(rows.some(([label, value]) => label === 'NP' && value === '205 W'))
  const source = byClass(buildActivity(factory, activity), 'tri-act-source')[0]
  assert.equal(text(source), 'Garmin')
  assert.equal(source.properties.tabIndex, 0)
  assert.equal(source.properties.dataGloss, '')
  assert.equal(
    source.properties.dataGlossDef,
    'Strava · 101\nMorning ride\n\nGarmin · connect:55\nVirtual course\ncourse.fit\n\nWahoo · wahoo:99\nPower & cadence\nride.fit',
  )
  assert.ok(String(source.properties.ariaLabel).includes('course.fit'))
  activity.wahoo = {
    activityId: 'wahoo:99',
    fitPath: null,
    sha256: 'a'.repeat(64),
    sourceDevice: 'ELEMNT BOLT',
    startOffsetS: 0,
    distanceM: 28_100,
    metrics: { ...emptyWahooMetrics(), trainingStressScore: 0 },
    summarySources: {},
    streamFallback: 'strava',
  }
  assert.deepEqual(
    activityTableRows(METRIC_TRIATHLON_PRESENTATION, activity).filter(
      ([label]) => label === 'source',
    ),
    [['source', 'Wahoo']],
  )
  assert.deepEqual(
    activityTableRows(METRIC_TRIATHLON_PRESENTATION, activity).filter(([label]) => label === 'TSS'),
    [['TSS', '0']],
  )
  const stravaRide = {
    ...activity,
    wahoo: undefined,
    virtual: false,
    distanceSource: 'strava' as const,
  }
  assert.deepEqual(
    activityTableRows(METRIC_TRIATHLON_PRESENTATION, stravaRide).filter(
      ([label]) => label === 'source',
    ),
    [['source', 'Strava']],
  )
})

test('renders WeatherKit humidity below wind in shared server and hydrated activity rows', () => {
  const activity = detail({
    windKph: 11,
    windDir: 'E',
    windGustKph: 17,
    averageRelativeHumidityPct: 68,
    relativeHumidityProvenance: {
      source: 'weatherkit',
      sourceKind: 'modeled',
      samplingMethod: 'route-hour',
      inputTimestamp: '2026-07-09T12:00:00Z',
      coveragePct: 75,
    },
    analyses: {
      native: {
        myWindsock: {
          source: 'provider-native',
          provider: 'mywindsock',
          transport: 'strava-description',
          schemaVersion: 1,
          activityId: 101,
          retrievedAt: 1,
          weatherImpactPct: 0.6,
          cdaM2: null,
          feelsLikeElevationM: null,
          headwindPct: null,
          headwindMinKph: null,
          headwindMaxKph: null,
          longestHeadwindS: null,
          airSpeedKph: null,
          averageTemperatureC: null,
          precipitationProbabilityPct: null,
          precipitationRateMmPerHour: null,
        },
        pelotan: null,
      },
      derived: { environment: null, uvScore: null, apparentWind: null },
    },
  })
  const rows = activityTableRows(METRIC_TRIATHLON_PRESENTATION, activity)
  const renderedRows = byTag(
    byClass(buildActivity(factory, activity), 'tri-act-stats')[0],
    'tr',
  ).map(row => [row.properties.dataStatKey, text(byClass(row, 'tri-act-stat-v')[0])])
  const humidityRow = byTag(buildActivity(factory, activity), 'tr').find(
    row => row.properties.dataStatKey === 'humidity',
  )

  assert.deepEqual(rows.slice(-3), [
    ['wind', '11 km/h E / gust 17'],
    ['humidity', '68%'],
    ['weather impact', '0.6%'],
  ])
  assert.deepEqual(renderedRows, rows)
  assert.equal(humidityRow?.properties.dataWeatherSource, 'weatherkit')
  assert.equal(humidityRow?.properties.dataWeatherSourceKind, 'modeled')
  assert.equal(humidityRow?.properties.dataWeatherSamplingMethod, 'route-hour')
  assert.equal(humidityRow?.properties.dataWeatherInputTimestamp, '2026-07-09T12:00:00Z')
  assert.equal(humidityRow?.properties.dataWeatherCoveragePct, '75')
  assert.deepEqual(
    moreStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({ windKph: 0, averageRelativeHumidityPct: 0 }),
    ).slice(-2),
    [
      ['wind', '0 km/h'],
      ['humidity', '0%'],
    ],
  )
  assert.equal(
    moreStatRows(METRIC_TRIATHLON_PRESENTATION, detail()).some(([label]) => label === 'humidity'),
    false,
  )
})

test('calculates missing exercise load from Garmin intensity without replacing native load', () => {
  assert.deepEqual(
    calculateActivityExerciseLoad(
      detail({ garmin: garminVerification({ intensityFactor: 0.803 }) }),
    ),
    { value: 86, source: 'garmin' },
  )
  assert.equal(
    calculateActivityExerciseLoad(
      detail({ garmin: garminVerification({ intensityFactor: 0.803, exerciseLoad: 301.7 }) }),
    ),
    null,
  )
})

test('summarizes Garmin training metrics with the dominant effect for every activity kind', () => {
  const garmin = garminVerification({
    intensityFactor: 0.803,
    aerobicTrainingEffect: 4.5,
    anaerobicTrainingEffect: 0,
    exerciseLoad: 301.7,
    trainingEffectLabel: 'AEROBIC_BASE',
    aerobicTrainingEffectMessage: 'HIGHLY_IMPROVING_AEROBIC_ENDURANCE_10',
    anaerobicTrainingEffectMessage: 'NO_ANAEROBIC_BENEFIT_0',
  })
  const expected: [string, string][] = [
    ['intensity factor', '0.803'],
    ['training effect', 'base'],
    ['exercise load', '302'],
  ]

  assert.deepEqual(
    activityStatRows(METRIC_TRIATHLON_PRESENTATION, detail({ garmin })).slice(-3),
    expected,
  )
  assert.deepEqual(
    activityStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({ sport: 'strength', route: [], bestEfforts: null, garmin }),
    ).slice(-3),
    expected,
  )
  assert.deepEqual(activityStatRows(frenchPresentation, detail({ garmin })).slice(-3), [
    ['intensity factor', '0,803'],
    ['training effect', 'base'],
    ['exercise load', '302'],
  ])

  const rendered = buildActivity(factory, detail({ garmin }))
  const labels = byClass(rendered, 'tri-act-stat-k')
  const exerciseLoad = labels.find(label => text(label) === 'exercise load')
  assert.equal(exerciseLoad?.properties.dataI18n, 'exercise load')
  const summaryEffect = byTag(rendered, 'tr').find(
    row => row.properties.dataStatKey === 'training effect',
  )
  assert.equal(summaryEffect?.properties.dataTrainingEffectGroup, 'low-aerobic')

  const highAerobic = buildActivity(
    factory,
    detail({ garmin: garminVerification({ trainingEffectLabel: 'VO2_MAX' }) }),
  )
  assert.equal(
    byTag(highAerobic, 'tr').find(row => row.properties.dataStatKey === 'training effect')
      ?.properties.dataTrainingEffectGroup,
    'high-aerobic',
  )

  const anaerobic = buildActivity(
    factory,
    detail({ garmin: garminVerification({ trainingEffectLabel: 'ANAEROBIC_CAPACITY' }) }),
  )
  assert.equal(
    byTag(anaerobic, 'tr').find(row => row.properties.dataStatKey === 'training effect')?.properties
      .dataTrainingEffectGroup,
    'anaerobic',
  )
})

test('uses base by default and recovery for strength, yoga, and treatment', () => {
  assert.deepEqual(activityStatRows(METRIC_TRIATHLON_PRESENTATION, detail()).slice(-1), [
    ['training effect', 'base'],
  ])
  assert.deepEqual(
    activityStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({ garmin: garminVerification({ intensityFactor: 0.803, exerciseLoad: 301.7 }) }),
    ).slice(-3),
    [
      ['intensity factor', '0.803'],
      ['training effect', 'base'],
      ['exercise load', '302'],
    ],
  )
  assert.deepEqual(
    activityStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({ sport: 'strength', route: [], bestEfforts: null }),
    ).slice(-1),
    [['training effect', 'recovery']],
  )
  for (const sport of ['yoga', 'treatment'] as const)
    assert.deepEqual(
      activityStatRows(
        METRIC_TRIATHLON_PRESENTATION,
        detail({ sport, route: [], bestEfforts: null }),
      ).slice(-1),
      [['training effect', 'recovery']],
    )

  const rendered = buildActivity(factory, detail())
  const row = byTag(rendered, 'tr').find(
    candidate => candidate.properties.dataStatKey === 'training effect',
  )
  assert.ok(row)
  assert.equal(text(row), 'training effectbase')
  assert.equal(row.properties.dataTrainingEffectGroup, 'low-aerobic')

  const renderedStrength = buildActivity(
    factory,
    detail({ sport: 'strength', route: [], bestEfforts: null }),
  )
  const strengthRow = byTag(renderedStrength, 'tr').find(
    candidate => candidate.properties.dataStatKey === 'training effect',
  )
  assert.ok(strengthRow)
  assert.equal(text(strengthRow), 'training effectrecovery')
  assert.equal(strengthRow.properties.dataTrainingEffectGroup, 'low-aerobic')
})

test('classifies calculated anaerobic capacity and preserves Garmin primary benefits', () => {
  const calculatedTrainingEffect = {
    aerobic: 4.5,
    anaerobic: 4.5,
    evidence: {
      aerobic: { source: 'relative-effort' as const, load: 109 },
      anaerobic: {
        source: 'power' as const,
        effect: 4.5,
        effortCount: 140,
        stimulus: 9.6,
        criticalPowerWatts: 250.8,
        wPrimeKilojoules: 10.1,
      },
    },
  }
  const ride = detail({ calculatedTrainingEffect })
  assert.equal(activityTrainingEffectLabel(ride), 'anaerobic capacity')
  assert.deepEqual(activityStatRows(METRIC_TRIATHLON_PRESENTATION, ride).slice(-1), [
    ['training effect', 'anaerobic capacity'],
  ])
  const summary = byTag(buildActivity(factory, ride), 'tr').find(
    row => row.properties.dataStatKey === 'training effect',
  )
  assert.equal(summary?.properties.dataTrainingEffectGroup, 'anaerobic')

  assert.equal(
    activityTrainingEffectLabel(
      detail({ calculatedTrainingEffect: { ...calculatedTrainingEffect, anaerobic: 3.3 } }),
    ),
    'base',
  )
  assert.equal(
    activityTrainingEffectLabel(
      detail({ calculatedTrainingEffect: { ...calculatedTrainingEffect, anaerobic: 2.9 } }),
    ),
    'base',
  )
  assert.equal(
    activityTrainingEffectLabel(
      detail({
        calculatedTrainingEffect: {
          ...calculatedTrainingEffect,
          evidence: {
            ...calculatedTrainingEffect.evidence,
            anaerobic: { source: 'heart-rate', seconds: 1_800 },
          },
        },
      }),
    ),
    'base',
  )
  assert.equal(
    activityTrainingEffectLabel(
      detail({
        garmin: garminVerification({ trainingEffectLabel: 'SPEED' }),
        calculatedTrainingEffect,
      }),
    ),
    'sprint',
  )
  assert.equal(
    activityTrainingEffectLabel(detail({ sport: 'sauna', calculatedTrainingEffect })),
    'recovery',
  )
})

test('labels calculated sprint work separately from longer anaerobic work', () => {
  const calculatedTrainingEffect = {
    aerobic: 3.4,
    anaerobic: 3.7,
    evidence: {
      aerobic: { source: 'relative-effort' as const, load: 70 },
      anaerobic: {
        source: 'power' as const,
        effect: 3.7,
        effortCount: 12,
        sprintEffortCount: 10,
        stimulus: 5.6,
        sprintStimulus: 4.8,
        criticalPowerWatts: 300,
        wPrimeKilojoules: 18,
      },
    },
  }
  const ride = detail({ calculatedTrainingEffect })
  assert.equal(activityTrainingEffectLabel(ride), 'sprint')
  assert.equal(
    activityTrainingEffectLabel(
      detail({
        calculatedTrainingEffect: {
          ...calculatedTrainingEffect,
          evidence: {
            ...calculatedTrainingEffect.evidence,
            anaerobic: {
              ...calculatedTrainingEffect.evidence.anaerobic,
              sprintEffortCount: 2,
              sprintStimulus: 1.2,
            },
          },
        },
      }),
    ),
    'anaerobic capacity',
  )
})

test('classifies sustained calculated aerobic work from recorded power zones', () => {
  const calculatedTrainingEffect = {
    aerobic: 3.8,
    anaerobic: 1.1,
    evidence: {
      aerobic: { source: 'relative-effort' as const, load: 80 },
      anaerobic: { source: 'heart-rate' as const, seconds: 0 },
    },
  }
  const trainingRide = (powerZones: number[], durationS: number, averageWatts: number) =>
    detail({
      movingTimeS: 3_600,
      npWatts: 300,
      calculatedIntensityFactor: { source: 'power', value: 1 },
      powerZones,
      bestEfforts: {
        weightKg: null,
        weightDate: null,
        distance: [],
        climbs: [],
        power: [
          { durationS, averageWatts, wattsPerKg: null, averageHeartRate: null, elevationDeltaM: 0 },
        ],
      },
      calculatedTrainingEffect,
    })
  assert.equal(
    activityTrainingEffectLabel(trainingRide([0, 2_400, 300, 300, 600, 0, 0], 300, 330)),
    'VO2max',
  )
  assert.equal(
    activityTrainingEffectLabel(trainingRide([0, 2_400, 300, 900, 0, 0, 0], 1_200, 285)),
    'threshold',
  )
  assert.equal(
    activityTrainingEffectLabel(trainingRide([0, 2_400, 1_200, 0, 0, 0, 0], 1_800, 240)),
    'tempo',
  )
  assert.equal(
    activityTrainingEffectLabel(trainingRide([0, 2_400, 300, 300, 600, 0, 0], 20, 500)),
    'base',
  )
  assert.equal(
    activityTrainingEffectLabel(trainingRide([0, 3_600, 0, 0, 0, 0, 0], 1_800, 240)),
    'base',
  )
})

test('uses recovery for a calculated easy session with a small aerobic effect', () => {
  const easy = detail({
    calculatedIntensityFactor: { source: 'power', value: 0.52 },
    calculatedTrainingEffect: {
      aerobic: 1.4,
      anaerobic: 0,
      evidence: {
        aerobic: { source: 'relative-effort', load: 12 },
        anaerobic: { source: 'heart-rate', seconds: 0 },
      },
    },
  })
  assert.equal(activityTrainingEffectLabel(easy), 'recovery')
  assert.equal(
    activityTrainingEffectLabel(detail({ ...easy, calculatedIntensityFactor: null })),
    'base',
  )
})

test('labels sauna as recovery while retaining Garmin training effect scores', () => {
  for (const trainingEffectLabel of [null, 'UNKNOWN', 'AEROBIC_BASE', 'VO2_MAX']) {
    const garmin = garminVerification({
      trainingEffectLabel,
      aerobicTrainingEffect: 0.8,
      anaerobicTrainingEffect: 0,
      exerciseLoad: 14,
    })
    const rendered = buildActivity(factory, detail({ sport: 'sauna', route: [], garmin }), true)
    const summary = byTag(rendered, 'tr').find(
      row => row.properties.dataStatKey === 'training effect',
    )
    assert.ok(summary)
    assert.equal(text(summary), 'training effectrecovery')
    assert.equal(summary.properties.dataTrainingEffectGroup, 'low-aerobic')
    const scores = byClass(rendered, 'tri-training-effect')[0]
    assert.equal(scores.properties.dataTrainingEffectSource, 'garmin')
    assert.deepEqual(byClass(scores, 'tri-training-effect-score').map(text), ['0.8', '0.0'])
    assert.equal(garmin.trainingEffectLabel, trainingEffectLabel)
  }
})

test('summarizes calculated intensity when Garmin does not provide it', () => {
  assert.deepEqual(
    activityStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({ sport: 'run', calculatedIntensityFactor: { value: 0.909, source: 'pace' } }),
    ).slice(-2),
    [
      ['intensity factor', '0.909'],
      ['training effect', 'base'],
    ],
  )
  assert.deepEqual(
    activityStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({
        sport: 'strength',
        route: [],
        bestEfforts: null,
        calculatedIntensityFactor: { value: 0.798, source: 'heart-rate' },
      }),
    ).slice(-2),
    [
      ['intensity factor', '0.798'],
      ['training effect', 'recovery'],
    ],
  )
  assert.deepEqual(
    activityStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({
        garmin: garminVerification({ intensityFactor: 0.803 }),
        calculatedIntensityFactor: { value: 0.727, source: 'power' },
      }),
    ).slice(-2),
    [
      ['intensity factor', '0.803'],
      ['training effect', 'base'],
    ],
  )
})

test('renders Garmin training effect scores and notes immediately above heart rate zones', () => {
  const garmin = garminVerification({
    aerobicTrainingEffect: 4.5,
    anaerobicTrainingEffect: 2.7,
    trainingEffectLabel: 'VO2_MAX',
    aerobicTrainingEffectMessage: 'HIGHLY_IMPROVING_VO2_MAX_11',
    anaerobicTrainingEffectMessage: 'MAINTAINING_FAST_FORCE_PRODUCTION_6',
  })
  const activity = detail({ garmin, hrZones: [300, 600, 900, 120, 30] })
  const rendered = buildActivity(factory, activity, true, ctx())
  const details = byClass(rendered, 'tri-training-effect')[0]
  assert.ok(details)
  assert.equal(details.properties.dataTrainingEffectSource, 'garmin')
  assert.deepEqual(byClass(details, 'tri-training-effect-label').map(text), [
    'aerobic',
    'anaerobic',
  ])
  assert.deepEqual(byClass(details, 'tri-training-effect-score').map(text), ['4.5', '2.7'])
  assert.deepEqual(byClass(details, 'tri-training-effect-note').map(text), [
    'highly improving VO2max',
    'maintaining fast force production',
  ])
  const items = byClass(details, 'tri-training-effect-item')
  assert.deepEqual(
    items.map(item => item.properties.dataTrainingEffectGroup),
    ['high-aerobic', 'anaerobic'],
  )
  assert.deepEqual(
    byClass(details, 'tri-training-effect-meter').map(meter => [
      meter.properties.role,
      meter.properties.ariaValueMin,
      meter.properties.ariaValueMax,
      meter.properties.ariaValueNow,
    ]),
    [
      ['meter', 0, 5, 4.5],
      ['meter', 0, 5, 2.7],
    ],
  )
  assert.deepEqual(
    byClass(details, 'tri-training-effect-meter-fill').map(fill => fill.properties.style),
    ['--tri-training-effect-progress:90.0%', '--tri-training-effect-progress:54.0%'],
  )
  const baseDetails = buildTrainingEffectDetails(
    factory,
    detail({
      garmin: garminVerification({
        aerobicTrainingEffect: 3.2,
        trainingEffectLabel: 'AEROBIC_BASE',
        aerobicTrainingEffectMessage: 'IMPROVING_AEROBIC_BASE_8',
      }),
    }),
  )
  assert.ok(baseDetails)
  assert.equal(
    byClass(baseDetails, 'tri-training-effect-item')[0].properties.dataTrainingEffectGroup,
    'low-aerobic',
  )
  const more = byClass(rendered, 'tri-act-more')[0]
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const detailIndex = children.findIndex(child => classNames(child).includes('tri-training-effect'))
  const zonesIndex = children.findIndex(child =>
    byClass(child, 'tri-zone-title').some(title => text(title) === 'heart rate zones'),
  )
  assert.equal(detailIndex + 1, zonesIndex)
  assert.equal(formatTrainingEffectLabel('AEROBIC_BASE'), 'base')
  assert.equal(formatTrainingEffectLabel('LACTATE_THRESHOLD'), 'threshold')
  assert.equal(formatTrainingEffectLabel('VO2_MAX'), 'VO2max')
  assert.equal(formatTrainingEffectLabel('SPEED'), 'sprint')
  assert.equal(formatTrainingEffectLabel('ANAEROBIC_CAPACITY'), 'anaerobic capacity')
  assert.equal(dominantTrainingEffectGroup('RECOVERY'), 'low-aerobic')
  assert.equal(dominantTrainingEffectGroup('LACTATE_THRESHOLD'), 'high-aerobic')
  assert.equal(dominantTrainingEffectGroup('SPRINT'), 'anaerobic')
  assert.equal(formatTrainingEffectNote('NO_ANAEROBIC_BENEFIT_0'), 'no anaerobic benefit')
  assert.ok(buildTrainingEffectDetails(factory, activity))

  const fallbackDetails = buildTrainingEffectDetails(
    factory,
    detail({
      garmin: garminVerification({ aerobicTrainingEffect: 3, anaerobicTrainingEffect: 1.3 }),
    }),
  )
  assert.ok(fallbackDetails)
  assert.deepEqual(byClass(fallbackDetails, 'tri-training-effect-note').map(text), [
    'improving aerobic fitness',
    'minor anaerobic benefit',
  ])

  const calculatedDetails = buildTrainingEffectDetails(
    factory,
    detail({
      sport: 'yoga',
      calculatedTrainingEffect: {
        aerobic: 3.4,
        anaerobic: 1.2,
        evidence: {
          aerobic: { source: 'relative-effort', load: 42 },
          anaerobic: {
            source: 'power',
            effect: 1.2,
            effortCount: 8,
            stimulus: 1.14,
            criticalPowerWatts: 248.9,
            wPrimeKilojoules: 10.3,
          },
        },
      },
    }),
  )
  assert.ok(calculatedDetails)
  assert.equal(calculatedDetails.properties.dataTrainingEffectSource, 'calculated')
  const calculatedTitle = byClass(calculatedDetails, 'tri-zone-title')[0]
  assert.equal(text(calculatedTitle), 'training effect')
  assert.equal(calculatedTitle.properties.dataGloss, '')
  assert.equal(calculatedTitle.properties.tabIndex, 0)
  assert.equal(
    calculatedTitle.properties.dataGlossDef,
    'Calculated estimate. Aerobic 3.4 uses relative effort 42.0. Anaerobic 1.2 uses 8 short power efforts above eCP 248.9 W and eW′ 10.3 kJ. The estimate spreads those efforts across 1h20m of moving time.',
  )
  assert.deepEqual(byClass(calculatedDetails, 'tri-training-effect-score').map(text), [
    '3.4',
    '1.2',
  ])
  assert.deepEqual(byClass(calculatedDetails, 'tri-training-effect-note').map(text), [
    'improving aerobic fitness',
    'minor anaerobic benefit',
  ])
  assert.deepEqual(
    byClass(calculatedDetails, 'tri-training-effect-meter-fill').map(fill => fill.properties.style),
    ['--tri-training-effect-progress:68.0%', '--tri-training-effect-progress:24.0%'],
  )
  const frenchCalculatedDetails = buildTrainingEffectDetails(
    factoryFor(frenchPresentation),
    detail({
      sport: 'swim',
      calculatedTrainingEffect: {
        aerobic: 2.4,
        anaerobic: 0.4,
        evidence: {
          aerobic: { source: 'exercise-load', load: 25.5 },
          anaerobic: { source: 'pace', weightedSeconds: 72 },
        },
      },
    }),
  )
  assert.ok(frenchCalculatedDetails)
  assert.equal(frenchCalculatedDetails.properties.ariaLabel, "effet d'entraînement")
  const frenchCalculatedTitle = byClass(frenchCalculatedDetails, 'tri-zone-title')[0]
  assert.equal(text(frenchCalculatedTitle), "effet d'entraînement")
  assert.equal(
    frenchCalculatedTitle.properties.dataGlossDef,
    "Estimation calculée. Aérobie 2,4 à partir de la charge d'exercice 25,5. Anaérobie 0,4 à partir de 1m12s d'efforts brefs pondérés par l'allure.",
  )

  const nativeDetails = buildTrainingEffectDetails(
    factory,
    detail({
      garmin,
      calculatedTrainingEffect: {
        aerobic: 1.1,
        anaerobic: 4.9,
        evidence: {
          aerobic: { source: 'relative-effort', load: 10 },
          anaerobic: { source: 'heart-rate', seconds: 600 },
        },
      },
    }),
  )
  assert.ok(nativeDetails)
  assert.equal(nativeDetails.properties.dataTrainingEffectSource, 'garmin')
  assert.equal(byClass(nativeDetails, 'tri-zone-title')[0].properties.dataGloss, undefined)
  assert.deepEqual(byClass(nativeDetails, 'tri-training-effect-score').map(text), ['4.5', '2.7'])
})

test('renders strength volume, totals, exercises, and exact loaded sets', () => {
  const strength: NonNullable<StravaActivityDetail['strength']> = {
    volumeKg: 816.512,
    totalSets: 15,
    totalReps: 90,
    exercises: [
      {
        name: 'Press Up Position Walk Out',
        setCount: 2,
        repetitions: 20,
        durationS: null,
        sets: [],
      },
      {
        name: 'KB Straight Leg Deadlift',
        setCount: 2,
        repetitions: 20,
        durationS: null,
        sets: [
          { repetitions: 10, durationS: null, weightKg: 22.68 },
          { repetitions: 10, durationS: null, weightKg: 22.68 },
        ],
      },
    ],
    source: 'manual',
  }
  const activity = detail({
    sport: 'strength',
    movingTimeS: 533,
    avgHr: 101,
    windKph: 18,
    windDir: 'SW',
    windGustKph: 31,
    strength,
    route: [],
    bestEfforts: null,
  })
  assert.deepEqual(activityStatRows(imperialPresentation, activity), [
    ['time', "9'"],
    ['volume', '1,800.1 lb'],
    ['sets', '15'],
    ['reps', '90'],
    ['avg hr', '101 bpm'],
    ['training effect', 'recovery'],
  ])
  const rendered = buildActivity(factoryFor(imperialPresentation), activity)
  assert.deepEqual(byClass(rendered, 'tri-strength-exercise-name').map(text), [
    'Press Up Position Walk Out',
    'KB Straight Leg Deadlift',
  ])
  assert.deepEqual(byClass(rendered, 'tri-strength-exercise-summary').map(text), [
    '2 sets · 20 reps',
    '2 sets · 10 reps @ 50 lb each',
  ])

  assert.equal(activityStatRows(METRIC_TRIATHLON_PRESENTATION, activity)[1][1], '816.5 kg')
  assert.deepEqual(moreStatRows(METRIC_TRIATHLON_PRESENTATION, activity).slice(-3), [
    ['device temp', '24°C'],
    ['ambient temp', '22°C'],
    ['wind', '18 km/h SW / gust 31'],
  ])
})

test('places sauna exercises in one expanded block after activity graphs', () => {
  const parsed = parseTrackingBlock(
    null,
    [
      'date: 2026-09-15',
      'activity: sauna',
      'time: 18:30',
      'duration: 71m',
      'temperature: 73C',
      'humidity: 11%',
      'cooldown: cold plunge',
      'exercise: Glute Bridge | 30s | 30s',
      'exercise: Figure-4 Glute Stretch | 40s | 40s | 40s | 40s',
    ].join('\n'),
  )
  assert.ok(parsed?.strength)
  const recordedStrength: NonNullable<StravaActivityDetail['strength']> = {
    ...parsed.strength,
    source: 'manual',
  }
  for (const expanded of [false, true]) {
    for (const embedded of [false, true]) {
      const activity = detail({
        sport: 'sauna',
        route: [],
        heartRateTrace: [heartRateTracePoint(0, 0, 90), heartRateTracePoint(0, 4_260, 100)],
        strength: recordedStrength,
      })
      const rendered = buildActivity(factory, activity, expanded, ctx(), false, embedded)
      const more = byClass(rendered, 'tri-act-more')[0]
      const exercises = byClass(rendered, 'tri-act-strength')
      assert.equal(exercises.length, 1)
      assert.ok(more.children.includes(exercises[0]))
      assert.equal(byClass(rendered, 'tri-act-figs--sauna').length, 0)
      assert.deepEqual(byClass(exercises[0], 'tri-strength-exercise-name').map(text), [
        'Glute Bridge',
        'Figure-4 Glute Stretch',
      ])
      assert.deepEqual(byClass(exercises[0], 'tri-strength-exercise-summary').map(text), [
        '2 sets · 30s each',
        '4 sets · 40s each',
      ])
      const workout = byClass(more, 'tri-workout-analysis')[0]
      assert.ok(workout)
      assert.ok(more.children.indexOf(workout) < more.children.indexOf(exercises[0]))
      assert.equal(byClass(rendered, 'tri-act-toggle')[0].properties.ariaExpanded, String(expanded))

      for (const strength of [null, { ...recordedStrength, exercises: [] }]) {
        const empty = buildActivity(
          factory,
          { ...activity, strength },
          expanded,
          ctx(),
          false,
          embedded,
        )
        assert.equal(byClass(empty, 'tri-act-strength').length, 0)
      }
    }
  }
})

test('renders activity moves with exact repetitions and per-side labels', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'treatment',
      route: [],
      moves: {
        source: 'manual',
        entries: [
          { name: 'Roll Eagle', sets: [{ repetitions: 4, perSide: true }] },
          { name: 'Roll Center', sets: [{ repetitions: 4, perSide: false }] },
          {
            name: 'Roll Hurdler',
            sets: [
              { repetitions: 4, perSide: true },
              { repetitions: 4, perSide: true },
            ],
          },
        ],
      },
    }),
  )

  assert.deepEqual(byClass(rendered, 'tri-move-name').map(text), [
    'Roll Eagle',
    'Roll Center',
    'Roll Hurdler',
  ])
  assert.deepEqual(byClass(rendered, 'tri-move-summary').map(text), [
    '1 set · 4 reps per side',
    '1 set · 4 reps',
    '2 sets · 4 reps per side each',
  ])
  assert.equal(text(byClass(rendered, 'tri-act-moves-h')[0]), 'moves')
})

test('renders route-less strength heart rate against elapsed time', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'strength',
      distanceKm: 0,
      movingTimeS: 1_560,
      route: [],
      heartRateTrace: [
        heartRateTracePoint(0, 0, 90),
        heartRateTracePoint(0, 520, 120),
        heartRateTracePoint(0, 1_040, null),
        heartRateTracePoint(0, 1_560, 140),
      ],
      bestEfforts: null,
    }),
    true,
  )
  const trace = byClass(rendered, 'tri-elev-wrap').find(
    element => element.properties.dataTriTrace === 'hr',
  )

  assert.ok(trace)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['80bpm', '100bpm', '120bpm', '140bpm'])
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['0s', '13:00', '26:00'])
  assert.match(String(byClass(trace, 'tri-elev-line')[0]?.properties.d), /^M 0 /)
  const graph = byTag(trace, 'svg')[0]
  assert.equal(graph?.properties.dataDomainStartElapsedS, 0)
  assert.equal(graph?.properties.dataDomainEndElapsedS, 1_560)
  assert.equal(graph?.properties.dataDomainStartDistanceKm, undefined)
})

test('renders parsed sauna location maps in compact, expanded, and embedded cards', () => {
  for (const name of ['Othership Adelaide', 'Othership Yorkville', 'Local sauna']) {
    const parsed = parseTrackingBlock(
      null,
      [
        'activity: sauna',
        'date: 2026-06-07',
        'time: 07:30',
        'duration: 75 mins',
        'temperature: 85C',
        'humidity: 11%',
        'cooldown: natural',
        `location: ${name}`,
      ].join('\n'),
    )
    assert.ok(parsed?.sauna?.location)
    const location = parsed.sauna.location
    const sauna = detail({
      sport: 'sauna',
      route: [],
      mapRoute: [],
      distanceKm: 0,
      elapsedTimeS: 4_500,
      heartRateTrace: [heartRateTracePoint(0, 0, 90), heartRateTracePoint(0, 4_500, 100)],
      analysisRanges: analysisRanges()
        .filter(range => range.kind === 'lap')
        .map((range, index) => ({
          ...range,
          id: `sauna:${index}`,
          startElapsedS: index * 1_000,
          endElapsedS: (index + 1) * 1_000,
          startDistanceKm: 0,
          endDistanceKm: 0,
          distanceKm: 0,
          durationS: 1_000,
        })),
      sauna: { ...parsed.sauna, heatTrainingLoad: 7.6, heartRateSource: null, source: 'manual' },
    })
    for (const expanded of [false, true]) {
      for (const embedded of [false, true]) {
        const rendered = buildActivity(factory, sauna, expanded, ctx(), false, embedded)
        const figure = byClass(rendered, 'tri-sauna-location')[0]
        assert.ok(figure)
        assert.equal(text(byClass(figure, 'tri-sauna-location-name')[0]), name)
        const summary = byClass(rendered, 'tri-sauna-summary')[0]
        const summaryDetails = byClass(summary, 'tri-sauna-summary-details')[0]
        assert.equal(byClass(rendered, 'tri-sauna-laps').length, 1)
        assert.equal(byClass(rendered, 'tri-sauna-laps-timeline').length, 0)
        const workout = byClass(rendered, 'tri-sauna-workout')[0]
        assert.ok(workout)
        assert.equal(byClass(workout, 'tri-elev').length, 1)
        assert.equal(byClass(workout, 'tri-sauna-laps').length, 0)
        const phaseLegend = byClass(summaryDetails, 'tri-sauna-laps-legend')[0]
        assert.deepEqual(byClass(phaseLegend, 'tri-sauna-phase').map(text), [
          'hot sauna',
          'cold plunge',
          'break',
        ])
        assert.equal(byClass(summaryDetails, 'tri-analysis-readout').length, 0)
        for (const [index, lap] of byClass(summaryDetails, 'tri-sauna-lap-legend').entries()) {
          assert.equal(text(byClass(lap, 'tri-sauna-phase')[0]), `lap ${index + 1}`)
          assert.equal(text(byClass(lap, 'tri-sauna-lap-duration')[0]), '16:40')
          assert.equal(lap.properties.dataSaunaPhase, 'hot sauna')
        }
        assert.equal(
          byClass(summaryDetails, 'tri-sauna-lap-legend').length,
          sauna.analysisRanges.length,
        )
        assert.equal(byClass(summaryDetails, 'tri-sauna-htl').length, 1)
        assert.equal(byClass(rendered, 'tri-sauna-htl').length, 1)
        assert.equal(byClass(summaryDetails, 'tri-sauna-htl-meter')[0].properties.ariaValueNow, 7.6)
        const maps = byClass(figure, 'tri-sauna-location-map')
        if (location.latitude == null || location.longitude == null) {
          assert.equal(maps.length, 0)
          assert.equal(byClass(figure, 'tri-sauna-location-pin').length, 0)
          continue
        }
        assert.equal(maps.length, 1)
        assert.equal(maps[0].tagName, 'a')
        assert.equal(
          maps[0].properties.href,
          `https://www.openstreetmap.org/?mlat=${location.latitude}&mlon=${location.longitude}#map=16/${location.latitude}/${location.longitude}`,
        )
        assert.equal(byClass(figure, 'tri-sauna-location-pin').length, 1)
        const images = byClass(figure, 'tri-sauna-location-image')
        assert.equal(images.length, 2)
        for (const [index, image] of images.entries()) {
          assert.equal(image.tagName, 'img')
          assert.equal(image.properties.loading, 'lazy')
          assert.equal(image.properties.dataNoPopover, 'true')
          assert.equal(
            image.properties.src,
            `/api/sauna-map?location=${encodeURIComponent(name)}&theme=${index === 0 ? 'light' : 'dark'}`,
          )
        }
        assert.equal(byClass(figure, 'tri-sauna-location-attribution').length, 1)
        assert.ok(byClass(rendered, 'tri-act-figs--sauna')[0].children.includes(summary))
      }
    }
    assert.equal(
      byClass(buildActivity(factory, { ...sauna, sauna: null }), 'tri-sauna-location').length,
      0,
    )
    assert.equal(
      byClass(buildActivity(factory, { ...sauna, sport: 'treatment' }), 'tri-sauna-location')
        .length,
      0,
    )
  }
})

test('renders manual sauna conditions and Oura heart rate without distance metrics', () => {
  const sauna = detail({
    sport: 'sauna',
    name: 'Untangle',
    distanceKm: 0,
    movingTimeS: 4_500,
    avgHr: 120,
    maxHr: 130,
    avgWatts: null,
    npWatts: null,
    maxWatts: null,
    kilojoules: null,
    deviceWatts: false,
    avgCadence: null,
    calories: null,
    deviceTemperatureC: 27,
    windKph: 10,
    windDir: 'NNW',
    windGustKph: 16,
    strength: null,
    sauna: {
      time: '18:30',
      temperatureC: 91.111,
      humidityPct: 11,
      cooldown: 'cold plunge',
      heatTrainingLoad: 7.7,
      heartRateSource: 'oura',
      source: 'manual',
    },
    calculatedTrainingEffect: {
      aerobic: 1.6,
      anaerobic: 0,
      evidence: {
        aerobic: { source: 'relative-effort', load: 8 },
        anaerobic: { source: 'heart-rate', seconds: 0 },
      },
    },
    route: [],
    heartRateTrace: [heartRateTracePoint(0, 300, 110), heartRateTracePoint(0, 3_600, 130)],
    bestEfforts: null,
  })

  assert.deepEqual(activityStatRows(imperialPresentation, sauna), [
    ['time', '18:30'],
    ['duration', "1h15'"],
    ['temperature', '196°F'],
    ['humidity', '11%'],
    ['avg hr', '120 bpm · Oura'],
    ['HTL', '7.7'],
    ['training effect', 'recovery'],
    ['cooldown', 'cold plunge'],
  ])
  assert.deepEqual(moreStatRows(imperialPresentation, sauna), [
    ['max hr', '130 bpm'],
    ['device temp', '81°F'],
    ['ambient temp', '72°F'],
    ['wind', '10 km/h NNW / gust 16'],
  ])
  const rendered = buildActivity(factoryFor(imperialPresentation), sauna, true)
  assert.equal(rendered.properties.dataActivityTitle, 'Untangle')
  const trainingEffect = byClass(rendered, 'tri-training-effect')[0]
  assert.ok(trainingEffect)
  assert.equal(trainingEffect.properties.dataTrainingEffectSource, 'calculated')
  assert.deepEqual(byClass(trainingEffect, 'tri-training-effect-score').map(text), ['1.6', '0.0'])
  assert.equal(
    byClass(rendered, 'tri-elev-wrap').find(element => element.properties.dataTriTrace === 'hr')
      ?.properties.dataTriTrace,
    'hr',
  )
  const withExercises: StravaActivityDetail = {
    ...sauna,
    strength: {
      volumeKg: null,
      totalSets: null,
      totalReps: null,
      source: 'manual',
      exercises: [
        { name: 'Glute Bridge', durations: [30, 30] },
        { name: 'Single-Leg Glute Bridge', durations: [40, 40, 40, 40] },
        { name: 'Single-Leg Glute Bridge to Abductor', durations: [40, 40, 40, 40] },
      ].map(({ name, durations }) => ({
        name,
        setCount: durations.length,
        durationS: durations.reduce((total, seconds) => total + seconds, 0),
        repetitions: null,
        sets: durations.map(durationS => ({ durationS, repetitions: null, weightKg: null })),
      })),
    },
  }
  const renderedExercises = buildActivity(factoryFor(imperialPresentation), withExercises, true)
  assert.deepEqual(byClass(renderedExercises, 'tri-strength-exercise-name').map(text), [
    'Glute Bridge',
    'Single-Leg Glute Bridge',
    'Single-Leg Glute Bridge to Abductor',
  ])
  assert.deepEqual(byClass(renderedExercises, 'tri-strength-exercise-summary').map(text), [
    '2 sets · 30s each',
    '4 sets · 40s each',
    '4 sets · 40s each',
  ])
  assert.deepEqual(
    activityStatRows(imperialPresentation, withExercises),
    activityStatRows(imperialPresentation, sauna),
  )
})

test('renders route-less treatment heart rate when samples are available', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'treatment',
      distanceKm: 0,
      movingTimeS: 1_291,
      route: [],
      heartRateTrace: [
        heartRateTracePoint(0, 0, null),
        heartRateTracePoint(0, 211, 60),
        heartRateTracePoint(0, 433, 62),
        heartRateTracePoint(0, 451, 67),
        heartRateTracePoint(0, 457, 64),
        heartRateTracePoint(0, 1_291, null),
      ],
      bestEfforts: null,
    }),
    true,
  )
  const trace = byClass(rendered, 'tri-elev-wrap').find(
    element => element.properties.dataTriTrace === 'hr',
  )

  assert.ok(trace)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), [
    '60bpm',
    '65bpm',
    '70bpm',
    '75bpm',
    '80bpm',
  ])
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['0s', '10:46', '21:31'])
  assert.match(String(byClass(trace, 'tri-elev-line')[0]?.properties.d), / 27\.10 /)
  assert.match(String(byClass(trace, 'tri-elev-line')[0]?.properties.d), / 19\.85 /)
  const graph = byTag(trace, 'svg')[0]
  assert.equal(graph?.properties.dataDomainStartElapsedS, 0)
  assert.equal(graph?.properties.dataDomainEndElapsedS, 1_291)
})

test('renders route-less yoga heart rate and CORE thermal traces against elapsed time', () => {
  const yoga = detail({
    sport: 'yoga',
    distanceKm: 0,
    movingTimeS: 1_560,
    avgHr: 87,
    route: [],
    heartRateTrace: [
      heartRateTracePoint(0, 0, 81),
      heartRateTracePoint(0, 520, 88, {
        heatStrainIndex: 0,
        heatStrainSource: 'core-app',
        coreTemperatureC: 37.05,
        coreTemperatureSource: 'core-app',
        skinTemperatureC: 34.2,
        skinTemperatureSource: 'core-app',
      }),
      heartRateTracePoint(0, 1_040, 90, {
        heatStrainIndex: 0,
        heatStrainSource: 'core-app',
        coreTemperatureC: 37.1,
        coreTemperatureSource: 'core-app',
        skinTemperatureC: 34.6,
        skinTemperatureSource: 'core-app',
      }),
      heartRateTracePoint(0, 1_560, 86, {
        heatStrainIndex: 0,
        heatStrainSource: 'core-app',
        coreTemperatureC: 37.14,
        coreTemperatureSource: 'core-app',
        skinTemperatureC: 34.9,
        skinTemperatureSource: 'core-app',
      }),
    ],
    bestEfforts: null,
  })

  const rendered = buildActivity(factory, yoga, true)
  const traces = byClass(rendered, 'tri-elev-wrap').filter(
    element => typeof element.properties.dataTriTrace === 'string',
  )

  assert.deepEqual(
    traces.map(trace => trace.properties.dataTriTrace),
    ['hr', 'heat-strain-index', 'core-temperature', 'skin-temperature'],
  )
  for (const trace of traces) {
    assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['0s', '13:00', '26:00'])
    const graph = byTag(trace, 'svg')[0]
    assert.equal(graph?.properties.dataDomainStartElapsedS, 0)
    assert.equal(graph?.properties.dataDomainEndElapsedS, 1_560)
  }
  assert.deepEqual(activityStatRows(METRIC_TRIATHLON_PRESENTATION, yoga).slice(0, 2), [
    ['time', "26'"],
    ['avg hr', '87 bpm'],
  ])
})

test('keeps inclusive cycling power by default and exposes the zero-excluded view on demand', () => {
  const source = detail({
    avgWatts: 150,
    npWatts: 210,
    powerZones: [20, 10],
    powerHist: [8, 4],
    powerWithoutZeros: { avgWatts: 200, powerZones: [12, 10], powerHist: [0, 4] },
  })

  assert.equal(powerViewActivity(METRIC_TRIATHLON_PRESENTATION, source), source)

  const filtered = powerViewActivity(excludeZeroPresentation, source)
  assert.notEqual(filtered, source)
  assert.equal(filtered.avgWatts, 200)
  assert.equal(filtered.npWatts, 210)
  assert.deepEqual(filtered.powerZones, [12, 10])
  assert.deepEqual(filtered.powerHist, [0, 4])
  assert.equal(powerViewActivity(excludeZeroPresentation, detail({ sport: 'run' })).sport, 'run')
})

test('interpolates omitted cycling samples on the existing distance axis', () => {
  const route = detail().route.map((point, index) => ({
    ...point,
    d: [0, 5, 20, 30][index],
    w: [100, 0, 300, 0][index],
  }))

  assert.deepEqual(
    interpolatePositiveMetricSeries(route, point => point.w),
    [100, 150, 300, 300],
  )
})

test('normalizes zero-excluded bike power and cadence traces', () => {
  const route = detail().route.map((point, index) => ({
    ...point,
    w: [100, 0, 0, 400][index],
    cad: [80, 0, 0, 110][index],
  }))

  const inclusive = buildActivity(factory, detail({ route }), true)
  const inclusivePower = byClass(inclusive, 'tri-elev-wrap').find(
    graph => graph.properties.dataTriTrace === 'power',
  )
  assert.ok(inclusivePower)
  assert.match(
    String(byClass(inclusivePower, 'tri-elev-line')[0].properties.d),
    /L 33\.33 30\.00 L 66\.67 30\.00/,
  )

  const normalized = buildActivity(factoryFor(excludeZeroPresentation), detail({ route }), true)
  const power = byClass(normalized, 'tri-elev-wrap').find(
    graph => graph.properties.dataTriTrace === 'power',
  )
  const cadence = byClass(normalized, 'tri-elev-wrap').find(
    graph => graph.properties.dataTriTrace === 'cadence',
  )
  assert.ok(power)
  assert.ok(cadence)
  assert.deepEqual(byClass(power, 'tri-cax-yt').map(text), ['100w', '200w', '300w', '400w'])
  assert.deepEqual(byClass(cadence, 'tri-cax-yt').map(text), ['80rpm', '90rpm', '100rpm', '110rpm'])
  assert.match(
    String(byClass(power, 'tri-elev-line')[0].properties.d),
    /M 0 30\.00 L 33\.33 20\.33 L 66\.67 10\.67 L 100\.00 1\.00/,
  )
  assert.match(
    String(byClass(cadence, 'tri-elev-line')[0].properties.d),
    /M 0 30\.00 L 33\.33 20\.33 L 66\.67 10\.67 L 100\.00 1\.00/,
  )
})

const analysisRanges = (): ActivityAnalysisRange[] => [
  {
    kind: 'segment',
    id: 'segment-boardwalk',
    label: 'Boardwalk east',
    startElapsedS: 2_400,
    endElapsedS: 3_600,
    startDistanceKm: 15,
    endDistanceKm: 22.5,
    durationS: 1_200,
    distanceKm: 7.5,
    elevationGainM: 8,
    averageSpeedKph: 22.5,
    averageHeartRate: 154,
    averageWatts: 205,
    averageCadence: 89,
  },
  {
    kind: 'climb',
    source: 'garmin-climbpro',
    id: 'climb-bay',
    label: 'Bay rise',
    startElapsedS: 800,
    endElapsedS: 1_600,
    startDistanceKm: 5,
    endDistanceKm: 10,
    durationS: 800,
    distanceKm: 5,
    elevationGainM: 48,
    averageSpeedKph: 22.5,
    averageHeartRate: 151,
    averageWatts: 232,
    averageCadence: 84,
  },
  {
    kind: 'lap',
    id: 'lap-2',
    label: 'Lap 2',
    startElapsedS: 1_200,
    endElapsedS: 2_400,
    startDistanceKm: 7.5,
    endDistanceKm: 15,
    durationS: 1_200,
    distanceKm: 7.5,
    elevationGainM: 24,
    averageSpeedKph: 22.5,
    averageHeartRate: 149,
    averageWatts: 214,
    averageCadence: 87,
  },
]

const analysisDetail = (): StravaActivityDetail =>
  detail({
    analysisRanges: analysisRanges(),
    mapRoute: [
      [
        { lat: 43.6, lng: -79.4, d: 0 },
        { lat: 43.63, lng: -79.37, d: 3.75 },
        { lat: 43.66, lng: -79.34, d: 7.5 },
        { lat: 43.69, lng: -79.31, d: 11.25 },
        { lat: 43.72, lng: -79.28, d: 15 },
        { lat: 43.75, lng: -79.25, d: 18.75 },
        { lat: 43.78, lng: -79.22, d: 22.5 },
        { lat: 43.81, lng: -79.19, d: 26.25 },
        { lat: 43.84, lng: -79.16, d: 30 },
      ],
    ],
  })

test('renders per-run best efforts with pace, telemetry gaps, and calculation provenance', () => {
  const run = detail({
    sport: 'run',
    bestEfforts: {
      distanceSource: 'calculated',
      weightKg: null,
      weightDate: null,
      power: [],
      climbs: [],
      distance: [
        {
          label: '400m',
          targetDistanceM: 400,
          elapsedTimeS: 114,
          averageSpeedKph: (400 / 114) * 3.6,
          averageHeartRate: 162,
          elevationDeltaM: -1.2,
        },
        {
          label: '1K',
          targetDistanceM: 1000,
          elapsedTimeS: 343,
          averageSpeedKph: (1000 / 343) * 3.6,
          averageHeartRate: null,
          elevationDeltaM: null,
        },
      ],
    },
  })
  const rendered = buildBestEfforts(factory, run)
  assert.ok(rendered)
  assert.equal(rendered.properties.ariaLabel, 'Running best efforts')
  assert.equal(rendered.properties.dataEffortSource, 'calculated')
  assert.deepEqual(headerText(table(rendered, 'distance')), [
    'distance',
    'time',
    'pace',
    'heart rate',
    'elev',
  ])
  assert.deepEqual(bodyRows(table(rendered, 'distance')), [
    ['400m', '1:54', '4:45/km', '162 bpm', '-1 m'],
    ['1K', '5:43', '5:43/km', '—', '—'],
  ])
  assert.equal(byClass(rendered, 'tri-effort-block').length, 1)
  assert.match(
    text(rendered),
    /Calculated from recorded distance and elapsed time, including pauses\./,
  )
  const imperial = buildBestEfforts(factoryFor(imperialPresentation), run)
  assert.ok(imperial)
  assert.deepEqual(bodyRows(table(imperial, 'distance'))[0], [
    '400m',
    '1:54',
    '7:39/mi',
    '162 bpm',
    '-4 ft',
  ])
  const day = buildDayCard(factory, run.date, { details: { [run.id]: run }, health: {} })
  assert.equal(byClass(day, 'tri-efforts').length, 1)
  assert.equal(buildBestEfforts(factory, detail({ sport: 'run', bestEfforts: null })), null)
  assert.equal(buildBestEfforts(factory, detail({ sport: 'walk' })), null)
})

test('builds semantic distance, power, and climbing tables in metric units', () => {
  const rendered = buildBestEfforts(factory, detail())
  assert.ok(rendered)

  assert.equal(rendered.tagName, 'section')
  assert.equal(rendered.properties.ariaLabel, 'Cycling best efforts')
  assert.equal(byTag(rendered, 'caption').length, 0)
  assert.deepEqual(
    byClass(rendered, 'tri-effort-title').map(title => [title.tagName, text(title)]),
    [
      ['div', 'distance'],
      ['div', 'power'],
      ['div', 'climbpro'],
    ],
  )
  assert.deepEqual(
    byClass(rendered, 'tri-effort-block').map(block => block.tagName),
    ['div', 'div', 'div'],
  )
  assert.equal(byClass(rendered, 'tri-effort-viewport').length, 3)
  for (const scroll of byClass(rendered, 'tri-effort-scroll'))
    assert.equal(byClass(scroll, 'tri-effort-title').length, 0)
  assert.deepEqual(
    byClass(rendered, 'tri-effort-scroll').map(scroll => [
      scroll.properties.role,
      scroll.properties.ariaLabel,
      scroll.properties.tabIndex,
    ]),
    [
      ['region', 'Distance efforts', 0],
      ['region', 'Power efforts', 0],
      ['region', 'ClimbPro efforts', 0],
    ],
  )

  const distance = table(rendered, 'distance')
  assert.equal(distance.properties.ariaLabel, 'Distance efforts')
  assert.deepEqual(headerText(distance), ['distance', 'time', 'speed', 'heart rate', 'elev'])
  assert.deepEqual(bodyRows(distance), [['10K', '24:31', '24.5 km/h', '151 bpm', '-30 m']])

  const power = table(rendered, 'power')
  assert.deepEqual(headerText(power), [
    'time',
    'power',
    'w/kg',
    'heart rate',
    'cadence',
    'torque',
    'coverage',
    'elev',
  ])
  assert.deepEqual(bodyRows(power), [
    ['5 sec', '565 W', '6.45 W/kg', '150 bpm', '—', '—', '—', '4 m'],
  ])

  const climbing = table(rendered, 'climbing')
  assert.deepEqual(headerText(climbing), [
    'climb',
    'time',
    'distance',
    'gain',
    'grade',
    'speed',
    'heart rate',
    'power',
    'w/kg',
    'vam',
  ])
  assert.deepEqual(bodyRows(climbing), [
    [
      'Snake Road',
      '8:00',
      '2.50 km',
      '120 m',
      '4.8%',
      '18.8 km/h',
      '155 bpm',
      '240 W',
      '2.74 W/kg',
      '900 m/h',
    ],
  ])

  const note = byClass(rendered, 'tri-effort-note')[0]
  assert.ok(note)
  assert.equal(text(note), 'W/kg from 87.55 kg Garmin weight · Jul 9')
  for (const heading of byTag(rendered, 'thead').flatMap(head => byTag(head, 'th')))
    assert.equal(heading.properties.scope, 'col')
  for (const body of byTag(rendered, 'tbody')) {
    const rowHeading = byTag(body, 'th')[0]
    assert.ok(rowHeading)
    assert.equal(rowHeading.properties.scope, 'row')
  }
})

test('renders cycling efforts in the expanded shared activity section', () => {
  const rendered = buildActivity(factory, detail(), true)
  assert.equal(byClass(rendered, 'tri-act--expanded').length, 1)
  assert.equal(byClass(rendered, 'tri-act-more').length, 1)
  assert.equal(byClass(rendered, 'tri-efforts').length, 1)
})

test('renders route stream graphs in the server activity markup', () => {
  const rendered = buildDayCard(
    factory,
    '2026-07-09',
    { details: { 101: detail() }, health: {} },
    { expanded: true },
  )
  const activity = byClass(rendered, 'tri-act')[0]
  assert.ok(activity)
  assert.equal(activity.properties.dataActivityId, '101')
  assert.equal(activity.properties.dataActivityTitle, 'Threshold ride')
  const traces = byClass(rendered, 'tri-elev-wrap').filter(
    graph => graph.properties.dataTriTrace != null,
  )
  assert.deepEqual(
    traces.map(graph => graph.properties.dataTriTrace),
    ['hr', 'temperature', 'cadence', 'respiration', 'power', 'speed'],
  )
  for (const graph of traces) {
    assert.equal(byClass(graph, 'tri-elev').length, 1)
    assert.equal(byClass(graph, 'tri-elev-area').length, 1)
    assert.equal(byClass(graph, 'tri-elev-line').length, 1)
    assert.equal(byClass(graph, 'tri-analysis-selection').length, 1)
  }
  assert.equal(byClass(activity, 'tri-analysis-selection').length, 7)
  const respiration = traces.find(graph => graph.properties.dataTriTrace === 'respiration')
  assert.ok(respiration)
  assert.deepEqual(
    byClass(respiration, 'tri-elev-cap')
      .flatMap(cap => byTag(cap, 'span'))
      .map(text),
    ['respiration', '26.0 brpm avg'],
  )
  assert.deepEqual(byClass(respiration, 'tri-cax-yt').map(text), ['20brpm', '30brpm'])
  const temperature = traces.find(graph => graph.properties.dataTriTrace === 'temperature')
  assert.ok(temperature)
  assert.deepEqual(
    byClass(temperature, 'tri-elev-cap')
      .flatMap(cap => byTag(cap, 'span'))
      .map(text),
    ['temperature', '22°C avg'],
  )
  assert.deepEqual(byClass(temperature, 'tri-cax-yt').map(text), ['22°C', '24°C', '26°C'])
})

test('renders CORE bike graphs before heart rate with sub-degree domains', () => {
  const thermal = detail({
    route: detail().route.map((point, index) => ({
      ...point,
      heatStrainIndex: [0, 1.4, 3, 3.1][index],
      heatStrainSource: 'core-fit',
      coreTemperatureC: [37.16, 37.17, 37.19, 37.18][index],
      coreTemperatureSource: 'core-fit',
      skinTemperatureC: [33.4, 33.45, 33.5, 33.55][index],
      skinTemperatureSource: 'core-fit',
    })),
  })
  const rendered = buildActivity(factory, thermal, true)
  const traces = byClass(rendered, 'tri-elev-wrap').filter(
    graph => graph.properties.dataTriTrace != null,
  )
  assert.deepEqual(
    traces.map(graph => graph.properties.dataTriTrace),
    [
      'heat-strain-index',
      'core-temperature',
      'skin-temperature',
      'hr',
      'temperature',
      'cadence',
      'respiration',
      'power',
      'speed',
    ],
  )

  const coreTemperature = traces.find(graph => graph.properties.dataTriTrace === 'core-temperature')
  assert.ok(coreTemperature)
  assert.deepEqual(byClass(coreTemperature, 'tri-cax-yt').map(text), [
    '37.16°C',
    '37.18°C',
    '37.20°C',
  ])
  assert.match(text(byClass(coreTemperature, 'tri-elev-cap')[0]), /37\.17°C avg/)

  const skinTemperature = traces.find(graph => graph.properties.dataTriTrace === 'skin-temperature')
  assert.ok(skinTemperature)
  assert.ok(
    byClass(skinTemperature, 'tri-cax-yt')
      .map(text)
      .every(label => /^\d+\.\d{2}°C$/.test(label)),
  )
})

test('renders cycling speed in initial activity HTML with metric and imperial units', () => {
  const bike = detail({
    route: detail().route.map((point, index) => ({ ...point, speedKph: index * 12 })),
  })
  for (const [presentation, peak, unit] of [
    [METRIC_TRIATHLON_PRESENTATION, '36.0 km/h peak', 'km/h'],
    [imperialPresentation, '22.4 mph peak', 'mph'],
  ] as const) {
    const rendered = buildActivity(factoryFor(presentation), bike, false, undefined, false, true)
    const traces = byClass(byClass(rendered, 'tri-act-more')[0], 'tri-elev-wrap')
    const speedIndex = traces.findIndex(trace => trace.properties.dataTriTrace === 'speed')
    const speed = traces[speedIndex]
    assert.ok(speed)
    assert.equal(traces[speedIndex - 1].properties.dataTriTrace, 'power')
    assert.equal(speed.properties.dataSpeedSource, 'distance-time')
    assert.equal(byClass(speed, 'tri-elev-d').map(text).join(''), 'speed')
    assert.equal(byClass(speed, 'tri-elev-range').map(text).join(''), peak)
    assert.ok(
      byClass(speed, 'tri-cax-yt')
        .slice(1)
        .every(tick => text(tick).endsWith(unit)),
    )
    assert.match(String(byClass(speed, 'tri-elev-line')[0].properties.d), /^M 0 30\.00 L /)
    assert.equal(byClass(speed, 'tri-analysis-selection').length, 1)
  }
  const filtered = buildActivity(factory, bike, false, undefined, false, true, { speed: false })
  assert.equal(
    byClass(filtered, 'tri-elev-wrap').some(trace => trace.properties.dataTriTrace === 'speed'),
    false,
  )
  for (const sport of ['run', 'walk', 'swim'] as const)
    assert.equal(
      byClass(buildActivity(factory, { ...bike, sport }), 'tri-elev-wrap').some(
        trace => trace.properties.dataTriTrace === 'speed',
      ),
      false,
    )
})

test('cycling speed preserves stops, breaks at invalid samples, and omits unavailable traces', () => {
  const traceFor = (speeds: number[]) =>
    byClass(
      buildActivity(
        factory,
        detail({
          route: detail().route.map((point, index) => ({ ...point, speedKph: speeds[index] })),
        }),
      ),
      'tri-elev-wrap',
    ).find(trace => trace.properties.dataTriTrace === 'speed')
  const stopped = traceFor([0, 0, 0, 0])
  assert.ok(stopped)
  assert.equal(byClass(stopped, 'tri-elev-range').map(text).join(''), '0.0 km/h peak')
  const gaps = traceFor([12, Number.NaN, -1, 24])
  assert.ok(gaps)
  assert.equal(String(byClass(gaps, 'tri-elev-line')[0].properties.d).match(/M /g)?.length, 2)
  assert.equal(traceFor([12, Number.NaN, -1, Number.POSITIVE_INFINITY]), undefined)
})

test('renders muscle oxygen as a percentage trace', () => {
  const oxygen = detail({
    route: detail().route.map((point, index) => ({
      ...point,
      muscleOxygenPct: [64, 62, 60, 58][index],
    })),
  })
  const rendered = buildActivity(factory, oxygen, true)
  const trace = byClass(rendered, 'tri-elev-wrap').find(
    graph => graph.properties.dataTriTrace === 'muscle-oxygen',
  )
  assert.ok(trace)
  assert.match(text(byClass(trace, 'tri-elev-cap')[0]), /muscle oxygen61\.0% \\mathrm\{SmO\}_2 avg/)
  assert.equal(byClass(trace, 'tri-math').length, 1)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['55.0%', '60.0%', '65.0%'])
})

test('keeps environment after the full-width activity traces', () => {
  const activity = detail({
    analyses: environmentAnalyses(),
    garmin: garminVerification({ aerobicTrainingEffect: 3.2, anaerobicTrainingEffect: 1.1 }),
    route: detail().route.map((point, index) => ({
      ...point,
      muscleOxygenPct: [64, 62, 60, 58][index],
      heatStrainIndex: [0, 1.4, 3, 3.1][index],
      heatStrainSource: 'core-fit',
      coreTemperatureC: [37.16, 37.17, 37.19, 37.18][index],
      coreTemperatureSource: 'core-fit',
      skinTemperatureC: [33.4, 33.45, 33.5, 33.55][index],
      skinTemperatureSource: 'core-fit',
    })),
  })
  const rendered = buildActivity(factory, activity, true)
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.equal(byClass(more, 'tri-environment-layout').length, 0)
  assert.equal(byClass(more, 'tri-environment-traces').length, 0)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  assert.deepEqual(
    children
      .filter(child => typeof child.properties.dataTriTrace === 'string')
      .slice(-3)
      .map(child => child.properties.dataTriTrace),
    ['power', 'speed', 'muscle-oxygen'],
  )
  const environment = children.find(child => classNames(child).includes('tri-environment'))
  assert.ok(environment)
  const environmentIndex = children.indexOf(environment)
  assert.equal(children[environmentIndex - 1].properties.dataTriTrace, 'muscle-oxygen')
  assert.ok(children.indexOf(byClass(more, 'tri-training-effect')[0]) < environmentIndex)
})

test('renders timestamp-aligned CORE app graphs for runs without Garmin thermal data', () => {
  const thermal = detail({
    sport: 'run',
    deviceWatts: false,
    garmin: null,
    route: detail().route.map((point, index) => ({
      ...point,
      w: 0,
      resp: null,
      tempC: null,
      heatStrainIndex: [0, 0.4, 0.8, 1.2][index],
      heatStrainSource: 'core-app',
      coreTemperatureC: [37.32, 37.61, 37.97, 38.33][index],
      coreTemperatureSource: 'core-app',
      skinTemperatureC: [32.54, 32.4, 32.21, 32.04][index],
      skinTemperatureSource: 'core-app',
    })),
  })
  const rendered = buildActivity(factory, thermal, true)
  const thermalTraces = byClass(rendered, 'tri-elev-wrap')
    .filter(graph => graph.properties.dataTriTrace != null)
    .filter(graph =>
      ['heat-strain-index', 'core-temperature', 'skin-temperature'].includes(
        String(graph.properties.dataTriTrace),
      ),
    )

  assert.deepEqual(
    thermalTraces.map(graph => graph.properties.dataTriTrace),
    ['heat-strain-index', 'core-temperature', 'skin-temperature'],
  )
  assert.match(text(byClass(thermalTraces[1], 'tri-elev-cap')[0]), /37\.81°C avg/)
  assert.match(text(byClass(thermalTraces[2], 'tri-elev-cap')[0]), /32\.30°C avg/)
})

test('connects every missing heat strain range with dotted straight lines', () => {
  const source = detail()
  const route = Array.from({ length: 7 }, (_, index) => {
    const heatStrainIndex = [null, 1.4, null, null, 3, null, null][index]
    return {
      ...source.route[Math.min(index, source.route.length - 1)],
      d: index * 5,
      heatStrainIndex,
      heatStrainSource: thermalSource(heatStrainIndex, 'core-fit'),
    }
  })
  const rendered = buildActivity(factory, detail({ route }), true)
  const heatStrain = byClass(rendered, 'tri-elev-wrap').find(
    graph => graph.properties.dataTriTrace === 'heat-strain-index',
  )
  assert.ok(heatStrain)
  const missing = byClass(heatStrain, 'tri-elev-line--missing')[0]
  assert.ok(missing)
  const path = String(missing.properties.d)
  assert.equal(path.match(/M /g)?.length, 3)
  assert.match(path, /^M 0 ([\d.]+) L 16\.67 \1 /)
  assert.match(path, /M 16\.67 [\d.]+ L 66\.67 [\d.]+/)
  assert.match(path, /M 66\.67 ([\d.]+) L 100 \1 $/)
})

test('renders estimated run stride length without bridging missing cadence samples', () => {
  const run = detail({
    sport: 'run',
    deviceWatts: false,
    route: detail().route.map((point, index) => ({
      ...point,
      cad: index === 1 ? 0 : 80 + index * 5,
      speedKph: index === 1 ? 0 : 10 + index,
    })),
  })
  const first = runStrideLengthM(run.route[0])
  assert.ok(first)
  assert.equal(first.toFixed(3), '1.042')
  assert.equal(runStrideLengthM(run.route[1]), null)
  assert.equal(formatStrideLength(METRIC_TRIATHLON_PRESENTATION, first), '1.04 m')

  const trace = buildRunStrideTrace(factory, run, null)
  assert.ok(trace)
  assert.equal(trace.properties.dataTriTrace, 'estimated-stride-length')
  assert.deepEqual(
    byClass(trace, 'tri-elev-cap')
      .flatMap(cap => byTag(cap, 'span'))
      .map(text),
    ['estimated stride length', '1.10 m avg'],
  )
  const line = byClass(trace, 'tri-elev-line')[0]
  assert.ok(line)
  assert.equal(String(line.properties.d).match(/M /g)?.length, 2)

  assert.equal(formatStrideLength(imperialPresentation, first), '3.42 ft')
})

test('prefers native running dynamics for the whole activity and preserves sensor gaps', () => {
  const run = detail({
    sport: 'run',
    deviceWatts: false,
    route: detail().route.map((point, index) => ({
      ...point,
      speedKph: 10 + index,
      cad: 80,
      strideLengthM: index === 1 ? null : 1.1 + index * 0.05,
      groundContactTimeMs: index === 1 ? null : 245 - index * 3,
      verticalOscillationCm: index === 1 ? null : 9.8 - index * 0.1,
    })),
  })

  assert.equal(runStrideLengthM(run.route[1])?.toFixed(3), '1.146')
  assert.equal(runStrideLengthValue(run, run.route[1]), null)
  assert.equal(formatGroundContactTime(241.4), '241 ms')
  assert.equal(formatVerticalOscillation(METRIC_TRIATHLON_PRESENTATION, 9.76), '9.8 cm')
  assert.equal(formatVerticalOscillation(imperialPresentation, 9.76), '3.8 in')

  const traces = [
    buildRunStrideTrace(factory, run, null),
    buildRunGroundContactTrace(factory, run, null),
    buildRunVerticalOscillationTrace(factory, run, null),
  ]
  assert.ok(traces.every(trace => trace != null))
  assert.deepEqual(
    traces.map(trace => trace?.properties.dataTriTrace),
    ['stride-length', 'ground-contact-time', 'vertical-oscillation'],
  )
  assert.deepEqual(
    traces.map(trace =>
      byClass(trace!, 'tri-elev-cap')
        .flatMap(cap => byTag(cap, 'span'))
        .map(text),
    ),
    [
      ['stride length', '1.18 m avg'],
      ['ground contact time', '240 ms avg'],
      ['vertical oscillation', '9.6 cm avg'],
    ],
  )
  for (const trace of traces) {
    const line = byClass(trace!, 'tri-elev-line')[0]
    assert.equal(String(line.properties.d).match(/M /g)?.length, 2)
  }
})

test('renders calculated cycling performance condition with its model inputs', () => {
  const ride = detail({
    performanceConditionTrace: {
      source: 'garden-estimate',
      method: 'garden-cycling-performance-condition-v1',
      ftpWatts: 287,
      lactateThresholdHeartRateBpm: 173,
      restingHeartRateBpm: 50,
      windowSeconds: 360,
    },
    route: detail().route.map((point, index) => ({
      ...point,
      performanceCondition: [-2, 0, 2, 4][index],
    })),
  })
  const rendered = buildActivity(factory, ride, true)
  const trace = byClass(rendered, 'tri-elev-wrap').find(
    element => element.properties.dataTriTrace === 'performance-condition',
  )

  assert.ok(trace)
  assert.equal(trace.properties.dataPerformanceConditionSource, 'garden-estimate')
  assert.equal(
    text(byClass(trace, 'tri-elev-cap')[0]),
    'performance condition+1 avgbaselinecalculated',
  )
  const source = byClass(trace, 'tri-performance-condition-source')[0]
  assert.ok(source)
  assert.match(
    String(source.properties.dataGlossDef),
    /6 min \(NP \/ FTP\) ÷ \(\(HR − RHR\) \/ \(LTHR − RHR\)\)/,
  )
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['-4', '-2', '0', '+2', '+4'])
  const reference = byClass(trace, 'tri-trace-reference')[0]
  const area = byClass(trace, 'tri-elev-area')[0]
  assert.ok(reference)
  assert.ok(area)
  assert.match(String(area.properties.d), new RegExp(`^M 0 ${reference.properties.y1} `))

  const mapMetric = metricSpecs(METRIC_TRIATHLON_PRESENTATION, ride, ctx()).find(
    spec => spec.label === 'performance condition',
  )
  assert.ok(mapMetric)
  assert.equal(mapMetric.shortLabel, 'PC')
  assert.equal(mapMetric.pick(ride.route[3], 3), 4)
})

test('renders the complete native Forerunner running dynamics set with paired controls', () => {
  const performanceCondition = [-4, -6, -8, -10]
  const strideLengthM = [1.04, 1.08, 1.1, 1.08]
  const verticalRatioPct = [11.1, 11.2, 11.4, 11.5]
  const verticalOscillationCm = [12.2, 12.4, 12.5, 12.5]
  const groundContactBalanceLeftPct = [49.1, 49.2, 49.4, 49.5]
  const groundContactTimeMs = [244, 246, 248, 250]
  const stepSpeedLossMps = [0.075, 0.078, 0.08, 0.083]
  const stepSpeedLossPct = [2.7, 2.76, 2.8, 2.86]
  const impactLoadFactor = [0.8, 1, 1.1, 1.1]
  const run = detail({
    sport: 'run',
    deviceWatts: false,
    staminaTrace: {
      source: 'garmin',
      method: 'garmin-native',
      ftpWatts: null,
      maxHeartRateBpm: null,
    },
    garmin: garminVerification({
      runningDynamics: {
        source: 'garmin',
        averageRespirationRate: 35.15,
        averageStrideLengthCm: 108.22,
        averageVerticalRatioPct: 11.3,
        averageVerticalOscillationCm: 12.38,
        averageGroundContactBalanceLeftPct: 49.26,
        averageGroundContactTimeMs: 246.5,
        averageStepSpeedLossMps: 0.079,
        averageStepSpeedLossPct: 2.78,
        impactLoadM: 5_320,
      },
    }),
    runWalk: {
      source: 'garmin',
      elapsedTimeS: 1_800.479,
      runTimeS: 1_746.741,
      walkTimeS: 46.836,
      idleTimeS: 6.902,
      segments: [
        { state: 'run', startElapsedS: 0, endElapsedS: 900 },
        { state: 'walk', startElapsedS: 900, endElapsedS: 920 },
        { state: 'run', startElapsedS: 920, endElapsedS: 1_766.741 },
        { state: 'walk', startElapsedS: 1_766.741, endElapsedS: 1_793.577 },
        { state: 'idle', startElapsedS: 1_793.577, endElapsedS: 1_800.479 },
      ],
    },
    route: detail().route.map((point, index) => ({
      ...point,
      resp: [34, 35, 35, 36][index],
      stamina: [87, 82, 76, 71][index],
      potentialStamina: [88, 84, 78, 71][index],
      performanceCondition: performanceCondition[index],
      strideLengthM: strideLengthM[index],
      verticalRatioPct: verticalRatioPct[index],
      verticalOscillationCm: verticalOscillationCm[index],
      groundContactBalanceLeftPct: groundContactBalanceLeftPct[index],
      groundContactTimeMs: groundContactTimeMs[index],
      stepSpeedLossMps: stepSpeedLossMps[index],
      stepSpeedLossPct: stepSpeedLossPct[index],
      impactLoadFactor: impactLoadFactor[index],
    })),
  })
  const rendered = buildActivity(factory, run, true)
  const traces = byClass(rendered, 'tri-elev-wrap').map(trace => trace.properties.dataTriTrace)

  for (const graph of byClass(rendered, 'tri-elev')) {
    const selection = byClass(graph, 'tri-analysis-selection')
    assert.equal(selection.length, 1, `${classNames(graph).join(' ')} has a lap highlight`)
    assert.equal(selection[0].properties.width, '0.00')
    assert.ok(
      graph.properties.dataDomainEndDistanceKm != null ||
        graph.properties.dataDomainEndElapsedS != null,
    )
  }

  for (const trace of [
    'stamina',
    'performance-condition',
    'stride-length',
    'vertical-oscillation',
    'vertical-ratio',
    'ground-contact-time',
    'ground-contact-balance',
    'step-speed-loss',
    'step-speed-loss-percent',
    'respiration',
    'run-walk',
    'impact-load-factor',
  ])
    assert.ok(traces.includes(trace), `${trace} trace is present`)

  const stamina = byClass(rendered, 'tri-elev-wrap').find(
    element => element.properties.dataTriTrace === 'stamina',
  )
  assert.ok(stamina)
  assert.deepEqual(byClass(stamina, 'tri-stamina-legend-item').map(text), ['current', 'potential'])
  const staminaMapMetric = metricSpecs(METRIC_TRIATHLON_PRESENTATION, run, ctx()).find(
    spec => spec.label === 'stamina',
  )
  assert.ok(staminaMapMetric)

  const paired = byClass(rendered, 'tri-run-metric-chart')
  assert.deepEqual(
    paired.map(chart => [chart.properties.dataRunMetricGroup, chart.properties.dataRunMetricMode]),
    [
      ['vertical', 'vertical-oscillation'],
      ['ground-contact', 'ground-contact-time'],
      ['step-speed-loss', 'step-speed-loss'],
    ],
  )
  for (const chart of paired) {
    const panes = byClass(chart, 'tri-run-metric-pane')
    assert.equal(panes.length, 2)
    assert.equal(panes[0].properties.ariaHidden, 'false')
    assert.equal(panes[0].properties.hidden, undefined)
    assert.equal(panes[1].properties.ariaHidden, 'true')
    assert.equal(panes[1].properties.hidden, true)
    assert.deepEqual(
      byClass(panes[0], 'tri-run-metric-mode').map(button => button.properties.ariaPressed),
      ['true', 'false'],
    )
  }

  const summaries: [string, string][] = [
    ['performance-condition', 'performance condition-7 avgbaseline'],
    ['stride-length', 'stride length1.08 m avg'],
    ['vertical-oscillation', 'vertical oscillation12.4 cm avgvertical oscillationvertical ratio'],
    ['vertical-ratio', 'vertical ratio11.3% avgvertical oscillationvertical ratio'],
    [
      'ground-contact-time',
      'ground contact time246.5 ms avgground contact timeground contact balance',
    ],
    [
      'ground-contact-balance',
      'ground contact balance49.3% L / 50.7% R avg50 / 50ground contact timeground contact balance',
    ],
    ['step-speed-loss', 'step speed loss7.9 cm/s avgstep speed lossstep speed loss percent'],
    [
      'step-speed-loss-percent',
      'step speed loss percent2.78% avgstep speed lossstep speed loss percent',
    ],
    ['respiration', 'respiration35.1 brpm avg'],
    ['impact-load-factor', 'impact load factor1.00 avg'],
  ]
  for (const [traceName, expected] of summaries) {
    const trace = byClass(rendered, 'tri-elev-wrap').find(
      element => element.properties.dataTriTrace === traceName,
    )
    assert.ok(trace)
    const cap = byClass(trace, 'tri-elev-cap')[0]
    assert.ok(cap)
    assert.equal(text(cap), expected)
  }

  const runWalk = byClass(rendered, 'tri-run-walk-chart')[0]
  assert.ok(runWalk)
  assert.equal(runWalk.properties.dataRunWalkSource, 'garmin')
  assert.equal(runWalk.properties.ariaLabel, 'run 29:07, walk 0:46.8, idle 0:06.9')
  assert.deepEqual(byClass(runWalk, 'tri-run-walk-time').map(text), ['29:07', '0:46.8', '0:06.9'])
  const runWalkSegments = byClass(runWalk, 'tri-run-walk-segment')
  const runWalkSvg = byClass(runWalk, 'tri-run-walk')[0]
  assert.equal(runWalkSvg.properties.role, 'slider')
  assert.equal(runWalkSvg.properties.tabIndex, 0)
  assert.equal(runWalkSvg.properties.ariaHidden, undefined)
  assert.equal(runWalkSvg.properties.ariaValueMax, 1_800.479)
  assert.equal(byClass(runWalk, 'tri-chart-cursor').length, 1)
  assert.equal(byClass(runWalk, 'tri-elev-cursor')[0], byClass(runWalk, 'tri-chart-cursor')[0])
  assert.equal(byClass(runWalk, 'tri-chart-readout').length, 1)
  assert.equal(byClass(runWalk, 'tri-fig-readout').length, 1)
  assert.deepEqual(
    runWalkSegments.map(segment => [
      segment.properties.dataStartElapsedS,
      segment.properties.dataEndElapsedS,
    ]),
    run.runWalk?.segments.map(segment => [segment.startElapsedS, segment.endElapsedS]),
  )
  assert.deepEqual(
    runWalkSegments.map(segment => segment.properties.dataRunWalkState),
    ['run', 'walk', 'run', 'walk', 'idle'],
  )
  assert.deepEqual(
    runWalkSegments.map(segment => [segment.properties.dataRunWalkState, segment.properties.y]),
    [
      ['run', 21],
      ['walk', 11],
      ['run', 21],
      ['walk', 11],
      ['idle', 1],
    ],
  )
})

test('run/walk hover follows exact interval boundaries, including short idle periods', () => {
  const segments: GarminRunWalkSegment[] = [
    { state: 'run', startElapsedS: 0, endElapsedS: 900 },
    { state: 'walk', startElapsedS: 900, endElapsedS: 920 },
    { state: 'idle', startElapsedS: 920, endElapsedS: 920.125 },
    { state: 'run', startElapsedS: 920.125, endElapsedS: 1800 },
  ]
  for (const [elapsedS, expected] of [
    [0, 'run'],
    [899.999, 'run'],
    [900, 'walk'],
    [919.999, 'walk'],
    [920, 'idle'],
    [920.124, 'idle'],
    [920.125, 'run'],
    [1800, 'run'],
  ])
    assert.equal(runWalkSegmentAt(segments, Number(elapsedS))?.state, expected)
  for (const elapsedS of [-1, 1800.001, NaN, Infinity])
    assert.equal(runWalkSegmentAt(segments, elapsedS), null)
  assert.equal(runWalkSegmentAt([], 0), null)
  assert.equal(runWalkSegmentAt([segments[0], segments[3]], 910), null)
  assert.equal(runWalkSegmentAt([segments[0]], 450), segments[0])
})

test('summarizes a dragged graph range from either pointer direction', () => {
  const route = detail().route
  const forward = activitySelectionSummary(route, 1, 3)
  const backward = activitySelectionSummary(route, 3, 1)

  assert.deepEqual(forward, backward)
  assert.deepEqual(forward, {
    startElapsedS: 1_600,
    endElapsedS: 4_800,
    startDistanceKm: 10,
    endDistanceKm: 30,
    durationS: 3_200,
    distanceKm: 20,
    elevationGainM: 21,
    averageSpeedKph: 22.5,
    averageHeartRate: 150,
    averageWatts: 201.25,
    averageCadence: 87.5,
    averageRespirationRate: 28,
    averageTemperatureC: 24,
  })
  assert.equal(activitySelectionSummary(route, 2, 2), null)
})

test('renders compact positional analysis bars beneath the existing activity figures', () => {
  const rendered = buildActivity(factory, analysisDetail(), true)
  const figures = byClass(rendered, 'tri-act-figs')[0]
  const analysis = byClass(figures, 'tri-analysis')[0]
  assert.ok(analysis)
  assert.equal(analysis.tagName, 'section')
  assert.equal(analysis.properties.dataActivityId, '101')
  assert.equal(analysis.properties.dataSelectedKind, undefined)
  assert.equal(analysis.properties.dataSelectedId, undefined)
  assert.equal(analysis.properties.ariaLabel, 'Activity analysis')

  const directClasses = figures.children
    .filter((child): child is Element => child.type === 'element')
    .map(child => classNames(child))
  assert.deepEqual(directClasses.slice(0, 3), [['tri-route'], ['tri-elev-wrap'], ['tri-analysis']])

  const route = byClass(figures, 'tri-route')[0]
  assert.ok(route)
  assert.equal(route.properties.viewBox, '0 0 100 100')
  assert.equal(route.properties.preserveAspectRatio, 'xMidYMid meet')
  assert.equal(byClass(route, 'tri-route-path').length, 1)
  assert.equal(byClass(route, 'tri-route-selected').length, 1)
  assert.equal(byClass(route, 'tri-route-cursor').length, 1)

  const bands = byClass(analysis, 'tri-analysis-band')
  assert.deepEqual(
    bands.map(band => [
      band.properties.dataAnalysisKind,
      band.properties.role,
      band.properties.ariaLabel,
    ]),
    [
      ['lap', 'group', 'Laps'],
      ['segment', 'group', 'Segments'],
      ['climb', 'group', 'ClimbPro'],
    ],
  )

  const buttons = byClass(analysis, 'tri-analysis-range')
  assert.deepEqual(
    buttons.map(button => [button.properties.dataRangeKind, button.properties.dataRangeId]),
    [
      ['lap', 'lap-2'],
      ['segment', 'segment-boardwalk'],
      ['climb', 'climb-bay'],
    ],
  )
  assert.deepEqual(
    buttons.map(button => button.properties.ariaPressed),
    ['false', 'false', 'false'],
  )
  assert.match(String(buttons[0].properties.style), /--tri-analysis-start:25\.000%/)
  assert.match(String(buttons[0].properties.style), /--tri-analysis-width:25\.000%/)
  assert.match(String(buttons[0].properties.style), /--tri-analysis-lane:0/)
  assert.equal(buttons[0].properties.title, undefined)
  assert.match(String(buttons[0].properties.ariaLabel), /^Lap 2, 7\.50 km, \+24 m, 20:00/)
  assert.equal(byClass(analysis, 'tri-analysis-range-title').length, 0)
  assert.equal(byClass(analysis, 'tri-analysis-range-stats').length, 0)

  const readout = byClass(analysis, 'tri-analysis-readout')[0]
  assert.ok(readout)
  assert.equal(readout.properties.dataTriAnalysisReadout, '')
  assert.equal(readout.properties.dataVisible, 'false')
  assert.equal(readout.properties.ariaHidden, 'true')
  assert.equal(readout.properties.ariaLive, 'polite')
  assert.deepEqual(byClass(readout, 'tri-analysis-readout-label').map(text), [''])
  assert.deepEqual(byClass(readout, 'tri-analysis-readout-metrics').map(text), [''])
  assert.equal(byClass(analysis, 'tri-analysis-tooltip').length, 0)
})

test('labels Wahoo climb analysis as Summit and efforts as Summit Segments', () => {
  const activity = analysisDetail()
  activity.computer = 'wahoo'
  activity.analysisRanges = activity.analysisRanges.map(range =>
    range.kind === 'climb'
      ? { ...range, source: 'wahoo-summit-segment', label: 'Summit 2/6' }
      : range,
  )
  const efforts = activity.bestEfforts
  assert.ok(efforts)
  efforts.climbs = efforts.climbs.map(climb => ({
    ...climb,
    source: 'wahoo-summit-segment',
    name: 'Summit 2/6',
  }))

  const rendered = buildActivity(factory, activity, true)
  const band = byClass(rendered, 'tri-analysis-band').find(
    candidate => candidate.properties.dataAnalysisKind === 'climb',
  )
  assert.ok(band)
  assert.equal(band.properties.ariaLabel, 'Summit Segments')
  assert.deepEqual(byClass(band, 'tri-analysis-band-label').map(text), ['Summit'])
  assert.deepEqual(
    byClass(band, 'tri-analysis-range').map(button => button.properties.ariaLabel),
    ['Summit 2/6, 5.00 km, +48 m, 13:20, 22.5 km/h, 232 W, 151 bpm, 84 rpm'],
  )

  const climbing = table(rendered, 'climbing')
  assert.equal(climbing.properties.ariaLabel, 'Summit Segments efforts')
  assert.deepEqual(headerText(climbing), [
    'Segment',
    'Time',
    'Distance',
    'Gain',
    'Grade',
    'Speed',
    'Heart rate',
    'Power',
    'W/kg',
    'VAM',
  ])
  assert.equal(bodyRows(climbing)[0][0], 'Summit 2/6')
})

test('preserves available climb grades in analysis labels and hydration data', () => {
  for (const grade of [6.1, 0, -2.4, null, undefined, Number.NaN, Number.POSITIVE_INFINITY]) {
    const activity = analysisDetail()
    activity.analysisRanges = activity.analysisRanges.map(range =>
      range.kind === 'climb' ? { ...range, averageGradePct: grade } : range,
    )
    const rendered = buildActivity(factory, activity, true)
    const button = byClass(rendered, 'tri-analysis-range').find(
      candidate => candidate.properties.dataRangeKind === 'climb',
    )
    assert.ok(button)
    if (grade != null && Number.isFinite(grade)) {
      assert.equal(button.properties.dataAverageGradePct, String(grade))
      assert.ok(String(button.properties.ariaLabel).includes(`${grade.toFixed(1)}% avg grade`))
    } else {
      assert.equal(button.properties.dataAverageGradePct, undefined)
      assert.doesNotMatch(String(button.properties.ariaLabel), /avg grade/)
    }
  }
})

test('renders cycling laps as selectable power bars over the elevation profile', () => {
  const bike = analysisDetail()
  bike.analysisRanges = [
    {
      kind: 'lap',
      id: 'lap-1',
      label: 'Lap 1',
      startElapsedS: 0,
      endElapsedS: 1_600,
      startDistanceKm: 0,
      endDistanceKm: 10,
      durationS: 1_600,
      movingTimeS: 1_200,
      distanceKm: 10,
      elevationGainM: 14,
      averageSpeedKph: 30,
      averageHeartRate: 142,
      averageWatts: 180,
      averageCadence: 84,
    },
    {
      kind: 'lap',
      id: 'lap-2',
      label: 'Lap 2',
      startElapsedS: 1_600,
      endElapsedS: 3_200,
      startDistanceKm: 10,
      endDistanceKm: 20,
      durationS: 1_600,
      movingTimeS: 1_200,
      distanceKm: 10,
      elevationGainM: 14,
      averageSpeedKph: 30,
      averageHeartRate: 154,
      averageWatts: 300,
      averageCadence: 92,
    },
    {
      kind: 'lap',
      id: 'lap-3',
      label: 'Lap 3',
      startElapsedS: 3_200,
      endElapsedS: 4_800,
      startDistanceKm: 20,
      endDistanceKm: 30,
      durationS: 1_600,
      movingTimeS: 1_200,
      distanceKm: 10,
      elevationGainM: 7,
      averageSpeedKph: 30,
      averageHeartRate: 136,
      averageWatts: 120,
      averageCadence: 78,
    },
    ...analysisRanges().filter(range => range.kind !== 'lap'),
  ]

  const rendered = buildActivity(factory, bike, true)
  const workout = byClass(rendered, 'tri-cycling-workout')[0]
  assert.ok(workout)
  assert.equal(workout.tagName, 'section')
  assert.equal(workout.properties.ariaLabel, 'Cycling workout analysis')
  assert.equal(byClass(workout, 'tri-cycling-workout-plot')[0].properties.dataSiteCursorLine, '')
  assert.equal(byClass(workout, 'tri-cycling-workout-title').length, 0)
  for (const embedded of [false, true]) {
    const card = buildActivity(factory, bike, true, undefined, false, embedded)
    const analysis = byClass(card, 'tri-workout-analysis')[0]
    assert.ok(analysis)
    assert.equal(analysis.properties.dataSport, 'bike')
    const tabs = byClass(analysis, 'tri-workout-analysis-tab')
    const panels = byClass(analysis, 'tri-workout-analysis-panel')
    assert.deepEqual(tabs.map(text), [embedded ? 'WA' : 'workout analysis'])
    assert.deepEqual(
      tabs.map(tab => tab.properties.ariaLabel),
      ['workout analysis'],
    )
    assert.equal(tabs[0].properties.role, 'tab')
    assert.equal(tabs[0].properties.ariaSelected, 'true')
    assert.equal(tabs[0].properties.tabIndex, 0)
    assert.equal(panels.length, 1)
    assert.equal(panels[0].properties.role, 'tabpanel')
    assert.equal(panels[0].properties.hidden, undefined)
    assert.deepEqual(tabs[0].properties.ariaControls, [panels[0].properties.id])
    assert.equal(byClass(panels[0], 'tri-cycling-workout').length, 1)
  }
  assert.deepEqual(
    byClass(workout, 'tri-cycling-workout-stats')
      .flatMap(stat => byTag(stat, 'span'))
      .map(text),
    ['highest 300 W', 'avg 200 W', 'lowest 120 W'],
  )
  assert.deepEqual(byClass(workout, 'tri-cycling-workout-y-tick').map(text), ['100', '200', '300'])
  assert.deepEqual(byClass(workout, 'tri-cycling-workout-y-unit').map(text), ['W'])
  const elevation = byClass(workout, 'tri-workout-elevation')[0]
  assert.ok(elevation)
  assert.equal(elevation.tagName, 'svg')
  assert.equal(elevation.properties.viewBox, '0 0 100 100')
  assert.equal(elevation.properties.ariaHidden, 'true')
  assert.match(
    String(byClass(elevation, 'tri-workout-elevation-area')[0].properties.d),
    /^M 0\.000 100 L 0\.000 100\.000 L 33\.333 60\.000 L 66\.667 20\.000 L 100\.000 0\.000 L 100\.000 100 Z$/,
  )
  assert.equal(byClass(elevation, 'tri-cycling-workout-grade').length, 0)
  assert.match(
    String(byClass(workout, 'tri-cycling-workout-average-line')[0].properties.style),
    /top:33\.333%/,
  )

  const laps = byClass(workout, 'tri-cycling-workout-lap')
  assert.deepEqual(
    laps.map(lap => [
      lap.properties.dataRangeKind,
      lap.properties.dataRangeId,
      lap.properties.ariaPressed,
    ]),
    [
      ['lap', 'lap-1', 'false'],
      ['lap', 'lap-2', 'false'],
      ['lap', 'lap-3', 'false'],
    ],
  )
  assert.match(
    String(laps[0].properties.style),
    /--tri-cycling-workout-start:0\.000%;--tri-cycling-workout-width:33\.333%;--tri-cycling-workout-height:60\.000%/,
  )
  assert.match(
    String(laps[1].properties.style),
    /--tri-cycling-workout-start:33\.333%;--tri-cycling-workout-width:33\.333%;--tri-cycling-workout-height:100\.000%/,
  )
  assert.match(
    String(laps[2].properties.style),
    /--tri-cycling-workout-start:66\.667%;--tri-cycling-workout-width:33\.333%;--tri-cycling-workout-height:40\.000%/,
  )
  assert.match(String(laps[0].properties.ariaLabel), /^Lap 1, 10\.00 km, \+14 m, 20:00/)
  assert.match(String(laps[0].properties.ariaLabel), /180 W/)
  assert.deepEqual(byClass(workout, 'tri-workout-lap-tooltip').map(text), [
    '30.0 km/h · 180 W',
    '30.0 km/h · 300 W',
    '30.0 km/h · 120 W',
  ])
  assert.deepEqual(byClass(workout, 'tri-cycling-workout-label').map(text), ['1', '2', '3'])
})

test('renders workout lap hover metrics in the selected units and preserves zero and missing power', () => {
  const bike = analysisDetail()
  const lap = bike.analysisRanges.find(range => range.kind === 'lap')!
  const values = [
    { averageSpeedKph: 0, averageWatts: 0 },
    { averageSpeedKph: null, averageWatts: null },
    { averageSpeedKph: 30, averageWatts: 200 },
  ]
  bike.analysisRanges = values.map((value, index) => ({
    ...lap,
    ...value,
    id: `lap-${index + 1}`,
    startElapsedS: index * 1_600,
    endElapsedS: (index + 1) * 1_600,
    startDistanceKm: index * 10,
    endDistanceKm: (index + 1) * 10,
  }))

  for (const selected of [METRIC_TRIATHLON_PRESENTATION, imperialPresentation]) {
    for (const embedded of [false, true]) {
      const rendered = buildWorkoutAnalysis(factoryFor(selected), bike, embedded)
      assert.ok(rendered)
      const laps = byClass(rendered, 'tri-workout-lap')
      assert.equal(laps.length, 3)
      assert.deepEqual(
        laps.map(lap => text(byClass(lap, 'tri-workout-lap-tooltip')[0])),
        selected.distance === 'imperial'
          ? ['0.0 mph · 0 W', '—', '18.6 mph · 200 W']
          : ['0.0 km/h · 0 W', '—', '30.0 km/h · 200 W'],
      )
      for (const lap of laps) {
        assert.equal(lap.tagName, 'button')
        assert.equal(lap.properties.type, 'button')
        assert.equal(byClass(lap, 'tri-workout-lap-bar').length, 1)
        assert.equal(byClass(lap, 'tri-workout-lap-tooltip')[0].properties.ariaHidden, 'true')
      }
    }
  }
})

test('maps local elevation gradient to distinct climb color bands', () => {
  assert.deepEqual(
    [Number.NaN, -1, 0, 0.1, 3.9, 4, 7.9, 8, 11.9, 12, 19.9, 20, 28].map(climbGradeBand),
    [
      null,
      null,
      null,
      'gentle',
      'gentle',
      'moderate',
      'moderate',
      'hard',
      'hard',
      'steep',
      'steep',
      'wall',
      'wall',
    ],
  )
})

test('overlays climb grade colors on the standalone elevation profile', () => {
  const seed = detail().route[0]
  const route = [
    [0, 0],
    [0.1, 2],
    [0.2, 6],
    [0.3, 18],
    [0.4, 17],
    [0.5, 37],
    [0.6, 65],
  ].map(([d, alt], index) => ({ ...seed, d, alt, elapsedS: index * 60 }))
  const elevation = buildElevation(
    factory,
    detail({ route, distanceKm: 0.6, minAlt: 0, maxAlt: 65 }),
  )
  const svg = byClass(elevation, 'tri-elev')[0]
  assert.ok(svg)
  assert.equal(byClass(svg, 'tri-elev-area').length, 1)
  assert.equal(byClass(svg, 'tri-elev-grades').length, 1)
  const grades = byClass(svg, 'tri-elev-grade')
  assert.deepEqual(
    grades.map(grade => grade.properties.dataGradeBand),
    ['moderate', 'hard', 'steep', 'wall'],
  )
  assert.deepEqual(
    grades.map(grade => grade.properties.d),
    [
      'M 0.00 30 L 0.00 30.00 L 16.67 29.08 L 33.33 27.23 L 33.33 30 Z',
      'M 33.33 30 L 33.33 27.23 L 50.00 21.69 L 50.00 30 Z',
      'M 66.67 30 L 66.67 22.15 L 83.33 12.92 L 83.33 30 Z',
      'M 83.33 30 L 83.33 12.92 L 100.00 0.00 L 100.00 30 Z',
    ],
  )
})

test('omits cycling workout analysis when lap power is unavailable', () => {
  const bike = analysisDetail()
  bike.analysisRanges = bike.analysisRanges.map(range =>
    range.kind === 'lap' ? { ...range, averageWatts: null } : range,
  )
  assert.equal(byClass(buildActivity(factory, bike, true), 'tri-cycling-workout').length, 0)
  assert.equal(
    byClass(buildActivity(factory, { ...bike, sport: 'run' }, true), 'tri-cycling-workout').length,
    0,
  )
})

test('renders consecutive pool swim pace bars while preserving elapsed selection ranges', () => {
  const swim = swimTrendDetail({
    distanceKm: 0.15,
    movingTimeS: 195,
    elapsedTimeS: 300,
    analysisRanges: [
      {
        kind: 'lap',
        id: 'garmin-swim-lap:1',
        label: 'Lap 1',
        startElapsedS: 72,
        endElapsedS: 192,
        startDistanceKm: 0,
        endDistanceKm: 0.1,
        durationS: 120,
        movingTimeS: 120,
        distanceKm: 0.1,
        elevationGainM: null,
        averageSpeedKph: 3,
        averageHeartRate: 130,
        averageWatts: null,
        averageCadence: 25,
      },
      {
        kind: 'lap',
        id: 'garmin-swim-lap:2',
        label: 'Lap 2',
        startElapsedS: 222,
        endElapsedS: 297,
        startDistanceKm: 0.1,
        endDistanceKm: 0.15,
        durationS: 75,
        movingTimeS: 75,
        distanceKm: 0.05,
        elevationGainM: null,
        averageSpeedKph: 2.4,
        averageHeartRate: 140,
        averageWatts: null,
        averageCadence: 24,
      },
    ],
  })

  const rendered = buildActivity(factory, swim, true)
  const workout = byClass(rendered, 'tri-swim-workout')[0]
  assert.ok(workout)
  assert.equal(workout.properties.ariaLabel, 'Swim workout analysis')
  for (const embedded of [false, true]) {
    const card = buildActivity(factory, swim, true, undefined, false, embedded)
    const analysis = byClass(card, 'tri-workout-analysis')[0]
    assert.ok(analysis)
    assert.equal(analysis.properties.dataSport, 'swim')
    const tabs = byClass(analysis, 'tri-workout-analysis-tab')
    const panels = byClass(analysis, 'tri-workout-analysis-panel')
    assert.deepEqual(tabs.map(text), [embedded ? 'WA' : 'workout analysis'])
    assert.deepEqual(
      tabs.map(tab => tab.properties.ariaLabel),
      ['workout analysis'],
    )
    assert.equal(tabs[0].properties.role, 'tab')
    assert.equal(tabs[0].properties.ariaSelected, 'true')
    assert.equal(tabs[0].properties.tabIndex, 0)
    assert.equal(panels.length, 1)
    assert.equal(panels[0].properties.role, 'tabpanel')
    assert.equal(panels[0].properties.hidden, undefined)
    assert.deepEqual(tabs[0].properties.ariaControls, [panels[0].properties.id])
    assert.equal(byClass(panels[0], 'tri-swim-workout').length, 1)
  }
  assert.equal(workout.properties.dataSwimWorkoutElevation, 'false')
  assert.equal(byClass(workout, 'tri-workout-elevation').length, 0)
  assert.equal(byClass(workout, 'tri-swim-workout-plot')[0].properties.dataSiteCursorLine, '')
  assert.deepEqual(
    byClass(workout, 'tri-swim-workout-stats')
      .flatMap(stat => byTag(stat, 'span'))
      .map(text),
    ['fastest 2:00 /100m', 'avg 2:10 /100m', 'slowest 2:30 /100m'],
  )
  assert.deepEqual(byClass(workout, 'tri-swim-workout-y-tick').map(text), ['2:00', '2:30'])
  assert.deepEqual(byClass(workout, 'tri-swim-workout-pace').map(text), [
    '2:00 /100m',
    '2:30 /100m',
  ])
  const laps = byClass(workout, 'tri-swim-workout-lap')
  assert.deepEqual(
    laps.map(lap => [lap.properties.dataRangeId, lap.properties.ariaPressed]),
    [
      ['garmin-swim-lap:1', 'false'],
      ['garmin-swim-lap:2', 'false'],
    ],
  )
  assert.match(
    String(laps[0].properties.style),
    /--tri-swim-workout-start:0\.000%;--tri-swim-workout-width:61\.538%;--tri-swim-workout-height:100\.000%/,
  )
  assert.match(
    String(laps[1].properties.style),
    /--tri-swim-workout-start:61\.538%;--tri-swim-workout-width:38\.462%;--tri-swim-workout-height:3\.000%/,
  )
  assert.deepEqual(
    laps.map(lap => [lap.properties.dataStartElapsedS, lap.properties.dataEndElapsedS]),
    [
      ['72', '192'],
      ['222', '297'],
    ],
  )
  assert.match(String(laps[0].properties.ariaLabel), /^Lap 1, 100 m, 2:00, 2:00 \/100m/)
  assert.match(String(laps[0].properties.ariaLabel), /130 bpm, 25 spm$/)

  const openWater = buildWorkoutAnalysis(factory, { ...swim, swimLocation: 'openWater' })
  assert.ok(openWater)
  const openWaterLaps = byClass(openWater, 'tri-swim-workout-lap')
  assert.match(
    String(openWaterLaps[0].properties.style),
    /--tri-swim-workout-start:24\.000%;--tri-swim-workout-width:40\.000%/,
  )
  assert.match(
    String(openWaterLaps[1].properties.style),
    /--tri-swim-workout-start:74\.000%;--tri-swim-workout-width:25\.000%/,
  )
})

test('adds elevation behind open-water swim laps only when GPS data exists', () => {
  const seed = detail().route[0]
  const route = [
    { ...seed, x: 0, y: 0, d: 0, alt: 75, elapsedS: 0 },
    { ...seed, x: 0.5, y: 0.5, d: 0.1, alt: 76, elapsedS: 60 },
    { ...seed, x: 1, y: 1, d: 0.2, alt: 74, elapsedS: 120 },
  ]
  const swim = swimTrendDetail({
    name: 'Lake swim',
    distanceKm: 0.2,
    movingTimeS: 120,
    elapsedTimeS: 120,
    route,
    mapRoute: [route.map(point => ({ lat: point.lat, lng: point.lng, d: point.d }))],
    swimLocation: 'openWater',
    analysisRanges: [
      {
        kind: 'lap',
        id: 'garmin-swim-lap:1',
        label: 'Lap 1',
        startElapsedS: 0,
        endElapsedS: 120,
        startDistanceKm: 0,
        endDistanceKm: 0.2,
        durationS: 120,
        movingTimeS: 120,
        distanceKm: 0.2,
        elevationGainM: 2,
        averageSpeedKph: 6,
        averageHeartRate: 145,
        averageWatts: null,
        averageCadence: 32,
      },
    ],
  })

  const withGps = buildActivity(factory, swim, true)
  const workout = byClass(withGps, 'tri-swim-workout')[0]
  assert.ok(workout)
  assert.equal(workout.properties.dataSwimWorkoutElevation, 'true')
  const elevation = byClass(workout, 'tri-workout-elevation')[0]
  assert.ok(elevation)
  assert.match(
    String(byClass(elevation, 'tri-workout-elevation-area')[0].properties.d),
    /^M 0\.000 100 L 0\.000 50\.000 L 50\.000 0\.000 L 100\.000 100\.000 L 100\.000 100 Z$/,
  )
  assert.match(
    String(byClass(workout, 'tri-swim-workout-lap')[0].properties.ariaLabel),
    /^Lap 1, 200 m, \+2 m, 2:00, 1:00 \/100m, 145 bpm, 32 spm$/,
  )

  const withoutGps = buildActivity(factory, { ...swim, route: [], mapRoute: [] }, true)
  const routeLessWorkout = byClass(withoutGps, 'tri-swim-workout')[0]
  assert.ok(routeLessWorkout)
  assert.equal(routeLessWorkout.properties.dataSwimWorkoutElevation, 'false')
  assert.equal(byClass(routeLessWorkout, 'tri-workout-elevation').length, 0)

  const pool = buildWorkoutAnalysis(factory, { ...swim, swimLocation: 'pool' })
  assert.ok(pool)
  assert.equal(byClass(pool, 'tri-swim-workout-lap').length, 1)
  assert.equal(byClass(pool, 'tri-workout-elevation').length, 0)
})

test('aligns GPS run elevation and unequal lap widths on elapsed time, including rest gaps', () => {
  const seed = detail().route[0]
  const route = [
    { ...seed, d: 0, alt: 0, elapsedS: 0 },
    { ...seed, d: 0.4, alt: 10, elapsedS: 120 },
    { ...seed, d: 0.4, alt: 20, elapsedS: 180 },
    { ...seed, d: 1.8, alt: 0, elapsedS: 600 },
  ]
  const lap = analysisRanges().find(range => range.kind === 'lap')!
  const run = detail({
    sport: 'run',
    distanceKm: 1.8,
    movingTimeS: 540,
    elapsedTimeS: 600,
    route,
    mapRoute: [route.map(point => ({ lat: point.lat, lng: point.lng, d: point.d }))],
    analysisRanges: [
      {
        ...lap,
        id: 'lap-1',
        startElapsedS: 0,
        endElapsedS: 120,
        startDistanceKm: 0,
        endDistanceKm: 0.4,
        durationS: 120,
        distanceKm: 0.4,
        averageSpeedKph: 12,
      },
      {
        ...lap,
        id: 'lap-2',
        startElapsedS: 180,
        endElapsedS: 600,
        startDistanceKm: 0.4,
        endDistanceKm: 1.8,
        durationS: 420,
        distanceKm: 1.4,
        averageSpeedKph: 12,
      },
    ],
  })

  for (const embedded of [false, true]) {
    const rendered = buildWorkoutAnalysis(factory, run, embedded)
    assert.ok(rendered)
    const workout = byClass(rendered, 'tri-run-workout')[0]
    assert.equal(workout.properties.dataRunWorkoutElevation, 'true')
    const plot = byClass(workout, 'tri-run-workout-plot')[0]
    const layers = plot.children.filter((child): child is Element => child.type === 'element')
    assert.deepEqual(
      layers.map(child => classNames(child)[0]),
      ['tri-workout-elevation', 'tri-workout-grid', 'tri-run-workout-bars'],
    )
    assert.equal(layers[0].properties.ariaHidden, 'true')
    assert.equal(
      byClass(workout, 'tri-workout-elevation-area')[0].properties.d,
      'M 0.000 100 L 0.000 100.000 L 20.000 50.000 L 30.000 0.000 L 100.000 100.000 L 100.000 100 Z',
    )
    const laps = byClass(workout, 'tri-run-workout-lap')
    assert.match(
      String(laps[0].properties.style),
      /--tri-run-workout-start:0\.000%;--tri-run-workout-width:20\.000%/,
    )
    assert.match(
      String(laps[1].properties.style),
      /--tri-run-workout-start:30\.000%;--tri-run-workout-width:70\.000%/,
    )
  }

  const unavailable: Partial<StravaActivityDetail>[] = [
    { route: [], mapRoute: [] },
    { mapRoute: [] },
    { route: route.map(point => ({ ...point, lat: Number.NaN })) },
    { route: route.map((point, index) => ({ ...point, alt: index ? Number.NaN : point.alt })) },
  ]
  for (const override of unavailable) {
    const rendered = buildWorkoutAnalysis(factory, { ...run, ...override })
    assert.ok(rendered)
    assert.equal(
      byClass(rendered, 'tri-run-workout')[0].properties.dataRunWorkoutElevation,
      'false',
    )
    assert.equal(byClass(rendered, 'tri-workout-elevation').length, 0)
    assert.equal(byClass(rendered, 'tri-run-workout-lap').length, 2)
  }
})

test('closes workout elevation at recorded endpoints across cycling, running, and open-water swimming', () => {
  const seed = detail().route[0]
  const route = [
    { ...seed, d: 0.2, alt: 80, elapsedS: 20 },
    { ...seed, d: 0.5, alt: 100, elapsedS: 50 },
    { ...seed, d: 0.8, alt: 90, elapsedS: 80 },
  ]
  const lap = analysisRanges().find(range => range.kind === 'lap')!
  for (const sport of ['bike', 'run', 'swim'] as const) {
    const activity = detail({
      sport,
      elapsedTimeS: 100,
      movingTimeS: 100,
      distanceKm: 1,
      swimLocation: 'openWater',
      route,
      mapRoute: [route.map(point => ({ lat: point.lat, lng: point.lng, d: point.d }))],
      analysisRanges: [
        {
          ...lap,
          startElapsedS: 0,
          endElapsedS: 100,
          startDistanceKm: 0,
          endDistanceKm: 1,
          durationS: 100,
          distanceKm: 1,
          averageSpeedKph: 36,
          averageWatts: 200,
        },
      ],
    })
    for (const embedded of [false, true]) {
      const workout = buildWorkoutAnalysis(factory, activity, embedded)
      assert.ok(workout)
      assert.equal(
        byClass(workout, 'tri-workout-elevation-area')[0].properties.d,
        'M 20.000 100 L 20.000 100.000 L 50.000 0.000 L 80.000 50.000 L 80.000 100 Z',
        sport,
      )
    }
  }
})

test('renders run laps as selectable pace splits against the lap-weighted average', () => {
  const run = analysisDetail()
  run.sport = 'run'
  run.runPaceZones = {
    zoneSeconds: [354, 416, 397, 62, 227, 329],
    boundsSPerKm: [387.114, 333.676, 299.501, 280.238, 263.461],
    tenKmRaceTimeS: 3_000,
  }
  run.analysisRanges = [
    {
      kind: 'lap',
      id: 'lap-1',
      label: 'Lap 1',
      startElapsedS: 0,
      endElapsedS: 1_600,
      startDistanceKm: 0,
      endDistanceKm: 10,
      durationS: 330,
      movingTimeS: 300,
      distanceKm: 1,
      elevationGainM: 4,
      averageSpeedKph: 12,
      averageHeartRate: 145,
      averageWatts: null,
      averageCadence: 84,
    },
    {
      kind: 'lap',
      id: 'lap-2',
      label: 'Lap 2',
      startElapsedS: 1_600,
      endElapsedS: 3_200,
      startDistanceKm: 10,
      endDistanceKm: 20,
      durationS: 360,
      distanceKm: 1,
      elevationGainM: 5,
      averageSpeedKph: 10,
      averageHeartRate: 148,
      averageWatts: null,
      averageCadence: 82,
    },
    {
      kind: 'lap',
      id: 'lap-3',
      label: 'Lap 3',
      startElapsedS: 3_200,
      endElapsedS: 4_800,
      startDistanceKm: 20,
      endDistanceKm: 30,
      durationS: 240,
      distanceKm: 1,
      elevationGainM: 3,
      averageSpeedKph: 15,
      averageHeartRate: 152,
      averageWatts: null,
      averageCadence: 87,
    },
    ...analysisRanges().filter(range => range.kind !== 'lap'),
  ]

  const rendered = buildActivity(factory, run, true)
  const analysis = byClass(rendered, 'tri-analysis')[0]
  const more = byClass(rendered, 'tri-act-more')[0]
  const workoutAnalysis = byClass(more, 'tri-workout-analysis')[0]
  const workout = byClass(more, 'tri-run-workout')[0]
  const splits = byClass(more, 'tri-run-splits')[0]
  assert.ok(workoutAnalysis)
  assert.equal(workoutAnalysis.properties.ariaLabel, 'Run analysis')
  assert.equal(workoutAnalysis.properties.dataSport, 'run')
  assert.equal(workoutAnalysis.properties.dataWorkoutAnalysisView, 'workout')
  const tabs = byClass(workoutAnalysis, 'tri-workout-analysis-tab')
  assert.deepEqual(tabs.map(text), ['workout analysis', 'lap splits', 'pace distribution'])
  assert.deepEqual(
    tabs.map(tab => tab.properties.ariaLabel),
    ['workout analysis', 'lap splits', 'pace distribution'],
  )
  assert.deepEqual(
    tabs.map(tab => [tab.properties.role, tab.properties.ariaSelected, tab.properties.tabIndex]),
    [
      ['tab', 'true', 0],
      ['tab', 'false', -1],
      ['tab', 'false', -1],
    ],
  )
  const panels = byClass(workoutAnalysis, 'tri-workout-analysis-panel')
  assert.deepEqual(
    panels.map(panel => [
      panel.properties.role,
      panel.properties.dataWorkoutAnalysisPanel,
      panel.properties.hidden,
      panel.properties.ariaHidden,
    ]),
    [
      ['tabpanel', 'workout', undefined, 'false'],
      ['tabpanel', 'laps', true, 'true'],
      ['tabpanel', 'pace', true, 'true'],
    ],
  )
  assert.deepEqual(
    tabs.map(tab => tab.properties.ariaControls),
    panels.map(panel => [panel.properties.id]),
  )
  assert.deepEqual(
    panels.map(panel => panel.properties.inert),
    [undefined, true, true],
  )
  const embedded = buildActivity(factory, run, true, undefined, false, true)
  const embeddedTabs = byClass(embedded, 'tri-workout-analysis-tab')
  assert.deepEqual(embeddedTabs.map(text), ['WA', 'LS', 'PD'])
  assert.deepEqual(
    embeddedTabs.map(tab => tab.properties.ariaLabel),
    ['workout analysis', 'lap splits', 'pace distribution'],
  )
  assert.ok(workout)
  assert.equal(workout.tagName, 'section')
  assert.equal(workout.properties.ariaLabel, 'Run workout analysis')
  assert.equal(byClass(workout, 'tri-run-workout-plot')[0].properties.dataSiteCursorLine, '')
  assert.equal(byClass(workout, 'tri-run-workout-title').length, 0)
  assert.deepEqual(
    byClass(workout, 'tri-run-workout-stats')
      .flatMap(stat => byTag(stat, 'span'))
      .map(text),
    ['fastest 4:00 /km', 'avg 5:00 /km', 'slowest 6:00 /km'],
  )
  assert.deepEqual(byClass(workout, 'tri-run-workout-y-tick').map(text), [
    '4:00',
    '4:30',
    '5:00',
    '5:30',
    '6:00',
  ])
  assert.deepEqual(byClass(workout, 'tri-run-workout-label').map(text), ['1', '2', '3'])
  assert.deepEqual(byClass(workout, 'tri-run-workout-pace').map(text), [
    '5:00 /km',
    '6:00 /km',
    '4:00 /km',
  ])
  assert.equal(byClass(workout, 'tri-run-workout-column').length, 3)
  assert.ok(
    byClass(workout, 'tri-run-workout-pace').every(pace => pace.properties.ariaHidden === 'true'),
  )
  const workoutLaps = byClass(workout, 'tri-run-workout-lap')
  assert.deepEqual(
    workoutLaps.map(lap => [lap.properties.dataRangeKind, lap.properties.dataRangeId]),
    [
      ['lap', 'lap-1'],
      ['lap', 'lap-2'],
      ['lap', 'lap-3'],
    ],
  )
  assert.match(
    String(workoutLaps[0].properties.style),
    /--tri-run-workout-height:50\.000%;--tri-run-workout-opacity:0\.620/,
  )
  assert.match(String(workoutLaps[1].properties.style), /--tri-run-workout-height:3\.000%/)
  assert.match(String(workoutLaps[2].properties.style), /--tri-run-workout-height:100\.000%/)
  assert.equal(workoutLaps[0].properties.ariaPressed, 'false')
  assert.match(String(workoutLaps[0].properties.ariaLabel), /^Lap 1, 1\.00 km, \+4 m, 5:00/)
  assert.equal(workoutLaps[0].properties.dataDurationS, '300')
  assert.match(String(workoutLaps[1].properties.ariaLabel), /^Lap 2, 1\.00 km, \+5 m, 6:00/)
  assert.ok(splits)
  assert.equal(splits.tagName, 'section')
  assert.equal(splits.properties.ariaLabel, 'Run lap splits')
  assert.equal(byClass(splits, 'tri-run-splits-title').length, 0)
  assert.deepEqual(byClass(splits, 'tri-run-splits-average').map(text), ['avg 5:00 /km'])
  assert.deepEqual(
    byClass(splits, 'tri-run-splits-columns')[0]
      .children.filter((child): child is Element => child.type === 'element')
      .map(text),
    ['split', 'km', 'pace', '+/−'],
  )

  const rows = byClass(splits, 'tri-run-split')
  assert.deepEqual(
    rows.map(row => [row.properties.dataRangeKind, row.properties.dataRangeId]),
    [
      ['lap', 'lap-1'],
      ['lap', 'lap-2'],
      ['lap', 'lap-3'],
    ],
  )
  assert.deepEqual(byClass(splits, 'tri-run-split-lap').map(text), ['1', '2', '3'])
  assert.deepEqual(byClass(splits, 'tri-run-split-distance').map(text), ['1.00', '1.00', '1.00'])
  assert.deepEqual(byClass(splits, 'tri-run-split-pace').map(text), [
    '5:00 /km',
    '6:00 /km',
    '4:00 /km',
  ])
  assert.deepEqual(byClass(splits, 'tri-run-split-delta').map(text), ['—', '−1:00', '+2:00'])
  assert.equal(byClass(splits, 'tri-run-split-delta--slower').length, 1)
  assert.equal(byClass(splits, 'tri-run-split-delta--faster').length, 1)
  assert.match(String(rows[0].properties.style), /--tri-run-split-width:80\.000%/)
  assert.match(String(rows[0].properties.style), /--tri-run-split-average:80\.000%/)
  assert.equal(byClass(splits, 'tri-run-split-track').length, 3)
  assert.equal(byClass(splits, 'tri-run-split-fill').length, 3)
  assert.equal(byClass(splits, 'tri-run-split-average-marker').length, 3)
  assert.equal(rows[0].properties.ariaPressed, 'false')
  assert.match(String(rows[1].properties.ariaLabel), /−1:00 versus previous lap$/)

  const pace = byClass(workoutAnalysis, 'tri-run-pace-distribution')[0]
  assert.ok(pace)
  assert.equal(pace.properties.ariaLabel, 'Run pace distribution')
  assert.deepEqual(byClass(pace, 'tri-training-zone-summary-value').map(text), ['23% in zone 2'])
  assert.deepEqual(byClass(pace, 'tri-training-zone-summary-time').map(text), ['29:45'])
  assert.deepEqual(byClass(pace, 'tri-training-zone-name').map(text), [
    'Z6',
    'Z5',
    'Z4',
    'Z3',
    'Z2',
    'Z1',
  ])
  assert.deepEqual(byClass(pace, 'tri-training-zone-range').map(text), [
    '<4:23/km',
    '4:23–4:40/km',
    '4:40–5:00/km',
    '5:00–5:34/km',
    '5:34–6:27/km',
    '>6:27/km',
  ])
  assert.deepEqual(byClass(pace, 'tri-training-zone-source').map(text), [
    'based on 10 km race time 50:00',
  ])
  assert.equal(byClass(pace, 'tri-training-zone-row').length, 6)

  const bands = byClass(analysis, 'tri-analysis-band')
  assert.deepEqual(
    bands.map(band => band.properties.dataAnalysisKind),
    ['lap', 'segment', 'climb'],
  )
  assert.equal(byClass(analysis, 'tri-analysis-range').length, 5)
  assert.deepEqual(
    more.children
      .filter((child): child is Element => child.type === 'element')
      .slice(0, 2)
      .map(child => classNames(child)),
    [['tri-workout-analysis'], ['tri-elev-wrap', 'tri-elev-wrap--unavailable']],
  )
})

test('keeps the available workout-analysis tabs when pace telemetry is missing', () => {
  const run = analysisDetail()
  run.sport = 'run'
  const rendered = buildActivity(factory, run, true)
  assert.deepEqual(byClass(rendered, 'tri-workout-analysis-tab').map(text), [
    'workout analysis',
    'lap splits',
  ])
})

test('renders the configured 50 minute 10 km pace bands in imperial units', () => {
  const run = analysisDetail()
  run.sport = 'run'
  run.runPaceZones = {
    zoneSeconds: [354, 416, 397, 62, 227, 329],
    boundsSPerKm: [387.114, 333.676, 299.501, 280.238, 263.461],
    tenKmRaceTimeS: 3_000,
  }
  const rendered = buildActivity(factoryFor(imperialPresentation), run, true)
  assert.deepEqual(byClass(rendered, 'tri-training-zone-range').map(text), [
    '<7:04/mi',
    '7:04–7:31/mi',
    '7:31–8:02/mi',
    '8:02–8:57/mi',
    '8:57–10:23/mi',
    '>10:23/mi',
  ])
})

test('selects Strava metric or standard run splits from the active distance unit', () => {
  const run = analysisDetail()
  run.sport = 'run'
  run.runSplitsMetric = [
    {
      split: 1,
      distanceKm: 1,
      elapsedTimeS: 305,
      movingTimeS: 300,
      averageSpeedKph: 12,
      elevationDifferenceM: 4,
      paceZone: 2,
    },
    {
      split: 2,
      distanceKm: 0.5,
      elapsedTimeS: 190,
      movingTimeS: 180,
      averageSpeedKph: 10,
      elevationDifferenceM: -2,
      paceZone: 3,
    },
  ]
  run.runSplitsStandard = [
    {
      split: 1,
      distanceKm: 1.609344,
      elapsedTimeS: 490,
      movingTimeS: 480,
      averageSpeedKph: 12.07008,
      elevationDifferenceM: 5,
      paceZone: 2,
    },
    {
      split: 2,
      distanceKm: 0.804672,
      elapsedTimeS: 280,
      movingTimeS: 270,
      averageSpeedKph: 10.72896,
      elevationDifferenceM: -3,
      paceZone: 3,
    },
  ]

  const metric = buildActivity(factory, run, true)
  const metricSplits = byClass(metric, 'tri-run-splits')[0]
  assert.deepEqual(byClass(metricSplits, 'tri-run-split-distance').map(text), ['1.00', '0.50'])
  assert.deepEqual(byClass(metricSplits, 'tri-run-split-pace').map(text), ['5:00 /km', '6:00 /km'])
  assert.deepEqual(
    byClass(metricSplits, 'tri-run-split').map(row => row.properties.dataRangeId),
    ['split:metric:1', 'split:metric:2'],
  )

  const standard = buildActivity(factoryFor(imperialPresentation), run, true)
  const standardSplits = byClass(standard, 'tri-run-splits')[0]
  assert.deepEqual(byClass(standardSplits, 'tri-run-split-distance').map(text), ['1.00', '0.50'])
  assert.deepEqual(byClass(standardSplits, 'tri-run-split-pace').map(text), [
    '8:00 /mi',
    '9:00 /mi',
  ])
  assert.deepEqual(byClass(standardSplits, 'tri-run-split-delta').map(text), ['—', '−1:00'])
  assert.deepEqual(
    byClass(standardSplits, 'tri-run-split').map(row => row.properties.dataRangeId),
    ['split:standard:1', 'split:standard:2'],
  )
})

test('reserves fixed segment and climb lanes when an activity only has laps', () => {
  const lapsOnly = analysisDetail()
  lapsOnly.analysisRanges = lapsOnly.analysisRanges.filter(range => range.kind === 'lap')
  const rendered = buildActivity(factory, lapsOnly, true)
  const analysis = byClass(rendered, 'tri-analysis')[0]
  assert.ok(analysis)

  const bands = byClass(analysis, 'tri-analysis-band')
  assert.deepEqual(
    bands.map(band => band.properties.dataAnalysisKind),
    ['lap', 'segment', 'climb'],
  )
  assert.deepEqual(
    bands.flatMap(band =>
      byClass(band, 'tri-analysis-band-items').map(items => items.properties.style),
    ),
    ['--tri-analysis-lanes:1', '--tri-analysis-lanes:4', '--tri-analysis-lanes:1'],
  )
  for (const emptyBand of bands.slice(1)) {
    assert.equal(emptyBand.properties.role, undefined)
    assert.equal(emptyBand.properties.ariaLabel, undefined)
    assert.equal(emptyBand.properties.ariaHidden, 'true')
    assert.deepEqual(byClass(emptyBand, 'tri-analysis-band-label').map(text), [''])
    assert.equal(byClass(emptyBand, 'tri-analysis-range').length, 0)
  }
})

test('reserves an inaccessible analysis stack when a routed activity has no valid ranges', () => {
  const rendered = buildActivity(factory, detail({ analysisRanges: [] }), true)
  const analysis = byClass(rendered, 'tri-analysis')[0]
  assert.ok(analysis)
  assert.equal(analysis.properties.ariaLabel, undefined)
  assert.equal(analysis.properties.ariaHidden, 'true')

  const bands = byClass(analysis, 'tri-analysis-band')
  assert.deepEqual(
    bands.map(band => band.properties.dataAnalysisKind),
    ['lap', 'segment', 'climb'],
  )
  assert.deepEqual(
    bands.flatMap(band =>
      byClass(band, 'tri-analysis-band-items').map(items => items.properties.style),
    ),
    ['--tri-analysis-lanes:1', '--tri-analysis-lanes:4', '--tri-analysis-lanes:1'],
  )
  for (const band of bands) {
    assert.equal(band.properties.ariaHidden, 'true')
    assert.equal(byClass(band, 'tri-analysis-range').length, 0)
  }
})

test('starts the route and stream graphs with empty analysis highlights', () => {
  const rendered = buildActivity(factory, analysisDetail(), true)
  assert.equal(byClass(rendered, 'tri-analysis-map').length, 0)
  assert.equal(byClass(rendered, 'tri-analysis-graphs').length, 0)
  assert.equal(byClass(rendered, 'tri-analysis-trace').length, 0)

  const route = byClass(rendered, 'tri-route')[0]
  assert.ok(route)
  const selectedRoute = byClass(route, 'tri-route-selected')[0]
  assert.ok(selectedRoute)
  assert.equal(selectedRoute.properties.d, '')

  const selections = byClass(rendered, 'tri-analysis-selection')
  assert.equal(selections.length, 7)
  for (const selection of selections) {
    assert.equal(selection.tagName, 'rect')
    assert.equal(selection.properties.x, '0.00')
    assert.equal(selection.properties.width, '0.00')
  }

  const traces = byClass(rendered, 'tri-elev-wrap').filter(
    graph => graph.properties.dataTriTrace != null,
  )
  assert.deepEqual(
    traces.map(trace => trace.properties.dataTriTrace),
    ['hr', 'temperature', 'cadence', 'respiration', 'power', 'speed'],
  )
  assert.equal(byClass(rendered, 'tri-elev-cursor').length, 7)
})

test('keeps an empty selected-route overlay available after deselection', () => {
  const route = buildRoute(factory, analysisDetail().route)
  const selectedRoute = byClass(route, 'tri-route-selected')[0]
  assert.ok(selectedRoute)
  assert.equal(selectedRoute.properties.d, '')
  assert.match(String(byClass(route, 'tri-route-path')[0].properties.d), /^M /)
})

test('keeps virtual ride highlights on each chart distance domain', () => {
  const activity = { ...analysisDetail(), virtual: true, distanceKm: 31 }
  const rendered = buildActivity(factory, activity, true)
  const graphs = byClass(rendered, 'tri-elev')
  assert.equal(graphs.length, 7)
  for (const graph of graphs) {
    assert.equal(graph.properties.dataDomainStartDistanceKm, 0)
    assert.equal(graph.properties.dataDomainEndDistanceKm, activity.route.at(-1)?.d)
    assert.equal(byClass(graph, 'tri-analysis-selection').length, 1)
  }
})

test('keeps the run lap block visible when no lap is available', () => {
  const rendered = buildActivity(
    factory,
    detail({ sport: 'run', route: [], bestEfforts: null }),
    true,
  )
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const splits = byClass(more, 'tri-run-splits')[0]
  assert.ok(splits)
  assert.equal(splits.properties.ariaLabel, 'Run lap splits')
  assert.equal(byClass(splits, 'tri-run-splits-title').length, 0)
  assert.deepEqual(byClass(more, 'tri-workout-analysis-tab').map(text), ['lap splits'])
  assert.equal(byClass(more, 'tri-workout-analysis-tab')[0].properties.ariaSelected, 'true')
  assert.deepEqual(byClass(splits, 'tri-run-splits-columns').map(text), [''])
  assert.deepEqual(byClass(splits, 'tri-run-splits-empty').map(text), ['no lap found'])
  assert.equal(byClass(splits, 'tri-run-split').length, 0)
})

test('falls back to legacy stream traces without complete analysis telemetry', () => {
  const fallback = detail({
    analysisRanges: analysisRanges(),
    route: detail().route.map((point, index) =>
      index === 1 ? { ...point, speedKph: Number.NaN } : point,
    ),
  })
  const rendered = buildActivity(factory, fallback, true)
  assert.equal(byClass(rendered, 'tri-analysis').length, 0)
  const traces = byClass(rendered, 'tri-elev-wrap').filter(
    graph => graph.properties.dataTriTrace != null,
  )
  assert.deepEqual(
    traces.map(trace => trace.properties.dataTriTrace),
    ['hr', 'temperature', 'cadence', 'respiration', 'power', 'speed'],
  )
})

test('labels the activity disclosure and exposes its expanded state and controlled panel', () => {
  const collapsed = buildActivity(factory, detail({ id: 42 }))
  const collapsedToggle = byClass(collapsed, 'tri-act-toggle')[0]
  const collapsedPanel = byClass(collapsed, 'tri-act-more')[0]
  assert.ok(collapsedToggle)
  assert.ok(collapsedPanel)
  assert.equal(collapsed.properties.id, 'tri-activity-42')
  assert.equal(text(collapsedToggle), '+ see more')
  assert.equal(collapsedToggle.properties.ariaExpanded, 'false')
  assert.deepEqual(collapsedToggle.properties.ariaControls, ['tri-act-more-42'])
  assert.equal(collapsedPanel.properties.id, 'tri-act-more-42')

  const expanded = buildActivity(factory, detail({ id: 42 }), true)
  const expandedToggle = byClass(expanded, 'tri-act-toggle')[0]
  assert.ok(expandedToggle)
  assert.equal(text(expandedToggle), '− see less')
  assert.equal(expandedToggle.properties.ariaExpanded, 'true')
})

test('uses an explicit embed setting to choose the activity disclosure state', () => {
  const payload = {
    details: { 42: detail({ id: 42, date: '2026-08-18', sport: 'bike' }) },
    health: {},
  }
  const expanded = buildDayCard(factory, '2026-08-18', payload, {
    embedded: true,
    settings: { expanded: true },
  })
  const expandedActivity = byClass(expanded, 'tri-act')[0]
  const expandedToggle = byClass(expanded, 'tri-act-toggle')[0]
  assert.ok(classNames(expandedActivity).includes('tri-act--expanded'))
  assert.equal(expandedToggle.properties.ariaExpanded, 'true')

  const collapsed = buildDayCard(factory, '2026-08-18', payload, {
    embedded: true,
    sport: 'bike',
    settings: { expanded: false },
  })
  const collapsedActivity = byClass(collapsed, 'tri-act')[0]
  const collapsedToggle = byClass(collapsed, 'tri-act-toggle')[0]
  assert.equal(classNames(collapsedActivity).includes('tri-act--expanded'), false)
  assert.equal(collapsedToggle.properties.ariaExpanded, 'false')
})

test('marks every routed sport for the shared desktop figure split', () => {
  const routedSports: StravaActivityDetail['sport'][] = ['bike', 'run', 'walk']
  for (const sport of routedSports) {
    const rendered = buildActivity(factory, detail({ sport }), true)
    assert.equal(byClass(rendered, 'tri-act-figs--route').length, 1)
    assert.equal(byClass(rendered, 'tri-act-figs--split').length, 1)
  }

  const swim = buildActivity(
    factory,
    detail({ sport: 'swim', strokes: { freestyle: 1_500 } }),
    true,
  )
  assert.equal(byClass(swim, 'tri-act-figs--route').length, 1)
  assert.equal(byClass(swim, 'tri-act-figs--split').length, 1)

  const routeOnlySwim = detail({ id: 2026, date: '2026-07-26', sport: 'swim', strokes: null })
  const routeOnlyPayload = { details: { 2026: routeOnlySwim }, health: {} }
  const fullPageSwim = buildDayCard(factory, routeOnlySwim.date, routeOnlyPayload, {
    expanded: true,
  })
  assert.equal(byClass(fullPageSwim, 'tri-act-figs--route').length, 1)
  assert.equal(byClass(fullPageSwim, 'tri-act-figs--split').length, 0)
  assert.equal(byClass(fullPageSwim, 'tri-elev-unavailable').length, 1)

  const embeddedSwim = buildDayCard(factory, routeOnlySwim.date, routeOnlyPayload, {
    embedded: true,
  })
  assert.equal(byClass(embeddedSwim, 'tri-act-figs--route').length, 1)
  assert.equal(byClass(embeddedSwim, 'tri-act-figs--split').length, 1)
  const unavailable = byClass(embeddedSwim, 'tri-elev-unavailable')[0]
  assert.ok(unavailable)
  assert.equal(unavailable.tagName, 'div')
  assert.equal(text(unavailable), 'no data available')
  assert.equal(unavailable.properties.dataI18n, 'no data available')
  const unavailableWrap = byClass(embeddedSwim, 'tri-elev-wrap--unavailable')[0]
  assert.ok(unavailableWrap)
  assert.equal(byClass(unavailableWrap, 'tri-elev').length, 0)
  const unavailableCap = byClass(unavailableWrap, 'tri-elev-cap--unavailable')[0]
  assert.ok(unavailableCap)
  assert.equal(unavailableCap.properties.ariaHidden, 'true')
})

test('prefers active swim pace and adds stroke rate and count to the main stats', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'swim',
      distanceKm: 1,
      movingTimeS: 1_200,
      route: [],
      bestEfforts: null,
      swimPaceSPer100m: 95.4,
      strokeRateSpm: 31.5,
      strokeCount: 876,
      calculatedIntensityFactor: { value: 1.011, source: 'pace' },
      calculatedExerciseLoad: { value: 25.6, source: 'pace' },
      swimIntervals: [
        {
          startElapsedS: 0,
          endElapsedS: 25,
          distanceM: 25,
          durationS: 25,
          cumulativeDistanceM: 25,
          paceSPer100m: 100,
          strokeCount: 10,
          strokeTimeS: 25,
          strokeRateSpm: 24,
          stroke: 'freestyle',
        },
        {
          startElapsedS: 30,
          endElapsedS: 56,
          distanceM: 25,
          durationS: 26,
          cumulativeDistanceM: 50,
          paceSPer100m: 104,
          strokeCount: 11,
          strokeTimeS: 26,
          strokeRateSpm: 25.4,
          stroke: 'freestyle',
        },
      ],
      strokes: { freestyle: 800, breaststroke: 200 },
      swimLocation: 'pool',
      windKph: 18,
      windDir: 'SW',
      windGustKph: 31,
    }),
  )
  const stats = byClass(rendered, 'tri-act-stats')[0]
  assert.ok(stats)
  assert.equal(byClass(rendered, 'tri-act-figs--pool').length, 1)
  assert.deepEqual(bodyRows(stats), [
    ['distance', '1,000 m'],
    ['time', "20'"],
    ['pace', '1:35 /100m'],
    ['stroke rate', '32 spm'],
    ['avg hr', '148 bpm'],
    ['intensity factor', '1.011'],
    ['training effect', 'base'],
    ['exercise load', '26'],
    ['SWOLF', '36'],
    ['1.9k / 3.8k', "30' / 1h00'"],
    ['stroke type', 'freestyle'],
    ['strokes', '876 · 1.14 m/str'],
    ['NP', '205 W'],
    ['avg power', '188 W'],
    ['max power', '565 W'],
    ['energy', '900 kJ'],
    ['calories', '960 kcal'],
    ['cadence', '10.5 /length'],
    ['max hr', '171 bpm'],
    ['device temp', '24°C'],
    ['ambient temp', '22°C'],
    ['wind', '18 km/h SW / gust 31'],
  ])
})

test('keeps a missing swim stroke rate visible as an em dash', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'swim',
      distanceKm: 1.5,
      movingTimeS: 2_460,
      route: [],
      bestEfforts: null,
      strokeRateSpm: null,
      strokeCount: null,
      avgCadence: null,
      swimLocation: 'pool',
    }),
  )
  const stats = byClass(rendered, 'tri-act-stats')[0]
  assert.ok(stats)
  assert.deepEqual(bodyRows(stats).slice(0, 10), [
    ['distance', '1,500 m'],
    ['time', "41'"],
    ['pace', '2:44 /100m'],
    ['stroke rate', '—'],
    ['avg hr', '148 bpm'],
    ['training effect', 'base'],
    ['SWOLF', '—'],
    ['1.9k / 3.8k', "52' / 1h44'"],
    ['stroke type', 'freestyle'],
    ['strokes', '—'],
  ])
})

test('keeps water temperature and adds the full open-water swim profile', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'swim',
      distanceKm: 1.5,
      movingTimeS: 2_460,
      route: [],
      bestEfforts: null,
      strokeRateSpm: 31.5,
      strokeCount: null,
      avgCadence: null,
      swimLocation: 'openWater',
      waterTemperatureC: 14.4,
    }),
  )
  const stats = byClass(rendered, 'tri-act-stats')[0]
  assert.ok(stats)
  assert.deepEqual(bodyRows(stats).slice(0, 11), [
    ['distance', '1,500 m'],
    ['time', "41'"],
    ['pace', '2:44 /100m'],
    ['stroke rate', '32 spm'],
    ['avg hr', '148 bpm'],
    ['training effect', 'base'],
    ['water temp', '14°C'],
    ['SWOLF', '—'],
    ['1.9k / 3.8k', "52' / 1h44'"],
    ['stroke type', 'freestyle'],
    ['strokes', '—'],
  ])
})

test('adds max speed directly below the bike speed row', () => {
  const metric = buildActivity(factory, detail({ maxSpeedKph: 41.8 }))
  const metricStats = byClass(metric, 'tri-act-stats')[0]
  assert.ok(metricStats)
  assert.deepEqual(bodyRows(metricStats), [
    ['distance', '30.0 km'],
    ['time', "1h20'"],
    ['speed', '22.5 km/h'],
    ['max speed', '41.8 km/h'],
    ['avg hr', '148 bpm'],
    ['training effect', 'base'],
    ['NP', '205 W'],
    ['avg power', '188 W'],
    ['max power', '565 W'],
    ['energy', '900 kJ'],
    ['calories', '960 kcal'],
    ['cadence', '88 rpm'],
    ['max hr', '171 bpm'],
    ['device temp', '24°C'],
    ['ambient temp', '22°C'],
  ])

  const imperial = buildActivity(factoryFor(imperialPresentation), detail({ maxSpeedKph: 41.8 }))
  const imperialStats = byClass(imperial, 'tri-act-stats')[0]
  assert.ok(imperialStats)
  assert.deepEqual(bodyRows(imperialStats), [
    ['distance', '18.6 mi'],
    ['time', "1h20'"],
    ['speed', '14.0 mph'],
    ['max speed', '26.0 mph'],
    ['avg hr', '148 bpm'],
    ['training effect', 'base'],
    ['NP', '205 W'],
    ['avg power', '188 W'],
    ['max power', '565 W'],
    ['energy', '900 kJ'],
    ['calories', '960 kcal'],
    ['cadence', '88 rpm'],
    ['max hr', '171 bpm'],
    ['device temp', '75°F'],
    ['ambient temp', '72°F'],
  ])
  const withoutMax = buildActivity(factory, detail())
  const plainStats = byClass(withoutMax, 'tri-act-stats')[0]
  assert.ok(plainStats)
  assert.deepEqual(
    bodyRows(plainStats).map(([label]) => label),
    [
      'distance',
      'time',
      'speed',
      'avg hr',
      'training effect',
      'NP',
      'avg power',
      'max power',
      'energy',
      'calories',
      'cadence',
      'max hr',
      'device temp',
      'ambient temp',
    ],
  )
})

test('places one combined stats table above route figures with a disclosure for every activity', () => {
  const rendered = buildActivity(factory, detail(), true)
  const children = rendered.children.filter((child): child is Element => child.type === 'element')
  const statsIndex = children.findIndex(child => classNames(child).includes('tri-act-stats'))
  const figuresIndex = children.findIndex(child => classNames(child).includes('tri-act-figs'))
  assert.ok(statsIndex >= 0)
  assert.ok(figuresIndex > statsIndex)
  assert.equal(byClass(rendered, 'tri-act-stats').length, 1)
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  assert.equal(byClass(more, 'tri-act-stats').length, 0)

  const rowsOnly = buildActivity(
    factory,
    detail({ route: [], analysisRanges: [], bestEfforts: null, deviceWatts: false }),
  )
  assert.equal(byClass(rowsOnly, 'tri-act-stats').length, 1)
  assert.ok(
    bodyRows(byClass(rowsOnly, 'tri-act-stats')[0]).some(([label]) => label === 'est power'),
  )
  assert.equal(byClass(rowsOnly, 'tri-act-toggle').length, 1)
  assert.equal(byClass(rowsOnly, 'tri-act-more').length, 1)
})

test('projects each sub-marathon run to the next standard race distance', () => {
  const trendLabel = (distanceKm: number): string | null =>
    activityStatRows(
      METRIC_TRIATHLON_PRESENTATION,
      detail({ sport: 'run', distanceKm, movingTimeS: 3_600, maxSpeedKph: null }),
    ).find(([label]) => label.endsWith(' trend'))?.[0] ?? null

  assert.deepEqual([4.999, 5, 9.999, 10, 21.0974, 21.0975, 42.194, 42.195, 50].map(trendLabel), [
    '5k trend',
    '10k trend',
    '10k trend',
    'half trend',
    'half trend',
    'marathon trend',
    'marathon trend',
    null,
    null,
  ])
})

test('renders the run trend between pace and heart rate in server activity markup', () => {
  const rendered = buildActivity(
    factory,
    detail({ sport: 'run', distanceKm: 11.1, movingTimeS: 4_320, maxSpeedKph: null }),
  )
  const stats = byClass(rendered, 'tri-act-stats')[0]
  assert.ok(stats)
  assert.deepEqual(bodyRows(stats), [
    ['distance', '11.1 km'],
    ['time', "1h12'"],
    ['pace', '6:29 /km'],
    ['half trend', "2h22'"],
    ['avg hr', '148 bpm'],
    ['training effect', 'base'],
    ['NP', '205 W'],
    ['avg power', '188 W'],
    ['max power', '565 W'],
    ['energy', '900 kJ'],
    ['calories', '960 kcal'],
    ['cadence', '176 spm'],
    ['max hr', '171 bpm'],
    ['device temp', '24°C'],
    ['ambient temp', '22°C'],
  ])
})

test('keeps a missing run cadence visible as an em dash', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'run',
      distanceKm: 9.5,
      movingTimeS: 3_220,
      maxSpeedKph: null,
      avgCadence: null,
    }),
  )
  const stats = byClass(rendered, 'tri-act-stats')[0]
  assert.ok(stats)
  const rows = bodyRows(stats)
  const cadenceIndex = rows.findIndex(([label]) => label === 'cadence')
  assert.deepEqual(rows.slice(cadenceIndex - 1, cadenceIndex + 2), [
    ['calories', '960 kcal'],
    ['cadence', '—'],
    ['max hr', '171 bpm'],
  ])
})

test('renders estimated walking power in cards, map metrics, and the workspace while retaining native power', () => {
  const time = [0, 10, 20, 30, 40, 50, 60]
  const estimate = buildWalkPowerEstimate({
    inputSource: 'strava',
    weight: { kg: 80, date: '2026-09-29', source: 'garmin' },
    streams: { time, distance: time, altitude: time.map(() => 100) },
    elapsedTimeS: 60,
  })
  assert.ok(estimate)
  const base = detail()
  const activity = detail({
    sport: 'walk',
    date: '2026-09-29',
    deviceWatts: false,
    avgWatts: null,
    distanceKm: 0.06,
    elapsedTimeS: 60,
    movingTimeS: 60,
    walkPower: estimate,
    route: time.map((elapsedS, index) => ({ ...base.route[0], elapsedS, d: index / 100, w: 0 })),
  })
  assert.ok(isActivityDetail(JSON.parse(JSON.stringify(activity))))
  assert.equal(
    isActivityDetail({ ...activity, walkPower: { ...estimate, source: 'garmin' } }),
    false,
  )
  assert.equal(isActivityDetail({ ...activity, sport: 'run' }), false)
  assert.equal(
    isActivityDetail({
      ...activity,
      walkPower: { ...estimate, points: estimate.points.map(point => ({ ...point, watts: -1 })) },
    }),
    false,
  )
  for (const embedded of [false, true]) {
    const card = buildActivity(factory, activity, true, undefined, false, embedded)
    const chart = descendants(card, node => node.properties.dataTriTrace === 'walking-power')[0]
    assert.ok(chart)
    assert.equal(chart.properties.dataWalkingPowerSource, 'garden-estimate')
    assert.match(text(chart), /walking power.*200 W.*calculated/)
    assert.deepEqual(
      moreStatRows(METRIC_TRIATHLON_PRESENTATION, activity).find(
        ([label]) => label === 'walking power',
      ),
      ['walking power', '200 W · estimated'],
    )
  }
  const workspace = workspaceTraces(activity, METRIC_TRIATHLON_PRESENTATION)
  assert.equal(workspace.find(trace => trace.id === 'walk-power')?.estimated, true)
  const map = metricSpecs(METRIC_TRIATHLON_PRESENTATION, activity, ctx()).find(
    metric => metric.label === 'walking power',
  )
  assert.ok(map)
  assert.equal(map.pick(activity.route[3], 3), 200)
  const native = detail({
    ...activity,
    deviceWatts: true,
    avgWatts: 123,
    route: activity.route.map(point => ({ ...point, w: 123 })),
  })
  const card = buildActivity(factory, native, true)
  assert.equal(
    descendants(card, node => node.properties.dataTriTrace === 'walking-power').length,
    0,
  )
  assert.equal(descendants(card, node => node.properties.dataTriTrace === 'power').length, 1)
  assert.equal(
    workspaceTraces(native, METRIC_TRIATHLON_PRESENTATION).filter(
      trace => trace.id === 'walk-power',
    ).length,
    0,
  )
  assert.ok(
    metricSpecs(METRIC_TRIATHLON_PRESENTATION, native, ctx()).some(
      metric => metric.label === 'power',
    ),
  )
})

test('renders Garmin walk pace, cadence, respiration, and elevation', () => {
  const base = detail()
  const route = base.route.map((point, index) => ({
    ...point,
    d: (0.618 * index) / (base.route.length - 1),
    alt: [86.4, 91.6, 89.2, 104.8][index],
    cad: [48, 47, 49, 46][index],
    resp: [19, 26, 32, 27][index],
    elapsedS: (403 * index) / (base.route.length - 1),
    speedKph: [4.2, 5.8, 4.8, 5.1][index],
  }))
  const activity = detail({
    sport: 'walk',
    deviceWatts: false,
    distanceKm: 0.618,
    movingTimeS: 403,
    elapsedTimeS: 475,
    elevationM: 18,
    descentM: 1,
    minAlt: 86.4,
    maxAlt: 104.8,
    avgCadence: 56,
    garmin: garminVerification({ avgCadence: 95 }),
    route,
  })

  assert.deepEqual(
    activityTableRows(imperialPresentation, activity).find(([label]) => label === 'cadence'),
    ['cadence', '95 spm'],
  )

  const rendered = buildActivity(factoryFor(imperialPresentation), activity, true)
  const traces = byClass(rendered, 'tri-elev-wrap').filter(
    graph => graph.properties.dataTriTrace != null,
  )
  assert.deepEqual(
    traces.map(graph => graph.properties.dataTriTrace),
    ['hr', 'temperature', 'cadence', 'respiration', 'pace'],
  )
  const pace = traces.find(graph => graph.properties.dataTriTrace === 'pace')
  const cadence = traces.find(graph => graph.properties.dataTriTrace === 'cadence')
  assert.ok(pace)
  assert.ok(cadence)
  assert.match(text(byClass(pace, 'tri-elev-cap')[0]), /pace.*\/mi avg/)
  assert.ok(
    byClass(cadence, 'tri-cax-yt')
      .map(text)
      .every(label => label === '0' || label.endsWith('spm')),
  )

  const specs = metricSpecs(imperialPresentation, activity, ctx())
  assert.deepEqual(
    specs.map(spec => spec.label),
    ['pace', 'heart rate', 'cadence', 'respiration', 'elevation', 'temperature'],
  )
  const cadenceSpec = specs.find(spec => spec.label === 'cadence')
  assert.ok(cadenceSpec)
  assert.equal(cadenceSpec.pick(route[0], 0), 96)
  assert.equal(cadenceSpec.fmt(96), '96 spm')
})

const swimTrendDetail = (overrides: Partial<StravaActivityDetail> = {}): StravaActivityDetail =>
  detail({
    id: 5,
    sport: 'swim',
    name: 'Pool swim',
    date: '2026-07-05',
    start: '2026-07-05T12:00:00Z',
    distanceKm: 0.1,
    movingTimeS: 100,
    route: [],
    bestEfforts: null,
    swimPaceSPer100m: 100,
    strokeRateSpm: 28,
    strokeCount: 700,
    swimDurationS: 180,
    swimLocation: 'pool',
    swimIntervals: [
      {
        startElapsedS: 0,
        endElapsedS: 25,
        distanceM: 25,
        durationS: 25,
        cumulativeDistanceM: 25,
        paceSPer100m: 100,
        strokeCount: 10,
        strokeTimeS: 25,
        strokeRateSpm: 24,
        stroke: 'freestyle',
      },
      {
        startElapsedS: 40,
        endElapsedS: 66,
        distanceM: 25,
        durationS: 26,
        cumulativeDistanceM: 50,
        paceSPer100m: 104,
        strokeCount: 11,
        strokeTimeS: 25.4,
        strokeRateSpm: 26,
        stroke: 'freestyle',
      },
      {
        startElapsedS: 80,
        endElapsedS: 105,
        distanceM: 25,
        durationS: 25,
        cumulativeDistanceM: 75,
        paceSPer100m: 100,
        strokeCount: null,
        strokeTimeS: null,
        strokeRateSpm: null,
        stroke: 'kickboard',
      },
      {
        startElapsedS: 120,
        endElapsedS: 144,
        distanceM: 25,
        durationS: 24,
        cumulativeDistanceM: 100,
        paceSPer100m: 96,
        strokeCount: 12,
        strokeTimeS: 24,
        strokeRateSpm: 30,
        stroke: 'freestyle',
      },
    ],
    ...overrides,
  })

const swimToggleDetail = (): StravaActivityDetail => {
  const durations = [25, 26, 25, 24, 30, 29, 31, 30]
  const swimIntervals: SwimActivityInterval[] = durations.map((durationS, index) => {
    const firstBlock = index < 4
    const strokeCount = firstBlock ? 8 : 12
    const strokeTimeS = firstBlock ? 20 : 24
    return {
      startElapsedS: index * 40,
      endElapsedS: index * 40 + durationS,
      distanceM: 25,
      durationS,
      cumulativeDistanceM: (index + 1) * 25,
      paceSPer100m: durationS * 4,
      strokeCount,
      strokeTimeS,
      strokeRateSpm: (strokeCount / strokeTimeS) * 60,
      stroke: 'freestyle',
    }
  })
  return swimTrendDetail({
    distanceKm: 0.2,
    movingTimeS: 220,
    swimPaceSPer100m: 110,
    strokeRateSpm: 27.3,
    swimDurationS: 310,
    swimIntervals,
  })
}

test('pool analysis accepts recorded telemetry without GPS and omits empty activities', () => {
  const pool = swimTrendDetail({ heartRateTrace: [] })
  const button = buildActivityAnalyzeButton(factory, pool)
  assert.equal(button.properties.disabled, undefined)
  assert.equal(button.properties.title, 'analyze')
  const empty = buildActivityAnalyzeButton(factory, detail({ route: [], heartRateTrace: [] }))
  assert.equal(empty.properties.disabled, true)
  assert.equal(empty.properties.title, 'No recorded telemetry for this activity.')
})

test('pool overlays retain measured distance, length metrics, rest gaps, and missing strokes', () => {
  const pool = swimTrendDetail({
    heartRateTrace: [heartRateTracePoint(0, 0, 100), heartRateTracePoint(0.1, 144, 130)],
  })
  const traces = workspaceTraces(pool, METRIC_TRIATHLON_PRESENTATION)
  const pace = traces.find(trace => trace.id === 'swim-pace')
  const cadence = traces.find(trace => trace.id === 'swim-cadence')
  const swolf = traces.find(trace => trace.id === 'swolf')
  const heartRate = traces.find(trace => trace.id === 'hr')
  assert.ok(pace && cadence && swolf && heartRate)
  assert.equal(pace.format(104), '1:44 /100m')
  assert.equal(workspaceValueAt(pace, 12), 100)
  assert.equal(workspaceValueAt(pace, 30), null)
  assert.equal(workspaceValueAt(pace, 50), 104)
  assert.equal(workspaceValueAt(cadence, 90), null)
  assert.equal(workspaceValueAt(swolf, 50), 37)
  assert.equal(heartRate.samples.at(-1)?.distanceKm, 0.1)
  assert.equal(pace.samples.at(-1)?.distanceKm, 0.1)
  assert.ok(workspaceTracePaths(pace, 'time', 0, 144).line.split('M').length > 2)
  const timeline = workspaceTimeline(pool)
  assert.deepEqual(workspaceLocationAt(timeline, 'time', 12.5), {
    elapsedS: 12.5,
    distanceKm: 0.0125,
  })
  assert.deepEqual(workspaceLocationAt(timeline, 'time', 30), { elapsedS: 30, distanceKm: 0.025 })
  assert.deepEqual(workspaceLocationAt(timeline, 'distance', 0.0375), {
    elapsedS: 53,
    distanceKm: 0.0375,
  })
})

test('pool overlays connect rounded length boundaries while preserving recorded rests', () => {
  const pool = swimTrendDetail()
  const first = pool.swimIntervals[0]
  const second = pool.swimIntervals[1]
  pool.swimIntervals = [first, { ...second, startElapsedS: 25.4, endElapsedS: 51.4 }]
  const pace = workspaceTraces(pool, METRIC_TRIATHLON_PRESENTATION).find(
    trace => trace.id === 'swim-pace',
  )
  assert.ok(pace)
  assert.equal(workspaceValueAt(pace, 25.2), 104)
  assert.equal(workspaceTracePaths(pace, 'time', 0, 60).line.split('M').length, 2)
  const location = workspaceLocationAt(workspaceTimeline(pool), 'time', 25.2)
  assert.equal(location.elapsedS, 25.2)
  assert.ok(Math.abs(location.distanceKm - 0.025189393939) < 1e-10)
})

test('workspace range clipping retains a continuous length spanning both range edges', () => {
  const pace = workspaceTraces(swimTrendDetail(), METRIC_TRIATHLON_PRESENTATION).find(
    trace => trace.id === 'swim-pace',
  )
  assert.ok(pace)
  assert.match(workspaceTracePaths(pace, 'time', 10, 20).line, /^M 0 [\d.]+ L 100 [\d.]+/)
  assert.equal(workspaceTracePaths(pace, 'time', 26, 39).line, '')
})

test('pool swimming retains calculated condition and recorded physiology without HR stamina', () => {
  const pool = swimTrendDetail({
    elapsedTimeS: 600,
    heartRateTrace: Array.from({ length: 61 }, (_, i) =>
      heartRateTracePoint(i / 100, i * 10, i <= 6 ? 140 : 160),
    ),
  })
  applyHeartRatePhysiology(pool, 200)
  assert.ok(pool.heartRatePhysiology)
  assert.equal(pool.heartRatePhysiology.points.at(-1)?.performanceCondition, -10)
  assert.ok(
    pool.heartRatePhysiology.points.every(
      point => point.stamina == null && point.potentialStamina == null,
    ),
  )
  const rendered = buildActivity(factory, pool, true)
  assert.equal(byClass(rendered, 'tri-stamina-chart').length, 0)
  const condition = descendants(
    rendered,
    node => node.properties.dataTriTrace === 'performance-condition',
  )[0]
  assert.ok(condition)
  assert.equal(condition.properties.dataPerformanceConditionSource, 'garden-estimate')
  const source = byClass(condition, 'tri-performance-condition-source')[0]
  assert.equal(text(source), 'calculated')
  assert.match(String(source.properties.dataGlossDef), /Garden HR change proxy/)
  const traces = workspaceTraces(pool, METRIC_TRIATHLON_PRESENTATION)
  assert.ok(traces.some(trace => trace.id === 'hr'))
  assert.ok(traces.some(trace => trace.id === 'condition' && trace.estimated))
  assert.ok(traces.every(trace => trace.id !== 'stamina'))
  const native = detail({ sport: 'swim' })
  native.route = native.route.map(point => ({ ...point, stamina: 70, potentialStamina: 80 }))
  applyHeartRatePhysiology(native, 200)
  assert.ok(buildStaminaChart(factory, native))
})

const swimTrendPoints: SwimTrendPoint[] = [
  {
    id: 1,
    date: '2026-07-01',
    start: '2026-07-01T12:00:00Z',
    paceSPer100m: 112,
    paceSource: 'stroke',
    strokeRateSpm: 20,
  },
  {
    id: 2,
    date: '2026-07-02',
    start: '2026-07-02T12:00:00Z',
    paceSPer100m: 110,
    paceSource: 'stroke',
    strokeRateSpm: 22,
  },
  {
    id: 3,
    date: '2026-07-03',
    start: '2026-07-03T12:00:00Z',
    paceSPer100m: 108,
    paceSource: 'stroke',
    strokeRateSpm: 24,
  },
  {
    id: 4,
    date: '2026-07-04',
    start: '2026-07-04T12:00:00Z',
    paceSPer100m: 106,
    paceSource: 'stroke',
    strokeRateSpm: 26,
  },
  {
    id: 5,
    date: '2026-07-05',
    start: '2026-07-05T12:00:00Z',
    paceSPer100m: 100,
    paceSource: 'stroke',
    strokeRateSpm: 28,
  },
  {
    id: 7,
    date: '2026-07-05',
    start: '2026-07-05T18:00:00Z',
    paceSPer100m: 90,
    paceSource: 'stroke',
    strokeRateSpm: 40,
  },
  {
    id: 6,
    date: '2026-07-06',
    start: '2026-07-06T12:00:00Z',
    paceSPer100m: 90,
    paceSource: 'stroke',
    strokeRateSpm: 40,
  },
]

test('renders aligned swim trends with the selected activity average', () => {
  const rendered = buildSwimTrends(factory, swimTrendDetail())
  assert.ok(rendered)
  assert.equal(rendered.tagName, 'section')
  assert.equal(rendered.properties.ariaLabel, 'Swim activity analysis')
  assert.deepEqual(
    byClass(rendered, 'tri-swim-trend').map(chart => chart.properties.dataTriTrace),
    ['pace', 'stroke-rate', 'cadence', 'swolf'],
  )
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-title').map(text), [
    'pace /100m',
    'stroke rate spm',
    'cadence str/length',
    'SWOLF',
  ])
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-value').map(text), ['1:40', '28', '11', '36'])
  assert.equal(byClass(rendered, 'tri-swim-trend-delta').length, 0)

  const pace = byClass(rendered, 'tri-swim-trend--pace')[0]
  const cadence = byClass(rendered, 'tri-swim-trend--cadence')[0]
  const swolf = byClass(rendered, 'tri-swim-trend--swolf')[0]
  assert.ok(pace)
  assert.ok(cadence)
  assert.ok(swolf)
  assert.equal(byClass(rendered, 'tri-swim-chart-grid').length, 1)
  assert.equal(byClass(rendered, 'tri-swim-mode-toggle').length, 0)
  assert.ok(classNames(pace).includes('tri-zone'))
  assert.ok(classNames(cadence).includes('tri-zone'))
  assert.ok(classNames(swolf).includes('tri-zone'))
  assert.deepEqual(byClass(pace, 'tri-cax-xt').map(text), ['0 m', '50 m', '100 m'])
  assert.deepEqual(
    byClass(cadence, 'tri-cax-xt').map(tick => [text(tick), tick.properties.style]),
    byClass(pace, 'tri-cax-xt').map(tick => [text(tick), tick.properties.style]),
  )
  const paceSvg = byClass(pace, 'tri-swim-trend-svg')[0]
  assert.ok(paceSvg)
  for (const graph of byClass(rendered, 'tri-swim-trend-svg')) {
    assert.equal(graph.properties.dataDomainStartDistanceKm, 0)
    assert.equal(graph.properties.dataDomainEndDistanceKm, 0.1)
    const selections = byClass(graph, 'tri-analysis-selection')
    assert.equal(selections.length, 1)
    assert.equal(selections[0].properties.x, '0.00')
    assert.equal(selections[0].properties.width, '0.00')
    assert.equal(selections[0].properties.height, 30)
  }
  assert.deepEqual(byClass(pace, 'tri-cax-yt').map(text), ['0:00', '0:50', '1:40', '2:30'])
  assert.deepEqual(byClass(cadence, 'tri-cax-yt').map(text), [
    '10.0',
    '10.5',
    '11.0',
    '11.5',
    '12.0',
  ])
  assert.deepEqual(byClass(swolf, 'tri-cax-yt').map(text), ['35.0', '35.5', '36.0', '36.5', '37.0'])
  assert.equal(paceSvg.properties.role, 'slider')
  assert.equal(paceSvg.properties.tabIndex, 0)
  assert.equal(paceSvg.properties.ariaOrientation, 'horizontal')
  assert.equal(paceSvg.properties.ariaValueMin, 0)
  assert.equal(paceSvg.properties.ariaValueMax, 100)
  assert.equal(paceSvg.properties.ariaValueNow, 100)
  assert.match(
    String(paceSvg.properties.ariaValueText),
    /100 metres, 2:24 elapsed, swim pace 1:36 per 100 metres\. Activity average 1:40 \/100m\./,
  )
  assert.equal(paceSvg.properties.dataSwimKind, 'pace')
  assert.equal(paceSvg.properties.dataSwimIndex, 3)
  const paceSeries = JSON.parse(
    String(paceSvg.properties.dataSwimSeriesLengths),
  ) as SwimTrendChartPoint[]
  assert.deepEqual(paceSeries[0], {
    elapsedS: 25,
    cumulativeDistanceM: 25,
    value: 100,
    xPct: 25,
    yPct: 66.66666666666666,
  })
  assert.deepEqual(paceSeries.at(-1), {
    elapsedS: 144,
    cumulativeDistanceM: 100,
    value: 96,
    xPct: 100,
    yPct: 64,
  })
  const pacePath = byClass(paceSvg, 'tri-swim-trend-line')[0]
  const paceArea = byClass(paceSvg, 'tri-swim-trend-area')[0]
  assert.ok(pacePath)
  assert.ok(paceArea)
  assert.match(
    String(pacePath.properties.d),
    /^M 0\.00 20\.00 L 25\.00 20\.00 .* L 100\.00 19\.20$/,
  )
  assert.match(
    String(paceArea.properties.d),
    /^M 0\.00 30 L 0\.00 20\.00 L 25\.00 20\.00 .* L 100\.00 19\.20 L 100\.00 30 Z$/,
  )
  assert.equal(byClass(rendered, 'tri-swim-trend-current').length, 0)
  assert.equal(byClass(rendered, 'tri-swim-trend-area').length, 4)
  assert.deepEqual(
    byClass(rendered, 'tri-swim-trend-hover').map(point => point.properties.hidden),
    [true, true, true, true],
  )
  assert.equal(byClass(rendered, 'tri-chart-cursor').length, 4)
  assert.deepEqual(byClass(pace, 'tri-swim-trend-readout').map(text), [
    '100 m · 2:24 elapsed1:36 /100m',
  ])
})

test('renders one shared lengths and 100 metre toggle for all swim charts', () => {
  const rendered = buildSwimTrends(factory, swimToggleDetail())
  assert.ok(rendered)
  const toggle = byClass(rendered, 'tri-swim-mode-toggle')[0]
  assert.ok(toggle)
  assert.equal(rendered.properties.dataI18nAriaLabel, 'swim activity analysis')
  assert.equal(toggle.properties.role, 'group')
  assert.equal(toggle.properties.ariaLabel, 'swim chart aggregation')
  assert.equal(toggle.properties.dataSwimMode, 'lengths')
  const paceHead = byClass(byClass(rendered, 'tri-swim-trend--pace')[0], 'tri-swim-trend-head')[0]
  assert.ok(paceHead)
  assert.equal(byClass(paceHead, 'tri-swim-mode-toggle').length, 1)
  assert.equal(byClass(paceHead, 'tri-swim-trend-title').length, 0)
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-title').map(text), [
    'stroke rate spm',
    'cadence str/length',
    'SWOLF',
  ])
  assert.deepEqual(
    byClass(toggle, 'tri-swim-mode').map(button => [
      text(button),
      button.properties.dataSwimMode,
      button.properties.ariaPressed,
    ]),
    [
      ['lengths', 'lengths', 'true'],
      ['100 m', '100m', 'false'],
    ],
  )

  const paceSvg = byClass(rendered, 'tri-swim-trend-svg--pace')[0]
  const cadenceSvg = byClass(rendered, 'tri-swim-trend-svg--cadence')[0]
  const swolfSvg = byClass(rendered, 'tri-swim-trend-svg--swolf')[0]
  assert.ok(paceSvg)
  assert.ok(cadenceSvg)
  assert.ok(swolfSvg)
  const paceLengths = JSON.parse(
    String(paceSvg.properties.dataSwimSeriesLengths),
  ) as SwimTrendChartPoint[]
  const paceHundreds = JSON.parse(
    String(paceSvg.properties.dataSwimSeriesHundred),
  ) as SwimTrendChartPoint[]
  const cadenceHundreds = JSON.parse(
    String(cadenceSvg.properties.dataSwimSeriesHundred),
  ) as SwimTrendChartPoint[]
  const swolfHundreds = JSON.parse(
    String(swolfSvg.properties.dataSwimSeriesHundred),
  ) as SwimTrendChartPoint[]
  assert.equal(paceLengths.length, 8)
  assert.deepEqual(
    paceHundreds.map(point => [
      point.windowStartDistanceM,
      point.cumulativeDistanceM,
      point.elapsedS,
      point.value,
      point.xPct,
    ]),
    [
      [0, 100, 144, 100, 50],
      [100, 200, 310, 120, 100],
    ],
  )
  assert.deepEqual(
    cadenceHundreds.map(point => [point.cumulativeDistanceM, point.value]),
    [
      [100, 8],
      [200, 12],
    ],
  )
  assert.deepEqual(
    swolfHundreds.map(point => [point.cumulativeDistanceM, point.value]),
    [
      [100, 33],
      [200, 42],
    ],
  )
  assert.match(
    String(byClass(paceSvg, 'tri-swim-trend-line--100m')[0]?.properties.d),
    /^M 0\.00 .* L 50\.00 .* L 50\.00 .* L 100\.00/,
  )
  assert.match(
    String(byClass(cadenceSvg, 'tri-swim-trend-area--100m')[0]?.properties.d),
    /^M 0\.00 30 L 0\.00 .* L 50\.00 .* L 50\.00 .* L 100\.00 .* L 100\.00 30 Z$/,
  )
  assert.equal(paceSvg.properties.dataSwimMode, 'lengths')
  assert.equal(cadenceSvg.properties.dataSwimMode, 'lengths')
  assert.equal(swolfSvg.properties.dataSwimMode, 'lengths')
  assert.equal(byClass(rendered, 'tri-swim-series').length, 8)
  assert.equal(byClass(rendered, 'tri-swim-series--active').length, 4)
  assert.equal(byClass(rendered, 'tri-swim-trend-area').length, 8)
  assert.equal(byClass(rendered, 'tri-swim-trend-current').length, 0)
})

test('plots only the selected swim intervals even when history contains same-date activities', () => {
  const rendered = buildSwimTrends(
    factory,
    swimTrendDetail({
      id: 7,
      start: '2026-07-05T18:00:00Z',
      swimPaceSPer100m: 90,
      strokeRateSpm: 40,
    }),
  )
  assert.ok(rendered)
  const paceSvg = byClass(rendered, 'tri-swim-trend-svg--pace')[0]
  assert.ok(paceSvg)
  const series = JSON.parse(
    String(paceSvg.properties.dataSwimSeriesLengths),
  ) as SwimTrendChartPoint[]

  assert.deepEqual(
    series.map(point => [point.cumulativeDistanceM, point.elapsedS, point.xPct]),
    [
      [25, 25, 25],
      [50, 66, 50],
      [75, 105, 75],
      [100, 144, 100],
    ],
  )
  assert.doesNotMatch(String(paceSvg.properties.ariaValueText), /Jul|2026/)
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-readout-position').map(text), [
    '100 m · 2:24 elapsed',
    '100 m · 2:24 elapsed',
    '100 m · 2:24 elapsed',
    '100 m · 2:24 elapsed',
  ])
})

test('filters swim traces before assigning the shared aggregation toggle', () => {
  const rendered = buildSwimTrends(factory, swimToggleDetail(), {
    pace: false,
    'stroke-rate': false,
  })

  assert.ok(rendered)
  assert.deepEqual(
    byClass(rendered, 'tri-swim-trend').map(chart => chart.properties.dataTriTrace),
    ['cadence', 'swolf'],
  )
  const cadence = byClass(rendered, 'tri-swim-trend--cadence')[0]
  assert.ok(cadence)
  assert.equal(byClass(cadence, 'tri-swim-mode-toggle').length, 1)
  assert.equal(byClass(cadence, 'tri-swim-trend-title').length, 0)
  assert.equal(
    buildSwimTrends(factory, swimToggleDetail(), {
      pace: false,
      'stroke-rate': false,
      cadence: false,
      swolf: false,
    }),
    null,
  )
})

test('keeps missing length metrics as graph gaps and renders pace alone when needed', () => {
  const current = swimTrendDetail()
  const rendered = buildSwimTrends(factory, current)
  assert.ok(rendered)
  assert.equal(byClass(rendered, 'tri-swim-trend').length, 4)
  const paceSvg = byClass(rendered, 'tri-swim-trend-svg--pace')[0]
  const cadenceSvg = byClass(rendered, 'tri-swim-trend-svg--cadence')[0]
  const swolfSvg = byClass(rendered, 'tri-swim-trend-svg--swolf')[0]
  const cadencePath = byClass(
    byClass(rendered, 'tri-swim-trend--cadence')[0],
    'tri-swim-trend-line',
  )[0]
  assert.ok(paceSvg)
  assert.ok(cadenceSvg)
  assert.ok(swolfSvg)
  assert.ok(cadencePath)
  assert.equal(String(cadencePath.properties.d).match(/[ML]/g)?.length, 6)
  assert.match(
    String(cadencePath.properties.d),
    /^M 0\.00 .* L 25\.00 .* L 25\.00 .* L 50\.00 .* M 75\.00 .* L 100\.00/,
  )
  const paceSeries = JSON.parse(
    String(paceSvg.properties.dataSwimSeriesLengths),
  ) as SwimTrendChartPoint[]
  const cadenceSeries = JSON.parse(
    String(cadenceSvg.properties.dataSwimSeriesLengths),
  ) as SwimTrendChartPoint[]
  const swolfSeries = JSON.parse(
    String(swolfSvg.properties.dataSwimSeriesLengths),
  ) as SwimTrendChartPoint[]
  assert.deepEqual(
    paceSeries.map(point => point.xPct),
    [25, 50, 75, 100],
  )
  assert.deepEqual(
    cadenceSeries.map(point => [point.cumulativeDistanceM, point.value, point.xPct]),
    [
      [25, 10, 25],
      [50, 11, 50],
      [100, 12, 100],
    ],
  )
  assert.deepEqual(
    swolfSeries.map(point => [point.cumulativeDistanceM, point.value, point.xPct]),
    [
      [25, 35, 25],
      [50, 37, 50],
      [100, 36, 100],
    ],
  )

  const paceOnly = buildSwimTrends(
    factory,
    swimTrendDetail({
      strokeRateSpm: null,
      swimIntervals: current.swimIntervals.map(interval => ({
        ...interval,
        strokeCount: null,
        strokeTimeS: null,
        strokeRateSpm: null,
      })),
    }),
  )
  assert.ok(paceOnly)
  assert.equal(byClass(paceOnly, 'tri-swim-trend--pace').length, 1)
  assert.equal(byClass(paceOnly, 'tri-swim-trend--cadence').length, 0)
  assert.equal(byClass(paceOnly, 'tri-swim-trend--swolf').length, 0)

  assert.equal(
    buildSwimTrends(factory, swimTrendDetail({ swimIntervals: current.swimIntervals.slice(0, 1) })),
    null,
  )
})

test('renders cadence and SWOLF independently when pace is unavailable', () => {
  const current = swimToggleDetail()
  const rendered = buildSwimTrends(
    factory,
    swimTrendDetail({
      swimPaceSPer100m: null,
      swimIntervals: current.swimIntervals.map(interval => ({ ...interval, paceSPer100m: null })),
    }),
  )

  assert.ok(rendered)
  assert.equal(byClass(rendered, 'tri-swim-trend--pace').length, 0)
  assert.equal(byClass(rendered, 'tri-swim-trend--rate').length, 1)
  assert.equal(byClass(rendered, 'tri-swim-trend--cadence').length, 1)
  assert.equal(byClass(rendered, 'tri-swim-trend--swolf').length, 1)
  const rateHead = byClass(byClass(rendered, 'tri-swim-trend--rate')[0], 'tri-swim-trend-head')[0]
  assert.ok(rateHead)
  assert.equal(byClass(rateHead, 'tri-swim-mode-toggle').length, 1)
  assert.equal(byClass(rateHead, 'tri-swim-trend-title').length, 0)
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-title').map(text), [
    'cadence str/length',
    'SWOLF',
  ])
})

test('includes swim trends in the default server-rendered day card', () => {
  const current = swimTrendDetail()
  const rendered = buildDayCard(factory, current.date, {
    details: { [current.id]: current },
    swimTrend: swimTrendPoints,
    health: {},
  })

  assert.equal(byClass(rendered, 'tri-swim-trends').length, 1)
  assert.equal(byClass(rendered, 'tri-act-toggle').length, 1)
  assert.equal(byClass(rendered, 'tri-act-more').length, 1)
})

test('parses ampersand-separated activity exclusions', () => {
  assert.deepEqual(parseExcludedActivityIds('filter=19471122670&19476629599&19471122670'), [
    '19471122670',
    '19476629599',
  ])
  assert.deepEqual(parseExcludedActivityIds('filter=19471122670&&19476629599'), [])
  assert.deepEqual(parseExcludedActivityIds('filter='), [])
})

test('omits excluded activities from a day card', () => {
  const activities = [
    detail({ id: 19471122670, date: '2026-07-26', name: 'Warmup legs for SuperTri' }),
    detail({ id: 19475891673, date: '2026-07-26', name: 'SuperTri 2026 Bike Leg' }),
    detail({ id: 19476629599, date: '2026-07-26', name: 'Warm down' }),
  ]
  const rendered = buildDayCard(
    factory,
    '2026-07-26',
    {
      details: Object.fromEntries(activities.map(activity => [activity.id, activity])),
      health: {},
    },
    { excludedActivityIds: ['19471122670', '19476629599'] },
  )

  assert.deepEqual(
    byClass(rendered, 'tri-act').map(activity => activity.properties.dataActivityId),
    ['19475891673'],
  )
})

test('focused display keeps triathlon sports and composes with sport and ID filters', () => {
  const date = '2026-09-08'
  const activities = [
    detail({ id: 1, date, sport: 'bike' }),
    detail({ id: 2, date, sport: 'run' }),
    detail({ id: 3, date, sport: 'swim' }),
    detail({ id: 4, date, sport: 'walk' }),
    detail({ id: 5, date, sport: 'strength' }),
    detail({ id: 6, date, sport: 'yoga' }),
    detail({ id: 7, date, sport: 'treatment' }),
    detail({ id: 8, date, sport: 'sauna' }),
  ]
  const payload = {
    details: Object.fromEntries(activities.map(activity => [activity.id, activity])),
    health: {},
  }
  const extras = { analytics: true, settings: TRIATHLON_TRACE_DISPLAY_SETTINGS.focused }
  const activityIds = (card: Element) =>
    byClass(card, 'tri-act').map(activity => activity.properties.dataActivityId)
  assert.deepEqual(activityIds(buildDayCard(factory, date, payload, extras)), ['1', '2', '3'])
  assert.deepEqual(
    activityIds(buildDayCard(factory, date, payload, { ...extras, excludedActivityIds: ['1'] })),
    ['2', '3'],
  )
  assert.deepEqual(activityIds(buildDayCard(factory, date, payload, { ...extras, sport: 'run' })), [
    '2',
  ])
  assert.deepEqual(
    activityIds(buildDayCard(factory, date, payload, { ...extras, sport: 'walk' })),
    [],
  )
  assert.equal(activityIds(buildDayCard(factory, date, payload)).length, activities.length)
})

test('renders only the selected activity and expands it', () => {
  const selected = detail({ id: 19731411847, date: '2026-08-13', name: 'Toronto-Nobleton-Toronto' })
  const other = detail({ id: 19731411848, date: '2026-08-13', sport: 'run' })
  const rendered = buildDayCard(
    factory,
    '2026-08-13',
    {
      details: { [selected.id]: selected, [other.id]: other },
      health: { '2026-08-13': { ...emptyHealth(), readiness: 80 } },
    },
    { activityId: `${selected.id}`, embedded: true },
  )

  assert.deepEqual(
    byClass(rendered, 'tri-act').map(activity => activity.properties.dataActivityId),
    ['19731411847'],
  )
  assert.equal(byClass(rendered, 'tri-act--expanded').length, 1)
  assert.equal(byClass(rendered, 'tri-act-health').length, 0)
})

test('renders exact-date analytics and appends day-card sleep after recovery', () => {
  const date = '2026-08-16'
  const ride = detail({ id: 19771722076, date, name: 'Recovery Crit' })
  const run = detail({ id: 19771722077, date, name: 'Evening run', sport: 'run' })
  const summary: TriathlonDayAnalytics = {
    date,
    sleepMetrics: resolveSleepMetrics(
      { avgBreath: 17.25 },
      {
        source: 'garmin',
        date,
        startTime: '2026-08-16T01:30:00-04:00',
        endTime: '2026-08-16T12:20:00-04:00',
        averageBreathsPerMinute: 15,
        lowestBreathsPerMinute: 11,
        highestBreathsPerMinute: 19,
        averageSpO2: 97,
        lowestSpO2: 91,
        bodyBatteryStart: 64,
        bodyBatteryEnd: 100,
        bodyBatteryChange: 36,
        averageStress: 11,
        restlessMoments: 39,
      },
    ),
    body: {
      date,
      kg: 86.06,
      bmi: 24.3,
      ffmi: 19.65,
      bodyFatPct: 19.3,
      bodyWaterPct: 58.9,
      muscleMassKg: 36.51,
      boneMassKg: 6.05,
    },
    recovery: {
      status: 'firm',
      baselineDays: 28,
      readiness: 75,
      readinessBaseline: 77,
      hrv: 54,
      hrvBaseline: 53.4,
      hrvZ: 0.1,
      rhr: 54,
      rhrBaseline: 55.2,
      rhrZ: -0.3,
      temperatureDeviationC: 0.39,
      sleepDurationS: 33_810,
      sleepBaselineS: 29_880,
      sleepTargetS: 30_600,
      sleepDebtS: 17_040,
    },
    sleep: {
      bedtimeStart: '2026-08-16T01:28:59-04:00',
      bedtimeEnd: '2026-08-16T12:25:05-04:00',
      phase5Min: '4444222111222333',
      efficiency: 86,
      latencyS: 2_100,
      timeInBedS: 39_366,
      totalSleepS: 33_810,
      deepS: 6_150,
      lightS: 21_720,
      remS: 5_940,
      awakeS: 5_556,
      averageBreathsPerMinute: 17.25,
      averageHeartRate: 62.875,
      averageHrv: 54,
      lowestHeartRate: 54,
      restlessPeriods: 201,
      hrv: { startTs: '2026-08-16T01:28:59-04:00', intervalS: 300, items: [20, 32, null, 48, 54] },
      heartRate: {
        startTs: '2026-08-16T01:28:59-04:00',
        intervalS: 300,
        items: [64, 60, 57, 54, 56],
      },
      readinessScore: 75,
      readinessContrib: { activity_balance: 34, hrv_balance: 82 },
      sleepScore: 84,
      sleepContrib: { deep_sleep: 96, latency: 46 },
    },
    training: {
      activityCount: 1,
      load: 87.3,
      relativeEffort: 12,
      ctl: 132,
      atl: 244.8,
      tsb: -112.8,
      garminTss: 52.8,
      exerciseLoad: 60.6,
      exerciseLoadSource: 'garmin',
      vo2max: { value: 55.2, method: 'garmin', confidence: 'firm', asOfDate: date },
    },
    heat: {
      date,
      temperatureC: 37.8,
      heatStrainIndex: 0.9,
      source: 'core',
      coreOrigin: 'app',
      observedMinutes: 74,
      hotMinutes: 0,
      saunaMinutes: 0,
      saunaHtl: null,
      dose: 0,
      acclimatisationPct: 100,
    },
  }
  const rendered = buildDayCard(
    factory,
    date,
    {
      details: { [ride.id]: ride, [run.id]: run },
      health: { [date]: { ...emptyHealth(), readiness: 75 } },
      dailyAnalytics: { [date]: summary },
    },
    { analytics: true, sport: 'bike', embedded: true },
  )

  assert.deepEqual(
    rendered.children
      .filter((child): child is Element => child.type === 'element')
      .map(child => classNames(child)[0]),
    ['tri-pop-head', 'tri-day-analytics', 'tri-ana-block-title', 'tri-act'],
  )
  assert.equal(byClass(rendered, 'tri-day-analytics').length, 1)
  assert.deepEqual(
    byClass(rendered, 'tri-act').map(activity => activity.properties.dataActivityId),
    ['19771722076'],
  )
  assert.equal(byClass(rendered, 'tri-day-analytics')[0].properties.ariaLabel, 'daily analytics')
  assert.equal(
    byClass(rendered, 'tri-day-analytics')[0].properties.dataI18nAriaLabel,
    'daily analytics',
  )
  assert.equal(byClass(rendered, 'tri-day-analytics-title').length, 0)
  assert.equal(byClass(rendered, 'tri-day-analytics-group').length, 4)
  assert.equal(byClass(rendered, 'tri-day-analytics-group--body-recovery').length, 1)
  assert.equal(byClass(rendered, 'tri-day-analytics-group--state-load').length, 1)
  assert.equal(byClass(rendered, 'tri-sleep-contrib').length, 2)
  assert.equal(byClass(rendered, 'tri-day-sleep-stages').length, 1)
  assert.equal(byClass(rendered, 'tri-day-sleep-series--hrv').length, 1)
  assert.equal(byClass(rendered, 'tri-day-sleep-series--heart-rate').length, 1)
  assert.equal(byClass(rendered, 'tri-day-sleep-line-svg').length, 2)
  assert.ok(
    byClass(rendered, 'tri-day-sleep-line-svg').every(
      chart =>
        chart.properties.role === 'slider' &&
        chart.properties.tabIndex === 0 &&
        String(chart.properties.ariaDescribedBy).includes('tri-day-2026-08-16-sleep'),
    ),
  )
  assert.equal(byClass(rendered, 'tri-ana-cursor').length, 3)
  assert.equal(byClass(rendered, 'tri-chart-readout').length, 3)
  const stageChart = byClass(rendered, 'tri-day-sleep-stages')[0]
  assert.equal(stageChart.properties.dataDaySleepSeries, 'stages')
  assert.equal(stageChart.properties.dataDaySleepInterval, '300')
  assert.match(String(stageChart.properties.dataDaySleepValues), /^[0-3,]+$/)
  assert.equal(byClass(rendered, 'tri-day-sleep-stage-svg')[0].properties.role, 'slider')
  assert.match(
    String(byClass(rendered, 'tri-day-sleep-series--hrv')[0].properties.dataDaySleepValues),
    /54/,
  )
  assert.ok(
    byClass(rendered, 'tri-day-analytics-detail').every(
      detail => detail.properties.role === 'tooltip',
    ),
  )
  assert.equal(byClass(rendered, 'tri-act-health').length, 0)
  const sleepMarkup = byClass(rendered, 'tri-day-analytics-group--sleep')[0]
  const sleepBar = byClass(sleepMarkup, 'tri-sleep-metrics')[0]
  const sleepReadings = byClass(sleepBar, 'tri-day-analytics-metric')
  assert.equal(sleepReadings.length, 5)
  assert.deepEqual(byClass(sleepBar, 'tri-day-analytics-label').map(text), [
    'respiration',
    'Pulse Ox',
    'Body Battery change',
    'sleep stress',
    'restless moments',
  ])
  assert.ok(sleepReadings.every(metric => metric.properties.tabIndex === 0))
  for (const metric of sleepReadings) {
    const tooltip = byClass(metric, 'tri-day-analytics-detail')[0]
    assert.equal(String(metric.properties.ariaDescribedBy), tooltip.properties.id)
  }
  assert.match(text(sleepMarkup), /respiration17\.3 brpmOura/)
  assert.doesNotMatch(text(sleepMarkup), /respiration15\.0 brpm/)
  assert.doesNotMatch(text(sleepMarkup), / · (Oura|Garmin)/)
  assert.equal(
    text(byClass(sleepReadings[1], 'tri-day-analytics-detail')[0]),
    'Garmin\nlowest Pulse Ox 91%',
  )
  assert.equal(
    text(byClass(sleepReadings[2], 'tri-day-analytics-detail')[0]),
    'Garmin\nBody Battery at bedtime 64\nBody Battery at wake-up 100',
  )
  assert.match(text(sleepMarkup), /Pulse Ox97\.0%Garmin/)
  assert.match(text(sleepMarkup), /Body Battery change\+36Garmin/)
  assert.match(text(sleepMarkup), /sleep stress11\.0Garmin/)
  assert.match(text(sleepMarkup), /restless moments39Garmin/)
  const frenchSleep = byClass(
    buildDayAnalytics(factoryFor(frenchPresentation), summary),
    'tri-day-analytics-group--sleep',
  )[0]
  assert.match(text(frenchSleep), /respiration17,3 brpmOura/)
  assert.match(text(frenchSleep), /SpO₂97,0%Garmin/)
  assert.match(text(frenchSleep), /SpO₂ minimale 91%/)
  assert.match(text(frenchSleep), /Body Battery au réveil 100/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /86\.1 kg/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /19\.65/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /55\.2 ml\/kg\/min/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /today load · TSS87\.3site/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /Garmin TSS52\.8/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /exercise load60\.6Garmin/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /relative effort12Strava/)
  assert.match(text(byClass(rendered, 'tri-day-analytics')[0]), /HSI0\.9/)
  assert.doesNotMatch(text(byClass(rendered, 'tri-day-analytics')[0]), /CORE app/)

  const rest = buildDayCard(factory, date, {
    details: {},
    health: { [date]: { ...emptyHealth(), readiness: 75 } },
    dailyAnalytics: { [date]: summary },
  })
  assert.equal(text(byClass(rest, 'tri-pop-rest-label')[0]), 'rest')
  assert.equal(byClass(rest, 'tri-day-analytics').length, 1)
  assert.equal(byClass(rest, 'tri-day-sleep-analytics').length, 1)
  assert.equal(byClass(rest, 'tri-day-analytics-group').length, 1)
  assert.equal(byClass(rest, 'tri-day-analytics-group--sleep').length, 1)
  assert.equal(byClass(rest, 'tri-day-analytics-group--body-recovery').length, 0)
  assert.equal(byClass(rest, 'tri-day-analytics-group--state-load').length, 0)
  assert.equal(byClass(rest, 'tri-day-analytics-group--thermal').length, 0)
  assert.equal(byClass(rest, 'tri-day-sleep-stages').length, 1)
  assert.equal(byClass(rest, 'tri-day-sleep-series--hrv').length, 1)
  assert.equal(byClass(rest, 'tri-day-sleep-series--heart-rate').length, 1)
  assert.deepEqual(rest.children.slice(-2), [
    byClass(rest, 'tri-act-health')[0],
    byClass(rest, 'tri-day-sleep-analytics')[0],
  ])

  const restAnalytics = buildDayCard(
    factory,
    date,
    { details: {}, health: {}, dailyAnalytics: { [date]: summary } },
    { embedded: true, analytics: true },
  )
  assert.equal(byClass(restAnalytics, 'tri-act').length, 0)
  assert.equal(byClass(restAnalytics, 'tri-day-activities-title').length, 0)
  assert.equal(text(byClass(restAnalytics, 'tri-pop-rest-label')[0]), 'rest')
  assert.equal(byClass(restAnalytics, 'tri-day-analytics').length, 1)
  assert.equal(byClass(restAnalytics, 'tri-day-analytics')[0].properties.dataAnalyticsDate, date)
  assert.deepEqual(byClass(restAnalytics, 'tri-day-analytics-group-title').map(text), [
    'body · recovery',
    'sleep details',
    'state · load',
    'thermal',
  ])
  assert.equal(byClass(restAnalytics, 'tri-day-sleep-stages').length, 1)
  assert.equal(byClass(restAnalytics, 'tri-day-sleep-series--hrv').length, 1)
  assert.equal(byClass(restAnalytics, 'tri-day-sleep-series--heart-rate').length, 1)

  const payload = {
    details: { [ride.id]: ride, [run.id]: run },
    health: { [date]: { ...emptyHealth(), readiness: 75 } },
    dailyAnalytics: { [date]: summary },
  }
  for (const extras of [{}, { expanded: true }, { embedded: true }]) {
    const active = buildDayCard(factory, date, payload, extras)
    const sleepSection = byClass(active, 'tri-day-sleep-analytics')[0]
    assert.equal(byClass(active, 'tri-act').length, 2)
    assert.equal(byClass(active, 'tri-day-analytics').length, 1)
    assert.equal(byClass(active, 'tri-day-analytics-group').length, 1)
    assert.equal(sleepSection.properties.dataAnalyticsDate, date)
    assert.deepEqual(active.children.slice(-2), [
      byClass(active, 'tri-act-health')[0],
      sleepSection,
    ])
    assert.deepEqual(byClass(sleepSection, 'tri-day-analytics-group--sleep'), [sleepMarkup])
  }

  for (const extras of [{ sport: ride.sport }, { activityId: `${ride.id}` }]) {
    const selected = buildDayCard(factory, date, payload, extras)
    assert.equal(byClass(selected, 'tri-act-health').length, 0)
    assert.equal(byClass(selected, 'tri-day-sleep-analytics').length, 0)
  }
  const analyticsCard = buildDayCard(factory, date, payload, { analytics: true })
  assert.equal(byClass(analyticsCard, 'tri-day-analytics-group--sleep').length, 1)
  assert.equal(byClass(analyticsCard, 'tri-day-sleep-analytics').length, 0)
  assert.equal(byClass(buildDayCard(factory, '2026-08-17', payload), 'tri-day-analytics').length, 0)
  const noSleep = buildDayCard(factory, date, {
    ...payload,
    dailyAnalytics: { [date]: { ...summary, sleep: null, sleepMetrics: null, recovery: null } },
  })
  assert.equal(byClass(noSleep, 'tri-day-analytics').length, 0)

  assert.ok(summary.sleepMetrics?.garmin)
  const garminOnly: TriathlonDayAnalytics = {
    ...summary,
    recovery: null,
    sleep: null,
    sleepMetrics: resolveSleepMetrics(null, {
      ...summary.sleepMetrics.garmin,
      bodyBatteryChange: 0,
      averageStress: 0,
      restlessMoments: 0,
    }),
  }
  const garminRest = buildDayCard(factory, date, {
    details: {},
    health: {},
    dailyAnalytics: { [date]: garminOnly },
  })
  assert.equal(byClass(garminRest, 'tri-day-analytics-group--sleep').length, 1)
  assert.match(text(garminRest), /respiration15\.0 brpmGarmin/)
  assert.match(text(garminRest), /Body Battery change0Garmin/)
  assert.match(text(garminRest), /sleep stress0\.0Garmin/)
  assert.match(text(garminRest), /restless moments0Garmin/)
  assert.equal(byClass(garminRest, 'tri-day-sleep-stages').length, 0)
  assert.equal(byClass(garminRest, 'tri-day-sleep-line-svg').length, 0)
  assert.doesNotMatch(text(garminRest), /total sleep|sleep score|time in bed/)
  const garminActive = buildDayCard(factory, date, {
    ...payload,
    dailyAnalytics: { [date]: garminOnly },
  })
  assert.deepEqual(
    byClass(garminActive, 'tri-day-analytics-group--sleep'),
    byClass(garminRest, 'tri-day-analytics-group--sleep'),
  )

  assert.ok(summary.heat)
  const saunaSummary: TriathlonDayAnalytics = {
    ...summary,
    heat: { ...summary.heat, source: 'mixed', saunaMinutes: 65, saunaHtl: 7.7 },
  }
  const thermal = byClass(
    buildDayAnalytics(factory, saunaSummary),
    'tri-day-analytics-group--thermal',
  )[0]
  assert.match(text(thermal), /CORE temperature/)
  assert.doesNotMatch(text(thermal), /ambient temperature/)
  assert.match(text(thermal), /sauna min65/)
  assert.match(text(thermal), /recorded sauna HTL7\.7/)
})

test('renders native sleep respiration with real time spacing, gaps, and Garmin provenance', () => {
  const date = '2026-09-08'
  const start = Date.parse(`${date}T05:30:56Z`)
  const offsets = [0, 64, 184, 304, 424, 1200, 1320]
  const values = [19, 15, null, 16, 17, 18, 14]
  const garmin: GarminSleepSummary = {
    source: 'garmin',
    date,
    startTime: new Date(start).toISOString(),
    endTime: new Date(start + 1320_000).toISOString(),
    utcOffsetMinutes: -240,
    averageBreathsPerMinute: 15,
    lowestBreathsPerMinute: 14,
    highestBreathsPerMinute: 19,
    averageSpO2: null,
    lowestSpO2: null,
    bodyBatteryStart: null,
    bodyBatteryEnd: null,
    bodyBatteryChange: null,
    averageStress: null,
    restlessMoments: null,
    respiration: offsets.map((seconds, index) => ({
      timestamp: start + seconds * 1000,
      breathsPerMinute: values[index],
    })),
  }
  const sleepMetrics = resolveSleepMetrics({ avgBreath: 16.1 }, garmin)
  const chart = buildSleepRespirationChart(factory, date, sleepMetrics)
  assert.ok(chart)
  assert.equal(chart.properties.dataDaySleepTimes, offsets.join(','))
  assert.equal(chart.properties.dataDaySleepValues, '19,15,,16,17,18,14')
  assert.equal(chart.properties.dataDaySleepUnit, 'brpm')
  assert.equal(chart.properties.dataDaySleepSource, 'Garmin')
  assert.equal(chart.properties.dataDaySleepStart, `${date}T01:30:56.000-04:00`)
  assert.equal(Date.parse(String(chart.properties.dataDaySleepStart)), start)
  const paths = byClass(chart, 'tri-day-sleep-line--respiration')
  assert.equal(paths.length, 3)
  assert.match(String(paths[0].properties.d), /^M0\.000 [\d.]+L4\.848 /)
  const svg = byClass(chart, 'tri-day-sleep-line-svg')[0]
  assert.equal(svg.properties.role, 'slider')
  assert.equal(svg.properties.tabIndex, 0)
  assert.equal(svg.properties.ariaValueText, '01:52 · 14 brpm')
  const onlyGarmin = buildDayAnalytics(factory, {
    date,
    body: null,
    recovery: null,
    sleep: null,
    sleepMetrics: resolveSleepMetrics(null, garmin),
    training: null,
    heat: null,
  })
  assert.equal(byClass(onlyGarmin, 'tri-day-sleep-series--respiration').length, 1)
  assert.equal(byClass(onlyGarmin, 'tri-day-sleep-stages').length, 0)
  const samplesOnly = buildDayAnalytics(factory, {
    date,
    body: null,
    recovery: null,
    sleep: null,
    sleepMetrics: resolveSleepMetrics(null, {
      ...garmin,
      averageBreathsPerMinute: null,
      lowestBreathsPerMinute: null,
      highestBreathsPerMinute: null,
    }),
    training: null,
    heat: null,
  })
  assert.equal(byClass(samplesOnly, 'tri-sleep-metrics').length, 0)
  assert.equal(byClass(samplesOnly, 'tri-day-sleep-series--respiration').length, 1)
  assert.equal(
    buildSleepRespirationChart(factory, date, resolveSleepMetrics({ avgBreath: 16.1 }, null)),
    null,
  )
  assert.equal(
    buildSleepRespirationChart(
      factory,
      date,
      resolveSleepMetrics(null, { ...garmin, respiration: [] }),
    ),
    null,
  )
  const isolated = buildSleepRespirationChart(
    factory,
    date,
    resolveSleepMetrics(null, {
      ...garmin,
      respiration: [
        { timestamp: start, breathsPerMinute: 15 },
        { timestamp: start + 600_000, breathsPerMinute: 17 },
      ],
    }),
  )
  assert.ok(isolated)
  assert.equal(byClass(isolated, 'tri-day-sleep-line--respiration').length, 2)
  assert.ok(
    byClass(isolated, 'tri-day-sleep-line--respiration').every(path =>
      String(path.properties.d).endsWith('l0 0'),
    ),
  )
})

test('day-card date renders as a month link only when extras provide an href', () => {
  const current = detail({ id: 7, date: '2026-07-09' })
  const payload = { details: { 7: current }, health: {} }
  const linked = buildDayCard(factory, '2026-07-09', payload, {
    dateHref: '../../../triathlon/on/2026/07',
  })
  const anchor = byClass(linked, 'tri-pop-date')[0]
  assert.equal(anchor.tagName, 'a')
  assert.equal(anchor.properties.href, '../../../triathlon/on/2026/07')
  const plain = buildDayCard(factory, '2026-07-09', payload)
  assert.equal(byClass(plain, 'tri-pop-date')[0].tagName, 'span')
})

test('timeline day cards keep activity measurements and the date inert', () => {
  const ride = detail({ id: 1, date: '2026-07-09', name: 'Lunch ride', distanceKm: 30 })
  const strength = detail({
    id: 2,
    date: '2026-07-09',
    sport: 'strength',
    name: 'Upper body',
    distanceKm: 0,
    movingTimeS: 2_700,
  })
  const card = buildTimelineDayCard(factory, '2026-07-09', {
    details: { 1: ride, 2: strength },
    health: {},
  })

  const entries = byClass(card, 'tri-timeline-activity')
  assert.equal(byClass(card, 'tri-act').length, 0)
  assert.equal(byClass(card, 'tri-timeline-name').length, 0)
  assert.equal(byClass(card, 'tri-pop-loc').length, 0)
  assert.deepEqual(
    entries.map(entry => entry.tagName),
    ['span', 'span'],
  )
  assert.ok(entries.every(entry => entry.properties.href === undefined))
  assert.ok(entries.every(entry => entry.properties.role === 'group'))
  assert.equal(byClass(card, 'tri-timeline-row').length, 2)
  assert.equal(byTag(card, 'a').length, 0)
  assert.deepEqual(byClass(card, 'tri-timeline-value').map(text), ['30.0 km', "45'"])
  assert.equal(byClass(card, 'tri-pop-date')[0].tagName, 'span')
  assert.equal(byClass(card, 'tri-pop-date')[0].properties.href, undefined)

  const rest = buildTimelineDayCard(factory, '2026-07-10', { details: {}, health: {} })
  assert.equal(byClass(rest, 'tri-pop-date').length, 1)
  assert.equal(byClass(rest, 'tri-timeline-activity').length, 0)
  assert.equal(byClass(rest, 'tri-timeline-row').length, 1)
  assert.equal(byClass(rest, 'tri-timeline-rest').length, 1)
  assert.equal(byClass(rest, 'tri-pop-rest').length, 0)
  assert.equal(byClass(rest, 'tri-battery').length, 1)
  assert.equal(text(byClass(rest, 'tri-timeline-value')[0]), 'rest')

  const loading = buildTimelineDayCard(factory, '2026-07-10', null)
  assert.equal(byClass(loading, 'tri-battery').length, 0)
  assert.equal(text(byClass(loading, 'tri-pop-rest')[0]), '·')
})

test('embedded day cards align activity summaries to their largest row count', () => {
  const ride = detail({
    id: 1,
    date: '2026-07-09',
    windKph: 10,
    windDir: 'NW',
    windGustKph: 21,
    fueling: {
      caloriesConsumed: 200,
      carbsConsumedG: null,
      fluidMl: null,
      carbsRecommendedG: null,
      fluidRecommendedMl: null,
      sweatLossMl: null,
      sodiumLossMg: null,
      sourceDevice: 'Edge 1050',
      source: 'garmin',
    },
  })
  const run = detail({
    id: 2,
    date: '2026-07-09',
    sport: 'run',
    avgWatts: null,
    npWatts: null,
    maxWatts: null,
    kilojoules: null,
    deviceWatts: false,
  })
  const payload = { details: { 1: ride, 2: run }, health: {} }
  const embedded = buildDayCard(factory, '2026-07-09', payload, { embedded: true })
  const rowCounts = byClass(embedded, 'tri-act').map(
    activity => byTag(byClass(activity, 'tri-act-stats')[0], 'tr').length,
  )

  assert.equal(
    embedded.properties.style,
    `--tri-embedded-summary-rows:${Math.max(...rowCounts)};--tri-embedded-fueling-rows:2`,
  )
  const wind = byTag(embedded, 'tr').find(row => text(byClass(row, 'tri-act-stat-k')[0]) === 'wind')
  assert.equal(wind?.properties.dataStatKey, 'wind')
  const emptyFueling = byClass(embedded, 'tri-act-fueling--empty')[0]
  assert.ok(emptyFueling)
  assert.equal(emptyFueling.properties.ariaHidden, 'true')
  assert.equal(byTag(byClass(emptyFueling, 'tri-act-stats')[0], 'tr').length, 0)

  const hydratedReservations: boolean[] = []
  buildDayCard(factory, '2026-07-09', payload, { embedded: true }, (activity, reserveFueling) => {
    hydratedReservations.push(reserveFueling)
    return buildActivity(
      factory,
      activity,
      false,
      undefined,
      false,
      true,
      undefined,
      reserveFueling,
    )
  })
  assert.deepEqual(hydratedReservations, [true, true])

  const stacked = buildDayCard(factory, '2026-07-09', payload)
  const single = buildDayCard(
    factory,
    '2026-07-09',
    { details: { 1: ride }, health: {} },
    { embedded: true },
  )
  assert.equal(stacked.properties.style, undefined)
  assert.equal(single.properties.style, undefined)
})

test('embedded graphless activities reserve the shared visual slot before their disclosure', () => {
  const graphless = detail({
    id: 2,
    date: '2026-07-09',
    sport: 'yoga',
    route: [],
    heartRateTrace: [],
    analysisRanges: [],
    bestEfforts: null,
  })
  const embedded = buildDayCard(
    factory,
    '2026-07-09',
    { details: { 1: detail({ id: 1, date: '2026-07-09' }), 2: graphless }, health: {} },
    { embedded: true, expanded: true },
  )
  const graphlessActivity = byClass(embedded, 'tri-act').find(
    activity => activity.properties.dataActivityId === '2',
  )
  assert.ok(graphlessActivity)
  const emptyVisual = byClass(graphlessActivity, 'tri-act-figs--empty')[0]
  assert.ok(emptyVisual)
  assert.equal(emptyVisual.properties.ariaHidden, 'true')
  assert.equal(text(byClass(graphlessActivity, 'tri-act-toggle')[0]), '− see less')
  assert.equal(byClass(graphlessActivity, 'tri-act-more').length, 1)

  const standalone = buildActivity(factory, graphless, true)
  assert.equal(byClass(standalone, 'tri-act-figs--empty').length, 0)
  assert.equal(text(byClass(standalone, 'tri-act-toggle')[0]), '− see less')
})

test('expanded day-card extras render every activity pre-expanded', () => {
  const first = detail({ id: 1, date: '2026-07-09' })
  const second = detail({ id: 2, date: '2026-07-09', sport: 'run' })
  const rendered = buildDayCard(
    factory,
    '2026-07-09',
    { details: { 1: first, 2: second }, health: {} },
    { expanded: true },
  )
  assert.equal(byClass(rendered, 'tri-act--expanded').length, 2)
  const toggles = byClass(rendered, 'tri-act-toggle')
  assert.ok(toggles.length >= 1)
  for (const toggle of toggles) {
    assert.equal(text(toggle), '− see less')
    assert.equal(toggle.properties.ariaExpanded, 'true')
  }
})

test('race cards preserve missing run power rows for transitions', () => {
  const transitions = [
    detail({
      id: 1,
      name: 'SuperTri T1',
      sport: 'run',
      avgWatts: null,
      npWatts: null,
      maxWatts: null,
      kilojoules: null,
      deviceWatts: false,
    }),
    detail({
      id: 2,
      name: 'SuperTri T2',
      sport: 'run',
      avgWatts: null,
      npWatts: null,
      maxWatts: null,
      kilojoules: null,
      deviceWatts: false,
    }),
  ]
  const rendered = buildDayCard(
    factory,
    '2026-07-09',
    {
      details: Object.fromEntries(transitions.map(transition => [transition.id, transition])),
      health: {},
    },
    { event: 'SuperTri' },
  )

  for (const activity of byClass(rendered, 'tri-act')) {
    const stats = byClass(activity, 'tri-act-stats')[0]
    assert.ok(stats)
    assert.deepEqual(
      bodyRows(stats).filter(([label]) =>
        ['NP', 'avg power', 'max power', 'energy'].includes(label),
      ),
      [
        ['NP', '—'],
        ['avg power', '—'],
        ['max power', '—'],
        ['energy', '—'],
      ],
    )
  }

  const ordinaryRun = buildDayCard(factory, '2026-07-09', {
    details: { 1: transitions[0] },
    health: {},
  })
  const ordinaryStats = byClass(ordinaryRun, 'tri-act-stats')[0]
  assert.ok(ordinaryStats)
  assert.equal(
    bodyRows(ordinaryStats).some(([label]) => label === 'NP'),
    false,
  )
})

test('renders imperial effort values and elevation axes with feet grid increments', () => {
  assert.equal(formatAltitude(imperialPresentation, -0.1), '0 ft')
  const ride = detail()
  const imperialFactory = factoryFor(imperialPresentation)
  const efforts = buildBestEfforts(imperialFactory, ride)
  assert.ok(efforts)
  assert.deepEqual(bodyRows(table(efforts, 'distance')), [
    ['10K', '24:31', '15.2 mph', '151 bpm', '-98 ft'],
  ])
  assert.deepEqual(bodyRows(table(efforts, 'climbing')), [
    [
      'Snake Road',
      '8:00',
      '1.55 mi',
      '394 ft',
      '4.8%',
      '11.7 mph',
      '155 bpm',
      '240 W',
      '2.74 W/kg',
      '2,953 ft/h',
    ],
  ])

  const elevation = buildElevation(imperialFactory, ride)
  assert.equal(byClass(elevation, 'tri-cax-frame').length, 1)
  assert.deepEqual(byClass(elevation, 'tri-cax-yt').map(text).filter(Boolean), [
    '260 ft',
    '280 ft',
    '300 ft',
    '320 ft',
    '340 ft',
    '360 ft',
  ])
  assert.deepEqual(byClass(elevation, 'tri-cax-xt').map(text), ['5 mi', '10 mi', '15 mi'])
  assert.equal(byClass(elevation, 'tri-elev-grid').length, 6)
  assert.deepEqual(
    byClass(elevation, 'tri-elev-cap')
      .flatMap(cap => byTag(cap, 'span'))
      .map(text),
    ['+328 ft', '−66 ft', '246 ft–361 ft'],
  )
})

const zonedDetail = (): StravaActivityDetail =>
  detail({
    hrZones: [600, 1_200, 900, 300, 60],
    powerZones: [400, 900, 1_100, 700, 300, 120, 40],
    powerHist: [30, 300, 600, 420, 60],
    powerCurve: [
      { s: 1, w: 565 },
      { s: 5, w: 540 },
      { s: 60, w: 320 },
      { s: 300, w: 250 },
      { s: 1_200, w: 230 },
      { s: 3_600, w: 210 },
    ],
  })

const shiftedDetail = (): StravaActivityDetail =>
  detail({
    gearShifts: [
      {
        elapsedS: 0,
        distanceKm: 0,
        frontGearNum: 2,
        frontTeeth: 52,
        rearGearNum: 3,
        rearTeeth: 27,
      },
      {
        elapsedS: 1_600,
        distanceKm: 10,
        frontGearNum: 2,
        frontTeeth: 52,
        rearGearNum: 6,
        rearTeeth: 19,
      },
      {
        elapsedS: 3_200,
        distanceKm: 20,
        frontGearNum: 1,
        frontTeeth: 36,
        rearGearNum: 6,
        rearTeeth: 19,
      },
      {
        elapsedS: 4_800,
        distanceKm: 30,
        frontGearNum: 1,
        frontTeeth: 36,
        rearGearNum: 11,
        rearTeeth: 11,
      },
    ],
  })

const cyclingDynamicsDetail = (): StravaActivityDetail =>
  detail({
    route: detail().route.map((point, index) => ({
      ...point,
      rightPowerPct: [48, 49, 52, 51][index],
    })),
    cyclingDynamics: {
      elapsedS: [0, 10, 20, 30],
      distanceKm: [0, 10, 20, 30],
      leftPedalSmoothness: [21, null, 24, 25],
      rightPedalSmoothness: [23, null, 26, 27],
      leftTorqueEffectiveness: [70, null, 76, 78],
      rightTorqueEffectiveness: [72, null, 78, 80],
      leftPowerPhaseStart: [350, 355, 2, 4],
      leftPowerPhaseEnd: [190, 192, 194, 196],
      rightPowerPhaseStart: [348, 352, 354, 356],
      rightPowerPhaseEnd: [198, 200, 202, 204],
      positionChanges: [
        { elapsedS: 0, distanceKm: 0, position: 'seated' },
        { elapsedS: 10, distanceKm: 10, position: 'standing' },
        { elapsedS: 20, distanceKm: 20, position: 'seated' },
      ],
      seatedTimeS: 3_600,
      standingTimeS: 1_200,
    },
  })

test('emits kebab-case trace names across bike, run, and swim charts', () => {
  const bike = cyclingDynamicsDetail()
  bike.gearShifts = shiftedDetail().gearShifts
  bike.route = bike.route.map((point, index) => ({
    ...point,
    stamina: [100, 76, 54, 32][index],
    potentialStamina: [100, 88, 67, 40][index],
    heatStrainIndex: [0, 1.4, 3, 3.1][index],
    coreTemperatureC: [37.16, 37.17, 37.19, 37.18][index],
    skinTemperatureC: [33.4, 33.45, 33.5, 33.55][index],
  }))
  const run = detail({
    sport: 'run',
    deviceWatts: false,
    route: detail().route.map((point, index) => ({
      ...point,
      speedKph: 10 + index,
      cad: 80,
      strideLengthM: index === 1 ? null : 1.1 + index * 0.05,
      groundContactTimeMs: index === 1 ? null : 245 - index * 3,
      verticalOscillationCm: index === 1 ? null : 9.8 - index * 0.1,
    })),
  })
  const estimatedStride = buildRunStrideTrace(
    factory,
    detail({
      sport: 'run',
      deviceWatts: false,
      route: detail().route.map((point, index) => ({
        ...point,
        cad: 80 + index * 5,
        speedKph: 10 + index,
      })),
    }),
    null,
  )
  const swim = buildSwimTrends(factory, swimToggleDetail())
  assert.ok(estimatedStride)
  assert.ok(swim)

  const traceNames = [
    buildActivity(factory, bike, true),
    buildActivity(factory, run, true),
    estimatedStride,
    swim,
  ]
    .flatMap(root =>
      descendants(root, element => typeof element.properties.dataTriTrace === 'string'),
    )
    .map(element => String(element.properties.dataTriTrace))

  assert.deepEqual([...new Set(traceNames)].sort(), [
    'cadence',
    'core-temperature',
    'electronic-shifting',
    'estimated-stride-length',
    'ground-contact-time',
    'heat-strain-index',
    'hr',
    'pace',
    'pedal-smoothness',
    'power',
    'power-balance',
    'power-phase',
    'respiration',
    'rider-position',
    'skin-temperature',
    'speed',
    'stamina',
    'stride-length',
    'stroke-rate',
    'swolf',
    'temperature',
    'torque-effectiveness',
    'vertical-oscillation',
  ])
  assert.ok(traceNames.every(name => /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name)))
})

const ctx = (overrides: Partial<DetailCtx> = {}): DetailCtx => ({
  zones: { hr: [120, 140, 160, 180], power: [150, 200, 250, 300, 350, 400], ftp: 260 },
  curveRef: [],
  curveYearRef: [],
  runCurveRef: [],
  runCurveYearRef: [],
  curveYear: null,
  criticalPower: null,
  criticalPowerYear: null,
  ftp: 260,
  goalFtp: 280,
  vt1: 150,
  ...overrides,
})

const criticalPower = (
  window: CriticalPowerEstimate['window'] = 'six-weeks',
): CriticalPowerEstimate => ({
  criticalPowerWatts: 249,
  wPrimeJoules: 10_300,
  method: 'two-parameter-power-space',
  window,
  windowFrom:
    window === 'activity' ? '2026-07-09' : window === 'six-weeks' ? '2026-07-03' : '2026-01-01',
  windowTo: window === 'activity' ? '2026-07-09' : '2026-08-13',
  anchors: [
    {
      durationS: 180,
      meanPowerWatts: 306.5,
      activityId: 102,
      activityDate: '2026-08-09',
      startElapsedS: 6_618,
      endElapsedS: 6_798,
    },
    {
      durationS: 420,
      meanPowerWatts: 272,
      activityId: 103,
      activityDate: '2026-08-05',
      startElapsedS: 1_018,
      endElapsedS: 1_438,
    },
    {
      durationS: 720,
      meanPowerWatts: 264.3,
      activityId: 103,
      activityDate: '2026-08-05',
      startElapsedS: 1_018,
      endElapsedS: 1_738,
    },
  ],
  independentEffortCount: 2,
  rmseWatts: 1.4,
  normalizedRmse: 0.005,
  confidence: 'provisional',
})

test('renders traces with numbered value and distance axes', () => {
  const trace = buildTrace(
    factory,
    detail(),
    p => p.hr,
    'hr',
    max => `${max} bpm peak`,
    value => `${Math.round(value)}bpm`,
  )
  assert.equal(byClass(trace, 'tri-cax-frame').length, 1)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['0', '50bpm', '100bpm', '150bpm'])
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['10 km', '20 km'])
  assert.equal(byClass(trace, 'tri-elev-grid').length, 4)
  assert.equal(byClass(trace, 'tri-cax-ax').length, 2)
  assert.deepEqual(
    byClass(trace, 'tri-elev-cap')
      .flatMap(cap => byTag(cap, 'span'))
      .map(text),
    ['hr', '153 bpm peak'],
  )
})

test('starts heart rate traces at 80 bpm', () => {
  const trace = buildHeartRateTrace(factory, detail())

  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['80bpm', '100bpm', '120bpm', '140bpm'])
  assert.doesNotMatch(
    String(byClass(trace, 'tri-elev-line')[0]?.properties.d),
    / 30(?:\.00)?(?: |$)/,
  )
})

test('renders sauna laps by elapsed time with duration and HR change', () => {
  const lap: ActivityAnalysisRange = {
    kind: 'lap',
    id: 'lap:sauna-1',
    label: 'Lap 1',
    startElapsedS: 300,
    endElapsedS: 344,
    startDistanceKm: 0,
    endDistanceKm: 0,
    distanceKm: 0,
    durationS: 44,
    movingTimeS: 41,
    elevationGainM: null,
    averageSpeedKph: null,
    averageHeartRate: 108.5,
    averageWatts: null,
    averageCadence: null,
    heartRateChange: { source: 'strava', startBpm: 137, endBpm: 99 },
  }
  const sauna = detail({
    sport: 'sauna',
    route: [],
    mapRoute: [],
    distanceKm: 0,
    elapsedTimeS: 1000,
    analysisRanges: [lap, { ...lap }, { ...lap, id: 'invalid', endElapsedS: 300 }],
    heartRateTrace: [
      heartRateTracePoint(0, 0, 80),
      heartRateTracePoint(0, 300, 137),
      heartRateTracePoint(0, 344, 99),
      heartRateTracePoint(0, 1000, 70),
    ],
  })
  const rendered = buildActivity(factory, sauna, true, ctx())
  const bands = byClass(rendered, 'tri-analysis-band')
  assert.equal(bands.length, 1)
  assert.equal(bands[0].properties.dataAnalysisKind, 'lap')
  const buttons = byClass(rendered, 'tri-analysis-range')
  assert.equal(buttons.length, 1)
  assert.equal(byClass(bands[0], 'tri-analysis-band-items')[0].properties.dataSiteCursorLine, '')
  assert.equal(buttons[0].properties.ariaLabel, 'Lap 1, 0:44, 109 bpm avg, 137 → 99 bpm (-38)')
  assert.equal(buttons[0].properties.dataDurationS, '44')
  assert.equal(buttons[0].properties.dataHeartRateChangeSource, 'strava')
  assert.match(
    String(buttons[0].properties.style),
    /--tri-analysis-start:30\.000%;--tri-analysis-width:4\.400%/,
  )
  const trace = buildHeartRateTrace(factory, sauna, lap)
  assert.equal(byClass(trace, 'tri-analysis-selection')[0].properties.x, '30.00')
  assert.equal(byClass(trace, 'tri-analysis-selection')[0].properties.width, '4.40')
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['0s', '8:20', '16:40'])
  const withoutLaps = buildActivity(factory, { ...sauna, analysisRanges: [] }, true, ctx())
  assert.equal(byClass(withoutLaps, 'tri-analysis-range').length, 0)

  const withExercises: StravaActivityDetail = {
    ...sauna,
    strength: {
      volumeKg: null,
      totalSets: null,
      totalReps: null,
      source: 'manual',
      exercises: [
        {
          name: 'Glute Bridge',
          setCount: 2,
          durationS: 60,
          repetitions: null,
          sets: [
            { durationS: 30, repetitions: null, weightKg: null },
            { durationS: 30, repetitions: null, weightKg: null },
          ],
        },
      ],
    },
  }
  for (const embedded of [false, true]) {
    for (const expanded of [false, true]) {
      const card = buildActivity(factory, withExercises, expanded, ctx(), false, embedded)
      const figures = byClass(card, 'tri-act-figs')
      assert.equal(figures.length, 1)
      const more = byClass(card, 'tri-act-more')[0]
      const exercises = byClass(more, 'tri-act-strength')[0]
      const analysis = byClass(figures[0], 'tri-analysis')[0]
      assert.ok(exercises)
      assert.ok(analysis)
      assert.equal(byClass(figures[0], 'tri-act-strength').length, 0)
      assert.ok(more.children.includes(exercises))
      const toggle = byClass(card, 'tri-act-toggle')[0]
      assert.equal(card.children.indexOf(toggle), card.children.indexOf(figures[0]) + 1)
      assert.equal(toggle.properties.ariaExpanded, String(expanded))
      assert.equal(byClass(analysis, 'tri-analysis-range').length, 1)
    }
  }
})

test('renders sauna phases in chronological lap order with recorded timing and HR', () => {
  const durations = [600, 1_500, 480, 1_320, 540, 60]
  let elapsedS = 0
  const laps: ActivityAnalysisRange[] = durations.map((durationS, index) => {
    const startElapsedS = elapsedS
    elapsedS += durationS
    return {
      kind: 'lap',
      id: `lap:sauna-${index}`,
      label: `Lap ${index}`,
      startElapsedS,
      endElapsedS: elapsedS,
      startDistanceKm: 0,
      endDistanceKm: 0,
      distanceKm: 0,
      durationS,
      movingTimeS: durationS - 1,
      elevationGainM: null,
      averageSpeedKph: null,
      averageHeartRate: 101,
      averageWatts: null,
      averageCadence: null,
      heartRateChange: { source: 'strava', startBpm: 130, endBpm: 100 },
    }
  })
  const sauna = detail({
    sport: 'sauna',
    route: [],
    mapRoute: [],
    distanceKm: 0,
    elapsedTimeS: elapsedS,
    analysisRanges: [...laps.toReversed(), laps[0]],
    heartRateTrace: [heartRateTracePoint(0, 0, 90), heartRateTracePoint(0, elapsedS, 100)],
  })
  for (const embedded of [false, true]) {
    const workout = buildWorkoutAnalysis(factory, sauna, embedded)
    assert.ok(workout)
    assert.equal(workout.properties.dataWorkoutAnalysisMetric, 'hr')
    const phases = byClass(workout, 'tri-sauna-lap')
    assert.deepEqual(phases.map(text), ['1', '2', '3', '4', '5', '6'])
    assert.deepEqual(
      phases.map(phase => phase.properties.dataSaunaPhase),
      ['hot sauna', 'hot sauna', 'cold plunge', 'hot sauna', 'break', 'break'],
    )
    assert.deepEqual(
      phases.map(phase => phase.properties.dataRangeId),
      laps.map(lap => lap.id),
    )
    assert.equal(phases[2].properties.dataRangeLabel, 'lap 3 · cold plunge')
    assert.equal(
      phases[2].properties.ariaLabel,
      'lap 3 · cold plunge, 8:00, 101 bpm avg, 130 → 100 bpm (-30)',
    )
    assert.equal(phases[2].properties.dataDurationS, '480')
    assert.equal(phases[2].properties.dataStartElapsedS, '2100')
    assert.equal(phases[2].properties.dataEndElapsedS, '2580')
    assert.equal(phases[2].properties.dataHeartRateChangeSource, 'strava')
    assert.equal(byClass(workout, 'tri-sauna-laps')[0].properties.dataSaunaPhaseSource, 'lap-order')
    assert.equal(byClass(workout, 'tri-sauna-laps-timeline')[0].properties.dataSiteCursorLine, '')
    assert.doesNotMatch(text(workout), /phases inferred from lap order/)
    assert.equal(byClass(workout, 'tri-analysis-selection').length, 1)
    const trace = byClass(workout, 'tri-elev')[0]
    assert.equal(trace.properties.dataDomainEndElapsedS, elapsedS)
  }

  const noHeartRate = buildWorkoutAnalysis(factory, { ...sauna, heartRateTrace: [] })
  assert.ok(noHeartRate)
  assert.equal(noHeartRate.properties.dataWorkoutAnalysisMetric, 'elapsed')
  assert.equal(byClass(noHeartRate, 'tri-sauna-lap').length, 6)
  assert.equal(byClass(noHeartRate, 'tri-elev').length, 0)
  assert.equal(
    buildWorkoutAnalysis(factory, { ...sauna, analysisRanges: [], heartRateTrace: [] }),
    null,
  )
  const unsplit = buildWorkoutAnalysis(factory, { ...sauna, analysisRanges: [laps[0]] })
  assert.ok(unsplit)
  assert.equal(byClass(unsplit, 'tri-sauna-lap').length, 0)
  const naturalCooldown = buildWorkoutAnalysis(factory, {
    ...sauna,
    sauna: {
      time: '19:30',
      temperatureC: 80,
      humidityPct: 10,
      cooldown: 'natural',
      heatTrainingLoad: null,
      heartRateSource: null,
      source: 'manual',
    },
  })
  assert.ok(naturalCooldown)
  assert.equal(byClass(naturalCooldown, 'tri-sauna-lap')[2].properties.dataSaunaPhase, 'break')
})

test('renders authored sauna phases on chronological laps in cards and timelines', () => {
  const parsed = parseTrackingBlock(
    null,
    [
      'activity: sauna',
      'date: 2026-09-17',
      'time: 19:30',
      'duration: 75 mins',
      'temperature: 168F',
      'humidity: 11%',
      'cooldown: cold plunge',
      'location: Othership Adelaide',
      'lap-phases: hot sauna | cold plunge | hot sauna | break',
    ].join('\n'),
  )
  assert.ok(parsed?.sauna)
  const durations = [2_482, 356, 1_634, 265]
  const lap = analysisRanges().find(range => range.kind === 'lap')
  assert.ok(lap)
  let elapsedS = 0
  const laps = durations.map((durationS, index) => {
    const startElapsedS = elapsedS
    elapsedS += durationS
    return {
      ...lap,
      id: `sauna:${index}`,
      startElapsedS,
      endElapsedS: elapsedS,
      startDistanceKm: 0,
      endDistanceKm: 0,
      distanceKm: 0,
      durationS,
    }
  })
  const sauna = detail({
    sport: 'sauna',
    route: [],
    mapRoute: [],
    distanceKm: 0,
    elapsedTimeS: elapsedS,
    analysisRanges: laps.toReversed(),
    heartRateTrace: [heartRateTracePoint(0, 0, 90), heartRateTracePoint(0, elapsedS, 100)],
    sauna: { ...parsed.sauna, heartRateSource: null, source: 'manual' },
  })
  for (const embedded of [false, true]) {
    for (const location of [parsed.sauna.location, null]) {
      const rendered: Element = buildActivity(
        factory,
        { ...sauna, sauna: { ...parsed.sauna, location, heartRateSource: null, source: 'manual' } },
        true,
        ctx(),
        false,
        embedded,
      )
      const wrap = byClass(rendered, 'tri-sauna-laps')[0]
      assert.equal(wrap.properties.dataSaunaPhaseSource, 'manual')
      const phases = byClass(wrap, location ? 'tri-sauna-lap-legend' : 'tri-sauna-lap')
      assert.deepEqual(
        phases.map(phase => phase.properties.dataSaunaPhase),
        parsed.sauna.lapPhases,
      )
      assert.deepEqual(
        phases.map(phase => phase.properties.dataRangeId),
        laps.map(lap => lap.id),
      )
      assert.deepEqual(
        phases.map(phase => phase.properties.dataDurationS),
        durations.map(String),
      )
      assert.deepEqual(
        phases.map(phase => phase.properties.dataRangeLabel),
        ['lap 1 · hot sauna', 'lap 2 · cold plunge', 'lap 3 · hot sauna', 'lap 4 · break'],
      )
      assert.match(String(phases[1].properties.ariaLabel), /^lap 2 · cold plunge, 5:56,/)
    }
  }
  const unmatched = buildWorkoutAnalysis(factory, {
    ...sauna,
    sauna: {
      ...parsed.sauna,
      location: null,
      lapPhases: ['break'],
      heartRateSource: null,
      source: 'manual',
    },
  })
  assert.ok(unmatched)
  assert.equal(byClass(unmatched, 'tri-sauna-laps')[0].properties.dataSaunaPhaseSource, 'lap-order')
  assert.equal(byClass(unmatched, 'tri-sauna-lap')[0].properties.dataSaunaPhase, 'hot sauna')
})

test('renders recorded sauna HTL as ten one-point segments after activity graphs', () => {
  const sauna = detail({
    sport: 'sauna',
    route: [],
    mapRoute: [],
    distanceKm: 0,
    heartRateTrace: [heartRateTracePoint(0, 0, 90), heartRateTracePoint(0, 4_800, 100)],
    garmin: garminVerification({ aerobicTrainingEffect: 0.3 }),
    sauna: {
      time: '18:30',
      temperatureC: 72,
      humidityPct: 11,
      cooldown: 'cold plunge',
      heatTrainingLoad: 7.6,
      heartRateSource: null,
      source: 'manual',
    },
  })
  for (const embedded of [false, true]) {
    const card = buildActivity(factory, sauna, true, ctx(), false, embedded)
    const htl = byClass(card, 'tri-sauna-htl')[0]
    assert.ok(htl)
    assert.equal(htl.properties.dataSaunaHtlSource, 'manual')
    assert.equal(text(byClass(htl, 'tri-sauna-htl-score')[0]), '7.6 / 10')
    const meter = byClass(htl, 'tri-sauna-htl-meter')[0]
    assert.deepEqual(
      [
        meter.properties.role,
        meter.properties.ariaValueMin,
        meter.properties.ariaValueMax,
        meter.properties.ariaValueNow,
      ],
      ['meter', 0, 10, 7.6],
    )
    assert.equal(byClass(meter, 'tri-sauna-htl-segment').length, 10)
    assert.deepEqual(
      byClass(meter, 'tri-sauna-htl-fill').map(fill => fill.properties.style),
      [
        ...Array.from({ length: 7 }, () => 'width:100.0%'),
        'width:60.0%',
        'width:0.0%',
        'width:0.0%',
      ],
    )
    const more = byClass(card, 'tri-act-more')[0]
    assert.ok(
      more.children.indexOf(htl) > more.children.indexOf(byClass(card, 'tri-workout-analysis')[0]),
    )
    assert.ok(
      more.children.indexOf(byClass(card, 'tri-training-effect')[0]) < more.children.indexOf(htl),
    )
  }
  assert.ok(sauna.sauna)
  const boundaryScores: [number, string][] = [
    [0, 'width:0.0%'],
    [10, 'width:100.0%'],
    [-1, 'width:0.0%'],
    [11, 'width:100.0%'],
  ]
  for (const [value, fill] of boundaryScores) {
    const meter: Element | null = buildSaunaHeatTrainingLoad(factory, {
      ...sauna,
      sauna: { ...sauna.sauna, heatTrainingLoad: value },
    })
    assert.ok(meter)
    assert.deepEqual(
      byClass(meter, 'tri-sauna-htl-fill').map(node => node.properties.style),
      Array.from({ length: 10 }, () => fill),
    )
  }
  for (const heatTrainingLoad of [null, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(
      buildSaunaHeatTrainingLoad(factory, {
        ...sauna,
        sauna: { ...sauna.sauna, heatTrainingLoad },
      }),
      null,
    )
  }
  assert.equal(buildSaunaHeatTrainingLoad(factory, { ...sauna, sauna: null }), null)
  assert.equal(buildSaunaHeatTrainingLoad(factory, { ...sauna, sport: 'strength' }), null)
  const frenchHtl = buildSaunaHeatTrainingLoad(factoryFor(frenchPresentation), sauna)
  assert.ok(frenchHtl)
  assert.equal(text(byClass(frenchHtl, 'tri-sauna-htl-score')[0]), '7,6 / 10')
  assert.equal(
    byClass(frenchHtl, 'tri-sauna-htl-meter')[0].properties.ariaLabel,
    'charge d’entraînement à la chaleur',
  )
})

test('renders a route-less pool swim heart rate trace against metres', () => {
  const lap: ActivityAnalysisRange = {
    kind: 'lap',
    id: 'garmin-swim-lap:1',
    label: 'Lap 1',
    startElapsedS: 0,
    endElapsedS: 120,
    startDistanceKm: 0,
    endDistanceKm: 0.1,
    durationS: 120,
    movingTimeS: 120,
    distanceKm: 0.1,
    elevationGainM: null,
    averageSpeedKph: 3,
    averageHeartRate: 130,
    averageWatts: null,
    averageCadence: 25,
  }
  const swim = swimTrendDetail({
    distanceKm: 0.1,
    analysisRanges: [lap],
    heartRateTrace: [
      heartRateTracePoint(0, 0, 110),
      heartRateTracePoint(0.025, 30, 120),
      heartRateTracePoint(0.05, 60, null),
      heartRateTracePoint(0.075, 90, 140),
      heartRateTracePoint(0.1, 120, 150),
    ],
  })
  const rendered = buildActivity(factory, swim, true, ctx())
  const trace = byClass(rendered, 'tri-elev-wrap').find(
    element => element.properties.dataTriTrace === 'hr',
  )

  assert.ok(trace)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['80bpm', '100bpm', '120bpm', '140bpm'])
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['0 m', '50 m', '100 m'])
  assert.equal(byClass(trace, 'tri-analysis-selection').length, 1)
  assert.match(String(byClass(trace, 'tri-elev-line')[0]?.properties.d), /^M 0 /)

  const selectedTrace = buildHeartRateTrace(factory, swim, lap)
  assert.equal(byClass(selectedTrace, 'tri-analysis-selection')[0].properties.width, '100.00')
})

test('renders a route-less pool swim heart rate trace without distance against elapsed time', () => {
  const rendered = buildActivity(
    factory,
    swimTrendDetail({
      heartRateTrace: [
        heartRateTracePoint(0, 0, 110),
        heartRateTracePoint(0, 1_599, 140),
        heartRateTracePoint(0, 3_198, 123),
      ],
    }),
    true,
    ctx(),
  )
  const trace = byClass(rendered, 'tri-elev-wrap').find(
    element => element.properties.dataTriTrace === 'hr',
  )

  assert.ok(trace)
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['0s', '26:39', '53:18'])
  assert.equal(byClass(trace, 'tri-analysis-selection').length, 1)
  const graph = byClass(trace, 'tri-elev')[0]
  assert.equal(graph.properties.dataDomainStartElapsedS, 0)
  assert.equal(graph.properties.dataDomainEndElapsedS, 3_198)
})

test('renders activity graphs against a selected distance domain', () => {
  const ride = detail()
  const domain = { startDistanceKm: 10, endDistanceKm: 20 }
  const graphs = [
    buildElevation(factory, ride, null, domain),
    buildTrace(
      factory,
      ride,
      point => point.hr,
      'hr',
      max => `${max} bpm peak`,
      value => `${Math.round(value)}bpm`,
      undefined,
      null,
      domain,
    ),
  ]

  for (const graph of graphs) {
    const svg = byClass(graph, 'tri-elev')[0]
    assert.ok(svg)
    assert.equal(svg.properties.viewBox, '33.3333 0 33.3333 30')
    assert.equal(svg.properties.dataDomainStartDistanceKm, 10)
    assert.equal(svg.properties.dataDomainEndDistanceKm, 20)
    assert.deepEqual(byClass(graph, 'tri-cax-xt').map(text), ['12 km', '14 km', '16 km', '18 km'])
  }
})

test('resolves the active shifting pairing at distance', () => {
  const shifts = shiftedDetail().gearShifts

  assert.deepEqual(gearShiftAtFraction(shifts, 30, 0.4), { ...shifts[1], index: 1, xPct: 40 })
  assert.equal(gearShiftAtFraction(shifts, 30, -1)?.index, 0)
  assert.equal(gearShiftAtFraction(shifts, 30, 2)?.index, 3)
})

test('renders Garmin stamina and potential stamina on one fixed percentage scale', () => {
  const ride = detail({
    staminaTrace: {
      source: 'garmin',
      method: 'garmin-native',
      ftpWatts: null,
      maxHeartRateBpm: null,
    },
    route: detail().route.map((point, index) => ({
      ...point,
      stamina: [100, 76, 54, 32][index],
      potentialStamina: [100, 88, 67, 40][index],
    })),
  })
  const chart = buildStaminaChart(factory, ride, null)
  assert.ok(chart)
  assert.equal(chart.properties.dataTriTrace, 'stamina')
  assert.deepEqual(byClass(chart, 'tri-elev-d').map(text), ['stamina'])
  assert.deepEqual(byClass(chart, 'tri-stamina-legend-item').map(text), ['current', 'potential'])
  assert.deepEqual(byClass(chart, 'tri-cax-yt').map(text), ['0%', '25%', '50%', '75%', '100%'])
  assert.equal(byClass(chart, 'tri-stamina-area').length, 1)
  assert.equal(byClass(chart, 'tri-stamina-line--current').length, 1)
  assert.equal(byClass(chart, 'tri-stamina-line--potential').length, 1)
  assert.equal(byClass(chart, 'tri-elev-d')[0].properties.dataGlossDef, 'Garmin Connect')
  assert.equal(byClass(chart, 'tri-elev-d')[0].properties.tabIndex, 0)
  assert.equal(byClass(chart, 'tri-analysis-selection').length, 1)
  assert.equal(byClass(chart, 'tri-elev-cursor').length, 1)
  assert.deepEqual(byClass(chart, 'tri-elev-d').map(text), ['stamina'])
  assert.equal(byClass(chart, 'tri-trace-reference-k').length, 0)
})

test('keeps estimated stamina provenance in the title gloss', () => {
  const ride = detail({
    staminaTrace: {
      source: 'garden-estimate',
      method: 'garden-stamina-v2',
      ftpWatts: 287,
      maxHeartRateBpm: 196,
    },
    route: detail().route.map((point, index) => ({
      ...point,
      stamina: [100, 76, 54, 32][index],
      potentialStamina: [100, 88, 67, 40][index],
    })),
  })
  const chart = buildStaminaChart(factory, ride, null)

  assert.ok(chart)
  const title = byClass(chart, 'tri-elev-d')[0]
  assert.equal(text(title), 'stamina')
  assert.equal(title.properties.dataGloss, '')
  assert.equal(title.properties.dataGlossDef, 'estimate · FTP 287 W · max hr 196 bpm')
  assert.equal(title.properties.tabIndex, 0)
  assert.equal(byClass(chart, 'tri-trace-reference-k').length, 0)
})

test('renders power-weighted left and right pedal balance on a symmetric percentage scale', () => {
  const ride = detail({
    route: detail().route.map((point, index) => ({
      ...point,
      rightPowerPct: [48, 49, 52, 51][index],
    })),
  })
  const chart = buildPowerBalanceChart(factory, ride, null)
  assert.ok(chart)
  assert.equal(chart.properties.dataTriTrace, 'power-balance')
  assert.equal(chart.properties.dataCyclingChartMode, 'distance')
  assert.equal(byClass(chart, 'tri-power-balance-svg')[0].properties.ariaLabel, 'power balance')
  assert.deepEqual(byClass(chart, 'tri-elev-d').map(text), ['power balance'])
  assert.deepEqual(byClass(chart, 'tri-elev-range').map(text), ['L 49.7% / R 50.3% avg'])
  assert.deepEqual(byClass(chart, 'tri-power-balance-legend-item').map(text), ['left', 'right'])
  const modes = byClass(chart, 'tri-cycling-chart-modes')[0]
  assert.equal(modes.properties.role, 'group')
  assert.equal(modes.properties.ariaLabel, 'cycling charts view')
  assert.deepEqual(
    byClass(modes, 'tri-cycling-chart-mode').map(button => [
      text(button),
      button.properties.dataCyclingChartMode,
      button.properties.ariaPressed,
    ]),
    [
      ['distance', 'distance', 'true'],
      ['watts', 'power', 'false'],
    ],
  )
  const distancePane = byClass(chart, 'tri-power-balance-pane--distance')[0]
  const powerPane = byClass(chart, 'tri-power-balance-pane--power')[0]
  assert.ok(distancePane)
  assert.ok(powerPane)
  assert.equal(distancePane.properties.hidden, undefined)
  assert.equal(distancePane.properties.ariaHidden, 'false')
  assert.equal(powerPane.properties.hidden, true)
  assert.equal(powerPane.properties.ariaHidden, 'true')
  assert.deepEqual(byClass(distancePane, 'tri-cax-yt').map(text), ['45%', '50%', '55%'])
  assert.deepEqual(byClass(powerPane, 'tri-cax-yt').map(text), ['100% L', '50/50', '100% R'])
  assert.deepEqual(byClass(powerPane, 'tri-cax-xt').map(text), [
    '0 W',
    '100 W',
    '200 W',
    '300 W',
    '400 W',
    '500 W',
    '600 W',
  ])
  const heatmap = byClass(chart, 'tri-power-balance-heatmap')[0]
  assert.equal(heatmap.properties.ariaLabel, 'power balance by watts')
  assert.equal(heatmap.properties.dataPowerBalanceSamples, 4)
  assert.equal(heatmap.properties.dataPowerBalanceMaxWatts, 600)
  assert.equal(
    byClass(chart, 'tri-cycling-watts-heat-cell').reduce(
      (samples, cell) => samples + Number(cell.properties.dataSamples),
      0,
    ),
    4,
  )
  assert.equal(byClass(chart, 'tri-power-balance-reference').length, 1)
  assert.equal(byClass(chart, 'tri-power-balance-line--left').length, 1)
  assert.equal(byClass(chart, 'tri-power-balance-line--right').length, 1)
  assert.equal(byClass(chart, 'tri-analysis-selection').length, 1)
  assert.equal(byClass(chart, 'tri-elev-cursor').length, 1)

  const selected = buildPowerBalanceChart(factory, ride, null, false, {
    startDistanceKm: 0,
    endDistanceKm: 10,
  })
  assert.ok(selected)
  assert.equal(
    byClass(selected, 'tri-power-balance-heatmap')[0].properties.dataPowerBalanceSamples,
    2,
  )

  const embedded = buildActivity(factory, ride, true, undefined, false, true)
  const embeddedChart = byClass(embedded, 'tri-power-balance-chart')[0]
  assert.ok(embeddedChart)
  assert.deepEqual(byClass(embeddedChart, 'tri-elev-d').map(text), ['power balance'])
  assert.deepEqual(byClass(embeddedChart, 'tri-elev-range').map(text), ['L 49.7% / R 50.3%'])
  assert.deepEqual(byClass(embeddedChart, 'tri-power-balance-legend-item').map(text), [])
  assert.equal(
    byClass(embeddedChart, 'tri-power-balance-svg')[0].properties.ariaLabel,
    'power balance',
  )
})

test('bridges zero-power pedal balance ranges with dotted left and right traces', () => {
  const ride = detail({
    route: detail().route.map((point, index) => ({
      ...point,
      w: [180, 0, 0, 220][index],
      rightPowerPct: [48, 50, 50, 52][index],
    })),
  })
  const chart = buildPowerBalanceChart(factory, ride, null)
  assert.ok(chart)
  const missing = byClass(chart, 'tri-power-balance-line--missing')
  assert.equal(missing.length, 2)
  assert.ok(
    missing.every(path => String(path.properties.d).match(/^M [\d.]+ [\d.]+ L [\d.]+ [\d.]+ $/)),
  )
  assert.equal(byClass(chart, 'tri-power-balance-line--left').length, 2)
  assert.equal(byClass(chart, 'tri-power-balance-line--right').length, 2)
})

test('places pedal balance after the available common traces', () => {
  const ride = detail({
    route: detail().route.map((point, index) => ({
      ...point,
      rightPowerPct: [48, 49, 52, 51][index],
    })),
  })
  const rendered = buildActivity(factory, ride, true)
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const powerIndex = children.findIndex(child => child.properties.dataTriTrace === 'power')
  assert.ok(powerIndex >= 0)
  assert.equal(children[powerIndex + 1].properties.dataTriTrace, 'power-balance')
})

test('renders cycling dynamics and rider position immediately below pedal balance', () => {
  const ride = cyclingDynamicsDetail()
  const rendered = buildActivity(factory, ride, true)
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const balanceIndex = children.findIndex(
    child => child.properties.dataTriTrace === 'power-balance',
  )
  assert.ok(balanceIndex >= 0)
  assert.deepEqual(
    children.slice(balanceIndex, balanceIndex + 5).map(child => child.properties.dataTriTrace),
    ['power-balance', 'torque-effectiveness', 'pedal-smoothness', 'power-phase', 'rider-position'],
  )

  const torque = byClass(rendered, 'tri-torque-effectiveness-chart')[0]
  const smoothness = byClass(rendered, 'tri-pedal-smoothness-chart')[0]
  const phase = byClass(rendered, 'tri-power-phase-chart')[0]
  const position = byClass(rendered, 'tri-rider-position-chart')[0]
  assert.ok(torque)
  assert.ok(smoothness)
  assert.ok(phase)
  assert.ok(position)
  assert.equal(byClass(rendered, 'tri-cycling-chart-modes').length, 1)
  assert.equal(byClass(rendered, 'tri-cycling-chart-mode').length, 2)
  assert.deepEqual(byClass(torque, 'tri-elev-d').map(text), ['torque effectiveness'])
  assert.deepEqual(byClass(smoothness, 'tri-elev-d').map(text), ['pedal smoothness'])
  assert.deepEqual(byClass(phase, 'tri-elev-d').map(text), ['power phase'])
  assert.deepEqual(byClass(position, 'tri-elev-d').map(text), ['rider position'])
  assert.deepEqual(
    [torque, smoothness, phase].map(chart => {
      const title = byClass(chart, 'tri-elev-d')[0]
      return [title.properties.dataGloss, title.properties.tabIndex]
    }),
    [
      ['torque effectiveness', 0],
      ['pedal smoothness', 0],
      ['power phase', 0],
    ],
  )
  assert.deepEqual(glossFor('en', 'torque effectiveness'), {
    term: 'torque effectiveness (TE)',
    def: 'Torque effectiveness compares the positive torque that drives the crank with negative torque that resists it during each revolution. A value of 100% means no negative torque was recorded. Interpret left and right trends alongside power and cadence; the metric has no universal target.',
  })
  assert.deepEqual(glossFor('fr', 'pedal smoothness'), {
    term: 'fluidité du pédalage (PS)',
    def: "La fluidité du pédalage est la puissance moyenne divisée par la puissance maximale sur un tour de manivelle. Une valeur plus élevée signifie que la puissance est répartie plus uniformément sur le tour. Elle décrit la forme de l'application de la puissance; la puissance totale et le rendement sont des mesures distinctes.",
  })
  assert.equal(glossFor('en', 'power phase')?.def.includes('360°→0°'), true)
  assert.equal(byClass(torque, 'tri-cycling-dynamics-line--left').length, 2)
  assert.equal(byClass(smoothness, 'tri-cycling-dynamics-line--right').length, 2)
  assert.deepEqual(byClass(torque, 'tri-cycling-dynamics-legend-item').map(text), ['left', 'right'])
  assert.deepEqual(byClass(smoothness, 'tri-cycling-dynamics-legend-item').map(text), [
    'left',
    'right',
  ])
  for (const [chart, title, className, sampleProperty] of [
    [torque, 'torque effectiveness', 'torque-effectiveness', 'dataTorqueEffectivenessSamples'],
    [smoothness, 'pedal smoothness', 'pedal-smoothness', 'dataPedalSmoothnessSamples'],
  ] as const) {
    assert.equal(chart.properties.dataCyclingChartMode, 'distance')
    assert.deepEqual(byClass(chart, 'tri-cycling-chart-modes'), [])
    const distancePane = byClass(chart, `tri-${className}-pane--distance`)[0]
    const powerPane = byClass(chart, `tri-${className}-pane--power`)[0]
    assert.equal(distancePane.properties.hidden, undefined)
    assert.equal(distancePane.properties.ariaHidden, 'false')
    assert.equal(powerPane.properties.hidden, true)
    assert.equal(powerPane.properties.ariaHidden, 'true')
    assert.deepEqual(byClass(powerPane, 'tri-cax-xt').map(text), [
      '0 W',
      '100 W',
      '200 W',
      '300 W',
      '400 W',
      '500 W',
      '600 W',
    ])
    const heatmap = byClass(chart, `tri-${className}-heatmap`)[0]
    assert.equal(heatmap.properties.ariaLabel, `${title} by watts`)
    assert.equal(heatmap.properties[sampleProperty], 6)
    assert.equal(
      byClass(heatmap, 'tri-cycling-watts-heat-cell').reduce(
        (samples, cell) => samples + Number(cell.properties.dataSamples),
        0,
      ),
      6,
    )
    assert.ok(byClass(heatmap, 'tri-cycling-watts-heat-cell--left').length > 0)
    assert.ok(byClass(heatmap, 'tri-cycling-watts-heat-cell--right').length > 0)
  }
  const selectedTorque = buildTorqueEffectivenessChart(factory, ride, null, false, {
    startDistanceKm: 0,
    endDistanceKm: 0,
  })
  assert.ok(selectedTorque)
  assert.equal(
    byClass(selectedTorque, 'tri-torque-effectiveness-heatmap')[0].properties
      .dataTorqueEffectivenessSamples,
    2,
  )
  assert.equal(byClass(phase, 'tri-power-phase-line--start').length, 2)
  assert.equal(byClass(phase, 'tri-power-phase-line--end').length, 2)
  assert.equal(byClass(phase, 'tri-cycling-dynamics-legend-item').length, 4)
  assert.equal(byClass(position, 'tri-rider-position-standing').length, 1)
  assert.deepEqual(byClass(position, 'tri-elev-range').map(text), ['standing 20:00 · 25.0%'])
  assert.ok(ride.cyclingDynamics)
  assert.equal(cyclingDynamicsIndexAtDistance(ride.cyclingDynamics, 16), 2)
  assert.equal(riderPositionAtDistance(ride.cyclingDynamics, 15), 'standing')

  const embedded = buildActivity(factory, ride, true, undefined, false, true)
  const embeddedTorque = byClass(embedded, 'tri-torque-effectiveness-chart')[0]
  const embeddedSmoothness = byClass(embedded, 'tri-pedal-smoothness-chart')[0]
  const embeddedPhase = byClass(embedded, 'tri-power-phase-chart')[0]
  assert.ok(embeddedTorque)
  assert.ok(embeddedSmoothness)
  assert.ok(embeddedPhase)
  assert.deepEqual(byClass(embeddedTorque, 'tri-cycling-dynamics-legend-item'), [])
  assert.deepEqual(byClass(embeddedSmoothness, 'tri-cycling-dynamics-legend-item'), [])
  assert.equal(byClass(embeddedPhase, 'tri-cycling-dynamics-legend-item').length, 4)
})

test('groups stamina, performance condition, and thermal graphs before heart rate and sport-specific charts', () => {
  const ride = cyclingDynamicsDetail()
  ride.gearShifts = shiftedDetail().gearShifts
  ride.staminaTrace = {
    source: 'garmin',
    method: 'garmin-native',
    ftpWatts: null,
    maxHeartRateBpm: null,
  }
  ride.route = ride.route.map((point, index) => ({
    ...point,
    stamina: [100, 76, 54, 32][index],
    potentialStamina: [100, 88, 67, 40][index],
    performanceCondition: [2, 1, -1, -2][index],
    heatStrainIndex: [0, 0.4, 0.8, 1.2][index],
    heatStrainSource: 'core-app',
    coreTemperatureC: [37.2, 37.5, 37.8, 38.1][index],
    coreTemperatureSource: 'core-app',
    skinTemperatureC: [32.5, 32.4, 32.2, 32.0][index],
    skinTemperatureSource: 'core-app',
  }))
  const run = detail({
    sport: 'run',
    staminaTrace: ride.staminaTrace,
    route: ride.route.map(point => ({ ...point, rightPowerPct: null })),
  })
  const walk = detail({ ...run, sport: 'walk', deviceWatts: false })
  for (const activity of [ride, run, walk]) {
    for (const embedded of [false, true]) {
      const rendered = buildActivity(factory, activity, true, undefined, false, embedded)
      const more = byClass(rendered, 'tri-act-more')[0]
      assert.ok(more)
      const traces = more.children
        .filter((child): child is Element => child.type === 'element')
        .map(child => child.properties.dataTriTrace)
        .filter(trace => trace != null)
      const common = [
        'stamina',
        'performance-condition',
        'heat-strain-index',
        'core-temperature',
        'skin-temperature',
        ...(activity.sport === 'walk' ? [] : ['hr']),
        'temperature',
        'cadence',
        'respiration',
        activity.sport === 'walk' ? 'pace' : 'power',
      ]
      assert.deepEqual(traces.slice(0, common.length), common)
      if (activity.sport === 'bike') {
        assert.equal(traces[common.length], 'power-balance')
        assert.ok(traces.indexOf('electronic-shifting') > common.length)
        assert.ok(traces.indexOf('speed') > common.length)
      }
    }
  }
})

test('renders front and rear shifting on separate overlaid y axes', () => {
  const chart = buildShiftingChart(factory, shiftedDetail(), null)
  assert.ok(chart)
  assert.deepEqual(byClass(chart, 'tri-elev-d').map(text), ['electronic shifting'])
  assert.deepEqual(byClass(chart, 'tri-elev-range').map(text), ['52×27 · 26:40'])
  assert.deepEqual(byClass(chart, 'tri-shift-legend-item').map(text), ['front', 'rear'])
  assert.equal(byClass(chart, 'tri-shift-legend-line').length, 2)
  assert.equal(
    classNames(byClass(chart, 'tri-elev-cap')[0]).includes('tri-elev-cap--summary'),
    true,
  )
  const distancePane = byClass(chart, 'tri-shift-pane--distance')[0]
  const powerPane = byClass(chart, 'tri-shift-pane--power')[0]
  assert.deepEqual(byClass(distancePane, 'tri-cax-yt').map(text), ['36T', '52T'])
  assert.equal(byClass(chart, 'tri-cax-yt--right').length, 0)
  assert.deepEqual(byClass(distancePane, 'tri-cax-xt').map(text), ['10 km', '20 km'])
  assert.equal(chart.properties.dataCyclingChartMode, 'distance')
  assert.equal(distancePane.properties.hidden, undefined)
  assert.equal(distancePane.properties.ariaHidden, 'false')
  assert.equal(powerPane.properties.hidden, true)
  assert.equal(powerPane.properties.ariaHidden, 'true')
  assert.deepEqual(byClass(chart, 'tri-cycling-chart-modes'), [])
  assert.deepEqual(byClass(powerPane, 'tri-cax-yt').map(text), ['36×19', '52×27', '52×19', '36×11'])
  assert.deepEqual(byClass(powerPane, 'tri-cax-xt').map(text), [
    '0 W',
    '100 W',
    '200 W',
    '300 W',
    '400 W',
    '500 W',
    '600 W',
  ])
  const heatmap = byClass(chart, 'tri-shift-heatmap')[0]
  assert.equal(heatmap.properties.ariaLabel, 'electronic shifting by watts')
  assert.equal(heatmap.properties.dataElectronicShiftingSamples, 4)
  assert.equal(heatmap.properties.dataElectronicShiftingMaxWatts, 600)
  assert.equal(
    byClass(heatmap, 'tri-cycling-watts-heat-cell').reduce(
      (samples, cell) => samples + Number(cell.properties.dataSamples),
      0,
    ),
    4,
  )
  const selected = buildShiftingChart(factory, shiftedDetail(), null, {
    startDistanceKm: 0,
    endDistanceKm: 0,
  })
  assert.ok(selected)
  assert.equal(
    byClass(selected, 'tri-shift-heatmap')[0].properties.dataElectronicShiftingSamples,
    1,
  )
  assert.equal(byClass(chart, 'tri-shift-line').length, 2)
  assert.equal(byClass(chart, 'tri-analysis-selection').length, 1)
  assert.equal(chart.properties.dataTriTrace, 'electronic-shifting')
  const svg = byClass(chart, 'tri-shift-svg')[0]
  assert.ok(svg)
  assert.equal(classNames(svg).includes('tri-elev'), true)
  assert.equal(byClass(svg, 'tri-elev-cursor').length, 1)
})

test('aggregates repeated visits when choosing the longest-held gear pairing', () => {
  const ride = shiftedDetail()
  ride.gearShifts = [
    { ...ride.gearShifts[0], elapsedS: 0, distanceKm: 0, frontTeeth: 52, rearTeeth: 19 },
    { ...ride.gearShifts[1], elapsedS: 600, distanceKm: 4, frontTeeth: 36, rearTeeth: 27 },
    { ...ride.gearShifts[2], elapsedS: 1_200, distanceKm: 8, frontTeeth: 52, rearTeeth: 19 },
    { ...ride.gearShifts[3], elapsedS: 3_600, distanceKm: 24, frontTeeth: 36, rearTeeth: 11 },
  ]
  const chart = buildShiftingChart(factory, ride)
  assert.ok(chart)
  assert.deepEqual(byClass(chart, 'tri-elev-range').map(text), ['52×19 · 50:00'])
})

test('normalizes electronic shifting time by exact gear ratio', () => {
  const ride = shiftedDetail()
  ride.gearShifts = [
    { ...ride.gearShifts[0], elapsedS: 0, frontTeeth: 52, rearTeeth: 19 },
    { ...ride.gearShifts[1], elapsedS: 600, frontTeeth: 36, rearTeeth: 27 },
    { ...ride.gearShifts[2], elapsedS: 1_200, frontTeeth: 52, rearTeeth: 19 },
    { ...ride.gearShifts[3], elapsedS: 3_600, frontTeeth: 36, rearTeeth: 11 },
  ]

  assert.deepEqual(activityGearRatioDistribution(ride), [
    { ratio: 1.333333, percentage: 12.5, pairings: [{ frontTeeth: 36, rearTeeth: 27 }] },
    { ratio: 2.736842, percentage: 62.5, pairings: [{ frontTeeth: 52, rearTeeth: 19 }] },
    { ratio: 3.272727, percentage: 25, pairings: [{ frontTeeth: 36, rearTeeth: 11 }] },
  ])
  assert.deepEqual(activityGearRatioDistribution({ ...ride, sport: 'run' }), [])
})

test('gear ratio distributions retain distinct tooth combinations with the same ratio', () => {
  const ride = shiftedDetail()
  ride.route = []
  ride.movingTimeS = 400
  ride.gearShifts = [
    { ...ride.gearShifts[0], elapsedS: 0, frontTeeth: 52, rearTeeth: 26 },
    { ...ride.gearShifts[0], elapsedS: 100, frontTeeth: 36, rearTeeth: 18 },
    { ...ride.gearShifts[0], elapsedS: 200, frontTeeth: 52, rearTeeth: 26 },
    { ...ride.gearShifts[0], elapsedS: 300, frontTeeth: 52, rearTeeth: 13 },
    { ...ride.gearShifts[0], elapsedS: 400, frontTeeth: 36, rearTeeth: 9 },
  ]
  assert.deepEqual(activityGearRatioDistribution(ride), [
    {
      ratio: 2,
      percentage: 75,
      pairings: [
        { frontTeeth: 36, rearTeeth: 18 },
        { frontTeeth: 52, rearTeeth: 26 },
      ],
    },
    { ratio: 4, percentage: 25, pairings: [{ frontTeeth: 52, rearTeeth: 13 }] },
  ])
  assert.deepEqual(activityGearRatioDistribution({ ...ride, gearShifts: [] }), [])
})

test('places electronic shifting after the available common traces', () => {
  const rendered = buildActivity(factory, shiftedDetail(), true)
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const powerIndex = children.findIndex(child => child.properties.dataTriTrace === 'power')
  assert.ok(powerIndex >= 0)
  assert.equal(classNames(children[powerIndex + 1]).includes('tri-shift-chart'), true)
})

test('centres a fixed front chainring while the rear cassette changes', () => {
  const ride = shiftedDetail()
  ride.gearShifts = ride.gearShifts.map(shift => ({ ...shift, frontGearNum: 1, frontTeeth: 40 }))
  const chart = buildShiftingChart(factory, ride)
  assert.ok(chart)
  assert.deepEqual(
    byClass(byClass(chart, 'tri-shift-pane--distance')[0], 'tri-cax-yt')
      .filter(tick => !classNames(tick).includes('tri-cax-yt--right'))
      .map(text),
    ['40T'],
  )
  assert.match(String(byClass(chart, 'tri-shift-line--front')[0].properties.d), /^M 0 15\.00/)
})

test('extends the first measured trace value to distance zero', () => {
  const trace = buildTrace(
    factory,
    detail({
      route: detail().route.map((point, index) => ({ ...point, d: index === 0 ? 0.183 : point.d })),
    }),
    point => point.hr,
    'hr',
    max => `${max} bpm peak`,
    value => `${Math.round(value)}bpm`,
  )
  const area = byClass(trace, 'tri-elev-area')[0]
  const line = byClass(trace, 'tri-elev-line')[0]

  assert.ok(area)
  assert.ok(line)
  assert.match(String(area.properties.d), /^M 0 30 L 0 ([\d.]+) L 0\.61 \1 /)
  assert.match(String(line.properties.d), /^M 0 ([\d.]+) L 0\.61 \1 /)
})

test('places training effect and zones directly after performance condition across activity cards', () => {
  const sports: StravaActivityDetail['sport'][] = [
    'bike',
    'run',
    'walk',
    'swim',
    'sauna',
    'strength',
  ]
  for (const sport of sports) {
    for (const embedded of [false, true]) {
      for (const hasRoute of [false, true]) {
        const activity = detail({
          ...zonedDetail(),
          sport,
          garmin: garminVerification({ aerobicTrainingEffect: 3.2, anaerobicTrainingEffect: 0 }),
          route: hasRoute
            ? detail().route.map((point, index) => ({ ...point, performanceCondition: index - 2 }))
            : [],
          heartRateTrace: [heartRateTracePoint(0, 0, 110), heartRateTracePoint(0, 4_800, 140)],
        })
        const rendered = buildActivity(factory, activity, true, ctx(), false, embedded)
        const more = byClass(rendered, 'tri-act-more')[0]
        const children = more.children.filter((child): child is Element => child.type === 'element')
        const trainingEffect = byClass(more, 'tri-training-effect')
        const zones = byClass(more, 'tri-zone-duo')
        assert.equal(trainingEffect.length, 1)
        assert.equal(zones.length, 1)
        const effectIndex = children.indexOf(trainingEffect[0])
        assert.equal(children.indexOf(zones[0]), effectIndex + 1)
        const conditionIndex = children.findIndex(
          child =>
            child.properties.dataTriTrace === 'performance-condition' ||
            child.properties.dataTriUnavailable === 'performance-condition',
        )
        assert.ok(conditionIndex >= 0)
        assert.equal(effectIndex, conditionIndex + 1)
      }
    }
  }
})

test('pairs hr/power zones with aligned captions', () => {
  const rendered = buildActivity(factory, zonedDetail(), true, ctx())
  const duos = byClass(rendered, 'tri-zone-duo')
  assert.equal(duos.length, 1)
  assert.deepEqual(
    duos.flatMap(duo =>
      duo.children
        .filter((child): child is Element => child.type === 'element')
        .map(child => child.properties.dataTriTrace),
    ),
    ['heart-rate-zones', 'power-zones'],
  )
  assert.deepEqual(byClass(duos[0], 'tri-zone-title').map(text), [
    'heart rate zones',
    'power zones',
  ])
  assert.deepEqual(byClass(duos[0], 'tri-zone-cap').map(text), [
    'based on vt1 150 bpm',
    'based on FTP 260 W',
  ])
  const zoneTables = byClass(duos[0], 'tri-zone')
  assert.deepEqual(byClass(zoneTables[0], 'tri-zone-name').map(text), [
    'anaerobic',
    'threshold',
    'tempo',
    'endurance',
    'recovery',
  ])
  assert.deepEqual(byClass(zoneTables[1], 'tri-zone-name').map(text), [
    'neuromuscular',
    'anaerobic',
    'VO2max',
    'threshold',
    'tempo',
    'endurance',
    'recovery',
  ])
  assert.equal(byClass(zoneTables[0], 'tri-zone-grid')[0].properties.role, 'list')
  const zoneRows = byClass(zoneTables[1], 'tri-zone-row')
  assert.equal(zoneRows[0].properties.role, 'listitem')
  assert.match(String(zoneRows[0].properties.ariaLabel), /^Z7, neuromuscular, > 400w, 40s, /)
  assert.equal(byClass(zoneTables[1], 'tri-zone-z')[0].properties.tabIndex, undefined)
})

test('gives cycling and running power curves and distributions their own activity rows', () => {
  const sports: StravaActivityDetail['sport'][] = ['bike', 'run']
  for (const sport of sports) {
    for (const embedded of [false, true]) {
      const activity = { ...zonedDetail(), sport }
      const rendered = buildActivity(factory, activity, true, ctx(), false, embedded)
      const more = byClass(rendered, 'tri-act-more')[0]
      assert.ok(more)
      const children = more.children.filter((child): child is Element => child.type === 'element')
      const curveIndex = children.findIndex(
        child => child.properties.dataTriTrace === 'power-curve',
      )
      assert.ok(curveIndex >= 0)
      assert.equal(children[curveIndex + 1].properties.dataTriTrace, '25w-power-distribution')
    }
  }
})

test('removes zone duos from simplified activity details', () => {
  const rendered = buildActivity(
    factory,
    zonedDetail(),
    true,
    ctx(),
    false,
    true,
    TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified,
  )

  assert.equal(byClass(rendered, 'tri-zone-duo').length, 0)
  for (const trace of ['heart-rate-zones', 'power-zones', 'power-curve', '25w-power-distribution'])
    assert.equal(
      descendants(rendered, element => element.properties.dataTriTrace === trace).length,
      0,
      `${trace} should be hidden`,
    )
})

test('places heart rate zones before swim charts in expanded details', () => {
  const rendered = buildActivity(
    factory,
    swimTrendDetail({ hrZones: [20, 40, 30, 10, 0] }),
    true,
    ctx(),
  )
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const swimIndex = children.findIndex(child => classNames(child).includes('tri-swim-trends'))
  const zonesIndex = children.findIndex(child =>
    byClass(child, 'tri-zone-title').some(title => text(title) === 'heart rate zones'),
  )

  assert.ok(swimIndex >= 0)
  assert.ok(zonesIndex >= 0 && zonesIndex < swimIndex)
})

test('glosses the swim rate, cadence, and SWOLF titles', () => {
  const rendered = buildActivity(factory, swimTrendDetail(), true, ctx())
  const glossed = new Map(
    byClass(rendered, 'tri-swim-trend-title').map(title => [
      text(title),
      title.properties.dataGloss,
    ]),
  )
  assert.equal(glossed.get('stroke rate spm'), 'strokerate')
  assert.equal(glossed.get('cadence str/length'), 'swimcadence')
  assert.equal(glossed.get('SWOLF'), 'swolf')
  assert.equal(glossed.get('pace /100m'), undefined)
  for (const key of ['strokerate', 'swimcadence', 'swolf']) {
    assert.ok(glossFor('en', key)?.def)
    assert.ok(glossFor('fr', key)?.def)
  }
})

test('places the pool overview above stroke distances in full and embedded swim summaries', () => {
  const activity = { ...swimToggleDetail(), strokes: { freestyle: 75, breaststroke: 25 } }
  for (const embedded of [false, true]) {
    const rendered = buildActivity(factory, activity, true, undefined, false, embedded)
    const summary = byClass(rendered, 'tri-act-figs--pool')[0]
    const more = byClass(rendered, 'tri-act-more')[0]
    assert.ok(summary)
    assert.ok(more)
    assert.equal(byClass(rendered, 'tri-pool').length, 1)
    assert.equal(byClass(more, 'tri-pool').length, 0)
    const pool = byClass(summary, 'tri-pool-wrap')[0]
    assert.ok(pool)
    const children = pool.children.filter((child): child is Element => child.type === 'element')
    assert.ok(classNames(children[0]).includes('tri-pool'))
    assert.ok(classNames(children[1]).includes('tri-pool-cap'))
    assert.ok(classNames(children[2]).includes('tri-pool-strokes'))
    assert.deepEqual(byClass(pool, 'tri-stroke-name').map(text), ['freestyle', 'breast'])
    assert.deepEqual(byClass(pool, 'tri-stroke-distance').map(text), ['75 m', '25 m'])
  }
})

test('places cycling efforts after the expanded charts', () => {
  const rendered = buildActivity(factory, zonedDetail(), true, ctx())
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const last = children[children.length - 1]
  assert.ok(last)
  assert.equal(classNames(last).includes('tri-efforts'), true)
})

test('renders power curve unit controls and activity-day weight for full and embedded cards', () => {
  for (const embedded of [false, true]) {
    const activity = zonedDetail()
    const curve = buildPowerCurve(factory, activity, ctx(), embedded)
    assert.ok(curve)
    const buttons = byClass(curve, 'tri-curve-unit')
    assert.deepEqual(buttons.map(text), ['W', 'W/kg'])
    assert.deepEqual(
      buttons.map(button => button.properties.ariaPressed),
      ['true', 'false'],
    )
    assert.ok(
      buttons.every(button => button.tagName === 'button' && button.properties.type === 'button'),
    )
    assert.equal(buttons[1].properties.disabled, undefined)
    assert.equal(byClass(curve, 'tri-curve-units')[0].properties.ariaLabel, 'power curve units')
    const svg = byClass(curve, 'tri-curve-svg')[0]
    assert.equal(svg.properties.dataCurveWeightKg, 87.55)
    assert.equal(svg.properties.dataCurveWattStep, 100)
    assert.deepEqual(decodedPowerCurves(svg)[0], activity.powerCurve)
    const note = byClass(curve, 'tri-curve-weight-note')[0]
    assert.equal(note.properties.hidden, true)
    assert.equal(note.tagName, 'span')
    assert.equal(text(note), '87.55 kg')
    assert.equal(note.properties.title, 'Garmin · 2026-07-09')
    assert.ok(byClass(curve, 'tri-elev-cap')[0].children.includes(note))
    assert.equal(curve.children.includes(note), false)
    assert.deepEqual(
      byClass(curve, 'tri-curve-power-value').map(value => Number(value.properties.dataCurveWatts)),
      embedded ? [540, 320, 250, 230] : [540, 320, 250, 230, 260, 280],
    )
  }
  const run = {
    ...zonedDetail(),
    sport: 'run',
    bestEfforts: null,
    powerCurveWeight: { kg: 75, date: '2026-07-09', source: 'garmin' },
  } satisfies StravaActivityDetail
  const curve = buildPowerCurve(factory, run, ctx())
  assert.ok(curve)
  assert.equal(byClass(curve, 'tri-curve-unit')[1].properties.disabled, undefined)
  assert.equal(byClass(curve, 'tri-curve-svg')[0].properties.dataCurveWeightKg, 75)
})

test('renders power curve W/kg with earlier Garmin weight and its measurement date', () => {
  for (const embedded of [false, true]) {
    const weight = { kg: 76.79, date: '2026-07-08', source: 'garmin' } satisfies NonNullable<
      StravaActivityDetail['powerCurveWeight']
    >
    const activity = { ...zonedDetail(), powerCurveWeight: weight }
    const curve = buildPowerCurve(factory, activity, ctx(), embedded)
    assert.ok(curve)
    assert.deepEqual(powerCurveWeight(activity), weight)
    assert.equal(byClass(curve, 'tri-curve-unit')[1].properties.disabled, undefined)
    assert.equal(byClass(curve, 'tri-curve-svg')[0].properties.dataCurveWeightKg, weight.kg)
    const note = byClass(curve, 'tri-curve-weight-note')[0]
    assert.equal(text(note), '76.79 kg · 2026-07-08')
    assert.equal(note.properties.title, 'Garmin · 2026-07-08')
    assert.equal(note.properties.hidden, true)
    assert.ok(activity.bestEfforts)
    assert.deepEqual(
      powerCurveWeight({
        ...activity,
        powerCurveWeight: undefined,
        bestEfforts: { ...activity.bestEfforts, weightKg: weight.kg, weightDate: weight.date },
      }),
      weight,
    )
  }
})

test('disables power curve W/kg without a valid weight on or before the activity date', () => {
  for (const kg of [null, 0, -75, NaN, Infinity]) {
    const activity = zonedDetail()
    assert.ok(activity.bestEfforts)
    activity.bestEfforts = { ...activity.bestEfforts, weightKg: kg }
    const curve = buildPowerCurve(factory, activity, ctx())
    assert.ok(curve)
    assert.equal(powerCurveWeight(activity), undefined)
    assert.equal(byClass(curve, 'tri-curve-unit')[1].properties.disabled, true)
    assert.equal(byClass(curve, 'tri-curve-svg')[0].properties.dataCurveWeightKg, '')
    assert.equal(byClass(curve, 'tri-curve-weight-note').length, 0)
    assert.equal(
      byClass(curve, 'tri-curve-unit')[1].properties.title,
      'W/kg unavailable: no weight recorded on or before this activity',
    )
  }
  assert.equal(powerCurveWeight(detail({ bestEfforts: null })), undefined)
  assert.equal(
    powerCurveWeight(
      detail({
        bestEfforts: null,
        powerCurveWeight: { kg: 75, date: '2026-07-10', source: 'garmin' },
      }),
    ),
    undefined,
  )
  for (const date of ['', '2026-7-08', 'unknown'])
    assert.equal(
      powerCurveWeight(
        detail({ bestEfforts: null, powerCurveWeight: { kg: 75, date, source: 'garmin' } }),
      ),
      undefined,
    )
})

test('converts power curve values with fractional kilograms and preserves measured zero', () => {
  assert.equal(powerCurveValueText(264, 87.09, 'en'), '3.03 W/kg')
  assert.equal(powerCurveValueText(0, 87.09, 'en'), '0.00 W/kg')
  assert.equal(powerCurveValueText(264, 87.09, 'fr'), '3,03 W/kg')
  assert.equal(powerCurveValueText(234.3, 87.09, 'en'), '2.69 W/kg')
  assert.equal(powerCurveValueText(264, null, 'en'), '264 W')
  assert.equal(powerCurveValueText(264, null, 'en', true), '264W')
  assert.equal(powerCurveValueText(234.3, null, 'en'), '234.3 W')
})

test('keeps power curve W/kg ticks on the common watt geometry and restores the watt axis', () => {
  const watts = powerCurveAxisTicks(1_200, 200, null, 'en')
  const relative = powerCurveAxisTicks(1_200, 200, 80, 'en')
  assert.deepEqual(relative, [
    { label: '0', watts: 0 },
    { label: '5 W/kg', watts: 400 },
    { label: '10 W/kg', watts: 800 },
    { label: '15 W/kg', watts: 1_200 },
  ])
  assert.deepEqual(powerCurveAxisTicks(1_200, 200, null, 'en'), watts)
  assert.deepEqual(
    watts.map(tick => tick.label),
    ['0', '200w', '400w', '600w', '800w', '1,000w', '1,200w'],
  )
  assert.ok(powerCurveAxisTicks(1_500, 500, 87.09, 'en').every(tick => tick.watts <= 1_500))
  assert.deepEqual(powerCurveAxisTicks(1_200, 200, 0, 'en'), watts)
})

test('scales power curve y axis with nice watt ticks', () => {
  const curve = buildPowerCurve(factory, zonedDetail(), ctx())
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-yt').map(text), [
    '0',
    '100w',
    '200w',
    '300w',
    '400w',
    '500w',
    '600w',
  ])
})

test('labels a power curve through its endpoint beyond three hours', () => {
  const curve = buildPowerCurve(
    factory,
    detail({
      powerCurve: [
        { s: 1, w: 565 },
        { s: 20_107, w: 180 },
      ],
    }),
    ctx(),
  )
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-xt').map(text), [
    '1s',
    '5s',
    '10s',
    '20s',
    '30s',
    '1m',
    '2m',
    '3m',
    '5m',
    '6m',
    '12m',
    '20m',
    '1h',
    '5h35m',
  ])
  const lastTick = byClass(curve, 'tri-cax-xt').at(-1)
  assert.ok(lastTick)
  assert.equal(classNames(lastTick).includes('tri-cax-xt--last'), true)
  const ticks = byClass(curve, 'tri-curve-tick')
  assert.equal(
    ticks.every(tick => tick.tagName === 'button'),
    true,
  )
  assert.deepEqual(
    ticks.map(tick => tick.properties.dataCurveSeconds),
    ['1', '5', '10', '20', '30', '60', '120', '180', '300', '360', '720', '1200', '3600', '20107'],
  )
  assert.deepEqual(
    ticks.map(tick => tick.properties.ariaPressed),
    ticks.map((_, index) => String(index === 0)),
  )
})

test('uses sparse second markers in embedded power curves', () => {
  const activity = detail({
    date: '2026-08-04',
    powerCurve: [
      { s: 1, w: 565 },
      { s: 3_600, w: 180 },
    ],
  })
  const card = buildDayCard(
    factory,
    activity.date,
    { details: { [activity.id]: activity }, health: {} },
    { embedded: true, expanded: true },
    undefined,
    ctx(),
  )
  assert.deepEqual(byClass(card, 'tri-curve-tick').map(text), [
    '1s',
    '5s',
    '30s',
    '1m',
    '2m',
    '3m',
    '5m',
    '6m',
    '12m',
    '20m',
    '1h',
  ])
})

test('keeps precise embedded power curve endpoints clear of the previous marker', () => {
  const activity = detail({
    date: '2026-08-11',
    powerCurve: [
      { s: 1, w: 525 },
      { s: 3_200, w: 195 },
    ],
  })
  const card = buildDayCard(
    factory,
    activity.date,
    { details: { [activity.id]: activity }, health: {} },
    { embedded: true, expanded: true },
    undefined,
    ctx(),
  )
  assert.deepEqual(byClass(card, 'tri-curve-tick').map(text), [
    '1s',
    '5s',
    '30s',
    '1m',
    '2m',
    '3m',
    '5m',
    '6m',
    '12m',
    '53m20s',
  ])
})

test('keeps shared power curve duration markers inside the visible domain', () => {
  assert.deepEqual(
    powerCurveDurationTicks(5, 300, [1, 15, 60, 300, 600]),
    [5, 10, 15, 20, 30, 60, 120, 180, 300],
  )
  assert.deepEqual(
    powerCurveDurationTicks(1, 20_107, [1, 60, 300, 1_200, 3_600, 10_800]),
    [1, 5, 10, 20, 30, 60, 120, 180, 300, 360, 720, 1_200, 3_600, 20_107],
  )
})

test('keeps every hover value while bounding a dense power curve path', () => {
  const powerCurve = Array.from({ length: 10_800 }, (_, index) => ({
    s: index + 1,
    w: 700 - Math.floor(index / 20),
  }))
  const curve = buildPowerCurve(factory, detail({ powerCurve }), ctx())
  assert.ok(curve)
  const svg = byClass(curve, 'tri-curve-svg')[0]
  const path = byClass(curve, 'tri-curve-line')[0]
  assert.ok(svg)
  assert.ok(path)
  const encoded = String(svg.properties.dataPowerCurves)
  const decoded = decodedPowerCurves(svg)[0]
  assert.equal(decoded.length, powerCurve.length)
  for (const seconds of [61, 3_601, 7_200, 10_800]) {
    assert.deepEqual(decoded[seconds - 1], powerCurve[seconds - 1])
    assert.equal(
      powerCurveHoverAt(
        decoded,
        [],
        powerCurveFraction(seconds, decoded[0].s, decoded[decoded.length - 1].s),
      )?.durationS,
      seconds,
    )
  }
  assert.equal((String(path.properties.d).match(/[ML]/g) ?? []).length <= 1_024, true)
  assert.equal(encoded.length < JSON.stringify(powerCurve).length / 2, true)
})

test('serializes daily cards with long activity and reference power curves', async t => {
  const pointCount = 53_736
  const powerCurve = Array.from({ length: pointCount }, (_, index) => ({
    s: index + 1,
    w: index === pointCount - 1 ? 1_600 : 600,
  }))
  const reference = (watts: number) =>
    powerCurve.map(point => ({
      ...point,
      w: point.s === pointCount - 1 ? watts : 700,
      activityId: 202,
      activityDate: '2026-07-08',
    }))
  const sixWeeks = reference(2_000)
  const year = [...reference(2_600), { s: pointCount + 1, w: 20_000 }]
  for (const sport of ['bike', 'run'] satisfies StravaActivityDetail['sport'][]) {
    for (const embedded of [false, true]) {
      await t.test(`${sport}, embedded=${embedded}`, () => {
        const activity = detail({ sport, powerCurve })
        const card = triathlonDayCard(
          activity.date,
          { details: { [activity.id]: activity }, health: {} },
          { embedded },
          ctx({
            curveRef: sixWeeks,
            curveYearRef: year,
            runCurveRef: sixWeeks,
            runCurveYearRef: year,
          }),
        )
        const html = toHtml(card)
        const svg = byClass(card, 'tri-curve-svg')[0]
        assert.ok(svg)
        assert.match(html, /class="tri-curve-svg"/)
        assert.doesNotMatch(html, /NaN|Infinity/)
        assert.equal(svg.properties.ariaValueMax, pointCount)
        assert.ok(Number(svg.properties.dataCurveDomainMax) >= 2_600)
        assert.ok(Number(svg.properties.dataCurveDomainMax) < 20_000)
        const series = decodePowerCurves(String(svg.properties.dataPowerCurves))
        assert.equal(series.length, 3)
        assert.equal(series[0].durations, series[1].durations)
        assert.equal(series[0].durations, series[2].durations)
        for (const [seriesIndex, expected] of [
          [0, powerCurve],
          [1, sixWeeks],
          [2, year.slice(0, -1)],
        ] satisfies [number, typeof powerCurve][]) {
          const decoded = decodedPowerCurves(svg)[seriesIndex]
          assert.equal(decoded.length, pointCount)
          for (const index of [0, 3_600, pointCount - 1])
            assert.deepEqual(decoded[index], expected[index])
        }
        for (const path of [...byClass(card, 'tri-curve-line'), ...byClass(card, 'tri-curve-ref')])
          assert.ok((String(path.properties.d).match(/[ML]/g) ?? []).length <= 1_024)
      })
    }
  }
})

test('scales the power curve axis above the six-week peak and renders selected points', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveRef: [
        { s: 1, w: 1_060 },
        { s: 5, w: 1_020 },
        { s: 60, w: 400 },
        { s: 300, w: 300 },
        { s: 1_200, w: 240 },
        { s: 3_600, w: 210 },
      ],
    }),
  )
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-yt').map(text), [
    '0',
    '200w',
    '400w',
    '600w',
    '800w',
    '1,000w',
    '1,200w',
  ])
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.dataCurveDomainMax, 1_200)
  const ridePoint = byClass(curve, 'tri-curve-point--ride')[0]
  const referencePoint = byClass(curve, 'tri-curve-point--ref')[0]
  assert.ok(ridePoint)
  assert.ok(referencePoint)
  assert.equal(ridePoint.properties.ariaHidden, 'true')
  assert.equal(referencePoint.properties.ariaHidden, 'true')
})

test('renders six-week and calendar-year comparison ranges on one watt domain', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveRef: [
        { s: 1, w: 700, activityId: 102, activityDate: '2026-07-10' },
        { s: 60, w: 400, activityId: 102, activityDate: '2026-07-10' },
        { s: 3_600, w: 220, activityId: 103, activityDate: '2026-07-11' },
      ],
      curveYearRef: [
        { s: 1, w: 1_060 },
        { s: 60, w: 440 },
        { s: 3_600, w: 240 },
      ],
      curveYear: 2026,
    }),
  )
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-yt').map(text), [
    '0',
    '200w',
    '400w',
    '600w',
    '800w',
    '1,000w',
    '1,200w',
  ])
  const ranges = byClass(curve, 'tri-curve-range')
  assert.deepEqual(ranges.map(text), ['6 weeks', 'all of 2026'])
  const controls = byClass(curve, 'tri-curve-controls')[0]
  assert.deepEqual(controls.children, [
    byClass(curve, 'tri-curve-ranges')[0],
    byClass(curve, 'tri-curve-units')[0],
  ])
  assert.deepEqual(
    ranges.map(button => button.properties.ariaPressed),
    ['true', 'false'],
  )
  const paths = byClass(curve, 'tri-curve-ref')
  assert.equal(paths.length, 2)
  assert.equal('hidden' in paths[0].properties, false)
  assert.equal('hidden' in paths[1].properties, true)
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.dataCurveRange, 'six-weeks')
  assert.equal(svg.properties.dataCurveYear, 2026)
  assert.equal(decodedPowerCurves(svg)[1][0].w, 700)
  assert.equal(decodedPowerCurves(svg)[2][0].w, 1_060)
  const stage = byClass(curve, 'tri-cax-stage')[0]
  assert.ok(stage)
  assert.equal(byClass(stage, 'tri-curve-readout').length, 1)
  const referenceRow = byClass(stage, 'tri-curve-readout-row--ref')[0]
  assert.equal(referenceRow.tagName, 'a')
  assert.equal(referenceRow.properties.href, '/triathlon/on/2026/07/10#tri-activity-102')
  assert.equal(referenceRow.properties.dataPowerActivityId, '102')
})

test('renders only the ride critical power model and keeps FTP and goal in the efforts row', () => {
  const sixWeeks = criticalPower()
  const year = { ...criticalPower('calendar-year'), criticalPowerWatts: 252 }
  const ride = {
    ...criticalPower('activity'),
    criticalPowerWatts: 245,
    wPrimeJoules: 9_600,
    anchors: criticalPower('activity').anchors.map(anchor => ({
      ...anchor,
      activityId: 101,
      activityDate: '2026-07-09',
    })),
    independentEffortCount: 1,
  }
  const context = ctx({
    curveRef: [
      { s: 1, w: 700 },
      { s: 180, w: 307 },
      { s: 720, w: 264 },
    ],
    curveYearRef: [
      { s: 1, w: 720 },
      { s: 180, w: 310 },
      { s: 720, w: 267 },
    ],
    curveYear: 2026,
    criticalPower: sixWeeks,
    criticalPowerYear: year,
  })
  const bike = zonedDetail()
  bike.activityCriticalPower = ride
  bike.powerHist = Array.from({ length: 13 }, () => 1)
  const curve = buildPowerCurve(factory, bike, context)
  assert.ok(curve)
  assert.equal(byClass(curve, 'tri-curve-model').length, 1)
  assert.equal(byClass(curve, 'tri-curve-model--ride').length, 1)
  assert.equal(byClass(curve, 'tri-curve-model--ref').length, 0)
  assert.equal(byClass(curve, 'tri-curve-cp').length, 1)
  assert.equal(byClass(curve, 'tri-curve-cp--ride').length, 1)
  const modelRows = byClass(curve, 'tri-curve-readout-row--model')
  assert.equal(modelRows.length, 1)
  assert.deepEqual(byClass(curve, 'tri-curve-readout-label--model').map(text), [
    'this ride eCP model',
  ])
  assert.equal(modelRows[0].properties.dataCurveCriticalPower, '245')
  assert.equal(modelRows[0].properties.dataCurveWPrime, '9600')
  assert.equal(modelRows[0].properties.dataCurveModelMinSeconds, '180')
  assert.equal(modelRows[0].properties.dataCurveModelMaxSeconds, '720')
  const summaries = byClass(curve, 'tri-curve-cp-k')
  assert.equal(summaries.length, 1)
  assert.equal(text(summaries[0]), 'this ride · eCP 245 W · eW′ 9.6 kJ · 87.55 kg')
  const weightNote = byClass(curve, 'tri-curve-weight-note')[0]
  assert.ok(summaries[0].children.includes(weightNote))
  assert.equal(weightNote.properties.hidden, true)
  assert.equal(curve.children.includes(weightNote), false)
  assert.equal(summaries[0].properties.dataGlossDef, '1 independent effort · provisional')
  assert.equal(summaries[0].properties.tabIndex, 0)
  assert.equal('hidden' in summaries[0].properties, false)
  const anchors = byClass(curve, 'tri-critical-power-anchor')
  assert.equal(anchors.length, 0)
  const thresholds = byClass(curve, 'tri-curve-thresholds')
  assert.equal(thresholds.length, 1)
  assert.deepEqual((thresholds[0].children as Element[]).map(text), [
    'this ride · eCP 245 W · eW′ 9.6 kJ · 87.55 kg',
  ])
  const cap = byClass(curve, 'tri-elev-cap')[0]
  const capChildren = cap.children as Element[]
  assert.deepEqual(capChildren.slice(0, 6).map(text), [
    '5s 540W',
    '1m 320W',
    '5m 250W',
    '20m 230W',
    'FTP 260W',
    'goal 280W',
  ])
  assert.equal(capChildren[6], thresholds[0])

  const embeddedCurve = buildPowerCurve(factory, bike, context, true)
  assert.ok(embeddedCurve)
  assert.equal(byClass(embeddedCurve, 'tri-critical-power-anchor').length, 0)
  assert.equal(byClass(embeddedCurve, 'tri-critical-power-anchor-duration').length, 0)
  assert.equal(byClass(embeddedCurve, 'tri-curve-thresholds').length, 0)
  assert.equal(byClass(embeddedCurve, 'tri-curve-cp-k').length, 0)
  assert.equal(byClass(embeddedCurve, 'tri-curve-ftp-k').length, 0)
  assert.equal(byClass(embeddedCurve, 'tri-curve-goal-k').length, 0)

  const histogram = buildPowerHist(factory, bike)
  assert.ok(histogram)
  assert.equal(byClass(histogram, 'tri-hist-cp').length, 0)
  assert.equal(byClass(histogram, 'tri-hist-cp-k').length, 0)

  const activity = buildActivity(factory, bike, true, context)
  assert.equal(byClass(activity, 'tri-trace-reference').length, 1)
  assert.equal(text(byClass(activity, 'tri-trace-reference-k')[0]), 'eCP 245 W')
})

test('renders running power comparison ranges with run sources and keeps cycling thresholds off', () => {
  const run = detail({
    sport: 'run',
    deviceWatts: true,
    powerCurve: zonedDetail().powerCurve,
    powerHist: Array.from({ length: 13 }, () => 1),
  })
  const context = ctx({
    criticalPower: criticalPower(),
    curveRef: [{ s: 1, w: 1_200 }],
    curveYearRef: [{ s: 1, w: 1_400 }],
    runCurveRef: [
      { s: 1, w: 700, activityId: 201, activityDate: '2026-07-10' },
      { s: 3_600, w: 320, activityId: 201, activityDate: '2026-07-10' },
    ],
    runCurveYearRef: [
      { s: 1, w: 800, activityId: 202, activityDate: '2026-01-10' },
      { s: 3_600, w: 350, activityId: 202, activityDate: '2026-01-10' },
    ],
    curveYear: 2026,
  })
  const curve = buildPowerCurve(factory, run, context)
  assert.ok(curve)
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.equal(svg.properties.dataCurveSport, 'run')
  assert.equal(svg.properties.dataCurveDomainMax, 800)
  assert.deepEqual(decodedPowerCurves(svg)[1], context.runCurveRef)
  assert.deepEqual(decodedPowerCurves(svg)[2], context.runCurveYearRef)
  assert.deepEqual(byClass(curve, 'tri-curve-range').map(text), ['6 weeks', 'all of 2026'])
  assert.deepEqual(byClass(curve, 'tri-curve-readout-label').map(text), ['this run', '6-week best'])
  assert.equal(
    byClass(curve, 'tri-curve-readout-row--ref')[0].properties.href,
    '/triathlon/on/2026/07/10#tri-activity-201',
  )
  assert.equal(byClass(curve, 'tri-curve-model').length, 0)
  assert.equal(byClass(curve, 'tri-curve-cp').length, 0)
  assert.equal(byClass(curve, 'tri-curve-ref').length, 2)
  assert.equal(byClass(curve, 'tri-curve-ftp').length, 0)
  assert.equal(byClass(curve, 'tri-curve-goal').length, 0)
  assert.equal(byClass(curve, 'tri-curve-thresholds').length, 0)
  const histogram = buildPowerHist(factory, run)
  assert.ok(histogram)
  assert.equal(byClass(histogram, 'tri-hist-cp').length, 0)

  const embedded = buildPowerCurve(factory, run, context, true)
  assert.ok(embedded)
  assert.deepEqual(byClass(embedded, 'tri-curve-range').map(text), ['6 weeks', 'all of 2026'])
  const yearOnly = buildPowerCurve(factory, run, { ...context, runCurveRef: [] })
  assert.ok(yearOnly)
  assert.equal(byClass(yearOnly, 'tri-curve-range')[0].properties.disabled, true)
  assert.equal(byClass(yearOnly, 'tri-curve-svg')[0].properties.dataCurveRange, 'year')
  assert.equal(
    byClass(yearOnly, 'tri-curve-readout-row--ref')[0].properties.href,
    '/triathlon/on/2026/01/10#tri-activity-202',
  )
  const unavailable = buildPowerCurve(factory, run, {
    ...context,
    runCurveRef: [],
    runCurveYearRef: [],
  })
  assert.ok(unavailable)
  assert.equal(byClass(unavailable, 'tri-curve-ref').length, 0)
  assert.equal(byClass(unavailable, 'tri-curve-range').length, 0)
})

test('suppresses a power reference link back to its enclosing activity', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveRef: [
        { s: 1, w: 700, activityId: 101, activityDate: '2026-07-09' },
        { s: 60, w: 400, activityId: 101, activityDate: '2026-07-09' },
      ],
    }),
  )
  assert.ok(curve)
  const referenceRow = byClass(curve, 'tri-curve-readout-row--ref')[0]
  assert.equal(referenceRow.properties.ariaDisabled, 'true')
  assert.equal('href' in referenceRow.properties, false)
  assert.equal(byClass(curve, 'tri-curve-readout-row')[0].tagName, 'span')
})

test('uses the calendar year when no six-week power reference exists', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveYearRef: [
        { s: 1, w: 900 },
        { s: 60, w: 380 },
        { s: 3_600, w: 215 },
      ],
      curveYear: 2026,
    }),
  )
  assert.ok(curve)
  const ranges = byClass(curve, 'tri-curve-range')
  assert.equal(ranges[0].properties.disabled, true)
  assert.deepEqual(
    ranges.map(button => button.properties.ariaPressed),
    ['false', 'true'],
  )
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.dataCurveRange, 'year')
  assert.deepEqual(byClass(curve, 'tri-curve-readout-label').map(text), ['this ride', '2026 best'])
})

test('renders a keyboard-focusable comparison readout for the power curve', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveRef: [
        { s: 1, w: 700 },
        { s: 5, w: 650 },
        { s: 60, w: 400 },
        { s: 300, w: 300 },
        { s: 1_200, w: 260 },
        { s: 3_600, w: 220 },
      ],
    }),
  )
  assert.ok(curve)
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.role, 'slider')
  assert.equal(svg.properties.tabIndex, 0)
  assert.equal(svg.properties.ariaReadonly, undefined)
  assert.equal(svg.properties.ariaValueMin, 1)
  assert.equal(svg.properties.ariaValueMax, 3_600)
  const readout = byClass(curve, 'tri-curve-readout')[0]
  assert.ok(readout)
  assert.deepEqual(byClass(readout, 'tri-curve-readout-label').map(text), [
    'this ride',
    '6-week best',
  ])
})

test('zoneDuo unwraps when one side is missing', () => {
  const solo = factory.el('div', 'tri-zone')
  assert.equal(zoneDuo(factory, solo, null), solo)
  assert.equal(zoneDuo(factory, null, solo), solo)
  assert.equal(zoneDuo(factory, null, null), null)
})

test('renders metric elevation ticks, distance ticks, and dotted grid nodes', () => {
  assert.equal(formatAltitude(METRIC_TRIATHLON_PRESENTATION, -0.1), '0 m')
  const elevation = buildElevation(factory, detail())
  assert.deepEqual(byClass(elevation, 'tri-cax-yt').map(text).filter(Boolean), [
    '80 m',
    '90 m',
    '100 m',
    '110 m',
  ])
  assert.deepEqual(byClass(elevation, 'tri-cax-xt').map(text), ['10 km', '20 km'])
  assert.equal(byClass(elevation, 'tri-elev-grid').length, 4)
})

const comparisonActivity = (
  id: number,
  overrides: Partial<StravaActivityDetail> = {},
): StravaActivityDetail => {
  const activity = detail({
    id,
    name: `Activity ${id}`,
    date: `2026-07-${String((id % 20) + 1).padStart(2, '0')}`,
    ...overrides,
  })
  if (activity.mapRoute.length > 0) return activity
  return {
    ...activity,
    mapRoute: [activity.route.map(point => ({ lat: point.lat, lng: point.lng, d: point.d }))],
  }
}

const comparisonChart = (root: Element, kind: string): Element => {
  const chart = byClass(root, 'tri-compare-chart').find(
    element => element.properties.dataCompareChart === kind,
  )
  assert.ok(chart)
  return chart
}

test('counts route coverage for the comparison map from gapped map routes', () => {
  const first = comparisonActivity(201, {
    route: detail().route.map((point, index) => ({
      ...point,
      lat: 43.6 + index * 0.01,
      lng: -79.4 + index * 0.01,
    })),
    mapRoute: [
      [
        { lat: 43.6, lng: -79.4, d: 0 },
        { lat: 43.61, lng: -79.39, d: 10 },
      ],
      [
        { lat: 43.62, lng: -79.38, d: 20 },
        { lat: 43.63, lng: -79.37, d: 30 },
      ],
    ],
  })
  const second = comparisonActivity(202, {
    route: detail().route.map((point, index) => ({
      ...point,
      lat: 43.8 + index / 300,
      lng: -79.2 + index / 300,
    })),
    mapRoute: [],
  })
  const rendered = buildActivityComparison(factory, [first, second])
  const map = byClass(rendered, 'tri-compare-map')[0]

  assert.ok(map)
  assert.equal(map.properties.dataAvailable, '2')
  assert.equal(map.properties.dataDomainXMax, '30')
  assert.equal(activityComparisonEligible(first), true)
  assert.equal(activityComparisonEligible(second), true)

  const routeless = { ...second, id: 203, route: [], mapRoute: [] }
  assert.equal(activityComparisonEligible(routeless), false)
})

test('formats only the active comparison metric and clamps keyboard navigation', () => {
  const activity = comparisonActivity(210)
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'elevation', 5), '82 m · +0.1%')
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'speed', 5), '23.0 km/h')
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'hr', 5), '138 bpm')
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'power', 5), '180 W')
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'cadence', 5), '84 rpm')
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'respiration', 5), '22.0 brpm')
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'temperature', 5), '23°C')
  assert.equal(
    activityComparisonDisplayValueAtDistance(
      comparisonActivity(209, { route: detail().route.map(point => ({ ...point, hr: 0 })) }),
      'hr',
      5,
    ),
    '—',
  )
  assert.equal(
    activityComparisonDisplayValueAtDistance(
      comparisonActivity(208, { route: detail().route.map(point => ({ ...point, speedKph: 0 })) }),
      'speed',
      5,
    ),
    '—',
  )

  assert.equal(activityComparisonFractionForKey('ArrowRight', 0.4, 0.1), 0.5)
  assert.equal(activityComparisonFractionForKey('ArrowUp', 0.4, 0.1), 0.5)
  assert.equal(activityComparisonFractionForKey('ArrowLeft', 0, 0.1), 0)
  assert.equal(activityComparisonFractionForKey('ArrowDown', 0, 0.1), 0)
  assert.equal(activityComparisonFractionForKey('Home', 0.7, 0.1), 0)
  assert.equal(activityComparisonFractionForKey('End', 0.2, 0.1), 1)
  assert.equal(activityComparisonFractionForKey('Enter', 0.2, 0.1), null)
})

test('uses normalized bike power and cadence values in comparison readouts', () => {
  const activity = comparisonActivity(210, {
    route: detail().route.map((point, index) => ({
      ...point,
      w: [100, 0, 300, 400][index],
      cad: [80, 0, 100, 110][index],
    })),
  })

  assert.equal(activityComparisonMetricAtDistance(activity, 'power', 10), 0)
  assert.equal(activityComparisonMetricAtDistance(activity, 'cadence', 10), null)

  assert.equal(
    activityComparisonMetricAtDistance(activity, 'power', 10, excludeZeroPresentation),
    200,
  )
  assert.equal(
    activityComparisonMetricAtDistance(activity, 'cadence', 10, excludeZeroPresentation),
    90,
  )
})

test('renders every comparison graph with stable selectors, cursors, and readout rows', () => {
  const first = comparisonActivity(211, {
    route: detail().route.map((point, index) => ({
      ...point,
      skinTemperatureC: 33.4 + index * 0.05,
    })),
    gearShifts: shiftedDetail().gearShifts,
    powerCurve: [
      { s: 1, w: 700 },
      { s: 60, w: 400 },
      { s: 3_600, w: 220 },
    ],
    powerHist: [10, 30, 50, 10],
    hrZones: [100, 200, 300, 200, 100],
    powerZones: [50, 100, 200, 300, 200, 100, 50],
  })
  const second = comparisonActivity(212, {
    route: detail().route.map((point, index) => ({
      ...point,
      skinTemperatureC: 33.5 + index * 0.05,
    })),
    gearShifts: shiftedDetail().gearShifts.map(shift => ({
      ...shift,
      rearTeeth: shift.rearTeeth === 19 ? 17 : shift.rearTeeth,
    })),
    powerCurve: [
      { s: 1, w: 680 },
      { s: 60, w: 390 },
      { s: 3_600, w: 210 },
    ],
    powerHist: [20, 40, 30, 10],
    hrZones: [90, 180, 270, 180, 90],
    powerZones: [40, 80, 160, 240, 160, 80, 40],
  })
  const rendered = buildActivityComparison(factory, [first, second])

  assert.equal(byClass(rendered, 'tri-compare').length, 1)
  assert.equal(rendered.properties.dataCompareState, 'ready')
  const legend = byClass(rendered, 'tri-compare-legend')[0]
  assert.ok(legend)
  assert.equal(legend.properties.role, 'list')
  assert.equal(legend.properties.ariaLabel, 'selected activities')
  assert.equal(legend.properties.dataI18nAriaLabel, 'selected activities')
  assert.deepEqual(
    byClass(rendered, 'tri-compare-legend-item').map(item => [
      item.properties.dataActivityId,
      item.properties.dataActivityIndex,
      item.properties.style,
    ]),
    [
      ['211', '0', `--tri-compare-color:${activityCompareColor(0)}`],
      ['212', '1', `--tri-compare-color:${activityCompareColor(1)}`],
    ],
  )
  const removeButtons = byClass(rendered, 'tri-compare-legend-remove')
  assert.deepEqual(
    removeButtons.map(button => [
      button.properties.dataCompareActivityRemove,
      button.properties.type,
      button.properties.ariaLabel,
      button.properties.dataI18nAriaLabel,
      button.properties.disabled,
    ]),
    [
      ['211', 'button', 'remove activity', 'remove activity', true],
      ['212', 'button', 'remove activity', 'remove activity', true],
    ],
  )
  for (const button of removeButtons)
    assert.equal(byClass(button, 'tri-compare-legend-remove-icon').length, 1)
  const chartViewport = byClass(rendered, 'tri-compare-charts-viewport')[0]
  const chartContainer = byClass(rendered, 'tri-compare-charts')[0]
  assert.ok(chartViewport)
  assert.ok(chartContainer)
  assert.equal(chartViewport.children.includes(chartContainer), true)
  assert.deepEqual(
    byClass(rendered, 'tri-compare-chart').map(chart => chart.properties.dataCompareChart),
    [
      'elevation',
      'speed',
      'hr',
      'power',
      'cadence',
      'respiration',
      'temperature',
      'skin-temperature',
      'gear-ratio-distribution',
      'power-distribution',
      'power-curve',
      'hr-zones',
      'power-zones',
    ],
  )
  for (const chart of byClass(rendered, 'tri-compare-chart')) {
    const graph = byClass(chart, 'tri-compare-graph')[0]
    assert.ok(graph)
    assert.equal(graph.properties.role, 'slider')
    assert.equal(graph.properties.tabIndex, 0)
    assert.equal(graph.properties.ariaOrientation, 'horizontal')
    assert.equal(graph.properties.ariaLabel, graph.properties.dataI18nAriaLabel)
    assert.equal(byClass(chart, 'tri-compare-cursor').length, 1)
    const selection = byClass(chart, 'tri-compare-selection-region')
    const selectionClip = byClass(chart, 'tri-compare-selection-clip')
    const selectionLines = byClass(chart, 'tri-compare-selection-line')
    const distanceChart = [
      'elevation',
      'speed',
      'hr',
      'power',
      'cadence',
      'respiration',
      'temperature',
      'skin-temperature',
    ].includes(String(chart.properties.dataCompareChart))
    assert.equal(selection.length, distanceChart ? 1 : 0)
    assert.equal(selectionClip.length, distanceChart ? 1 : 0)
    assert.equal(selectionLines.length, distanceChart ? 2 : 0)
    if (distanceChart) {
      assert.equal(selection[0].properties.x, 0)
      assert.equal(selection[0].properties.width, 0)
      assert.equal(selection[0].properties.ariaHidden, 'true')
      assert.equal(selectionClip[0].properties.x, 0)
      assert.equal(selectionClip[0].properties.width, 0)
    }
    assert.equal(
      byClass(chart, 'tri-compare-readout').length,
      chart.properties.dataCompareChart === 'elevation' ? 1 : 0,
    )
    assert.equal(byClass(chart, 'tri-compare-coverage').length, 0)
  }
  assert.equal(byClass(rendered, 'tri-compare-zone-band').length, 0)
  const maps = byClass(rendered, 'tri-compare-map')
  const mapPanels = byClass(rendered, 'tri-compare-map-panel')
  const mapStages = byClass(rendered, 'tri-compare-map-stage')
  const readouts = byClass(rendered, 'tri-compare-readout')
  assert.equal(maps.length, 1)
  assert.equal(mapPanels.length, 1)
  assert.equal(mapStages.length, 1)
  assert.equal(readouts.length, 1)
  const map = maps[0]
  const mapPanel = mapPanels[0]
  const mapStage = mapStages[0]
  const readout = readouts[0]
  assert.ok(map)
  assert.ok(mapPanel)
  assert.ok(mapStage)
  assert.ok(readout)
  assert.equal(mapPanel.properties.ariaLabel, 'route overlay')
  assert.equal(mapPanel.properties.dataI18nAriaLabel, 'route overlay')
  assert.equal(map.properties.dataCompareMap, '')
  assert.equal(map.properties.dataAvailable, '2')
  assert.equal(mapPanel.children.includes(mapStage), true)
  assert.equal(mapStage.children.includes(map), true)
  assert.equal(mapStage.children.includes(readout), false)
  const elevationHead = byClass(comparisonChart(rendered, 'elevation'), 'tri-compare-chart-head')[0]
  assert.ok(elevationHead)
  assert.equal(elevationHead.children.includes(readout), true)
  assert.equal(byClass(rendered, 'tri-compare-coverage').length, 0)
  assert.equal(byClass(rendered, 'tri-compare-readout').length, 1)
  assert.equal(readout.properties.role, undefined)
  assert.equal(readout.properties.ariaLabel, undefined)
  assert.equal(readout.properties.dataI18nAriaLabel, undefined)
  assert.equal(readout.properties.dataCompareReadout, '')
  assert.equal(readout.properties.dataVisible, 'false')
  assert.equal(readout.properties.ariaHidden, 'true')
  const context = byClass(readout, 'tri-compare-readout-context')
  assert.equal(context.length, 1)
  assert.equal(context[0].properties.dataCompareReadoutContext, '')
  assert.equal(context[0].properties.hidden, true)
  assert.equal(text(context[0]), '')
  assert.equal(byClass(readout, 'tri-compare-readout-position').length, 0)
  assert.equal(byClass(readout, 'tri-compare-readout-label').length, 0)
  const rows = byClass(readout, 'tri-compare-readout-row')
  assert.deepEqual(
    rows.map(row => [row.properties.dataActivityId, row.properties.dataActivityIndex]),
    [
      ['211', '0'],
      ['212', '1'],
    ],
  )
  for (const [index, row] of rows.entries()) {
    assert.equal(row.children.length, 2)
    assert.equal(row.properties.style, `--tri-compare-color:${activityCompareColor(index)}`)
    const swatch = row.children[0]
    const value = row.children[1]
    assert.ok(swatch?.type === 'element')
    assert.ok(value?.type === 'element')
    assert.deepEqual(classNames(swatch), ['tri-compare-readout-swatch'])
    assert.equal(swatch.properties.ariaHidden, 'true')
    assert.deepEqual(classNames(value), ['tri-compare-readout-value'])
    assert.equal(value.properties.dataCompareReadoutValue, '')
    assert.equal(text(row), '')
  }
  for (const line of [
    ...byClass(rendered, 'tri-compare-line'),
    ...byClass(rendered, 'tri-compare-selection-line'),
  ]) {
    assert.notEqual(line.properties.dataActivityIndex, undefined)
    assert.equal(line.properties.strokeDasharray, undefined)
  }
  assert.equal(byClass(rendered, 'tri-hist-svg').length, 0)

  const removable = buildActivityComparison(factory, [first, second, comparisonActivity(213)])
  assert.deepEqual(
    byClass(removable, 'tri-compare-legend-remove').map(button => button.properties.disabled),
    [undefined, undefined, undefined],
  )
  const embedded = buildActivityComparison(
    factory,
    [first, second, comparisonActivity(213)],
    undefined,
    { removable: false },
  )
  assert.equal(byClass(embedded, 'tri-compare-legend-remove').length, 0)
  assert.equal(byClass(embedded, 'tri-compare-legend--static').length, 1)
})

test('uses running dynamics instead of respiration for run comparisons', () => {
  const runRoute = detail().route.map((point, index) => ({
    ...point,
    speedKph: 11 + index,
    cad: 82 + index,
    strideLengthM: 1.08 + index * 0.04,
    groundContactTimeMs: 252 - index * 4,
    verticalOscillationCm: 9.2 + index * 0.15,
  }))
  const first = comparisonActivity(213, { sport: 'run', route: runRoute, mapRoute: [] })
  const second = comparisonActivity(214, {
    sport: 'run',
    route: runRoute.map(point => ({
      ...point,
      strideLengthM: (point.strideLengthM ?? 0) + 0.05,
      groundContactTimeMs: (point.groundContactTimeMs ?? 0) - 6,
      verticalOscillationCm: (point.verticalOscillationCm ?? 0) - 0.25,
    })),
    mapRoute: [],
  })
  const rendered = buildActivityComparison(factory, [first, second])

  assert.deepEqual(activityComparisonMetricsForSport('run'), [
    'elevation',
    'speed',
    'hr',
    'power',
    'cadence',
    'stride-length',
    'ground-contact-time',
    'vertical-oscillation',
    'temperature',
  ])
  assert.deepEqual(
    byClass(rendered, 'tri-compare-chart').map(chart => chart.properties.dataCompareChart),
    [
      'elevation',
      'speed',
      'hr',
      'power',
      'cadence',
      'stride-length',
      'ground-contact-time',
      'vertical-oscillation',
      'temperature',
      'power-distribution',
      'power-curve',
      'hr-zones',
      'power-zones',
    ],
  )
  assert.equal(byClass(comparisonChart(rendered, 'stride-length'), 'tri-compare-line').length, 2)
  assert.equal(
    byClass(comparisonChart(rendered, 'ground-contact-time'), 'tri-compare-line').length,
    2,
  )
  assert.equal(
    byClass(comparisonChart(rendered, 'vertical-oscillation'), 'tri-compare-line').length,
    2,
  )
  assert.equal(activityComparisonDisplayValueAtDistance(first, 'stride-length', 10), '1.12 m')
  assert.equal(activityComparisonDisplayValueAtDistance(first, 'ground-contact-time', 10), '248 ms')
  assert.equal(
    activityComparisonDisplayValueAtDistance(first, 'vertical-oscillation', 10),
    '9.3 cm',
  )
})

test('compares route-less pool swims on interval distance, pace, and stroke rate', () => {
  const first = swimTrendDetail({ id: 215, name: 'Pool 1', hrZones: [20, 40, 30, 10, 0] })
  const second = swimTrendDetail({
    id: 216,
    name: 'Pool 2',
    date: '2026-07-06',
    swimIntervals: swimTrendDetail().swimIntervals.map(interval => ({
      ...interval,
      paceSPer100m: interval.paceSPer100m == null ? null : interval.paceSPer100m + 4,
      strokeRateSpm: interval.strokeRateSpm == null ? null : interval.strokeRateSpm + 2,
    })),
    hrZones: [10, 30, 40, 20, 0],
  })
  const rendered = buildActivityComparison(factory, [first, second])

  assert.equal(activityComparisonEligible(first), true)
  assert.deepEqual(activityComparisonMetricsForSport('swim'), ['swim-pace', 'stroke-rate'])
  assert.equal(rendered.properties.dataCompareState, 'ready')
  assert.deepEqual(
    byClass(rendered, 'tri-compare-chart').map(chart => chart.properties.dataCompareChart),
    ['swim-pace', 'stroke-rate', 'hr-zones'],
  )
  assert.equal(byClass(rendered, 'tri-compare-map').length, 0)
  const head = byClass(comparisonChart(rendered, 'swim-pace'), 'tri-compare-chart-head')[0]
  assert.equal(byClass(head, 'tri-compare-readout-row').length, 2)
  for (const kind of ['swim-pace', 'stroke-rate']) {
    const chart = comparisonChart(rendered, kind)
    assert.equal(chart.properties.dataAvailable, '2')
    assert.equal(byClass(chart, 'tri-compare-line').length, 2)
    assert.deepEqual(byClass(chart, 'tri-cax-xt').map(text), ['0 m', '50 m', '100 m'])
  }
  assert.equal(activityComparisonDisplayValueAtDistance(first, 'swim-pace', 0.05), '1:44 /100m')
  assert.equal(activityComparisonDisplayValueAtDistance(first, 'stroke-rate', 0.05), '26 str/min')
  assert.equal(activityComparisonMetricAtDistance(first, 'stroke-rate', 0.0625), null)
})

test('uses one absolute-distance and y domain for every activity line', () => {
  const first = comparisonActivity(221)
  const secondRoute = detail().route.map((point, index) => ({
    ...point,
    d: index * 5,
    alt: 180 + index * 10,
    hr: 110 + index * 5,
  }))
  const second = comparisonActivity(222, {
    distanceKm: 15,
    route: secondRoute,
    minAlt: 180,
    maxAlt: 210,
  })
  const rendered = buildActivityComparison(factory, [first, second])
  const elevation = comparisonChart(rendered, 'elevation')
  const graph = byClass(elevation, 'tri-compare-graph')[0]
  const lines = byClass(elevation, 'tri-compare-line')

  assert.ok(graph)
  assert.equal(graph.properties.dataDomainXMin, 0)
  assert.equal(graph.properties.dataDomainXMax, 30)
  assert.ok(Number(graph.properties.dataDomainYMin) <= 75)
  assert.ok(Number(graph.properties.dataDomainYMax) >= 210)
  assert.deepEqual(
    lines.map(line => String(line.properties.dataActivityId)),
    ['221', '222'],
  )
  assert.match(String(lines[1].properties.d), /L 50\.00 /)
  for (const distanceGraph of byClass(rendered, 'tri-compare-distance-graph')) {
    assert.equal(distanceGraph.properties.dataDomainXMin, 0)
    assert.equal(distanceGraph.properties.dataDomainXMax, 30)
  }
  assert.equal(byClass(rendered, 'tri-compare-map')[0].properties.dataDomainXMax, '30')
})

test('bounds dense comparison power curves on one logarithmic duration domain', () => {
  const dense = Array.from({ length: 10_800 }, (_, index) => ({
    s: index + 1,
    w: Math.round(900 - Math.log(index + 1) * 65),
  }))
  const first = comparisonActivity(231, { powerCurve: dense })
  const second = comparisonActivity(232, {
    powerCurve: dense.map(point => ({ s: point.s, w: point.w - 20 })),
  })
  const rendered = buildActivityComparison(factory, [first, second])
  const curve = comparisonChart(rendered, 'power-curve')
  const graph = byClass(curve, 'tri-compare-curve-graph')[0]
  const paths = byClass(curve, 'tri-compare-line')

  assert.ok(graph)
  assert.equal(graph.properties.dataDomainXScale, 'log')
  assert.equal(graph.properties.dataDomainXMin, 1)
  assert.equal(graph.properties.dataDomainXMax, 10_800)
  assert.equal(graph.properties.dataCurve, undefined)
  assert.equal(paths.length, 2)
  for (const path of paths) {
    const commands = String(path.properties.d).match(/[ML]/g) ?? []
    assert.ok(commands.length <= 1_024)
    assert.ok(commands.length > 500)
    const coordinates =
      String(path.properties.d)
        .match(/-?\d+(?:\.\d+)?/g)
        ?.map(Number) ?? []
    for (let index = 0; index < coordinates.length; index += 2) {
      assert.ok(coordinates[index] >= 0 && coordinates[index] <= 100)
      assert.ok(coordinates[index + 1] >= 0 && coordinates[index + 1] <= 34)
    }
  }
  const normalized = normalizePowerCurvePoints(dense)
  assert.equal(nearestPowerCurveValue(normalized, 60), dense[59].w)
  assert.equal(nearestPowerCurveValue(normalized, 10_801), null)
  assert.deepEqual(
    normalizePowerCurvePoints([
      { s: 60, w: 400 },
      { s: 5, w: 600 },
      { s: 60, w: 390 },
      { s: Number.NaN, w: 300 },
    ]),
    [
      { s: 5, w: 600 },
      { s: 60, w: 400 },
    ],
  )
})

test('overlays the six-week best and threshold lines on the comparison power curve', () => {
  const curve = [
    { s: 1, w: 700 },
    { s: 5, w: 620 },
    { s: 60, w: 380 },
    { s: 300, w: 300 },
    { s: 1_200, w: 260 },
  ]
  const first = comparisonActivity(241, { powerCurve: curve })
  const second = comparisonActivity(242, {
    powerCurve: curve.map(point => ({ s: point.s, w: point.w - 40 })),
  })
  const reference = ctx({
    curveRef: [
      { s: 1, w: 1_100 },
      { s: 5, w: 900 },
      { s: 60, w: 440 },
      { s: 300, w: 330 },
      { s: 1_200, w: 280 },
      { s: 3_600, w: 250 },
    ],
    ftp: 260,
    goalFtp: 290,
  })
  const bare = comparisonChart(buildActivityComparison(factory, [first, second]), 'power-curve')
  const chart = comparisonChart(
    buildActivityComparison(factory, [first, second], reference),
    'power-curve',
  )
  const graph = byClass(chart, 'tri-compare-curve-graph')[0]
  const refPath = byClass(chart, 'tri-compare-curve-ref')[0]

  assert.equal(byClass(bare, 'tri-compare-curve-ref').length, 0)
  assert.equal(byClass(bare, 'tri-compare-curve-ftp').length, 0)
  assert.equal(byClass(bare, 'tri-compare-curve-goal').length, 0)
  assert.equal(byClass(bare, 'tri-elev-cap').length, 0)
  assert.ok(Number(byClass(bare, 'tri-compare-curve-graph')[0].properties.dataDomainYMax) < 1_100)

  assert.ok(refPath)
  assert.ok(Number(graph.properties.dataDomainYMax) >= 1_100)
  assert.deepEqual(
    decodedPowerCurves(graph)[0],
    [
      { s: 1, w: 1_100 },
      { s: 5, w: 900 },
      { s: 60, w: 440 },
      { s: 300, w: 330 },
      { s: 1_200, w: 280 },
    ],
    'reference is clipped to the compared activities’ duration domain',
  )

  const yFor = (watts: number): number => {
    const min = Number(graph.properties.dataDomainYMin)
    const max = Number(graph.properties.dataDomainYMax)
    return 34 - ((watts - min) / (max - min)) * 33
  }
  const ftpLine = byClass(chart, 'tri-compare-curve-ftp')[0]
  const goalLine = byClass(chart, 'tri-compare-curve-goal')[0]
  assert.equal(Number(ftpLine.properties.y1), Number(yFor(260).toFixed(2)))
  assert.equal(Number(ftpLine.properties.y2), Number(yFor(260).toFixed(2)))
  assert.equal(Number(ftpLine.properties.x2), 100)
  assert.equal(Number(goalLine.properties.y1), Number(yFor(290).toFixed(2)))
  assert.ok(Number(goalLine.properties.y1) < Number(ftpLine.properties.y1))

  const cap = byClass(chart, 'tri-elev-cap')[0]
  assert.ok(cap)
  assert.ok((chart.children as Element[]).includes(cap))
  assert.deepEqual(byClass(cap, 'tri-compare-curve-reference-label').map(text), ['6-week best'])
  const thresholds = byClass(cap, 'tri-curve-thresholds')[0]
  assert.deepEqual((thresholds.children as Element[]).map(text), ['FTP 260W', 'goal 290W'])
  assert.ok(
    (cap.children as Element[]).indexOf(byClass(cap, 'tri-compare-curve-reference-label')[0]) <
      (cap.children as Element[]).indexOf(thresholds),
  )

  const marks = graph.children as Element[]
  const refIndex = marks.indexOf(refPath)
  const lineIndex = marks.findIndex(mark =>
    String(mark.properties?.className ?? '').includes('tri-compare-line'),
  )
  assert.ok(refIndex >= 0 && lineIndex >= 0)
  assert.ok(refIndex < lineIndex, 'reference sits under the compared activities')
})

test('adds critical power references to bike comparison charts', () => {
  const curve = Array.from({ length: 900 }, (_, index) => ({
    s: index + 1,
    w: Math.round(700 - Math.log(index + 1) * 60),
  }))
  const powerHist = Array.from({ length: 13 }, () => 1)
  const first = comparisonActivity(243, { powerCurve: curve, powerHist })
  const second = comparisonActivity(244, {
    powerCurve: curve.map(point => ({ s: point.s, w: point.w - 20 })),
    powerHist,
  })
  const rendered = buildActivityComparison(
    factory,
    [first, second],
    ctx({
      criticalPower: criticalPower(),
      criticalPowerYear: criticalPower('calendar-year'),
      curveRef: curve,
      curveYearRef: curve,
      curveYear: 2026,
    }),
  )
  const powerCurve = comparisonChart(rendered, 'power-curve')
  assert.equal(byClass(powerCurve, 'tri-compare-curve-model').length, 2)
  assert.equal(byClass(powerCurve, 'tri-compare-curve-cp').length, 2)
  const summaries = byClass(powerCurve, 'tri-curve-cp-k')
  assert.equal(summaries.length, 2)
  assert.equal(text(summaries[0]), 'eCP 249 W · eW′ 10.3 kJ')
  assert.equal(summaries[0].properties.dataGlossDef, '2 independent efforts · provisional')
  assert.equal(byClass(powerCurve, 'tri-critical-power-anchor').length, 6)
  assert.deepEqual(
    (byClass(powerCurve, 'tri-curve-thresholds')[0].children as Element[]).map(text),
    ['eCP 249 W · eW′ 10.3 kJ', 'eCP 249 W · eW′ 10.3 kJ', 'FTP 260W', 'goal 280W'],
  )
  const distribution = comparisonChart(rendered, 'power-distribution')
  assert.equal(byClass(distribution, 'tri-compare-distribution-cp').length, 0)
  assert.equal(byClass(distribution, 'tri-hist-cp-k').length, 0)
})

test('keeps cycling power references off run comparison charts', () => {
  const curve = [
    { s: 1, w: 500 },
    { s: 180, w: 280 },
    { s: 420, w: 250 },
    { s: 720, w: 240 },
  ]
  const first = comparisonActivity(251, { sport: 'run', powerCurve: curve })
  const second = comparisonActivity(252, {
    sport: 'run',
    powerCurve: curve.map(point => ({ ...point, w: point.w - 20 })),
  })
  const bare = comparisonChart(buildActivityComparison(factory, [first, second]), 'power-curve')
  const rendered = comparisonChart(
    buildActivityComparison(
      factory,
      [first, second],
      ctx({
        criticalPower: criticalPower(),
        criticalPowerYear: criticalPower('calendar-year'),
        curveRef: curve,
        curveYearRef: curve,
      }),
    ),
    'power-curve',
  )

  assert.equal(byClass(rendered, 'tri-compare-curve-ref').length, 0)
  assert.equal(byClass(rendered, 'tri-compare-curve-model').length, 0)
  assert.equal(byClass(rendered, 'tri-compare-curve-cp').length, 0)
  assert.equal(byClass(rendered, 'tri-compare-curve-ftp').length, 0)
  assert.equal(byClass(rendered, 'tri-compare-curve-goal').length, 0)
  assert.equal(byClass(rendered, 'tri-curve-thresholds').length, 0)
  assert.equal(
    byClass(rendered, 'tri-compare-curve-graph')[0].properties.dataDomainYMax,
    byClass(bare, 'tri-compare-curve-graph')[0].properties.dataDomainYMax,
  )
})

test('gives comparison power curves the shared ranges and clickable duration segments', () => {
  const curve = [
    { s: 1, w: 700 },
    { s: 5, w: 620 },
    { s: 60, w: 380 },
    { s: 300, w: 300 },
    { s: 1_200, w: 260 },
  ]
  const first = comparisonActivity(245, { powerCurve: curve })
  const second = comparisonActivity(246, {
    powerCurve: curve.map(point => ({ s: point.s, w: point.w - 40 })),
  })
  const chart = comparisonChart(
    buildActivityComparison(
      factory,
      [first, second],
      ctx({
        curveRef: curve.map(point => ({ s: point.s, w: point.w + 40 })),
        curveYearRef: curve.map(point => ({ s: point.s, w: point.w + 80 })),
        curveYear: 2026,
      }),
    ),
    'power-curve',
  )
  const graph = byClass(chart, 'tri-compare-curve-graph')[0]
  const ranges = byClass(chart, 'tri-curve-range')
  const references = byClass(chart, 'tri-compare-curve-ref')
  const ticks = byClass(chart, 'tri-curve-tick')

  assert.deepEqual(ranges.map(text), ['6 weeks', 'all of 2026'])
  assert.deepEqual(
    ranges.map(button => button.properties.ariaPressed),
    ['true', 'false'],
  )
  assert.equal(graph.properties.dataCurveRange, 'six-weeks')
  assert.equal(graph.properties.dataCurveYear, 2026)
  assert.equal(decodedPowerCurves(graph)[0][0].w, 740)
  assert.equal(decodedPowerCurves(graph)[1][0].w, 780)
  assert.equal(references.length, 2)
  assert.equal('hidden' in references[0].properties, false)
  assert.equal('hidden' in references[1].properties, true)
  assert.deepEqual(ticks.map(text), [
    '1s',
    '5s',
    '10s',
    '20s',
    '30s',
    '1m',
    '2m',
    '3m',
    '5m',
    '6m',
    '20m',
  ])
  assert.equal(
    ticks.every(tick => tick.tagName === 'button'),
    true,
  )
  assert.deepEqual(
    ticks.map(tick => tick.properties.dataCurveSeconds),
    ['1', '5', '10', '20', '30', '60', '120', '180', '300', '360', '1200'],
  )
  assert.equal(ticks[0].properties.ariaPressed, 'true')
  assert.deepEqual(byClass(chart, 'tri-compare-curve-reference-label').map(text), ['6-week best'])
})

test('normalizes and overlays 25W power distributions on one percentage domain', () => {
  const first = comparisonActivity(239, { powerHist: [10, 30, 50, 10] })
  const second = comparisonActivity(240, { powerHist: [20, 40, 30, 10, 0] })
  const rendered = buildActivityComparison(factory, [first, second])
  const chart = comparisonChart(rendered, 'power-distribution')
  const graph = byClass(chart, 'tri-compare-distribution-graph')[0]
  const paths = byClass(chart, 'tri-compare-line')

  assert.deepEqual(activityPowerDistributionPercentages([10, 30, 50, 10]), [10, 30, 50, 10])
  assert.deepEqual(activityPowerDistributionPercentages([1]), [])
  assert.deepEqual(activityPowerDistributionPercentages([10, Number.NaN, 20]), [])
  assert.ok(graph)
  assert.equal(graph.properties.dataBinCount, 5)
  assert.equal(graph.properties.dataBinWidthWatts, 25)
  assert.equal(graph.properties.dataDomainXMin, 0)
  assert.equal(graph.properties.dataDomainXMax, 100)
  assert.equal(graph.properties.ariaValueText, '0–24 W')
  assert.equal(paths.length, 2)
  assert.deepEqual(
    paths.map(path => String(path.properties.dataActivityId)),
    ['239', '240'],
  )
  assert.match(String(paths[0].properties.d), /L 100\.00 34\.00$/)
  assert.deepEqual(byClass(chart, 'tri-cax-yt').map(text), ['0%', '20%', '40%', '60%'])
  assert.deepEqual(byClass(chart, 'tri-cax-xt').map(text), ['0 W', '100 W'])
})

test('compares precise skin temperature and time-normalized gear ratios between rides', () => {
  const first = comparisonActivity(243, {
    route: detail().route.map((point, index) => ({
      ...point,
      skinTemperatureC: 33.4 + index * 0.05,
    })),
    gearShifts: shiftedDetail().gearShifts.map((shift, index) => ({
      ...shift,
      elapsedS: index === 3 ? 4_000 : shift.elapsedS,
    })),
  })
  const second = comparisonActivity(244, {
    route: detail().route.map((point, index) => ({
      ...point,
      skinTemperatureC: 33.5 + index * 0.05,
    })),
    gearShifts: shiftedDetail().gearShifts.map((shift, index) => ({
      ...shift,
      elapsedS: index === 3 ? 4_000 : shift.elapsedS,
      rearTeeth: shift.rearTeeth === 19 ? 17 : shift.rearTeeth,
    })),
  })
  const rendered = buildActivityComparison(factory, [first, second])
  const skin = comparisonChart(rendered, 'skin-temperature')
  const skinGraph = byClass(skin, 'tri-compare-graph')[0]
  const ratios = comparisonChart(rendered, 'gear-ratio-distribution')
  const ratioGraph = byClass(ratios, 'tri-compare-distribution-graph')[0]

  assert.deepEqual(activityComparisonMetricsForSport('bike'), [
    'elevation',
    'speed',
    'hr',
    'power',
    'cadence',
    'respiration',
    'temperature',
    'skin-temperature',
  ])
  assert.equal(activityComparisonDisplayValueAtDistance(first, 'skin-temperature', 10), '33.45°C')
  assert.equal(skin.properties.dataAvailable, '2')
  assert.equal(byClass(skin, 'tri-compare-line').length, 2)
  assert.ok(Math.abs(Number(skinGraph.properties.dataDomainYMin) - 33.35) < 1e-9)
  assert.ok(Math.abs(Number(skinGraph.properties.dataDomainYMax) - 33.7) < 1e-9)
  for (const tick of byClass(skin, 'tri-cax-yt').map(text)) assert.match(tick, /^\d+\.\d{2}°C$/)

  assert.equal(ratios.properties.dataAvailable, '2')
  assert.equal(ratioGraph.properties.dataRatioCount, 6)
  assert.equal(ratioGraph.properties.dataDomainXMin, 0)
  assert.equal(ratioGraph.properties.dataDomainXMax, 5)
  assert.equal(byClass(ratios, 'tri-compare-line').length, 2)
  assert.deepEqual(byClass(ratios, 'tri-cax-xt').map(text), [
    '1.89×',
    '1.93×',
    '2.12×',
    '2.74×',
    '3.06×',
    '3.27×',
  ])
  for (const activity of [first, second]) {
    const total = activityGearRatioDistribution(activity).reduce(
      (sum, point) => sum + point.percentage,
      0,
    )
    assert.ok(Math.abs(total - 100) < 1e-9)
  }
})

test('normalizes heart-rate and power zone overlays to percentages', () => {
  const first = comparisonActivity(241, { hrZones: [60, 40], powerZones: [900, 100] })
  const second = comparisonActivity(242, { hrZones: [10, 90], powerZones: [1, 1] })
  const rendered = buildActivityComparison(factory, [first, second])

  assert.deepEqual(activityZonePercentages([60, 40]), [60, 40])
  assert.deepEqual(activityZonePercentages([10, Number.NaN, -5, 30]), [])
  for (const kind of ['hr-zones', 'power-zones']) {
    const chart = comparisonChart(rendered, kind)
    const graph = byClass(chart, 'tri-compare-zone-graph')[0]
    assert.ok(graph)
    assert.equal(graph.properties.dataZoneUnit, 'percent')
    assert.equal(byClass(chart, 'tri-compare-line').length, 2)
    assert.deepEqual(byClass(chart, 'tri-cax-yt').map(text), ['0%', '25%', '50%', '75%', '100%'])
    assert.doesNotMatch(text(chart), /\b\d+:\d{2}\b/)
  }
})

test('preserves mixed sensor availability without turning missing samples into zero lines', () => {
  const measured = comparisonActivity(251, {
    route: detail().route.map((point, index) => ({
      ...point,
      skinTemperatureC: 33.4 + index * 0.05,
    })),
  })
  const missing = comparisonActivity(252, {
    deviceWatts: false,
    route: detail().route.map(point => ({
      ...point,
      w: 0,
      hr: 0,
      cad: 0,
      resp: null,
      tempC: null,
      skinTemperatureC: null,
    })),
  })
  const rendered = buildActivityComparison(factory, [measured, missing])

  for (const kind of ['hr', 'power', 'cadence', 'respiration', 'temperature', 'skin-temperature']) {
    const chart = comparisonChart(rendered, kind)
    assert.equal(chart.properties.dataAvailable, '1')
    assert.equal(chart.properties.dataSelected, '2')
    assert.deepEqual(
      byClass(chart, 'tri-compare-line').map(line => String(line.properties.dataActivityId)),
      ['251'],
    )
    assert.equal(byClass(chart, 'tri-compare-readout').length, 0)
  }
  assert.equal(byClass(rendered, 'tri-compare-readout-row').length, 2)
  const elevation = comparisonChart(rendered, 'elevation')
  assert.equal(elevation.properties.dataAvailable, '2')
  assert.equal(byClass(elevation, 'tri-compare-line').length, 2)
})

test('keeps zero-coverage and single-sample plots visible but noninteractive', () => {
  const singleSample = comparisonActivity(256, {
    route: detail().route.map((point, index) => ({ ...point, hr: index === 1 ? 140 : 0 })),
  })
  const missing = comparisonActivity(257, {
    route: detail().route.map(point => ({ ...point, hr: 0 })),
  })
  const rendered = buildActivityComparison(factory, [singleSample, missing])

  for (const kind of [
    'hr',
    'gear-ratio-distribution',
    'power-curve',
    'power-distribution',
    'hr-zones',
    'power-zones',
  ]) {
    const chart = comparisonChart(rendered, kind)
    const graph = byClass(chart, 'tri-compare-graph')[0]
    assert.ok(graph)
    assert.equal(chart.properties.dataAvailable, '0')
    assert.equal(byClass(chart, 'tri-compare-line').length, 0)
    assert.equal(graph.properties.role, 'img')
    assert.equal(graph.properties.ariaDisabled, 'true')
    assert.equal(graph.properties.tabIndex, undefined)
    assert.equal(graph.properties.ariaValueMin, undefined)
    assert.equal(graph.properties.ariaValueMax, undefined)
    assert.equal(graph.properties.ariaValueNow, undefined)
    assert.equal(graph.properties.ariaValueText, undefined)
    assert.equal(graph.properties.ariaLabel, graph.properties.dataI18nAriaLabel)
  }
  const elevation = comparisonChart(rendered, 'elevation')
  assert.equal(byClass(elevation, 'tri-compare-graph')[0].properties.role, 'slider')
})

test('comparison elevation readouts include signed smoothed grades in either unit system', () => {
  const base = detail().route[0]
  for (const slope of [0.08, -0.04, 0]) {
    const activity = comparisonActivity(258, {
      route: Array.from({ length: 7 }, (_, index) => ({
        ...base,
        d: index * 0.1,
        alt: 100 + slope * index * 100,
      })),
    })
    for (const units of [METRIC_TRIATHLON_PRESENTATION, imperialPresentation]) {
      for (const distance of [0, 0.25, activity.route[6].d]) {
        const altitude = formatAltitude(units, 100 + slope * distance * 1000)
        const grade = `${slope >= 0 ? '+' : ''}${(slope * 100).toFixed(1)}%`
        assert.equal(
          activityComparisonDisplayValueAtDistance(activity, 'elevation', distance, units),
          `${altitude} · ${grade}`,
        )
      }
    }
    assert.equal(activityComparisonDisplayValueAtDistance(activity, 'elevation', -1), '—')
    assert.equal(activityComparisonDisplayValueAtDistance(activity, 'elevation', 1), '—')
  }
  const smoothed = comparisonActivity(259, {
    route: [100, 110, 140, 110, 120].map((alt, index) => ({ ...base, d: index * 0.1, alt })),
  })
  assert.equal(
    activityComparisonDisplayValueAtDistance(smoothed, 'elevation', 0.2),
    '140 m · +5.0%',
  )
})

test('comparison elevation omits grades across invalid samples or without a distance span', () => {
  const base = detail().route[0]
  const missing = comparisonActivity(260, {
    route: [100, Number.NaN, 120, 130, 140].map((alt, index) => ({ ...base, d: index * 0.1, alt })),
  })
  assert.equal(activityComparisonDisplayValueAtDistance(missing, 'elevation', 0.2), '120 m')
  assert.equal(activityComparisonDisplayValueAtDistance(missing, 'elevation', 0.1), '—')
  const stationary = comparisonActivity(260, {
    route: [100, 110].map(alt => ({ ...base, d: 0, alt })),
  })
  assert.equal(activityComparisonDisplayValueAtDistance(stationary, 'elevation', 0), '110 m')
})

test('interpolates only finite adjacent metrics and returns null past telemetry', () => {
  const measured = comparisonActivity(261)
  assert.equal(activityComparisonMetricAtDistance(measured, 'elevation', 5), 82)
  assert.equal(activityComparisonMetricAtDistance(measured, 'elevation', -1), null)
  assert.equal(activityComparisonMetricAtDistance(measured, 'elevation', 31), null)

  const missingHeartRate = comparisonActivity(262, {
    route: detail().route.map((point, index) => ({ ...point, hr: index === 1 ? 0 : point.hr })),
  })
  assert.equal(activityComparisonMetricAtDistance(missingHeartRate, 'hr', 5), null)
  assert.equal(activityComparisonMetricAtDistance(missingHeartRate, 'hr', 10), null)
  assert.equal(activityComparisonMetricAtDistance(missingHeartRate, 'hr', 15), null)

  const incapablePower = comparisonActivity(263, {
    deviceWatts: false,
    route: detail().route.map(point => ({ ...point, w: 0 })),
  })
  assert.equal(activityComparisonMetricAtDistance(incapablePower, 'power', 0), null)

  const capablePower = comparisonActivity(264, {
    deviceWatts: false,
    route: detail().route.map((point, index) => ({
      ...point,
      w: index === 0 ? 0 : 100 + index * 10,
    })),
  })
  assert.equal(activityComparisonMetricAtDistance(capablePower, 'power', 0), 0)
  assert.equal(activityComparisonMetricAtDistance(capablePower, 'power', 5), 55)

  const temperature = comparisonActivity(265, {
    route: detail().route.map((point, index) => ({
      ...point,
      tempC: index === 0 ? 0 : index === 1 ? null : point.tempC,
    })),
  })
  assert.equal(activityComparisonMetricAtDistance(temperature, 'temperature', 0), 0)
  assert.equal(activityComparisonMetricAtDistance(temperature, 'temperature', 5), null)

  const reset = comparisonActivity(266, {
    route: detail().route.map((point, index) => ({
      ...point,
      d: [0, 10, 5, 15][index],
      hr: [100, 120, 200, 220][index],
    })),
  })
  assert.equal(activityComparisonMetricAtDistance(reset, 'hr', 7), 114)
  const resetChart = comparisonChart(
    buildActivityComparison(factory, [reset, comparisonActivity(267)]),
    'hr',
  )
  const resetPath = byClass(resetChart, 'tri-compare-line').find(
    line => String(line.properties.dataActivityId) === '266',
  )
  assert.ok(resetPath)
  assert.equal(String(resetPath.properties.d).match(/M/g)?.length, 2)

  const plateau = comparisonActivity(268, {
    route: detail().route.map((point, index) => ({
      ...point,
      d: [0, 10, 10, 20][index],
      hr: [100, 110, 130, 140][index],
    })),
  })
  assert.equal(activityComparisonMetricAtDistance(plateau, 'hr', 10), 130)
  assert.equal(activityComparisonMetricAtDistance(plateau, 'hr', 15), 135)
  assert.equal(
    nearestPowerCurveValue(
      normalizePowerCurvePoints([
        { s: 5, w: 500 },
        { s: 10, w: 400 },
      ]),
      7,
    ),
    500,
  )
})

test('degrades deterministically for empty, single, mixed-sport, and unrouted selections', () => {
  const empty = buildActivityComparison(factory, [])
  const single = buildActivityComparison(factory, [comparisonActivity(271)])
  const mixed = buildActivityComparison(factory, [
    comparisonActivity(272),
    comparisonActivity(273, { sport: 'run' }),
  ])
  const unrouted = buildActivityComparison(factory, [
    comparisonActivity(274),
    comparisonActivity(275, { route: detail().route.slice(0, 1) }),
  ])

  assert.equal(empty.properties.dataCompareState, 'empty')
  assert.equal(single.properties.dataCompareState, 'insufficient')
  assert.equal(mixed.properties.dataCompareState, 'mixed-sport')
  assert.equal(unrouted.properties.dataCompareState, 'route-unavailable')
  for (const rendered of [empty, single, mixed, unrouted]) {
    assert.equal(byClass(rendered, 'tri-compare').length, 1)
    assert.equal(byClass(rendered, 'tri-compare-empty').length, 1)
    assert.equal(byClass(rendered, 'tri-compare-chart').length, 0)
    assert.equal(byClass(rendered, 'tri-compare-map-stage').length, 0)
    assert.equal(byClass(rendered, 'tri-compare-readout').length, 0)
  }
})

test('renders the IF graph and session VI table value in server HTML', () => {
  const d = detail()
  d.wahoo = {
    activityId: 'wahoo:1',
    fitPath: null,
    sha256: 'a'.repeat(64),
    sourceDevice: 'ELEMNT BOLT',
    startOffsetS: 0,
    distanceM: 30_000,
    metrics: {
      ...emptyWahooMetrics(),
      intensityFactor: 0.814,
      normalizedPower: 203.5,
      avgPower: 200,
    },
    summarySources: {},
    streamFallback: 'strava',
  }
  d.cyclingIntensityTrace = buildCyclingIntensityTrace({
    streams: { time: Array.from({ length: 4801 }, (_, i) => i), watts: Array(4801).fill(200) },
    metrics: d.wahoo.metrics,
    startOffsetS: 0,
    elapsedTimeS: 4800,
    route: d.route,
    athleteFtp: 300,
  })
  assert.ok(d.cyclingIntensityTrace)
  const card = buildActivity(factory, d, true)
  const intensity = descendants(card, node => node.properties.dataTriTrace === 'intensity-factor')
  const variability = descendants(
    card,
    node => node.properties.dataTriTrace === 'variability-index',
  )
  assert.equal(intensity.length, 1)
  assert.equal(variability.length, 0)
  assert.match(text(intensity[0]), /0\.814 · Wahoo/)
  const viRow = byTag(card, 'tr').find(row => row.properties.dataStatKey === 'variability index')
  assert.ok(viRow)
  assert.equal(text(byTag(viRow, 'td')[0]), '1.018')
  assert.doesNotMatch(text(card), /calculated from Wahoo/)
  const viStat = (activity: StravaActivityDetail) =>
    activityStatRows(METRIC_TRIATHLON_PRESENTATION, activity).find(
      ([key]) => key === 'variability index',
    )
  assert.deepEqual(viStat({ ...d, cyclingIntensityTrace: null }), ['variability index', '1.018'])
  assert.deepEqual(
    viStat({ ...d, wahoo: { ...d.wahoo, metrics: { ...d.wahoo.metrics, normalizedPower: 0 } } }),
    ['variability index', '0.000'],
  )
  assert.equal(
    viStat({ ...d, wahoo: { ...d.wahoo, metrics: { ...d.wahoo.metrics, avgPower: 0 } } }),
    undefined,
  )
  assert.equal(
    viStat({ ...d, wahoo: { ...d.wahoo, metrics: { ...d.wahoo.metrics, normalizedPower: null } } }),
    undefined,
  )
  assert.equal(viStat({ ...d, sport: 'run' }), undefined)
  assert.deepEqual(
    activityStatRows(frenchPresentation, d).find(([key]) => key === 'variability index'),
    ['variability index', '1,018'],
  )
  assert.match(text(intensity[0]), /cumulative/)
  assert.equal(intensity[0].properties.dataCyclingIntensitySource, 'calculated-wahoo')
  for (const chart of intensity) {
    assert.ok(String(byClass(chart, 'tri-elev-line')[0].properties.d).includes('L'))
    assert.equal(byClass(chart, 'tri-elev-cursor').length, 1)
    assert.ok(byClass(chart, 'tri-elev')[0].properties.dataDomainEndDistanceKm)
  }
  const simplified = buildActivity(factory, d, true, undefined, false, false, {
    'intensity-factor': false,
  })
  assert.equal(
    descendants(simplified, node => node.properties.dataTriTrace === 'intensity-factor').length,
    0,
  )
  assert.equal(
    descendants(simplified, node => node.properties.dataTriTrace === 'variability-index').length,
    0,
  )
  const indoor = buildIntensityFactorChart(factory, { ...d, route: [] })
  assert.ok(indoor)
  assert.equal(byClass(indoor, 'tri-elev')[0].properties.dataDomainEndElapsedS, 4800)
  assert.equal(buildIntensityFactorChart(factory, { ...d, sport: 'run' }), null)
  assert.equal(buildIntensityFactorChart(factory, { ...d, wahoo: undefined }), null)
  const frenchChart = buildIntensityFactorChart(factoryFor(frenchPresentation), d)
  assert.ok(frenchChart)
  assert.match(text(frenchChart), /facteur d'intensité/)
})

test('renders HR workout analysis and session estimates for activities without power', () => {
  const sports: StravaActivityDetail['sport'][] = [
    'walk',
    'run',
    'swim',
    'yoga',
    'strength',
    'treatment',
    'sauna',
    'bike',
  ]
  for (const sport of sports) {
    for (const embedded of [false, true]) {
      const heartRateTrace = Array.from({ length: 31 }, (_, index) =>
        heartRateTracePoint(0, index * 10, index < 7 ? 100 : 120, {
          heatStrainIndex: index / 10,
          heatStrainSource: 'core-app',
          coreTemperatureC: 37 + index / 30,
          coreTemperatureSource: 'core-app',
          skinTemperatureC: 33 + index / 30,
          skinTemperatureSource: 'core-app',
        }),
      )
      const activity = detail({
        sport,
        route: [],
        distanceKm: 0,
        deviceWatts: false,
        elapsedTimeS: 300,
        movingTimeS: 300,
        heartRateTrace,
        heartRatePhysiology: estimateHeartRatePhysiology(heartRateTrace, 200, 'walk'),
      })
      const rendered = buildActivity(factory, activity, true, undefined, false, embedded)
      const analysis = byClass(rendered, 'tri-workout-analysis')[0]
      assert.ok(analysis, sport)
      assert.equal(analysis.properties.dataWorkoutAnalysisMetric, 'hr')
      const heartRateGraphs = descendants(rendered, node => node.properties.dataTriTrace === 'hr')
      assert.equal(heartRateGraphs.length, 1, `${sport} renders HR once`)
      assert.equal(
        descendants(analysis, node => node.properties.dataTriTrace === 'hr')[0],
        heartRateGraphs[0],
      )
      assert.match(
        text(byClass(analysis, 'tri-workout-stats')[0]),
        /highest 120 bpm.*avg.*lowest 100 bpm/,
      )
      const more = byClass(rendered, 'tri-act-more')[0]
      const children = more.children.filter((node): node is Element => node.type === 'element')
      assert.equal(children[0], analysis)
      assert.deepEqual(
        children.slice(1, 6).map(node => node.properties.dataTriTrace),
        [
          'stamina',
          'performance-condition',
          'heat-strain-index',
          'core-temperature',
          'skin-temperature',
        ],
      )
      const stamina = byClass(rendered, 'tri-stamina-chart')[0]
      assert.equal(stamina.properties.dataStaminaSource, 'garden-estimate')
      assert.match(String(byTag(stamina, 'svg')[0].properties.dataDomainEndElapsedS), /300/)
      assert.match(
        String(byClass(stamina, 'tri-elev-d')[0].properties.dataGlossDef),
        /carried from earlier sessions/,
      )
      assert.match(
        String(byClass(rendered, 'tri-performance-condition-source')[0].properties.dataGlossDef),
        /HR change proxy/,
      )
    }
  }
})

test('keeps standalone HR for pace, power, and activities without workout analysis', () => {
  const bike = analysisDetail()
  const activities = [
    { activity: bike, metric: 'power' },
    { activity: { ...bike, sport: 'run' }, metric: 'pace' },
    { activity: { ...bike, sport: 'swim' }, metric: 'pace' },
    { activity: { ...bike, analysisRanges: [] }, metric: null },
  ] satisfies { activity: StravaActivityDetail; metric: string | null }[]
  for (const { activity, metric } of activities) {
    for (const embedded of [false, true]) {
      const rendered = buildActivity(factory, activity, true, undefined, false, embedded)
      const analysis = byClass(rendered, 'tri-workout-analysis')[0]
      assert.equal(analysis?.properties.dataWorkoutAnalysisMetric ?? null, metric)
      const heartRateGraphs = descendants(rendered, node => node.properties.dataTriTrace === 'hr')
      assert.equal(heartRateGraphs.length, 1)
      const more = byClass(rendered, 'tri-act-more')[0]
      assert.ok(more.children.includes(heartRateGraphs[0]))
    }
  }
})

test('renders swim physiology and a duration drag curve in full and embedded activity cards', () => {
  const samples = Array.from({ length: 61 }, (_, i) => ({
    elapsedS: i * 10,
    distanceKm: i * 0.007,
    heartRate: 140,
    speedMps: 0.7,
    strokeRateSpm: 24,
  }))
  const swim = detail({
    sport: 'swim',
    route: [],
    elapsedTimeS: 600,
    movingTimeS: 600,
    distanceKm: 0.42,
    swimPhysiology: estimateSwimPhysiology(samples, 200, 'activity-average'),
    swimPower: buildSwimPowerEstimate(
      'route',
      [{ startElapsedS: 0, endElapsedS: 600, durationS: 600, distanceM: 420, stroke: null }],
      'ground-speed',
    ),
  })
  for (const embedded of [false, true]) {
    const node = buildActivity(factory, swim, true, undefined, false, embedded)
    assert.equal(
      byClass(node, 'tri-stamina-chart')[0].properties.dataStaminaSource,
      'garden-estimate',
    )
    assert.match(
      String(byClass(node, 'tri-performance-condition-source')[0].properties.dataGlossDef),
      /swim/,
    )
    const curve = byClass(node, 'tri-swim-drag-chart')[0]
    assert.ok(curve)
    assert.match(text(curve), /idx/)
    assert.doesNotMatch(text(curve), /2:30| index/)
    const title = byClass(curve, 'tri-zone-title')[0]
    assert.match(String(title.properties.dataGlossDef), /2:30\/100m = 100/)
    assert.equal(title.properties.tabIndex, 0)
    assert.doesNotMatch(text(curve), /W\/kg|FTP/)
    assert.equal(byTag(curve, 'svg')[0].properties.ariaLabel, 'modeled swim drag power curve')
    const distribution = byClass(node, 'tri-swim-power-distribution')[0]
    assert.ok(distribution)
    assert.match(text(distribution), /25 idx drag distribution/)
    assert.doesNotMatch(text(distribution), /\bW\b|FTP|2:30/)
    const histogram = byClass(distribution, 'tri-hist-svg')[0]
    assert.equal(histogram.properties.dataHistUnit, 'idx')
    assert.equal(histogram.properties.role, 'slider')
    assert.equal(histogram.properties.tabIndex, 0)
    assert.equal(histogram.properties.ariaLabel, 'modeled swim drag distribution')
    const bins: unknown = JSON.parse(String(histogram.properties.dataHist))
    assert.deepEqual(bins, [0, 0, 0, 0, 600])
    assert.match(
      String(byClass(distribution, 'tri-zone-title')[0].properties.dataGlossDef),
      /2:30\/100m = 100/,
    )
  }
  assert.equal(buildSwimPowerCurve(factory, detail({ sport: 'swim' })), null)
  assert.equal(buildPowerHist(factory, detail({ sport: 'swim', powerHist: [10, 20] })), null)
  const slow = detail({
    sport: 'swim',
    swimPower: buildSwimPowerEstimate('apple', [
      { startElapsedS: 0, endElapsedS: 80, durationS: 80, distanceM: 25, stroke: 'freestyle' },
    ]),
  })
  assert.ok(buildPowerHist(factory, slow))
})

test('retains a labeled unavailable condition row for every activity kind and honors display settings', () => {
  const sports: StravaActivityDetail['sport'][] = [
    'bike',
    'run',
    'swim',
    'walk',
    'yoga',
    'strength',
    'treatment',
    'sauna',
  ]
  for (const sport of sports) {
    for (const embedded of [false, true]) {
      const activity = detail({ sport, route: [], heartRateTrace: [], deviceWatts: false })
      const rendered = buildActivity(factory, activity, true, undefined, false, embedded)
      const unavailable = descendants(
        rendered,
        node => node.properties.dataTriUnavailable === 'performance-condition',
      )
      assert.equal(unavailable.length, 1, sport)
      assert.match(text(unavailable[0]), /performance condition.*no data available/)
      assert.equal(byTag(unavailable[0], 'svg').length, 0)
      const hidden = buildActivity(factory, activity, true, undefined, false, embedded, {
        'performance-condition': false,
      })
      assert.equal(
        descendants(hidden, node => node.properties.dataTriUnavailable === 'performance-condition')
          .length,
        0,
      )
    }
  }
})

test('keeps native physiology ahead of the HR fallback and omits empty workout analysis', () => {
  const activity = cyclingDynamicsDetail()
  const points = Array.from({ length: 31 }, (_, index) => heartRateTracePoint(0, index * 10, 100))
  activity.heartRatePhysiology = estimateHeartRatePhysiology(points, 200, 'bike')
  activity.staminaTrace = {
    source: 'garmin',
    method: 'garmin-native',
    ftpWatts: null,
    maxHeartRateBpm: null,
  }
  activity.route = activity.route.map(point => ({
    ...point,
    stamina: 0,
    potentialStamina: 0,
    performanceCondition: 0,
  }))
  const rendered = buildActivity(factory, activity, true)
  assert.equal(byClass(rendered, 'tri-stamina-chart')[0].properties.dataStaminaSource, 'garmin')
  assert.equal(
    byClass(
      buildActivity(factory, detail({ sport: 'treatment', route: [], heartRateTrace: [] }), true),
      'tri-workout-analysis',
    ).length,
    0,
  )
})

test('cycling PD uses the recorded power zones and FTP boundaries in full and embedded cards', () => {
  const ride = detail({
    powerZones: [0, 60, 120, 180, 120, 60, 60],
    analysisRanges: analysisRanges(),
  })
  for (const embedded of [false, true]) {
    const card = buildActivity(factory, ride, true, ctx(), false, embedded)
    const analysis = byClass(card, 'tri-workout-analysis')[0]
    const tabs = byClass(analysis, 'tri-workout-analysis-tab')
    assert.deepEqual(
      tabs.map(text),
      embedded ? ['WA', 'PD'] : ['workout analysis', 'power distribution'],
    )
    assert.equal(tabs[1].properties.ariaLabel, 'power distribution')
    const panel = byClass(analysis, 'tri-workout-analysis-panel')[1]
    assert.equal(panel.properties.dataWorkoutAnalysisPanel, 'power')
    assert.equal(panel.properties.hidden, true)
    assert.equal(panel.properties.inert, true)
    assert.deepEqual(tabs[1].properties.ariaControls, [panel.properties.id])
    const power = byClass(panel, 'tri-cycling-power-distribution')[0]
    assert.deepEqual(byClass(power, 'tri-training-zone-name').map(text), [
      'Z7',
      'Z6',
      'Z5',
      'Z4',
      'Z3',
      'Z2',
      'Z1',
    ])
    assert.deepEqual(byClass(power, 'tri-training-zone-range').map(text), [
      '> 400 W',
      '351–400 W',
      '301–350 W',
      '251–300 W',
      '201–250 W',
      '151–200 W',
      '≤ 150 W',
    ])
    assert.deepEqual(byClass(power, 'tri-training-zone-pct').map(text), [
      '10.0%',
      '10.0%',
      '20.0%',
      '30.0%',
      '20.0%',
      '10.0%',
      '0.0%',
    ])
    assert.deepEqual(byClass(power, 'tri-training-zone-summary-value').map(text), ['30% in zone 4'])
    assert.deepEqual(byClass(power, 'tri-training-zone-summary-time').map(text), ['10:00'])
    assert.deepEqual(byClass(power, 'tri-training-zone-source').map(text), ['based on FTP 260 W'])
    assert.equal(
      byClass(power, 'tri-training-zone-row')[3].properties.ariaLabel,
      'Z4 threshold, 3:00, 30.0%, 251–300 W',
    )
  }
})

test('cycling PD omits missing or invalid distributions and supports rides without laps', () => {
  for (const powerZones of [
    null,
    [],
    [1, 2],
    [0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0],
    [0, NaN, 0, 0, 0, 0, 0],
  ]) {
    const card = buildActivity(factory, detail({ powerZones }), true, ctx())
    assert.equal(byClass(card, 'tri-cycling-power-distribution').length, 0)
  }
  const ride = detail({ powerZones: [60, 0, 0, 0, 0, 0, 0], analysisRanges: [] })
  for (const zones of [
    null,
    { hr: [], power: [150, 200], ftp: 260 },
    { hr: [], power: [150, 200, 200, 300, 350, 400], ftp: 260 },
  ]) {
    const card = buildActivity(factory, ride, true, ctx({ zones }))
    assert.equal(byClass(card, 'tri-cycling-power-distribution').length, 0)
  }
  const analysis = buildWorkoutAnalysis(factory, ride, false, ctx().zones)
  assert.ok(analysis)
  assert.equal(byClass(analysis, 'tri-cycling-power-distribution').length, 1)
  const run = buildActivity(factory, { ...ride, sport: 'run' }, true, ctx())
  assert.equal(byClass(run, 'tri-cycling-power-distribution').length, 0)
  const filtered = powerViewActivity(excludeZeroPresentation, {
    ...ride,
    powerWithoutZeros: { avgWatts: 200, powerZones: [0, 60, 0, 0, 0, 0, 0], powerHist: [] },
  })
  const card = buildActivity(factory, filtered, true, ctx())
  assert.deepEqual(byClass(card, 'tri-training-zone-summary-value').map(text), ['100% in zone 2'])
})

test('renders crank torque views and supports missing cadence', () => {
  const activity = detail({ elapsedTimeS: 120 })
  const time = Array.from({ length: 120 }, (_, i) => i)
  const samples = cyclingTorqueSamples(
    { time, watts: time.map(() => 250), cadence: time.map(() => 90) },
    0,
    120,
  )
  activity.cyclingTorque = buildCyclingTorqueTrace(samples, 120, [
    { elapsedS: 0, d: 0 },
    { elapsedS: 120, d: 30 },
  ])
  const chart = buildCrankTorqueChart(factory, activity)
  assert.ok(chart)
  assert.match(text(chart), /crank torque/)
  assert.match(text(chart), /26.5 N·m/)
  const headers = byClass(chart, 'tri-elev-cap')
  assert.equal(headers.length, 2)
  for (const header of headers) {
    assert.equal(byClass(header, 'tri-torque-mode').length, 2)
    assert.equal(byClass(header, 'tri-curve-range').length, 2)
  }
  assert.equal(byClass(chart, 'tri-torque-density')[0].properties.role, 'slider')
  const frenchChart = buildCrankTorqueChart(factoryFor(frenchPresentation), activity)
  assert.ok(frenchChart)
  assert.match(text(frenchChart), /couple au pédalier/)
  const rendered = buildActivity(factory, activity)
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  assert.equal(byClass(more, 'tri-torque-panel').length, 1)
  assert.equal(buildCrankTorqueChart(factory, detail()), null)
  assert.equal(buildCrankTorqueChart(factory, { ...activity, sport: 'run' }), null)
})

const cyclingPowerDetail = (): StravaActivityDetail =>
  detail({
    elapsedTimeS: 1_000,
    cyclingPowerTrace: {
      source: 'wahoo',
      terrainSource: 'wahoo',
      method: 'recorded-power-average-v1',
      points: [0, 30, 300, 400, 700, 1_000].map((elapsedS, index) => ({
        elapsedS,
        distanceKm: elapsedS / 100,
        elevationM: [100, 120, 140, null, 80, 100][index],
        power30sWatts: [null, 0, 200, null, 0, 160][index],
        power5mWatts: [null, null, 180, null, null, 160][index],
        cumulativePowerWatts: [null, 0, 100, null, 75, 90][index],
      })),
    },
  })

test('cycling power renders elapsed power over elevation with named averaging controls and gaps', () => {
  const activity = cyclingPowerDetail()
  const selection = { ...analysisRanges()[0], startElapsedS: 250, endElapsedS: 750 }
  const chart = buildCyclingPowerChart(factory, activity, selection)
  assert.ok(chart)
  assert.equal(chart.properties.dataTriTrace, 'cycling-power')
  assert.equal(chart.properties.dataCyclingPowerWindow, '30')
  assert.equal(chart.properties.dataCyclingPowerSource, 'wahoo')
  const caption = byClass(chart, 'tri-elev-cap')[0]
  assert.match(text(caption), /30 s averageride average/)
  assert.equal(byClass(caption, 'tri-cycling-power-legend').length, 1)
  assert.equal(byClass(chart, 'tri-cycling-power-key--elevation').length, 0)
  const controls = byClass(chart, 'tri-cycling-power-window')
  assert.equal(byClass(byClass(chart, 'tri-elev-cap')[0], 'tri-chart-controls').length, 1)
  assert.equal(
    byClass(byClass(chart, 'tri-cycling-power-legend')[0], 'tri-chart-controls').length,
    0,
  )
  assert.deepEqual(
    controls.map(control => text(control)),
    ['30 s', '5 min'],
  )
  assert.deepEqual(
    controls.map(control => control.properties.ariaPressed),
    ['true', 'false'],
  )
  assert.ok(
    controls.every(control => control.tagName === 'button' && control.properties.type === 'button'),
  )
  const graph = byClass(chart, 'tri-cycling-power-plot')[0]
  assert.equal(graph.properties.dataDomainStartElapsedS, 0)
  assert.equal(graph.properties.dataDomainEndElapsedS, 1_000)
  assert.equal(graph.properties.role, 'slider')
  assert.equal(graph.properties.tabIndex, 0)
  assert.equal(graph.properties.ariaValueMax, 1_000)
  const selected = byClass(chart, 'tri-analysis-selection')[0]
  assert.equal(selected.properties.x, '25.00')
  assert.equal(selected.properties.width, '50.00')
  assert.equal(byClass(chart, 'tri-elev-cursor').length, 1)
  const series = byClass(chart, 'tri-cycling-power-line')
  assert.deepEqual(
    series.map(line => line.properties.dataCyclingPowerSeries),
    ['30', '300', 'cumulative'],
  )
  assert.equal(series[0].properties.hidden, undefined)
  assert.equal(series[1].properties.hidden, '')
  assert.equal(String(series[0].properties.d).match(/M /g)?.length, 2)
  assert.match(String(series[0].properties.d), /^M 3\.00 28\.00 L /)
  assert.match(String(series[0].properties.d), /M 70\.00 28\.00 L /)
  assert.equal(String(series[2].properties.d).match(/M /g)?.length, 2)
  const backdrop = byClass(chart, 'tri-cycling-power-elevation')[0]
  assert.equal(String(backdrop.properties.d).match(/Z/g)?.length, 2)
  assert.ok(byClass(chart, 'tri-cax-yt').some(tick => text(tick) === '0 W'))
  assert.deepEqual(
    byClass(chart, 'tri-cax-yt--right').map(tick => text(tick)),
    ['80 m', '110 m', '140 m'],
  )
  assert.deepEqual(
    activityCyclingPowerPoints(activity).map(point => point.d),
    [0, 30, 300, 400, 700, 1_000],
  )
})

test('cycling power retains provider metadata without visible source text', () => {
  const activity = cyclingPowerDetail()
  const trace = activity.cyclingPowerTrace
  assert.ok(trace)
  const sameSource = buildCyclingPowerChart(factory, activity)
  assert.ok(sameSource)
  assert.doesNotMatch(text(sameSource), /Wahoo|Strava|Garmin/)
  const virtualCourse: StravaActivityDetail = {
    ...activity,
    cyclingPowerTrace: { ...trace, terrainSource: 'garmin' },
  }
  const chart = buildCyclingPowerChart(factory, virtualCourse)
  assert.ok(chart)
  assert.equal(chart.properties.dataCyclingPowerSource, 'wahoo')
  assert.equal(chart.properties.dataCyclingPowerTerrainSource, 'garmin')
  assert.doesNotMatch(text(chart), /Wahoo|Strava|Garmin/)
  const frenchChart = buildCyclingPowerChart(factoryFor(frenchPresentation), virtualCourse)
  assert.ok(frenchChart)
  assert.doesNotMatch(text(frenchChart), /Wahoo|Strava|Garmin/)
})

test('cycling power renders continuous terrain and cumulative power across a recording pause', () => {
  const time = Array.from({ length: 401 }, (_, index) => (index < 60 ? index : index + 120))
  const cyclingPowerTrace = buildCyclingPowerTrace({
    source: 'wahoo',
    startOffsetS: 0,
    elapsedTimeS: 520,
    streams: {
      time,
      watts: time.map(() => 200),
      altitude: time.map(second => (second < 180 ? 120 : 121)),
    },
  })
  assert.ok(cyclingPowerTrace)
  const chart = buildCyclingPowerChart(factory, detail({ elapsedTimeS: 520, cyclingPowerTrace }))
  assert.ok(chart)
  const terrain = String(byClass(chart, 'tri-cycling-power-elevation')[0].properties.d)
  assert.equal(terrain.match(/M /g)?.length, 1)
  assert.equal(terrain.match(/Z/g)?.length, 1)
  const lines = byClass(chart, 'tri-cycling-power-line')
  assert.equal(String(lines[0].properties.d).match(/M /g)?.length, 2)
  assert.equal(String(lines[2].properties.d).match(/M /g)?.length, 1)
})

test('cycling power keeps wind gaps empty and preserves observed calm and signed direction', () => {
  const activity = cyclingPowerDetail()
  const analyses = environmentAnalyses()
  const environment = analyses.derived.environment
  assert.ok(environment)
  const sample = environment.samples[0]
  environment.samples = [0, 0, 8, -4, null, -2, 2].map((headwindKph, index) => ({
    ...sample,
    elapsedS: index * 150,
    headwindKph,
  }))
  const chart = buildCyclingPowerChart(factory, { ...activity, analyses })
  assert.ok(chart)
  assert.equal(byClass(chart, 'tri-cycling-power-wind--calm').length, 1)
  assert.equal(byClass(chart, 'tri-cycling-power-wind--headwind').length, 3)
  assert.equal(byClass(chart, 'tri-cycling-power-wind--tailwind').length, 2)
  assert.ok(
    byClass(chart, 'tri-cycling-power-wind').every(segment => {
      const start = Number(segment.properties.x)
      const end = start + Number(segment.properties.width)
      return end <= 45 || start >= 75
    }),
  )
  assert.match(text(byClass(chart, 'tri-elev-cap')[0]), /\+ headwind− tailwind/)
  assert.doesNotMatch(text(chart), /estimated wind/)
  const noWeather = buildCyclingPowerChart(factory, activity)
  assert.ok(noWeather)
  assert.equal(byClass(noWeather, 'tri-cycling-power-wind').length, 0)
  assert.match(text(noWeather), /wind unavailable/)
})

test('cycling power uses presentation units and locale while preserving recorded zeroes', () => {
  const activity = cyclingPowerDetail()
  const imperial = buildCyclingPowerChart(factoryFor(imperialPresentation), activity)
  assert.ok(imperial)
  assert.deepEqual(
    byClass(imperial, 'tri-cax-yt--right').map(tick => text(tick)),
    ['262 ft', '361 ft', '459 ft'],
  )
  const frenchChart = buildCyclingPowerChart(factoryFor(frenchPresentation), activity)
  assert.ok(frenchChart)
  assert.match(text(frenchChart), /puissance et relief/)
  assert.match(text(frenchChart), /moyenne sur 30 smoyenne de la sortie/)
  const filtered = buildCyclingPowerChart(
    factoryFor(excludeZeroPresentation),
    powerViewActivity(excludeZeroPresentation, activity),
  )
  assert.ok(filtered)
  assert.match(
    String(byClass(filtered, 'tri-cycling-power-line')[0].properties.d),
    /^M 3\.00 28\.00 /,
  )
  const trace = activity.cyclingPowerTrace
  assert.ok(trace)
  const zero = buildCyclingPowerChart(factory, {
    ...activity,
    cyclingPowerTrace: {
      ...trace,
      points: trace.points.map(point => ({
        ...point,
        power30sWatts: 0,
        power5mWatts: 0,
        cumulativePowerWatts: 0,
      })),
    },
  })
  assert.ok(zero)
  assert.ok(
    byClass(zero, 'tri-cycling-power-line').every(
      line => !String(line.properties.d).includes('NaN'),
    ),
  )
  const noElevation = buildCyclingPowerChart(factory, {
    ...activity,
    cyclingPowerTrace: {
      ...trace,
      points: trace.points.map(point => ({ ...point, elevationM: null })),
    },
  })
  assert.ok(noElevation)
  assert.equal(byClass(noElevation, 'tri-cycling-power-elevation').length, 0)
  assert.equal(byClass(noElevation, 'tri-cax-yt--right').length, 0)
})

test('places cycling graphs after muscle oxygen and before environment in cards and embeds', () => {
  const activity = cyclingPowerDetail()
  activity.analyses = environmentAnalyses()
  activity.route = activity.route.map((point, index) => ({
    ...point,
    muscleOxygenPct: [64, 62, 60, 58][index],
  }))
  const time = Array.from({ length: 120 }, (_, index) => index)
  activity.cyclingTorque = buildCyclingTorqueTrace(
    cyclingTorqueSamples({ time, watts: time.map(() => 250), cadence: time.map(() => 90) }, 0, 120),
    120,
    [
      { elapsedS: 0, d: 0 },
      { elapsedS: 120, d: 30 },
    ],
  )
  for (const embedded of [false, true]) {
    const card = buildActivity(factory, activity, true, ctx(), false, embedded)
    const more = byClass(card, 'tri-act-more')[0]
    assert.ok(more)
    const children = more.children.filter((child): child is Element => child.type === 'element')
    const oxygenIndex = children.findIndex(
      child => child.properties.dataTriTrace === 'muscle-oxygen',
    )
    assert.ok(oxygenIndex >= 0)
    assert.equal(children[oxygenIndex + 1], byClass(more, 'tri-torque-panel')[0])
    assert.equal(children[oxygenIndex + 2], byClass(more, 'tri-cycling-power')[0])
    assert.equal(children[oxygenIndex + 3], byClass(more, 'tri-environment')[0])
    const disabled = buildActivity(factory, activity, true, ctx(), false, embedded, {
      'cycling-power': false,
    })
    assert.equal(byClass(disabled, 'tri-cycling-power').length, 0)
  }
  assert.equal(buildCyclingPowerChart(factory, { ...activity, sport: 'run' }), null)
  assert.equal(buildCyclingPowerChart(factory, detail()), null)
})

test('renders night and individual naps with separate charts, unique readouts, and no overnight Garmin leakage', () => {
  const date = '2026-09-21'
  const details: Record<string, OuraDayDetail> = {}
  const days: Record<string, OuraDaily> = {}
  const sleepRow = (id: string, type: string, hour: string) => ({
    id,
    type,
    day: date,
    bedtime_start: `${date}T${hour}:00:00-04:00`,
    bedtime_end: `${date}T${hour}:30:00-04:00`,
    total_sleep_duration: 1200,
    sleep_phase_5_min: '423214',
    average_breath: 14,
    average_hrv: 45,
    heart_rate: { timestamp: `${date}T${hour}:00:00-04:00`, interval: 300, items: [55, null, 58] },
    hrv: { timestamp: `${date}T${hour}:00:00-04:00`, interval: 300, items: [42, null, 44] },
    sleep_score_delta: 0,
  })
  applyOuraSleepRows(
    [
      sleepRow('night', 'long_sleep', '01'),
      sleepRow('nap1', 'sleep', '14'),
      { ...sleepRow('nap2', 'late_nap', '19'), day: '2026-09-22' },
    ],
    days,
    details,
    date,
    date,
  )
  const summary = buildTriathlonDailyAnalytics(buildAnalytics(null), details)[date]
  summary.garminHealth = garminHealthFixture
  assert.ok(summary.sleep)
  summary.sleep.sleepContrib = { efficiency: 80 }
  summary.sleep.readinessContrib = { hrv_balance: 85 }
  const tree = buildDayAnalytics(factory, summary)
  const scoreArea = byClass(tree, 'tri-day-sleep-contributions')[0]
  assert.equal(byClass(scoreArea, 'tri-sleep-contrib').length, 4)
  assert.equal(byClass(tree, 'tri-health-recovery').length, 1)
  assert.equal(byClass(tree, 'tri-health-training').length, 1)
  const nightPeriod = byClass(tree, 'tri-sleep-period')[0]
  assert.ok(
    nightPeriod.children.indexOf(scoreArea) <
      nightPeriod.children.indexOf(byClass(nightPeriod, 'tri-day-sleep-stages')[0]),
  )
  const panes = byClass(tree, 'tri-sleep-pane')
  assert.equal(panes.length, 2)
  assert.equal(panes[0].properties.hidden, undefined)
  assert.equal(panes[1].properties.hidden, true)
  assert.deepEqual(
    byClass(tree, 'tri-sleep-view-button').map(button => button.properties.ariaPressed),
    ['true', 'false'],
  )
  const naps = byClass(tree, 'tri-sleep-nap')
  assert.equal(naps.length, 2)
  assert.match(text(naps[0]), /sleep score change0Oura/)
  assert.match(text(naps[1]), /score date2026-09-22Oura/)
  for (const nap of naps) {
    assert.equal(byClass(nap, 'tri-day-sleep-stages').length, 1)
    assert.equal(byClass(nap, 'tri-day-sleep-series--hrv').length, 1)
    assert.equal(byClass(nap, 'tri-day-sleep-series--heart-rate').length, 1)
    assert.equal(byClass(nap, 'tri-day-sleep-series--respiration').length, 0)
    assert.doesNotMatch(text(nap), /Garmin|sleep debt|sleep baseline|sleep target/)
  }
  const readouts = byClass(tree, 'tri-chart-readout')
    .map(node => node.properties.id)
    .filter(Boolean)
  assert.equal(new Set(readouts).size, readouts.length)
  const old = buildDayAnalytics(factory, { ...summary, naps: undefined })
  assert.match(text(old), /nap data unavailable/)
  assert.match(text(buildDayAnalytics(factory, { ...summary, naps: [] })), /no naps recorded/)
  const onlyNaps = buildDayAnalytics(factory, {
    ...summary,
    sleep: null,
    recovery: null,
    sleepMetrics: null,
    garminHealth: null,
  })
  assert.match(text(byClass(onlyNaps, 'tri-sleep-pane')[0]), /no sleep logged/)
  assert.equal(byClass(onlyNaps, 'tri-sleep-nap').length, 2)
})
