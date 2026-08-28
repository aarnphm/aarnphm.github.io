import type { Element, ElementContent } from 'hast'
import { h, s } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { CriticalPowerEstimate } from '../plugins/stores/critical-power'
import type {
  ActivityAnalysisRange,
  ActivityHeartRateTracePoint,
  StravaActivityDetail,
  SwimActivityInterval,
  SwimTrendPoint,
} from '../plugins/stores/strava'
import type { TriathlonDayAnalytics } from './triathlon-day-analytics'
import { createTriathlonFormatter } from '../components/triathlon/runtime/formatter'
import { calculateActivityExerciseLoad, emptyHealth } from '../plugins/stores/strava'
import {
  activityCompareColor,
  activityComparisonDisplayValueAtDistance as displayValueAtDistance,
  activityComparisonEligible,
  activityComparisonFractionForKey,
  activityComparisonMetricAtDistance as metricAtDistance,
  activityComparisonMetricsForSport,
  activityGearRatioDistribution,
  activityPowerDistributionPercentages,
  activitySelectionSummary,
  activityStatRows,
  activityZonePercentages,
  axisFrame,
  buildActivity,
  buildActivityComparison,
  buildCyclingBestEfforts,
  buildDayCard,
  buildElevation,
  buildHeartRateTrace,
  buildPowerBalanceChart,
  buildPowerCurve,
  buildPowerHist,
  buildShiftingChart,
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
  clock,
  decodePowerCurve,
  dlabel,
  encodePowerCurve,
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
  powerCurveFraction,
  powerCurveHoverAt,
  powerViewActivity,
  riderPositionAtDistance,
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

const METRIC_TRIATHLON_PRESENTATION: TriathlonPresentation = Object.freeze({
  ...DEFAULT_TRIATHLON_PRESENTATION,
  distance: 'metric',
})

const factory: TriNodeFactory<Element> = {
  presentation: METRIC_TRIATHLON_PRESENTATION,
  el: (tag, cls, text, attrs) =>
    h(tag, { ...(cls ? { class: cls } : {}), ...attrs }, text === undefined ? [] : [text]),
  svg: (tag, attrs) => s(tag, attrs),
  add: (parent, ...children) => parent.children.push(...children),
}

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
  }
  assert.equal(
    vo2SourceText('fr', 'garmin', null),
    "Cette valeur vient de Garmin Connect ou d'une saisie manuelle.",
  )
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
    'FTP 230 W (athlète). La puissance aérobie maximale estimée est de 307 W. Le poids est de 88,9 kg.',
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
    'FTP 230 W (athlete). Estimated maximum aerobic power is 307 W. Body weight is 88.9 kg.',
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

test('round trips dense and sparse power curve attributes', () => {
  const dense = [
    { s: 1, w: 700 },
    { s: 2, w: 660 },
    { s: 3, w: 635 },
  ]
  const sparse = [
    { s: 1, w: 700 },
    { s: 5, w: 590 },
    { s: 60, w: 350 },
  ]
  const sourced = [
    { s: 1, w: 700, activityId: 11, activityDate: '2026-08-01' },
    { s: 2, w: 660, activityId: 11, activityDate: '2026-08-01' },
    { s: 3, w: 635, activityId: 12, activityDate: '2026-08-02' },
  ]
  assert.deepEqual(decodePowerCurve(encodePowerCurve(dense)), dense)
  assert.deepEqual(decodePowerCurve(encodePowerCurve(sparse)), sparse)
  assert.deepEqual(decodePowerCurve(encodePowerCurve(sourced)), sourced)
  assert.deepEqual(nearestPowerCurvePoint(sourced, 2.7), sourced[2])
  assert.deepEqual(decodePowerCurve('d|1|700,nope'), [])
  assert.deepEqual(decodePowerCurve('d|1|700,'), [])
  assert.deepEqual(decodePowerCurve('d|1|700|11,2026-08-01,2'), [])
  assert.deepEqual(decodePowerCurve('s|1:700,2:'), [])
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
      'heatStrainIndex' | 'coreTemperatureC' | 'skinTemperatureC' | 'coreTemperatureSource'
    >
  > = {},
): ActivityHeartRateTracePoint => ({
  distanceKm,
  elapsedS,
  heartRate,
  heatStrainIndex: null,
  coreTemperatureC: null,
  skinTemperatureC: null,
  coreTemperatureSource: null,
  ...thermal,
})

const detail = (overrides: Partial<StravaActivityDetail> = {}): StravaActivityDetail => ({
  id: 101,
  sport: 'bike',
  name: 'Threshold ride',
  date: '2026-07-09',
  start: '2026-07-09T12:00:00Z',
  distanceKm: 30,
  movingTimeS: 4_800,
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
  avgTemp: 24,
  windKph: null,
  windDir: null,
  windDirDeg: null,
  windGustKph: null,
  location: 'Toronto',
  fueling: null,
  strength: null,
  sauna: null,
  garmin: null,
  computer: null,
  calculatedIntensityFactor: null,
  calculatedExerciseLoad: null,
  calculatedTrainingEffect: null,
  gearShifts: [],
  cyclingDynamics: null,
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
      coreTemperatureC: null,
      skinTemperatureC: null,
      coreTemperatureSource: null,
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
      coreTemperatureC: null,
      skinTemperatureC: null,
      coreTemperatureSource: null,
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
      coreTemperatureC: null,
      skinTemperatureC: null,
      coreTemperatureSource: null,
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
      coreTemperatureC: null,
      skinTemperatureC: null,
      coreTemperatureSource: null,
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
  ...overrides,
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

test('renders the cycling computer as an activity table row', () => {
  const garmin = buildActivity(factory, detail({ computer: 'garmin' }))
  const wahoo = buildActivity(factory, detail({ computer: 'wahoo' }))
  const absent = buildActivity(factory, detail())
  const computerRow = (node: TestNode): TestNode | undefined =>
    byTag(node, 'tr').find(row => row.properties.dataStatKey === 'computer')

  assert.equal(text(byClass(computerRow(garmin)!, 'tri-act-stat-v')[0]), 'Edge 1050')
  assert.equal(text(byClass(computerRow(wahoo)!, 'tri-act-stat-v')[0]), 'ELEMNT BOLT 3')
  assert.equal(computerRow(absent), undefined)
  assert.equal(byClass(garmin, 'tri-act-computer').length, 0)
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
  assert.equal(formatTrainingEffectLabel('SPEED'), 'speed')
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
    detail({ sport: 'run', calculatedTrainingEffect: { aerobic: 3.4, anaerobic: 1.2 } }),
  )
  assert.ok(calculatedDetails)
  assert.equal(calculatedDetails.properties.dataTrainingEffectSource, 'calculated')
  assert.equal(text(byClass(calculatedDetails, 'tri-zone-title')[0]), 'training effect')
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
    detail({ sport: 'swim', calculatedTrainingEffect: { aerobic: 2.4, anaerobic: 0.4 } }),
  )
  assert.ok(frenchCalculatedDetails)
  assert.equal(frenchCalculatedDetails.properties.ariaLabel, "effet d'entraînement")
  assert.equal(text(byClass(frenchCalculatedDetails, 'tri-zone-title')[0]), "effet d'entraînement")

  const nativeDetails = buildTrainingEffectDetails(
    factory,
    detail({ garmin, calculatedTrainingEffect: { aerobic: 1.1, anaerobic: 4.9 } }),
  )
  assert.ok(nativeDetails)
  assert.equal(nativeDetails.properties.dataTrainingEffectSource, 'garmin')
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
  assert.deepEqual(moreStatRows(METRIC_TRIATHLON_PRESENTATION, activity).slice(-2), [
    ['air temp', '24°C'],
    ['wind', '18 km/h SW / gust 31'],
  ])
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
    avgTemp: null,
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
    route: [],
    heartRateTrace: [heartRateTracePoint(0, 300, 110), heartRateTracePoint(0, 3_600, 130)],
    bestEfforts: null,
  })

  assert.deepEqual(activityStatRows(imperialPresentation, sauna), [
    ['time', '18:30'],
    ['duration', "1h15'"],
    ['temperature', '196°F'],
    ['humidity', '11%'],
    ['cooldown', 'cold plunge'],
    ['HTL', '7.7'],
    ['avg hr', '120 bpm · Oura'],
  ])
  assert.deepEqual(moreStatRows(imperialPresentation, sauna), [['max hr', '130 bpm']])
  const rendered = buildActivity(factoryFor(imperialPresentation), sauna, true)
  assert.equal(rendered.properties.dataActivityTitle, 'Untangle')
  assert.equal(
    byClass(rendered, 'tri-elev-wrap').find(element => element.properties.dataTriTrace === 'hr')
      ?.properties.dataTriTrace,
    'hr',
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
        coreTemperatureC: 37.05,
        skinTemperatureC: 34.2,
        coreTemperatureSource: 'core-app',
      }),
      heartRateTracePoint(0, 1_040, 90, {
        heatStrainIndex: 0,
        coreTemperatureC: 37.1,
        skinTemperatureC: 34.6,
        coreTemperatureSource: 'core-app',
      }),
      heartRateTracePoint(0, 1_560, 86, {
        heatStrainIndex: 0,
        coreTemperatureC: 37.14,
        skinTemperatureC: 34.9,
        coreTemperatureSource: 'core-app',
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

test('builds semantic distance, power, and climbing tables in metric units', () => {
  const rendered = buildCyclingBestEfforts(factory, detail())
  assert.ok(rendered)

  assert.equal(rendered.tagName, 'section')
  assert.equal(rendered.properties.ariaLabel, 'Cycling best efforts')
  assert.equal(byTag(rendered, 'caption').length, 0)
  assert.deepEqual(
    byClass(rendered, 'tri-effort-title').map(title => [title.tagName, text(title)]),
    [
      ['div', 'Distance'],
      ['div', 'Power'],
      ['div', 'Climbing'],
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
      ['region', 'Climbing efforts', 0],
    ],
  )

  const distance = table(rendered, 'distance')
  assert.equal(distance.properties.ariaLabel, 'Distance efforts')
  assert.deepEqual(headerText(distance), ['Distance', 'Time', 'Speed', 'Heart rate', 'Elev'])
  assert.deepEqual(bodyRows(distance), [['10K', '24:31', '24.5 km/h', '151 bpm', '-30 m']])

  const power = table(rendered, 'power')
  assert.deepEqual(headerText(power), ['Time', 'Power', 'W/kg', 'Heart rate', 'Elev'])
  assert.deepEqual(bodyRows(power), [['5 sec', '565 W', '6.45 W/kg', '150 bpm', '4 m']])

  const climbing = table(rendered, 'climbing')
  assert.deepEqual(headerText(climbing), [
    'Climb',
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
    ['hr', 'power', 'cadence', 'respiration', 'temperature'],
  )
  for (const graph of traces) {
    assert.equal(byClass(graph, 'tri-elev').length, 1)
    assert.equal(byClass(graph, 'tri-elev-area').length, 1)
    assert.equal(byClass(graph, 'tri-elev-line').length, 1)
    assert.equal(byClass(graph, 'tri-analysis-selection').length, 1)
  }
  assert.equal(byClass(activity, 'tri-analysis-selection').length, 6)
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
    ['temperature', '24°C avg'],
  )
  assert.deepEqual(byClass(temperature, 'tri-cax-yt').map(text), ['22°C', '24°C', '26°C'])
})

test('renders CORE bike graphs after ambient temperature with sub-degree domains', () => {
  const thermal = detail({
    route: detail().route.map((point, index) => ({
      ...point,
      heatStrainIndex: [0, 1.4, 3, 3.1][index],
      coreTemperatureC: [37.16, 37.17, 37.19, 37.18][index],
      skinTemperatureC: [33.4, 33.45, 33.5, 33.55][index],
    })),
  })
  const rendered = buildActivity(factory, thermal, true)
  const traces = byClass(rendered, 'tri-elev-wrap').filter(
    graph => graph.properties.dataTriTrace != null,
  )
  assert.deepEqual(
    traces.map(graph => graph.properties.dataTriTrace),
    [
      'hr',
      'power',
      'cadence',
      'respiration',
      'temperature',
      'heat-strain-index',
      'core-temperature',
      'skin-temperature',
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
  assert.match(text(byClass(trace, 'tri-elev-cap')[0]), /muscle oxygen61\.0% SmO₂ avg/)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['55.0%', '60.0%', '65.0%'])
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
      coreTemperatureC: [37.32, 37.61, 37.97, 38.33][index],
      skinTemperatureC: [32.54, 32.4, 32.21, 32.04][index],
      coreTemperatureSource: 'core-app',
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
  const route = Array.from({ length: 7 }, (_, index) => ({
    ...source.route[Math.min(index, source.route.length - 1)],
    d: index * 5,
    heatStrainIndex: [null, 1.4, null, null, 3, null, null][index],
  }))
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
      ['climb', 'group', 'Climbs'],
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
  const runAnalysis = byClass(more, 'tri-run-analysis')[0]
  const workout = byClass(more, 'tri-run-workout')[0]
  const splits = byClass(more, 'tri-run-splits')[0]
  assert.ok(runAnalysis)
  assert.equal(runAnalysis.properties.ariaLabel, 'Run analysis')
  assert.equal(runAnalysis.properties.dataRunAnalysisView, 'workout')
  const tabs = byClass(runAnalysis, 'tri-run-analysis-tab')
  assert.deepEqual(tabs.map(text), ['workout analysis', 'lap splits', 'pace distribution'])
  assert.deepEqual(
    tabs.map(tab => [tab.properties.role, tab.properties.ariaSelected, tab.properties.tabIndex]),
    [
      ['tab', 'true', 0],
      ['tab', 'false', -1],
      ['tab', 'false', -1],
    ],
  )
  const panels = byClass(runAnalysis, 'tri-run-analysis-panel')
  assert.deepEqual(
    panels.map(panel => [
      panel.properties.role,
      panel.properties.dataRunAnalysisPanel,
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
  assert.ok(workout)
  assert.equal(workout.tagName, 'section')
  assert.equal(workout.properties.ariaLabel, 'Run workout analysis')
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

  const pace = byClass(runAnalysis, 'tri-run-pace-distribution')[0]
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
    [['tri-run-analysis'], ['tri-elev-wrap']],
  )
})

test('keeps the available run-analysis tabs when pace telemetry is missing', () => {
  const run = analysisDetail()
  run.sport = 'run'
  const rendered = buildActivity(factory, run, true)
  assert.deepEqual(byClass(rendered, 'tri-run-analysis-tab').map(text), [
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
  assert.equal(selections.length, 6)
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
    ['hr', 'power', 'cadence', 'respiration', 'temperature'],
  )
  assert.equal(byClass(rendered, 'tri-elev-cursor').length, 6)
})

test('keeps an empty selected-route overlay available after deselection', () => {
  const route = buildRoute(factory, analysisDetail().route)
  const selectedRoute = byClass(route, 'tri-route-selected')[0]
  assert.ok(selectedRoute)
  assert.equal(selectedRoute.properties.d, '')
  assert.match(String(byClass(route, 'tri-route-path')[0].properties.d), /^M /)
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
  assert.deepEqual(byClass(splits, 'tri-run-splits-title').map(text), ['lap splits'])
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
    ['hr', 'power', 'cadence', 'respiration', 'temperature'],
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
  assert.equal(byClass(fullPageSwim, 'tri-elev-unavailable').length, 0)

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
    ['air temp', '24°C'],
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
    ['temp', '24°C'],
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
    ['temp', '75°F'],
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
      'temp',
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
    ['temp', '24°C'],
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

test('renders exact-date analytics and limits automatic rest-day analytics to sleep', () => {
  const date = '2026-08-16'
  const ride = detail({ id: 19771722076, date, name: 'Recovery Crit' })
  const run = detail({ id: 19771722077, date, name: 'Evening run', sport: 'run' })
  const summary: TriathlonDayAnalytics = {
    date,
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
  assert.equal(byClass(rest, 'tri-day-rest-analytics').length, 1)
  assert.equal(byClass(rest, 'tri-day-analytics-group').length, 1)
  assert.equal(byClass(rest, 'tri-day-analytics-group--sleep').length, 1)
  assert.equal(byClass(rest, 'tri-day-analytics-group--body-recovery').length, 0)
  assert.equal(byClass(rest, 'tri-day-analytics-group--state-load').length, 0)
  assert.equal(byClass(rest, 'tri-day-analytics-group--thermal').length, 0)
  assert.equal(byClass(rest, 'tri-day-sleep-stages').length, 1)
  assert.equal(byClass(rest, 'tri-day-sleep-series--hrv').length, 1)
  assert.equal(byClass(rest, 'tri-day-sleep-series--heart-rate').length, 1)
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
  const efforts = buildCyclingBestEfforts(imperialFactory, ride)
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

test('renders a route-less pool swim heart rate trace against metres', () => {
  const rendered = buildActivity(
    factory,
    swimTrendDetail({
      heartRateTrace: [
        heartRateTracePoint(0, 0, 110),
        heartRateTracePoint(0.025, 30, 120),
        heartRateTracePoint(0.05, 60, null),
        heartRateTracePoint(0.075, 90, 140),
        heartRateTracePoint(0.1, 120, 150),
      ],
    }),
    true,
    ctx(),
  )
  const trace = byClass(rendered, 'tri-elev-wrap').find(
    element => element.properties.dataTriTrace === 'hr',
  )

  assert.ok(trace)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['80bpm', '100bpm', '120bpm', '140bpm'])
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['0 m', '50 m', '100 m'])
  assert.match(String(byClass(trace, 'tri-elev-line')[0]?.properties.d), /^M 0 /)
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
  assert.equal(byClass(chart, 'tri-analysis-selection').length, 1)
  assert.equal(byClass(chart, 'tri-elev-cursor').length, 1)
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

test('places pedal balance immediately below the ride power trace', () => {
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
  const powerIndex = children.findIndex(child => child.properties.dataTriTrace === 'power')
  assert.ok(powerIndex >= 0)
  assert.deepEqual(
    children.slice(powerIndex + 1, powerIndex + 6).map(child => child.properties.dataTriTrace),
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

test('places stamina below power and above electronic shifting', () => {
  const ride = shiftedDetail()
  ride.route = ride.route.map((point, index) => ({
    ...point,
    stamina: [100, 76, 54, 32][index],
    potentialStamina: [100, 88, 67, 40][index],
  }))
  const rendered = buildActivity(factory, ride, true)
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const powerIndex = children.findIndex(child => child.properties.dataTriTrace === 'power')
  assert.ok(powerIndex >= 0)
  assert.equal(children[powerIndex + 1].properties.dataTriTrace, 'stamina')
  assert.equal(children[powerIndex + 2].properties.dataTriTrace, 'electronic-shifting')
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
    { ratio: 1.333333, percentage: 12.5 },
    { ratio: 2.736842, percentage: 62.5 },
    { ratio: 3.272727, percentage: 25 },
  ])
  assert.deepEqual(activityGearRatioDistribution({ ...ride, sport: 'run' }), [])
})

test('places electronic shifting immediately below the ride power trace', () => {
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

test('pairs hr/power zones and curve/hist into duos with aligned captions', () => {
  const rendered = buildActivity(factory, zonedDetail(), true, ctx())
  const duos = byClass(rendered, 'tri-zone-duo')
  assert.equal(duos.length, 2)
  assert.deepEqual(
    duos.flatMap(duo =>
      duo.children
        .filter((child): child is Element => child.type === 'element')
        .map(child => child.properties.dataTriTrace),
    ),
    ['heart-rate-zones', 'power-zones', 'power-curve', '25w-power-distribution'],
  )
  assert.deepEqual(byClass(duos[0], 'tri-zone-title').map(text), [
    'heart rate zones',
    'power zones',
  ])
  assert.deepEqual(byClass(duos[1], 'tri-zone-title').map(text), [
    'power curve',
    '25W power distribution',
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

test('places the pool overview and swim charts before heart rate zones in expanded details', () => {
  const rendered = buildActivity(
    factory,
    swimTrendDetail({ hrZones: [20, 40, 30, 10, 0] }),
    true,
    ctx(),
  )
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const overviewIndex = children.findIndex(child => classNames(child).includes('tri-pool'))
  const swimIndex = children.findIndex(child => classNames(child).includes('tri-swim-trends'))
  const zonesIndex = children.findIndex(child =>
    byClass(child, 'tri-zone-title').some(title => text(title) === 'heart rate zones'),
  )

  assert.ok(overviewIndex >= 0)
  assert.equal(swimIndex, overviewIndex + 1)
  assert.ok(zonesIndex > swimIndex)
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

test('places the pool overview immediately before non-embedded swim trends', () => {
  const activity = { ...swimToggleDetail(), strokes: { freestyle: 75, breaststroke: 25 } }
  const rendered = buildActivity(factory, activity, true)
  const summary = byClass(rendered, 'tri-act-figs--pool')[0]
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(summary)
  assert.ok(more)
  assert.equal(byClass(summary, 'tri-pool').length, 0)
  assert.equal(byClass(summary, 'tri-pool-cap').length, 1)
  assert.equal(byClass(summary, 'tri-pool-strokes').length, 1)

  const children = more.children.filter((child): child is Element => child.type === 'element')
  const overviewIndex = children.findIndex(child => classNames(child).includes('tri-pool'))
  const trendsIndex = children.findIndex(child => classNames(child).includes('tri-swim-trends'))
  assert.ok(overviewIndex >= 0)
  assert.equal(trendsIndex, overviewIndex + 1)

  const embedded = buildActivity(factory, activity, true, undefined, false, true)
  const embeddedSummary = byClass(embedded, 'tri-act-figs--pool')[0]
  const embeddedMore = byClass(embedded, 'tri-act-more')[0]
  assert.ok(embeddedSummary)
  assert.ok(embeddedMore)
  assert.equal(byClass(embeddedSummary, 'tri-pool').length, 1)
  assert.equal(byClass(embeddedMore, 'tri-pool').length, 0)
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
    '5m',
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
    ['1', '5', '10', '20', '30', '60', '120', '300', '1200', '3600', '20107'],
  )
  assert.deepEqual(
    ticks.map(tick => tick.properties.ariaPressed),
    [
      'true',
      'false',
      'false',
      'false',
      'false',
      'false',
      'false',
      'false',
      'false',
      'false',
      'false',
    ],
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
    '5m',
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
    '5m',
    '53m20s',
  ])
})

test('keeps shared power curve duration markers inside the visible domain', () => {
  assert.deepEqual(
    powerCurveDurationTicks(5, 300, [1, 15, 60, 300, 600]),
    [5, 10, 15, 20, 30, 60, 120, 300],
  )
  assert.deepEqual(
    powerCurveDurationTicks(1, 20_107, [1, 60, 300, 1_200, 3_600, 10_800]),
    [1, 5, 10, 20, 30, 60, 120, 300, 1_200, 3_600, 20_107],
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
  const encoded = String(svg.properties.dataCurve)
  const decoded = decodePowerCurve(encoded)
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
  assert.equal(decodePowerCurve(String(svg.properties.dataCurveRefSixWeeks))[0].w, 700)
  assert.equal(decodePowerCurve(String(svg.properties.dataCurveRefYear))[0].w, 1_060)
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
  assert.equal(text(summaries[0]), 'this ride · eCP 245 W · eW′ 9.6 kJ')
  assert.equal(summaries[0].properties.dataGlossDef, '1 independent effort · provisional')
  assert.equal(summaries[0].properties.tabIndex, 0)
  assert.equal('hidden' in summaries[0].properties, false)
  const anchors = byClass(curve, 'tri-critical-power-anchor')
  assert.equal(anchors.length, 0)
  const thresholds = byClass(curve, 'tri-curve-thresholds')
  assert.equal(thresholds.length, 1)
  assert.deepEqual((thresholds[0].children as Element[]).map(text), [
    'this ride · eCP 245 W · eW′ 9.6 kJ',
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

test('keeps critical power references off run power charts', () => {
  const run = detail({
    sport: 'run',
    deviceWatts: true,
    powerCurve: zonedDetail().powerCurve,
    powerHist: Array.from({ length: 13 }, () => 1),
  })
  const context = ctx({ criticalPower: criticalPower() })
  const curve = buildPowerCurve(factory, run, context)
  assert.ok(curve)
  assert.equal(byClass(curve, 'tri-curve-model').length, 0)
  assert.equal(byClass(curve, 'tri-curve-cp').length, 0)
  assert.equal(byClass(curve, 'tri-curve-ref').length, 0)
  assert.equal(byClass(curve, 'tri-curve-ftp').length, 0)
  assert.equal(byClass(curve, 'tri-curve-goal').length, 0)
  assert.equal(byClass(curve, 'tri-curve-thresholds').length, 0)
  const histogram = buildPowerHist(factory, run)
  assert.ok(histogram)
  assert.equal(byClass(histogram, 'tri-hist-cp').length, 0)
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
  assert.equal(activityComparisonDisplayValueAtDistance(activity, 'elevation', 5), '82 m')
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
    assert.equal(byClass(chart, 'tri-compare-readout').length, 0)
    assert.equal(text(byClass(chart, 'tri-compare-coverage')[0]), '2/2 · sensor coverage')
  }
  assert.equal(byClass(rendered, 'tri-compare-zone-band').length, 0)
  const maps = byClass(rendered, 'tri-compare-map')
  const mapPanels = byClass(rendered, 'tri-compare-map-panel')
  const mapStages = byClass(rendered, 'tri-compare-map-stage')
  const readouts = byClass(rendered, 'tri-compare-map-readout')
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
  assert.equal(mapStage.children.includes(readout), true)
  assert.equal(byClass(rendered, 'tri-compare-readout').length, 1)
  assert.equal(readout.properties.role, undefined)
  assert.equal(readout.properties.ariaLabel, undefined)
  assert.equal(readout.properties.dataI18nAriaLabel, undefined)
  assert.equal(readout.properties.dataCompareReadout, '')
  assert.equal(readout.properties.dataVisible, 'false')
  assert.equal(readout.properties.ariaHidden, 'true')
  assert.equal(byClass(readout, 'tri-compare-readout-context').length, 0)
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
    decodePowerCurve(String(graph.properties.dataCurveRefSixWeeks)),
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
  assert.equal(decodePowerCurve(String(graph.properties.dataCurveRefSixWeeks))[0].w, 740)
  assert.equal(decodePowerCurve(String(graph.properties.dataCurveRefYear))[0].w, 780)
  assert.equal(references.length, 2)
  assert.equal('hidden' in references[0].properties, false)
  assert.equal('hidden' in references[1].properties, true)
  assert.deepEqual(ticks.map(text), ['1s', '5s', '10s', '20s', '30s', '1m', '2m', '5m', '20m'])
  assert.equal(
    ticks.every(tick => tick.tagName === 'button'),
    true,
  )
  assert.deepEqual(
    ticks.map(tick => tick.properties.dataCurveSeconds),
    ['1', '5', '10', '20', '30', '60', '120', '300', '1200'],
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

test('reports mixed sensor coverage without turning missing samples into zero lines', () => {
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
