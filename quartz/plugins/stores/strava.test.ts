import assert from 'node:assert/strict'
import test from 'node:test'
import { isRecord } from '../../util/type-guards'
import { buildAnalytics, buildDataFeed } from './analytics'
import {
  emptyGarminFueling,
  emptyGarminMetrics,
  type GarminCache,
  type GarminRunWalkData,
} from './garmin'
import {
  applyActivityTracking,
  applyManualFueling,
  applyManualSauna,
  applyManualStrength,
  buildPayload,
  calculateActivityIntensityFactor,
  calculateActivityTrainingEffect,
  calculateAnaerobicPowerEstimate,
  calculateCyclingPerformanceCondition,
  calculateExerciseLoad,
  calculateHeartRateTss,
  emptyPayload,
  hasFetchedActivityDetail,
  type RawStravaActivity,
  type RawStravaAnalysisRange,
  type StravaRawCache,
  type StravaStreams,
} from './strava'
import { parseTrackingBlock } from './tracking'
import { emptyWahooMetrics, type WahooCache } from './wahoo'
import { summarizeWeatherDays, type WeatherActivity, type WeatherCache } from './weather'

test('calculates cycling performance condition from six-minute power and heart-rate load', () => {
  const sampleCount = 721
  const observed = Array<number>(sampleCount).fill(1)
  const baseline = calculateCyclingPerformanceCondition(
    Array<number>(sampleCount).fill(150),
    observed,
    Array<number>(sampleCount).fill(122),
    250,
    170,
    50,
  )
  const elevated = calculateCyclingPerformanceCondition(
    Array<number>(sampleCount).fill(165),
    observed,
    Array<number>(sampleCount).fill(122),
    250,
    170,
    50,
  )
  const partialCoverage = calculateCyclingPerformanceCondition(
    Array<number>(sampleCount).fill(150),
    observed.map((value, index) => (index % 10 === 9 ? 0 : value)),
    Array<number>(sampleCount).fill(122),
    250,
    170,
    50,
  )

  assert.ok(baseline)
  assert.ok(elevated)
  assert.ok(partialCoverage)
  assert.equal(baseline.windowSeconds, 360)
  assert.ok(baseline.samples.every(sample => sample.value === 0))
  assert.ok(elevated.samples.every(sample => sample.value === 10))
  assert.ok(partialCoverage.samples.every(sample => sample.value === 0))
  assert.equal(
    calculateCyclingPerformanceCondition(
      Array<number>(sampleCount).fill(165),
      Array<number>(sampleCount).fill(0),
      Array<number>(sampleCount).fill(122),
      250,
      170,
      50,
    ),
    null,
  )
})

test('calculates activity intensity from the sport-specific threshold signal', () => {
  assert.deepEqual(
    calculateActivityIntensityFactor(
      { sport: 'run', avgHr: 152, npWatts: null, deviceWatts: false, garmin: null },
      0.9094,
      282,
      173,
    ),
    { value: 0.909, source: 'pace' },
  )
  assert.deepEqual(
    calculateActivityIntensityFactor(
      { sport: 'bike', avgHr: 152, npWatts: 205, deviceWatts: true, garmin: null },
      null,
      250,
      173,
    ),
    { value: 0.82, source: 'power' },
  )
  assert.deepEqual(
    calculateActivityIntensityFactor(
      { sport: 'strength', avgHr: 138, npWatts: null, deviceWatts: false, garmin: null },
      null,
      282,
      173,
    ),
    { value: 0.798, source: 'heart-rate' },
  )
  assert.equal(
    calculateActivityIntensityFactor(
      { sport: 'strength', avgHr: null, npWatts: null, deviceWatts: false, garmin: null },
      null,
      282,
      173,
    ),
    null,
  )
})

test('calculates exercise load from capped intensity and moving duration', () => {
  assert.equal(calculateExerciseLoad(0.8, 3_600), 64)
  assert.equal(calculateExerciseLoad(1.011, 900), 25.6)
  assert.equal(calculateExerciseLoad(2, 3_600), 132.3)
  assert.equal(calculateExerciseLoad(0, 3_600), null)
  assert.equal(calculateExerciseLoad(0.8, 0), null)
})

test('normalizes heart-rate TRIMP to 100 TSS for one hour at threshold', () => {
  assert.equal(calculateHeartRateTss(173, 3_600, 50, 173, 196, 'M'), 100)
  assert.equal(calculateHeartRateTss(83, 1_500, 50, 173, 196, 'M'), 3.4)
  assert.equal(calculateHeartRateTss(120.5, 1_200, 50, 173, 196, 'M'), 9.6)
  assert.equal(calculateHeartRateTss(50, 1_200, 50, 173, 196, 'M'), null)
  assert.equal(calculateHeartRateTss(120, 0, 50, 173, 196, 'M'), null)
  assert.equal(calculateHeartRateTss(120, 1_200, 50, 196, 196, 'M'), null)
})

test('scores only W-prime-depleting 10 to 120 second power efforts', () => {
  const watts = [
    ...Array<number>(20).fill(0),
    ...Array<number>(20).fill(320),
    ...Array<number>(20).fill(0),
    ...Array<number>(20).fill(600),
    ...Array<number>(20).fill(0),
    ...Array<number>(121).fill(600),
    ...Array<number>(20).fill(0),
  ]
  const estimate = calculateAnaerobicPowerEstimate(watts, 3_600, {
    criticalPowerWatts: 300,
    wPrimeJoules: 6_000,
  })
  assert.ok(estimate)
  assert.equal(estimate.effortCount, 1)
  assert.equal(estimate.criticalPowerWatts, 300)
  assert.equal(estimate.wPrimeKilojoules, 6)
  assert.ok(estimate.effect > 0.8 && estimate.effect < 1.5)
  assert.equal(calculateAnaerobicPowerEstimate([], 3_600, null), null)
  assert.equal(calculateAnaerobicPowerEstimate(watts, 3_600, null), null)
})

test('dilutes repeated anaerobic efforts across moving time', () => {
  const watts = Array.from({ length: 4 }, () => [
    ...Array<number>(20).fill(0),
    ...Array<number>(20).fill(600),
  ]).flat()
  const powerModel = { criticalPowerWatts: 300, wPrimeJoules: 6_000 }
  const short = calculateAnaerobicPowerEstimate(watts, 3_600, powerModel)
  const long = calculateAnaerobicPowerEstimate(watts, 8 * 3_600, powerModel)
  assert.ok(short)
  assert.ok(long)
  assert.equal(short.effortCount, long.effortCount)
  assert.ok(long.effect < short.effect / 2)
})

test('calculates missing run training effect from relative effort and upper-zone time', () => {
  assert.deepEqual(
    calculateActivityTrainingEffect({
      sport: 'run',
      distanceKm: 10,
      movingTimeS: 3_600,
      sufferScore: 30,
      garmin: null,
      calculatedIntensityFactor: { value: 0.9, source: 'pace' },
      calculatedExerciseLoad: { value: 81, source: 'pace' },
      anaerobicPowerEstimate: null,
      hrZones: [0, 0, 3_360, 240, 0],
      analysisRanges: [],
      swimPaceSPer100m: null,
      swimIntervals: [],
    }),
    {
      aerobic: 3,
      anaerobic: 2,
      evidence: {
        aerobic: { source: 'relative-effort', load: 30 },
        anaerobic: { source: 'heart-rate', seconds: 240 },
      },
    },
  )
})

test('calculates missing bike training effect from aerobic load and power intervals', () => {
  assert.deepEqual(
    calculateActivityTrainingEffect({
      sport: 'bike',
      distanceKm: 24,
      movingTimeS: 3_600,
      sufferScore: null,
      garmin: null,
      calculatedIntensityFactor: { value: 0.645, source: 'power' },
      calculatedExerciseLoad: { value: 40.4, source: 'power' },
      anaerobicPowerEstimate: {
        effect: 2.6,
        effortCount: 4,
        stimulus: 3,
        criticalPowerWatts: 250,
        wPrimeKilojoules: 10,
      },
      hrZones: [0, 0, 0, 120, 60],
      analysisRanges: [],
      swimPaceSPer100m: null,
      swimIntervals: [],
    }),
    {
      aerobic: 3,
      anaerobic: 2.6,
      evidence: {
        aerobic: { source: 'exercise-load', load: 40.4 },
        anaerobic: {
          source: 'power',
          effect: 2.6,
          effortCount: 4,
          stimulus: 3,
          criticalPowerWatts: 250,
          wPrimeKilojoules: 10,
        },
      },
    },
  )
})

test('calculates training effect for every non-endurance activity with observed load', () => {
  const kinds: Array<'strength' | 'walk' | 'yoga' | 'treatment' | 'sauna'> = [
    'strength',
    'walk',
    'yoga',
    'treatment',
    'sauna',
  ]
  for (const sport of kinds)
    assert.deepEqual(
      calculateActivityTrainingEffect({
        sport,
        distanceKm: 0,
        movingTimeS: 1_800,
        sufferScore: 11,
        garmin: null,
        calculatedIntensityFactor: { value: 0.7, source: 'heart-rate' },
        calculatedExerciseLoad: { value: 24.5, source: 'heart-rate' },
        anaerobicPowerEstimate: null,
        hrZones: [0, 0, 1_680, 120, 0],
        analysisRanges: [],
        swimPaceSPer100m: null,
        swimIntervals: [],
      }),
      {
        aerobic: 2,
        anaerobic: 1,
        evidence: {
          aerobic: { source: 'relative-effort', load: 11 },
          anaerobic: { source: 'heart-rate', seconds: 120 },
        },
      },
      sport,
    )
})

test('calculates recovery training effect for an Apple Watch sauna without Garmin', () => {
  assert.deepEqual(
    calculateActivityTrainingEffect({
      sport: 'sauna',
      distanceKm: 0,
      movingTimeS: 4_000,
      sufferScore: 8,
      garmin: null,
      calculatedIntensityFactor: null,
      calculatedExerciseLoad: null,
      anaerobicPowerEstimate: null,
      hrZones: [3_468, 437, 95, 0, 0],
      analysisRanges: [],
      swimPaceSPer100m: null,
      swimIntervals: [],
    }),
    {
      aerobic: 1.6,
      anaerobic: 0,
      evidence: {
        aerobic: { source: 'relative-effort', load: 8 },
        anaerobic: { source: 'heart-rate', seconds: 0 },
      },
    },
  )
})

test('requires measured load before calculating training effect', () => {
  assert.equal(
    calculateActivityTrainingEffect({
      sport: 'yoga',
      distanceKm: 0,
      movingTimeS: 1_800,
      sufferScore: null,
      garmin: null,
      calculatedIntensityFactor: null,
      calculatedExerciseLoad: null,
      anaerobicPowerEstimate: null,
      hrZones: null,
      analysisRanges: [],
      swimPaceSPer100m: null,
      swimIntervals: [],
    }),
    null,
  )
})

test('treats cached empty analysis arrays as a fetched activity detail', () => {
  assert.equal(hasFetchedActivityDetail(undefined), false)
  assert.equal(hasFetchedActivityDetail({ calories: null, laps: [], segmentEfforts: [] }), false)
  assert.equal(
    hasFetchedActivityDetail({
      description: null,
      fetchedAt: 1,
      calories: null,
      laps: [],
      segmentEfforts: [],
      splitsMetric: [],
      splitsStandard: [],
    }),
    true,
  )
})

test('projects a distance-aligned heart rate trace for route-less pool swims', () => {
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({
        name: 'Pool swim',
        sportType: 'Swim',
        distance: 75,
        movingTime: 90,
        elapsedTime: 90,
      }),
    },
    streams: {
      101: {
        time: [0, 30, 60, 90],
        latlng: [],
        altitude: [],
        distance: [0, 25, 50, 75],
        heartrate: [90, 110, 0, 130],
      },
    },
  }

  const activity = buildPayload(cache, null, null, '2026-06-01').details['101']

  assert.deepEqual(activity.route, [])
  assert.deepEqual(activity.heartRateTrace, [
    {
      distanceKm: 0,
      elapsedS: 0,
      heartRate: 90,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
    {
      distanceKm: 0.025,
      elapsedS: 30,
      heartRate: 110,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
    {
      distanceKm: 0.05,
      elapsedS: 60,
      heartRate: null,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
    {
      distanceKm: 0.075,
      elapsedS: 90,
      heartRate: 130,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
  ])
})

test('projects bounded Garmin thermal samples onto a route-less run', () => {
  const activity = ride({
    name: 'Thermal run',
    sportType: 'Run',
    distance: 3_000,
    movingTime: 300,
    elapsedTime: 300,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const cache: StravaRawCache = {
    version: 5,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time: [0, 60, 120, 300],
        latlng: [],
        altitude: [],
        distance: [0, 1_000, 2_000, 3_000],
        heartrate: [120, 130, 140, 150],
      },
    },
  }
  const garmin: GarminCache = {
    version: 7,
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      run: {
        id: 'run',
        name: 'Thermal run',
        sport: 'run',
        startDate: '2026-06-07T12:00:00Z',
        startDateLocal: '2026-06-07T08:00:00',
        distanceM: 3_000,
        movingTimeS: 300,
        elapsedTimeS: 300,
        sourceDevice: 'Forerunner 970',
        sourceFile: 'run.fit',
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Forerunner 970'),
      },
    },
    streams: {
      run: {
        time: [0, 60, 120],
        latlng: [],
        altitude: [],
        distance: [0, 1_000, 2_000],
        heatStrainIndex: [0, -1, 2],
        coreTemperatureC: [37.1, -1, 37.3],
        skinTemperatureC: [32.5, -1, 33],
      },
    },
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01').details['101']

  assert.equal(detail.device, 'garmin-forerunner-970')
  assert.deepEqual(
    detail.heartRateTrace.map(point => ({
      elapsedS: point.elapsedS,
      heartRate: point.heartRate,
      heatStrainIndex: point.heatStrainIndex,
      heatStrainSource: point.heatStrainSource,
      coreTemperatureC: point.coreTemperatureC,
      coreTemperatureSource: point.coreTemperatureSource,
      skinTemperatureC: point.skinTemperatureC,
      skinTemperatureSource: point.skinTemperatureSource,
    })),
    [
      {
        elapsedS: 0,
        heartRate: 120,
        heatStrainIndex: 0,
        heatStrainSource: 'core-fit',
        coreTemperatureC: 37.1,
        coreTemperatureSource: 'core-fit',
        skinTemperatureC: 32.5,
        skinTemperatureSource: 'core-fit',
      },
      {
        elapsedS: 60,
        heartRate: 130,
        heatStrainIndex: null,
        heatStrainSource: null,
        coreTemperatureC: null,
        coreTemperatureSource: null,
        skinTemperatureC: null,
        skinTemperatureSource: null,
      },
      {
        elapsedS: 120,
        heartRate: 140,
        heatStrainIndex: 2,
        heatStrainSource: 'core-fit',
        coreTemperatureC: 37.3,
        coreTemperatureSource: 'core-fit',
        skinTemperatureC: 33,
        skinTemperatureSource: 'core-fit',
      },
      {
        elapsedS: 300,
        heartRate: 150,
        heatStrainIndex: null,
        heatStrainSource: null,
        coreTemperatureC: null,
        coreTemperatureSource: null,
        skinTemperatureC: null,
        skinTemperatureSource: null,
      },
    ],
  )
})

test('projects Garmin thermal telemetry without GPS or heart rate onto a route-less walk', () => {
  const activity = ride({
    name: 'Thermal walk',
    sportType: 'Walk',
    distance: 1_000,
    movingTime: 120,
    elapsedTime: 120,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const cache: StravaRawCache = {
    version: 5,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
  }
  const garmin: GarminCache = {
    version: 7,
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      walk: {
        id: 'walk',
        name: 'Thermal walk',
        sport: 'walk',
        startDate: '2026-06-07T12:00:00Z',
        startDateLocal: '2026-06-07T08:00:00',
        distanceM: 1_000,
        movingTimeS: 120,
        elapsedTimeS: 120,
        sourceDevice: 'Forerunner 970',
        sourceFile: 'walk.fit',
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Forerunner 970'),
      },
    },
    streams: {
      walk: {
        time: [0, 60, 120],
        latlng: [],
        altitude: [],
        distance: [],
        heatStrainIndex: [0, 1, 2],
        coreTemperatureC: [37.1, 37.2, 37.3],
        skinTemperatureC: [32.5, 32.8, 33],
      },
    },
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01').details['101']

  assert.deepEqual(detail.route, [])
  assert.equal(detail.device, 'garmin-forerunner-970')
  assert.deepEqual(
    detail.heartRateTrace.map(point => ({
      elapsedS: point.elapsedS,
      heartRate: point.heartRate,
      heatStrainIndex: point.heatStrainIndex,
      heatStrainSource: point.heatStrainSource,
      coreTemperatureC: point.coreTemperatureC,
      coreTemperatureSource: point.coreTemperatureSource,
      skinTemperatureC: point.skinTemperatureC,
      skinTemperatureSource: point.skinTemperatureSource,
    })),
    [
      {
        elapsedS: 0,
        heartRate: null,
        heatStrainIndex: 0,
        heatStrainSource: 'core-fit',
        coreTemperatureC: 37.1,
        coreTemperatureSource: 'core-fit',
        skinTemperatureC: 32.5,
        skinTemperatureSource: 'core-fit',
      },
      {
        elapsedS: 60,
        heartRate: null,
        heatStrainIndex: 1,
        heatStrainSource: 'core-fit',
        coreTemperatureC: 37.2,
        coreTemperatureSource: 'core-fit',
        skinTemperatureC: 32.8,
        skinTemperatureSource: 'core-fit',
      },
      {
        elapsedS: 120,
        heartRate: null,
        heatStrainIndex: 2,
        heatStrainSource: 'core-fit',
        coreTemperatureC: 37.3,
        coreTemperatureSource: 'core-fit',
        skinTemperatureC: 33,
        skinTemperatureSource: 'core-fit',
      },
    ],
  )
})

test('projects a time-aligned heart rate trace for route-less strength training', () => {
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({
        name: 'Strength training',
        sportType: 'WeightTraining',
        distance: 0,
        movingTime: 90,
        elapsedTime: 90,
      }),
    },
    streams: {
      101: {
        time: [0, 30, 60, 90],
        latlng: [],
        altitude: [],
        distance: [],
        heartrate: [90, 110, 0, 130],
      },
    },
  }

  const activity = buildPayload(cache, null, null, '2026-06-01').details['101']

  assert.deepEqual(activity.route, [])
  assert.deepEqual(activity.heartRateTrace, [
    {
      distanceKm: 0,
      elapsedS: 0,
      heartRate: 90,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
    {
      distanceKm: 0,
      elapsedS: 30,
      heartRate: 110,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
    {
      distanceKm: 0,
      elapsedS: 60,
      heartRate: null,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
    {
      distanceKm: 0,
      elapsedS: 90,
      heartRate: 130,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    },
  ])
})

test('projects a time-aligned heart rate trace for route-less yoga', () => {
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({
        name: 'Yoga Session',
        sportType: 'Yoga',
        distance: 0,
        movingTime: 90,
        elapsedTime: 90,
      }),
    },
    streams: {
      101: {
        time: [0, 30, 60, 90],
        latlng: [],
        altitude: [],
        distance: [],
        heartrate: [90, 110, 0, 130],
      },
    },
  }

  const activity = buildPayload(cache, null, null, '2026-06-01').details['101']

  assert.deepEqual(activity.route, [])
  assert.deepEqual(
    activity.heartRateTrace.map(point => ({
      elapsedS: point.elapsedS,
      heartRate: point.heartRate,
      coreTemperatureSource: point.coreTemperatureSource,
    })),
    [
      { elapsedS: 0, heartRate: 90, coreTemperatureSource: null },
      { elapsedS: 30, heartRate: 110, coreTemperatureSource: null },
      { elapsedS: 60, heartRate: null, coreTemperatureSource: null },
      { elapsedS: 90, heartRate: 130, coreTemperatureSource: null },
    ],
  )
})

test('projects a time-aligned heart rate trace for route-less treatment', () => {
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({
        name: 'Post-race physio',
        sportType: 'PhysicalTherapy',
        distance: 0,
        movingTime: 90,
        elapsedTime: 90,
      }),
    },
    streams: {
      101: {
        time: [0, 30, 60, 90],
        latlng: [],
        altitude: [],
        distance: [],
        heartrate: [90, 110, 0, 130],
      },
    },
  }

  const activity = buildPayload(cache, null, null, '2026-06-01').details['101']

  assert.equal(activity.sport, 'treatment')
  assert.deepEqual(
    activity.heartRateTrace.map(point => ({
      elapsedS: point.elapsedS,
      heartRate: point.heartRate,
    })),
    [
      { elapsedS: 0, heartRate: 90 },
      { elapsedS: 30, heartRate: 110 },
      { elapsedS: 60, heartRate: null },
      { elapsedS: 90, heartRate: 130 },
    ],
  )
})

test('preserves the recorded peak when sampling route-less heart rate', () => {
  const sampleCount = 294
  const heartrate = Array.from({ length: sampleCount }, () => 100)
  heartrate[118] = 135
  heartrate[203] = 135
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({
        name: 'Strength training',
        sportType: 'WeightTraining',
        distance: 0,
        movingTime: sampleCount - 1,
        elapsedTime: sampleCount - 1,
      }),
    },
    streams: {
      101: {
        time: Array.from({ length: sampleCount }, (_, index) => index),
        latlng: [],
        altitude: [],
        distance: [],
        heartrate,
      },
    },
  }

  const trace = buildPayload(cache, null, null, '2026-06-01').details['101'].heartRateTrace

  assert.equal(Math.max(...trace.map(point => point.heartRate ?? 0)), 135)
  assert.ok(trace.length <= 140)
})

function ride(overrides: Partial<RawStravaActivity> = {}): RawStravaActivity {
  return {
    id: 101,
    name: 'Cadence training',
    sportType: 'Ride',
    distance: 61_400,
    movingTime: 7_200,
    elapsedTime: 7_500,
    totalElevationGain: 430,
    startDate: '2026-06-07T11:29:55Z',
    startDateLocal: '2026-06-07T07:29:55',
    averageSpeed: 8.52,
    ...overrides,
  }
}

test('merges an explicit virtual route by time and uses Garmin distance across payload and feed', () => {
  const activity = ride({
    id: 20037941355,
    distance: 66_800,
    movingTime: 10,
    elapsedTime: 10,
    averageWatts: 153,
    weightedAverageWatts: 170,
    averageHeartrate: 123,
    averageCadence: 74,
    deviceWatts: true,
    totalElevationGain: 0,
  })
  const time = Array.from({ length: 11 }, (_, index) => index)
  const cache: StravaRawCache = {
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: 0 },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: 0,
    activities: { [activity.id]: activity },
    streams: {
      [activity.id]: {
        time,
        latlng: [],
        altitude: [],
        distance: time.map(value => value * 6_680),
        watts: time.map(value => (value === 2 ? 0 : 100 + value)),
        heartrate: time.map(value => 120 + value),
        cadence: time.map(value => 70 + value),
      },
    },
  }
  const garminTime = [0, 2, 4, 6, 8, 10, 12, 14]
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {
      'connect:24239315396': {
        id: 'connect:24239315396',
        name: 'Virtual course',
        sport: 'bike',
        startDate: '2026-06-07T11:29:53Z',
        startDateLocal: '2026-06-07T07:29:53',
        distanceM: 28_000,
        movingTimeS: 10,
        elapsedTimeS: 14,
        sourceDevice: null,
        sourceFile: null,
        metrics: {
          ...emptyGarminMetrics(),
          totalAscentM: 1_244,
          totalDescentM: 1_058,
          normalizedPower: 179,
          trainingStressScore: 89.9,
          aerobicTrainingEffect: 0,
        },
        fueling: { ...emptyGarminFueling(), caloriesConsumed: 0, fluidMl: 700 },
      },
    },
    streams: {
      'connect:24239315396': {
        time: garminTime,
        latlng: garminTime.map(value => [45 + value / 10_000, 6 + value / 10_000]),
        altitude: garminTime.map(value => 758 + value),
        distance: [0, 0, 5_600, 11_200, 16_800, 22_400, 28_000, 28_000],
        watts: garminTime.map(() => 250),
        heartrate: garminTime.map(() => 0),
        cadence: garminTime.map(() => 0),
      },
    },
  }
  const entry = parseTrackingBlock(
    null,
    `activity: ${activity.id}\ngarmin: 24239315396\nvirtual: true`,
  )?.activity
  assert.ok(entry)
  cache.activityDetails = {
    [activity.id]: {
      calories: null,
      laps: [
        analysisRange('virtual-lap', 'Lap 1', {
          distance: 66_800,
          elapsedTime: 10,
          movingTime: 10,
          startIndex: 0,
          endIndex: 10,
        }),
      ],
      segmentEfforts: [],
      splitsMetric: [],
      splitsStandard: [],
    },
  }
  const before = structuredClone(cache)
  const payload = buildPayload(
    cache,
    null,
    garmin,
    undefined,
    null,
    250,
    undefined,
    undefined,
    null,
    190,
    170,
    cache.lastSync,
    [entry],
  )
  const detail = payload.details[String(activity.id)]
  assert.equal(detail.virtual, true)
  assert.equal(detail.distanceSource, 'garmin')
  assert.equal(detail.distanceKm, 28)
  assert.equal(detail.elevationM, 1_244)
  assert.equal(detail.descentM, 1_058)
  assert.equal(payload.totalKm, 28)
  assert.equal(payload.totals.find(total => total.sport === 'bike')?.distanceKm, 28)
  assert.equal(payload.days.flatMap(day => day.items)[0].distanceKm, 28)
  assert.equal(detail.avgWatts, 153)
  assert.equal(detail.npWatts, 170)
  assert.equal(detail.avgHr, 123)
  assert.equal(detail.avgCadence, 74)
  assert.equal(detail.garmin?.normalizedPower, 179)
  assert.equal(detail.garmin?.trainingStressScore, 89.9)
  assert.equal(detail.garmin?.aerobicTrainingEffect, 0)
  assert.equal(detail.garmin?.distanceDeltaM, -38_800)
  assert.equal(detail.fueling?.caloriesConsumed, 0)
  assert.equal(detail.fueling?.fluidMl, 700)
  assert.equal(detail.route[0].elapsedS, 0)
  assert.equal(detail.route[0].lat, 45.0002)
  assert.equal(detail.route[0].hr, 120)
  assert.equal(detail.route[0].w, 100)
  assert.equal(detail.route.find(point => point.elapsedS === 2)?.w, 0)
  assert.equal(detail.route.at(-1)?.elapsedS, 10)
  assert.equal(detail.route.at(-1)?.d, 28)
  assert.equal(detail.analysisRanges[0].distanceKm, 28)
  assert.equal(detail.analysisRanges[0].elevationGainM, null)
  assert.ok(detail.mapRoute.flat().length >= 2)
  assert.equal(detail.analyses.derived.environment, null)
  assert.deepEqual(cache, before)
  const trackedCache = applyActivityTracking(cache, garmin, [entry])
  assert.deepEqual(
    trackedCache?.streams?.[String(activity.id)].altitude,
    time.map(value => 760 + value),
  )
  const analytics = buildAnalytics(trackedCache, { activityDetails: payload.details, garmin })
  assert.equal(analytics.activities[0].distanceKm, 28)
  const feed = buildDataFeed(trackedCache, analytics, { activityDetails: payload.details, garmin })
  const rows: unknown[] = feed
    .trim()
    .split('\n')
    .map(line => JSON.parse(line))
  assert.equal(rows.filter(isRecord).find(row => row.kind === 'activity')?.distanceKm, 28)
  const serialized: unknown = JSON.parse(JSON.stringify(detail))
  assert.ok(isRecord(serialized))
  assert.equal(serialized.virtual, true)
  assert.equal(serialized.distanceSource, 'garmin')

  const missing = buildPayload(
    cache,
    null,
    garmin,
    undefined,
    null,
    250,
    undefined,
    undefined,
    null,
    190,
    170,
    cache.lastSync,
    [{ ...entry, garminActivityId: 999 }],
  ).details[String(activity.id)]
  assert.equal(missing.garmin, null)
  assert.equal(missing.distanceKm, 66.8)
  assert.equal(missing.distanceSource, undefined)
  assert.equal(missing.mapRoute.length, 0)
  assert.equal(missing.virtual, true)

  const regular = buildPayload(cache, null, garmin).details[String(activity.id)]
  assert.equal(regular.garmin, null)
  assert.equal(regular.virtual, false)
  assert.equal(regular.distanceKm, 66.8)
  const stream = cache.streams?.[String(activity.id)]
  assert.ok(stream)
  stream.time = [0, 1, 9, 10]
  stream.distance = [0, 6_680, 60_120, 66_800]
  stream.watts = [100, 101, 109, 110]
  stream.heartrate = [120, 121, 129, 130]
  stream.cadence = [70, 71, 79, 80]
  const gapped = buildPayload(
    cache,
    null,
    garmin,
    undefined,
    null,
    250,
    undefined,
    undefined,
    null,
    190,
    170,
    cache.lastSync,
    [entry],
  ).details[String(activity.id)]
  assert.equal(gapped.route.find(point => point.elapsedS === 4)?.w, 250)
  assert.equal(gapped.route.find(point => point.elapsedS === 4)?.hr, 0)
})

test('projects a calculated cycling performance condition when Garmin omits its native trace', () => {
  const sampleCount = 1_801
  const activity = ride({
    distance: 7_200,
    movingTime: sampleCount - 1,
    elapsedTime: sampleCount - 1,
    deviceWatts: true,
  })
  const time = Array.from({ length: sampleCount }, (_, index) => index)
  const cache: StravaRawCache = {
    version: 6,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time,
        latlng: time.map(index => [43.6 + index / 100_000, -79.4 + index / 100_000]),
        altitude: time.map(() => 80),
        distance: time.map(index => index * 10),
        watts: time.map(() => 165),
        heartrate: time.map(index => (index >= 600 && index < 900 ? 0 : 122)),
        cadence: time.map(() => 85),
      },
    },
  }
  const oura = {
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    days: {
      '2026-06-07': {
        date: '2026-06-07',
        readiness: null,
        sleepScore: null,
        hrv: null,
        rhr: 50,
        sleepDurationS: null,
        tempDeviationC: null,
        totalCalories: null,
        activeCalories: null,
      },
    },
  }

  const detail = buildPayload(
    cache,
    oura,
    null,
    '2026-06-01',
    null,
    250,
    null,
    'UTC',
    null,
    null,
    170,
  ).details['101']

  assert.deepEqual(detail.performanceConditionTrace, {
    source: 'garden-estimate',
    method: 'garden-cycling-performance-condition-v1',
    ftpWatts: 250,
    lactateThresholdHeartRateBpm: 170,
    restingHeartRateBpm: 50,
    windowSeconds: 360,
  })
  const firstCalculatedIndex = detail.route.findIndex(point => point.performanceCondition != null)
  assert.ok(firstCalculatedIndex > 0)
  assert.ok(
    detail.route.slice(firstCalculatedIndex).every(point => point.performanceCondition === 10),
  )
})

test('projects supported run, walk, and swim recording devices', () => {
  const cases: {
    sportType: string
    deviceName: string
    expected: 'apple-watch-ultra-3' | 'garmin-forerunner-970' | null
  }[] = [
    {
      sportType: 'Run',
      deviceName: 'Apple Watch Ultra 3 (GPS + Cellular)',
      expected: 'apple-watch-ultra-3',
    },
    { sportType: 'Walk', deviceName: 'Garmin Forerunner 970', expected: 'garmin-forerunner-970' },
    { sportType: 'Swim', deviceName: 'Apple Watch Ultra 3', expected: 'apple-watch-ultra-3' },
    { sportType: 'Run', deviceName: 'Apple Watch', expected: null },
    { sportType: 'Run', deviceName: 'Garmin Forerunner 965', expected: null },
    { sportType: 'Ride', deviceName: 'Garmin Forerunner 970', expected: null },
  ]

  for (const current of cases) {
    const activity = ride({ sportType: current.sportType, deviceName: current.deviceName })
    const cache: StravaRawCache = {
      version: 6,
      athleteId: 1,
      auth: { refreshToken: '', obtainedAt: Date.now() },
      lastSync: Date.parse('2026-06-08T00:00:00Z'),
      lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
      activities: { 101: activity },
    }

    assert.equal(
      buildPayload(cache, null, null, '2026-06-01').details['101'].device,
      current.expected,
    )
  }
})

test('projects exact provider reports and removes them when the refreshed description does', () => {
  const cache: StravaRawCache = {
    version: 5,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: 1 },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride() },
    activityDetails: {
      101: {
        description:
          '── pelotan.cc/uv UV Load™ Analysis ──\nUV Load™: 83 — High\nAvg. UV Index: 2.3\nTemperature: 20°C\nCloud Cover: 42%\n\n-- myWindsock Report --\nWeather Impact: 0.6%\nCdA: 0.324\nHeadwind: 56% · 5–15 mph\nLongest Headwind: 03:32:41\nAir Speed: 16.8 mph\n-- END --',
        fetchedAt: Date.parse('2026-06-08T00:00:00Z'),
        calories: null,
        laps: [],
        segmentEfforts: [],
        splitsMetric: [],
        splitsStandard: [],
      },
    },
  }
  const projected = buildPayload(cache, null, null, '2026-06-01').details['101']

  assert.equal(projected.analyses.native.pelotan?.score, 83)
  assert.equal(projected.analyses.native.pelotan?.averageCloudCoverPct, 42)
  assert.equal(projected.analyses.native.myWindsock?.weatherImpactPct, 0.6)
  assert.equal(projected.analyses.native.myWindsock?.cdaM2, 0.324)
  assert.equal(projected.analyses.native.myWindsock?.headwindPct, 56)
  assert.equal(projected.analyses.native.myWindsock?.longestHeadwindS, 12_761)
  assert.equal(projected.analyses.native.myWindsock?.airSpeedKph, 27.037)
  assert.doesNotMatch(JSON.stringify(projected), /Pelotan UV Load|myWindsock Report|strava\.com/)

  const detail = cache.activityDetails?.['101']
  assert.ok(detail)
  detail.description = null
  detail.fetchedAt = (detail.fetchedAt ?? 0) + 1
  const refreshed = buildPayload(cache, null, null, '2026-06-01').details['101']
  assert.equal(refreshed.analyses.native.pelotan, null)
  assert.equal(refreshed.analyses.native.myWindsock, null)
})

test('manual fueling overrides Garmin fueling by Strava activity ID', () => {
  const fueling = emptyGarminFueling('Edge 1050')
  fueling.caloriesConsumed = 200
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride() },
  }
  const garmin: GarminCache = {
    lastSync: Date.now(),
    activities: {
      edge: {
        id: 'edge',
        name: 'Cadence training',
        sport: 'bike',
        startDate: '2026-06-07T11:29:55Z',
        startDateLocal: '2026-06-07T07:29:55',
        distanceM: 61_400,
        movingTimeS: 7_200,
        elapsedTimeS: 7_500,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling,
      },
    },
  }
  const payload = buildPayload(cache, null, garmin, '2026-06-01')

  assert.equal(payload.details['101'].fueling?.source, 'garmin')
  applyManualFueling(payload, [{ date: '2026-06-08', activityId: 101, caloriesConsumed: 140 }])
  assert.equal(payload.details['101'].fueling?.source, 'garmin')
  applyManualFueling(payload, [{ date: '2026-06-07', activityId: 101, caloriesConsumed: 140 }])
  assert.deepEqual(payload.details['101'].fueling, {
    caloriesConsumed: 140,
    carbsConsumedG: null,
    fluidMl: null,
    carbsRecommendedG: null,
    fluidRecommendedMl: null,
    sweatLossMl: null,
    sourceDevice: null,
    sodiumLossMg: null,
    source: 'manual',
  })
})

test('projects Garmin intensity, training effect, and exercise load into activity details', () => {
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride() },
  }
  const metrics = emptyGarminMetrics()
  metrics.intensityFactor = 0.803
  metrics.aerobicTrainingEffect = 4.5
  metrics.anaerobicTrainingEffect = 2.7
  metrics.exerciseLoad = 301.7
  metrics.trainingEffectLabel = 'AEROBIC_BASE'
  metrics.aerobicTrainingEffectMessage = 'HIGHLY_IMPROVING_AEROBIC_ENDURANCE_10'
  metrics.anaerobicTrainingEffectMessage = 'MAINTAINING_FAST_FORCE_PRODUCTION_6'
  const garmin: GarminCache = {
    lastSync: Date.now(),
    activities: {
      edge: {
        id: 'edge',
        name: 'Cadence training',
        sport: 'bike',
        startDate: '2026-06-07T11:29:55Z',
        startDateLocal: '2026-06-07T07:29:55',
        distanceM: 61_400,
        movingTimeS: 7_200,
        elapsedTimeS: 7_500,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics,
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
  }

  const verification = buildPayload(cache, null, garmin, '2026-06-01').details['101'].garmin

  assert.equal(verification?.intensityFactor, 0.803)
  assert.equal(verification?.trainingEffectActivityId, 'edge')
  assert.equal(verification?.aerobicTrainingEffect, 4.5)
  assert.equal(verification?.anaerobicTrainingEffect, 2.7)
  assert.equal(verification?.exerciseLoad, 301.7)
  assert.equal(verification?.trainingEffectLabel, 'AEROBIC_BASE')
  assert.equal(verification?.aerobicTrainingEffectMessage, 'HIGHLY_IMPROVING_AEROBIC_ENDURANCE_10')
  assert.equal(verification?.anaerobicTrainingEffectMessage, 'MAINTAINING_FAST_FORCE_PRODUCTION_6')
})

test('uses an overlapping Garmin recording for training effect without replacing the primary match', () => {
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride() },
  }
  const trainingEffect = emptyGarminMetrics()
  trainingEffect.aerobicTrainingEffect = 4.3
  trainingEffect.anaerobicTrainingEffect = 0.7
  trainingEffect.exerciseLoad = 215
  trainingEffect.trainingEffectLabel = 'LACTATE_THRESHOLD'
  const garmin: GarminCache = {
    lastSync: Date.now(),
    activities: {
      primary: {
        id: 'primary',
        name: 'Cadence training',
        sport: 'bike',
        startDate: '2026-06-07T11:29:55Z',
        startDateLocal: '2026-06-07T07:29:55',
        distanceM: 61_400,
        movingTimeS: 7_200,
        elapsedTimeS: 7_500,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
      trainingEffect: {
        id: 'training-effect',
        name: 'Cadence training',
        sport: 'bike',
        startDate: '2026-06-07T11:30:25Z',
        startDateLocal: '2026-06-07T07:30:25',
        distanceM: 72_000,
        movingTimeS: 8_000,
        elapsedTimeS: 8_200,
        sourceDevice: 'Forerunner 965',
        sourceFile: null,
        metrics: trainingEffect,
        fueling: emptyGarminFueling('Forerunner 965'),
      },
    },
  }

  const verification = buildPayload(cache, null, garmin, '2026-06-01').details['101'].garmin

  assert.equal(verification?.activityId, 'primary')
  assert.equal(verification?.distanceM, 61_400)
  assert.equal(verification?.trainingEffectActivityId, 'training-effect')
  assert.equal(verification?.aerobicTrainingEffect, 4.3)
  assert.equal(verification?.anaerobicTrainingEffect, 0.7)
  assert.equal(verification?.exerciseLoad, 215)
  assert.equal(verification?.trainingEffectLabel, 'LACTATE_THRESHOLD')
})

test('manual strength attaches only to the matching strength activity and date', () => {
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride({ name: 'Strength', sportType: 'WeightTraining', distance: 0 }) },
  }
  const payload = buildPayload(cache, null, null, '2026-06-01')
  const entry = {
    date: '2026-06-07',
    activityId: 101,
    volumeKg: 816.512,
    totalSets: 15,
    totalReps: 90,
    exercises: [
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
  }

  applyManualStrength(payload, [{ ...entry, date: '2026-06-08' }])
  assert.equal(payload.details['101'].strength, null)
  applyManualStrength(payload, [entry])
  assert.deepEqual(payload.details['101'].strength, {
    volumeKg: 816.512,
    totalSets: 15,
    totalReps: 90,
    exercises: entry.exercises,
    source: 'manual',
  })
})

test('projects a manual sauna session with interval-matched Oura heart rate', () => {
  const payload = emptyPayload(1)
  applyManualSauna(
    payload,
    [
      {
        id: 8_202_608_231_830,
        stravaActivityId: null,
        garminActivityId: null,
        title: 'Untangle',
        date: '2026-08-23',
        time: '18:30',
        durationS: 4_500,
        temperatureC: 91.111,
        humidityPct: 11,
        cooldown: 'cold plunge',
        heatTrainingLoad: 7.7,
      },
    ],
    [
      { timestamp: '2026-08-23T22:25:00Z', bpm: 80, source: 'awake' },
      { timestamp: '2026-08-23T22:35:00Z', bpm: 110, source: 'awake' },
      { timestamp: '2026-08-23T23:30:00Z', bpm: 130, source: 'awake' },
      { timestamp: '2026-08-23T23:46:00Z', bpm: 100, source: 'rest' },
    ],
    'America/Toronto',
  )

  const detail = payload.details['8202608231830']
  assert.equal(detail.name, 'Untangle')
  assert.equal(detail.start, '2026-08-23T22:30:00.000Z')
  assert.equal(detail.avgHr, 120)
  assert.equal(detail.maxHr, 130)
  assert.deepEqual(
    detail.heartRateTrace.map(point => [point.elapsedS, point.heartRate]),
    [
      [300, 110],
      [3_600, 130],
    ],
  )
  assert.deepEqual(detail.sauna, {
    time: '18:30',
    temperatureC: 91.111,
    humidityPct: 11,
    cooldown: 'cold plunge',
    heatTrainingLoad: 7.7,
    heartRateSource: 'oura',
    source: 'manual',
  })
  assert.deepEqual(payload.days, [
    {
      date: '2026-08-23',
      durationS: 4_500,
      items: [{ id: 8_202_608_231_830, sport: 'sauna', distanceKm: 0, durationS: 4_500 }],
      dominant: 'sauna',
    },
  ])
  assert.equal(payload.totalCount, 1)
  assert.equal(payload.totalTimeS, 4_500)
})

test('attaches a manual sauna session to its canonical Strava activity', () => {
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({
        name: 'Sauna',
        sportType: 'Workout',
        distance: 0,
        movingTime: 4_500,
        elapsedTime: 4_500,
        totalElevationGain: 0,
        averageHeartrate: 101,
        maxHeartrate: 143,
      }),
    },
  }
  const payload = buildPayload(cache, null, null, '2026-06-01')
  const garminMetrics = emptyGarminMetrics()
  garminMetrics.aerobicTrainingEffect = 0.4
  garminMetrics.anaerobicTrainingEffect = 0
  garminMetrics.exerciseLoad = 7
  garminMetrics.trainingEffectLabel = 'RECOVERY'
  garminMetrics.aerobicTrainingEffectMessage = 'RECOVERY_5'
  garminMetrics.anaerobicTrainingEffectMessage = 'NO_ANAEROBIC_BENEFIT_0'
  const garminActivityId = 24_229_638_323
  const garmin: GarminCache = {
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      [`connect:${garminActivityId}`]: {
        id: `connect:${garminActivityId}`,
        name: 'Cardio',
        sport: null,
        startDate: '2026-06-07T11:30:00Z',
        startDateLocal: '2026-06-07T07:30:00',
        distanceM: null,
        movingTimeS: null,
        elapsedTimeS: 4_500,
        sourceDevice: 'Forerunner 970',
        sourceFile: null,
        metrics: garminMetrics,
        fueling: emptyGarminFueling('Forerunner 970'),
      },
    },
  }
  const weatherActivity: WeatherActivity = {
    activityId: 102,
    date: '2026-06-07',
    start: '2026-06-07T11:00:00Z',
    end: '2026-06-07T12:00:00Z',
    latitude: 43.6,
    longitude: -79.4,
    durationS: 3_600,
    windKph: 18,
    windDir: 'W',
    windDirDeg: 270,
    windGustKph: 29,
    averageRelativeHumidityPct: 68,
    relativeHumidityProvenance: {
      source: 'weatherkit',
      sourceKind: 'modeled',
      samplingMethod: 'route-hour',
      inputTimestamp: '2026-06-07T11:29:55Z',
      coveragePct: 80,
    },
    temperatureC: 27,
    source: 'weatherkit',
  }
  const weather: WeatherCache = {
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    current: null,
    attribution: null,
    uvCalibration: null,
    activities: { 102: weatherActivity },
    days: summarizeWeatherDays({ 102: weatherActivity }),
  }
  const entry = {
    id: 8_202_606_070_730,
    stravaActivityId: 101,
    garminActivityId,
    title: 'Untangle',
    date: '2026-06-07',
    time: '07:30',
    durationS: 4_500,
    temperatureC: 91.111,
    humidityPct: 11,
    cooldown: 'cold plunge' as const,
    heatTrainingLoad: 7.7,
  }

  applyManualSauna(
    payload,
    [entry],
    [{ timestamp: '2026-06-07T11:35:00Z', bpm: 110, source: 'awake' }],
    'America/Toronto',
    weather,
    garmin,
  )

  assert.deepEqual(Object.keys(payload.details), ['101'])
  assert.equal(payload.details['101'].sport, 'sauna')
  assert.equal(payload.details['101'].name, 'Untangle')
  assert.equal(payload.details['101'].start, '2026-06-07T11:29:55Z')
  assert.equal(payload.details['101'].avgHr, 101)
  assert.equal(payload.details['101'].maxHr, 143)
  assert.equal(payload.details['101'].ambientTemperatureC, 27)
  assert.equal(payload.details['101'].windKph, 18)
  assert.equal(payload.details['101'].windDir, 'W')
  assert.equal(payload.details['101'].windGustKph, 29)
  assert.equal(payload.details['101'].averageRelativeHumidityPct, 68)
  assert.deepEqual(payload.details['101'].relativeHumidityProvenance, {
    source: 'weatherkit',
    sourceKind: 'modeled',
    samplingMethod: 'route-hour',
    inputTimestamp: '2026-06-07T11:29:55Z',
    coveragePct: 80,
  })
  assert.equal(payload.details['101'].strength, null)
  assert.equal(
    payload.details['101'].garmin?.trainingEffectActivityId,
    `connect:${garminActivityId}`,
  )
  assert.equal(payload.details['101'].garmin?.aerobicTrainingEffect, 0.4)
  assert.equal(payload.details['101'].garmin?.anaerobicTrainingEffect, 0)
  assert.equal(payload.details['101'].garmin?.exerciseLoad, 7)
  assert.equal(payload.details['101'].garmin?.trainingEffectLabel, 'RECOVERY')
  assert.equal(payload.details['101'].garmin?.aerobicTrainingEffectMessage, 'RECOVERY_5')
  assert.equal(
    payload.details['101'].garmin?.anaerobicTrainingEffectMessage,
    'NO_ANAEROBIC_BENEFIT_0',
  )
  assert.deepEqual(payload.details['101'].sauna, {
    time: '07:30',
    temperatureC: 91.111,
    humidityPct: 11,
    cooldown: 'cold plunge',
    heatTrainingLoad: 7.7,
    heartRateSource: null,
    source: 'manual',
  })
  assert.deepEqual(payload.days.find(day => day.date === '2026-06-07')?.items, [
    { id: 101, sport: 'sauna', distanceKm: 0, durationS: 4_500 },
  ])
  assert.deepEqual(payload.strengthTotal, { count: 0, movingTimeS: 0 })
  assert.equal(payload.totalCount, 1)
  assert.equal(payload.totalTimeS, 4_500)

  const missing = emptyPayload(1)
  applyManualSauna(missing, [{ ...entry, stravaActivityId: 999 }], [], 'America/Toronto')
  assert.deepEqual(missing.details, {})
  assert.equal(missing.totalCount, 0)
})

test('emits geometry-preserved map data separately from compact telemetry route', () => {
  const latlng: [number, number][] = Array.from({ length: 1_000 }, (_, i) => [
    43 + i * 0.00001,
    -79 - i * 0.00002,
  ])
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({ averageTemp: 21, movingTime: latlng.length - 1, elapsedTime: latlng.length }),
    },
    streams: {
      101: {
        time: Array.from({ length: latlng.length }, (_, i) => i),
        latlng,
        altitude: Array.from({ length: latlng.length }, (_, i) => 80 + i / 100),
        distance: Array.from({ length: latlng.length }, (_, i) => i * 61.4),
        watts: Array.from({ length: latlng.length }, (_, i) => 140 + (i % 80)),
        heartrate: Array.from({ length: latlng.length }, (_, i) => 120 + (i % 35)),
        cadence: Array.from({ length: latlng.length }, (_, i) => 70 + (i % 20)),
      },
    },
  }

  const detail = buildPayload(cache, null, null, '2026-06-01').details['101']
  assert.ok(detail.route.length <= 141)
  assert.equal(detail.mapRoute.length, 1)
  assert.equal(detail.mapRoute[0].length, 2)
  assert.equal(Object.hasOwn(detail, 'mapBreaks'), false)
  assert.deepEqual(detail.mapRoute[0][0], { lat: 43, lng: -79, d: 0 })
  assert.deepEqual(detail.mapRoute[0].at(-1), { lat: 43.00999, lng: -79.01998, d: 61.3386 })
  assert.ok(detail.route.every(point => point.tempC === 21))
  assert.ok(detail.route.every(point => Number.isFinite(point.elapsedS)))
  assert.ok(detail.route.every(point => Number.isFinite(point.speedKph)))
})

test('aligns Garmin respiration and CORE samples onto the Strava route timeline', () => {
  const activity = ride({
    distance: 2_000,
    movingTime: 20,
    elapsedTime: 20,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const cache: StravaRawCache = {
    version: 3,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time: [0, 10, 20],
        latlng: [
          [43.64, -79.4],
          [43.65, -79.39],
          [43.66, -79.38],
        ],
        altitude: [80, 85, 90],
        distance: [0, 1_000, 2_000],
      },
    },
  }
  const garmin: GarminCache = {
    version: 5,
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      edge: {
        id: 'edge',
        name: 'Respiration ride',
        sport: 'bike',
        startDate: '2026-06-07T12:00:02Z',
        startDateLocal: '2026-06-07T08:00:02',
        distanceM: 2_000,
        movingTimeS: 20,
        elapsedTimeS: 20,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      edge: {
        time: [0, 10, 20],
        latlng: [
          [43.64, -79.4],
          [43.65, -79.39],
          [43.66, -79.38],
        ],
        altitude: [80, 85, 90],
        distance: [0, 1_000, 2_000],
        rightBalance: [47, 49, 52],
        stamina: [100, 80, 60],
        potentialStamina: [100, 92, 78],
        respiration: [18, 28, 38],
        heatStrainIndex: [0, 1.4, 3],
        coreTemperatureC: [37.16, 37.17, 37.19],
        skinTemperatureC: [33.4, 33.45, 33.5],
      },
    },
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01').details['101']
  assert.deepEqual(
    detail.route.map(point => point.rightPowerPct),
    [47, 48.6, 51.4],
  )
  assert.deepEqual(
    detail.route.map(point => point.stamina),
    [100, 84, 64],
  )
  assert.deepEqual(
    detail.route.map(point => point.potentialStamina),
    [100, 93.6, 80.8],
  )
  assert.deepEqual(detail.staminaTrace, {
    source: 'garmin',
    method: 'garmin-native',
    ftpWatts: null,
    maxHeartRateBpm: null,
  })
  assert.deepEqual(
    detail.route.map(point => point.resp),
    [18, 26, 36],
  )
  assert.deepEqual(
    detail.route.map(point => point.heatStrainIndex),
    [0, 1.1, 2.7],
  )
  assert.deepEqual(
    detail.route.map(point => point.coreTemperatureC),
    [37.16, 37.17, 37.19],
  )
  assert.deepEqual(
    detail.route.map(point => point.skinTemperatureC),
    [33.4, 33.44, 33.49],
  )
  assert.ok(
    detail.route.every(
      point =>
        point.heatStrainSource === 'core-fit' &&
        point.coreTemperatureSource === 'core-fit' &&
        point.skinTemperatureSource === 'core-fit',
    ),
  )
})

test('projects native Forerunner running dynamics and run/walk intervals', () => {
  const activity = ride({
    name: 'Easy Miles',
    sportType: 'Run',
    distance: 2_000,
    movingTime: 20,
    elapsedTime: 20,
    startDate: '2026-09-03T19:38:00Z',
    startDateLocal: '2026-09-03T15:38:00',
  })
  const cache: StravaRawCache = {
    version: 5,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-09-03T20:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time: [0, 10, 20],
        latlng: [
          [43.64, -79.4],
          [43.65, -79.39],
          [43.66, -79.38],
        ],
        altitude: [80, 85, 90],
        distance: [0, 1_000, 2_000],
      },
    },
  }
  const runWalk: GarminRunWalkData = {
    source: 'garmin',
    elapsedTimeS: 20,
    runTimeS: 17,
    walkTimeS: 2,
    idleTimeS: 1,
    segments: [
      { state: 'run', startElapsedS: 0, endElapsedS: 10 },
      { state: 'walk', startElapsedS: 10, endElapsedS: 12 },
      { state: 'idle', startElapsedS: 12, endElapsedS: 13 },
      { state: 'run', startElapsedS: 13, endElapsedS: 20 },
    ],
  }
  const garmin: GarminCache = {
    version: 13,
    lastSync: Date.parse('2026-09-03T20:00:00Z'),
    activities: {
      run: {
        id: 'run',
        name: 'Easy Miles',
        sport: 'run',
        startDate: activity.startDate,
        startDateLocal: activity.startDateLocal,
        distanceM: 2_000,
        movingTimeS: 20,
        elapsedTimeS: 20,
        sourceDevice: 'Forerunner 970',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
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
        fueling: emptyGarminFueling('Forerunner 970'),
      },
    },
    streams: {
      run: {
        time: [0, 10, 20],
        latlng: [],
        altitude: [0, 0, 0],
        distance: [0, 1_000, 2_000],
        stamina: [100, 80, 60],
        potentialStamina: [100, 92, 78],
        performanceCondition: [null, -4, -10],
        strideLengthCm: [null, 108, 77],
        verticalRatioPct: [null, 11.3, 16.1],
        verticalOscillationCm: [null, 12.4, 12.5],
        groundContactBalanceLeftPct: [null, 49.3, 50.1],
        groundContactTimeMs: [null, 246.5, 248],
        stepSpeedLossMps: [null, 0.079, 0.07],
        stepSpeedLossPct: [null, 2.78, 2.5],
        impactLoadFactor: [0, 1, 0.96],
      },
    },
    runWalks: { run: runWalk },
  }

  const detail = buildPayload(cache, null, garmin, '2026-09-01').details['101']
  assert.equal(detail.device, 'garmin-forerunner-970')
  assert.equal(detail.runWalk, runWalk)
  assert.deepEqual(detail.garmin?.runningDynamics, garmin.activities.run.runningDynamics)
  assert.deepEqual(
    detail.route.map(point => point.performanceCondition),
    [null, -4, -10],
  )
  assert.deepEqual(
    detail.route.map(point => point.strideLengthM),
    [null, 1.08, 0.77],
  )
  assert.deepEqual(
    detail.route.map(point => point.verticalRatioPct),
    [null, 11.3, 16.1],
  )
  assert.deepEqual(
    detail.route.map(point => point.groundContactBalanceLeftPct),
    [null, 49.3, 50.1],
  )
  assert.deepEqual(
    detail.route.map(point => point.stepSpeedLossMps),
    [null, 0.079, 0.07],
  )
  assert.deepEqual(
    detail.route.map(point => point.stepSpeedLossPct),
    [null, 2.78, 2.5],
  )
  assert.deepEqual(
    detail.route.map(point => point.impactLoadFactor),
    [0, 1, 0.96],
  )
  assert.deepEqual(detail.performanceConditionTrace, { source: 'garmin', method: 'garmin-native' })
  assert.deepEqual(
    detail.route.map(point => point.stamina),
    [100, 80, 60],
  )
  assert.deepEqual(
    detail.route.map(point => point.potentialStamina),
    [100, 92, 78],
  )
  assert.deepEqual(detail.staminaTrace, {
    source: 'garmin',
    method: 'garmin-native',
    ftpWatts: null,
    maxHeartRateBpm: null,
  })
})

test('projects Garmin-native performance condition onto a cycling route', () => {
  const activity = ride({
    name: 'Garmin ride',
    distance: 2_000,
    movingTime: 20,
    elapsedTime: 20,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const cache: StravaRawCache = {
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time: [0, 10, 20],
        latlng: [
          [43.64, -79.4],
          [43.65, -79.39],
          [43.66, -79.38],
        ],
        altitude: [80, 85, 90],
        distance: [0, 1_000, 2_000],
        watts: [180, 200, 220],
        heartrate: [125, 130, 135],
      },
    },
  }
  const garmin: GarminCache = {
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      ride: {
        id: 'ride',
        name: 'Garmin ride',
        sport: 'bike',
        startDate: activity.startDate,
        startDateLocal: activity.startDateLocal,
        distanceM: 2_000,
        movingTimeS: 20,
        elapsedTimeS: 20,
        sourceDevice: 'Forerunner 970',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Forerunner 970'),
      },
    },
    streams: {
      ride: {
        time: [0, 10, 20],
        latlng: [],
        altitude: [0, 0, 0],
        distance: [0, 1_000, 2_000],
        performanceCondition: [null, 2, 4],
      },
    },
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01').details['101']

  assert.deepEqual(
    detail.route.map(point => point.performanceCondition),
    [null, 2, 4],
  )
  assert.deepEqual(detail.performanceConditionTrace, { source: 'garmin', method: 'garmin-native' })
})

test('projects Garmin FIT gear states and cycling dynamics onto ride distance', () => {
  const activity = ride({
    distance: 2_000,
    movingTime: 20,
    elapsedTime: 20,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const cache: StravaRawCache = {
    version: 3,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
  }
  const garmin: GarminCache = {
    version: 7,
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      edge: {
        id: 'edge',
        name: 'Electronic shifting ride',
        sport: 'bike',
        startDate: '2026-06-07T12:00:02Z',
        startDateLocal: '2026-06-07T08:00:02',
        distanceM: 2_000,
        movingTimeS: 20,
        elapsedTimeS: 20,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      edge: { time: [0, 10, 20], latlng: [], altitude: [0, 0, 0], distance: [0, 1_000, 2_000] },
    },
    gearShifts: {
      edge: [
        {
          timestamp: '2026-06-07T12:00:02.000Z',
          frontGearNum: 2,
          frontTeeth: 52,
          rearGearNum: 3,
          rearTeeth: 27,
        },
        {
          timestamp: '2026-06-07T12:00:12.000Z',
          frontGearNum: 2,
          frontTeeth: 52,
          rearGearNum: 6,
          rearTeeth: 19,
        },
        {
          timestamp: '2026-06-07T12:00:22.000Z',
          frontGearNum: 1,
          frontTeeth: 36,
          rearGearNum: 8,
          rearTeeth: 15,
        },
      ],
    },
    cyclingDynamics: {
      edge: {
        time: [0, 10, 20],
        distance: [0, 1_000, 2_000],
        leftPedalSmoothness: [21, 22, null],
        rightPedalSmoothness: [23, 24, 25],
        leftTorqueEffectiveness: [70, 74, null],
        rightTorqueEffectiveness: [72, 76, 78],
        leftPowerPhaseStart: [350, 352, 0],
        leftPowerPhaseEnd: [190, 192, 194],
        rightPowerPhaseStart: [348, 350, 352],
        rightPowerPhaseEnd: [198, 200, 202],
        positionChanges: [
          { elapsedS: 0, distanceM: 0, position: 'seated' },
          { elapsedS: 5, distanceM: 500, position: 'standing' },
          { elapsedS: 15, distanceM: 1_500, position: 'seated' },
        ],
        seatedTimeS: 10,
        standingTimeS: 10,
      },
    },
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01').details['101']
  assert.equal(detail.computer, 'garmin')
  assert.deepEqual(detail.gearShifts, [
    { elapsedS: 2, distanceKm: 0, frontGearNum: 2, frontTeeth: 52, rearGearNum: 3, rearTeeth: 27 },
    { elapsedS: 12, distanceKm: 1, frontGearNum: 2, frontTeeth: 52, rearGearNum: 6, rearTeeth: 19 },
    { elapsedS: 20, distanceKm: 2, frontGearNum: 1, frontTeeth: 36, rearGearNum: 8, rearTeeth: 15 },
  ])
  assert.deepEqual(detail.cyclingDynamics, {
    elapsedS: [2, 12, 22],
    distanceKm: [0, 1, 2],
    leftPedalSmoothness: [21, 22, null],
    rightPedalSmoothness: [23, 24, 25],
    leftTorqueEffectiveness: [70, 74, null],
    rightTorqueEffectiveness: [72, 76, 78],
    leftPowerPhaseStart: [350, 352, 0],
    leftPowerPhaseEnd: [190, 192, 194],
    rightPowerPhaseStart: [348, 350, 352],
    rightPowerPhaseEnd: [198, 200, 202],
    positionChanges: [
      { elapsedS: 2, distanceKm: 0, position: 'seated' },
      { elapsedS: 7, distanceKm: 0.5, position: 'standing' },
      { elapsedS: 17, distanceKm: 1.5, position: 'seated' },
    ],
    seatedTimeS: 10,
    standingTimeS: 10,
  })
})

test('projects Wahoo balance, respiration, shifting, and cycling dynamics onto a ride', () => {
  const activity = ride({
    distance: 2_000,
    movingTime: 20,
    elapsedTime: 20,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const cache: StravaRawCache = {
    version: 4,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time: [0, 10, 20],
        latlng: [
          [43.64, -79.4],
          [43.65, -79.39],
          [43.66, -79.38],
        ],
        altitude: [80, 85, 90],
        distance: [0, 1_000, 2_000],
      },
    },
  }
  const wahoo: WahooCache = {
    version: 4,
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      'wahoo:1': {
        id: 'wahoo:1',
        workoutId: 1,
        workoutTypeId: 15,
        workoutUpdatedAt: '2026-06-07T13:00:00Z',
        name: 'Wahoo ride',
        sport: 'bike',
        startDate: '2026-06-07T12:00:02Z',
        startDateLocal: '2026-06-07T08:00:02',
        distanceM: 2_000,
        movingTimeS: 20,
        elapsedTimeS: 20,
        sourceDevice: 'ELEMNT BOLT',
        sourceFile: {
          url: 'https://cdn.wahoofitness.com/ride.fit',
          sha256: 'a'.repeat(64),
          byteLength: 4_000,
          profileVersion: '21.171',
        },
        sweatLoss: { fluidMl: 900, sodiumMg: 740 },
        metrics: emptyWahooMetrics(),
        summary: {
          id: 2,
          name: 'Wahoo ride',
          timeZone: 'America/Toronto',
          manual: false,
          edited: false,
          fitnessAppId: 1,
          durationPausedS: 0,
          createdAt: '2026-06-07T13:00:00Z',
          updatedAt: '2026-06-07T13:00:00Z',
        },
      },
    },
    streams: {
      'wahoo:1': {
        timestamps: ['2026-06-07T12:00:02Z', '2026-06-07T12:00:12Z', '2026-06-07T12:00:22Z'],
        time: [0, 10, 20],
        latlng: [null, null, null],
        altitude: [80, 85, 90],
        distance: [0, 1_000, 2_000],
        watts: [200, 220, 240],
        rightBalance: [47, 49, 52],
        heartrate: [130, 140, 150],
        cadence: [80, 85, 90],
        speed: [8, 9, 10],
        temperature: [25, 25, 26],
        respiration: [18, 28, 38],
        muscleOxygenPercent: [62, 60, 58],
        totalHemoglobinConcentration: [12.1, 12.2, 12.3],
        heatStrainIndex: [0, 1.4, 3],
        coreTemperatureC: [37.16, 37.17, 37.19],
        skinTemperatureC: [33.4, 33.45, 33.5],
        minuteVentilation: [40, 50, 60],
        tidalVolume: [1.2, 1.3, 1.4],
        fluidLossMl: [0, 400, 900],
        sodiumLossMg: [0, 300, 740],
      },
    },
    gearShifts: {
      'wahoo:1': [
        {
          timestamp: '2026-06-07T12:00:02Z',
          frontGearNum: 2,
          frontTeeth: 54,
          rearGearNum: 5,
          rearTeeth: 21,
        },
        {
          timestamp: '2026-06-07T12:00:12Z',
          frontGearNum: 2,
          frontTeeth: 54,
          rearGearNum: 6,
          rearTeeth: 19,
        },
      ],
    },
    cyclingDynamics: {
      'wahoo:1': {
        time: [0, 10, 20],
        distance: [0, 1_000, 2_000],
        leftPedalSmoothness: [21, 22, 23],
        rightPedalSmoothness: [23, 24, 25],
        leftTorqueEffectiveness: [70, 74, 76],
        rightTorqueEffectiveness: [72, 76, 78],
        leftPowerPhaseStart: [null, null, null],
        leftPowerPhaseEnd: [null, null, null],
        rightPowerPhaseStart: [null, null, null],
        rightPowerPhaseEnd: [null, null, null],
        positionChanges: [],
        seatedTimeS: null,
        standingTimeS: null,
      },
    },
    summitSegments: {
      'wahoo:1': [
        {
          feature: 'summit-segment',
          uuid: 'WAHOO_ON_ROUTE_CLIMB-route',
          name: '1/2',
          startDate: '2026-06-07T12:00:02Z',
          endDate: '2026-06-07T12:00:12Z',
          distanceM: 1_000,
          durationS: 10,
          elevationGainM: 50,
          avgGradePct: 5,
          avgSpeedMps: 100,
          avgHeartRate: 140,
          avgPower: 220,
          avgCadence: 85,
        },
      ],
    },
  }

  const fueling = emptyGarminFueling('Edge 1050')
  fueling.caloriesConsumed = 260
  fueling.fluidMl = 1_200
  fueling.sweatLossMl = 800
  const garmin: GarminCache = {
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      edge: {
        id: 'edge',
        name: 'Wahoo ride',
        sport: 'bike',
        startDate: '2026-06-07T12:00:00Z',
        startDateLocal: '2026-06-07T08:00:00',
        distanceM: 2_000,
        movingTimeS: 20,
        elapsedTimeS: 20,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling,
      },
    },
    climbs: {
      edge: [
        {
          startDate: '2026-06-07T12:00:12Z',
          endDate: '2026-06-07T12:00:22Z',
          distanceM: 500,
          durationS: 10,
          movingTimeS: 10,
          elapsedTimeS: 10,
          elevationGainM: 25,
          elevationLossM: 0,
          startElevationM: 100,
          avgGradePct: 5,
          maxGradePct: 8,
          avgSpeedMps: 50,
          avgHeartRate: 150,
          maxHeartRate: 160,
          avgPower: 300,
          normalizedPower: 310,
          maxPower: 400,
          avgCadence: 90,
          difficulty: 'MODERATE',
        },
      ],
    },
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01', null, 230, null, 'UTC', wahoo, 196)
    .details['101']
  assert.equal(detail.computer, 'wahoo')
  assert.deepEqual(
    detail.analysisRanges.filter(range => range.kind === 'climb'),
    [
      {
        kind: 'climb',
        source: 'wahoo-summit-segment',
        id: 'wahoo-summit-segment:1:2026-06-07T12:00:02Z',
        label: 'Summit 1/2',
        startElapsedS: 0,
        endElapsedS: 10,
        startDistanceKm: 0,
        endDistanceKm: 1,
        durationS: 10,
        distanceKm: 1,
        elevationGainM: 50,
        averageSpeedKph: 360,
        averageHeartRate: 140,
        averageWatts: 220,
        averageCadence: 85,
      },
    ],
  )
  assert.deepEqual(detail.bestEfforts?.climbs, [
    {
      source: 'wahoo-summit-segment',
      name: 'Summit 1/2',
      durationS: 10,
      distanceM: 1_000,
      elevationGainM: 50,
      averageGradePct: 5,
      averageSpeedKph: 360,
      averageHeartRate: 140,
      averageWatts: 220,
      wattsPerKg: null,
      vamMPerHour: 18_000,
    },
  ])
  const withoutSummit = buildPayload(
    cache,
    null,
    garmin,
    '2026-06-01',
    null,
    230,
    null,
    'UTC',
    { ...wahoo, summitSegments: { 'wahoo:1': [] } },
    196,
  ).details['101']
  assert.deepEqual(
    withoutSummit.analysisRanges.filter(range => range.kind === 'climb'),
    [],
  )
  assert.deepEqual(withoutSummit.bestEfforts?.climbs, [])
  assert.deepEqual(detail.staminaTrace, {
    source: 'garden-estimate',
    method: 'garden-stamina-v1',
    ftpWatts: 230,
    maxHeartRateBpm: 196,
  })
  assert.ok(detail.route.every(point => point.stamina != null && point.potentialStamina != null))
  assert.deepEqual(
    detail.route.map(point => point.rightPowerPct),
    [47, 48.6, 51.4],
  )
  assert.deepEqual(
    detail.route.map(point => point.resp),
    [18, 26, 36],
  )
  assert.deepEqual(
    detail.route.map(point => point.muscleOxygenPct),
    [62, 60.4, 58.4],
  )
  assert.deepEqual(
    detail.route.map(point => point.coreTemperatureC),
    [37.16, 37.17, 37.19],
  )
  assert.deepEqual(detail.fueling, {
    caloriesConsumed: 260,
    carbsConsumedG: null,
    fluidMl: 1_200,
    carbsRecommendedG: null,
    fluidRecommendedMl: null,
    sweatLossMl: 900,
    sourceDevice: 'Edge 1050',
    sodiumLossMg: 740,
    source: 'garmin+wahoo',
  })
  assert.deepEqual(detail.gearShifts, [
    { elapsedS: 2, distanceKm: 0, frontGearNum: 2, frontTeeth: 54, rearGearNum: 5, rearTeeth: 21 },
    { elapsedS: 12, distanceKm: 1, frontGearNum: 2, frontTeeth: 54, rearGearNum: 6, rearTeeth: 19 },
  ])
  assert.deepEqual(detail.cyclingDynamics?.leftPedalSmoothness, [21, 22, 23])
  assert.deepEqual(detail.cyclingDynamics?.rightTorqueEffectiveness, [72, 76, 78])
  const sourceStream = cache.streams?.['101']
  assert.ok(sourceStream)
  const virtualGarmin: GarminCache = {
    ...garmin,
    activities: { 'connect:123': { ...garmin.activities.edge, id: 'connect:123', distanceM: 200 } },
    streams: { 'connect:123': { ...sourceStream, distance: [0, 100, 200] } },
    climbs: { 'connect:123': garmin.climbs?.edge ?? [] },
  }
  const virtual = buildPayload(
    cache,
    null,
    virtualGarmin,
    '2026-06-01',
    null,
    230,
    null,
    'UTC',
    wahoo,
    196,
    170,
    cache.lastSync,
    [{ activityId: 101, garminActivityId: 123, virtual: true }],
  ).details['101']
  assert.equal(virtual.distanceKm, 0.2)
  assert.equal(virtual.computer, 'wahoo')
  assert.deepEqual(
    virtual.gearShifts.map(shift => shift.distanceKm),
    [0.02, 0.12],
  )
  assert.deepEqual(virtual.cyclingDynamics?.distanceKm, [0.02, 0.12, 0.2])
  assert.deepEqual(virtual.cyclingDynamics?.leftPedalSmoothness, [21, 22, 23])
  assert.equal(virtual.fueling?.source, 'garmin+wahoo')
  assert.equal(virtual.route[0].rightPowerPct, 47)
  assert.ok(
    virtual.analysisRanges.some(
      range => range.kind === 'climb' && range.source === 'garmin-climbpro',
    ),
  )
})

test('projects Wahoo respiration at 500 ms from Garmin-calibrated heart rate', () => {
  const activity = ride({
    distance: 2_000,
    movingTime: 2,
    elapsedTime: 2,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const cache: StravaRawCache = {
    version: 4,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time: [0, 0.5, 1, 1.5, 2],
        latlng: [
          [43.64, -79.4],
          [43.645, -79.395],
          [43.65, -79.39],
          [43.655, -79.385],
          [43.66, -79.38],
        ],
        altitude: [80, 82, 85, 87, 90],
        distance: [0, 500, 1_000, 1_500, 2_000],
      },
    },
  }
  const calibrationHeartRate = Array.from({ length: 61 }, (_, index) => 100 + index)
  const calibrationRespiration = calibrationHeartRate.map(value => value * 0.25 - 5)
  const garmin: GarminCache = {
    version: 5,
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      calibration: {
        id: 'calibration',
        name: 'Garmin calibration ride',
        sport: 'bike',
        startDate: '2026-05-01T12:00:00Z',
        startDateLocal: '2026-05-01T08:00:00',
        distanceM: 61_000,
        movingTimeS: 60,
        elapsedTimeS: 60,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      calibration: {
        time: Array.from({ length: 61 }, (_, index) => index),
        latlng: [],
        altitude: [],
        distance: [],
        heartrate: calibrationHeartRate,
        respiration: calibrationRespiration,
      },
    },
  }
  const empty = [null, null, null]
  const wahoo: WahooCache = {
    version: 4,
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      'wahoo:1': {
        id: 'wahoo:1',
        workoutId: 1,
        workoutTypeId: 15,
        workoutUpdatedAt: '2026-06-07T13:00:00Z',
        name: 'Wahoo ride',
        sport: 'bike',
        startDate: activity.startDate,
        startDateLocal: activity.startDateLocal,
        distanceM: activity.distance,
        movingTimeS: activity.movingTime,
        elapsedTimeS: activity.elapsedTime,
        sourceDevice: 'ELEMNT BOLT',
        sourceFile: {
          url: 'https://cdn.wahoofitness.com/ride.fit',
          sha256: 'a'.repeat(64),
          byteLength: 4_000,
          profileVersion: '21.171',
        },
        sweatLoss: { fluidMl: null, sodiumMg: null },
        metrics: emptyWahooMetrics(),
        summary: {
          id: 2,
          name: 'Wahoo ride',
          timeZone: 'America/Toronto',
          manual: false,
          edited: false,
          fitnessAppId: 1,
          durationPausedS: 0,
          createdAt: '2026-06-07T13:00:00Z',
          updatedAt: '2026-06-07T13:00:00Z',
        },
      },
    },
    streams: {
      'wahoo:1': {
        timestamps: [activity.startDate, '2026-06-07T12:00:01Z', '2026-06-07T12:00:02Z'],
        time: [0, 1, 2],
        latlng: empty,
        altitude: empty,
        distance: [0, 1_000, 2_000],
        watts: empty,
        rightBalance: empty,
        heartrate: [120, 140, 160],
        cadence: empty,
        speed: empty,
        temperature: empty,
        respiration: empty,
        muscleOxygenPercent: empty,
        totalHemoglobinConcentration: empty,
        heatStrainIndex: empty,
        coreTemperatureC: empty,
        skinTemperatureC: empty,
        minuteVentilation: empty,
        tidalVolume: empty,
        fluidLossMl: empty,
        sodiumLossMg: empty,
      },
    },
    gearShifts: { 'wahoo:1': [] },
    cyclingDynamics: {},
    summitSegments: { 'wahoo:1': [] },
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01', null, null, null, 'UTC', wahoo)
    .details['101']

  assert.deepEqual(
    detail.route.map(point => point.resp),
    [25, 27.5, 30, 32.5, 35],
  )
})

function timedRideCache(increments: (i: number) => number, n = 100): StravaRawCache {
  const distance = [0]
  for (let i = 1; i < n; i++) distance.push(distance[i - 1] + increments(i))
  return {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride({ distance: distance[n - 1], movingTime: n - 1, elapsedTime: n }) },
    streams: {
      101: {
        time: Array.from({ length: n }, (_, i) => i),
        latlng: Array.from({ length: n }, (_, i) => [43 + i * 0.0001, -79] as [number, number]),
        altitude: Array.from({ length: n }, () => 80),
        distance,
      },
    },
  }
}

test('leaves anaerobic power effect unavailable without a fitted power model', () => {
  const watts = [
    ...Array<number>(20).fill(0),
    ...Array<number>(9).fill(600),
    ...Array<number>(20).fill(0),
    ...Array<number>(20).fill(600),
    ...Array<number>(20).fill(0),
    ...Array<number>(121).fill(600),
    ...Array<number>(20).fill(0),
  ]
  const cache = timedRideCache(() => 8, watts.length)
  const streams = cache.streams?.['101']
  assert.ok(streams)
  streams.watts = watts

  const detail = buildPayload(cache, null, null, '2026-06-01', null, 300).details['101']
  assert.equal(detail.anaerobicPowerEstimate, null)
})

test('derives max speed from the timed distance stream', () => {
  const surge = new Map([
    [50, 10],
    [51, 12],
    [52, 14],
    [53, 16],
    [54, 16],
    [55, 16],
    [56, 14],
    [57, 12],
    [58, 10],
  ])
  const cache = timedRideCache(i => surge.get(i) ?? 8)
  const detail = buildPayload(cache, null, null, '2026-06-01').details['101']
  assert.equal(detail.maxSpeedKph, 57.6)
})

test('rejects GPS teleports when deriving max speed', () => {
  const glitch = new Map([
    [50, 0],
    [51, 0],
    [52, 0],
    [53, 52],
    [54, 52],
    [55, 52],
  ])
  const cache = timedRideCache(i => glitch.get(i) ?? 8)
  const detail = buildPayload(cache, null, null, '2026-06-01').details['101']
  assert.equal(detail.maxSpeedKph, 30)
})

function analysisRange(
  id: string,
  name: string,
  overrides: Partial<RawStravaAnalysisRange> = {},
): RawStravaAnalysisRange {
  return {
    id,
    name,
    elapsedTime: 10,
    movingTime: 10,
    startDate: null,
    distance: 1_234,
    startIndex: null,
    endIndex: null,
    totalElevationGain: null,
    averageSpeed: null,
    averageHeartrate: null,
    averageWatts: null,
    averageCadence: null,
    ...overrides,
  }
}

test('emits exact elapsed time and bounded local speed at forced analysis boundaries', () => {
  const points = 201
  const time = Array.from({ length: points }, (_, index) => index + (index >= 50 ? 10 : 0))
  const distance = [0]
  for (let index = 1; index < points; index++) {
    const increment = index >= 40 && index <= 50 ? 0 : index === 100 ? 1_000 : 5
    distance.push(distance[index - 1] + increment)
  }
  const activity = ride({
    sportType: 'Run',
    distance: distance.at(-1) ?? 0,
    movingTime: points - 1,
    elapsedTime: time.at(-1) ?? points - 1,
  })
  const cache: StravaRawCache = {
    version: 3,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    activityDetails: {
      101: {
        calories: 500,
        laps: [
          analysisRange('lap-2', 'Lap 2', {
            elapsedTime: 20,
            startIndex: 45,
            endIndex: 55,
            totalElevationGain: 12.5,
            averageSpeed: 8,
            averageHeartrate: 151.2,
            averageWatts: 244.8,
            averageCadence: 89.4,
          }),
        ],
        segmentEfforts: [
          analysisRange('segment-9', 'Boardwalk east', { startDate: '2026-06-07T11:31:45Z' }),
        ],
        splitsMetric: [
          {
            split: 1,
            distance: 1_000,
            elapsedTime: 305,
            movingTime: 300,
            averageSpeed: 10 / 3,
            elevationDifference: 4.2,
            paceZone: 2,
          },
        ],
        splitsStandard: [
          {
            split: 1,
            distance: 1_609.344,
            elapsedTime: 495,
            movingTime: 480,
            averageSpeed: 3.3528,
            elevationDifference: -2.3,
            paceZone: 3,
          },
        ],
      },
    },
    streams: {
      101: {
        time,
        latlng: Array.from({ length: points }, (_, index) => [
          43 + index * 0.00001,
          -79 - index * 0.00001,
        ]),
        altitude: Array.from({ length: points }, () => 80),
        distance,
      },
    },
  }

  const detail = buildPayload(cache, null, null, '2026-06-01').details['101']
  const paused = detail.route.find(point => point.elapsedS === 45)
  const teleport = detail.route.find(point => point.elapsedS === 110)
  assert.ok(paused)
  assert.ok(teleport)
  assert.equal(paused.speedKph, 0)
  assert.equal(teleport.speedKph, 28.8)
  assert.equal(paused.d, 0.195)
  assert.deepEqual(detail.analysisRanges, [
    {
      kind: 'lap',
      id: 'lap:lap-2',
      label: 'Lap 2',
      startElapsedS: 45,
      endElapsedS: 65,
      startDistanceKm: 0.195,
      endDistanceKm: 0.22,
      durationS: 20,
      movingTimeS: 10,
      distanceKm: 1.234,
      elevationGainM: 12.5,
      averageSpeedKph: 28.8,
      averageHeartRate: 151.2,
      averageWatts: 244.8,
      averageCadence: 89.4,
    },
    {
      kind: 'segment',
      id: 'segment:segment-9',
      label: 'Boardwalk east',
      startElapsedS: 110,
      endElapsedS: 120,
      startDistanceKm: 1.44,
      endDistanceKm: 1.49,
      durationS: 10,
      movingTimeS: 10,
      distanceKm: 1.234,
      elevationGainM: null,
      averageSpeedKph: null,
      averageHeartRate: null,
      averageWatts: null,
      averageCadence: null,
    },
  ])
  assert.deepEqual(detail.runSplitsMetric, [
    {
      split: 1,
      distanceKm: 1,
      elapsedTimeS: 305,
      movingTimeS: 300,
      averageSpeedKph: 12,
      elevationDifferenceM: 4.2,
      paceZone: 2,
    },
  ])
  assert.deepEqual(detail.runSplitsStandard, [
    {
      split: 1,
      distanceKm: 1.609,
      elapsedTimeS: 495,
      movingTimeS: 480,
      averageSpeedKph: 12.07,
      elevationDifferenceM: -2.3,
      paceZone: 3,
    },
  ])
})

test('keeps even route coverage when segment boundaries exceed the sampling budget', () => {
  const points = 1_000
  const distance = Array.from({ length: points }, (_, index) => index * 61.4)
  const cache: StravaRawCache = {
    version: 3,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride({ movingTime: points - 1, elapsedTime: points }) },
    activityDetails: {
      101: {
        calories: 500,
        laps: [],
        segmentEfforts: Array.from({ length: 75 }, (_, index) =>
          analysisRange(`segment-${index}`, `Segment ${index}`, {
            startIndex: 10 + index * 3,
            endIndex: 11 + index * 3,
          }),
        ),
        splitsMetric: [],
        splitsStandard: [],
      },
    },
    streams: {
      101: {
        time: Array.from({ length: points }, (_, index) => index),
        latlng: Array.from({ length: points }, (_, index) => [
          43 + index * 0.00001,
          -79 - index * 0.00002,
        ]),
        altitude: Array.from({ length: points }, () => 80),
        distance,
      },
    },
  }

  const detail = buildPayload(cache, null, null, '2026-06-01').details['101']
  assert.equal(detail.analysisRanges.length, 75)
  assert.ok(detail.route.some(point => point.d === 0.614))
  for (let index = 1; index < detail.route.length; index++)
    assert.ok(detail.route[index].d - detail.route[index - 1].d <= 0.5)
})

test('keeps dense map route continuous across GPS jumps', () => {
  const latlng: [number, number][] = [
    [43.64, -79.4],
    [43.64001, -79.40001],
    [43.64292, -79.39886],
    [43.64293, -79.39887],
  ]
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride({ distance: 680, movingTime: 4 }) },
    streams: {
      101: {
        latlng,
        altitude: [80, 80, 81, 81],
        distance: [0, 2, 672, 680],
        watts: [140, 141, 142, 143],
        heartrate: [120, 121, 122, 123],
        cadence: [70, 71, 72, 73],
      },
    },
  }

  const detail = buildPayload(cache, null, null, '2026-06-01').details['101']
  assert.equal(detail.mapRoute.length, 1)
  assert.ok(detail.mapRoute[0].length >= 2)
  assert.equal(Object.hasOwn(detail, 'mapBreaks'), false)
})

test('keeps sparse but plausible map samples continuous', () => {
  const latlng: [number, number][] = [
    [43.64, -79.4],
    [43.64002, -79.40002],
    [43.6411, -79.40002],
    [43.64112, -79.40004],
  ]
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride({ distance: 126, movingTime: 600 }) },
    streams: {
      101: {
        latlng,
        altitude: [80, 80, 81, 81],
        distance: [0, 3, 123, 126],
        watts: [140, 141, 142, 143],
        heartrate: [120, 121, 122, 123],
        cadence: [70, 71, 72, 73],
      },
    },
  }

  const detail = buildPayload(cache, null, null, '2026-06-01').details['101']
  assert.equal(detail.mapRoute.length, 1)
  assert.ok(detail.mapRoute[0].length >= 2)
  assert.equal(Object.hasOwn(detail, 'mapBreaks'), false)
})

test('merges WeatherKit wind into activity detail and day health', () => {
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride() },
    streams: {
      101: {
        time: [0, 3750, 7500],
        latlng: [
          [43.64, -79.4],
          [43.65, -79.39],
          [43.66, -79.38],
        ],
        altitude: [80, 90, 85],
        distance: [0, 30_700, 61_400],
      },
    },
  }
  const activity: WeatherActivity = {
    activityId: 101,
    date: '2026-06-07',
    start: '2026-06-07T11:29:55.000Z',
    end: '2026-06-07T13:34:55.000Z',
    latitude: 43.64,
    longitude: -79.4,
    durationS: 7500,
    windKph: 18,
    windDir: 'SW',
    windDirDeg: 225,
    windGustKph: 31,
    averageRelativeHumidityPct: 68,
    relativeHumidityProvenance: {
      source: 'weatherkit',
      sourceKind: 'modeled',
      samplingMethod: 'route-hour',
      inputTimestamp: '2026-06-07T11:29:55.000Z',
      coveragePct: 100,
    },
    temperatureC: 24,
    temperatureSeries: [
      { elapsedS: 0, temperatureC: 22 },
      { elapsedS: 3750, temperatureC: 26 },
      { elapsedS: 7500, temperatureC: 24 },
    ],
    source: 'weatherkit',
  }
  const weather: WeatherCache = {
    version: 2,
    lastSync: cache.lastSync,
    current: null,
    attribution: null,
    uvCalibration: null,
    activities: { 101: activity },
    days: summarizeWeatherDays({ 101: activity }),
  }

  const payload = buildPayload(cache, null, null, '2026-06-01', weather)
  assert.equal(payload.details['101'].windKph, 18)
  assert.equal(payload.details['101'].averageRelativeHumidityPct, 68)
  assert.equal(payload.details['101'].relativeHumidityProvenance?.coveragePct, 100)
  assert.equal(payload.details['101'].windDir, 'SW')
  assert.equal(payload.details['101'].windGustKph, 31)
  assert.equal(payload.details['101'].ambientTemperatureC, 24)
  assert.deepEqual(
    payload.details['101'].route.map(point => point.tempC),
    [22, 26, 24],
  )
  assert.equal(payload.health['2026-06-07'].windKph, 18)
  assert.equal(payload.health['2026-06-07'].windDir, 'SW')
})

test('uses nearest same-day weather for route-less swim and strength activities', () => {
  const swim = ride({
    id: 102,
    name: 'Pool swim',
    sportType: 'Swim',
    distance: 900,
    movingTime: 1_200,
    elapsedTime: 1_800,
    startDate: '2026-06-07T14:00:00Z',
    startDateLocal: '2026-06-07T10:00:00',
  })
  const strength = ride({
    id: 103,
    name: 'Weight training',
    sportType: 'WeightTraining',
    distance: 0,
    movingTime: 1_800,
    elapsedTime: 2_000,
    startDate: '2026-06-07T15:00:00Z',
    startDateLocal: '2026-06-07T11:00:00',
  })
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(swim.startDate) / 1000),
    activities: { 101: ride(), 102: swim, 103: strength },
  }
  const activity: WeatherActivity = {
    activityId: 101,
    date: '2026-06-07',
    start: '2026-06-07T11:29:55.000Z',
    end: '2026-06-07T13:34:55.000Z',
    latitude: 43.64,
    longitude: -79.4,
    durationS: 7_500,
    windKph: 18,
    windDir: 'SW',
    windDirDeg: 225,
    windGustKph: 31,
    averageRelativeHumidityPct: null,
    relativeHumidityProvenance: null,
    temperatureC: 24,
    source: 'weatherkit',
  }
  const weather: WeatherCache = {
    version: 2,
    lastSync: cache.lastSync,
    current: null,
    attribution: null,
    uvCalibration: null,
    activities: { 101: activity },
    days: summarizeWeatherDays({ 101: activity }),
  }

  const payload = buildPayload(cache, null, null, '2026-06-01', weather)
  for (const id of ['102', '103']) {
    const detail = payload.details[id]
    assert.equal(detail.ambientTemperatureC, 24)
    assert.equal(detail.windKph, 18)
    assert.equal(detail.windDir, 'SW')
    assert.equal(detail.windGustKph, 31)
  }
})

test('buildPayload keeps late evening syncs on the local calendar day', () => {
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-07-01T02:45:00.000Z'),
    lastActivityStart: Math.floor(Date.parse('2026-07-01T01:11:01Z') / 1000),
    activities: {
      101: ride({ startDate: '2026-07-01T01:11:01Z', startDateLocal: '2026-06-30T21:11:01' }),
    },
  }

  const payload = buildPayload(cache, null, null, '2026-06-30', null, null, null, 'America/Toronto')

  assert.deepEqual(
    payload.days.map(day => day.date),
    ['2026-06-30'],
  )
})

test('uses an inclusive 42-day window for the six-week power reference', () => {
  const stream = (watts: number): StravaStreams => ({
    time: [0, 1, 2, 3, 4],
    latlng: [],
    altitude: [0, 0, 0, 0, 0],
    distance: [0, 1, 2, 3, 4],
    watts: [watts, watts, watts, watts, watts],
  })
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-07-13T12:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-02T12:00:00Z') / 1000),
    activities: {
      101: ride({
        id: 101,
        movingTime: 5,
        elapsedTime: 5,
        startDate: '2026-06-01T12:00:00Z',
        startDateLocal: '2026-06-01T12:00:00',
      }),
      102: ride({
        id: 102,
        movingTime: 5,
        elapsedTime: 5,
        startDate: '2026-06-02T12:00:00Z',
        startDateLocal: '2026-06-02T12:00:00',
      }),
    },
    streams: { 101: stream(900), 102: stream(500) },
  }

  const payload = buildPayload(cache, null, null, '2026-06-01', null, null, null, 'UTC')
  assert.deepEqual(
    payload.powerCurveRef.find(point => point.s === 1),
    { s: 1, w: 500, activityId: 102, activityDate: '2026-06-02' },
  )
})

test('builds the calendar-year power reference outside the visible activity window', () => {
  const stream = (watts: number): StravaStreams => ({
    time: [0, 1, 2, 3, 4],
    latlng: [],
    altitude: [0, 0, 0, 0, 0],
    distance: [0, 1, 2, 3, 4],
    watts: [watts, watts, watts, watts, watts],
  })
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-07-13T12:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-07-14T12:00:00Z') / 1000),
    activities: {
      101: ride({
        id: 101,
        movingTime: 5,
        elapsedTime: 5,
        startDate: '2026-01-15T12:00:00Z',
        startDateLocal: '2026-01-15T12:00:00',
      }),
      102: ride({
        id: 102,
        movingTime: 5,
        elapsedTime: 5,
        startDate: '2026-06-02T12:00:00Z',
        startDateLocal: '2026-06-02T12:00:00',
      }),
      103: ride({
        id: 103,
        movingTime: 5,
        elapsedTime: 5,
        startDate: '2026-07-14T12:00:00Z',
        startDateLocal: '2026-07-14T12:00:00',
      }),
    },
    streams: { 101: stream(900), 102: stream(500), 103: stream(1_200) },
  }

  const payload = buildPayload(cache, null, null, '2026-05-15', null, null, null, 'UTC')
  assert.equal(payload.details['101']?.date, '2026-01-15')
  assert.deepEqual(
    payload.powerCurveRef.find(point => point.s === 1),
    { s: 1, w: 500, activityId: 102, activityDate: '2026-06-02' },
  )
  assert.deepEqual(
    payload.powerCurveYearRef.find(point => point.s === 1),
    { s: 1, w: 900, activityId: 101, activityDate: '2026-01-15' },
  )
  assert.equal(payload.powerCurveYear, 2026)
})

test('retains the winning activity at every aggregate power duration', () => {
  const stream = (watts: number[]): StravaStreams => ({
    time: [0, 1, 2, 3],
    latlng: [],
    altitude: [0, 0, 0, 0],
    distance: [0, 1, 2, 3],
    watts,
  })
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-08-13T12:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-08-12T12:00:00Z') / 1000),
    activities: {
      101: ride({
        id: 101,
        movingTime: 4,
        elapsedTime: 4,
        startDate: '2026-08-11T12:00:00Z',
        startDateLocal: '2026-08-11T08:00:00',
      }),
      102: ride({
        id: 102,
        movingTime: 4,
        elapsedTime: 4,
        startDate: '2026-08-12T12:00:00Z',
        startDateLocal: '2026-08-12T08:00:00',
      }),
    },
    streams: { 101: stream([500, 0, 0, 0]), 102: stream([350, 350, 350, 350]) },
  }

  const curve = buildPayload(cache, null, null, '2026-08-01', null, null, null, 'UTC').powerCurveRef
  assert.deepEqual(
    curve.find(point => point.s === 1),
    { s: 1, w: 500, activityId: 101, activityDate: '2026-08-11' },
  )
  assert.deepEqual(
    curve.find(point => point.s === 2),
    { s: 2, w: 350, activityId: 102, activityDate: '2026-08-12' },
  )
})

test('fits critical power from complete device-power windows', () => {
  const power = (durationS: number): number => 250 + 10_000 / durationS
  const stream = (durationS: number, watts: number, missingSecond?: number): StravaStreams => {
    const time = Array.from({ length: durationS }, (_, index) => index).filter(
      second => second !== missingSecond,
    )
    return {
      time,
      latlng: [],
      altitude: time.map(() => 0),
      distance: time.map(second => second),
      watts: time.map(() => watts),
    }
  }
  const activity = (id: number, durationS: number, deviceWatts = true): RawStravaActivity =>
    ride({
      id,
      movingTime: durationS,
      elapsedTime: durationS,
      deviceWatts,
      startDate: `2026-08-0${id}T12:00:00Z`,
      startDateLocal: `2026-08-0${id}T08:00:00`,
    })
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-08-13T12:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-08-05T12:00:00Z') / 1000),
    activities: {
      1: activity(1, 180),
      2: activity(2, 420),
      3: activity(3, 720),
      4: activity(4, 180),
      5: activity(5, 180, false),
      6: activity(6, 720, false),
    },
    streams: {
      1: stream(180, power(180)),
      2: stream(420, power(420)),
      3: stream(720, power(720)),
      4: stream(180, 600, 90),
      5: stream(180, 700),
      6: stream(720, 800),
    },
  }

  const payload = buildPayload(cache, null, null, '2026-08-06', null, null, null, 'UTC')

  assert.ok(payload.criticalPower)
  assert.equal(payload.criticalPower.criticalPowerWatts, 250)
  assert.ok(Math.abs(payload.criticalPower.wPrimeJoules - 10_000) <= 2)
  assert.equal(payload.criticalPower.independentEffortCount, 3)
  assert.equal(payload.criticalPower.confidence, 'medium')
  assert.deepEqual(
    payload.criticalPower.anchors.map(anchor => anchor.activityId),
    [1, 2, 3],
  )
  assert.deepEqual(payload.criticalPowerYear, {
    ...payload.criticalPower,
    window: 'calendar-year',
    windowFrom: '2026-01-01',
  })
  assert.deepEqual(
    [1, 2, 3, 6].map(id => payload.details[String(id)]?.id),
    [1, 2, 3, 6],
  )
})

test('fits activity critical power from one ride only', () => {
  const durationS = 720
  const time = Array.from({ length: durationS }, (_, index) => index)
  const watts = time.map(second => (second < 180 ? 250 + 10_000 / 180 : 250))
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-08-13T12:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-08-12T12:00:00Z') / 1000),
    activities: {
      101: ride({
        id: 101,
        movingTime: durationS,
        elapsedTime: durationS,
        deviceWatts: true,
        startDate: '2026-08-12T12:00:00Z',
        startDateLocal: '2026-08-12T08:00:00',
      }),
    },
    streams: { 101: { time, latlng: [], altitude: time.map(() => 0), distance: time, watts } },
  }

  const estimate = buildPayload(cache, null, null, '2026-08-01').details['101']
    .activityCriticalPower

  assert.ok(estimate)
  assert.equal(estimate.window, 'activity')
  assert.equal(estimate.windowFrom, '2026-08-12')
  assert.equal(estimate.windowTo, '2026-08-12')
  assert.equal(estimate.criticalPowerWatts, 250)
  assert.ok(Math.abs(estimate.wPrimeJoules - 10_000) <= 2)
  assert.equal(estimate.independentEffortCount, 1)
  assert.deepEqual(
    estimate.anchors.map(anchor => anchor.activityId),
    [101, 101, 101],
  )
})

test('samples every second for the full power curve', () => {
  const durationS = 39 * 60
  const length = durationS
  const seconds = Array.from({ length }, (_, index) => index)
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride({ movingTime: durationS, elapsedTime: durationS, deviceWatts: true }) },
    streams: {
      101: {
        time: seconds,
        latlng: [],
        altitude: seconds.map(() => 80),
        distance: seconds.map(second => second * 8),
        watts: seconds.map(() => 200),
      },
    },
  }

  const payload = buildPayload(cache, null, null, '2026-06-01')
  const curve = payload.details['101'].powerCurve
  assert.ok(curve)
  assert.deepEqual(
    curve.map(point => point.s),
    Array.from({ length: durationS }, (_, index) => index + 1),
  )
  assert.equal(curve.find(point => point.s === 61)?.w, 200)
  assert.equal(curve.find(point => point.s === 2_339)?.w, 200)
  assert.equal(curve.find(point => point.s === 2_340)?.w, 200)
  assert.equal(payload.powerCurveRef.find(point => point.s === 61)?.w, 200)
  assert.equal(payload.powerCurveRef.find(point => point.s === 2_340)?.w, 200)
})

test('samples per-second power curves through the full stream duration', () => {
  const streamDurationS = 3 * 60 * 60 + 1
  const seconds = Array.from({ length: streamDurationS }, (_, index) => index)
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({ movingTime: streamDurationS, elapsedTime: streamDurationS, deviceWatts: true }),
    },
    streams: {
      101: {
        time: seconds,
        latlng: [],
        altitude: seconds.map(() => 80),
        distance: seconds.map(second => second * 8),
        watts: seconds.map(() => 200),
      },
    },
  }

  const payload = buildPayload(cache, null, null, '2026-06-01')
  const curve = payload.details['101'].powerCurve
  assert.ok(curve)
  assert.equal(curve.length, streamDurationS)
  assert.deepEqual(curve.at(-1), { s: streamDurationS, w: 200 })
  assert.equal(payload.powerCurveRef.length, streamDurationS)
  assert.deepEqual(payload.powerCurveRef.at(-1), {
    s: streamDurationS,
    w: 200,
    activityId: 101,
    activityDate: '2026-06-07',
  })
})

test('keeps Strava power inclusive while projecting a zero-excluded cycling view', () => {
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: {
      101: ride({ movingTime: 4, elapsedTime: 4, averageWatts: 150, deviceWatts: true }),
    },
    streams: {
      101: {
        time: [0, 1, 2, 3],
        latlng: [],
        altitude: [0, 0, 0, 0],
        distance: [0, 1, 2, 3],
        watts: [0, 100, 200, 300],
      },
    },
  }

  const detail = buildPayload(cache, null, null, '2026-06-01', null, 200).details['101']

  assert.equal(detail.avgWatts, 150)
  assert.deepEqual(detail.powerZones, [2, 0, 0, 1, 0, 1, 0])
  assert.equal(detail.powerHist?.[0], 1)
  assert.deepEqual(detail.powerWithoutZeros, {
    avgWatts: 200,
    powerZones: [1, 0, 0, 1, 0, 1, 0],
    powerHist: [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
  })
})

test('derives elapsed cycling efforts with Garmin weight and ClimbPro segments', () => {
  const activity = ride({
    distance: 10_000,
    movingTime: 10,
    elapsedTime: 15,
    deviceWatts: true,
    startDate: '2026-06-07T12:00:00Z',
    startDateLocal: '2026-06-07T08:00:00',
  })
  const points = 10
  const cache: StravaRawCache = {
    version: 2,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(activity.startDate) / 1000),
    activities: { 101: activity },
    streams: {
      101: {
        time: [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
        latlng: Array.from({ length: points }, (_, i) => [43 + i * 0.00001, -79 - i * 0.00001]),
        altitude: Array.from({ length: points }, (_, i) => 100 + i),
        distance: [0, 1_000, 2_000, 3_000, 4_000, 4_000, 5_500, 7_000, 8_500, 10_000],
        watts: [100, 200, 300, 400, 500, 0, 100, 100, 100, 100],
        heartrate: [140, 141, 142, 143, 144, 145, 146, 147, 148, 149],
        cadence: Array.from({ length: points }, () => 80),
      },
    },
  }
  const garmin: GarminCache = {
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    activities: {
      edge: {
        id: 'edge',
        name: 'Cadence training',
        sport: 'bike',
        startDate: activity.startDate,
        startDateLocal: activity.startDateLocal,
        distanceM: 10_000,
        movingTimeS: 10,
        elapsedTimeS: 15,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      edge: {
        latlng: Array.from({ length: points }, (_, i) => [43 + i * 0.00001, -79 - i * 0.00001]),
        altitude: Array.from({ length: points }, (_, i) => 100 + i),
        distance: Array.from({ length: points }, (_, i) => i * 1_000),
        watts: Array.from({ length: points }, () => 900),
        heartrate: Array.from({ length: points }, () => 180),
        cadence: Array.from({ length: points }, () => 100),
      },
    },
    climbs: {
      edge: [
        {
          startDate: '2026-06-07T12:00:02.000Z',
          endDate: '2026-06-07T12:00:12.000Z',
          distanceM: 500,
          durationS: 10,
          movingTimeS: 10,
          elapsedTimeS: 15,
          elevationGainM: 25,
          elevationLossM: 0,
          startElevationM: 100,
          avgGradePct: 5,
          maxGradePct: 8,
          avgSpeedMps: 5,
          avgHeartRate: 150,
          maxHeartRate: 160,
          avgPower: 225,
          normalizedPower: 235,
          maxPower: 400,
          avgCadence: 80,
          difficulty: 'MODERATE',
        },
      ],
    },
    weight: [
      {
        ts: Date.parse('2026-06-06T12:00:00Z'),
        date: '2026-06-06',
        weightKg: 76,
        bmi: null,
        bodyFatPct: null,
        bodyWaterPct: null,
        muscleMassKg: null,
        boneMassKg: null,
      },
      {
        ts: Date.parse('2026-06-07T11:00:00Z'),
        date: '2026-06-07',
        weightKg: 75,
        bmi: null,
        bodyFatPct: null,
        bodyWaterPct: null,
        muscleMassKg: null,
        boneMassKg: null,
      },
      {
        ts: Date.parse('2026-06-07T13:00:00Z'),
        date: '2026-06-07',
        weightKg: 74,
        bmi: null,
        bodyFatPct: null,
        bodyWaterPct: null,
        muscleMassKg: null,
        boneMassKg: null,
      },
    ],
  }

  const detail = buildPayload(cache, null, garmin, '2026-06-01').details['101']
  const efforts = detail.bestEfforts
  assert.ok(efforts)
  assert.deepEqual(detail.analysisRanges, [
    {
      kind: 'climb',
      source: 'garmin-climbpro',
      id: 'garmin-climbpro:1:2026-06-07T12:00:02.000Z',
      label: 'Climb 1',
      startElapsedS: 2,
      endElapsedS: 12,
      startDistanceKm: 2,
      endDistanceKm: 7,
      durationS: 10,
      distanceKm: 0.5,
      elevationGainM: 25,
      averageSpeedKph: 18,
      averageHeartRate: 150,
      averageWatts: 225,
      averageCadence: 80,
    },
  ])
  assert.equal(efforts.weightKg, 75)
  assert.equal(efforts.weightDate, '2026-06-07')
  assert.equal(efforts.distance.find(effort => effort.label === '10K')?.elapsedTimeS, 14)
  assert.deepEqual(efforts.power[0], {
    durationS: 5,
    averageWatts: 300,
    wattsPerKg: 4,
    averageHeartRate: 142,
    elevationDeltaM: 4,
  })
  assert.equal(efforts.power.find(effort => effort.durationS === 15)?.averageWatts, 126)
  assert.deepEqual(efforts.climbs, [
    {
      source: 'garmin-climbpro',
      name: 'Climb 1',
      durationS: 10,
      distanceM: 500,
      elevationGainM: 25,
      averageGradePct: 5,
      averageSpeedKph: 18,
      averageHeartRate: 150,
      averageWatts: 225,
      wattsPerKg: 3,
      vamMPerHour: 9000,
    },
  ])
  assert.deepEqual(
    detail.powerCurve?.map(point => point.s),
    Array.from({ length: 15 }, (_, index) => index + 1),
  )
  assert.equal(detail.powerCurve?.find(point => point.s === 15)?.w, 126)

  const baseClimb = garmin.climbs?.edge[0]
  assert.ok(baseClimb)
  const clippedClimb = buildPayload(
    cache,
    null,
    { ...garmin, climbs: { edge: [{ ...baseClimb, endDate: '2026-06-07T12:19:35.000Z' }] } },
    '2026-06-01',
  ).details['101'].analysisRanges[0]
  assert.equal(clippedClimb.startElapsedS, 2)
  assert.equal(clippedClimb.endElapsedS, 14)

  const climbOnly = buildPayload({ ...cache, streams: {} }, null, garmin, '2026-06-01').details[
    '101'
  ].bestEfforts
  assert.ok(climbOnly)
  assert.deepEqual(climbOnly.distance, [])
  assert.deepEqual(climbOnly.power, [])
  assert.equal(climbOnly.climbs.length, 1)

  const withoutSameDayWeight = buildPayload(
    cache,
    null,
    { ...garmin, weight: garmin.weight?.filter(sample => sample.date !== '2026-06-07') },
    '2026-06-01',
  ).details['101'].bestEfforts
  assert.ok(withoutSameDayWeight)
  assert.equal(withoutSameDayWeight.weightKg, null)
  assert.equal(withoutSameDayWeight.power[0].wattsPerKg, null)
})
