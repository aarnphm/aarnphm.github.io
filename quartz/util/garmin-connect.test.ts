import assert from 'node:assert/strict'
import test from 'node:test'
import {
  garminConnectActivities,
  garminConnectActivity,
  garminConnectClimbSegments,
  garminConnectStreams,
  garminConnectWeightSamples,
} from './garmin-connect'

test('normalizes Garmin Connect activity details into the Garmin cache shape', () => {
  const raw = {
    activities: [
      {
        activityId: 123,
        activityName: 'Morning ride',
        activityType: { typeKey: 'road_biking' },
        startTimeGMT: '2026-06-01 12:00:00',
      },
    ],
  }
  const items = garminConnectActivities(raw)
  const activity = garminConnectActivity(
    {
      activityId: 123,
      activityName: 'Morning ride',
      activityType: { typeKey: 'road_biking' },
      startTimeGMT: '2026-06-01 12:00:00',
      startTimeLocal: '2026-06-01 08:00:00',
      distance: 48_200,
      duration: 7500,
      movingDuration: 7200,
      calories: 1403,
      averageHR: 135,
      maxHR: 168,
      averagePower: 108,
      normalizedPower: 151,
      maxPower: 530,
      averageBikeCadence: 82,
      elevationGain: 430,
      elevationLoss: 421,
      kilojoules: 775.4,
      trainingStressScore: 67.8,
      intensityFactor: 0.713,
      deviceName: 'Edge 1050',
    },
    items[0].record,
    0,
  )

  assert.equal(items.length, 1)
  assert.equal(activity?.id, 'connect:123')
  assert.equal(activity?.sport, 'bike')
  assert.equal(activity?.startDate, '2026-06-01T12:00:00.000Z')
  assert.equal(activity?.startDateLocal, '2026-06-01T08:00:00')
  assert.equal(activity?.distanceM, 48_200)
  assert.equal(activity?.movingTimeS, 7200)
  assert.equal(activity?.elapsedTimeS, 7500)
  assert.equal(activity?.sourceDevice, 'Edge 1050')
  assert.deepEqual(activity?.metrics, {
    totalCalories: 1403,
    metabolicCalories: null,
    avgHeartRate: 135,
    maxHeartRate: 168,
    avgPower: 108,
    normalizedPower: 151,
    maxPower: 530,
    avgCadence: 82,
    totalAscentM: 430,
    totalDescentM: 421,
    totalWorkKJ: 775.4,
    trainingStressScore: 67.8,
    intensityFactor: 0.713,
  })
})

test('keeps Garmin Connect fueling fields when they appear in nested records', () => {
  const fallback = {
    activityId: 456,
    activityName: 'Long ride',
    activityType: { typeKey: 'cycling' },
    startTimeGMT: '2026-06-07 11:29:55',
    distance: 96_000,
    duration: 14_400,
  }
  const activity = garminConnectActivity(
    {
      activityDetailDTO: fallback,
      nutrition: { caloriesConsumed: 520, carbsConsumedG: 74 },
      hydration: { fluidConsumedMl: 1180, sweatLossMl: 2100 },
    },
    fallback,
    0,
  )

  assert.deepEqual(activity?.fueling, {
    caloriesConsumed: 520,
    carbsConsumedG: 74,
    fluidMl: 1180,
    carbsRecommendedG: null,
    fluidRecommendedMl: null,
    sweatLossMl: 2100,
    sourceDevice: null,
  })
})

test('deduplicates Garmin Connect list records by activity id', () => {
  const items = garminConnectActivities({
    activities: [
      { activityId: 1, startTimeGMT: '2026-06-01 12:00:00' },
      { activityIdStr: '1', startTimeGMT: '2026-06-01 12:00:00' },
      { activityId: 2, startTimeGMT: '2026-06-02 12:00:00' },
    ],
  })

  assert.deepEqual(
    items.map(item => item.id),
    ['1', '2'],
  )
})

test('normalizes Garmin Connect GraphQL activity scalars', () => {
  const items = garminConnectActivities({
    data: {
      searchActivitiesScalar: [
        { activityId: 3, startTimeGMT: '2026-06-03 12:00:00' },
        { activityId: 4, startTimeGMT: '2026-06-04 12:00:00' },
      ],
    },
  })

  assert.deepEqual(
    items.map(item => item.id),
    ['3', '4'],
  )
})

test('normalizes Garmin Connect weight samples across dayview aliases', () => {
  const samples = garminConnectWeightSamples({
    dailyWeightSummaries: [
      {
        summaryDate: '2026-06-18',
        weightMetrics: [
          {
            samplePk: 1_781_826_649,
            weightInGrams: 91_130,
            bodyMassIndex: 25.8,
            bodyFatPercentage: 22.2,
            bodyWaterPercentage: 56.8,
            muscleMassInGrams: 37_750,
            boneMassInGrams: 6_240,
          },
        ],
      },
    ],
  })

  assert.deepEqual(samples, [
    {
      ts: 1_781_826_649_000,
      date: '2026-06-18',
      weightKg: 91.13,
      bmi: 25.8,
      bodyFatPct: 22.2,
      bodyWaterPct: 56.8,
      muscleMassKg: 37.75,
      boneMassKg: 6.24,
    },
  ])
})

test('normalizes Garmin Connect weight records with timestamp strings and pound units', () => {
  const samples = garminConnectWeightSamples({
    weightMetrics: [
      {
        weighInTimestampGMT: '2026-06-19 11:03:00',
        weight: 200.6,
        weightUnit: 'pounds',
        bodyFatPercent: 22.1,
        bodyWaterPct: 56.7,
        muscleMassKg: 37.8,
        boneMassKg: 6.2,
      },
    ],
  })

  assert.equal(samples.length, 1)
  assert.equal(samples[0].date, '2026-06-19')
  assert.equal(samples[0].weightKg, 90.99)
  assert.equal(samples[0].bodyFatPct, 22.1)
  assert.equal(samples[0].bodyWaterPct, 56.7)
  assert.equal(samples[0].muscleMassKg, 37.8)
  assert.equal(samples[0].boneMassKg, 6.2)
})

test('normalizes Garmin ClimbPro parent splits and excludes their sections', () => {
  const climbs = garminConnectClimbSegments({
    splits: [
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB',
        startTimeGMT: '2026-07-09T22:12:18.0',
        endTimeGMT: '2026-07-09T22:15:24.0',
        distance: 1045.82,
        duration: 187.552,
        movingDuration: 186,
        elapsedDuration: 187.552,
        elevationGain: 29,
        elevationLoss: 3,
        startElevation: 81.8,
        averageGrade: 2.730000019,
        maxGrade: 8.739999771,
        averageSpeed: 5.576000213,
        averageHR: 155,
        maxHR: 165,
        averagePower: 225,
        normalizedPower: 262,
        maxPower: 451,
        averageBikeCadence: 80,
        climbProDifficulty: 'NONE',
      },
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB_SECTION',
        startTimeGMT: '2026-07-09T22:06:26.0',
        endTimeGMT: '2026-07-09T22:07:28.0',
        distance: 268.51,
        duration: 62,
        elevationGain: 18.2,
        averageGrade: 7.59,
        averagePower: 338.24,
        climbProDifficulty: 'STEEP',
      },
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB',
        startTimeGMT: '2026-07-09T22:06:14.0',
        endTimeGMT: '2026-07-09T22:09:34.0',
        distance: 1067.37,
        duration: 200.14,
        movingDuration: 200,
        elapsedDuration: 200.14,
        elevationGain: 27,
        elevationLoss: 4,
        startElevation: 83.8,
        averageGrade: 2.569999933,
        maxGrade: 9.029999733,
        averageSpeed: 5.333000183,
        averageHR: 150,
        maxHR: 162,
        averagePower: 203,
        normalizedPower: 257,
        maxPower: 473,
        averageBikeCadence: 79,
        climbProDifficulty: 'NONE',
      },
      {
        type: 'SURFACE_TYPE_PAVED',
        startTimeGMT: '2026-07-09T21:38:32.0',
        endTimeGMT: '2026-07-09T22:57:35.0',
        distance: 29298.34,
        duration: 4134.357,
      },
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB',
        startTimeGMT: '2026-07-09T23:00:00.0',
        endTimeGMT: '2026-07-09T23:01:00.0',
        distance: 300,
      },
    ],
  })

  assert.deepEqual(climbs, [
    {
      startDate: '2026-07-09T22:06:14.000Z',
      endDate: '2026-07-09T22:09:34.000Z',
      distanceM: 1067.37,
      durationS: 200.14,
      movingTimeS: 200,
      elapsedTimeS: 200.14,
      elevationGainM: 27,
      elevationLossM: 4,
      startElevationM: 83.8,
      avgGradePct: 2.57,
      maxGradePct: 9.03,
      avgSpeedMps: 5.333,
      avgHeartRate: 150,
      maxHeartRate: 162,
      avgPower: 203,
      normalizedPower: 257,
      maxPower: 473,
      avgCadence: 79,
      difficulty: 'NONE',
    },
    {
      startDate: '2026-07-09T22:12:18.000Z',
      endDate: '2026-07-09T22:15:24.000Z',
      distanceM: 1045.82,
      durationS: 187.552,
      movingTimeS: 186,
      elapsedTimeS: 187.552,
      elevationGainM: 29,
      elevationLossM: 3,
      startElevationM: 81.8,
      avgGradePct: 2.73,
      maxGradePct: 8.74,
      avgSpeedMps: 5.576,
      avgHeartRate: 155,
      maxHeartRate: 165,
      avgPower: 225,
      normalizedPower: 262,
      maxPower: 451,
      avgCadence: 80,
      difficulty: 'NONE',
    },
  ])
})

test('normalizes Garmin Connect detail metrics into streams', () => {
  const streams = garminConnectStreams({
    metricDescriptors: [
      { key: 'sumDistance' },
      { key: 'directLatitude' },
      { key: 'directLongitude' },
      { key: 'directElevation' },
      { key: 'directPower' },
      { key: 'directHeartRate' },
      { key: 'directBikeCadence' },
      { key: 'sumElapsedDuration' },
      { key: 'directRespirationRate' },
    ],
    activityDetailMetrics: [
      { metrics: [0, 43.1, -79.1, 101, 120, 135, 82, 0, null] },
      { metrics: [500, 43.2, -79.2, 104, 180, 142, 88, 15, 27.42] },
      { metrics: [1000, 43.3, -79.3, 109, 210, 149, 91, 30, 31.08] },
    ],
  })

  assert.deepEqual(streams?.latlng, [
    [43.1, -79.1],
    [43.2, -79.2],
    [43.3, -79.3],
  ])
  assert.deepEqual(streams?.distance, [0, 500, 1000])
  assert.deepEqual(streams?.altitude, [101, 104, 109])
  assert.deepEqual(streams?.watts, [120, 180, 210])
  assert.deepEqual(streams?.heartrate, [135, 142, 149])
  assert.deepEqual(streams?.cadence, [82, 88, 91])
  assert.deepEqual(streams?.time, [0, 15, 30])
  assert.deepEqual(streams?.respiration, [0, 27.42, 31.08])
})
