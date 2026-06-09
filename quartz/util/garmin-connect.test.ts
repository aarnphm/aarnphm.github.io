import assert from 'node:assert/strict'
import test from 'node:test'
import { garminConnectActivities, garminConnectActivity } from './garmin-connect'

test('normalizes Garmin Connect activity details into the Garmin cache shape', () => {
  const raw = {
    activities: [
      {
        activityId: 123,
        activityName: 'Morning ride',
        activityType: { typeKey: 'cycling' },
        startTimeGMT: '2026-06-01 12:00:00',
      },
    ],
  }
  const items = garminConnectActivities(raw)
  const activity = garminConnectActivity(
    {
      activityId: 123,
      activityName: 'Morning ride',
      activityType: { typeKey: 'cycling' },
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
