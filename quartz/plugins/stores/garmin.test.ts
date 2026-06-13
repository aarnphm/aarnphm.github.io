import assert from 'node:assert/strict'
import test from 'node:test'
import type { GarminCache } from './garmin'
import {
  emptyGarminFueling,
  emptyGarminMetrics,
  matchGarminActivity,
  matchGarminFueling,
} from './garmin'
import { buildPayload, type RawStravaActivity, type StravaRawCache } from './strava'

function strava(overrides: Partial<RawStravaActivity> = {}): RawStravaActivity {
  return {
    id: 101,
    name: 'Morning ride',
    sportType: 'Ride',
    distance: 48_200,
    movingTime: 7200,
    elapsedTime: 7500,
    totalElevationGain: 430,
    startDate: '2026-06-01T12:00:00Z',
    startDateLocal: '2026-06-01T08:00:00',
    averageSpeed: 6.69,
    ...overrides,
  }
}

test('matches Garmin fueling to a Strava activity by sport, start, distance, and duration', () => {
  const fueling = emptyGarminFueling()
  fueling.caloriesConsumed = 520
  fueling.carbsConsumedG = 74
  fueling.fluidMl = 1180
  const cache: GarminCache = {
    lastSync: Date.now(),
    activities: {
      edge: {
        id: 'edge',
        name: 'Morning ride',
        sport: 'bike',
        startDate: '2026-06-01T12:04:00Z',
        startDateLocal: '2026-06-01T08:04:00',
        distanceM: 48_450,
        movingTimeS: 7180,
        elapsedTimeS: 7520,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling,
      },
    },
  }

  assert.deepEqual(matchGarminFueling(strava(), 'bike', cache), {
    caloriesConsumed: 520,
    carbsConsumedG: 74,
    fluidMl: 1180,
    carbsRecommendedG: null,
    fluidRecommendedMl: null,
    sweatLossMl: null,
    sourceDevice: 'Edge 1050',
  })
})

test('rejects Garmin fueling outside sport and activity tolerances', () => {
  const fueling = emptyGarminFueling('Edge 1050')
  fueling.caloriesConsumed = 260
  const cache: GarminCache = {
    lastSync: Date.now(),
    activities: {
      run: {
        id: 'run',
        name: 'Wrong sport',
        sport: 'run',
        startDate: '2026-06-01T12:03:00Z',
        startDateLocal: '2026-06-01T08:03:00',
        distanceM: 48_200,
        movingTimeS: 7200,
        elapsedTimeS: 7500,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling,
      },
      far: {
        id: 'far',
        name: 'Wrong ride',
        sport: 'bike',
        startDate: '2026-06-01T12:04:00Z',
        startDateLocal: '2026-06-01T08:04:00',
        distanceM: 61_000,
        movingTimeS: 10_400,
        elapsedTimeS: 10_700,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling,
      },
    },
  }

  assert.equal(matchGarminFueling(strava(), 'bike', cache), null)
})

test('matches Garmin FIT baseline metrics without fueling', () => {
  const metrics = emptyGarminMetrics()
  metrics.totalCalories = 1403
  metrics.avgHeartRate = 135
  metrics.avgPower = 108
  const cache: GarminCache = {
    lastSync: Date.now(),
    activities: {
      fit: {
        id: 'fit:ride',
        name: 'ROAD',
        sport: 'bike',
        startDate: '2026-06-01T12:02:00Z',
        startDateLocal: '2026-06-01T12:02:00Z',
        distanceM: 48_320,
        movingTimeS: 7190,
        elapsedTimeS: 7512,
        sourceDevice: 'Edge 1050',
        sourceFile: 'ride.fit',
        metrics,
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
  }

  const match = matchGarminActivity(strava(), 'bike', cache)
  assert.equal(match?.activity.metrics.totalCalories, 1403)
  assert.equal(match?.distanceDiffM, 120)
  assert.equal(matchGarminFueling(strava(), 'bike', cache), null)
})

test('backfills Strava detail streams from a matched Garmin activity', () => {
  const ride = strava({ averageHeartrate: undefined, maxHeartrate: undefined })
  const stravaCache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(ride.startDate) / 1000),
    activities: { [ride.id]: ride },
    streams: { [ride.id]: { latlng: [], altitude: [], distance: [] } },
  }
  const garmin: GarminCache = {
    lastSync: Date.now(),
    activities: {
      edge: {
        id: 'edge',
        name: 'Morning ride',
        sport: 'bike',
        startDate: '2026-06-01T12:04:00Z',
        startDateLocal: '2026-06-01T08:04:00',
        distanceM: 48_450,
        movingTimeS: 7180,
        elapsedTimeS: 7520,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      edge: {
        latlng: [
          [43.1, -79.1],
          [43.2, -79.2],
          [43.3, -79.3],
        ],
        altitude: [101, 104, 109],
        distance: [0, 500, 1000],
        watts: [100, 200, 300],
        heartrate: [130, 140, 150],
        cadence: [80, 88, 96],
      },
    },
  }

  const detail = buildPayload(stravaCache, null, garmin, '2026-05-12').details[String(ride.id)]
  assert.equal(detail.route.length, 3)
  assert.equal(detail.elevationM, 8)
  assert.equal(detail.avgHr, 140)
  assert.equal(detail.maxHr, 150)
  assert.equal(detail.avgCadence, 88)
  assert.equal(detail.powerCurve?.[0]?.w, 300)
})

test('matches strength to a sport-null Garmin entry and backfills calories', () => {
  const lift = strava({
    id: 202,
    name: 'Legs & Core Strength',
    sportType: 'WeightTraining',
    distance: 0,
    movingTime: 2700,
    elapsedTime: 2895,
    totalElevationGain: 0,
    startDate: '2026-06-11T21:29:46Z',
    startDateLocal: '2026-06-11T17:29:46',
    calories: 0,
  })
  const metrics = emptyGarminMetrics()
  metrics.totalCalories = 356
  const garmin: GarminCache = {
    lastSync: Date.now(),
    activities: {
      lift: {
        id: 'connect:lift',
        name: 'Leg & Core Strength',
        sport: null,
        startDate: '2026-06-11T21:29:50Z',
        startDateLocal: '2026-06-11T17:29:50',
        distanceM: null,
        movingTimeS: null,
        elapsedTimeS: null,
        sourceDevice: null,
        sourceFile: null,
        metrics,
        fueling: emptyGarminFueling(),
      },
    },
  }

  assert.equal(matchGarminActivity(lift, 'strength', garmin)?.activity.id, 'connect:lift')

  const stravaCache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-11T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse(lift.startDate) / 1000),
    activities: { [lift.id]: lift },
    streams: {},
  }
  const detail = buildPayload(stravaCache, null, garmin).details[String(lift.id)]
  assert.equal(detail.calories, 356)
  assert.equal(detail.garmin?.activityId, 'connect:lift')
})

test('strength skips Garmin entries that carry distance', () => {
  const lift = strava({
    sportType: 'WeightTraining',
    distance: 0,
    movingTime: 2700,
    elapsedTime: 2895,
    startDate: '2026-06-11T21:29:46Z',
    startDateLocal: '2026-06-11T17:29:46',
  })
  const garmin: GarminCache = {
    lastSync: Date.now(),
    activities: {
      ride: {
        id: 'connect:ride',
        name: 'Toronto Road Cycling',
        sport: null,
        startDate: '2026-06-11T21:35:00Z',
        startDateLocal: '2026-06-11T17:35:00',
        distanceM: 21_386,
        movingTimeS: 2724,
        elapsedTimeS: 2810,
        sourceDevice: null,
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling(),
      },
    },
  }

  assert.equal(matchGarminActivity(lift, 'strength', garmin), null)
})

test('garminConnectVo2 parses maxmet daily payloads defensively', async () => {
  const { garminConnectVo2 } = await import('../../util/garmin-connect')
  const rows = garminConnectVo2([
    {
      calendarDate: '2026-06-10',
      generic: { vo2MaxPreciseValue: 54.3, vo2MaxValue: 54 },
      cycling: { vo2MaxValue: 50 },
    },
    { generic: { calendarDate: '2026-06-09', vo2MaxValue: 53 }, cycling: null },
    { calendarDate: 'not-a-date', generic: { vo2MaxValue: 50 } },
    { calendarDate: '2026-06-08', generic: {}, cycling: {} },
    'garbage',
  ])
  assert.deepEqual(rows, [
    { date: '2026-06-09', generic: 53, cycling: null },
    { date: '2026-06-10', generic: 54.3, cycling: 50 },
  ])
})

test('garminConnectWeightSamples flattens multiple weigh-ins per day with timestamps', async () => {
  const { garminConnectWeightSamples } = await import('../../util/garmin-connect')
  const morning = Date.parse('2026-06-12T07:30:00Z')
  const evening = Date.parse('2026-06-12T21:10:00Z')
  const samples = garminConnectWeightSamples({
    dailyWeightSummaries: [
      {
        summaryDate: '2026-06-12',
        numOfWeightEntries: 2,
        allWeightMetrics: [
          { timestampGMT: morning, weight: 91920, bmi: 26, bodyFat: 22 },
          { timestampGMT: evening, weight: 90800, bmi: 25.7, bodyFat: 21.4 },
        ],
      },
      { summaryDate: '2026-06-13', latestWeight: { weight: 90500, bodyFat: 21 } },
    ],
  })
  assert.equal(samples.length, 3)
  assert.deepEqual(
    samples.map(s => [s.date, s.ts, s.weightKg]),
    [
      ['2026-06-12', morning, 91.92],
      ['2026-06-12', evening, 90.8],
      ['2026-06-13', Date.parse('2026-06-13T12:00:00.000Z'), 90.5],
    ],
  )
  assert.equal(samples[0].bodyFatPct, 22)
  const legacy = garminConnectWeightSamples({
    dateWeightList: [{ calendarDate: '2026-06-10', weight: 89.1, bodyFat: 22 }],
  })
  assert.equal(legacy[0].weightKg, 89.1)
  assert.equal(legacy[0].bodyFatPct, 22)
  assert.equal(legacy[0].boneMassKg, null)
  assert.deepEqual(garminConnectWeightSamples('junk'), [])
})

test('garminConnectWeightGoal finds the weight-typed goal and ignores other goal kinds', async () => {
  const { garminConnectWeightGoal } = await import('../../util/garmin-connect')
  assert.equal(
    garminConnectWeightGoal([
      { goalType: 0, targetValue: 10000 },
      { userGoalTypePK: 4, goalValue: 80000 },
    ]),
    80,
  )
  assert.equal(garminConnectWeightGoal([{ goalType: 'WEIGHT', targetValue: 79.5 }]), 79.5)
  assert.equal(garminConnectWeightGoal([{ goalType: 0, targetValue: 10000 }]), null)
  assert.equal(garminConnectWeightGoal({ goals: [{ goalType: 4, targetValue: 81200 }] }), 81.2)
  assert.equal(garminConnectWeightGoal({ goalWeight: 80000 }), 80)
  assert.equal(garminConnectWeightGoal({ weightGoal: { goalWeight: 79.5 } }), 79.5)
  assert.equal(garminConnectWeightGoal({ nothing: true }), null)
  assert.equal(garminConnectWeightGoal(null), null)
})
