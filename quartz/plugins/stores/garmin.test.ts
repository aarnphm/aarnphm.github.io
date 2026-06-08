import assert from 'node:assert/strict'
import test from 'node:test'
import type { GarminCache } from './garmin'
import type { RawStravaActivity } from './strava'
import {
  emptyGarminFueling,
  emptyGarminMetrics,
  matchGarminActivity,
  matchGarminFueling,
} from './garmin'

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
