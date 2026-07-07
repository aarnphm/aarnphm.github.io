import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity, StravaRawCache } from '../plugins/stores/strava'
import { emptyGarminFueling, emptyGarminMetrics, type GarminCache } from '../plugins/stores/garmin'
import { garminConnectNumericActivityId, selectGarminTitleUpdates } from './garmin-title-sync'

function strava(overrides: Partial<RawStravaActivity> = {}): RawStravaActivity {
  return {
    id: 101,
    name: 'Tempo   Training',
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

function stravaCache(activities: RawStravaActivity[]): StravaRawCache {
  return {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: 0,
    activities: Object.fromEntries(activities.map(activity => [String(activity.id), activity])),
  }
}

function garminCache(overrides: Partial<GarminCache['activities'][string]> = {}): GarminCache {
  return {
    lastSync: Date.now(),
    activities: {
      edge: {
        id: 'connect:23227931231',
        name: 'Toronto Road Cycling',
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
        ...overrides,
      },
    },
  }
}

test('selects bike title updates from matched Strava and Garmin activities', () => {
  const updates = selectGarminTitleUpdates(stravaCache([strava()]), garminCache())

  assert.deepEqual(updates, [
    {
      stravaId: 101,
      garminId: 'connect:23227931231',
      garminActivityId: '23227931231',
      from: 'Toronto Road Cycling',
      to: 'Tempo Training',
      startDate: '2026-06-01T12:00:00Z',
      startDateLocal: '2026-06-01T08:00:00',
      score: updates[0].score,
      startDiffS: 240,
      distanceDiffM: 250,
      durationDiffS: 20,
    },
  ])
  assert.ok(updates[0].score > 0)
})

test('skips non-bike, same-title, nonnumeric Garmin ids, and filtered activities', () => {
  const cache = stravaCache([
    strava({ id: 1, name: 'Keep me' }),
    strava({ id: 2, name: 'Run Title', sportType: 'Run' }),
    strava({ id: 3, name: 'Late Ride', startDate: '2026-06-03T12:00:00Z' }),
  ])

  assert.equal(selectGarminTitleUpdates(cache, garminCache({ name: 'Keep me' })).length, 0)
  assert.equal(selectGarminTitleUpdates(cache, garminCache({ id: 'fit:file' })).length, 0)
  assert.equal(selectGarminTitleUpdates(cache, garminCache(), { ids: new Set(['3']) }).length, 0)
  assert.equal(selectGarminTitleUpdates(cache, garminCache(), { since: '2026-06-02' }).length, 0)
})

test('normalizes Garmin Connect numeric activity ids', () => {
  assert.equal(garminConnectNumericActivityId('connect:123'), '123')
  assert.equal(garminConnectNumericActivityId('123'), '123')
  assert.equal(garminConnectNumericActivityId('connect:fit:123'), null)
})
