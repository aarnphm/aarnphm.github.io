import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity } from './strava'
import {
  emptyWahooMetrics,
  matchWahooActivity,
  normalizeWahooSport,
  parseWahooCache,
  selectWahooTitleUpdates,
  type WahooActivity,
  type WahooCache,
} from './wahoo'

function strava(id: number, startDate = '2026-08-27T12:00:00.000Z'): RawStravaActivity {
  return {
    id,
    name: id === 1 ? 'Tempo Training' : 'Weaker Match',
    sportType: 'Ride',
    distance: id === 1 ? 48_200 : 47_000,
    movingTime: id === 1 ? 7200 : 6900,
    elapsedTime: id === 1 ? 7500 : 7100,
    totalElevationGain: 430,
    startDate,
    startDateLocal: '2026-08-27T08:00:00',
    averageSpeed: 6.69,
  }
}

function activity(edited = false): WahooActivity {
  return {
    id: 'wahoo:55',
    workoutId: 55,
    workoutTypeId: 15,
    workoutUpdatedAt: '2026-08-27T15:00:00.000Z',
    name: 'Toronto Road Cycling',
    sport: 'bike',
    startDate: '2026-08-27T12:04:00.000Z',
    startDateLocal: '2026-08-27T08:04:00',
    distanceM: 48_450,
    movingTimeS: 7180,
    elapsedTimeS: 7520,
    sourceDevice: 'ELEMNT BOLT',
    sourceFile: {
      url: 'https://cdn.wahoofitness.com/ride.fit',
      sha256: 'a'.repeat(64),
      byteLength: 4000,
      profileVersion: '21.208',
    },
    sweatLoss: { fluidMl: null, sodiumMg: null },
    metrics: emptyWahooMetrics(),
    summary: {
      id: 66,
      name: 'Toronto Road Cycling',
      timeZone: 'America/Toronto',
      manual: false,
      edited,
      fitnessAppId: 1,
      durationPausedS: 340,
      createdAt: '2026-08-27T15:00:00.000Z',
      updatedAt: '2026-08-27T15:00:00.000Z',
    },
  }
}

function cache(edited = false): WahooCache {
  const ride = activity(edited)
  return {
    version: 4,
    lastSync: Date.now(),
    activities: { [ride.id]: ride },
    streams: {
      [ride.id]: {
        timestamps: [],
        time: [],
        latlng: [],
        altitude: [],
        distance: [],
        watts: [],
        rightBalance: [],
        heartrate: [],
        cadence: [],
        speed: [],
        temperature: [],
        respiration: [],
        muscleOxygenPercent: [],
        totalHemoglobinConcentration: [],
        heatStrainIndex: [],
        coreTemperatureC: [],
        skinTemperatureC: [],
        minuteVentilation: [],
        tidalVolume: [],
        fluidLossMl: [],
        sodiumLossMg: [],
      },
    },
    gearShifts: { [ride.id]: [] },
    cyclingDynamics: {
      [ride.id]: {
        time: [],
        distance: [],
        leftPedalSmoothness: [],
        rightPedalSmoothness: [],
        leftTorqueEffectiveness: [],
        rightTorqueEffectiveness: [],
        leftPowerPhaseStart: [],
        leftPowerPhaseEnd: [],
        rightPowerPhaseStart: [],
        rightPowerPhaseEnd: [],
        positionChanges: [],
        seatedTimeS: null,
        standingTimeS: null,
      },
    },
    summitSegments: {
      [ride.id]: [
        {
          feature: 'summit-segment',
          uuid: 'WAHOO_ON_ROUTE_CLIMB-snake-road',
          name: 'Snake Road',
          startDate: '2026-08-27T12:30:00.000Z',
          endDate: '2026-08-27T12:35:00.000Z',
          distanceM: 1_500,
          durationS: 300,
          elevationGainM: 90,
          avgGradePct: 6,
          avgSpeedMps: 5,
          avgHeartRate: 155,
          avgPower: 280,
          avgCadence: 82,
        },
      ],
    },
  }
}

test('maps official Wahoo workout types to triathlon sports', () => {
  assert.equal(normalizeWahooSport(15), 'bike')
  assert.equal(normalizeWahooSport(67), 'run')
  assert.equal(normalizeWahooSport(25), 'swim')
  assert.equal(normalizeWahooSport(255, 'cycling'), 'bike')
})

test('rejects obsolete Wahoo cache versions', () => {
  assert.throws(() => parseWahooCache({ ...cache(), version: 3 }), /version 3 is unsupported/)
})

test('parses Summit segments and requires one entry for every Wahoo activity', () => {
  const value = cache()
  assert.deepEqual(parseWahooCache(value), value)

  const missing = { ...value, summitSegments: {} }
  assert.throws(() => parseWahooCache(missing), /activity wahoo:55 is missing summit segments/)

  const [segment] = value.summitSegments['wahoo:55']
  assert.ok(segment)
  assert.throws(
    () =>
      parseWahooCache({ ...value, summitSegments: { 'wahoo:55': [{ ...segment, durationS: 0 }] } }),
    /positive distance and duration/,
  )
  assert.throws(
    () =>
      parseWahooCache({
        ...value,
        summitSegments: { 'wahoo:55': [{ ...segment, feature: 'summit-freeride' }] },
      }),
    /uuid is invalid/,
  )
})

test('matches by sport, start, distance, and duration', () => {
  const match = matchWahooActivity(strava(1), 'bike', cache())
  assert.equal(match?.activity.id, 'wahoo:55')
  assert.equal(match?.startDiffMs, 240_000)
  assert.equal(match?.distanceDiffM, 250)
  assert.equal(match?.durationDiffS, 20)
  assert.equal(matchWahooActivity(strava(1, '2026-08-28T12:00:00.000Z'), 'bike', cache()), null)
})

test('rejects start-only matches before title mutation', () => {
  const missingDistance = cache()
  const distanceActivity = missingDistance.activities['wahoo:55']
  if (!distanceActivity) assert.fail('fixture omitted Wahoo activity')
  distanceActivity.distanceM = null
  assert.equal(matchWahooActivity(strava(1), 'bike', missingDistance), null)

  const missingDuration = cache()
  const durationActivity = missingDuration.activities['wahoo:55']
  if (!durationActivity) assert.fail('fixture omitted Wahoo activity')
  durationActivity.movingTimeS = null
  durationActivity.elapsedTimeS = null
  assert.equal(matchWahooActivity(strava(1), 'bike', missingDuration), null)
})

test('selects one strongest Strava cycling title and protects edited Wahoo workouts', () => {
  const stravaCache = { activities: { one: strava(1), two: strava(2) } }
  const updates = selectWahooTitleUpdates(stravaCache, cache())
  assert.equal(updates.length, 1)
  assert.equal(updates[0].stravaId, 1)
  assert.equal(updates[0].wahooWorkoutId, 55)
  assert.equal(updates[0].from, 'Toronto Road Cycling')
  assert.equal(updates[0].to, 'Tempo Training')
  assert.deepEqual(selectWahooTitleUpdates(stravaCache, cache(true)), [])
  assert.equal(selectWahooTitleUpdates(stravaCache, cache(true), { includeEdited: true }).length, 1)
})
