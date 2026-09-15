import assert from 'node:assert/strict'
import test from 'node:test'
import { normalizeKind, type RawStravaActivity } from './strava'
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
  for (const id of [6, 7, 8, 9, 10, 56]) assert.equal(normalizeWahooSport(id), 'walk')
  for (const id of [2, 18, 42, 43]) assert.equal(normalizeWahooSport(id), 'strength')
  assert.equal(normalizeWahooSport(66), 'yoga')
  assert.equal(normalizeWahooSport(255, 'training'), 'strength')
  assert.equal(normalizeWahooSport(255, 'walking'), 'walk')
  assert.equal(normalizeWahooSport(255, 'yoga'), 'yoga')
  assert.equal(normalizeWahooSport(255, 'unrecognized'), null)
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

test('selects titles for every supported kind, including workouts without distance', () => {
  for (const sportType of [
    'Ride',
    'Run',
    'Swim',
    'WeightTraining',
    'Workout',
    'Crossfit',
    'Walk',
    'Hike',
    'Yoga',
    'Pilates',
    'PhysicalTherapy',
  ]) {
    const kind = normalizeKind(sportType)
    assert.ok(kind)
    const distance = ['Ride', 'Run', 'Swim', 'Walk', 'Hike'].includes(sportType) ? 1000 : 0
    const source = { ...strava(1), sportType, distance }
    const data = cache()
    const target = data.activities['wahoo:55']
    target.sport = kind
    target.distanceM = distance
    const updates = selectWahooTitleUpdates({ activities: { one: source } }, data)
    assert.equal(updates.length, 1, sportType)
    assert.equal(updates[0].to, source.name)
    target.name = updates[0].to
    assert.deepEqual(selectWahooTitleUpdates({ activities: { one: source } }, data), [])
  }
})

test('parses non-triathlon kinds and recovers known kinds from existing uncategorized caches', () => {
  for (const { workoutTypeId, kind } of [
    { workoutTypeId: 6, kind: 'walk' },
    { workoutTypeId: 42, kind: 'strength' },
    { workoutTypeId: 43, kind: 'strength' },
    { workoutTypeId: 66, kind: 'yoga' },
  ]) {
    const data = cache()
    const target = data.activities['wahoo:55']
    target.workoutTypeId = workoutTypeId
    target.sport = null
    const parsed = parseWahooCache(data)
    assert.equal(parsed.activities[target.id].sport, kind)
    assert.deepEqual(parseWahooCache(parsed), parsed)
  }
})

test('stationary title matches require duration and reject other sports or recorded distance', () => {
  const source = { ...strava(1), sportType: 'Yoga', distance: 0 }
  const data = cache()
  const target = data.activities['wahoo:55']
  target.sport = 'yoga'
  target.distanceM = null
  assert.equal(selectWahooTitleUpdates({ activities: { one: source } }, data).length, 1)
  target.distanceM = 1000
  assert.deepEqual(selectWahooTitleUpdates({ activities: { one: source } }, data), [])
  target.distanceM = 0
  target.sport = 'strength'
  assert.deepEqual(selectWahooTitleUpdates({ activities: { one: source } }, data), [])
  target.sport = 'yoga'
  target.elapsedTimeS = null
  target.movingTimeS = null
  assert.deepEqual(selectWahooTitleUpdates({ activities: { one: source } }, data), [])
})

test('deduplicates uncategorized Wahoo recordings across activity kinds before comparing titles', () => {
  const source = { ...strava(1), sportType: 'Yoga', distance: 0, name: 'yoga session' }
  const weaker = { ...strava(2), sportType: 'Workout', distance: 0 }
  const unknown = { ...source, id: 3, sportType: 'UnsupportedSport' }
  const data = cache()
  data.activities['wahoo:55'].sport = null
  data.activities['wahoo:55'].distanceM = 0
  const sources = { activities: { one: source, two: weaker, three: unknown } }
  const updates = selectWahooTitleUpdates(sources, data)
  assert.equal(updates.length, 1)
  assert.equal(updates[0].stravaId, 1)
  data.activities['wahoo:55'].name = updates[0].to
  assert.deepEqual(selectWahooTitleUpdates(sources, data), [])
})
