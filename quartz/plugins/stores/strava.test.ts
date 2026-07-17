import assert from 'node:assert/strict'
import test from 'node:test'
import { emptyGarminFueling, emptyGarminMetrics, type GarminCache } from './garmin'
import {
  buildPayload,
  type RawStravaActivity,
  type StravaRawCache,
  type StravaStreams,
} from './strava'
import { summarizeWeatherDays, type WeatherActivity, type WeatherCache } from './weather'

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

test('emits dense map geometry separately from compact telemetry route', () => {
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
    activities: { 101: ride({ averageTemp: 21 }) },
    streams: {
      101: {
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
  assert.equal(detail.mapRoute?.length, 1_000)
  assert.equal(Object.hasOwn(detail, 'mapBreaks'), false)
  assert.deepEqual(detail.mapRoute?.[0], { lat: 43, lng: -79 })
  assert.deepEqual(detail.mapRoute?.at(-1), { lat: 43.00999, lng: -79.01998 })
  assert.ok(detail.route.every(point => point.tempC === 21))
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
  assert.equal(detail.mapRoute?.length, 4)
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
  assert.equal(detail.mapRoute?.length, 4)
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
    activities: { 101: activity },
    days: summarizeWeatherDays({ 101: activity }),
  }

  const payload = buildPayload(cache, null, null, '2026-06-01', weather)
  assert.equal(payload.details['101'].windKph, 18)
  assert.equal(payload.details['101'].windDir, 'SW')
  assert.equal(payload.details['101'].windGustKph, 31)
  assert.equal(payload.details['101'].avgTemp, 24)
  assert.deepEqual(
    payload.details['101'].route.map(point => point.tempC),
    [22, 26, 24],
  )
  assert.equal(payload.health['2026-06-07'].windKph, 18)
  assert.equal(payload.health['2026-06-07'].windDir, 'SW')
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
  assert.equal(payload.powerCurveRef.find(point => point.s === 1)?.w, 500)
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
  assert.equal(payload.details['101'], undefined)
  assert.equal(payload.powerCurveRef.find(point => point.s === 1)?.w, 500)
  assert.equal(payload.powerCurveYearRef.find(point => point.s === 1)?.w, 900)
  assert.equal(payload.powerCurveYear, 2026)
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

test('caps per-second power curves at three hours', () => {
  const maxDurationS = 3 * 60 * 60
  const streamDurationS = maxDurationS + 1
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
  assert.equal(curve.length, maxDurationS)
  assert.deepEqual(curve.at(-1), { s: maxDurationS, w: 200 })
  assert.equal(payload.powerCurveRef.length, maxDurationS)
  assert.deepEqual(payload.powerCurveRef.at(-1), { s: maxDurationS, w: 200 })
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
