import assert from 'node:assert/strict'
import test from 'node:test'
import { buildPayload, type RawStravaActivity, type StravaRawCache } from './strava'
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
    activities: { 101: ride() },
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
  assert.deepEqual(detail.mapRoute?.[0], { lat: 43, lng: -79 })
  assert.deepEqual(detail.mapRoute?.at(-1), { lat: 43.00999, lng: -79.01998 })
})

test('merges WeatherKit wind into activity detail and day health', () => {
  const cache: StravaRawCache = {
    version: 1,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: Date.now() },
    lastSync: Date.parse('2026-06-08T00:00:00Z'),
    lastActivityStart: Math.floor(Date.parse('2026-06-07T11:29:55Z') / 1000),
    activities: { 101: ride() },
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
    source: 'weatherkit',
  }
  const weather: WeatherCache = {
    version: 1,
    lastSync: cache.lastSync,
    activities: { 101: activity },
    days: summarizeWeatherDays({ 101: activity }),
  }

  const payload = buildPayload(cache, null, null, '2026-06-01', weather)
  assert.equal(payload.details['101'].windKph, 18)
  assert.equal(payload.details['101'].windDir, 'SW')
  assert.equal(payload.details['101'].windGustKph, 31)
  assert.equal(payload.health['2026-06-07'].windKph, 18)
  assert.equal(payload.health['2026-06-07'].windDir, 'SW')
})
