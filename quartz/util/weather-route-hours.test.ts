import assert from 'node:assert/strict'
import test from 'node:test'
import type {
  WeatherActivity,
  WeatherActivityCandidate,
  WeatherHour,
} from '../plugins/stores/weather'
import {
  routeHourQueries,
  routeWeatherFingerprint,
  routeWeatherNeedsRefresh,
  selectRouteHour,
  type RouteWeatherStream,
} from './weather-route-hours'

const candidate: WeatherActivityCandidate = {
  activityId: 101,
  date: '2026-06-11',
  start: '2026-06-11T13:30:00.000Z',
  end: '2026-06-11T15:15:00.000Z',
  latitude: 43.64,
  longitude: -79.4,
  durationS: 6_300,
}

const stream: RouteWeatherStream = {
  timeS: [0, 1_800, 3_600, 5_400],
  distanceM: [0, 10_000, 20_000, 30_000],
  latlng: [
    [43.64, -79.4],
    [43.65, -79.41],
    [43.66, -79.42],
    [43.67, -79.43],
  ],
}

test('builds one route-coordinate query for every intersecting UTC hour', () => {
  const queries = routeHourQueries(candidate, stream)

  assert.deepEqual(
    queries.map(query => [query.forecastStart, query.overlapStart, query.overlapEnd]),
    [
      ['2026-06-11T13:00:00.000Z', '2026-06-11T13:30:00.000Z', '2026-06-11T14:00:00.000Z'],
      ['2026-06-11T14:00:00.000Z', '2026-06-11T14:00:00.000Z', '2026-06-11T15:00:00.000Z'],
      ['2026-06-11T15:00:00.000Z', '2026-06-11T15:00:00.000Z', '2026-06-11T15:15:00.000Z'],
    ],
  )
  assert.deepEqual(
    queries.map(query => [query.latitude, query.longitude]),
    [
      [43.64, -79.4],
      [43.66, -79.42],
      [43.67, -79.43],
    ],
  )
  assert.equal(queries.at(-1)?.trailingCoordinate, true)
})

test('route fingerprints are stable and invalidate on route changes', () => {
  const first = routeWeatherFingerprint(101, candidate.start, candidate.end, stream)
  const same = routeWeatherFingerprint(101, candidate.start, candidate.end, {
    ...stream,
    latlng: stream.latlng.map(point => [...point]),
  })
  const changed = routeWeatherFingerprint(101, candidate.start, candidate.end, {
    ...stream,
    latlng: [...stream.latlng.slice(0, -1), [43.68, -79.43]],
  })

  assert.equal(first, same)
  assert.notEqual(first, changed)
  assert.match(first, /^[a-f0-9]{64}$/)
})

test('selects an exact hourly timestamp across ISO formatting variants', () => {
  const hour: WeatherHour = {
    forecastStart: '2026-06-11T14:00:00Z',
    windSpeed: null,
    windDirection: null,
    windGust: null,
    relativeHumidity: null,
    temperature: 20,
    uvIndex: 4,
    cloudCover: 0.5,
    pressure: null,
    daylight: true,
    conditionCode: null,
    precipitationChance: null,
    precipitationType: null,
  }
  assert.equal(selectRouteHour([hour], '2026-06-11T14:00:00.000Z'), hour)
  assert.equal(selectRouteHour([hour], '2026-06-11T15:00:00.000Z'), null)
})

test('refreshes old schemas, changed routes, recent activities, and incomplete coverage', () => {
  const fingerprint = 'a'.repeat(64)
  const activity: WeatherActivity = {
    activityId: 101,
    date: candidate.date,
    start: candidate.start,
    end: candidate.end,
    latitude: candidate.latitude,
    longitude: candidate.longitude,
    durationS: candidate.durationS,
    windKph: null,
    windDir: null,
    windDirDeg: null,
    windGustKph: null,
    averageRelativeHumidityPct: null,
    relativeHumidityProvenance: null,
    temperatureC: null,
    routeFingerprint: fingerprint,
    fetchedAt: 1,
    routeHours: [
      ['2026-06-11T13:00:00.000Z', candidate.start, '2026-06-11T14:00:00.000Z', 0, 1_800],
      [
        '2026-06-11T14:00:00.000Z',
        '2026-06-11T14:00:00.000Z',
        '2026-06-11T15:00:00.000Z',
        1_800,
        5_400,
      ],
      ['2026-06-11T15:00:00.000Z', '2026-06-11T15:00:00.000Z', candidate.end, 5_400, 6_300],
    ].map(([forecastStart, overlapStart, overlapEnd, elapsedStartS, elapsedEndS]) => ({
      forecastStart: String(forecastStart),
      overlapStart: String(overlapStart),
      overlapEnd: String(overlapEnd),
      elapsedStartS: Number(elapsedStartS),
      elapsedEndS: Number(elapsedEndS),
      latitude: candidate.latitude,
      longitude: candidate.longitude,
      uvIndex: 4,
      cloudCover: 0.5,
      temperatureC: 20,
      windSpeedKph: 10,
      windDirectionDeg: 0,
      windGustKph: 15,
      relativeHumidity: 0.5,
      pressureHpa: 1_015,
      daylight: true,
    })),
    source: 'weatherkit',
  }
  const afterActivity = Date.parse(candidate.start) + 1

  assert.equal(
    routeWeatherNeedsRefresh(activity, fingerprint, 5, 5, candidate.start, afterActivity, false),
    false,
  )
  assert.equal(
    routeWeatherNeedsRefresh(activity, fingerprint, 4, 5, candidate.start, afterActivity, false),
    true,
  )
  assert.equal(
    routeWeatherNeedsRefresh(activity, 'b'.repeat(64), 5, 5, candidate.start, afterActivity, false),
    true,
  )
  assert.equal(
    routeWeatherNeedsRefresh(
      activity,
      fingerprint,
      5,
      5,
      candidate.start,
      Date.parse(candidate.start),
      false,
    ),
    true,
  )
  assert.equal(
    routeWeatherNeedsRefresh(
      { ...activity, routeHours: [] },
      fingerprint,
      5,
      5,
      candidate.start,
      afterActivity,
      false,
    ),
    true,
  )
})
