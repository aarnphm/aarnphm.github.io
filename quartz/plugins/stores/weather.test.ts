import assert from 'node:assert/strict'
import test from 'node:test'
import type { SurfaceCurrentEstimate } from '../../util/surface-current'
import {
  compassFromDegrees,
  parseWeatherCache,
  summarizeWeatherDays,
  weatherActivityFromHours,
  weatherActivityFromRouteHours,
  weatherSnapshotFromHours,
  type WeatherActivity,
  type WeatherActivityCandidate,
  type WeatherHour,
} from './weather'

function currentForActivity(activity: WeatherActivity): SurfaceCurrentEstimate {
  return {
    source: 'noaa-loofs',
    sourceKind: 'modeled',
    formulaId: 'garden-surface-current-v1',
    formulaVersion: 1,
    activityId: activity.activityId,
    routeFingerprint: activity.routeFingerprint ?? 'route',
    start: activity.start,
    end: activity.end,
    computedAt: Date.parse('2026-06-12T01:00:00Z'),
    spatialSamplingModel: 'containing-element',
    temporalSamplingModel: 'hourly-linear-vector',
    layer: 0,
    summary: {
      averageSpeedMps: 1,
      averageDirectionDeg: 90,
      coveragePct: 100,
      coveredDurationS: activity.durationS,
      elapsedDurationS: activity.durationS,
    },
    samples: [0, activity.durationS].map(elapsedS => ({
      elapsedS,
      speedMps: 1,
      directionDeg: 90,
      uMps: 1,
      vMps: 0,
      element: 1,
      validTime: new Date(
        Math.floor((Date.parse(activity.start) + elapsedS * 1_000) / 3_600_000) * 3_600_000,
      ).toISOString(),
      cycleTime: '2026-06-11T18:00:00.000Z',
      sourceUrl:
        elapsedS === 0
          ? 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/2026/06/11/loofs.t18z.20260611.fields.n005.nc.ascii'
          : 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/2026/06/11/loofs.t18z.20260611.fields.n003.nc.ascii',
    })),
  }
}

function candidate(values: Partial<WeatherActivityCandidate> = {}): WeatherActivityCandidate {
  return {
    activityId: 101,
    date: '2026-06-11',
    start: '2026-06-11T13:30:00.000Z',
    end: '2026-06-11T15:00:00.000Z',
    latitude: 43.64,
    longitude: -79.4,
    durationS: 5400,
    ...values,
  }
}

function hour(values: Partial<WeatherHour>): WeatherHour {
  return {
    forecastStart: '2026-06-11T13:00:00.000Z',
    windSpeed: 0,
    windDirection: null,
    windGust: null,
    relativeHumidity: null,
    temperature: null,
    uvIndex: null,
    cloudCover: null,
    pressure: null,
    daylight: null,
    conditionCode: null,
    precipitationChance: null,
    precipitationType: null,
    ...values,
  }
}

test('compassFromDegrees returns 16-point compass labels', () => {
  assert.equal(compassFromDegrees(0), 'N')
  assert.equal(compassFromDegrees(44), 'NE')
  assert.equal(compassFromDegrees(226), 'SW')
  assert.equal(compassFromDegrees(359), 'N')
  assert.equal(compassFromDegrees(null), null)
})

test('weatherActivityFromHours weights wind by activity overlap', () => {
  const activity = weatherActivityFromHours(candidate(), [
    hour({
      forecastStart: '2026-06-11T13:00:00.000Z',
      windSpeed: 10,
      windDirection: 270,
      windGust: 20,
      relativeHumidity: 0.5,
      temperature: 22,
    }),
    hour({
      forecastStart: '2026-06-11T14:00:00.000Z',
      windSpeed: 20,
      windDirection: 270,
      windGust: 26,
      relativeHumidity: 0.75,
      temperature: 24,
    }),
  ])

  assert.equal(activity?.windKph, 17)
  assert.equal(activity?.windDir, 'W')
  assert.equal(activity?.windDirDeg, 270)
  assert.equal(activity?.windGustKph, 26)
  assert.equal(activity?.averageRelativeHumidityPct, 67)
  assert.deepEqual(activity?.relativeHumidityProvenance, {
    source: 'weatherkit',
    sourceKind: 'modeled',
    samplingMethod: 'route-hour',
    inputTimestamp: '2026-06-11T13:30:00.000Z',
    coveragePct: 100,
  })
  assert.equal(activity?.temperatureC, 23)
  assert.deepEqual(activity?.temperatureSeries, [
    { elapsedS: 0, temperatureC: 22 },
    { elapsedS: 1800, temperatureC: 24 },
    { elapsedS: 5400, temperatureC: 24 },
  ])
})

test('weatherActivityFromHours calculates humidity independently from wind coverage', () => {
  const activity = weatherActivityFromHours(candidate(), [
    hour({ windSpeed: null, relativeHumidity: 0.68 }),
    hour({ forecastStart: '2026-06-11T14:00:00.000Z', windSpeed: 20, relativeHumidity: null }),
  ])

  assert.equal(activity?.windKph, 20)
  assert.equal(activity?.averageRelativeHumidityPct, 68)
  assert.equal(activity?.relativeHumidityProvenance?.coveragePct, 33)
})

test('weatherActivityFromHours preserves zero humidity and rejects malformed fractions', () => {
  const dry = weatherActivityFromHours(candidate(), [hour({ relativeHumidity: 0 })])
  const malformed = weatherActivityFromHours(candidate(), [
    hour({ relativeHumidity: -0.01 }),
    hour({ forecastStart: '2026-06-11T14:00:00.000Z', relativeHumidity: 1.01 }),
  ])

  assert.equal(dry?.averageRelativeHumidityPct, 0)
  assert.equal(dry?.relativeHumidityProvenance?.coveragePct, 33)
  assert.equal(malformed?.averageRelativeHumidityPct, null)
  assert.equal(malformed?.relativeHumidityProvenance?.coveragePct, 0)
})

test('summarizeWeatherDays folds activity weather into duration-weighted day wind', () => {
  const first = weatherActivityFromHours(candidate({ activityId: 101, durationS: 3600 }), [
    hour({ windSpeed: 10, windDirection: 350, windGust: 19 }),
  ])
  const second = weatherActivityFromHours(
    candidate({
      activityId: 102,
      start: '2026-06-11T16:00:00.000Z',
      end: '2026-06-11T18:00:00.000Z',
      durationS: 7200,
    }),
    [
      hour({
        forecastStart: '2026-06-11T16:00:00.000Z',
        windSpeed: 20,
        windDirection: 10,
        windGust: 32,
      }),
    ],
  )
  const activities: Record<string, WeatherActivity> = {}
  if (first) activities[String(first.activityId)] = first
  if (second) activities[String(second.activityId)] = second

  const days = summarizeWeatherDays(activities)
  assert.equal(days['2026-06-11'].windKph, 17)
  assert.equal(days['2026-06-11'].windDir, 'N')
  assert.equal(days['2026-06-11'].windGustKph, 32)
  assert.equal(days['2026-06-11'].activityCount, 2)
})

test('weatherSnapshotFromHours selects the nearest forecast and clamps precipitation chance', () => {
  assert.deepEqual(
    weatherSnapshotFromHours(
      { latitude: 43.641234, longitude: -79.412345 },
      [
        hour({
          forecastStart: '2026-06-11T13:00:00.000Z',
          temperature: 21.44,
          conditionCode: 'MostlyCloudy',
          precipitationChance: 1.2,
          precipitationType: 'rain',
        }),
        hour({ forecastStart: '2026-06-11T15:00:00.000Z', temperature: 24 }),
      ],
      Date.parse('2026-06-11T13:20:00.000Z'),
    ),
    {
      forecastStart: '2026-06-11T13:00:00.000Z',
      latitude: 43.64123,
      longitude: -79.41234,
      temperatureC: 21.4,
      conditionCode: 'MostlyCloudy',
      precipitationChance: 1,
      precipitationType: 'rain',
      source: 'weatherkit',
    },
  )
})

test('parseWeatherCache keeps valid activities and recomputes day summaries', () => {
  const cache = parseWeatherCache({
    version: 1,
    lastSync: 100,
    activities: {
      good: {
        activityId: 101,
        date: '2026-06-11',
        start: '2026-06-11T13:00:00.000Z',
        end: '2026-06-11T14:00:00.000Z',
        latitude: 43.64,
        longitude: -79.4,
        durationS: 3600,
        windKph: 18,
        windDir: 'SW',
        windDirDeg: 225,
        windGustKph: 28,
        averageRelativeHumidityPct: 64,
        relativeHumidityProvenance: {
          source: 'weatherkit',
          sourceKind: 'modeled',
          samplingMethod: 'route-hour',
          inputTimestamp: '2026-06-11T13:00:00.000Z',
          coveragePct: 100,
        },
        temperatureC: 24,
        temperatureSeries: [
          { elapsedS: 0, temperatureC: 23 },
          { elapsedS: 3600, temperatureC: 24 },
        ],
        source: 'weatherkit',
      },
      bad: { activityId: 102 },
    },
  })

  assert.equal(cache?.lastSync, 100)
  assert.equal(cache?.current, null)
  assert.equal(Object.keys(cache?.activities ?? {}).length, 1)
  assert.equal(cache?.days['2026-06-11'].windKph, 18)
  assert.equal(cache?.activities.good.averageRelativeHumidityPct, 64)
  assert.deepEqual(cache?.activities.good.temperatureSeries, [
    { elapsedS: 0, temperatureC: 23 },
    { elapsedS: 3600, temperatureC: 24 },
  ])
})

test('keeps matching NOAA current through weather cache parsing and route-hour refresh', () => {
  const activity = weatherActivityFromHours(candidate({ routeFingerprint: 'route' }), [
    hour({ windSpeed: 10, windDirection: 270 }),
    hour({ forecastStart: '2026-06-11T14:00:00.000Z', windSpeed: 20, windDirection: 270 }),
  ])
  assert.ok(activity)
  const current = currentForActivity(activity)
  const cache = parseWeatherCache({
    version: 5,
    lastSync: 100,
    activities: { '101': { ...activity, surfaceCurrent: current } },
  })
  assert.deepEqual(cache?.activities['101'].surfaceCurrent, current)
  const refreshed = weatherActivityFromRouteHours(
    candidate({ routeFingerprint: 'route' }),
    activity.routeHours ?? [],
    200,
    current,
  )
  assert.deepEqual(refreshed?.surfaceCurrent, current)
  assert.equal(refreshed?.fetchedAt, 200)
})

test('discards mismatched or malformed current while preserving independent WeatherKit data', () => {
  const activity = weatherActivityFromHours(candidate({ routeFingerprint: 'route' }), [
    hour({ windSpeed: 10, windDirection: 270 }),
    hour({ forecastStart: '2026-06-11T14:00:00.000Z', windSpeed: 20, windDirection: 270 }),
  ])
  assert.ok(activity)
  const current = currentForActivity(activity)
  const mismatches: SurfaceCurrentEstimate[] = [
    { ...current, activityId: 102 },
    { ...current, routeFingerprint: 'another-route' },
    { ...current, start: '2026-06-11T13:00:00.000Z' },
    { ...current, end: '2026-06-11T15:30:00.000Z' },
    { ...current, summary: { ...current.summary, elapsedDurationS: activity.durationS - 1 } },
  ]
  for (const surfaceCurrent of [
    undefined,
    null,
    {},
    { ...current, sourceKind: 'measured' },
    ...mismatches,
  ]) {
    const cache = parseWeatherCache({
      version: 5,
      lastSync: 100,
      activities: { '101': { ...activity, surfaceCurrent } },
    })
    assert.equal(cache?.activities['101'].surfaceCurrent, undefined)
    assert.equal(cache?.activities['101'].windKph, 17)
  }
  for (const surfaceCurrent of mismatches) {
    const refreshed = weatherActivityFromRouteHours(
      candidate({ routeFingerprint: 'route' }),
      activity.routeHours ?? [],
      200,
      surfaceCurrent,
    )
    assert.equal(refreshed?.surfaceCurrent, undefined)
    assert.equal(refreshed?.windKph, 17)
  }
})

test('parseWeatherCache retains a valid current WeatherKit snapshot', () => {
  const cache = parseWeatherCache({
    version: 3,
    lastSync: 100,
    current: {
      forecastStart: '2026-08-17T13:00:00.000Z',
      latitude: 43.64,
      longitude: -79.4,
      temperatureC: 19.5,
      conditionCode: 'Rain',
      precipitationChance: 0.72,
      precipitationType: 'rain',
    },
    activities: {},
  })

  assert.deepEqual(cache?.current, {
    forecastStart: '2026-08-17T13:00:00.000Z',
    latitude: 43.64,
    longitude: -79.4,
    temperatureC: 19.5,
    conditionCode: 'Rain',
    precipitationChance: 0.72,
    precipitationType: 'rain',
    source: 'weatherkit',
  })
})
