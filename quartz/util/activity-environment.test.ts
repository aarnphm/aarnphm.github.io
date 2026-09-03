import assert from 'node:assert/strict'
import test from 'node:test'
import type { WeatherActivity, WeatherRouteHour } from '../plugins/stores/weather'
import { buildActivityEnvironment, type ActivityEnvironmentInput } from './activity-environment'

const start = '2026-06-11T13:00:00.000Z'

const routeHour = (
  elapsedStartS: number,
  elapsedEndS: number,
  values: Partial<WeatherRouteHour> = {},
): WeatherRouteHour => ({
  forecastStart: new Date(Date.parse(start) + elapsedStartS * 1_000).toISOString(),
  overlapStart: new Date(Date.parse(start) + elapsedStartS * 1_000).toISOString(),
  overlapEnd: new Date(Date.parse(start) + elapsedEndS * 1_000).toISOString(),
  elapsedStartS,
  elapsedEndS,
  latitude: 43.64,
  longitude: -79.4,
  uvIndex: 4,
  cloudCover: 0.5,
  temperatureC: 20,
  windSpeedKph: 0,
  windDirectionDeg: 0,
  windGustKph: 0,
  relativeHumidity: 0.5,
  pressureHpa: 1_015,
  daylight: true,
  ...values,
})

const weather = (durationS: number, routeHours: WeatherRouteHour[]): WeatherActivity => ({
  activityId: 101,
  date: '2026-06-11',
  start,
  end: new Date(Date.parse(start) + durationS * 1_000).toISOString(),
  latitude: 43.64,
  longitude: -79.4,
  durationS,
  windKph: 0,
  windDir: 'N',
  windDirDeg: 0,
  windGustKph: 0,
  averageRelativeHumidityPct: 50,
  relativeHumidityProvenance: {
    source: 'weatherkit',
    sourceKind: 'modeled',
    samplingMethod: 'route-hour',
    inputTimestamp: start,
    coveragePct: 100,
  },
  temperatureC: 20,
  temperatureSeries: [
    { elapsedS: 0, temperatureC: 20 },
    { elapsedS: durationS, temperatureC: 20 },
  ],
  routeFingerprint: 'route',
  fetchedAt: Date.parse('2026-06-12T00:00:00Z'),
  routeHours,
  source: 'weatherkit',
})

const input = (
  durationS: number,
  routeHours: WeatherRouteHour[],
  values: Partial<ActivityEnvironmentInput> = {},
): ActivityEnvironmentInput => ({
  activityId: 101,
  elapsedTimeS: durationS,
  movingTimeS: durationS,
  timeS: [0, durationS],
  distanceM: [0, durationS * 4],
  latlng: [
    [43.64, -79.4],
    [43.65, -79.4],
  ],
  weather: weather(durationS, routeHours),
  attribution: null,
  computedAt: Date.parse('2026-06-12T01:00:00Z'),
  ...values,
})

test('integrates one hour at UVI 4 into 4 UVI-hours and 3.6 SED', () => {
  const result = buildActivityEnvironment(input(3_600, [routeHour(0, 3_600)]))

  assert.equal(result.environment?.summary.uviHours, 4)
  assert.equal(result.environment?.summary.ambientSed, 3.6)
  assert.equal(result.environment?.summary.averageUvIndex, 4)
  assert.equal(result.environment?.coverage.uvPct, 100)
  assert.deepEqual(
    result.environment?.samples.map(sample => sample.cumulativeMovingTelemetrySed),
    [0, 3.6],
  )
})

test('rejects a WeatherKit entry attached to another Strava activity', () => {
  const values = input(3_600, [routeHour(0, 3_600)])
  values.weather.activityId = 102

  assert.deepEqual(buildActivityEnvironment(values), { environment: null, apparentWind: null })
})

test('uses exact partial-hour overlap and overlap-weighted weather values', () => {
  const result = buildActivityEnvironment(
    input(5_400, [
      routeHour(0, 1_800, { uvIndex: 6, temperatureC: 10, cloudCover: 0.2 }),
      routeHour(1_800, 5_400, { uvIndex: 2, temperatureC: 25, cloudCover: 0.8 }),
    ]),
  )

  assert.equal(result.environment?.summary.uviHours, 5)
  assert.equal(result.environment?.summary.ambientSed, 4.5)
  assert.equal(result.environment?.summary.averageUvIndex, 3.33)
  assert.equal(result.environment?.summary.averageAmbientTemperatureC, 20)
  assert.equal(result.environment?.summary.averageCloudCoverPct, 60)
})

test('preserves nighttime UVI zero and counts exposure while route distance is paused', () => {
  const result = buildActivityEnvironment(
    input(3_600, [routeHour(0, 3_600, { uvIndex: 0, daylight: false })], {
      distanceM: [0, 0],
      movingTimeS: 0,
    }),
  )

  assert.equal(result.environment?.summary.averageUvIndex, 0)
  assert.equal(result.environment?.summary.uviHours, 0)
  assert.equal(result.environment?.summary.ambientSed, 0)
  assert.equal(result.environment?.summary.daylightCoveragePct, 0)
  assert.equal(result.environment?.doseClocks.movingTelemetrySed, null)
})

test('retains partial traces while gaps suppress cumulative dose', () => {
  const result = buildActivityEnvironment(
    input(
      3_600,
      [
        routeHour(0, 1_200),
        routeHour(2_400, 3_600, { uvIndex: 2, temperatureC: null, cloudCover: null }),
      ],
      {
        timeS: [0, 1_200, 1_800, 2_400, 3_600],
        distanceM: [0, 4_800, 7_200, 9_600, 14_400],
        latlng: [
          [43.64, -79.4],
          [43.641, -79.4],
          [43.642, -79.4],
          [43.643, -79.4],
          [43.644, -79.4],
        ],
      },
    ),
  )

  assert.equal(result.environment?.summary.uviHours, null)
  assert.equal(result.environment?.summary.ambientSed, null)
  assert.equal(result.environment?.coverage.uvPct, 66.7)
  assert(result.environment?.samples.some(sample => sample.uvIndex === null))
  assert(result.environment?.samples.every(sample => sample.cumulativeSed === null))
  assert(result.environment?.samples.every(sample => sample.cumulativeMovingTelemetrySed === null))
})

test('uses WeatherKit UVI without applying cloud attenuation a second time', () => {
  const clear = buildActivityEnvironment(
    input(3_600, [routeHour(0, 3_600, { uvIndex: 5, cloudCover: 0 })]),
  )
  const overcast = buildActivityEnvironment(
    input(3_600, [routeHour(0, 3_600, { uvIndex: 5, cloudCover: 1 })]),
  )

  assert.equal(clear.environment?.summary.ambientSed, 4.5)
  assert.equal(overcast.environment?.summary.ambientSed, 4.5)
})

test('resolves meteorological from-direction into headwind, tailwind, and crosswind', () => {
  const northbound: Partial<ActivityEnvironmentInput> = {
    timeS: [0, 10],
    distanceM: [0, 100],
    latlng: [
      [43.64, -79.4],
      [43.641, -79.4],
    ],
    movingTimeS: 10,
  }
  const headwind = buildActivityEnvironment(
    input(10, [routeHour(0, 10, { windSpeedKph: 18, windDirectionDeg: 0 })], northbound),
  ).apparentWind
  const tailwind = buildActivityEnvironment(
    input(10, [routeHour(0, 10, { windSpeedKph: 18, windDirectionDeg: 180 })], northbound),
  ).apparentWind
  const crosswind = buildActivityEnvironment(
    input(10, [routeHour(0, 10, { windSpeedKph: 18, windDirectionDeg: 90 })], northbound),
  ).apparentWind
  const oppositeCrosswind = buildActivityEnvironment(
    input(10, [routeHour(0, 10, { windSpeedKph: 18, windDirectionDeg: 270 })], northbound),
  ).apparentWind

  assert.equal(headwind?.summary.averageHeadwindKph, 18)
  assert.equal(headwind?.summary.headwindSharePct, 100)
  assert.equal(tailwind?.summary.averageHeadwindKph, -18)
  assert.equal(tailwind?.summary.tailwindTimeS, 10)
  assert.equal(crosswind?.summary.averageHeadwindKph, 0)
  assert.equal(crosswind?.summary.averageCrosswindKph, 18)
  assert.equal(crosswind?.summary.maximumCrosswindKph, 18)
  assert((crosswind?.summary.averageYawDeg ?? 0) > 0)
  assert.equal(oppositeCrosswind?.summary.averageHeadwindKph, 0)
  assert.equal(oppositeCrosswind?.summary.averageCrosswindKph, -18)
  assert.equal(oppositeCrosswind?.summary.maximumCrosswindKph, 18)
  assert((oppositeCrosswind?.summary.averageYawDeg ?? 0) < 0)
})

test('represents calm air and rejects low-speed and telemetry-gap intervals', () => {
  const calm = buildActivityEnvironment(
    input(10, [routeHour(0, 10)], {
      timeS: [0, 10],
      distanceM: [0, 100],
      latlng: [
        [43.64, -79.4],
        [43.641, -79.4],
      ],
      movingTimeS: 10,
    }),
  )
  const lowSpeed = buildActivityEnvironment(
    input(10, [routeHour(0, 10)], {
      timeS: [0, 10],
      distanceM: [0, 20],
      latlng: [
        [43.64, -79.4],
        [43.6401, -79.4],
      ],
      movingTimeS: 10,
    }),
  )
  const gap = buildActivityEnvironment(
    input(60, [routeHour(0, 60)], {
      timeS: [0, 60],
      distanceM: [0, 600],
      latlng: [
        [43.64, -79.4],
        [43.645, -79.4],
      ],
      movingTimeS: 60,
    }),
  )

  assert.equal(calm.apparentWind?.summary.averageHeadwindKph, 0)
  assert.equal(calm.apparentWind?.summary.apparentAirRatio, 1)
  assert.equal(lowSpeed.apparentWind, null)
  assert.equal(gap.apparentWind, null)
})
