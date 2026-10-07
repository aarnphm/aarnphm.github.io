import assert from 'node:assert/strict'
import test from 'node:test'
import type { WeatherActivity, WeatherRouteHour } from '../plugins/stores/weather'
import type { SurfaceCurrentEstimate } from './surface-current'
import {
  buildActivityEnvironment,
  surfaceCurrentChartSamples,
  type ActivityEnvironmentInput,
} from './activity-environment'

const start = '2026-06-11T13:00:00.000Z'

const surfaceCurrent = (durationS: number): SurfaceCurrentEstimate => ({
  source: 'noaa-loofs',
  sourceKind: 'modeled',
  formulaId: 'garden-surface-current-v1',
  formulaVersion: 1,
  activityId: 101,
  routeFingerprint: 'route',
  start,
  end: new Date(Date.parse(start) + durationS * 1_000).toISOString(),
  computedAt: Date.parse('2026-06-12T01:00:00Z'),
  spatialSamplingModel: 'containing-element',
  temporalSamplingModel: 'hourly-linear-vector',
  layer: 0,
  summary: {
    averageSpeedMps: 1,
    averageDirectionDeg: 90,
    coveragePct: 100,
    coveredDurationS: durationS,
    elapsedDurationS: durationS,
  },
  samples: [0, durationS].map(elapsedS => ({
    elapsedS,
    speedMps: 1,
    directionDeg: 90,
    uMps: 1,
    vMps: 0,
    element: 1,
    validTime: new Date(
      Math.floor((Date.parse(start) + elapsedS * 1_000) / 3_600_000) * 3_600_000,
    ).toISOString(),
    cycleTime: '2026-06-11T18:00:00.000Z',
    sourceUrl:
      elapsedS === 0
        ? 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/2026/06/11/loofs.t18z.20260611.fields.n005.nc.ascii'
        : 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/2026/06/11/loofs.t18z.20260611.fields.n004.nc.ascii',
  })),
})

test('surface-current chart projection preserves gaps without fabricating weather or distance', () => {
  const samples = surfaceCurrent(120).samples
  const first = samples[0]
  const final = samples[1]
  const gap = {
    elapsedS: 60,
    speedMps: null,
    directionDeg: null,
    uMps: null,
    vMps: null,
    element: null,
    validTime: null,
    cycleTime: null,
    sourceUrl: null,
  }
  assert.deepEqual(surfaceCurrentChartSamples([first, gap, { ...final, speedMps: 0 }]), [
    { elapsedS: 0, surfaceCurrentSpeedMps: 1, surfaceCurrentDirectionDeg: 90 },
    { elapsedS: 60, surfaceCurrentSpeedMps: null, surfaceCurrentDirectionDeg: null },
    { elapsedS: 120, surfaceCurrentSpeedMps: 0, surfaceCurrentDirectionDeg: 90 },
  ])
  assert.deepEqual(surfaceCurrentChartSamples([]), [])
})

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
  openWater: false,
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

test('resolves 10 m from-direction wind into rider-height headwind, tailwind, and crosswind', () => {
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

  // 18 km/h at 10 m is 5.1 km/h at 1 m over suburban roughness.
  assert.equal(headwind?.summary.averageHeadwindKph, 5.1)
  assert.equal(headwind?.summary.headwindSharePct, 100)
  assert.equal(headwind?.summary.apparentAirRatio, 1.142)
  assert.equal(tailwind?.summary.averageHeadwindKph, -5.1)
  assert.equal(tailwind?.summary.tailwindTimeS, 10)
  assert.equal(crosswind?.summary.averageHeadwindKph, 0)
  assert.equal(crosswind?.summary.averageCrosswindKph, 5.1)
  assert.equal(crosswind?.summary.maximumCrosswindKph, 5.1)
  assert((crosswind?.summary.averageYawDeg ?? 0) > 0)
  assert.equal(oppositeCrosswind?.summary.averageHeadwindKph, 0)
  assert.equal(oppositeCrosswind?.summary.averageCrosswindKph, -5.1)
  assert.equal(oppositeCrosswind?.summary.maximumCrosswindKph, 5.1)
  assert((oppositeCrosswind?.summary.averageYawDeg ?? 0) < 0)
})

test('keeps open-water swim pace and resolves wind at the water surface', () => {
  const swimPace = {
    timeS: [0, 20],
    distanceM: [0, 11],
    latlng: [
      [43.64, -79.4],
      [43.6401, -79.4],
    ] as [number, number][],
    movingTimeS: 20,
  }
  const northWind = [routeHour(0, 20, { windSpeedKph: 18, windDirectionDeg: 0 })]
  const swim = buildActivityEnvironment(input(20, northWind, { ...swimPace, openWater: true }))
  const land = buildActivityEnvironment(input(20, northWind, swimPace))

  // 18 km/h at 10 m is 11.5 km/h at 0.2 m over open water.
  assert.equal(swim.apparentWind?.summary.averageHeadwindKph, 11.5)
  assert.equal(swim.apparentWind?.summary.coveragePct, 100)
  assert.equal(land.apparentWind?.summary.coveragePct ?? 0, 0)
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
      distanceM: [0, 5],
      latlng: [
        [43.64, -79.4],
        [43.64005, -79.4],
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

test('keeps running pace and measures headwind sections across short reversals and stops', () => {
  const runningRoute = (
    legs: { seconds: number; heading: 'north' | 'south' | 'stop' }[],
  ): Partial<ActivityEnvironmentInput> & { durationS: number } => {
    const stepDegrees = 2.5 / 111_195
    const timeS = [0]
    const distanceM = [0]
    const latlng: [number, number][] = [[43.64, -79.4]]
    for (const leg of legs) {
      if (leg.heading === 'stop') {
        timeS.push(timeS.at(-1)! + leg.seconds)
        distanceM.push(distanceM.at(-1)!)
        latlng.push(latlng.at(-1)!)
        continue
      }
      for (let second = 0; second < leg.seconds; second += 1) {
        const [latitude, longitude] = latlng.at(-1)!
        timeS.push(timeS.at(-1)! + 1)
        distanceM.push(distanceM.at(-1)! + 2.5)
        latlng.push([latitude + (leg.heading === 'north' ? stepDegrees : -stepDegrees), longitude])
      }
    }
    const durationS = timeS.at(-1)!
    return { timeS, distanceM, latlng, movingTimeS: durationS, durationS }
  }
  const apparentWind = (legs: Parameters<typeof runningRoute>[0]) => {
    const { durationS, ...route } = runningRoute(legs)
    return buildActivityEnvironment(
      input(durationS, [routeHour(0, durationS, { windSpeedKph: 18, windDirectionDeg: 0 })], route),
    ).apparentWind
  }
  const outAndBack = [
    { seconds: 120, heading: 'north' },
    { seconds: 10, heading: 'south' },
    { seconds: 100, heading: 'north' },
  ] as const
  const longStop = apparentWind([
    ...outAndBack,
    { seconds: 300, heading: 'stop' },
    { seconds: 60, heading: 'north' },
  ])
  const trafficLight = apparentWind([
    ...outAndBack,
    { seconds: 60, heading: 'stop' },
    { seconds: 60, heading: 'north' },
  ])

  // 2.5 m/s running pace is inside the wind estimate; the 10 s reversal stays in the section.
  assert.equal(longStop?.summary.averageGroundSpeedKph, 9)
  assert.equal(longStop?.summary.headwindTimeS, 280)
  assert.equal(longStop?.summary.tailwindTimeS, 10)
  assert.equal(longStop?.summary.longestHeadwindS, 230)
  assert.equal(trafficLight?.summary.longestHeadwindS, 350)
})

test('retains hourly ambient wind for slow outdoor routes without meteorological direction', () => {
  const result = buildActivityEnvironment(
    input(
      20,
      [
        routeHour(0, 5, { windSpeedKph: 12, windDirectionDeg: null }),
        routeHour(5, 20, { windSpeedKph: 20, windDirectionDeg: null }),
      ],
      {
        timeS: [0, 5, 10, 20],
        distanceM: [0, 5, 10, 20],
        latlng: [
          [43.64, -79.4],
          [43.64005, -79.4],
          [43.6401, -79.4],
          [43.6402, -79.4],
        ],
      },
    ),
  )

  assert.equal(result.environment?.summary.averageWindSpeedKph, 18)
  assert.equal(result.environment?.coverage.windPct, 100)
  assert.deepEqual(
    result.environment?.samples.map(sample => sample.windSpeedKph),
    [12, 20, 20, 20],
  )
  assert.equal(result.apparentWind, null)
  assert(result.environment?.samples.every(sample => sample.apparentAirSpeedKph === null))
})

test('reports ambient wind coverage independently and preserves missing hourly speed', () => {
  const result = buildActivityEnvironment(
    input(20, [routeHour(0, 10, { windSpeedKph: 0 }), routeHour(10, 20, { windSpeedKph: null })], {
      timeS: [0, 5, 10, 20],
      distanceM: [0, 5, 10, 20],
      latlng: [
        [43.64, -79.4],
        [43.64005, -79.4],
        [43.6401, -79.4],
        [43.6402, -79.4],
      ],
    }),
  )

  assert.equal(result.environment?.summary.averageWindSpeedKph, 0)
  assert.equal(result.environment?.coverage.windPct, 50)
  assert.deepEqual(
    result.environment?.samples.map(sample => sample.windSpeedKph),
    [0, 0, null, null],
  )
})

test('rejects invalid ambient wind speeds without discarding other weather estimates', () => {
  for (const windSpeedKph of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const result = buildActivityEnvironment(input(20, [routeHour(0, 20, { windSpeedKph })]))

    assert.equal(result.environment?.summary.averageWindSpeedKph, null)
    assert.equal(result.environment?.coverage.windPct, 0)
    assert(result.environment?.samples.every(sample => sample.windSpeedKph === null))
    assert.equal(result.environment?.summary.averageUvIndex, 4)
    assert.equal(result.apparentWind, null)
  }
})

test('projects matching NOAA surface current with modeled provenance and no private route identity', () => {
  const current = surfaceCurrent(3_600)
  const result = buildActivityEnvironment(
    input(3_600, [routeHour(0, 3_600)], {
      weather: { ...weather(3_600, [routeHour(0, 3_600)]), surfaceCurrent: current },
    }),
  )

  const { routeFingerprint: _fingerprint, ...projected } = current
  assert.deepEqual(result.environment?.surfaceCurrent, projected)
  assert.equal(result.environment?.surfaceCurrent?.sourceKind, 'modeled')
  assert.equal(result.environment?.summary.averageWindSpeedKph, 0)
})

test('suppresses surface current with mismatched identity or interval while retaining weather', () => {
  const current = surfaceCurrent(3_600)
  for (const mismatch of [
    { ...current, activityId: 102 },
    { ...current, routeFingerprint: 'another-route' },
    { ...current, start: '2026-06-11T12:00:00.000Z', end: start },
    { ...current, end: '2026-06-11T15:00:00.000Z' },
    { ...current, summary: { ...current.summary, elapsedDurationS: 3_599 } },
  ]) {
    const result = buildActivityEnvironment(
      input(3_600, [routeHour(0, 3_600)], {
        weather: { ...weather(3_600, [routeHour(0, 3_600)]), surfaceCurrent: mismatch },
      }),
    )
    assert.equal(result.environment?.surfaceCurrent, undefined)
    assert.equal(result.environment?.summary.averageUvIndex, 4)
  }
})
