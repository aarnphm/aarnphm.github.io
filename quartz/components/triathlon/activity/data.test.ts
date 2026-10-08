import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import test from 'node:test'
import type { PublicSurfaceCurrentEstimate } from '../../../util/surface-current'
import { buildAnalytics } from '../../../plugins/stores/analytics'
import { emptyWahooMetrics } from '../../../plugins/stores/wahoo'
import { buildCyclingTorqueTrace, cyclingTorqueSamples } from '../../../util/cycling-torque'
import { estimateHeartRatePhysiology } from '../../../util/heart-rate-physiology'
import { STRAVA_DETAIL_INDEX_KIND } from '../../../util/strava-detail'
import { estimateSwimPhysiology } from '../../../util/swim-physiology'
import { buildSwimPowerEstimate } from '../../../util/swim-power'
import { buildTriathlonDailyAnalytics } from '../../../util/triathlon-day-analytics'
import { detailContextFromPayload, isActivityDetail, readDetailPayload } from './data'

const emptyAnalyses = {
  native: { myWindsock: null, pelotan: null },
  derived: { environment: null, uvScore: null, apparentWind: null },
}

const detail = (id: number, date: string, sport: string): Record<string, unknown> => ({
  id,
  date,
  sport,
  device: null,
  staminaTrace: null,
  performanceConditionTrace: null,
  elapsedTimeS: 3_600,
  deviceTemperatureC: null,
  ambientTemperatureC: null,
  runWalk: null,
  route: [],
  heartRateTrace: [],
  analyses: emptyAnalyses,
})

test('swim estimates survive JSON and reject invalid values or use on another sport', () => {
  const samples = Array.from({ length: 61 }, (_, i) => ({
    elapsedS: i * 10,
    distanceKm: i * 0.007,
    heartRate: 140,
    speedMps: 0.7,
    strokeRateSpm: 24,
  }))
  const swimPhysiology = estimateSwimPhysiology(samples, 200, 'activity-average')
  const swimPower = buildSwimPowerEstimate(
    'route',
    [{ startElapsedS: 0, endElapsedS: 600, durationS: 600, distanceM: 420, stroke: null }],
    'ground-speed',
  )
  assert.ok(swimPhysiology)
  assert.ok(swimPower)
  const swim = { ...detail(1, '2026-09-27', 'swim'), swimPhysiology, swimPower }
  assert.equal(isActivityDetail(JSON.parse(JSON.stringify(swim))), true)
  assert.equal(isActivityDetail({ ...swim, sport: 'bike' }), false)
  for (const histogramS of [undefined, [], [-1], [Number.NaN], [601], '600'])
    assert.equal(isActivityDetail({ ...swim, swimPower: { ...swimPower, histogramS } }), false)
  assert.equal(
    isActivityDetail({ ...swim, swimPhysiology: { ...swimPhysiology, baselineSpeedMps: 0 } }),
    false,
  )
  assert.equal(
    isActivityDetail({
      ...swim,
      swimPower: { ...swimPower, curve: [{ ...swimPower.curve[0], index: -1 }] },
    }),
    false,
  )
})

test('validates nonempty authored computer text for cycling details', () => {
  const activity = detail(1, '2026-09-17', 'bike')
  assert.equal(isActivityDetail(activity), true)
  assert.equal(isActivityDetail({ ...activity, computerOverride: 'Wahoo ELEMNT BOLT 3' }), true)
  for (const computerOverride of ['', '   ', null, 3, {}])
    assert.equal(isActivityDetail({ ...activity, computerOverride }), false)
  assert.equal(
    isActivityDetail({ ...activity, sport: 'run', computerOverride: 'Wahoo ELEMNT BOLT 3' }),
    false,
  )
})

test('retains calculated run efforts through JSON and rejects invalid telemetry', () => {
  const effort = {
    label: '400m',
    targetDistanceM: 400,
    elapsedTimeS: 114,
    averageSpeedKph: 12.63,
    averageHeartRate: null,
    elevationDeltaM: null,
  }
  const bestEfforts = {
    distanceSource: 'calculated',
    weightKg: null,
    weightDate: null,
    distance: [effort],
    power: [],
    climbs: [],
  }
  const activity = { ...detail(1, '2026-09-24', 'run'), bestEfforts }
  const serialized: unknown = JSON.parse(JSON.stringify(activity))
  assert.ok(isActivityDetail(serialized))
  assert.deepEqual(serialized.bestEfforts, bestEfforts)
  assert.equal(isActivityDetail({ ...activity, bestEfforts: null }), true)
  for (const invalid of [
    { ...bestEfforts, distanceSource: 'strava' },
    ...[
      { targetDistanceM: 0 },
      { elapsedTimeS: -1 },
      { averageSpeedKph: NaN },
      { averageHeartRate: 0 },
      { elevationDeltaM: 'unknown' },
    ].map(fields => ({ ...bestEfforts, distance: [{ ...effort, ...fields }] })),
  ])
    assert.equal(isActivityDetail({ ...activity, bestEfforts: invalid }), false)
})

test('validates power curve weight on or before the activity date and retains it through JSON', () => {
  const weight = { kg: 87.09, date: '2026-09-17', source: 'garmin' }
  for (const sport of ['bike', 'run']) {
    const activity = detail(1, weight.date, sport)
    const serialized: unknown = JSON.parse(
      JSON.stringify({ ...activity, powerCurveWeight: weight }),
    )
    assert.ok(isActivityDetail(serialized))
    assert.deepEqual(serialized.powerCurveWeight, weight)
    assert.equal(isActivityDetail(activity), true)
    const earlierWeight = { ...weight, date: '2026-09-16' }
    const earlier: unknown = JSON.parse(
      JSON.stringify({ ...activity, powerCurveWeight: earlierWeight }),
    )
    assert.ok(isActivityDetail(earlier))
    assert.deepEqual(earlier.powerCurveWeight, earlierWeight)
    for (const powerCurveWeight of [
      null,
      {},
      ...[0, -1, NaN, Infinity, '87.09'].map(kg => ({ ...weight, kg })),
      { ...weight, date: '2026-09-18' },
      ...['', '2026-9-16', 'unknown'].map(date => ({ ...weight, date })),
      { ...weight, source: 'strava' },
    ])
      assert.equal(isActivityDetail({ ...activity, powerCurveWeight }), false)
  }
})

test('validates equipment identity, nullable names, and Strava provenance', () => {
  const activity = detail(1, '2026-09-15', 'bike')
  for (const name of ['Speedmax', null])
    assert.equal(
      isActivityDetail({ ...activity, equipment: { id: 'b123', name, source: 'strava' } }),
      true,
    )
  for (const equipment of [
    null,
    {},
    { id: '', name: 'Speedmax', source: 'strava' },
    { id: 'b123', name: '', source: 'strava' },
    { id: 'b123', name: 'Speedmax', source: 'garmin' },
  ])
    assert.equal(isActivityDetail({ ...activity, equipment }), false)
  assert.equal(isActivityDetail(activity), true)
})

test('loads activity shards alongside manual sauna daily analytics', async () => {
  const analytics = buildAnalytics(null)
  analytics.heat.series = [
    {
      date: '2026-08-02',
      temperatureC: null,
      heatStrainIndex: null,
      source: 'manual-sauna',
      observedMinutes: 65,
      hotMinutes: 0,
      saunaMinutes: 65,
      saunaHtl: 7.7,
      dose: 0.77,
      acclimatisationPct: 100,
    },
  ]
  const dailyAnalytics = buildTriathlonDailyAnalytics(analytics)
  const runPowerCurveRef = [{ s: 1, w: 700, activityId: 1, activityDate: '2026-07-31' }]
  const runPowerCurveYearRef = [{ s: 1, w: 800, activityId: 6, activityDate: '2026-01-01' }]
  const requested: string[] = []
  const server = createServer((request, response) => {
    const path = request.url ?? ''
    requested.push(path)
    const values: Record<string, unknown> = {
      '/static/strava-detail.json': {
        kind: STRAVA_DETAIL_INDEX_KIND,
        shards: ['strava-detail/2026-08.json', 'strava-detail/2026-07.json'],
        health: {},
        dailyAnalytics,
        runPowerCurveRef,
        runPowerCurveYearRef,
        ftp: 250,
      },
      '/static/strava-detail/2026-08.json': {
        details: {
          '2': detail(2, '2026-08-02', 'bike'),
          '3': detail(3, '2026-08-03', 'walk'),
          '4': detail(4, '2026-08-04', 'yoga'),
          '5': detail(5, '2026-08-05', 'treatment'),
        },
      },
      '/static/strava-detail/2026-07.json': { details: { '1': detail(1, '2026-07-31', 'run') } },
    }
    const value = values[path]
    if (!value) {
      response.writeHead(404)
      response.end()
      return
    }
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(value))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  try {
    const controller = new AbortController()
    const response = await fetch(`http://127.0.0.1:${address.port}/static/strava-detail.json`)
    const payload = await readDetailPayload(response, controller.signal)
    assert.deepEqual(Object.keys(payload.details).sort(), ['1', '2', '3', '4', '5'])
    assert.equal(payload.details['1'].date, '2026-07-31')
    assert.equal(payload.details['2'].date, '2026-08-02')
    assert.equal(payload.details['3'].sport, 'walk')
    assert.equal(payload.details['4'].sport, 'yoga')
    assert.equal(payload.details['5'].sport, 'treatment')
    assert.equal(payload.ftp, 250)
    assert.deepEqual(detailContextFromPayload(payload).runCurveRef, runPowerCurveRef)
    assert.deepEqual(detailContextFromPayload(payload).runCurveYearRef, runPowerCurveYearRef)
    assert.deepEqual(payload.dailyAnalytics, dailyAnalytics)
    assert.equal(payload.dailyAnalytics?.['2026-08-02'].heat?.source, 'manual-sauna')
    assert.deepEqual(requested.sort(), [
      '/static/strava-detail.json',
      '/static/strava-detail/2026-07.json',
      '/static/strava-detail/2026-08.json',
    ])
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    )
  }
})

const gardenEnvironment = (samples: Record<string, unknown>[]) => ({
  source: 'garden-estimate',
  formulaId: 'garden-environment-v1',
  formulaVersion: 1,
  inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1',
  normalizationVersion: 1,
  computedAt: 1,
  inputAsOf: 1,
  temporalSamplingModel: 'weatherkit-hourly-piecewise-constant',
  spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint',
  summary: {
    averageUvIndex: 0,
    peakUvIndex: 0,
    uviHours: 0,
    ambientSed: 0,
    averageAmbientTemperatureC: 0,
    averageCloudCoverPct: 0,
    daylightCoveragePct: 0,
    weatherCoveragePct: 100,
    coveredDurationS: 3_600,
    elapsedDurationS: 3_600,
  },
  doseClocks: { elapsedSed: 0, movingTelemetrySed: 0 },
  coverage: { weatherPct: 100, uvPct: 100, temperaturePct: 100, cloudPct: 100, daylightPct: 100 },
  samples,
  attribution: null,
})

const environmentSample = (elapsedS: number, distanceKm: number): Record<string, unknown> => ({
  elapsedS,
  distanceKm,
  uvIndex: 0,
  cumulativeSed: 0,
  cumulativeMovingTelemetrySed: 0,
  ambientTemperatureC: 0,
  cloudCoverPct: 0,
  headwindKph: 0,
  crosswindKph: 0,
  apparentAirSpeedKph: 0,
  yawDeg: 0,
})

test('accepts synced Wahoo provenance without a local FIT path', () => {
  const wahoo = {
    activityId: 'wahoo:495060543',
    fitPath: null,
    sha256: 'a'.repeat(64),
    sourceDevice: 'ELEMNT BOLT',
    startOffsetS: 0,
    distanceM: 48_513.64,
    metrics: { ...emptyWahooMetrics(), intensityFactor: 0, trainingStressScore: 0 },
    summarySources: { npWatts: 'wahoo' },
    streamFallback: 'strava',
  }
  const value = { ...detail(20092789179, '2026-09-08', 'bike'), wahoo }
  assert.equal(isActivityDetail(JSON.parse(JSON.stringify(value))), true)
  assert.equal(
    isActivityDetail({ ...value, wahoo: { ...wahoo, fitPath: 'triathlon/wahoo/ride.fit' } }),
    true,
  )
  assert.equal(
    isActivityDetail({ ...value, wahoo: { ...wahoo, fitPath: '/private/ride.fit' } }),
    false,
  )
})

test('validates public analysis contracts while preserving numeric zero', () => {
  const value = detail(9, '2026-08-09', 'bike')
  value.deviceTemperatureC = 0
  value.ambientTemperatureC = 0
  value.analyses = {
    native: { myWindsock: null, pelotan: null },
    derived: {
      environment: gardenEnvironment([environmentSample(0, 0), environmentSample(3_600, 20)]),
      uvScore: null,
      apparentWind: null,
    },
  }
  assert.equal(isActivityDetail(value), true)
})

test('preserves legacy environment data and validates optional ambient wind through JSON', () => {
  const validate = (environment: Record<string, unknown>): boolean =>
    isActivityDetail({
      ...detail(9, '2026-09-27', 'swim'),
      analyses: {
        native: { myWindsock: null, pelotan: null },
        derived: { environment, uvScore: null, apparentWind: null },
      },
    })
  const samples = [environmentSample(0, 0), environmentSample(3_600, 1)]
  const legacy = gardenEnvironment(samples)
  assert.equal(validate(legacy), true)
  for (const windSpeedKph of [null, 0, 18]) {
    const environment = {
      ...legacy,
      summary: { ...gardenEnvironment([]).summary, averageWindSpeedKph: windSpeedKph },
      coverage: {
        weatherPct: 100,
        uvPct: 100,
        temperaturePct: 100,
        cloudPct: 100,
        daylightPct: 100,
        windPct: 50,
      },
      samples: samples.map(sample => ({ ...sample, windSpeedKph })),
    }
    const value = {
      ...detail(9, '2026-09-27', 'swim'),
      analyses: {
        native: { myWindsock: null, pelotan: null },
        derived: { environment, uvScore: null, apparentWind: null },
      },
    }
    const serialized: unknown = JSON.parse(JSON.stringify(value))
    assert.ok(isActivityDetail(serialized))
    assert.equal(serialized.analyses.derived.environment?.summary.averageWindSpeedKph, windSpeedKph)
    assert.equal(serialized.analyses.derived.environment?.samples[0].windSpeedKph, windSpeedKph)
  }

  for (const windSpeedKph of [-1, 1_001, NaN, Infinity, '18', true, {}, undefined]) {
    assert.equal(
      validate({ ...legacy, samples: samples.map(sample => ({ ...sample, windSpeedKph })) }),
      false,
    )
    assert.equal(
      validate({
        ...legacy,
        summary: { ...gardenEnvironment([]).summary, averageWindSpeedKph: windSpeedKph },
      }),
      false,
    )
  }
  for (const windPct of [-1, 101, NaN, Infinity, null, '50', {}, undefined])
    assert.equal(
      validate({
        ...legacy,
        coverage: {
          weatherPct: 100,
          uvPct: 100,
          temperaturePct: 100,
          cloudPct: 100,
          daylightPct: 100,
          windPct,
        },
      }),
      false,
    )
})

test('validates public NOAA modeled current against activity identity and recorded interval', () => {
  const start = '2026-09-27T18:47:25.000Z'
  const current: PublicSurfaceCurrentEstimate = {
    source: 'noaa-loofs',
    sourceKind: 'modeled',
    formulaId: 'garden-surface-current-v1',
    formulaVersion: 1,
    activityId: 9,
    start,
    end: '2026-09-27T19:47:25.000Z',
    computedAt: Date.parse('2026-10-02T20:00:00Z'),
    spatialSamplingModel: 'containing-element',
    temporalSamplingModel: 'hourly-linear-vector',
    layer: 0,
    summary: {
      averageSpeedMps: 1,
      averageDirectionDeg: 90,
      coveragePct: 100,
      coveredDurationS: 3_600,
      elapsedDurationS: 3_600,
    },
    samples: [0, 3_600].map(elapsedS => ({
      elapsedS,
      speedMps: 1,
      directionDeg: 90,
      uMps: 1,
      vMps: 0,
      element: 1,
      validTime: new Date(
        Math.floor((Date.parse(start) + elapsedS * 1_000) / 3_600_000) * 3_600_000,
      ).toISOString(),
      cycleTime: '2026-09-28T00:00:00.000Z',
      sourceUrl:
        elapsedS === 0
          ? 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/2026/09/28/loofs.t00z.20260928.fields.n006.nc.ascii'
          : 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/2026/09/28/loofs.t00z.20260928.fields.n005.nc.ascii',
    })),
  }
  const validate = (surfaceCurrent: unknown, activityStart = start): boolean =>
    isActivityDetail({
      ...detail(9, '2026-09-27', 'swim'),
      start: activityStart,
      analyses: {
        native: { myWindsock: null, pelotan: null },
        derived: {
          environment: {
            ...gardenEnvironment([environmentSample(0, 0), environmentSample(3_600, 1)]),
            surfaceCurrent,
          },
          uvScore: null,
          apparentWind: null,
        },
      },
    })
  assert.equal(validate(current), true)
  assert.equal(validate(null), true)
  assert.equal(validate(undefined), true)
  assert.equal(validate(current, '2026-09-27T17:47:25.000Z'), false)
  for (const surfaceCurrent of [
    {},
    { ...current, activityId: 10 },
    { ...current, sourceKind: 'measured' },
    { ...current, routeFingerprint: 'private' },
    { ...current, start: '2026-09-27T17:47:25.000Z', end: start },
    { ...current, end: '2026-09-27T20:47:25.000Z' },
    { ...current, summary: { ...current.summary, averageSpeedMps: -1 } },
    { ...current, summary: { ...current.summary, elapsedDurationS: 3_599 } },
  ])
    assert.equal(validate(surfaceCurrent), false)
})

test('validates manual activity moves and their per-side repetition contract', () => {
  const value = detail(20093785889, '2026-09-08', 'treatment')
  const entries = [
    { name: 'Roll Eagle', sets: [{ repetitions: 4, perSide: true }] },
    { name: 'Roll Center', sets: [{ repetitions: 4, perSide: false }] },
  ]
  value.moves = { source: 'manual', entries }
  assert.equal(isActivityDetail(value), true)
  assert.equal(
    isActivityDetail({
      ...value,
      moves: {
        source: 'manual',
        entries: [{ name: 'Roll Eagle', sets: [{ repetitions: 0, perSide: true }] }],
      },
    }),
    false,
  )
  assert.equal(isActivityDetail({ ...value, moves: { source: 'strava', entries } }), false)
})

test('validates closed activity devices and independent thermal provenance', () => {
  const value = detail(14, '2026-08-14', 'run')
  value.device = 'apple-watch-ultra-3'
  const thermal = {
    heatStrainIndex: 0,
    heatStrainSource: 'core-fit',
    coreTemperatureC: 37.5,
    coreTemperatureSource: 'core-app',
    skinTemperatureC: null,
    skinTemperatureSource: null,
  }
  value.route = [thermal]
  assert.equal(isActivityDetail(value), true)

  const garmin = { ...value, device: 'garmin-forerunner-970' }
  assert.equal(isActivityDetail(garmin), true)
  assert.equal(isActivityDetail({ ...value, device: 'Apple Watch Ultra 3' }), false)
  assert.equal(
    isActivityDetail({ ...value, route: [{ ...thermal, heatStrainSource: 'watch' }] }),
    false,
  )
  const missing = { ...value }
  delete missing.device
  assert.equal(isActivityDetail(missing), false)
})

test('validates native Garmin running dynamics and run/walk segments', () => {
  const value = detail(15, '2026-09-03', 'run')
  value.route = [
    {
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
      performanceCondition: -10,
      strideLengthM: 1.08,
      verticalRatioPct: 11.3,
      verticalOscillationCm: 12.4,
      groundContactBalanceLeftPct: 49.3,
      groundContactTimeMs: 246.5,
      stepSpeedLossMps: 0.079,
      stepSpeedLossPct: 2.78,
      impactLoadFactor: 1,
    },
  ]
  value.staminaTrace = {
    source: 'garmin',
    method: 'garmin-native',
    ftpWatts: null,
    maxHeartRateBpm: null,
  }
  value.performanceConditionTrace = { source: 'garmin', method: 'garmin-native' }
  const runWalk = {
    source: 'garmin',
    elapsedTimeS: 1800.479,
    runTimeS: 1746.741,
    walkTimeS: 46.836,
    idleTimeS: 6.902,
    segments: [
      { state: 'run', startElapsedS: 0, endElapsedS: 1746.741 },
      { state: 'walk', startElapsedS: 1746.741, endElapsedS: 1793.577 },
      { state: 'idle', startElapsedS: 1793.577, endElapsedS: 1800.479 },
    ],
  }
  value.runWalk = runWalk
  assert.equal(isActivityDetail(value), true)
  assert.equal(
    isActivityDetail({
      ...value,
      staminaTrace: {
        source: 'garden-estimate',
        method: 'garden-stamina-v2',
        ftpWatts: 287,
        maxHeartRateBpm: 196,
      },
    }),
    false,
  )
  assert.equal(
    isActivityDetail({
      ...value,
      runWalk: {
        ...runWalk,
        segments: [
          { state: 'run', startElapsedS: 1, endElapsedS: 1747.741 },
          { state: 'walk', startElapsedS: 1747.741, endElapsedS: 1794.577 },
          { state: 'idle', startElapsedS: 1794.577, endElapsedS: 1801.479 },
        ],
      },
    }),
    false,
  )
  assert.equal(isActivityDetail({ ...value, sport: 'walk' }), false)

  const calculated = {
    ...value,
    sport: 'bike',
    runWalk: null,
    performanceConditionTrace: {
      source: 'garden-estimate',
      method: 'garden-cycling-performance-condition-v1',
      ftpWatts: 287,
      lactateThresholdHeartRateBpm: 173,
      restingHeartRateBpm: 50,
      windowSeconds: 360,
    },
  }
  assert.equal(isActivityDetail(calculated), true)
  assert.equal(isActivityDetail({ ...calculated, sport: 'run' }), false)
  assert.equal(
    isActivityDetail({
      ...calculated,
      performanceConditionTrace: { ...calculated.performanceConditionTrace, windowSeconds: 300 },
    }),
    false,
  )
})

test('rejects private fields, out-of-order samples, and ambiguous temperature contracts', () => {
  const privateValue = detail(10, '2026-08-10', 'run')
  privateValue.analyses = {
    native: { myWindsock: null, pelotan: null },
    derived: {
      environment: {
        ...gardenEnvironment([environmentSample(0, 0), environmentSample(3_600, 10)]),
        routeFingerprint: 'private',
      },
      uvScore: null,
      apparentWind: null,
    },
  }
  assert.equal(isActivityDetail(privateValue), false)

  const nonMonotonic = detail(11, '2026-08-11', 'run')
  nonMonotonic.analyses = {
    native: { myWindsock: null, pelotan: null },
    derived: {
      environment: gardenEnvironment([environmentSample(2_000, 8), environmentSample(1_000, 9)]),
      uvScore: null,
      apparentWind: null,
    },
  }
  assert.equal(isActivityDetail(nonMonotonic), false)

  const nonMonotonicMovingDose = detail(13, '2026-08-13', 'bike')
  const first = environmentSample(0, 0)
  const second = environmentSample(3_600, 10)
  first.cumulativeMovingTelemetrySed = 1
  second.cumulativeMovingTelemetrySed = 0
  nonMonotonicMovingDose.analyses = {
    native: { myWindsock: null, pelotan: null },
    derived: { environment: gardenEnvironment([first, second]), uvScore: null, apparentWind: null },
  }
  assert.equal(isActivityDetail(nonMonotonicMovingDose), false)

  const ambiguous = detail(12, '2026-08-12', 'bike')
  delete ambiguous.deviceTemperatureC
  assert.equal(isActivityDetail(ambiguous), false)
})

test('validates serialized cycling intensity provenance, gaps and sample ordering', () => {
  const trace = {
    source: 'wahoo',
    method: 'cumulative-power-30s-v1',
    ftpWatts: 250,
    ftpSource: 'wahoo-summary',
    points: [
      { elapsedS: 0, distanceKm: 0, intensityFactor: null },
      { elapsedS: 30, distanceKm: 0.3, intensityFactor: 0 },
      { elapsedS: 60, distanceKm: 0.6, intensityFactor: 0.8 },
    ],
  }
  const value = {
    ...detail(101, '2026-09-08', 'bike'),
    wahoo: {
      activityId: 'wahoo:1',
      fitPath: null,
      sha256: 'a'.repeat(64),
      sourceDevice: null,
      startOffsetS: 0,
      distanceM: 600,
      metrics: emptyWahooMetrics(),
      summarySources: {},
      streamFallback: null,
    },
    cyclingIntensityTrace: trace,
  }
  assert.equal(isActivityDetail(JSON.parse(JSON.stringify(value))), true)
  assert.equal(isActivityDetail({ ...value, sport: 'run' }), false)
  assert.equal(isActivityDetail({ ...value, wahoo: undefined }), false)
  for (const invalid of [
    { ...trace, source: 'garmin' },
    { ...trace, ftpSource: null },
    { ...trace, ftpWatts: null, ftpSource: null },
    { ...trace, points: trace.points.toReversed() },
    { ...trace, points: [...trace.points, { ...trace.points[2], elapsedS: 3601 }] },
    {
      ...trace,
      points: [...trace.points.slice(0, 2), { ...trace.points[2], intensityFactor: -1 }],
    },
    {
      ...trace,
      points: [...trace.points.slice(0, 2), { ...trace.points[2], intensityFactor: Infinity }],
    },
  ])
    assert.equal(isActivityDetail({ ...value, cyclingIntensityTrace: invalid }), false)
})

test('validates HR session estimates for walks and stationary recovery activities', () => {
  const physiology = estimateHeartRatePhysiology(
    Array.from({ length: 31 }, (_, index) => ({
      elapsedS: index * 10,
      distanceKm: 0,
      heartRate: 100,
    })),
    200,
    'walk',
  )
  assert.ok(physiology)
  for (const sport of ['walk', 'yoga', 'treatment', 'sauna', 'strength']) {
    const value: Record<string, unknown> = {
      ...detail(1, '2026-09-12', sport),
      heartRatePhysiology: physiology,
    }
    assert.equal(isActivityDetail(JSON.parse(JSON.stringify(value))), true)
    assert.equal(
      isActivityDetail({ ...value, heartRatePhysiology: { ...physiology, source: 'garmin' } }),
      false,
    )
    assert.equal(
      isActivityDetail({
        ...value,
        heartRatePhysiology: {
          ...physiology,
          points: [{ ...physiology.points[0], stamina: 101 }, ...physiology.points.slice(1)],
        },
      }),
      false,
    )
    assert.equal(
      isActivityDetail({
        ...value,
        heartRatePhysiology: { ...physiology, points: [...physiology.points].reverse() },
      }),
      false,
    )
  }
})

test('validates serialized Wahoo torque, bounds, ordering and provider identity', () => {
  const samples = cyclingTorqueSamples(
    { time: [0, 1, 2], watts: [250, 0, 250], cadence: [90, 90, 0] },
    0,
    3,
  )
  const cyclingTorque = buildCyclingTorqueTrace(samples, 3, [])
  assert.ok(cyclingTorque)
  const activity = {
    ...detail(101, '2026-09-19', 'bike'),
    elapsedTimeS: 3,
    wahoo: {
      activityId: 'wahoo:1',
      fitPath: null,
      sha256: 'a'.repeat(64),
      sourceDevice: null,
      startOffsetS: 0,
      distanceM: 0,
      metrics: emptyWahooMetrics(),
      summarySources: {},
      streamFallback: null,
    },
    cyclingTorque,
  }
  assert.equal(isActivityDetail(JSON.parse(JSON.stringify(activity))), true)
  assert.equal(isActivityDetail({ ...activity, wahoo: undefined }), false)
  assert.equal(isActivityDetail({ ...activity, sport: 'run' }), false)
  for (const invalid of [
    { ...cyclingTorque, source: 'garmin' },
    { ...cyclingTorque, points: cyclingTorque.points.toReversed() },
    { ...cyclingTorque, summary: { ...cyclingTorque.summary, coverage: 2 } },
    { ...cyclingTorque, cells: [{ cadenceRpm: 90, torqueNm: 30, seconds: -1 }] },
  ])
    assert.equal(isActivityDetail({ ...activity, cyclingTorque: invalid }), false)
})

test('validates recorded cycling power provenance, sample bounds and nullable terrain', () => {
  const trace = {
    source: 'strava',
    terrainSource: 'strava',
    method: 'recorded-power-average-v1',
    points: [
      {
        elapsedS: 0,
        distanceKm: 0,
        elevationM: -10,
        power30sWatts: null,
        power5mWatts: null,
        cumulativePowerWatts: null,
      },
      {
        elapsedS: 30,
        distanceKm: 0.3,
        elevationM: null,
        power30sWatts: 0,
        power5mWatts: null,
        cumulativePowerWatts: 0,
      },
      {
        elapsedS: 60,
        distanceKm: 0.6,
        elevationM: 20,
        power30sWatts: 200,
        power5mWatts: null,
        cumulativePowerWatts: 100,
      },
    ],
  }
  const activity = {
    ...detail(101, '2026-09-19', 'bike'),
    deviceWatts: true,
    cyclingPowerTrace: trace,
  }
  assert.equal(isActivityDetail(JSON.parse(JSON.stringify(activity))), true)
  assert.equal(isActivityDetail({ ...activity, sport: 'run' }), false)
  assert.equal(isActivityDetail({ ...activity, deviceWatts: false }), false)
  const wahooTrace = { ...trace, source: 'wahoo' }
  assert.equal(isActivityDetail({ ...activity, cyclingPowerTrace: wahooTrace }), false)
  assert.equal(
    isActivityDetail({
      ...activity,
      cyclingPowerTrace: wahooTrace,
      wahoo: {
        activityId: 'wahoo:1',
        fitPath: null,
        sha256: 'a'.repeat(64),
        sourceDevice: null,
        startOffsetS: 0,
        distanceM: 600,
        metrics: emptyWahooMetrics(),
        summarySources: {},
        streamFallback: null,
      },
    }),
    true,
  )
  for (const invalid of [
    { ...trace, source: 'garden-estimate' },
    { ...trace, terrainSource: 'garden-estimate' },
    { ...trace, method: 'wind-adjusted' },
    { ...trace, points: trace.points.toReversed() },
    { ...trace, points: [] },
    ...[
      { elapsedS: 3_601 },
      { elapsedS: -1 },
      { distanceKm: -1 },
      { elevationM: Infinity },
      { power30sWatts: -1 },
      { power5mWatts: NaN },
      { cumulativePowerWatts: undefined },
    ].map(overrides => ({
      ...trace,
      points: [trace.points[0], { ...trace.points[1], ...overrides }],
    })),
  ])
    assert.equal(isActivityDetail({ ...activity, cyclingPowerTrace: invalid }), false)
})
