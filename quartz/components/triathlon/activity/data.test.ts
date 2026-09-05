import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import test from 'node:test'
import { STRAVA_DETAIL_INDEX_KIND } from '../../../util/strava-detail'
import { isActivityDetail, readDetailPayload } from './data'

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

test('loads and reconstructs a Strava detail index from valid JSON shards', async () => {
  const requested: string[] = []
  const server = createServer((request, response) => {
    const path = request.url ?? ''
    requested.push(path)
    const values: Record<string, unknown> = {
      '/static/strava-detail.json': {
        kind: STRAVA_DETAIL_INDEX_KIND,
        shards: ['strava-detail/2026-08.json', 'strava-detail/2026-07.json'],
        health: {},
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

const gardenEnvironment = (samples: Record<string, unknown>[]): Record<string, unknown> => ({
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
        method: 'garden-stamina-v1',
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
