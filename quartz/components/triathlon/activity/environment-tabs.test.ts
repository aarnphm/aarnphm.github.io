import assert from 'node:assert/strict'
import test from 'node:test'
import {
  environmentCursorIndex,
  environmentSampleIndexAtElapsed,
  environmentViewFromKey,
  parseEnvironmentCurrentSamples,
  parseEnvironmentSamples,
  type EnvironmentView,
} from './environment-tabs'

test('environment tabs wrap with arrows and respect Home and End', () => {
  assert.equal(environmentViewFromKey('cloud-cover', 'ArrowRight'), 'wind')
  assert.equal(environmentViewFromKey('cumulative', 'ArrowLeft'), 'wind')
  assert.equal(environmentViewFromKey('wind', 'ArrowRight'), 'cumulative')
  assert.equal(environmentViewFromKey('wind', 'ArrowLeft'), 'cloud-cover')
  assert.equal(environmentViewFromKey('wind', 'Home'), 'cumulative')
  assert.equal(environmentViewFromKey('uv-index', 'End'), 'wind')
  assert.equal(environmentViewFromKey('uv-index', 'Enter'), null)
})

test('environment tabs navigate only among rendered views', () => {
  const views: readonly EnvironmentView[] = ['uv-index', 'temperature']
  assert.equal(environmentViewFromKey('uv-index', 'ArrowLeft', views), 'temperature')
  assert.equal(environmentViewFromKey('temperature', 'ArrowRight', views), 'uv-index')
  assert.equal(environmentViewFromKey('cumulative', 'ArrowRight', views), null)
})

test('open-water current tabs navigate in rendered order while retaining five-view defaults', () => {
  const views: readonly EnvironmentView[] = [
    'cumulative',
    'uv-index',
    'temperature',
    'cloud-cover',
    'wind',
    'current-speed',
    'current-direction',
  ]
  assert.equal(environmentViewFromKey('wind', 'ArrowRight', views), 'current-speed')
  assert.equal(environmentViewFromKey('current-speed', 'ArrowRight', views), 'current-direction')
  assert.equal(environmentViewFromKey('current-direction', 'ArrowRight', views), 'cumulative')
  assert.equal(environmentViewFromKey('cumulative', 'ArrowLeft', views), 'current-direction')
  assert.equal(environmentViewFromKey('wind', 'End', views), 'current-direction')
  assert.equal(environmentViewFromKey('current-speed', 'ArrowRight'), null)
})

test('environment cursor maps elapsed time across weather and current sample grids', () => {
  const weather = [0, 30, 60, 90, 120].map(elapsedS => ({ elapsedS }))
  const currents = [0, 40, 100, 120].map(elapsedS => ({ elapsedS }))
  assert.equal(environmentSampleIndexAtElapsed(currents, weather[3].elapsedS), 2)
  assert.equal(environmentSampleIndexAtElapsed(weather, currents[2].elapsedS), 3)
  assert.equal(environmentSampleIndexAtElapsed(currents, 0), 0)
  assert.equal(environmentSampleIndexAtElapsed(currents, 120), 3)
})

test('parses current chart series with calm values and unavailable gaps independently of weather', () => {
  const samples = [
    { elapsedS: 0, surfaceCurrentSpeedMps: 0, surfaceCurrentDirectionDeg: null },
    { elapsedS: 40, surfaceCurrentSpeedMps: null, surfaceCurrentDirectionDeg: null },
    { elapsedS: 120, surfaceCurrentSpeedMps: 0.14, surfaceCurrentDirectionDeg: 237 },
  ]
  assert.deepEqual(parseEnvironmentCurrentSamples(JSON.stringify(samples)), samples)
  assert.deepEqual(parseEnvironmentSamples(JSON.stringify(samples)), [])
  assert.deepEqual(parseEnvironmentCurrentSamples(undefined), [])
  assert.deepEqual(parseEnvironmentCurrentSamples('[]'), [])
  const maximum = Array.from({ length: 512 }, (_, elapsedS) => ({
    elapsedS,
    surfaceCurrentSpeedMps: 10,
    surfaceCurrentDirectionDeg: 359.99,
  }))
  assert.equal(parseEnvironmentCurrentSamples(JSON.stringify(maximum)).length, 512)
  assert.deepEqual(parseEnvironmentCurrentSamples(JSON.stringify(maximum.slice(0, 1))), [])
  assert.deepEqual(
    parseEnvironmentCurrentSamples(JSON.stringify([...maximum, { ...maximum[0], elapsedS: 512 }])),
    [],
  )
})

test('rejects malformed or unbounded current chart payloads and private/weather fields', () => {
  const first = { elapsedS: 0, surfaceCurrentSpeedMps: 0.14, surfaceCurrentDirectionDeg: 237 }
  const final = { ...first, elapsedS: 120 }
  const invalid = [
    { ...final, elapsedS: -1 },
    { ...final, elapsedS: '120' },
    { ...final, surfaceCurrentSpeedMps: -0.1 },
    { ...final, surfaceCurrentSpeedMps: 10.1 },
    { ...final, surfaceCurrentSpeedMps: '0.14' },
    { ...final, surfaceCurrentDirectionDeg: -1 },
    { ...final, surfaceCurrentDirectionDeg: 360 },
    { ...final, surfaceCurrentDirectionDeg: false },
    { elapsedS: 120, surfaceCurrentSpeedMps: 0.14 },
    { ...final, latitude: 43.64 },
    { ...final, routeFingerprint: 'private' },
    { ...final, distanceKm: 0 },
    { ...final, windSpeedKph: 14 },
  ]
  for (const sample of invalid)
    assert.deepEqual(parseEnvironmentCurrentSamples(JSON.stringify([first, sample])), [])
  assert.deepEqual(parseEnvironmentCurrentSamples(JSON.stringify([first, first])), [])
  assert.deepEqual(parseEnvironmentCurrentSamples(JSON.stringify([final, first])), [])
  assert.deepEqual(
    parseEnvironmentCurrentSamples(
      '[{"elapsedS":0,"surfaceCurrentSpeedMps":0,"surfaceCurrentDirectionDeg":null},{"elapsedS":1e309,"surfaceCurrentSpeedMps":0,"surfaceCurrentDirectionDeg":null}]',
    ),
    [],
  )
})

test('environment cursor preserves zero and clamps serialized indices', () => {
  assert.equal(environmentCursorIndex('0', 320), 0)
  assert.equal(environmentCursorIndex('1', 320), 1)
  assert.equal(environmentCursorIndex('999', 320), 319)
  assert.equal(environmentCursorIndex('-1', 320), 0)
  assert.equal(environmentCursorIndex(undefined, 320), 319)
  assert.equal(environmentCursorIndex('invalid', 320), 319)
})

test('parses legacy environment samples and retains nullable ambient wind values', () => {
  const samples = [0, 20].map(elapsedS => ({
    elapsedS,
    distanceKm: elapsedS / 1_000,
    uvIndex: 0,
    cumulativeSed: 0,
    cumulativeMovingTelemetrySed: null,
    ambientTemperatureC: 20,
    cloudCoverPct: 50,
    headwindKph: null,
    crosswindKph: null,
    apparentAirSpeedKph: null,
    yawDeg: null,
  }))
  assert.deepEqual(parseEnvironmentSamples(JSON.stringify(samples)), samples)
  for (const windSpeedKph of [null, 0, 18]) {
    const withWind = samples.map(sample => ({ ...sample, windSpeedKph }))
    assert.deepEqual(parseEnvironmentSamples(JSON.stringify(withWind)), withWind)
  }
  for (const windSpeedKph of [-1, 1_001, '18', true, {}])
    assert.deepEqual(
      parseEnvironmentSamples(JSON.stringify(samples.map(sample => ({ ...sample, windSpeedKph })))),
      [],
    )
})
