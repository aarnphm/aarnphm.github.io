import assert from 'node:assert/strict'
import test from 'node:test'
import { runBestEfforts } from './run-best-efforts'

test('finds the fastest distance anywhere in a run, including a fractional start', () => {
  const efforts = runBestEfforts({
    time: [0, 200, 250, 400],
    distance: [0, 200, 500, 600],
    altitude: [10, 20, 30, 40],
    heartrate: [120, 140, 160, 150],
  })
  assert.equal(efforts.length, 1)
  assert.deepEqual(efforts[0], {
    label: '400m',
    targetDistanceM: 400,
    elapsedTimeS: 150,
    averageSpeedKph: 9.6,
    averageHeartRate: 140,
    elevationDeltaM: 15,
  })
})

test('interpolates a fractional finish and weights heart rate by elapsed time', () => {
  const [effort] = runBestEfforts({
    time: [0, 40, 400],
    distance: [0, 300, 600],
    altitude: [0, 3, 9],
    heartrate: [100, 100, 160],
  })
  assert.equal(effort.elapsedTimeS, 160)
  assert.equal(effort.averageHeartRate, 108)
  assert.equal(effort.elevationDeltaM, 5)
})

test('counts pauses within efforts and starts after a stationary plateau when possible', () => {
  const [effort] = runBestEfforts({
    time: [0, 40, 100, 180],
    distance: [0, 200, 200, 600],
    altitude: [],
  })
  assert.equal(effort.elapsedTimeS, 80)
  const [paused] = runBestEfforts({
    time: [0, 40, 100, 140],
    distance: [0, 200, 200, 400],
    altitude: [],
  })
  assert.equal(paused.elapsedTimeS, 140)
})

test('keeps unavailable telemetry separate from measured zero elevation change', () => {
  const stream = { time: [10, 110], distance: [50, 450], altitude: [] }
  const [missing] = runBestEfforts({ ...stream, heartrate: [0, 0] })
  assert.equal(missing.averageHeartRate, null)
  assert.equal(missing.elevationDeltaM, null)
  const [flat] = runBestEfforts({ ...stream, altitude: [0, 0] })
  assert.equal(flat.elevationDeltaM, 0)
  assert.equal(runBestEfforts({ ...stream, distance: [50, 449.9] }).length, 0)
})

test('uses every supported running distance only when the recorded stream covers it', () => {
  const stream = { time: [0, 18_000], distance: [0, 50_000], altitude: [] }
  assert.deepEqual(
    runBestEfforts(stream).map(effort => effort.label),
    [
      '400m',
      '1/2 mile',
      '1K',
      '1 mile',
      '2 mile',
      '5K',
      '10K',
      '15K',
      '10 mile',
      '20K',
      'Half marathon',
      '30K',
      'Marathon',
      '50K',
    ],
  )
  assert.equal(runBestEfforts({ ...stream, distance: [0, 4_999.9] }).at(-1)?.label, '2 mile')
})

test('coalesces repeated whole-second timestamps before calculating efforts', () => {
  const [effort] = runBestEfforts({
    time: [0, 0, 100, 100, 200],
    distance: [0, 1, 200, 201, 401],
    altitude: [0, 1, 2, 3, 5],
    heartrate: [100, 120, 130, 140, 160],
  })
  assert.equal(effort.elapsedTimeS, 200)
  assert.equal(effort.averageHeartRate, 140)
  assert.equal(effort.elevationDeltaM, 4)
})

test('requires aligned, finite, monotonic elapsed time and distance without inventing a start', () => {
  const stream = { time: [0, 100], distance: [0, 400], altitude: [] }
  for (const invalid of [
    undefined,
    { ...stream, time: undefined },
    { ...stream, time: [0] },
    { ...stream, time: [0, 0] },
    { ...stream, time: [100, 0] },
    { ...stream, time: [0, NaN] },
    { ...stream, distance: [400, 0] },
    { ...stream, distance: [0, Infinity] },
    { ...stream, distance: [50, 400] },
  ])
    assert.deepEqual(runBestEfforts(invalid), [])
})
