import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildCyclingTorqueTrace,
  crankTorqueNm,
  cyclingMechanics,
  cyclingTorqueSamples,
  cyclingTorqueDensity,
} from './cycling-torque'

test('calculates mean crank torque and keeps zero power distinct from zero cadence', () => {
  assert.ok(Math.abs((crankTorqueNm(250, 60) ?? 0) - 39.78873577) < 1e-7)
  assert.equal(crankTorqueNm(0, 90), 0)
  for (const [watts, cadence] of [
    [250, 0],
    [null, 90],
    [250, null],
    [-1, 90],
    [Infinity, 90],
    [250, NaN],
  ])
    assert.equal(crankTorqueNm(watts, cadence), null)
})

test('aligns and clips timestamps without holding power across recording gaps', () => {
  const samples = cyclingTorqueSamples(
    { time: [0, 1, 60], watts: [200, 200, 200], cadence: [80, 80, 80] },
    10,
    80,
  )
  assert.deepEqual(
    samples.map(sample => [sample.elapsedS, sample.durationS]),
    [
      [10, 1],
      [11, 1],
      [70, 1],
    ],
  )
  assert.equal(cyclingMechanics(samples, 10, 71)?.observedSeconds, 3)
  assert.equal(cyclingMechanics(samples, 12, 70), null)
  const clipped = cyclingTorqueSamples(
    { time: [0, 1, 2], watts: [200, 200, 200], cadence: [80, 80, 80] },
    -0.5,
    1.8,
  )
  assert.deepEqual(
    clipped.map(sample => sample.durationS),
    [0.5, 1, 0.30000000000000004],
  )
})

test('averages paired sample torque rather than dividing average power by average cadence', () => {
  const samples = cyclingTorqueSamples(
    { time: [0, 1], watts: [100, 300], cadence: [50, 100] },
    0,
    2,
  )
  const summary = cyclingMechanics(samples, 0, 2)
  assert.ok(summary)
  assert.equal(summary.averageCadenceRpm, 75)
  assert.equal(summary.averageTorqueNm, 23.87)
  assert.notEqual(summary.averageTorqueNm, Math.round((crankTorqueNm(200, 75) ?? 0) * 100) / 100)
  assert.equal(summary.coverage, 1)
})

test('uses only paired observations inside the exact winning power window', () => {
  const samples = cyclingTorqueSamples(
    { time: [0, 1, 2, 3], watts: [1000, 200, null, 800], cadence: [40, 80, 80, 120] },
    0,
    4,
  )
  const summary = cyclingMechanics(samples, 1, 3)
  assert.equal(summary?.averageCadenceRpm, 80)
  assert.equal(summary?.averageTorqueNm, 23.87)
  assert.equal(summary?.coverage, 0.5)
})

test('bounds display points, breaks missing windows, and weights density by recorded time', () => {
  const time = Array.from({ length: 3600 }, (_, i) => i)
  const samples = cyclingTorqueSamples(
    {
      time,
      watts: time.map(i => (i >= 900 && i < 1200 ? null : 200)),
      cadence: time.map(() => 80),
    },
    0,
    3600,
  )
  const trace = buildCyclingTorqueTrace(samples, 3600, [
    { elapsedS: 0, d: 0 },
    { elapsedS: 3600, d: 40 },
  ])
  assert.ok(trace)
  assert.ok(trace.points.length <= 901)
  assert.ok(
    trace.points
      .filter(point => point.elapsedS >= 900 && point.elapsedS < 1200)
      .every(point => point.torqueNm === null),
  )
  assert.equal(
    trace.cells.reduce((sum, cell) => sum + cell.seconds, 0),
    3300,
  )
  assert.equal(trace.summary.observedSeconds, 3300)
})

test('rejects mismatched or unordered source channels', () => {
  assert.deepEqual(
    cyclingTorqueSamples({ time: [1, 0], watts: [200, 200], cadence: [80, 80] }, 0, 2),
    [],
  )
  assert.deepEqual(
    cyclingTorqueSamples({ time: [0, 1], watts: [200], cadence: [80, 80] }, 0, 2),
    [],
  )
  assert.equal(buildCyclingTorqueTrace([], 30, []), null)
})

test('splits fractional source intervals between display windows', () => {
  const samples = cyclingTorqueSamples(
    { time: [0, 1, 2], watts: [200, 200, 200], cadence: [80, 80, 80] },
    -0.5,
    2.5,
  )
  const trace = buildCyclingTorqueTrace(samples, 2.5, [])
  assert.deepEqual(
    trace?.points.map(point => point.torqueNm),
    [23.87, 23.87, 23.87],
  )
})

test('groups the high-torque tail without losing recorded seconds', () => {
  const density = cyclingTorqueDensity([
    { cadenceRpm: 80, torqueNm: 30, seconds: 1000 },
    { cadenceRpm: 5, torqueNm: 600, seconds: 1 },
    { cadenceRpm: 5, torqueNm: 650, seconds: 1 },
  ])
  assert.equal(density.limitNm, 60)
  assert.deepEqual(density.cells[0], { cadenceRpm: 5, torqueNm: 60, seconds: 2 })
  assert.equal(
    density.cells.reduce((sum, cell) => sum + cell.seconds, 0),
    1002,
  )
})
