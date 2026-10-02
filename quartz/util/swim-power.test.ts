import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildSwimPowerEstimate,
  buildSwimPowerCurveBlock,
  openWaterPowerIntervals,
} from './swim-power'

const length = (
  startElapsedS: number,
  durationS: number,
  distanceM = 25,
  stroke = 'freestyle',
) => ({ startElapsedS, endElapsedS: startElapsedS + durationS, durationS, distanceM, stroke })

test('open-water drag uses observed route distance and breaks gaps or impossible GPS jumps', () => {
  const route = Array.from({ length: 13 }, (_, i) => ({ elapsedS: i * 10, d: i / 150 }))
  const result = buildSwimPowerEstimate('route', openWaterPowerIntervals(route), 'ground-speed')
  assert.ok(result)
  assert.equal(result.method, 'open-water-drag-index-v1')
  assert.equal(result.speedBasis, 'ground-speed')
  assert.ok(Math.abs(result.averageIndex - 100) < 0.001)
  assert.equal(result.curve.at(-1)?.durationS, 120)
  const gapped = openWaterPowerIntervals([
    { elapsedS: 0, d: 0 },
    { elapsedS: 30, d: 0.02 },
    { elapsedS: 300, d: 0.04 },
    { elapsedS: 310, d: 1 },
  ])
  assert.deepEqual(buildSwimPowerEstimate('route', gapped, 'ground-speed')?.curve, [])
})

test('weights the cubic demand of each length by time, retaining model provenance', () => {
  const estimate = buildSwimPowerEstimate('garmin', [length(0, 30), length(30, 60)])
  assert.ok(estimate)
  assert.ok(Math.abs(estimate.averageIndex - 81.380208) < 0.001)
  assert.equal(estimate.source, 'garden-estimate')
  assert.equal(estimate.inputSource, 'garmin')
  assert.equal(estimate.distanceM, 50)
  assert.equal(estimate.activeTimeS, 90)
  assert.equal(estimate.curve.find(point => point.durationS === 60)?.index, 109.863)
})

test('bins valid swim seconds by drag demand, retaining fractions and excluding gaps and drills', () => {
  const estimate = buildSwimPowerEstimate('garmin', [
    length(0, 60),
    length(60, 37.5),
    length(120, 30),
    length(150, 30, 25, 'breaststroke'),
    length(180, 0),
  ])
  assert.ok(estimate)
  assert.deepEqual(estimate.histogramS, [60, 0, 0, 0, 37.5, 0, 0, 30])
  assert.equal(
    estimate.histogramS.reduce((sum, seconds) => sum + seconds, 0),
    127.5,
  )
  assert.equal(estimate.activeTimeS, 127.5)
  const slow = buildSwimPowerEstimate('apple', [length(0, 80)])
  assert.deepEqual(slow?.histogramS, [80])
})

test('outdoor distribution uses the same smoothed intervals as the drag curve', () => {
  const estimate = buildSwimPowerEstimate(
    'route',
    openWaterPowerIntervals([
      { elapsedS: 0, d: 0 },
      { elapsedS: 60, d: 0.036 },
      { elapsedS: 120, d: 0.084 },
    ]),
    'ground-speed',
  )
  assert.ok(estimate)
  assert.deepEqual(estimate.histogramS, [0, 0, 60, 0, 0, 0, 60])
  assert.equal(estimate.activeTimeS, 120)
})

test('rest, missing stroke, drills, impossible lengths, and overlapping records break efforts', () => {
  for (const intervals of [
    [length(0, 40), length(60, 40)],
    [length(0, 40), length(40, 40, 25, 'unknown'), length(80, 40)],
    [length(0, 40), length(40, 40, 25, 'kickboard'), length(80, 40)],
    [length(0, 40), length(40, 0.538), length(40.538, 40)],
    [length(0, 40), length(10, 40)],
  ]) {
    const estimate = buildSwimPowerEstimate('garmin', intervals)
    assert.ok(estimate)
    assert.deepEqual(estimate.curve, [])
  }
  assert.equal(buildSwimPowerEstimate('apple', [length(0, 40, 25, 'breaststroke')]), null)
})

test('accepts FIT timestamp rounding and integrates fractional window boundaries', () => {
  const estimate = buildSwimPowerEstimate('garmin', [length(0, 30.25), length(30, 30.25)])
  assert.ok(estimate)
  assert.equal(estimate.curve[0]?.durationS, 60)
  assert.ok(Math.abs((estimate.curve[0]?.index ?? 0) - 190.51) < 0.01)
  assert.equal(estimate.curve[0]?.startElapsedS, 0)
})

test('curve history keeps activity links and excludes out-of-window or missing data', () => {
  const estimate = buildSwimPowerEstimate('apple', [length(0, 75, 50), length(75, 75, 50)])
  assert.ok(estimate)
  const block = buildSwimPowerCurveBlock(
    [
      { id: 1, date: '2026-09-29', swimPower: estimate },
      { id: 2, date: '2026-01-02', swimPower: estimate },
      { id: 3, date: '2025-12-31', swimPower: estimate },
      { id: 4, date: '2026-09-30', swimPower: null },
      { id: 5, date: '2026-10-03', swimPower: estimate },
    ],
    '2026-10-02',
  )
  assert.equal(block.sixWeeks[0]?.activityId, 1)
  assert.equal(block.sixWeeks[0]?.inputSource, 'apple')
  assert.equal(block.year[0]?.activityId, 1)
  assert.equal(block.activityCount, 2)
  assert.equal(block.referencePaceSPer100m, 150)
})
