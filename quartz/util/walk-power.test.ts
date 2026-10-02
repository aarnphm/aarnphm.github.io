import assert from 'node:assert/strict'
import test from 'node:test'
import { buildWalkPowerEstimate, walkPowerAt, type WalkPowerInput } from './walk-power'

const input = (grade = 0, massKg = 80): WalkPowerInput => ({
  inputSource: 'strava',
  weight: { kg: massKg, date: '2026-09-29', source: 'garmin' },
  streams: {
    time: [0, 10, 20, 30, 40, 50, 60],
    distance: [0, 10, 20, 30, 40, 50, 60],
    altitude: [0, 10, 20, 30, 40, 50, 60].map(distance => 100 + distance * grade),
  },
  elapsedTimeS: 60,
})

test('walking metabolic power scales with recorded speed, grade, and measured mass', () => {
  const level = buildWalkPowerEstimate(input())
  const heavy = buildWalkPowerEstimate(input(0, 100))
  const uphill = buildWalkPowerEstimate(input(0.1))
  const downhill = buildWalkPowerEstimate(input(-0.1))
  assert.ok(level && heavy && uphill && downhill)
  assert.equal(level.averageWatts, 200)
  assert.equal(heavy.averageWatts / level.averageWatts, 1.25)
  assert.ok(uphill.averageWatts > level.averageWatts)
  assert.ok(downhill.averageWatts > 0 && downhill.averageWatts < level.averageWatts)
  assert.deepEqual(level.weight, input().weight)
  assert.equal(level.source, 'garden-estimate')
  assert.equal(level.kind, 'net-metabolic')
  assert.equal(walkPowerAt(level, 35), 200)
  assert.equal(walkPowerAt(level, 61), null)
})

test('walking power retains recorded stops and missing spans without inventing effort', () => {
  const paused = input()
  paused.streams.distance = [0, 10, 20, 20, 20, 30, 40]
  const estimate = buildWalkPowerEstimate(paused)
  assert.ok(estimate)
  assert.equal(estimate.points[3].watts, 0)
  assert.equal(estimate.points[4].watts, 0)
  const gap = input()
  gap.streams.time = [0, 10, 20, 70, 80, 90, 100]
  gap.elapsedTimeS = 100
  const missing = buildWalkPowerEstimate(gap)
  assert.ok(missing)
  assert.equal(missing.points[3].watts, null)
  assert.equal(walkPowerAt(missing, 40), null)
})

test('walking power rejects invalid inputs and leaves implausible GPS speeds and grades empty', () => {
  for (const massKg of [0, -1, NaN, Infinity])
    assert.equal(buildWalkPowerEstimate(input(0, massKg)), null)
  for (const grade of [-0.5, 0.5]) assert.equal(buildWalkPowerEstimate(input(grade)), null)
  const fast = input()
  fast.streams.distance = fast.streams.distance.map(distance => distance * 10)
  assert.equal(buildWalkPowerEstimate(fast), null)
  const reversed = input()
  reversed.streams.time = [...reversed.streams.time].reverse()
  assert.equal(buildWalkPowerEstimate(reversed), null)
  const missing = input()
  missing.streams.altitude = []
  assert.equal(buildWalkPowerEstimate(missing), null)
})
