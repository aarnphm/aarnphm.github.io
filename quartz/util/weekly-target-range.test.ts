import assert from 'node:assert/strict'
import test from 'node:test'
import {
  weeklyChartIndex,
  weeklyChartX,
  weeklyTargetRanges,
  type WeeklyTargetObservation,
} from './weekly-target-range'

const row = (value: number, complete = true, observed = true): WeeklyTargetObservation => ({
  value,
  complete,
  observed,
})

test('weekly target uses only the three preceding complete observations', () => {
  const observations = [row(100), row(200), row(300), row(400), row(10_000, false)]
  const ranges = weeklyTargetRanges(observations)

  assert.deepEqual(ranges.slice(0, 3), [
    [0, 0],
    [30, 60],
    [75, 150],
  ])
  assert.deepEqual(ranges[3], [120, 240])
  assert.deepEqual(ranges[4], [180, 360])
})

test('weekly target excludes partial and unobserved training rows', () => {
  const observations = [
    row(10_000, false),
    row(1, true, false),
    row(300),
    row(600),
    row(900),
    row(1),
  ]

  assert.deepEqual(weeklyTargetRanges(observations).at(-1), [360, 720])
})

test('weekly chart geometry anchors both endpoints and selects the nearest week', () => {
  assert.equal(weeklyChartX(0, 7), 0)
  assert.equal(weeklyChartX(3, 7), 0.5)
  assert.equal(weeklyChartX(6, 7), 1)
  assert.equal(weeklyChartIndex(0, 7), 0)
  assert.equal(weeklyChartIndex(0.5, 7), 3)
  assert.equal(weeklyChartIndex(1, 7), 6)
})
