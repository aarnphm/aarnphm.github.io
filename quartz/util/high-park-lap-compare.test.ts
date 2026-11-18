import assert from 'node:assert/strict'
import test from 'node:test'
import {
  binGrades,
  gradeAdjusted,
  lapBaseline,
  meanSpeed,
  minettiCost,
  PICK_SLOTS,
  signed,
  smoothBins,
  toggleLap,
} from './high-park-lap-compare'
import { LAP_BINS } from './high-park-lap-metrics'

test('picking keeps order, reuses the lowest free slot and drops on a second pick', () => {
  let picks = toggleLap([], 7)
  picks = toggleLap(picks, 2)
  picks = toggleLap(picks, 9)
  assert.deepEqual(picks, [
    { lap: 7, slot: 0 },
    { lap: 2, slot: 1 },
    { lap: 9, slot: 2 },
  ])
  picks = toggleLap(picks, 7)
  assert.deepEqual(picks, [
    { lap: 2, slot: 1 },
    { lap: 9, slot: 2 },
  ])
  assert.deepEqual(toggleLap(picks, 4).at(-1), { lap: 4, slot: 0 })
})

test('a pick past the last slot replaces the newest and leaves the reference alone', () => {
  let picks = Array.from({ length: PICK_SLOTS }, (_, i) => i).reduce(toggleLap, [])
  picks = toggleLap(picks, 20)
  assert.equal(picks.length, PICK_SLOTS)
  assert.equal(picks[0].lap, 0)
  assert.deepEqual(picks.at(-1), { lap: 20, slot: PICK_SLOTS - 1 })
})

test('smoothing keeps a flat line flat and spreads a spike over its neighbours', () => {
  assert.deepEqual(smoothBins([5, 5, 5, 5]), [5, 5, 5, 5])
  assert.deepEqual(smoothBins([0, 0, 4, 0, 0]), [0, 1, 2, 1, 0])
  assert.deepEqual(smoothBins([3, 0]), [2, 1])
})

test('baseline is the smoothed mean and spread across laps', () => {
  const flat = (value: number) => Array.from({ length: LAP_BINS }, () => value)
  const base = lapBaseline([flat(100), flat(200), []])
  assert.deepEqual(base?.mean, flat(150))
  assert.deepEqual(base?.sd, flat(50))
  assert.equal(lapBaseline([[]]), null)
  assert.equal(lapBaseline([flat(1).map((v, i) => (i === 3 ? null : v))]), null)
  assert.deepEqual(lapBaseline([flat(8)])?.sd, flat(0))
})

test('grades close the loop and sum to zero', () => {
  // 16 bins over 1.6 km: 100 m a bin. A 5 m step up and back down.
  const altitude = [0, 5, 10, 5, ...Array.from({ length: LAP_BINS - 4 }, () => 0)]
  const grades = binGrades(altitude, 1.6)
  assert.equal(grades[1], 0.05)
  assert.equal(grades[2], 0)
  assert.equal(grades[3], -0.05)
  assert.equal(grades[0], 0.025)
  assert.ok(Math.abs(grades.reduce((a, b) => a + b, 0)) < 1e-12)
  assert.deepEqual(
    binGrades(altitude, 0),
    altitude.map(() => 0),
  )
})

test('Minetti cost: 3.6 J/kg/m on the flat, dearer uphill, cheapest near -10 to -20 %', () => {
  assert.equal(minettiCost(0), 3.6)
  assert.ok(Math.abs(minettiCost(0.1) - 5.97) < 0.01)
  assert.ok(minettiCost(-0.1) < minettiCost(0))
  assert.ok(minettiCost(-0.2) < minettiCost(-0.4))
  assert.equal(minettiCost(0.9), minettiCost(0.45))
  assert.equal(gradeAdjusted(10, 0), 10)
  assert.ok(gradeAdjusted(10, 0.05) > 10)
  assert.ok(gradeAdjusted(10, -0.05) < 10)
})

test('mean speed over equal distances is harmonic', () => {
  assert.equal(meanSpeed([10, 30]), 15)
  assert.equal(meanSpeed([null, 12, null]), 12)
  assert.equal(meanSpeed([]), null)
})

test('signed differences name their sign', () => {
  assert.equal(signed(14), '+14')
  assert.equal(signed(-0.72, 1), '−0.7')
  assert.equal(signed(0.04, 1), '±0')
  assert.equal(signed(-0.4), '±0')
})
