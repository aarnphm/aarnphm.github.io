import assert from 'node:assert/strict'
import test from 'node:test'
import {
  criticalPowerAtDuration,
  criticalPowerCurve,
  fitCriticalPower,
  type CriticalPowerAnchor,
} from './critical-power'

const anchor = (durationS: number, activityId: number, startElapsedS = 0): CriticalPowerAnchor => ({
  durationS,
  meanPowerWatts: 250 + 10_000 / durationS,
  activityId,
  activityDate: `2026-08-0${activityId}`,
  startElapsedS,
  endElapsedS: startElapsedS + durationS,
})

test('fits critical power and W prime in power space', () => {
  const estimate = fitCriticalPower(
    [anchor(180, 1), anchor(180, 4), anchor(420, 2), anchor(720, 3)],
    'six-weeks',
    '2026-07-03',
    '2026-08-13',
  )

  assert.ok(estimate)
  assert.equal(estimate.criticalPowerWatts, 250)
  assert.equal(estimate.wPrimeJoules, 10_000)
  assert.equal(estimate.independentEffortCount, 3)
  assert.equal(estimate.anchors[0].activityId, 4)
  assert.equal(estimate.confidence, 'medium')
  assert.equal(estimate.rmseWatts, 0)
  assert.equal(criticalPowerAtDuration(estimate, 300), 250 + 10_000 / 300)
  const curve = criticalPowerCurve(estimate, 60, 900)
  assert.equal(curve[0].s, 180)
  assert.equal(curve.at(-1)?.s, 720)
})

test('marks overlapping anchors from one effort as provisional', () => {
  const estimate = fitCriticalPower(
    [anchor(180, 1, 1_000), anchor(420, 1, 1_000), anchor(720, 1, 1_000)],
    'activity',
    '2026-08-01',
    '2026-08-01',
  )

  assert.ok(estimate)
  assert.equal(estimate.window, 'activity')
  assert.equal(estimate.independentEffortCount, 1)
  assert.equal(estimate.confidence, 'provisional')
})

test('counts the largest set of non-overlapping efforts within one activity', () => {
  const estimate = fitCriticalPower(
    [anchor(180, 1, 0), anchor(420, 1, 300), anchor(720, 1, 0)],
    'calendar-year',
    '2026-01-01',
    '2026-08-13',
  )

  assert.ok(estimate)
  assert.equal(estimate.independentEffortCount, 2)
  assert.equal(estimate.confidence, 'provisional')
})

test('rejects implausible and poorly fitting anchors', () => {
  const flat = [180, 420, 720].map((durationS, index) => ({
    ...anchor(durationS, index + 1),
    meanPowerWatts: 250,
  }))
  const noisy = [anchor(180, 1), anchor(420, 2), { ...anchor(720, 3), meanPowerWatts: 210 }]

  assert.equal(fitCriticalPower(flat, 'six-weeks', '2026-07-03', '2026-08-13'), null)
  assert.equal(fitCriticalPower(noisy, 'six-weeks', '2026-07-03', '2026-08-13'), null)
  assert.equal(
    fitCriticalPower(
      [{ ...anchor(180, 1), endElapsedS: 179 }, anchor(420, 2), anchor(720, 3)],
      'six-weeks',
      '2026-07-03',
      '2026-08-13',
    ),
    null,
  )
})
