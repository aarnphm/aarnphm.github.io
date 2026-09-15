import assert from 'node:assert/strict'
import test from 'node:test'
import { POWER_TO_WEIGHT_DURATIONS } from '../../../../plugins/stores/analytics'
import {
  deconflictPowerToWeightMonthTicks,
  powerToWeightDurationLabel,
  restorePowerToWeightDurations,
} from './power-to-weight'

test('restores a saved duration subset and ignores invalid or unavailable entries', () => {
  assert.deepEqual(
    [
      ...restorePowerToWeightDurations(
        [300, 5, 5, 999, '60', null],
        new Set(POWER_TO_WEIGHT_DURATIONS),
      ),
    ],
    [5, 300],
  )
  assert.deepEqual([...restorePowerToWeightDurations([5, 300], new Set([300]))], [300])
})

test('falls back to available durations when saved selection cannot display a curve', () => {
  for (const stored of [null, {}, '5', [], [999], [5]]) {
    assert.deepEqual([...restorePowerToWeightDurations(stored, new Set([300, 1200]))], [300, 1200])
  }
  assert.deepEqual([...restorePowerToWeightDurations([5], new Set())], [])
})

test('labels the seven exact power-to-weight durations', () => {
  assert.equal(powerToWeightDurationLabel(5), '5s')
  assert.equal(powerToWeightDurationLabel(60), '1m')
  assert.equal(powerToWeightDurationLabel(180), '3m')
  assert.equal(powerToWeightDurationLabel(300), '5m')
  assert.equal(powerToWeightDurationLabel(360), '6m')
  assert.equal(powerToWeightDurationLabel(720), '12m')
  assert.equal(powerToWeightDurationLabel(1200), '20m')
})

test('drops an overlapping partial-month label at the chart origin', () => {
  assert.deepEqual(
    deconflictPowerToWeightMonthTicks([
      { label: 'May', pct: 0, cls: 'tri-cax-xt--first' },
      { label: 'Jun', pct: 1.2 },
      { label: 'Jul', pct: 34.5 },
    ]),
    [
      { label: 'Jun', pct: 1.2, cls: 'tri-cax-xt--first' },
      { label: 'Jul', pct: 34.5 },
    ],
  )
})
