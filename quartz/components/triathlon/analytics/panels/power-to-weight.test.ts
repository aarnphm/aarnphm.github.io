import assert from 'node:assert/strict'
import test from 'node:test'
import { deconflictPowerToWeightMonthTicks, powerToWeightDurationLabel } from './power-to-weight'

test('labels the four exact power-to-weight durations', () => {
  assert.equal(powerToWeightDurationLabel(5), '5s')
  assert.equal(powerToWeightDurationLabel(60), '1m')
  assert.equal(powerToWeightDurationLabel(300), '5m')
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
