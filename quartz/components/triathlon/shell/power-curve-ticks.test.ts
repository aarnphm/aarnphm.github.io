import assert from 'node:assert/strict'
import test from 'node:test'
import { powerCurveFraction } from '../../../util/triathlon-card'
import { powerCurveTickVisibility } from './power-curve-ticks'

test('fits extra minute markers between existing labels as the power curve widens', () => {
  const durations = [120, 180, 300, 360, 600, 720, 1_200]
  const visibleAtWidth = (width: number): number[] => {
    const ticks = durations.map(seconds => {
      const center = powerCurveFraction(seconds, 1, 20_107) * width
      const labelWidth = seconds >= 600 ? 14 : 10
      return {
        left: center - labelWidth / 2,
        right: center + labelWidth / 2,
        optional: [180, 360, 720].includes(seconds),
      }
    })
    const visible = powerCurveTickVisibility(ticks)
    return durations.filter((_, index) => visible[index])
  }
  assert.deepEqual(visibleAtWidth(220), [120, 300, 600, 1_200])
  assert.deepEqual(visibleAtWidth(400), [120, 180, 300, 600, 1_200])
  assert.deepEqual(visibleAtWidth(700), [120, 180, 300, 360, 600, 1_200])
  assert.deepEqual(visibleAtWidth(1_000), durations)
})

test('keeps endpoints and ignores labels already hidden by responsive styles', () => {
  assert.deepEqual(
    powerCurveTickVisibility([
      { left: 0, right: 0, optional: false },
      { left: 0, right: 14, optional: true },
      { left: 30, right: 44, optional: false },
    ]),
    [true, true, true],
  )
  assert.deepEqual(
    powerCurveTickVisibility([
      { left: 0, right: 14, optional: true },
      { left: 12, right: 40, optional: false },
    ]),
    [false, true],
  )
})
