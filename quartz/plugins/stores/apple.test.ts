import assert from 'node:assert/strict'
import test from 'node:test'
import { latestAppleDate, mergeAppleDay, type AppleDaily } from './apple'

function day(date: string, values: Partial<AppleDaily> = {}): AppleDaily {
  return { date, burnKcal: null, activeKcal: null, intakeKcal: null, weightKg: null, ...values }
}

test('latestAppleDate returns the newest valid Apple day', () => {
  assert.equal(
    latestAppleDate({
      old: day('2026-06-07'),
      bad: day('today'),
      current: day('2026-06-09'),
      middle: day('2026-06-08'),
    }),
    '2026-06-09',
  )
})

test('mergeAppleDay keeps prior values when a fresh import omits a metric', () => {
  assert.deepEqual(
    mergeAppleDay(
      day('2026-06-08', { burnKcal: 2500, activeKcal: 600, intakeKcal: 2100, weightKg: 90.5 }),
      day('2026-06-08', { burnKcal: 2600, activeKcal: null, intakeKcal: null, weightKg: 90.1 }),
    ),
    day('2026-06-08', { burnKcal: 2600, activeKcal: 600, intakeKcal: 2100, weightKg: 90.1 }),
  )
})
