import assert from 'node:assert/strict'
import test from 'node:test'
import {
  aggregateAppleRecords,
  latestAppleDate,
  matchAppleRecord,
  mergeAppleDay,
  parseAppleJson,
  type AppleDaily,
} from './apple'

function day(date: string, values: Partial<AppleDaily> = {}): AppleDaily {
  return {
    date,
    burnKcal: null,
    activeKcal: null,
    intakeKcal: null,
    weightKg: null,
    vo2max: null,
    ...values,
  }
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
      day('2026-06-08', {
        burnKcal: 2500,
        activeKcal: 600,
        intakeKcal: 2100,
        weightKg: 90.5,
        vo2max: 48.2,
      }),
      day('2026-06-08', { burnKcal: 2600, activeKcal: null, intakeKcal: null, weightKg: 90.1 }),
    ),
    day('2026-06-08', {
      burnKcal: 2600,
      activeKcal: 600,
      intakeKcal: 2100,
      weightKg: 90.1,
      vo2max: 48.2,
    }),
  )
})

test('Apple VO2 max survives XML aggregation and JSON import', () => {
  const record = matchAppleRecord(
    '<Record type="HKQuantityTypeIdentifierVO2Max" sourceName="Apple Watch" unit="mL/min/kg" startDate="2026-06-08 07:00:00 -0400" value="49.26"/>',
  )

  assert.deepEqual(record, {
    date: '2026-06-08',
    kind: 'vo2max',
    value: 49.26,
    unit: 'mL/min/kg',
    source: 'Apple Watch',
  })
  assert.deepEqual(aggregateAppleRecords(record ? [record] : []), [
    day('2026-06-08', { vo2max: 49.3 }),
  ])
  assert.deepEqual(parseAppleJson({ days: [{ date: '2026-06-08', vo2max: 49.26 }] }), [
    day('2026-06-08', { vo2max: 49.3 }),
  ])
})
