import assert from 'node:assert/strict'
import test from 'node:test'
import {
  localDayEndUtcMs,
  localDayStartUtcMs,
  localIsoDay,
  localIsoDayOffset,
  shiftIsoDay,
} from './local-date'

const TORONTO = 'America/Toronto'

test('formats the current day in the configured local timezone', () => {
  const lateEvening = Date.parse('2026-07-01T02:45:00.000Z')

  assert.equal(localIsoDay(lateEvening, TORONTO), '2026-06-30')
  assert.equal(localIsoDayOffset(0, lateEvening, TORONTO), '2026-06-30')
  assert.equal(localIsoDayOffset(1, lateEvening, TORONTO), '2026-07-01')
})

test('computes UTC bounds for a local day across daylight saving time', () => {
  assert.equal(
    new Date(localDayStartUtcMs('2026-06-30', TORONTO)).toISOString(),
    '2026-06-30T04:00:00.000Z',
  )
  assert.equal(
    new Date(localDayEndUtcMs('2026-06-30', TORONTO)).toISOString(),
    '2026-07-01T03:59:59.999Z',
  )
  assert.equal(
    new Date(localDayStartUtcMs('2026-12-30', TORONTO)).toISOString(),
    '2026-12-30T05:00:00.000Z',
  )
  assert.equal(
    new Date(localDayEndUtcMs('2026-12-30', TORONTO)).toISOString(),
    '2026-12-31T04:59:59.999Z',
  )
})

test('shifts ISO days without using the process timezone', () => {
  assert.equal(shiftIsoDay('2026-06-30', 1), '2026-07-01')
  assert.equal(shiftIsoDay('2026-03-01', -1), '2026-02-28')
})
