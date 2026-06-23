import assert from 'node:assert/strict'
import test from 'node:test'
import { ouraSleepCalendarDay } from './oura'

test('keys detailed sleep by local wake date', () => {
  assert.equal(
    ouraSleepCalendarDay({ day: '2026-06-22', bedtime_end: '2026-06-21T18:50:37.000-04:00' }),
    '2026-06-21',
  )
  assert.equal(
    ouraSleepCalendarDay({ day: '2026-06-22', bedtime_end: '2026-06-22T09:00:32.000-04:00' }),
    '2026-06-22',
  )
  assert.equal(ouraSleepCalendarDay({ day: '2026-06-22' }), '2026-06-22')
  assert.equal(ouraSleepCalendarDay({ day: 20260622, bedtime_end: 'bad-date' }), null)
})
