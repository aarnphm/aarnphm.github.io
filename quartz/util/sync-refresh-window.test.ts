import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_SYNC_REFRESH_WINDOW_DAYS,
  calendarRefreshStart,
  stravaSyncRefreshDays,
  syncRefreshDays,
} from './sync-refresh-window'

test('shared and Strava refresh windows honor their specific precedence', () => {
  const environment = { SYNC_REFRESH_DAYS: '5', STRAVA_SYNC_REFRESH_DAYS: '3' }

  assert.equal(syncRefreshDays(environment), 5)
  assert.equal(stravaSyncRefreshDays(environment), 3)
  assert.equal(syncRefreshDays({ STRAVA_SYNC_REFRESH_DAYS: '3' }), 3)
  assert.equal(stravaSyncRefreshDays({ SYNC_REFRESH_DAYS: '5' }), 5)
  assert.equal(syncRefreshDays({}), DEFAULT_SYNC_REFRESH_WINDOW_DAYS)
})

test('refresh windows reject negative, fractional, and nonnumeric days', () => {
  for (const value of ['-1', '1.5', 'three']) {
    assert.throws(
      () => syncRefreshDays({ SYNC_REFRESH_DAYS: value }),
      /SYNC_REFRESH_DAYS must be a nonnegative integer/,
    )
  }
})

test('calendarRefreshStart subtracts the configured local-day overlap', () => {
  const now = Date.parse('2026-09-01T16:00:00Z')

  assert.equal(calendarRefreshStart(3, now), '2026-08-29')
  assert.equal(calendarRefreshStart(1, now), '2026-08-31')
  assert.equal(calendarRefreshStart(0, now), '2026-09-01')
})
