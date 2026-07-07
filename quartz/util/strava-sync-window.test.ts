import assert from 'node:assert/strict'
import test from 'node:test'
import { stravaFetchAfter } from './strava-sync-window'

test('stravaFetchAfter refreshes a bounded recent overlap', () => {
  const last = Math.floor(Date.parse('2026-07-07T14:01:53Z') / 1000)

  assert.equal(stravaFetchAfter(last, false, 7), last - 7 * 86_400)
})

test('stravaFetchAfter preserves full refresh and empty-cache behavior', () => {
  assert.equal(stravaFetchAfter(123, true, 7), 0)
  assert.equal(stravaFetchAfter(0, false, 7), 0)
  assert.equal(stravaFetchAfter(undefined, false, 7), 0)
  assert.equal(stravaFetchAfter(123, false, 0), 123)
})
