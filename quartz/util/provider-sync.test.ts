import assert from 'node:assert/strict'
import test from 'node:test'
import { latestProviderSync } from './provider-sync'

test('latestProviderSync selects the newest valid provider timestamp', () => {
  assert.equal(
    latestProviderSync(
      { lastSync: Date.parse('2026-08-31T22:25:41.898Z') },
      { lastSync: Date.parse('2026-08-31T22:28:38.322Z') },
      Date.parse('2026-08-31T22:28:52.935Z'),
      { lastSync: Number.NaN },
      null,
    ),
    Date.parse('2026-08-31T22:28:52.935Z'),
  )
})

test('latestProviderSync returns zero without a valid timestamp', () => {
  assert.equal(latestProviderSync(undefined, null, Number.NaN, -1), 0)
})

test('a lactate-threshold-only refresh advances rendered data without changing activity freshness', () => {
  const garmin = { lastSync: 100, sleepLastSync: 150, lactateThresholdLastSync: 200 }
  assert.equal(latestProviderSync(garmin), 200)
  assert.equal(garmin.lastSync, 100)
  assert.equal(latestProviderSync({ lastSync: 100, lactateThresholdLastSync: NaN }), 100)
})

test('a sleep-only Garmin refresh advances rendered data without changing activity freshness', () => {
  const garmin = { lastSync: 100, sleepLastSync: 200 }
  assert.equal(latestProviderSync(garmin), 200)
  assert.equal(garmin.lastSync, 100)
  assert.equal(latestProviderSync({ lastSync: 100, sleepLastSync: Number.NaN }), 100)
})
