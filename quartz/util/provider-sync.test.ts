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
