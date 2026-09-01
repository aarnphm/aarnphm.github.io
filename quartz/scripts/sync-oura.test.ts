import assert from 'node:assert/strict'
import test from 'node:test'
import { ouraRefreshRange } from './sync-oura'

const NOW = Date.parse('2026-09-01T16:00:00Z')

test('Oura routine refresh uses the shared inclusive calendar window', () => {
  assert.deepEqual(ouraRefreshRange(false, 3, NOW), {
    start: '2026-08-29',
    end: '2026-09-01',
    endExclusive: '2026-09-02',
    heartRateStart: '2026-08-29',
  })
})

test('Oura schema refresh keeps the full daily backfill and bounded heart rate', () => {
  assert.deepEqual(ouraRefreshRange(true, 3, NOW), {
    start: '2025-09-01',
    end: '2026-09-01',
    endExclusive: '2026-09-02',
    heartRateStart: '2026-08-29',
  })
})
