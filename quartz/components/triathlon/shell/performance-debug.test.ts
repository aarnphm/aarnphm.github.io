import assert from 'node:assert/strict'
import test from 'node:test'
import { summarizeFrameDurations } from './performance-debug'

test('frame summary reports cadence, p95 duration, and frames beyond the budget', () => {
  const summary = summarizeFrameDurations([10, 12, 14, 16, 18, 20], 16)

  assert.equal(summary.fps.toFixed(1), '66.7')
  assert.equal(summary.p95, 20)
  assert.equal(summary.slowRatio.toFixed(2), '0.33')
})

test('frame summary handles an empty sample window', () => {
  assert.deepEqual(summarizeFrameDurations([]), { fps: 0, p95: 0, slowRatio: 0 })
})
