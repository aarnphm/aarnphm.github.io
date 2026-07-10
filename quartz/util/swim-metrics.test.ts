import assert from 'node:assert/strict'
import test from 'node:test'
import { swimPaceSeconds, swimStrokeRate } from './swim-metrics'

test('derives plausible swim pace and rejects broken distance data', () => {
  assert.equal(swimPaceSeconds(1_000, 1_590), 159)
  assert.equal(swimPaceSeconds(2_299.6, 306), null)
  assert.equal(swimPaceSeconds(0, 1_000), null)
})

test('derives stroke rate from count and active stroke time', () => {
  assert.equal(swimStrokeRate(420, 900), 28)
  assert.equal(swimStrokeRate(0, 900), null)
  assert.equal(swimStrokeRate(420, 0), null)
  assert.equal(swimStrokeRate(2_000, 300), null)
})
