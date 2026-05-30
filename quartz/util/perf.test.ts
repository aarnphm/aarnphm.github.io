import assert from 'node:assert/strict'
import test from 'node:test'
import { shouldLogBuildSpan, slowBuildThresholdMs } from './perf'

test('slow build threshold logs spans at or above the threshold', () => {
  const argv = { verbose: false, slowBuildThreshold: 100 }

  assert.equal(slowBuildThresholdMs(argv), 100)
  assert.equal(shouldLogBuildSpan(argv, 99.99), false)
  assert.equal(shouldLogBuildSpan(argv, 100), true)
  assert.equal(shouldLogBuildSpan(argv, 101), true)
})

test('slow build logging falls back to verbose mode without a threshold', () => {
  assert.equal(shouldLogBuildSpan({ verbose: false }, 1000), false)
  assert.equal(shouldLogBuildSpan({ verbose: true }, 1), true)
})

test('invalid slow build thresholds are ignored', () => {
  assert.equal(slowBuildThresholdMs({ slowBuildThreshold: 0 }), undefined)
  assert.equal(slowBuildThresholdMs({ slowBuildThreshold: Number.NaN }), undefined)
})
