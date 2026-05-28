import assert from 'node:assert/strict'
import test from 'node:test'
import { flushBuildSpans, logBuildSpan, shouldLogBuildSpan, slowBuildThresholdMs } from './perf'

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

test('verbose logging overrides slow build threshold filtering', () => {
  assert.equal(shouldLogBuildSpan({ verbose: true, slowBuildThreshold: 100 }, 1), true)
})

test('all build spans override slow build threshold filtering', () => {
  const argv = { verbose: false, slowBuildThreshold: 100, allBuildSpans: true }
  const lines: string[] = []
  const originalLog = console.log
  console.log = (line?: unknown) => {
    lines.push(String(line))
  }
  try {
    logBuildSpan(argv, 'write', 'public/a.html', 1)
    flushBuildSpans(argv)
  } finally {
    console.log = originalLog
  }

  assert.equal(shouldLogBuildSpan(argv, 1), true)
  assert.equal(lines.length, 1)
  assert.equal(lines[0].startsWith('[write] public/a.html'), true)
})

test('invalid slow build thresholds are ignored', () => {
  assert.equal(slowBuildThresholdMs({ slowBuildThreshold: 0 }), undefined)
  assert.equal(slowBuildThresholdMs({ slowBuildThreshold: Number.NaN }), undefined)
})

test('slow build spans are summarized when threshold logging is enabled', () => {
  const argv = { verbose: false, slowBuildThreshold: 100 }
  const lines: string[] = []
  const originalLog = console.log
  console.log = (line?: unknown) => {
    lines.push(String(line))
  }
  try {
    logBuildSpan(argv, 'write', 'public/a.html', 150)
    logBuildSpan(argv, 'write', 'public/b.html', 250)
    flushBuildSpans(argv)
  } finally {
    console.log = originalLog
  }

  assert.equal(
    lines.some(line => line.startsWith('[slow] 2 spans >= 100ms')),
    true,
  )
  assert.equal(
    lines.some(line => line.startsWith('[slow:write] 2 spans')),
    true,
  )
  assert.equal(
    lines.some(line => line.includes('public/b.html')),
    true,
  )
})
