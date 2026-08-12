import assert from 'node:assert/strict'
import test from 'node:test'
import {
  swimChartMetric,
  swimLengthAverages,
  swimLengthMetrics,
  swimPaceSeconds,
  swimStrokeRate,
} from './swim-metrics'

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

test('derives pool-length cadence and SWOLF while excluding drills', () => {
  assert.deepEqual(swimLengthMetrics({ durationS: 25.6, strokeCount: 11 }), {
    strokesPerLength: 11,
    swolf: 37,
  })
  assert.equal(swimLengthMetrics({ durationS: 25.6, strokeCount: 11, stroke: 'kickboard' }), null)
  assert.equal(swimLengthMetrics({ durationS: 25.6, strokeCount: null }), null)
  assert.deepEqual(
    swimLengthAverages([
      { durationS: 25.6, strokeCount: 11 },
      { durationS: 32.5, strokeCount: 10 },
      { durationS: 25, strokeCount: null, stroke: 'kickboard' },
      { durationS: 27.4, strokeCount: 11 },
    ]),
    { strokesPerLength: 10.7, swolf: 39.3 },
  )
})

test('parses serialized swim chart metrics', () => {
  assert.equal(swimChartMetric('cadence'), 'cadence')
  assert.equal(swimChartMetric('swolf'), 'swolf')
  assert.equal(swimChartMetric('pace'), 'pace')
  assert.equal(swimChartMetric('unknown'), 'pace')
})
