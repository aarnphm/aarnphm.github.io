import assert from 'node:assert/strict'
import test from 'node:test'
import {
  estimateFrameBudget,
  frameChartRatio,
  monotoneSlopes,
  smoothWindow,
  summarizeFrameDurations,
} from './performance-debug'

const hermite = (
  values: readonly number[],
  slopes: readonly number[],
  index: number,
  t: number,
): number => {
  const h00 = 2 * t ** 3 - 3 * t ** 2 + 1
  const h10 = t ** 3 - 2 * t ** 2 + t
  const h01 = -2 * t ** 3 + 3 * t ** 2
  const h11 = t ** 3 - t ** 2
  return (
    h00 * values[index] + h10 * slopes[index] + h01 * values[index + 1] + h11 * slopes[index + 1]
  )
}

test('frame summary reports cadence, p95 duration, and frames beyond the budget', () => {
  const summary = summarizeFrameDurations([10, 12, 14, 16, 18, 20], 16)

  assert.equal(summary.fps.toFixed(1), '66.7')
  assert.equal(summary.p95, 20)
  assert.equal(summary.slowRatio.toFixed(2), '0.33')
})

test('frame summary handles an empty sample window', () => {
  assert.deepEqual(summarizeFrameDurations([]), { fps: 0, p95: 0, slowRatio: 0 })
})

test('frame budget snaps to the refresh cadence the frames were actually drawn at', () => {
  const promotion = Array.from({ length: 60 }, (_, index) => 8.3 + (index % 5) * 0.4)
  const sixty = Array.from({ length: 60 }, (_, index) => 16.7 + (index % 5) * 0.4)

  assert.equal(estimateFrameBudget(promotion).toFixed(1), '8.3')
  assert.equal(estimateFrameBudget(sixty).toFixed(1), '16.7')
  assert.equal(estimateFrameBudget([9, 9, 9]).toFixed(1), '16.7')
})

test('smoothing damps single-bucket noise without leaving the sampled range', () => {
  const raw = [16, 16, 16, 40, 16, 16, 16]
  const smoothed = smoothWindow(raw)

  assert.equal(smoothed.length, raw.length)
  assert.ok(smoothed[3] < 40 && smoothed[3] > 16)
  assert.ok(smoothed.every(value => value >= 16 && value <= 40))
  assert.deepEqual(smoothWindow([16, 40]), [16, 40])
})

test('monotone slopes keep a spike from overshooting between buckets', () => {
  const values = [16, 16, 100, 16, 16]
  const slopes = monotoneSlopes(values)

  for (let index = 0; index < values.length - 1; index += 1) {
    const low = Math.min(values[index], values[index + 1])
    const high = Math.max(values[index], values[index + 1])
    for (let step = 0; step <= 20; step += 1) {
      const sampled = hermite(values, slopes, index, step / 20)
      assert.ok(sampled >= low - 1e-9 && sampled <= high + 1e-9)
    }
  }
})

test('monotone slopes follow a rising ramp instead of flattening it', () => {
  const slopes = monotoneSlopes([10, 20, 30, 40])

  assert.deepEqual(slopes, [10, 10, 10, 10])
})

test('chart ratio spreads frame durations across a logarithmic axis', () => {
  assert.equal(frameChartRatio(4), 0)
  assert.equal(frameChartRatio(250), 1)
  assert.equal(frameChartRatio(1), 0)
  assert.equal(frameChartRatio(4000), 1)

  const budget = frameChartRatio(1000 / 60)
  const half = frameChartRatio(1000 / 30)
  const quarter = frameChartRatio(1000 / 15)
  assert.ok(budget > 0.3 && budget < 0.4)
  assert.ok(Math.abs(half - budget - (quarter - half)) < 1e-9)
})
