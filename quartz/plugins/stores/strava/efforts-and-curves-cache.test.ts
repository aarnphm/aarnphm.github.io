import assert from 'node:assert/strict'
import test from 'node:test'
import { releaseModuleCaches } from '../../../util/module-caches'
import { meanMaxCurve, type EffortTimeline } from './efforts-and-curves'
import { createMeanMaxCurveCache, getMeanMaxCurveCache } from './power-curve-cache'

function timeline(watts: Float64Array): EffortTimeline {
  return {
    watts,
    distanceM: new Float64Array(watts.length),
    altitudeM: new Float64Array(watts.length),
    heartRate: new Float64Array(watts.length),
    wattsObserved: new Uint8Array(watts.length),
  }
}

test('equal watt content preserves exact curves across timelines and isolated return values', () => {
  releaseModuleCaches()
  const watts = new Float64Array([150, 250, 350, 450, 200])
  const first = meanMaxCurve(timeline(watts))
  const expected = structuredClone(first)
  first[0].w = 999
  first[0].activityId = 42
  first.splice(1, 1)
  const second = meanMaxCurve(timeline(Float64Array.from(watts)))
  assert.deepEqual(second, expected)
  assert.notEqual(second, first)
  watts[0] = 900
  assert.equal(meanMaxCurve(timeline(watts))[0].w, 900)
  assert.deepEqual(meanMaxCurve(timeline(new Float64Array([150, 250, 350, 450, 200]))), expected)
})

test('typed-array offset and length define the exact input', () => {
  const buffer = new Float64Array([9999, 210, 315, 420, 280, 8888])
  const other = new Float64Array([210, 315, 420, 280, 1234])
  const first = meanMaxCurve(timeline(buffer.subarray(1, 5)))
  assert.deepEqual(meanMaxCurve(timeline(other.subarray(0, 4))), first)
  assert.equal(meanMaxCurve(timeline(buffer.subarray(0, 4)))[0].w, 9999)
  assert.equal(meanMaxCurve(timeline(buffer.subarray(1, 6)))[0].w, 8888)
})

test('null, empty, nonfinite and fractional input retain their calculation behavior', () => {
  const empty = meanMaxCurve(null)
  empty.push({ s: 1, w: 500 })
  assert.deepEqual(meanMaxCurve(null), [])
  assert.deepEqual(meanMaxCurve(timeline(new Float64Array())), [])
  const values = [0, 100.9, 200.9, 0]
  assert.deepEqual(meanMaxCurve(timeline(Float64Array.from(values))), [
    { s: 1, w: 200 },
    { s: 2, w: 150 },
    { s: 3, w: 100 },
    { s: 4, w: 75 },
  ])
  assert.deepEqual(
    meanMaxCurve(timeline(new Float64Array([-0, 100.9, 200.9, -0]))),
    meanMaxCurve(timeline(Float64Array.from(values))),
  )
  for (const watts of [
    [120, 240, NaN, 500],
    [100, Infinity, 200],
    [-50, -25, 0],
  ]) {
    const expected = meanMaxCurve(timeline(Float64Array.from(watts)))
    assert.deepEqual(meanMaxCurve(timeline(Float64Array.from(watts))), expected)
  }
})

test('bounded LRU evicts old curves, preserves recent results and bypasses oversized input', () => {
  const cache = createMeanMaxCurveCache({ maxEntries: 2, maxTotalPoints: 8, maxEntryPoints: 4 })
  const compute = (values: number[]) => {
    const watts = Float64Array.from(values)
    return cache.getOrCompute(watts, () => meanMaxCurve(timeline(watts)))
  }
  const a = compute([100, 200, 300, 400])
  compute([110, 210, 310, 410])
  assert.deepEqual(compute([100, 200, 300, 400]), a)
  assert.equal(cache.stats().hits, 1)
  compute([120, 220, 320, 420])
  assert.equal(cache.stats().evictions, 1)
  assert.deepEqual(compute([100, 200, 300, 400]), a)
  compute([110, 210, 310, 410])
  assert.equal(cache.stats().misses, 4)
  assert.equal(compute([10, 20, 30, 40, 50]).length, 5)
  assert.equal(cache.stats().skippedOversized, 1)
  assert.equal(cache.stats().totalPoints, 8)
  cache.clear()
  assert.equal(cache.stats().entries, 0)
  assert.equal(cache.stats().totalPoints, 0)
  const byCount = createMeanMaxCurveCache({ maxEntries: 1, maxTotalPoints: 100 })
  for (const values of [
    [10, 20],
    [30, 40],
  ]) {
    const watts = Float64Array.from(values)
    byCount.getOrCompute(watts, () => meanMaxCurve(timeline(watts)))
  }
  assert.equal(byCount.stats().entries, 1)
  assert.equal(byCount.stats().evictions, 1)
})

test('module retirement drops retained curves and algorithm versions have separate identities', () => {
  const cache1 = createMeanMaxCurveCache({ algorithmVersion: 'v1' })
  const cache2 = createMeanMaxCurveCache({ algorithmVersion: 'v2' })
  const watts = new Float64Array([175, 225, 275, 325, 375])
  assert.notEqual(cache1.computeKey(watts), cache2.computeKey(watts))
  const expected = meanMaxCurve(timeline(watts))
  assert.ok(getMeanMaxCurveCache().stats().entries > 0)
  releaseModuleCaches()
  assert.equal(getMeanMaxCurveCache().stats().entries, 0)
  assert.equal(getMeanMaxCurveCache().stats().totalPoints, 0)
  assert.deepEqual(meanMaxCurve(timeline(watts)), expected)
})

test('hash collisions cannot return another watt series curve', () => {
  const cache = createMeanMaxCurveCache()
  const a = new Float64Array([243.7889529466629, 99.33694511651993])
  const b = new Float64Array([43.57571363449097, 189.170852124691])
  assert.equal(cache.computeKey(a), cache.computeKey(b))
  const first = cache.getOrCompute(a, () => meanMaxCurve(timeline(a)))
  const second = cache.getOrCompute(b, () => meanMaxCurve(timeline(b)))
  assert.notDeepEqual(second, first)
  assert.deepEqual(second, meanMaxCurve(timeline(b)))
  assert.deepEqual(
    cache.getOrCompute(a, () => meanMaxCurve(timeline(a))),
    first,
  )
})
