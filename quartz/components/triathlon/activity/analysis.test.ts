import assert from 'node:assert/strict'
import test from 'node:test'
import { activityScrubElapsedIndexAt, activityScrubIndexAt } from './analysis'

const routeLessSamples = [
  { d: 0, elapsedS: 0 },
  { d: 520, elapsedS: 520 },
  { d: 1_040, elapsedS: 1_040 },
  { d: 1_560, elapsedS: 1_560 },
]

test('route-less activity scrub selects the nearest elapsed-axis sample', () => {
  assert.equal(activityScrubIndexAt(routeLessSamples, -20), 0)
  assert.equal(activityScrubIndexAt(routeLessSamples, 510), 1)
  assert.equal(activityScrubIndexAt(routeLessSamples, 780), 1)
  assert.equal(activityScrubIndexAt(routeLessSamples, 800), 2)
  assert.equal(activityScrubIndexAt(routeLessSamples, 2_000), 3)
})

test('linked activity charts synchronize samples by elapsed time', () => {
  const distanceSamples = [
    { d: 0, elapsedS: 90 },
    { d: 4.8, elapsedS: 600 },
    { d: 10.2, elapsedS: 1_100 },
    { d: 15.4, elapsedS: 1_700 },
  ]
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 0), 0)
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 850), 1)
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 900), 2)
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 2_000), 3)
  assert.equal(activityScrubElapsedIndexAt([], 900), -1)
})
