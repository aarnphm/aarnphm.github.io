import assert from 'node:assert/strict'
import test from 'node:test'
import { trainingPeaksActivityPeaks } from './trainingpeaks-calendar-peaks'

// Live recordings cannot exercise missing seconds, source disagreement, or malformed clocks.
const trace = (values: number[], time = values.map((_, index) => index)) => ({
  time,
  heartrate: values,
})

test('uses complete elapsed HR windows and exact existing power-curve durations', () => {
  const peaks = trainingPeaksActivityPeaks({
    strava: trace([...Array<number>(60).fill(120), ...Array<number>(5).fill(180)]),
    powerCurve: [
      { s: 5, w: 510 },
      { s: 59, w: 301 },
      { s: 60, w: 300 },
    ],
  })
  assert.equal(peaks?.find(point => point.seconds === 5)?.heartRateBpm, 180)
  assert.equal(peaks?.find(point => point.seconds === 60)?.heartRateBpm, 125)
  assert.equal(peaks?.find(point => point.seconds === 5)?.powerWatts, 510)
  assert.equal(peaks?.find(point => point.seconds === 60)?.powerWatts, 300)
  assert.equal(peaks?.find(point => point.seconds === 300)?.powerWatts, null)
  assert.equal(peaks?.find(point => point.seconds === 300)?.heartRateBpm, null)
})

test('keeps missing HR seconds and invalid HR out of complete windows', () => {
  for (const strava of [
    trace([150, 150, 150, 150, 150], [0, 1, 2, 4, 5]),
    trace([150, 150, 0, 150, 150]),
    trace([150, 150, NaN, 150, 150]),
    trace([150, 150, 150, 150, 150], [0, 1, 1, 2, 3]),
    trace([150, 150, 150, 150, 150], [4, 3, 2, 1, 0]),
    trace([150, 150, 150, 150, 150], [0, 1]),
  ])
    assert.equal(trainingPeaksActivityPeaks({ strava }), undefined)
})

test('keeps provider clocks separate and reports the source of each HR interval', () => {
  const peaks = trainingPeaksActivityPeaks({
    garmin: trace(Array<number>(5).fill(170), [100, 101, 102, 103, 104]),
    strava: trace(Array<number>(60).fill(150)),
  })
  assert.equal(peaks?.find(point => point.seconds === 5)?.heartRateBpm, 170)
  assert.equal(peaks?.find(point => point.seconds === 5)?.heartRateSource, 'garmin')
  assert.equal(peaks?.find(point => point.seconds === 60)?.heartRateBpm, 150)
  assert.equal(peaks?.find(point => point.seconds === 60)?.heartRateSource, 'strava')
})

test('preserves observed zero power and rejects absent or invalid curves', () => {
  assert.equal(trainingPeaksActivityPeaks({}), undefined)
  assert.equal(trainingPeaksActivityPeaks({ powerCurve: [{ s: 5, w: NaN }] }), undefined)
  const peaks = trainingPeaksActivityPeaks({ powerCurve: [{ s: 5, w: 0 }] })
  assert.equal(peaks?.find(point => point.seconds === 5)?.powerWatts, 0)
  assert.equal(peaks?.find(point => point.seconds === 5)?.heartRateSource, null)
})
