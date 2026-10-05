import assert from 'node:assert/strict'
import test from 'node:test'
import { sampleTrend, trendChartGeometry } from './thresholds'

test('trend chart preserves asymmetric model bounds and maps its labels to the plot edges', () => {
  const samples = { centers: [100, 110], los: [100, 90], his: [100, 160], days: 1 }
  const chart = trendChartGeometry(false, samples, false)
  assert.equal(chart.low, 87)
  assert.equal(chart.high, 163)
  assert.equal(chart.line, 'M 0.00 24.87 L 100.00 20.92')
  assert.equal(chart.band, 'M 0.00 24.87 L 100.00 1.18 L 100.00 28.82 L 0.00 24.87 Z')
  assert.deepEqual(sampleTrend(samples, 1), { value: 110, lo: 90, hi: 160, days: 1 })
})

test('pace inversion reflects both the forecast line and its full interval', () => {
  const samples = { centers: [100, 110], los: [100, 90], his: [100, 160], days: 1 }
  const chart = trendChartGeometry(true, samples, false)
  assert.equal(chart.line, 'M 0.00 5.13 L 100.00 9.08')
  assert.equal(chart.band, 'M 0.00 5.13 L 100.00 28.82 L 100.00 1.18 L 0.00 5.13 Z')
  assert.deepEqual(sampleTrend(samples, 0.5), { value: 105, lo: 95, hi: 130, days: 0.5 })
})

test('a flat cycling forecast retains a finite scale and the full uncertainty range', () => {
  const chart = trendChartGeometry(
    false,
    { centers: [30, 30, 30], los: [30, 28, 27], his: [30, 33, 35], days: 2 },
    false,
  )
  assert.ok(chart.low < 27)
  assert.ok(chart.high > 35)
  assert.doesNotMatch(chart.line + chart.band, /NaN|Infinity/)
  const heights = chart.line.match(/(?:M|L) [\d.]+ ([\d.]+)/g)
  assert.equal(new Set(heights?.map(point => point.split(' ')[2])).size, 1)
})
