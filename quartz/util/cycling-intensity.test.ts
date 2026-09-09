import assert from 'node:assert/strict'
import test from 'node:test'
import { buildCyclingIntensityTrace } from './cycling-intensity'

const input = (watts: (number | null)[]) => ({
  streams: { time: watts.map((_, i) => i), watts },
  metrics: { normalizedPower: 200, intensityFactor: 0.8 },
  startOffsetS: 0,
  elapsedTimeS: watts.length - 1,
  route: [
    { elapsedS: 0, d: 0 },
    { elapsedS: watts.length - 1, d: 10 },
  ],
  athleteFtp: 400,
})

test('uses full Wahoo power and the historical FTP implied by native NP and IF', () => {
  const trace = buildCyclingIntensityTrace(input(Array(121).fill(200)))
  assert.ok(trace)
  assert.equal(trace.ftpWatts, 250)
  assert.equal(trace.ftpSource, 'wahoo-summary')
  assert.equal(trace.points[28].intensityFactor, null)
  assert.equal(trace.points[29].intensityFactor, 0.8)
  assert.equal(trace.points.at(-1)?.distanceKm, 10)
})

test('includes coasting zeros in the 30-second rolling NP', () => {
  const watts = Array.from({ length: 120 }, (_, i) => (i % 2 ? 0 : 400))
  const trace = buildCyclingIntensityTrace(input(watts))
  assert.ok(trace)
  assert.equal(trace.points.at(-1)?.intensityFactor, 0.8)
  const allZero = buildCyclingIntensityTrace(input(Array(60).fill(0)))
  assert.equal(allZero?.points.at(-1)?.intensityFactor, 0)
})

test('resets rolling power at missing data and requires 30 fresh seconds', () => {
  const watts: (number | null)[] = Array(121).fill(200)
  watts[60] = null
  const trace = buildCyclingIntensityTrace(input(watts))
  assert.ok(trace)
  assert.equal(trace.points[59].intensityFactor, 0.8)
  assert.equal(trace.points[60].intensityFactor, null)
  assert.equal(trace.points[89].intensityFactor, null)
  assert.equal(trace.points[90].intensityFactor, 0.8)
})

test('aligns a recording offset and clips the end without extrapolating watts', () => {
  const trace = buildCyclingIntensityTrace({
    ...input(Array(61).fill(200)),
    startOffsetS: 10,
    elapsedTimeS: 90,
  })
  assert.ok(trace)
  assert.equal(trace.points[38].intensityFactor, null)
  assert.equal(trace.points[39].intensityFactor, 0.8)
  assert.equal(trace.points[70].intensityFactor, 0.8)
  assert.equal(trace.points[71].intensityFactor, null)
})

test('requires FTP and records an athlete FTP fallback explicitly', () => {
  const base = {
    ...input(Array(61).fill(200)),
    metrics: { normalizedPower: null, intensityFactor: null },
  }
  const noFtp = buildCyclingIntensityTrace({ ...base, athleteFtp: null })
  assert.equal(noFtp, null)
  const fallback = buildCyclingIntensityTrace({ ...base, athleteFtp: 200 })
  assert.equal(fallback?.ftpSource, 'athlete')
  assert.equal(fallback?.points.at(-1)?.intensityFactor, 1)
})

test('rejects sparse, unordered and invalid streams instead of inventing a curve', () => {
  const base = input(Array(61).fill(200))
  assert.equal(
    buildCyclingIntensityTrace({
      ...base,
      streams: { time: base.streams.time.map(t => t * 10), watts: base.streams.watts },
      elapsedTimeS: 600,
    }),
    null,
  )
  assert.equal(
    buildCyclingIntensityTrace({
      ...base,
      streams: { ...base.streams, time: base.streams.time.toReversed() },
    }),
    null,
  )
  assert.equal(buildCyclingIntensityTrace({ ...base, streams: undefined }), null)
})

test('bounds the serialized trace and preserves a gap inside a downsampling bucket', () => {
  const watts: (number | null)[] = Array(3601).fill(200)
  watts[1000] = null
  const trace = buildCyclingIntensityTrace(input(watts))
  assert.ok(trace)
  assert.ok(trace.points.length <= 322)
  assert.equal(trace.points.find(point => point.elapsedS === 1008)?.intensityFactor, null)
  assert.equal(trace.points.at(-1)?.distanceKm, 10)
})

test('weights variable power by the fourth power of complete 30-second windows', () => {
  const trace = buildCyclingIntensityTrace(input([...Array(30).fill(0), ...Array(30).fill(400)]))
  assert.ok(trace)
  const sumFourthPowers = (30 * 31 * 61 * (3 * 30 ** 2 + 3 * 30 - 1)) / 30
  const np = (400 / 30) * (sumFourthPowers / 31) ** 0.25
  assert.ok(Math.abs((trace.points.at(-1)?.intensityFactor ?? 0) - np / 250) < 1e-12)
})
