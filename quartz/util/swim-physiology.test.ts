import assert from 'node:assert/strict'
import test from 'node:test'
import { estimateSwimPhysiology, type SwimPhysiologySample } from './swim-physiology'

const samples = (after: Partial<SwimPhysiologySample> = {}): SwimPhysiologySample[] => {
  let distanceKm = 0
  return Array.from({ length: 91 }, (_, i) => {
    const change = i > 36 ? after : {}
    const speed = change.speedMps ?? 0.7
    if (i > 0) distanceKm += (speed * 10) / 1000
    return {
      elapsedS: i * 10,
      distanceKm,
      heartRate: 140,
      speedMps: speed,
      strokeRateSpm: 24,
      ...change,
    }
  })
}

test('swim model uses duration, establishes an opening baseline, and labels inferred exertion', () => {
  const result = estimateSwimPhysiology(samples(), 200, 'stream')
  assert.ok(result)
  assert.equal(result.source, 'garden-estimate')
  assert.equal(result.exertionSource, 'heart-rate')
  assert.equal(result.strokeRateSource, 'stream')
  assert.equal(result.points[0].stamina, 100)
  assert.equal(result.points[12].performanceCondition, null)
  assert.ok(Math.abs(result.points.at(-1)?.performanceCondition ?? Infinity) < 1e-9)
  assert.ok((result.points.at(-1)?.stamina ?? 100) < 100)
  assert.equal(result.points.at(-1)?.stamina, result.points.at(-1)?.potentialStamina)
})

test('higher HR, faster swimming, and more strokes per metre increase modeled depletion', () => {
  const steady = estimateSwimPhysiology(samples(), 200, 'stream')
  assert.ok(steady)
  for (const change of [{ heartRate: 175 }, { speedMps: 0.9 }, { strokeRateSpm: 32 }]) {
    const result = estimateSwimPhysiology(samples(change), 200, 'stream')
    assert.ok(result)
    assert.ok((result.points.at(-1)?.stamina ?? 100) < (steady.points.at(-1)?.stamina ?? 0))
    if ('heartRate' in change || 'strokeRateSpm' in change)
      assert.ok((result.points.at(-1)?.performanceCondition ?? 0) < 0)
  }
})

test('swim model preserves gaps and works without inventing stroke data or RPE', () => {
  const missing = samples().map(p => ({ ...p, heartRate: p.elapsedS === 600 ? null : p.heartRate }))
  const result = estimateSwimPhysiology(missing, 200, 'activity-average')
  assert.ok(result)
  assert.equal(result.points[60].stamina, null)
  assert.equal(result.points[60].performanceCondition, null)
  const noStroke = estimateSwimPhysiology(
    samples().map(p => ({ ...p, strokeRateSpm: null })),
    200,
    'unavailable',
  )
  assert.ok(noStroke)
  assert.equal(noStroke.baselineStrokesPerM, null)
  assert.equal(
    estimateSwimPhysiology(
      samples().map(p => ({ ...p, heartRate: null })),
      200,
      'unavailable',
    ),
    null,
  )
  assert.equal(estimateSwimPhysiology(samples().slice(0, 12), 200, 'stream'), null)
  assert.equal(estimateSwimPhysiology(samples(), Number.NaN, 'stream'), null)
})
