import assert from 'node:assert/strict'
import test from 'node:test'
import { estimateHeartRatePhysiology } from './heart-rate-physiology'

const samples = (step = 10, heartRate = 120) =>
  Array.from({ length: 600 / step + 1 }, (_, index) => ({
    elapsedS: index * step,
    distanceKm: (index * step) / 1000,
    heartRate,
  }))

test('HR estimates use elapsed duration and a measured opening baseline', () => {
  const estimate = estimateHeartRatePhysiology(samples(), 200)
  assert.ok(estimate)
  assert.equal(estimate.source, 'garden-estimate')
  assert.equal(estimate.baselineHeartRateBpm, 120)
  assert.equal(estimate.points[0].stamina, 100)
  assert.equal(estimate.points[5].performanceCondition, null)
  assert.equal(estimate.points[6].performanceCondition, 0)
  const ending = estimate.points.at(-1)
  assert.ok(ending?.stamina != null && ending.stamina < 100 && ending.stamina > 99)
  assert.equal(ending.potentialStamina, ending.stamina)
  const coarse = estimateHeartRatePhysiology(samples(30), 200)
  assert.ok(coarse?.points.at(-1)?.stamina != null)
  assert.ok(Math.abs((coarse.points.at(-1)?.stamina ?? 0) - ending.stamina) < 1e-10)
})

test('HR change is negative for rising HR and positive for falling HR', () => {
  for (const endHr of [100, 140]) {
    const estimate = estimateHeartRatePhysiology(
      samples().map(point => ({ ...point, heartRate: point.elapsedS <= 60 ? 120 : endHr })),
      200,
    )
    assert.ok(estimate)
    assert.equal(estimate.points.at(-1)?.performanceCondition, endHr === 100 ? 10 : -10)
  }
})

test('HR estimates preserve missing samples and reject insufficient or invalid coverage', () => {
  const points = samples().map(point => ({
    ...point,
    heartRate: point.elapsedS === 300 ? null : point.heartRate,
  }))
  const estimate = estimateHeartRatePhysiology(points, 200)
  assert.ok(estimate)
  assert.equal(estimate.points[30].stamina, null)
  assert.equal(estimate.points[30].performanceCondition, null)
  assert.equal(estimate.points[31].stamina, null)
  assert.equal(
    estimateHeartRatePhysiology(
      samples().filter(point => point.elapsedS < 100 || point.elapsedS > 500),
      200,
    ),
    null,
  )
  assert.equal(estimateHeartRatePhysiology(points, Number.NaN), null)
  assert.equal(estimateHeartRatePhysiology([points[1], points[0]], 200), null)
  assert.equal(
    estimateHeartRatePhysiology(
      samples().map(point => ({ ...point, heartRate: null })),
      200,
    ),
    null,
  )
})
