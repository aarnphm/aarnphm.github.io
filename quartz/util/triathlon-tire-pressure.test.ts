import assert from 'node:assert/strict'
import test from 'node:test'
import {
  calculateTirePressure,
  DEFAULT_TIRE_PRESSURE_SELECTION,
  latestMorningBodyWeight,
} from './triathlon-tire-pressure'

test('uses the latest valid morning body-composition weight', () => {
  assert.deepEqual(
    latestMorningBodyWeight([
      { date: '2026-08-15', kg: 86.4 },
      { date: '2026-08-17', kg: null },
      { date: '2026-08-16', kg: 86.06 },
    ]),
    { date: '2026-08-16', kg: 86.06 },
  )
})

test('matches the published SILCA equation for the equipped Cervélo', () => {
  const recommendation = calculateTirePressure(86.06, DEFAULT_TIRE_PRESSURE_SELECTION)

  assert.ok(recommendation)
  assert.equal(recommendation.frontPsi, 78.5)
  assert.equal(recommendation.rearPsi, 80.5)
  assert.equal(Number(recommendation.bikeKg.toFixed(3)), 9.979)
  assert.equal(Number(recommendation.systemKg.toFixed(3)), 96.039)
  assert.equal(recommendation.measuredWidthMm, 28)
  assert.equal(recommendation.diameterMm, 622)
  assert.equal(recommendation.wheelCompatibilityWarning, false)
})

test('uses the Speedmax system mass and even triathlon load distribution', () => {
  const recommendation = calculateTirePressure(86.06, {
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    bike: 'speedmax',
  })

  assert.ok(recommendation)
  assert.equal(recommendation.frontPsi, 80)
  assert.equal(recommendation.rearPsi, 80)
  assert.equal(Number(recommendation.bikeKg.toFixed(3)), 11.793)
})

test('keeps pressure stable across equal measured casing widths and flags Reserve compatibility', () => {
  const princeton = calculateTirePressure(86.06, DEFAULT_TIRE_PRESSURE_SELECTION)
  const reserve = calculateTirePressure(86.06, {
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    wheel: 'reserve',
  })

  assert.ok(princeton)
  assert.ok(reserve)
  assert.equal(reserve.frontPsi, princeton.frontPsi)
  assert.equal(reserve.rearPsi, princeton.rearPsi)
  assert.equal(reserve.wheelCompatibilityWarning, true)
  assert.equal(reserve.wheel.frontInnerWidthMm, 25.4)
  assert.equal(reserve.wheel.rearInnerWidthMm, 24.8)
})

test('keeps Pirelli TPU and tubeless setups at the high-performance 1.00 coefficient', () => {
  const tpu = calculateTirePressure(86.06, DEFAULT_TIRE_PRESSURE_SELECTION)
  const tubeless = calculateTirePressure(86.06, {
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    tire: 'tubeless',
  })

  assert.ok(tpu)
  assert.ok(tubeless)
  assert.equal(tubeless.frontPsi, tpu.frontPsi)
  assert.equal(tubeless.rearPsi, tpu.rearPsi)
  assert.equal(tubeless.tire.pressureCoefficient, 1)
})

test('rejects pressure inputs outside the SILCA system-weight and speed domains', () => {
  assert.equal(calculateTirePressure(10, DEFAULT_TIRE_PRESSURE_SELECTION), null)
  assert.equal(
    calculateTirePressure(86.06, { ...DEFAULT_TIRE_PRESSURE_SELECTION, speedMph: 34 }),
    null,
  )
})
