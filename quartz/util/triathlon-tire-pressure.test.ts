import assert from 'node:assert/strict'
import test from 'node:test'
import {
  calculateTirePressure,
  DEFAULT_TIRE_PRESSURE_SELECTION,
  formatTirePressureWeight,
  latestMorningBodyWeight,
  tirePressureWeightToKg,
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

test('converts rider weight units while retaining canonical kilograms', () => {
  assert.equal(formatTirePressureWeight(86.06, 'kg'), '86.06')
  assert.equal(formatTirePressureWeight(86.06, 'lb'), '189.7')
  assert.equal(Number(tirePressureWeightToKg(189.7, 'lb').toFixed(2)), 86.05)
})

test('matches the published SILCA equation for the equipped Cervélo', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
  })

  assert.ok(recommendation)
  assert.equal(recommendation.frontPsi, 76.5)
  assert.equal(recommendation.rearPsi, 83)
  assert.equal(Number(recommendation.bikeKg.toFixed(3)), 9.979)
  assert.equal(Number(recommendation.systemKg.toFixed(3)), 96.039)
  assert.equal(recommendation.measuredWidthMm, 28)
  assert.equal(recommendation.diameterMm, 622)
  assert.equal(recommendation.wheelCompatibilityWarning, false)
})

test('uses the Speedmax system mass with a selected even load distribution', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    bike: 'speedmax',
    balance: '50-50',
  })

  assert.ok(recommendation)
  assert.equal(recommendation.frontPsi, 80)
  assert.equal(recommendation.rearPsi, 80)
  assert.equal(Number(recommendation.bikeKg.toFixed(3)), 11.793)
})

test('uses a customized equipped-bike mass without changing its load distribution', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    bikeMassesLb: { ...DEFAULT_TIRE_PRESSURE_SELECTION.bikeMassesLb, cervelo: 23.5 },
  })

  assert.ok(recommendation)
  assert.equal(recommendation.bikeMassLb, 23.5)
  assert.equal(Number(recommendation.bikeKg.toFixed(3)), 10.659)
  assert.ok(recommendation.frontPsi < recommendation.rearPsi)
})

test('calculates a balanced recommendation for a custom bike', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    bike: 'custom',
    bikeMassesLb: { ...DEFAULT_TIRE_PRESSURE_SELECTION.bikeMassesLb, custom: 31.5 },
    balance: '50-50',
  })

  assert.ok(recommendation)
  assert.equal(recommendation.bike.label, 'custom bike')
  assert.equal(recommendation.bikeMassLb, 31.5)
  assert.equal(recommendation.frontPsi, recommendation.rearPsi)
})

test('applies selectable front and rear load distributions', () => {
  const even = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    balance: '50-50',
  })
  const rearHeavy = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    balance: '40-60',
  })

  assert.ok(even)
  assert.ok(rearHeavy)
  assert.equal(even.frontPsi, even.rearPsi)
  assert.equal(rearHeavy.balance.frontPercent, 40)
  assert.equal(rearHeavy.balance.rearPercent, 60)
  assert.ok(rearHeavy.frontPsi < even.frontPsi)
  assert.ok(rearHeavy.rearPsi > even.rearPsi)
})

test('keeps pressure stable across equal measured casing widths and flags Reserve compatibility', () => {
  const hunt = calculateTirePressure({ ...DEFAULT_TIRE_PRESSURE_SELECTION, riderKg: 86.06 })
  const reserve = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    wheel: 'reserve',
  })

  assert.ok(hunt)
  assert.ok(reserve)
  assert.equal(hunt.wheel.label, 'HUNT 54_58 Aerodynamicist UD')
  assert.equal(hunt.wheel.frontInnerWidthMm, 22)
  assert.equal(hunt.wheel.rearInnerWidthMm, 22)
  assert.equal(reserve.frontPsi, hunt.frontPsi)
  assert.equal(reserve.rearPsi, hunt.rearPsi)
  assert.equal(reserve.wheelCompatibilityWarning, true)
  assert.equal(reserve.wheel.frontInnerWidthMm, 25.4)
  assert.equal(reserve.wheel.rearInnerWidthMm, 24.8)
})

test('uses custom front and rear internal rim widths without inventing casing growth', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    wheel: 'custom',
    customWheel: { frontInnerWidthMm: 21.5, rearInnerWidthMm: 24 },
  })

  assert.ok(recommendation)
  assert.equal(recommendation.wheel.label, 'custom wheelset')
  assert.equal(recommendation.wheel.frontInnerWidthMm, 21.5)
  assert.equal(recommendation.wheel.rearInnerWidthMm, 24)
  assert.equal(recommendation.measuredWidthMm, 28)
  assert.equal(recommendation.wheelCompatibilityWarning, false)
})

test('keeps Pirelli TPU and tubeless setups at the high-performance 1.00 coefficient', () => {
  const tpu = calculateTirePressure({ ...DEFAULT_TIRE_PRESSURE_SELECTION, riderKg: 86.06 })
  const tubeless = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    tire: 'tubeless',
  })

  assert.ok(tpu)
  assert.ok(tubeless)
  assert.equal(tubeless.frontPsi, tpu.frontPsi)
  assert.equal(tubeless.rearPsi, tpu.rearPsi)
  assert.equal(tubeless.tire.pressureCoefficient, 1)
})

test('rejects pressure inputs outside the SILCA system-weight and speed domains', () => {
  assert.equal(calculateTirePressure({ ...DEFAULT_TIRE_PRESSURE_SELECTION, riderKg: 10 }), null)
  assert.equal(
    calculateTirePressure({ ...DEFAULT_TIRE_PRESSURE_SELECTION, riderKg: 86.06, speedMph: 34 }),
    null,
  )
  assert.equal(
    calculateTirePressure({
      ...DEFAULT_TIRE_PRESSURE_SELECTION,
      riderKg: 86.06,
      bikeMassesLb: { ...DEFAULT_TIRE_PRESSURE_SELECTION.bikeMassesLb, cervelo: 9.5 },
    }),
    null,
  )
  assert.equal(
    calculateTirePressure({
      ...DEFAULT_TIRE_PRESSURE_SELECTION,
      riderKg: 86.06,
      wheel: 'custom',
      customWheel: { frontInnerWidthMm: 12.5, rearInnerWidthMm: 23 },
    }),
    null,
  )
})
