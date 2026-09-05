import assert from 'node:assert/strict'
import test from 'node:test'
import {
  calculateTirePressure,
  DEFAULT_TIRE_PRESSURE_SELECTION,
  formatTirePressureWeight,
  KG_PER_LB,
  latestMorningBodyWeight,
  TIRE_PRESSURE_WHEELS,
  type TirePressureSelection,
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
  assert.equal(recommendation.frontPsi, 64)
  assert.equal(recommendation.rearPsi, 81)
  assert.equal(Number(recommendation.bikeKg.toFixed(3)), 11.884)
  assert.equal(Number(recommendation.systemKg.toFixed(3)), 97.944)
  assert.equal(recommendation.frontMeasuredWidthMm, 32)
  assert.equal(recommendation.rearMeasuredWidthMm, 28)
  assert.equal(recommendation.diameterMm, 622)
  assert.equal(recommendation.wheel.id, 'reserve-40-44')
  assert.equal(recommendation.wheel.label, 'Reserve 40|44 Road')
  assert.equal(recommendation.wheel.frontInnerWidthMm, 25.4)
  assert.equal(recommendation.wheel.rearInnerWidthMm, 25)
  assert.equal(recommendation.wheelCompatibilityWarning, false)
})

test('matches controlled outputs from the live SILCA calculator', () => {
  const systemKg = 100
  const riderKg = systemKg - DEFAULT_TIRE_PRESSURE_SELECTION.bikeMassesLb.custom * KG_PER_LB
  const selection: TirePressureSelection = {
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg,
    bike: 'custom',
    balance: '48-52',
  }
  const width32 = calculateTirePressure({
    ...selection,
    measuredTire: { frontWidthMm: 32, rearWidthMm: 32 },
  })
  const width28 = calculateTirePressure({
    ...selection,
    measuredTire: { frontWidthMm: 28, rearWidthMm: 28 },
  })

  assert.ok(width32)
  assert.ok(width28)
  assert.equal(width32.systemKg, systemKg)
  assert.equal(width32.frontPsi, 64)
  assert.equal(width32.rearPsi, 65.5)
  assert.equal(width28.frontPsi, 79)
  assert.equal(width28.rearPsi, 81)
})

test('uses the Speedmax system mass with a selected even load distribution', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    bike: 'speedmax',
    balance: '50-50',
  })

  assert.ok(recommendation)
  assert.equal(recommendation.frontPsi, 64.5)
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

test('uses axle-specific widths for a custom bike with an even load distribution', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    bike: 'custom',
    bikeMassesLb: { ...DEFAULT_TIRE_PRESSURE_SELECTION.bikeMassesLb, custom: 31.5 },
    balance: '50-50',
  })

  assert.ok(recommendation)
  assert.equal(recommendation.bike.label, 'Custom')
  assert.equal(recommendation.bikeMassLb, 31.5)
  assert.ok(recommendation.frontPsi < recommendation.rearPsi)
})

test('applies SILCA front and rear load-distribution coefficients', () => {
  const road = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    balance: '48-52',
  })
  const rearHeavy = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    balance: '46.5-53.5',
  })

  assert.ok(road)
  assert.ok(rearHeavy)
  assert.equal(road.balance.frontPressureCoefficient, 0.985)
  assert.equal(road.balance.rearPressureCoefficient, 1.01)
  assert.equal(rearHeavy.balance.frontPercent, 46.5)
  assert.equal(rearHeavy.balance.rearPercent, 53.5)
  assert.equal(rearHeavy.balance.frontPressureCoefficient, 0.97)
  assert.equal(rearHeavy.balance.rearPressureCoefficient, 1.03)
  assert.ok(rearHeavy.frontPsi < road.frontPsi)
  assert.ok(rearHeavy.rearPsi > road.rearPsi)
})

test('keeps axle pressures stable across the equipped wheelset rotation', () => {
  const road4044 = calculateTirePressure({ ...DEFAULT_TIRE_PRESSURE_SELECTION, riderKg: 86.06 })
  const hunt = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    wheel: 'hunt-54-58',
  })
  const ta4249 = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    wheel: 'reserve-42-49',
  })

  assert.ok(road4044)
  assert.ok(hunt)
  assert.ok(ta4249)
  assert.deepEqual(
    TIRE_PRESSURE_WHEELS.map(wheel => wheel.id),
    ['reserve-40-44', 'reserve-42-49', 'hunt-54-58', 'custom'],
  )
  assert.equal(road4044.wheel.label, 'Reserve 40|44 Road')
  assert.equal(road4044.wheel.frontInnerWidthMm, 25.4)
  assert.equal(road4044.wheel.rearInnerWidthMm, 25)
  assert.equal(road4044.wheelCompatibilityWarning, false)
  assert.equal(hunt.wheel.label, 'HUNT 54_58 Aerodynamicist UD')
  assert.equal(hunt.wheel.frontInnerWidthMm, 22)
  assert.equal(hunt.wheel.rearInnerWidthMm, 22)
  assert.equal(ta4249.frontPsi, hunt.frontPsi)
  assert.equal(ta4249.rearPsi, hunt.rearPsi)
  assert.equal(ta4249.wheelCompatibilityWarning, true)
  assert.equal(ta4249.wheel.frontInnerWidthMm, 25.4)
  assert.equal(ta4249.wheel.rearInnerWidthMm, 24.8)
})

test('uses custom front and rear internal rim widths without inventing casing growth', () => {
  const recommendation = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    wheel: 'custom',
    customWheel: { frontInnerWidthMm: 21.5, rearInnerWidthMm: 24 },
  })

  assert.ok(recommendation)
  assert.equal(recommendation.wheel.label, 'Custom Wheelset')
  assert.equal(recommendation.wheel.frontInnerWidthMm, 21.5)
  assert.equal(recommendation.wheel.rearInnerWidthMm, 24)
  assert.equal(recommendation.frontMeasuredWidthMm, 32)
  assert.equal(recommendation.rearMeasuredWidthMm, 28)
  assert.equal(recommendation.wheelCompatibilityWarning, false)
})

test('uses measured front and rear tire widths as independent calculation inputs', () => {
  const staggered = calculateTirePressure({ ...DEFAULT_TIRE_PRESSURE_SELECTION, riderKg: 86.06 })
  const equalWidths = calculateTirePressure({
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 86.06,
    measuredTire: { frontWidthMm: 28, rearWidthMm: 28 },
  })

  assert.ok(staggered)
  assert.ok(equalWidths)
  assert.ok(equalWidths.frontPsi > staggered.frontPsi)
  assert.equal(equalWidths.rearPsi, staggered.rearPsi)
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
  assert.equal(
    calculateTirePressure({
      ...DEFAULT_TIRE_PRESSURE_SELECTION,
      riderKg: 86.06,
      measuredTire: { frontWidthMm: 19, rearWidthMm: 28 },
    }),
    null,
  )
})
