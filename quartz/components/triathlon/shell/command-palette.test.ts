import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_TIRE_PRESSURE_SELECTION,
  type TirePressureSelection,
} from '../../../util/triathlon-tire-pressure'
import {
  nextMapMetricShortcutIndex,
  nextTirePressurePaletteStep,
  previousTirePressurePaletteStep,
  tirePressurePaletteSelectionIndex,
} from './command-palette'

test('map metric shortcuts select the next matching tab and wrap duplicate initials', () => {
  const shortcuts = ['w', 'h', 'c', 'r', 's', 'e', 'r', 's', 'e', 't']

  assert.equal(nextMapMetricShortcutIndex(shortcuts, 0, 'R'), 3)
  assert.equal(nextMapMetricShortcutIndex(shortcuts, 3, 'r'), 6)
  assert.equal(nextMapMetricShortcutIndex(shortcuts, 6, 'R'), 3)
  assert.equal(nextMapMetricShortcutIndex(shortcuts, 4, 's'), 7)
  assert.equal(nextMapMetricShortcutIndex(shortcuts, 8, 'e'), 5)
  assert.equal(nextMapMetricShortcutIndex(shortcuts, 9, 'T'), 9)
})

test('map metric shortcuts reject non-character keys and missing initials', () => {
  const shortcuts = ['w', 'h', 't']

  assert.equal(nextMapMetricShortcutIndex(shortcuts, 0, 'ArrowRight'), -1)
  assert.equal(nextMapMetricShortcutIndex(shortcuts, 0, 'x'), -1)
  assert.equal(nextMapMetricShortcutIndex([], -1, 'w'), -1)
})

test('tire pressure palette advances through every physical selection', () => {
  assert.equal(nextTirePressurePaletteStep('weightUnit'), 'riderMass')
  assert.equal(nextTirePressurePaletteStep('riderMass'), 'bike')
  assert.equal(nextTirePressurePaletteStep('bike'), 'bikeMass')
  assert.equal(nextTirePressurePaletteStep('bikeMass'), 'balance')
  assert.equal(nextTirePressurePaletteStep('balance'), 'wheel')
  assert.equal(nextTirePressurePaletteStep('wheel'), 'measuredTireFront')
  assert.equal(nextTirePressurePaletteStep('measuredTireFront'), 'measuredTireRear')
  assert.equal(nextTirePressurePaletteStep('measuredTireRear'), 'tire')
  assert.equal(nextTirePressurePaletteStep('tire'), 'surface')
  assert.equal(nextTirePressurePaletteStep('surface'), 'speed')
  assert.equal(nextTirePressurePaletteStep('speed'), 'result')

  const customWheel: TirePressureSelection = { ...DEFAULT_TIRE_PRESSURE_SELECTION, wheel: 'custom' }
  assert.equal(nextTirePressurePaletteStep('wheel', customWheel), 'customWheelFront')
  assert.equal(nextTirePressurePaletteStep('customWheelFront', customWheel), 'customWheelRear')
  assert.equal(nextTirePressurePaletteStep('customWheelRear', customWheel), 'measuredTireFront')
})

test('tire pressure palette returns to the result after editing one configuration row', () => {
  assert.equal(
    nextTirePressurePaletteStep('bikeMass', DEFAULT_TIRE_PRESSURE_SELECTION, 'bikeMass'),
    'result',
  )
  assert.equal(
    nextTirePressurePaletteStep(
      'measuredTireFront',
      DEFAULT_TIRE_PRESSURE_SELECTION,
      'measuredTireRear',
    ),
    'measuredTireRear',
  )
  assert.equal(
    nextTirePressurePaletteStep(
      'measuredTireRear',
      DEFAULT_TIRE_PRESSURE_SELECTION,
      'measuredTireRear',
    ),
    'result',
  )
})

test('tire pressure palette backtracks without skipping selection state', () => {
  assert.equal(previousTirePressurePaletteStep('result'), 'commands')
  assert.equal(previousTirePressurePaletteStep('speed'), 'surface')
  assert.equal(previousTirePressurePaletteStep('surface'), 'tire')
  assert.equal(previousTirePressurePaletteStep('tire'), 'measuredTireRear')
  assert.equal(previousTirePressurePaletteStep('measuredTireRear'), 'measuredTireFront')
  assert.equal(previousTirePressurePaletteStep('measuredTireFront'), 'wheel')
  assert.equal(previousTirePressurePaletteStep('wheel'), 'balance')
  assert.equal(previousTirePressurePaletteStep('balance'), 'bikeMass')
  assert.equal(previousTirePressurePaletteStep('bikeMass'), 'bike')
  assert.equal(previousTirePressurePaletteStep('bike'), 'riderMass')
  assert.equal(previousTirePressurePaletteStep('riderMass'), 'weightUnit')
  assert.equal(previousTirePressurePaletteStep('weightUnit'), 'commands')

  const customWheel: TirePressureSelection = { ...DEFAULT_TIRE_PRESSURE_SELECTION, wheel: 'custom' }
  assert.equal(previousTirePressurePaletteStep('measuredTireFront', customWheel), 'customWheelRear')
  assert.equal(previousTirePressurePaletteStep('customWheelRear', customWheel), 'customWheelFront')
  assert.equal(previousTirePressurePaletteStep('customWheelFront', customWheel), 'wheel')
})

test('tire pressure palette backtracks within one configuration row', () => {
  assert.equal(
    previousTirePressurePaletteStep(
      'measuredTireRear',
      DEFAULT_TIRE_PRESSURE_SELECTION,
      'measuredTireFront',
    ),
    'measuredTireFront',
  )
  assert.equal(
    previousTirePressurePaletteStep(
      'measuredTireFront',
      DEFAULT_TIRE_PRESSURE_SELECTION,
      'measuredTireFront',
    ),
    'result',
  )
})

test('tire pressure palette highlights the persisted choice at every step', () => {
  const selection: TirePressureSelection = {
    riderKg: 86.2,
    weightUnit: 'lb',
    bike: 'speedmax',
    bikeMassesLb: { cervelo: 22.4, speedmax: 26.8, custom: 19.5 },
    balance: '47-53',
    wheel: 'reserve',
    customWheel: { frontInnerWidthMm: 21.5, rearInnerWidthMm: 24 },
    measuredTire: { frontWidthMm: 32, rearWidthMm: 28 },
    tire: 'tubeless',
    surface: 'worn-pavement',
    speedMph: 23,
  }

  assert.equal(tirePressurePaletteSelectionIndex('weightUnit', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('riderMass', selection), 0)
  assert.equal(tirePressurePaletteSelectionIndex('bike', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('bikeMass', selection), 0)
  assert.equal(tirePressurePaletteSelectionIndex('balance', selection), 2)
  assert.equal(tirePressurePaletteSelectionIndex('wheel', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('measuredTireFront', selection), 0)
  assert.equal(tirePressurePaletteSelectionIndex('measuredTireRear', selection), 0)
  assert.equal(tirePressurePaletteSelectionIndex('tire', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('surface', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('speed', selection), 2)
  assert.equal(tirePressurePaletteSelectionIndex('result', selection), 0)
})
