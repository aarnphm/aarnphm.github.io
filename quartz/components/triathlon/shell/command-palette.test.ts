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
  assert.equal(nextTirePressurePaletteStep('wheel'), 'widthMode')
  assert.equal(nextTirePressurePaletteStep('widthMode'), 'measuredTireFront')
  assert.equal(nextTirePressurePaletteStep('measuredTireFront'), 'measuredTireRear')
  assert.equal(nextTirePressurePaletteStep('measuredTireRear'), 'tire')
  assert.equal(nextTirePressurePaletteStep('tire'), 'setup')
  assert.equal(nextTirePressurePaletteStep('setup'), 'surface')
  assert.equal(nextTirePressurePaletteStep('surface'), 'speed')
  assert.equal(nextTirePressurePaletteStep('speed'), 'result')

  const customWheel: TirePressureSelection = { ...DEFAULT_TIRE_PRESSURE_SELECTION, wheel: 'custom' }
  assert.equal(nextTirePressurePaletteStep('wheel', customWheel), 'customWheelFront')
  assert.equal(nextTirePressurePaletteStep('customWheelFront', customWheel), 'customWheelRear')
  assert.equal(nextTirePressurePaletteStep('customWheelRear', customWheel), 'widthMode')
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

test('shared width flow skips the rear input and keeps result edits bounded', () => {
  const selection: TirePressureSelection = {
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    widthMode: 'shared',
  }
  assert.equal(nextTirePressurePaletteStep('widthMode', selection), 'measuredTireFront')
  assert.equal(nextTirePressurePaletteStep('measuredTireFront', selection), 'tire')
  assert.equal(
    nextTirePressurePaletteStep('measuredTireFront', selection, 'measuredTireRear'),
    'result',
  )
  assert.equal(previousTirePressurePaletteStep('tire', selection), 'measuredTireFront')
  assert.equal(
    previousTirePressurePaletteStep('measuredTireFront', selection, 'widthMode'),
    'widthMode',
  )
  assert.equal(previousTirePressurePaletteStep('widthMode', selection, 'widthMode'), 'result')
  assert.equal(tirePressurePaletteSelectionIndex('widthMode', selection), 0)
  assert.equal(tirePressurePaletteSelectionIndex('widthMode', DEFAULT_TIRE_PRESSURE_SELECTION), 1)
})

test('tire pressure palette highlights tire models and their compatible setups independently', () => {
  for (const [tire, setup, tireIndex, setupIndex] of [
    ['race-sl-r', 'tpu', 0, 0],
    ['race-tlr-sl-r', 'tpu', 1, 0],
    ['race-tlr-sl-r', 'tubeless', 1, 1],
  ] as const) {
    const selection: TirePressureSelection = { ...DEFAULT_TIRE_PRESSURE_SELECTION, tire, setup }

    assert.equal(tirePressurePaletteSelectionIndex('tire', selection), tireIndex)
    assert.equal(tirePressurePaletteSelectionIndex('setup', selection), setupIndex)
    for (const step of ['tire', 'setup'] as const) {
      assert.equal(nextTirePressurePaletteStep(step, selection, step), 'result')
      assert.equal(previousTirePressurePaletteStep(step, selection, step), 'result')
    }
  }
})

test('tire pressure palette backtracks without skipping selection state', () => {
  assert.equal(previousTirePressurePaletteStep('result'), 'commands')
  assert.equal(previousTirePressurePaletteStep('speed'), 'surface')
  assert.equal(previousTirePressurePaletteStep('surface'), 'setup')
  assert.equal(previousTirePressurePaletteStep('setup'), 'tire')
  assert.equal(previousTirePressurePaletteStep('tire'), 'measuredTireRear')
  assert.equal(previousTirePressurePaletteStep('measuredTireRear'), 'measuredTireFront')
  assert.equal(previousTirePressurePaletteStep('measuredTireFront'), 'widthMode')
  assert.equal(previousTirePressurePaletteStep('widthMode'), 'wheel')
  assert.equal(previousTirePressurePaletteStep('wheel'), 'balance')
  assert.equal(previousTirePressurePaletteStep('balance'), 'bikeMass')
  assert.equal(previousTirePressurePaletteStep('bikeMass'), 'bike')
  assert.equal(previousTirePressurePaletteStep('bike'), 'riderMass')
  assert.equal(previousTirePressurePaletteStep('riderMass'), 'weightUnit')
  assert.equal(previousTirePressurePaletteStep('weightUnit'), 'commands')

  const customWheel: TirePressureSelection = { ...DEFAULT_TIRE_PRESSURE_SELECTION, wheel: 'custom' }
  assert.equal(previousTirePressurePaletteStep('widthMode', customWheel), 'customWheelRear')
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
    bikeMassesLb: { cervelo: 22.4, speedmax: 26.8, aeroad: 15.7, custom: 19.5 },
    balance: '47-53',
    wheel: 'reserve-42-49',
    customWheel: { frontInnerWidthMm: 21.5, rearInnerWidthMm: 24 },
    widthMode: 'separate',
    measuredTire: { frontWidthMm: 32, rearWidthMm: 28 },
    tire: 'race-tlr-sl-r',
    setup: 'tubeless',
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
  assert.equal(tirePressurePaletteSelectionIndex('setup', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('surface', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('speed', selection), 2)
  assert.equal(tirePressurePaletteSelectionIndex('result', selection), 0)
})
