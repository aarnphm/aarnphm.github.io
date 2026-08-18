import assert from 'node:assert/strict'
import test from 'node:test'
import type { TirePressureSelection } from '../../../util/triathlon-tire-pressure'
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
  assert.equal(nextTirePressurePaletteStep('bike'), 'bikeMass')
  assert.equal(nextTirePressurePaletteStep('bikeMass'), 'balance')
  assert.equal(nextTirePressurePaletteStep('balance'), 'wheel')
  assert.equal(nextTirePressurePaletteStep('wheel'), 'tire')
  assert.equal(nextTirePressurePaletteStep('tire'), 'surface')
  assert.equal(nextTirePressurePaletteStep('surface'), 'speed')
  assert.equal(nextTirePressurePaletteStep('speed'), 'result')
})

test('tire pressure palette backtracks without skipping selection state', () => {
  assert.equal(previousTirePressurePaletteStep('result'), 'speed')
  assert.equal(previousTirePressurePaletteStep('speed'), 'surface')
  assert.equal(previousTirePressurePaletteStep('surface'), 'tire')
  assert.equal(previousTirePressurePaletteStep('tire'), 'wheel')
  assert.equal(previousTirePressurePaletteStep('wheel'), 'balance')
  assert.equal(previousTirePressurePaletteStep('balance'), 'bikeMass')
  assert.equal(previousTirePressurePaletteStep('bikeMass'), 'bike')
  assert.equal(previousTirePressurePaletteStep('bike'), 'commands')
})

test('tire pressure palette highlights the persisted choice at every step', () => {
  const selection: TirePressureSelection = {
    bike: 'speedmax',
    bikeMassesLb: { cervelo: 22.4, speedmax: 26.8, custom: 19.5 },
    balance: '45-55',
    wheel: 'reserve',
    tire: 'tubeless',
    surface: 'worn-pavement',
    speedMph: 23,
  }

  assert.equal(tirePressurePaletteSelectionIndex('bike', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('bikeMass', selection), 0)
  assert.equal(tirePressurePaletteSelectionIndex('balance', selection), 2)
  assert.equal(tirePressurePaletteSelectionIndex('wheel', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('tire', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('surface', selection), 1)
  assert.equal(tirePressurePaletteSelectionIndex('speed', selection), 2)
  assert.equal(tirePressurePaletteSelectionIndex('result', selection), 0)
})
