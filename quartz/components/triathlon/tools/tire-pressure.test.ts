import assert from 'node:assert/strict'
import test, { type TestContext } from 'node:test'
import {
  DEFAULT_TIRE_PRESSURE_SELECTION,
  type TirePressureSelection,
} from '../../../util/triathlon-tire-pressure'
import { readTirePressureSelection, storeTirePressureSelection } from './tire-pressure'

const mockLocalStorage = (t: TestContext): void => {
  const values = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return values.size
    },
    clear: () => values.clear(),
    getItem: key => values.get(key) ?? null,
    key: index => Array.from(values.keys())[index] ?? null,
    removeItem: key => values.delete(key),
    setItem: (key, value) => values.set(key, value),
  }
  t.mock.getter(globalThis, 'localStorage', () => storage)
}

for (const [tire, setup] of [
  ['race-sl-r', 'tpu'],
  ['race-tlr-sl-r', 'tpu'],
  ['race-tlr-sl-r', 'tubeless'],
] as const) {
  test(`tire pressure restores the saved ${tire} / ${setup} selection`, t => {
    mockLocalStorage(t)
    const selection: TirePressureSelection = {
      ...DEFAULT_TIRE_PRESSURE_SELECTION,
      riderKg: 72,
      tire,
      setup,
    }

    storeTirePressureSelection(selection, '2026-09-07')

    assert.equal(localStorage.getItem('triathlon-tire-pressure-tire'), tire)
    assert.equal(localStorage.getItem('triathlon-tire-pressure-setup'), setup)
    assert.deepEqual(readTirePressureSelection(null, '2026-09-07'), selection)
  })
}

test('an invalid saved tire model falls back without discarding the other selections', t => {
  mockLocalStorage(t)
  const selection: TirePressureSelection = {
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    riderKg: 72,
    tire: 'race-tlr-sl-r',
    setup: 'tubeless',
    wheel: 'custom',
  }
  storeTirePressureSelection(selection)
  localStorage.setItem('triathlon-tire-pressure-tire', 'unknown-setup')

  assert.deepEqual(readTirePressureSelection(), {
    ...selection,
    tire: DEFAULT_TIRE_PRESSURE_SELECTION.tire,
    setup: DEFAULT_TIRE_PRESSURE_SELECTION.setup,
  })
})

for (const [legacy, tire, setup] of [
  ['tpu', 'race-sl-r', 'tpu'],
  ['tubeless', 'race-tlr-sl-r', 'tubeless'],
] as const) {
  test(`migrates the previous ${legacy} profile to separate tire and setup choices`, t => {
    mockLocalStorage(t)
    localStorage.setItem('triathlon-tire-pressure-tire', legacy)
    localStorage.setItem('triathlon-tire-pressure-measured-front', '30')
    const selection = readTirePressureSelection(72)
    assert.equal(selection.tire, tire)
    assert.equal(selection.setup, setup)
    assert.equal(selection.measuredTire.frontWidthMm, 30)
    storeTirePressureSelection(selection)
    assert.equal(localStorage.getItem('triathlon-tire-pressure-tire'), tire)
    assert.equal(localStorage.getItem('triathlon-tire-pressure-setup'), setup)
    assert.deepEqual(readTirePressureSelection(), selection)
  })
}

test('normalizes a stale unsupported setup while preserving the selected tire model', t => {
  mockLocalStorage(t)
  storeTirePressureSelection(DEFAULT_TIRE_PRESSURE_SELECTION)
  localStorage.setItem('triathlon-tire-pressure-setup', 'tubeless')
  assert.deepEqual(readTirePressureSelection(), DEFAULT_TIRE_PRESSURE_SELECTION)
})

test('an unknown setup falls back to TPU without losing the TLR tire choice', t => {
  mockLocalStorage(t)
  const selection: TirePressureSelection = {
    ...DEFAULT_TIRE_PRESSURE_SELECTION,
    tire: 'race-tlr-sl-r',
  }
  storeTirePressureSelection(selection)
  localStorage.setItem('triathlon-tire-pressure-setup', 'unknown')
  assert.deepEqual(readTirePressureSelection(), selection)
})
