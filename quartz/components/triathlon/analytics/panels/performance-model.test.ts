import assert from 'node:assert/strict'
import test from 'node:test'
import { initialPerformanceModel, updatePerformance } from './performance-model'

test('performance reducer owns projection load, hover, leave, and locked selection', () => {
  const bounds = { lastObservedIndex: 89, maximumIndex: 116, maximumLoad: 200 }
  const initial = initialPerformanceModel(75, bounds)
  assert.deepEqual(initial, { futureDailyLoad: 75, activeIndex: 89, lockedIndex: null })

  const hovered = updatePerformance(initial, { type: 'hover', index: 12.6 }, bounds)
  assert.equal(hovered.activeIndex, 13)
  assert.equal(updatePerformance(hovered, { type: 'leave' }, bounds).activeIndex, 89)

  const locked = updatePerformance(hovered, { type: 'toggle-lock', index: 20 }, bounds)
  assert.deepEqual(updatePerformance(locked, { type: 'hover', index: 30 }, bounds), locked)
  assert.deepEqual(updatePerformance(locked, { type: 'toggle-lock', index: 20 }, bounds), initial)

  assert.equal(
    updatePerformance(initial, { type: 'set-load', load: 250 }, bounds).futureDailyLoad,
    200,
  )
})
