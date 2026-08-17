import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_TRIATHLON_PRESENTATION,
  distanceSystemFromStoredUnit,
} from './triathlon-presentation'

test('defaults distance presentation to imperial', () => {
  assert.equal(DEFAULT_TRIATHLON_PRESENTATION.distance, 'imperial')
  assert.equal(distanceSystemFromStoredUnit(null), 'imperial')
  assert.equal(distanceSystemFromStoredUnit('mi'), 'imperial')
})

test('preserves an explicit stored metric preference', () => {
  assert.equal(distanceSystemFromStoredUnit('km'), 'metric')
})
