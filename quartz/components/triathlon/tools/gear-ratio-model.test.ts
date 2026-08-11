import assert from 'node:assert/strict'
import test from 'node:test'
import { initialGearRatioModel, updateGearRatio } from './gear-ratio-model'

test('gear ratio reducer owns cassette, layout, and chainring transitions', () => {
  const initial = initialGearRatioModel('11-34', 52, 36)
  const compact = updateGearRatio(initial, {
    type: 'set-cassette',
    cassetteId: '10-44',
    maximumChainrings: 1,
  })
  assert.equal(compact.model.cassetteId, '10-44')
  assert.equal(compact.model.layout, 1)
  assert.deepEqual(compact.effects, [{ type: 'render' }])

  const changed = updateGearRatio(compact.model, { type: 'set-chainring', index: 0, value: 54 })
  assert.deepEqual(changed.model.chainrings, [54, 36])
})
