import assert from 'node:assert/strict'
import test from 'node:test'
import { initialGearRatioModel, updateGearRatio } from './gear-ratio-model'

test('gear ratio reducer owns cassette, layout, and chainring transitions', () => {
  const initial = initialGearRatioModel('11-34', '54-40', 54, 40)
  const compact = updateGearRatio(initial, {
    type: 'set-cassette',
    cassetteId: '10-44',
    maximumChainrings: 1,
  })
  assert.equal(compact.model.cassetteId, '10-44')
  assert.equal(compact.model.layout, 1)
  assert.deepEqual(compact.effects, [{ type: 'render' }])

  const selected = updateGearRatio(compact.model, {
    type: 'set-chainring-preset',
    presetId: '53-39',
    chainrings: [53, 39],
  })
  assert.equal(selected.model.layout, 2)
  assert.equal(selected.model.chainringPresetId, '53-39')
  assert.deepEqual(selected.model.chainrings, [53, 39])

  const changed = updateGearRatio(selected.model, { type: 'set-chainring', index: 0, value: 54 })
  assert.equal(changed.model.chainringPresetId, null)
  assert.deepEqual(changed.model.chainrings, [54, 39])
})
