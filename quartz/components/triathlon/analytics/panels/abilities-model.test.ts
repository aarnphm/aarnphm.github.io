import assert from 'node:assert/strict'
import test from 'node:test'
import { initialAbilitiesModel, updateAbilities } from './abilities-model'

test('abilities reducer owns sport, average, and restored selection', () => {
  const available = ['swim', 'bike', 'run'] as const
  const initial = initialAbilitiesModel(available)
  assert.deepEqual(initial, { average: false, sports: ['bike'] })

  const multi = updateAbilities(initial, { type: 'toggle-sport', sport: 'run' }, available)
  assert.deepEqual(multi, { average: false, sports: ['bike', 'run'] })

  const average = updateAbilities(multi, { type: 'toggle-average' }, available)
  assert.deepEqual(average, { average: true, sports: ['bike', 'run'] })

  const single = updateAbilities(average, { type: 'toggle-sport', sport: 'swim' }, available)
  assert.deepEqual(single, { average: false, sports: ['swim'] })

  const restored = updateAbilities(
    single,
    { type: 'restore', model: { average: true, sports: ['run', 'run'] } },
    available,
  )
  assert.deepEqual(restored, { average: true, sports: ['run'] })
})
