import assert from 'node:assert/strict'
import test from 'node:test'
import { initialCalculatorModel, updateCalculator } from './calculator-model'

const input = {
  swimKm: 1.5,
  bikeKm: 40,
  runKm: 10,
  swimPaceSec: 120,
  t1Sec: 180,
  bikeMph: 20,
  t2Sec: 120,
  runPaceSec: 480,
}

test('calculator reducer owns inputs, sources, and projection selection', () => {
  const average = updateCalculator(initialCalculatorModel(input), {
    type: 'select-source',
    source: 'avg',
  })
  assert.equal(average.model.source, 'avg')
  assert.equal(average.model.userEdited, false)

  const edited = updateCalculator(average.model, {
    type: 'sync-input',
    input: { ...input, bikeMph: 21 },
    userEdited: true,
  })
  assert.equal(edited.model.source, 'manual')
  assert.equal(edited.model.input.bikeMph, 21)

  const projected = updateCalculator(edited.model, { type: 'select-source', source: 'projection' })
  assert.equal(projected.model.projection.active, true)
  const zone = updateCalculator(projected.model, { type: 'select-zone', zone: 4 })
  assert.equal(zone.model.projection.zone, 4)
})
