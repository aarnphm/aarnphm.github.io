import assert from 'node:assert/strict'
import test from 'node:test'
import { initialPaceForecastModel, updatePaceForecast } from './pace-forecast-model'

test('pace forecast reducer owns sport and comparison selection generations', () => {
  const sport = updatePaceForecast(initialPaceForecastModel(), {
    type: 'select-sport',
    sport: 'bike',
  })
  assert.equal(sport.model.sport, 'bike')
  assert.equal(sport.model.generation, 1)

  const date = updatePaceForecast(sport.model, { type: 'select-date', date: '2026-07-01' })
  assert.equal(date.model.comparison, 'custom')
  assert.equal(date.model.comparisonDate, '2026-07-01')
  assert.deepEqual(date.effects, [{ type: 'render', generation: 2 }])

  const cleared = updatePaceForecast(date.model, { type: 'clear-date' })
  assert.equal(cleared.model.comparison, '7')
  assert.equal(cleared.model.comparisonDate, undefined)
})
