import assert from 'node:assert/strict'
import test from 'node:test'
import {
  initialPaceForecastModel,
  paceForecastBounds,
  paceForecastComparisonDate,
  updatePaceForecast,
} from './pace-forecast-model'

test('pace predictor comparison history follows the analytics window', () => {
  const available = { min: '2026-05-15', max: '2026-09-22' }
  const selected = { min: '2026-07-25', max: '2026-09-22' }
  const bounds = paceForecastBounds(available, selected)
  assert.deepEqual(bounds, selected)
  assert.ok(bounds)
  assert.equal(paceForecastComparisonDate(bounds, 7), '2026-09-15')
  assert.equal(paceForecastComparisonDate(bounds, 30), '2026-08-23')
  assert.equal(paceForecastComparisonDate(bounds, 59), '2026-07-25')
  assert.equal(paceForecastComparisonDate(bounds, 60), null)

  const all = paceForecastBounds(available, available)
  assert.ok(all)
  assert.equal(paceForecastComparisonDate(all, 60), '2026-07-24')
})

test('pace predictor bounds account for independently updated and missing forecast data', () => {
  const selected = { min: '2026-07-25', max: '2026-09-22' }
  assert.deepEqual(paceForecastBounds({ min: '2026-08-01', max: '2026-09-21' }, selected), {
    min: '2026-08-01',
    max: '2026-09-21',
  })
  assert.deepEqual(paceForecastBounds({ min: '2026-05-15', max: '2026-09-23' }, selected), selected)
  assert.equal(paceForecastBounds(null, selected), null)
  assert.equal(paceForecastBounds({ min: '2026-05-15', max: '2026-07-24' }, selected), null)
  assert.equal(paceForecastBounds({ min: '2026-09-23', max: '2026-09-24' }, selected), null)
})

test('pace predictor comparisons count calendar days across month and daylight saving boundaries', () => {
  const bounds = { min: '2026-01-09', max: '2026-03-09' }
  assert.equal(paceForecastComparisonDate(bounds, 7), '2026-03-02')
  assert.equal(paceForecastComparisonDate(bounds, 30), '2026-02-07')
  assert.equal(paceForecastComparisonDate(bounds, 59), '2026-01-09')
  assert.equal(paceForecastComparisonDate(bounds, 60), null)
})

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
