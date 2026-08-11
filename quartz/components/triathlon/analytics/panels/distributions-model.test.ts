import assert from 'node:assert/strict'
import test from 'node:test'
import { initialDistributionModel, updateDistributions } from './distributions-model'

test('distribution reducer owns sport, range, custom date, and restored selection', () => {
  const bounds = {
    minimumDate: '2026-01-01',
    maximumDate: '2026-03-31',
    sports: ['swim', 'bike', 'run'] as const,
  }
  const initial = initialDistributionModel(bounds)
  assert.deepEqual(initial, { sport: 'bike', range: '30', startDate: '2026-03-02' })

  const week = updateDistributions(initial, { type: 'select-range', range: '7' }, bounds)
  assert.deepEqual(week, { sport: 'bike', range: '7', startDate: '2026-03-25' })

  const custom = updateDistributions(week, { type: 'select-date', date: '2025-12-01' }, bounds)
  assert.deepEqual(custom, { sport: 'bike', range: 'custom', startDate: '2026-01-01' })

  const restored = updateDistributions(
    custom,
    { type: 'restore', model: { sport: 'run', range: '60', startDate: '2026-02-20' } },
    bounds,
  )
  assert.deepEqual(restored, { sport: 'run', range: '60', startDate: '2026-01-31' })

  assert.deepEqual(updateDistributions(restored, { type: 'clear-date' }, bounds), {
    sport: 'run',
    range: '30',
    startDate: '2026-03-02',
  })
})
