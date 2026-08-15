import assert from 'node:assert/strict'
import test from 'node:test'
import {
  initialDistributionModel,
  telemetryTrend,
  telemetryWeightedAverage,
  updateDistributions,
} from './distributions-model'

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

test('telemetry trend compares the latest two observed activities', () => {
  assert.equal(telemetryTrend([120, null, 138]), 'up')
  assert.equal(telemetryTrend([81, 76]), 'down')
  assert.equal(telemetryTrend([33.4, Number.NaN, 33.4]), 'flat')
  assert.equal(telemetryTrend([null, 2.1]), null)
})

test('telemetry range summaries weight values by their observed duration', () => {
  assert.equal(
    telemetryWeightedAverage([
      { value: 100, observedSeconds: 1_800 },
      { value: 200, observedSeconds: 3_600 },
    ]),
    500 / 3,
  )
  assert.equal(
    telemetryWeightedAverage([
      { value: 32, observedSeconds: 900 },
      { value: null, observedSeconds: 3_600 },
      { value: Number.NaN, observedSeconds: 1_800 },
      { value: 36, observedSeconds: 0 },
    ]),
    32,
  )
  assert.equal(telemetryWeightedAverage([{ value: 120, observedSeconds: 0 }]), null)
})
