import assert from 'node:assert/strict'
import test from 'node:test'
import {
  completeDistributionPaceRanges,
  distributionMetricForSport,
  distributionMetrics,
  initialDistributionModel,
  telemetryTrend,
  telemetryWeightedAverage,
  updateDistributions,
} from './distributions-model'

test('pace ranges complete open outer zones and single missing interior zones', () => {
  assert.deepEqual(
    completeDistributionPaceRanges([
      { fastestSPerKm: 406.3, slowestSPerKm: 1074.6 },
      { fastestSPerKm: 333.3, slowestSPerKm: 381.8 },
      { fastestSPerKm: 308.7, slowestSPerKm: 350.9 },
      null,
      { fastestSPerKm: 268.9, slowestSPerKm: 270.3 },
      null,
    ]),
    [
      { fastestSPerKm: 406.3, slowestSPerKm: null, fillGap: false },
      { fastestSPerKm: 333.3, slowestSPerKm: 381.8, fillGap: false },
      { fastestSPerKm: 308.7, slowestSPerKm: 350.9, fillGap: false },
      { fastestSPerKm: 270.3, slowestSPerKm: 308.7, fillGap: true },
      { fastestSPerKm: 268.9, slowestSPerKm: 270.3, fillGap: false },
      { fastestSPerKm: null, slowestSPerKm: 268.9, fillGap: false },
    ],
  )
})

test('pace range completion does not fabricate boundaries across multiple missing zones', () => {
  assert.deepEqual(
    completeDistributionPaceRanges([
      null,
      { fastestSPerKm: 330, slowestSPerKm: 390 },
      null,
      null,
      { fastestSPerKm: 270, slowestSPerKm: 280 },
      null,
    ]),
    [
      { fastestSPerKm: 390, slowestSPerKm: null, fillGap: false },
      { fastestSPerKm: 330, slowestSPerKm: 390, fillGap: false },
      null,
      null,
      { fastestSPerKm: 270, slowestSPerKm: 280, fillGap: false },
      { fastestSPerKm: null, slowestSPerKm: 270, fillGap: false },
    ],
  )
})

test('distribution reducer owns sport, range, custom date, and restored selection', () => {
  const bounds = {
    minimumDate: '2026-01-01',
    maximumDate: '2026-03-31',
    sports: ['swim', 'bike', 'run'] as const,
  }
  const initial = initialDistributionModel(bounds)
  assert.deepEqual(initial, {
    sport: 'bike',
    metric: 'power',
    range: '30',
    startDate: '2026-03-02',
  })

  const week = updateDistributions(initial, { type: 'select-range', range: '7' }, bounds)
  assert.deepEqual(week, { sport: 'bike', metric: 'power', range: '7', startDate: '2026-03-25' })

  const custom = updateDistributions(week, { type: 'select-date', date: '2025-12-01' }, bounds)
  assert.deepEqual(custom, {
    sport: 'bike',
    metric: 'power',
    range: 'custom',
    startDate: '2026-01-01',
  })

  const restored = updateDistributions(
    custom,
    {
      type: 'restore',
      model: { sport: 'run', metric: 'pace', range: '60', startDate: '2026-02-20' },
    },
    bounds,
  )
  assert.deepEqual(restored, { sport: 'run', metric: 'pace', range: '60', startDate: '2026-01-31' })

  assert.deepEqual(updateDistributions(restored, { type: 'clear-date' }, bounds), {
    sport: 'run',
    metric: 'pace',
    range: '30',
    startDate: '2026-03-02',
  })
})

test('distribution metrics follow each sport and preserve heart rate across sport changes', () => {
  assert.deepEqual(distributionMetrics('bike'), ['heart-rate', 'power'])
  assert.deepEqual(distributionMetrics('run'), ['heart-rate', 'pace'])
  assert.deepEqual(distributionMetrics('swim'), ['heart-rate'])
  assert.equal(distributionMetricForSport('bike'), 'power')
  assert.equal(distributionMetricForSport('run'), 'pace')
  assert.equal(distributionMetricForSport('swim'), 'heart-rate')
  assert.equal(distributionMetricForSport('run', 'heart-rate'), 'heart-rate')
  assert.equal(distributionMetricForSport('run', 'power'), 'pace')
  assert.equal(distributionMetricForSport('swim', 'pace'), 'heart-rate')
})

test('distribution metric selection switches both directions for bike and run', () => {
  const bounds = {
    minimumDate: '2026-01-01',
    maximumDate: '2026-03-31',
    sports: ['swim', 'bike', 'run'] as const,
  }
  const bikePower = initialDistributionModel(bounds)
  const bikeHeartRate = updateDistributions(
    bikePower,
    { type: 'select-metric', metric: 'heart-rate' },
    bounds,
  )
  assert.equal(bikeHeartRate.metric, 'heart-rate')
  assert.equal(
    updateDistributions(bikeHeartRate, { type: 'select-metric', metric: 'power' }, bounds).metric,
    'power',
  )

  const runPace = updateDistributions(bikePower, { type: 'select-sport', sport: 'run' }, bounds)
  assert.equal(runPace.metric, 'pace')
  const runHeartRate = updateDistributions(
    runPace,
    { type: 'select-metric', metric: 'heart-rate' },
    bounds,
  )
  assert.equal(runHeartRate.metric, 'heart-rate')
  assert.equal(
    updateDistributions(runHeartRate, { type: 'select-metric', metric: 'pace' }, bounds).metric,
    'pace',
  )
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
