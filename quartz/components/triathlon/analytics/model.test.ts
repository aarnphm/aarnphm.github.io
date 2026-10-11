import assert from 'node:assert/strict'
import test from 'node:test'
import { initialAnalyticsModel, updateAnalytics } from './model'
import { initialAbilitiesModel, updateAbilities } from './panels/abilities-model'

test('analytics range changes rerender from the loaded artifact without refetching', () => {
  const initial = initialAnalyticsModel()
  assert.equal(initial.range, '60d')
  const loading = updateAnalytics(initial, { type: 'load' })
  const full = updateAnalytics(loading.model, { type: 'set-range', range: 'all' })
  assert.equal(full.model.range, 'all')
  assert.deepEqual(full.effects, [])
  const loaded = updateAnalytics(full.model, { type: 'loaded', request: loading.model.request })
  assert.equal(loaded.model.range, 'all')
  assert.deepEqual(loaded.effects, [{ type: 'render-panels' }])

  const search = updateAnalytics(loaded.model, { type: 'query', value: 'filter:run' })
  const recent = updateAnalytics(search.model, { type: 'set-range', range: '60d' })
  assert.equal(recent.model.range, '60d')
  assert.equal(recent.model.request, loaded.model.request)
  assert.equal(recent.model.query, 'filter:run')
  assert.deepEqual(recent.effects, [{ type: 'render-panels' }])
  const unchanged = updateAnalytics(recent.model, { type: 'set-range', range: '60d' })
  assert.equal(unchanged.model, recent.model)
  assert.deepEqual(unchanged.effects, [])
})

test('analytics keeps its selected range across closing and resetting the panel', () => {
  const initial = initialAnalyticsModel('all')
  const closed = updateAnalytics(initial, { type: 'close' })
  const reset = updateAnalytics(closed.model, { type: 'reset' })
  assert.equal(reset.model.range, 'all')
})

test('analytics reducer ignores stale loads and renders the current artifact', () => {
  const first = updateAnalytics(initialAnalyticsModel(), { type: 'load' })
  const second = updateAnalytics(first.model, { type: 'load' })
  const stale = updateAnalytics(second.model, { type: 'loaded', request: 1 })
  assert.equal(stale.model.status, 'loading')
  assert.deepEqual(stale.effects, [])

  const ready = updateAnalytics(stale.model, { type: 'loaded', request: 2 })
  assert.equal(ready.model.status, 'ready')
  assert.deepEqual(ready.effects, [{ type: 'render-panels' }])
})

test('ability selection leaves averaging mode and deduplicates restored sports', () => {
  const available = ['swim', 'bike', 'run'] as const
  const initial = initialAbilitiesModel(available)
  const selected = updateAbilities(
    { ...initial, average: true },
    { type: 'toggle-sport', sport: 'swim' },
    available,
  )
  assert.deepEqual(selected, { average: false, sports: ['swim'] })
  const restored = updateAbilities(
    selected,
    { type: 'restore', model: { average: true, sports: ['run', 'run'] } },
    available,
  )
  assert.deepEqual(restored, { average: true, sports: ['run'] })
})
