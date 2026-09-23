import assert from 'node:assert/strict'
import test from 'node:test'
import { initialAnalyticsModel, updateAnalytics } from './model'

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

test('analytics reducer keeps search, detail, and comparison transitions explicit', () => {
  const search = updateAnalytics(initialAnalyticsModel(), { type: 'query', value: 'sport:run' })
  assert.equal(search.model.mode, 'search')
  assert.equal(search.model.selectedResult, 0)
  assert.deepEqual(search.effects, [{ type: 'render-search' }])

  const detail = updateAnalytics(search.model, { type: 'show-activity', id: '42' })
  assert.equal(detail.model.mode, 'detail')
  assert.equal(detail.model.selectedActivityId, '42')

  const first = updateAnalytics(detail.model, { type: 'toggle-comparison-activity', id: '42' })
  const second = updateAnalytics(first.model, { type: 'toggle-comparison-activity', id: '84' })
  assert.equal(second.model.mode, 'compare')
  assert.deepEqual(second.model.comparisonActivityIds, ['42', '84'])
  assert.deepEqual(second.effects, [{ type: 'render-search' }])

  const filtered = updateAnalytics(second.model, { type: 'query', value: 'filter:bike' })
  assert.equal(filtered.model.mode, 'compare')
  assert.deepEqual(filtered.model.comparisonActivityIds, ['42', '84'])
  assert.deepEqual(filtered.effects, [{ type: 'render-search' }])

  const submitted = updateAnalytics(filtered.model, { type: 'submit-comparison' })
  assert.deepEqual(submitted.effects, [{ type: 'render-comparison', ids: ['42', '84'] }])

  const closed = updateAnalytics(submitted.model, { type: 'close' })
  assert.equal(closed.model.mode, 'main')
  assert.deepEqual(closed.model.comparisonActivityIds, [])
  assert.deepEqual(closed.effects, [{ type: 'restore-focus' }])
})
