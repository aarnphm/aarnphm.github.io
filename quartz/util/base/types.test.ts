import assert from 'node:assert'
import test from 'node:test'
import { parseViews, parseViewSummaries } from './types'

test('parseViews preserves raw filters', () => {
  const views = parseViews([
    { type: 'table', name: 'test', filters: 'status == "done"', order: ['file.name'] },
  ])

  assert.strictEqual(views.length, 1)
  assert.strictEqual(views[0].filters, 'status == "done"')
  assert.deepStrictEqual(views[0].order, ['file.name'])
})

test('parseViews rejects missing type/name', () => {
  assert.throws(() => parseViews([{}]))
})

test('parseViewSummaries resolves builtin and formula refs', () => {
  const summaries = parseViewSummaries(
    { price: 'Average', score: 'avgScore', extra: 'values.length' },
    { avgScore: 'values.mean()' },
  )

  assert.ok(summaries)
  if (!summaries) return
  assert.strictEqual(summaries.columns.price.type, 'builtin')
  assert.strictEqual(summaries.columns.score.type, 'formula')
  assert.strictEqual(summaries.columns.score.formulaRef, 'avgScore')
  assert.strictEqual(summaries.columns.extra.type, 'formula')
})
