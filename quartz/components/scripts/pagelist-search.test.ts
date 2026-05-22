import assert from 'node:assert'
import test, { describe } from 'node:test'
import { extractTagsFromQuery, extractTextQuery, pageListItemMatchesQuery } from './pagelist-search'

describe('page list search predicates', () => {
  const tsfm = { title: 'toronto school of foundational modeling', tags: ['tsfm'] }

  test('matches plain text without requiring a tag token', () => {
    assert.strictEqual(pageListItemMatchesQuery(tsfm, 'toronto'), true)
    assert.strictEqual(pageListItemMatchesQuery(tsfm, 'montreal'), false)
  })

  test('keeps tag filters active when query contains tags', () => {
    assert.strictEqual(pageListItemMatchesQuery(tsfm, '#tsfm'), true)
    assert.strictEqual(pageListItemMatchesQuery(tsfm, '#ml'), false)
  })

  test('requires text and tag clauses when both are present', () => {
    assert.strictEqual(pageListItemMatchesQuery(tsfm, 'toronto #tsfm'), true)
    assert.strictEqual(pageListItemMatchesQuery(tsfm, 'toronto #ml'), false)
    assert.strictEqual(pageListItemMatchesQuery(tsfm, 'montreal #tsfm'), false)
  })

  test('extracts text and tag clauses independently', () => {
    assert.deepStrictEqual(extractTagsFromQuery('toronto #tsfm #school/fm'), ['tsfm', 'school/fm'])
    assert.strictEqual(extractTextQuery('toronto #tsfm #school/fm'), 'toronto')
  })
})
