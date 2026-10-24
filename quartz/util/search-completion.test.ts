import assert from 'node:assert/strict'
import { test } from 'node:test'
import { parseCodeQuery } from './code-query'
import { searchCompletions } from './search-completion'
import { parseSearchQuery } from './search-scope'

// Failure cases: unsupported filters, aliases that cannot be parsed, replacing
// surrounding query text, and interpreting quoted source or AST patterns as operators.
test('filter completions use supported scopes and language aliases', () => {
  const choices = searchCompletions('filter:', 7)
  for (const value of ['notes', 'images', 'pdf', 'code', 'all', 'python', 'typescript', 'tsx'])
    assert.ok(
      choices.some(choice => choice.value === value),
      value,
    )
  for (const choice of choices) {
    const query = choice.insert.trim()
    assert.ok(parseCodeQuery(query) || parseSearchQuery(query).explicit, query)
  }
  assert.equal(searchCompletions('filter:py', 9)[0]?.value, 'python')
  assert.equal(searchCompletions('filter:js', 9)[0]?.value, 'javascript')
  assert.equal(searchCompletions('filter:python ', 14).length, 0)
})

test('completion replaces only the active operator, including spaced filters', () => {
  const raw = 'jax.jit filter: py trailing'
  const choice = searchCompletions(raw, 18)[0]
  assert.ok(choice)
  assert.equal(
    raw.slice(0, choice.from) + choice.insert + raw.slice(choice.to),
    'jax.jit filter:python trailing',
  )
  const middle = searchCompletions('filter:pytho ast:np.exp($X)', 9)[0]
  assert.ok(middle)
  assert.equal('filter:pytho ast:np.exp($X)'.slice(middle.to), 'ast:np.exp($X)')
})

test('code operators are offered only within code search', () => {
  assert.equal(searchCompletions('fil', 3)[0]?.insert, 'filter:')
  assert.equal(searchCompletions('filter:python as', 16)[0]?.insert, 'ast:')
  assert.equal(searchCompletions('filter:code pa', 14)[0]?.insert, 'path:')
  assert.deepEqual(searchCompletions('notes as', 8), [])
})

test('source strings, AST patterns and commands stay opaque', () => {
  for (const query of [
    'filter:python "filter:py',
    "filter:python 'filter:py'",
    'filter:javascript `filter:py',
    'filter:python ast:filter:py',
    'filter:python ast:console.log($A) fil',
    '> filter:py',
  ])
    assert.deepEqual(searchCompletions(query, query.length), [], query)
})
