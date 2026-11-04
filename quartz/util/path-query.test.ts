import assert from 'node:assert/strict'
import test from 'node:test'
import { buildPathTree, completePath, matchPaths, parsePathQuery, routePath } from './path-query'

const tree = buildPathTree(
  [
    'index',
    'embeddings/index',
    'thoughts/index',
    'thoughts/attention',
    'thoughts/autoencoder',
    'thoughts/LW/index',
    'thoughts/LW/rationality',
    'thoughts/papers/attention-is-all',
    'posts/25/index',
    'posts/25/notes',
  ].map(slug => ({ slug, title: slug })),
)

test('parsePathQuery reads only slash-led single tokens', () => {
  assert.deepEqual(parsePathQuery('/embeddings'), { path: 'embeddings', suffix: '' })
  assert.deepEqual(parsePathQuery('  /thoughts/  '), { path: 'thoughts/', suffix: '' })
  assert.deepEqual(parsePathQuery('/posts/25#notes'), { path: 'posts/25', suffix: '#notes' })
  assert.deepEqual(parsePathQuery('/'), { path: '', suffix: '' })
  assert.equal(parsePathQuery('embeddings'), null)
  assert.equal(parsePathQuery('/a b'), null)
  assert.deepEqual(parsePathQuery('path:/thoughts/'), { path: 'thoughts/', suffix: '' })
  assert.deepEqual(parsePathQuery('PATH: /posts/25#notes'), { path: 'posts/25', suffix: '#notes' })
  assert.equal(parsePathQuery('path:thoughts'), null)
  assert.equal(parsePathQuery('filter:code path:/a.py'), null)
})

test('routePath folds folder index slugs', () => {
  assert.equal(routePath('index'), '')
  assert.equal(routePath('thoughts/LW/index'), 'thoughts/LW')
  assert.equal(routePath('thoughts/attention'), 'thoughts/attention')
})

test('exact page comes first and is not repeated', () => {
  const { exact, items } = matchPaths(tree, 'embeddings')
  assert.equal(exact?.slug, 'embeddings/index')
  assert.ok(!items.some(item => item.path === 'embeddings'))
})

test('trailing slash lists children, folders first', () => {
  const { exact, items } = matchPaths(tree, 'thoughts/')
  assert.equal(exact?.slug, 'thoughts/index')
  assert.deepEqual(
    items.map(item => item.path),
    ['thoughts/LW', 'thoughts/papers', 'thoughts/attention', 'thoughts/autoencoder'],
  )
  assert.equal(items[0].pages, 1)
})

test('partial segment completes, then reaches deeper matches', () => {
  const { exact, items } = matchPaths(tree, 'thoughts/att')
  assert.equal(exact, undefined)
  assert.deepEqual(
    items.map(item => item.path),
    ['thoughts/attention', 'thoughts/papers/attention-is-all'],
  )
})

test('subsequence children rank below deeper real matches', () => {
  assert.deepEqual(
    matchPaths(tree, 'thoughts/aenc').items.map(item => item.path),
    ['thoughts/autoencoder'],
  )
  assert.deepEqual(
    matchPaths(tree, 'rat').items.map(item => item.path),
    ['thoughts/LW/rationality'],
  )
})

test('case-insensitive directories', () => {
  assert.deepEqual(
    matchPaths(tree, 'THOUGHTS/lw/').items.map(item => item.path),
    ['thoughts/LW/rationality'],
  )
})

test('root lists top-level paths', () => {
  assert.deepEqual(
    matchPaths(tree, '').items.map(item => item.path),
    ['posts', 'thoughts', 'embeddings'],
  )
})

test('completePath descends into folders', () => {
  const lw = tree.entries.get('thoughts/LW')!
  const note = tree.entries.get('thoughts/attention')!
  assert.equal(completePath(lw), '/thoughts/LW/')
  assert.equal(completePath(note), '/thoughts/attention')
})
