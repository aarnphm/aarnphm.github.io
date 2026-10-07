import assert from 'node:assert/strict'
import test from 'node:test'
import { aggregateSemanticResults, fuseSearchRanks } from './semantic-search'

test('a strong passage outranks repeated weaker matches from a long document', () => {
  const ids = ['short', ...Array.from({ length: 30 }, (_, i) => `long#chunk${i}`)]
  const chunkMetadata = Object.fromEntries(
    ids.slice(1).map((slug, chunkId) => [slug, { parentSlug: 'long', chunkId }]),
  )
  const hits = ids.map((_, id) => ({ id, score: id === 0 ? 0.99 : 0.7 }))
  assert.deepEqual(aggregateSemanticResults(hits, { ids, chunkMetadata }), [
    { slug: 'short', score: 0.99 },
    { slug: 'long', score: 0.7 },
  ])
})

test('aggregation keeps the best passage and ignores invalid vector results', () => {
  assert.deepEqual(
    aggregateSemanticResults(
      [
        { id: 0, score: 0.4 },
        { id: 1, score: 0.9 },
        { id: 1, score: 0.9 },
        { id: 2, score: Number.NaN },
        { id: 3, score: Infinity },
        { id: 99, score: 1 },
      ],
      {
        ids: ['note#chunk0', 'note#chunk1', 'bad', 'bad-too'],
        chunkMetadata: {
          'note#chunk0': { parentSlug: 'note', chunkId: 0 },
          'note#chunk1': { parentSlug: 'note', chunkId: 1 },
        },
      },
    ),
    [{ slug: 'note', score: 0.9 }],
  )
})

test('hybrid ranking rewards agreement while retaining lexical-only notes', () => {
  const ranked = fuseSearchRanks([
    { ids: ['literal', 'shared'], weight: 1 },
    { ids: ['semantic', 'shared'], weight: 1 },
  ])
  assert.deepEqual(
    ranked.map(hit => hit.id),
    ['shared', 'literal', 'semantic'],
  )
})

test('duplicate IDs cannot inflate a ranking contribution', () => {
  assert.deepEqual(
    fuseSearchRanks([{ ids: ['first', 'first', 'second'], weight: 1 }]),
    fuseSearchRanks([{ ids: ['first', 'second'], weight: 1 }]),
  )
})

test('title boosts can preserve an exact lexical match alongside semantic results', () => {
  const ranked = fuseSearchRanks([
    { ids: [4, 5], weight: 1, boosts: new Map([[4, 1.5]]) },
    { ids: [8, 9], weight: 1 },
  ])
  assert.equal(ranked[0].id, 4)
})
