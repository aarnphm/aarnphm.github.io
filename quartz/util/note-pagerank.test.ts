import assert from 'node:assert/strict'
import test from 'node:test'
import { createNoteRanker, type NoteRankMetadata } from './note-pagerank'

function note(links: string[] = [], overrides: Partial<NoteRankMetadata> = {}): NoteRankMetadata {
  return { links, filePath: 'content/note.md', fileName: 'note', ...overrides }
}

test('homepage omits course navigation indexes while course pages keep their ranking', () => {
  const data = {
    index: note(),
    garden: note(),
    'courses/example/index': note(['courses/example/resources/unit/']),
    'courses/example/resources/unit/index': note(['courses/example/']),
    'courses/example/pages/syllabus/index': note(['courses/example/']),
    'courses/example/pages/index': note(['courses/example/']),
    'courses/example/resources/index': note(['courses/example/']),
    'courses/example/lectures/index': note(),
    'courses/example/lectures/01': note(['courses/example/resources/unit/']),
  }
  const ranker = createNoteRanker(data)
  const courseBefore = ranker.related('courses/example/lectures/01', 10)
  assert.ok(courseBefore.some(slug => slug === 'courses/example/resources/unit/index'))
  assert.ok(courseBefore.some(slug => slug === 'courses/example/index'))
  assert.deepEqual(ranker.related('index', 10), [
    'courses/example/lectures/01',
    'courses/example/lectures/index',
    'garden',
  ])
  assert.deepEqual(ranker.related('/#welcome', 10), ranker.related('index', 10))
  assert.deepEqual(ranker.related('courses/example/lectures/01', 10), courseBefore)
  assert.ok(ranker.related('missing', 10).some(slug => slug === 'courses/example/index'))
})

test('homepage policy remains distinct from missing-page cache without a homepage node', () => {
  const ranker = createNoteRanker({ 'courses/example/index': note(), garden: note() })
  assert.deepEqual(ranker.related('missing', 10), ['courses/example/index', 'garden'])
  assert.deepEqual(ranker.related('index', 10), ['garden'])
  assert.deepEqual(ranker.related('/', 10), ['garden'])
})

test('nearby notes lead and disconnected global results fill remaining slots', () => {
  const ranker = createNoteRanker({ a: note(['b']), b: note(), c: note(['d']), d: note() })
  assert.deepEqual(ranker.related('a', 3), ['b', 'd', 'c'])
  assert.deepEqual(ranker.related('b', 1), ['a'])
  assert.deepEqual(ranker.related('index', 4), ['b', 'd', 'a', 'c'])
})

test('empty, isolated, singleton, and invalid limits have defined behavior', () => {
  assert.deepEqual(createNoteRanker({}).related('index', 10), [])
  assert.deepEqual(createNoteRanker({ a: note() }).related('a', 10), [])
  const ranker = createNoteRanker({ b: note(), a: note(), c: note() })
  assert.deepEqual(ranker.related('missing', 10), ['a', 'b', 'c'])
  assert.deepEqual(ranker.related('a', 10), ['b', 'c'])
  for (const limit of [0, -1, NaN, Infinity]) assert.deepEqual(ranker.related('a', limit), [])
  assert.deepEqual(ranker.related('missing', 1.9), ['a'])
})

test('protected, excluded and asset nodes cannot bridge the local graph', () => {
  const ranker = createNoteRanker({
    a: note(['hidden', 'hinterland/note', 'asset', 'path-excluded']),
    hidden: note(['z'], { protected: true }),
    'hinterland/note': note(['z']),
    'path-excluded': note(['z'], { fileName: 'content/embeddings/model' }),
    asset: note(['z'], { filePath: 'content/image.png' }),
    z: note(),
    b: note(['c']),
    c: note(),
  })
  assert.deepEqual(ranker.related('a', 10), ['c', 'b', 'z'])
})

test('index paths and fragments resolve to canonical rows', () => {
  const ranker = createNoteRanker({
    'topic/index': note(['child#section']),
    child: note(),
    source: note(['topic/#heading']),
  })
  assert.deepEqual(ranker.related('topic/#section', 2), ['child', 'source'])
  assert.deepEqual(ranker.related('source', 1), ['topic/index'])
})

test('ambiguous simplified slugs do not depend on insertion order', () => {
  const data = { source: note(['topic/#heading']), topic: note(), 'topic/index': note() }
  assert.deepEqual(createNoteRanker(data).related('source', 10), ['topic', 'topic/index'])
  assert.deepEqual(
    createNoteRanker(Object.fromEntries(Object.entries(data).reverse())).related('source', 10),
    createNoteRanker(data).related('source', 10),
  )
})

test('duplicates, self links, unresolved links and reciprocal edges preserve local order', () => {
  const original = { a: note(['b', 'c']), b: note(), c: note(), d: note() }
  const redundant = {
    d: note(),
    c: note(['a']),
    b: note(['a']),
    a: note(['b', 'b', 'c', 'a', 'missing', 'https://example.com']),
  }
  assert.deepEqual(createNoteRanker(original).related('a', 2), ['b', 'c'])
  assert.deepEqual(createNoteRanker(redundant).related('a', 2), ['b', 'c'])
})

test('dangling global nodes retain mass and rank ahead of their sole source', () => {
  assert.deepEqual(createNoteRanker({ a: note(['b']), b: note() }).related('index', 2), ['b', 'a'])
})

test('global ranking exposes normalized scores, stable positions and the complete eligible inventory', () => {
  const ranker = createNoteRanker({
    a: note(['b']),
    b: note(),
    asset: note([], { filePath: 'a.png' }),
  })
  const rows = ranker.global()
  assert.deepEqual(
    rows.map(row => [row.slug, row.position]),
    [
      ['b', 1],
      ['a', 2],
    ],
  )
  assert.ok(Math.abs(rows[0].score - 1.85 / 2.85) < 1e-9)
  assert.ok(Math.abs(rows.reduce((sum, row) => sum + row.score, 0) - 1) < 1e-9)
  assert.ok(Object.isFrozen(rows) && Object.isFrozen(rows[0]))
  assert.equal(ranker.position('b#section'), rows[0])
  assert.equal(ranker.position('asset'), undefined)
})

test('ranker snapshots metadata and callers cannot mutate cached results', () => {
  const links = ['b']
  const data = { a: note(links), b: note(), c: note() }
  const ranker = createNoteRanker(data)
  links.push('c')
  const first = ranker.related('a', 2)
  assert.ok(Object.isFrozen(first))
  assert.deepEqual(first, ['b', 'c'])
  assert.deepEqual(ranker.related('a', 2), first)
})
