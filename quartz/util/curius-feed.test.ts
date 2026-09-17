import assert from 'node:assert/strict'
import test from 'node:test'
import {
  arenaFeedArticleId,
  arenaFeedSourceNames,
  buildArenaFeedManifest,
  isArenaReadingEntry,
  parseArenaFeedManifest,
} from './arena-feed'
import { mergeCuriusFeed, parseCuriusFeedLinks, type CuriusSavedLink } from './curius-feed'

const base = 'https://aarnphm.xyz'
const saved: CuriusSavedLink = {
  id: 236997,
  link: 'https://example.com/essay?curius=3584&utm_source=curius',
  title: 'An essay from Curius',
  toRead: null,
  createdDate: '2026-09-15T00:09:31.811Z',
}

test('the Curius index validates the entire response and accepts its nullable fields', () => {
  assert.deepEqual(parseCuriusFeedLinks({ links: [saved] }), [saved])
  assert.deepEqual(parseCuriusFeedLinks({ links: [] }), [])
  for (const value of [
    {},
    { links: null },
    { links: [saved, { link: 'https://example.com' }] },
    { links: [{ ...saved, id: -1 }] },
    { links: [{ ...saved, toRead: 'true' }] },
  ]) {
    assert.equal(parseCuriusFeedLinks(value), null)
  }
})

test('duplicates retain Arena identity, notes and tags while adding Curius provenance', async () => {
  const arena = await buildArenaFeedManifest(
    [
      {
        id: 'essays',
        slug: 'essay',
        name: '#essay',
        tags: ['writing'],
        blocks: [
          {
            id: 'block-9',
            title: 'Authored title',
            content: 'Authored title',
            url: 'https://example.com/essay',
            metadata: { date: '2026-09-16' },
            subItems: [{ id: 'note-1', content: 'Keep this saved note.' }],
          },
        ],
      },
    ],
    base,
  )
  const original = structuredClone(arena)
  const merged = await mergeCuriusFeed(arena, [saved, { ...saved, id: 236998, toRead: true }], base)
  assert.equal(merged.entries.length, 1)
  const entry = merged.entries[0]
  assert.equal(entry.articleId, arena.entries[0].articleId)
  assert.equal(entry.sourceUrl, 'https://example.com/essay')
  assert.equal(entry.title, 'Authored title')
  assert.equal(entry.savedAt, saved.createdDate)
  assert.equal(entry.later, true)
  assert.deepEqual(entry.tags, ['writing'])
  assert.deepEqual(entry.occurrences, arena.entries[0].occurrences)
  assert.match(entry.occurrences[0].notesHtml ?? '', /Keep this saved note/)
  assert.deepEqual(entry.curius, [
    { userId: 3584, linkId: 236997 },
    { userId: 3584, linkId: 236998 },
  ])
  assert.deepEqual(arenaFeedSourceNames(entry), ['#essay', 'curius'])
  assert.deepEqual(arena, original)
  assert.ok(parseArenaFeedManifest(merged))
})

test('all Curius index entries are ingested without a page limit and share source classification', async () => {
  const empty = await buildArenaFeedManifest([], base)
  const links = Array.from({ length: 1423 }, (_, index) => ({
    ...saved,
    id: index + 1,
    link: `https://example.com/${index}`,
  }))
  links.push(
    { ...saved, id: 2001, link: 'https://arxiv.org/abs/2601.00417' },
    { ...saved, id: 2002, link: 'https://example.com/paper.pdf' },
    { ...saved, id: 2003, link: `${base}/thoughts/reading` },
    { ...saved, id: 2004, link: 'https://www.youtube.com/watch?v=NB1PNDN24cE' },
    { ...saved, id: 2005, link: 'javascript:alert(1)' },
    { ...saved, id: 2006, link: 'https://user:password@example.com/' },
  )
  const merged = await mergeCuriusFeed(empty, links, base)
  assert.equal(merged.entries.length, 1427)
  assert.ok(merged.entries.some(entry => entry.sourceUrl === 'https://example.com/1422'))
  assert.equal(merged.entries.find(entry => entry.sourceUrl.includes('arxiv.org'))?.kind, 'pdf')
  assert.equal(merged.entries.find(entry => entry.sourceUrl.endsWith('paper.pdf'))?.kind, 'pdf')
  assert.equal(merged.entries.find(entry => entry.sourceUrl.startsWith(base))?.kind, 'internal')
  assert.equal(merged.entries.filter(isArenaReadingEntry).length, 1426)
  assert.ok(merged.entries.every(entry => entry.occurrences.length === 0 && !entry.later))
  assert.ok(parseArenaFeedManifest(merged))
})

test('Curius URL identity survives source order, duplicate imports, title edits and an Arena save', async () => {
  const empty = await buildArenaFeedManifest([], base)
  const links = [saved, { ...saved, id: 12, link: 'https://example.com/essay?utm_medium=share' }]
  const original = await mergeCuriusFeed(empty, links, base)
  const reordered = await mergeCuriusFeed(empty, links.toReversed(), base)
  const repeated = await mergeCuriusFeed(original, links, base)
  assert.deepEqual(reordered, original)
  assert.deepEqual(repeated, original)
  assert.equal(original.entries[0].articleId, await arenaFeedArticleId('https://example.com/essay'))
  const edited = await mergeCuriusFeed(
    empty,
    [{ ...saved, title: 'Updated title', toRead: true }],
    base,
  )
  assert.equal(edited.entries[0].articleId, original.entries[0].articleId)
  assert.notEqual(edited.revision, original.revision)
  const arena = await buildArenaFeedManifest(
    [
      {
        id: 'reading',
        slug: 'reading',
        name: 'Reading',
        blocks: [
          { id: 'new-block', content: 'Saved later in Arena', url: 'https://example.com/essay' },
        ],
      },
    ],
    base,
  )
  assert.equal(arena.entries[0].articleId, original.entries[0].articleId)
  assert.equal((await mergeCuriusFeed(arena, links, base)).entries.length, 1)
})

test('deduplication retains content parameters, fragments and HTTP distinctions', async () => {
  const empty = await buildArenaFeedManifest([], base)
  const urls = [
    'https://example.com/essay',
    'http://example.com/essay',
    'https://example.com/essay/',
    'https://example.com/essay?version=2',
    'https://example.com/essay#part-2',
  ]
  const merged = await mergeCuriusFeed(
    empty,
    urls.map((link, id) => ({ ...saved, id: id + 1, link })),
    base,
  )
  assert.equal(merged.entries.length, urls.length)
  assert.equal(new Set(merged.entries.map(entry => entry.articleId)).size, urls.length)
  const entry = merged.entries[0]
  for (const curius of [[], [{ userId: 3584, linkId: 'bad' }], [{ userId: -1, linkId: 1 }]])
    assert.equal(parseArenaFeedManifest({ ...merged, entries: [{ ...entry, curius }] }), null)
})
