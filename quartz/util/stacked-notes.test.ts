import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildStackedNoteHtml,
  failedNoteData,
  hashSlug,
  normalizeStackedNoteSlug,
} from '../../worker/stacked'
import {
  cacheStackedNotePayload,
  clearStackedNotePayloadCache,
  decodeStackedNoteHash,
  getCachedStackedNotePayload,
  hashStackedNoteSlug,
  MAX_STACKED_NOTE_PAYLOAD_CACHE_ENTRIES,
  readStackedNotePayload,
  stackedNoteSlugs,
  stackedNotesUrl,
} from './stacked-notes'

test('buildStackedNoteHtml renders a visible failed note state', () => {
  const note = failedNoteData('thoughts/kant', 'Kant')
  const html = buildStackedNoteHtml(note, 1, 3)

  assert.equal(note.state, 'failed')
  assert.ok(html.includes(`id="${hashSlug('thoughts/kant')}"`))
  assert.ok(html.includes('class="stacked-note failed"'))
  assert.ok(html.includes('data-state="failed"'))
  assert.ok(html.includes('Kant'))
  assert.ok(html.includes('data-stacked-retry'))
  assert.ok(html.includes('<svg'))
  assert.equal(html.includes('>retry<'), false)
})

test('normalizeStackedNoteSlug accepts only route-like slugs', () => {
  assert.equal(normalizeStackedNoteSlug('/thoughts/kant/'), 'thoughts/kant')
  assert.equal(normalizeStackedNoteSlug('courses/notes_b/index'), 'courses/notes_b/index')
  assert.equal(normalizeStackedNoteSlug('thoughts/antinomy-öⁿ'), 'thoughts/antinomy-öⁿ')
  assert.equal(normalizeStackedNoteSlug(''), null)
  assert.equal(normalizeStackedNoteSlug('/'), 'index')
  assert.equal(normalizeStackedNoteSlug('../secrets'), null)
  assert.equal(normalizeStackedNoteSlug('thoughts/kant?x=1'), null)
  assert.equal(normalizeStackedNoteSlug('thoughts\\kant'), null)
})

test('stacked note hashes round-trip real slugs', () => {
  const slugs = ['notes', 'courses/notes_b/index', 'thoughts/antinomy-öⁿ', 'thoughts/a.b']

  for (const slug of slugs) {
    assert.equal(decodeStackedNoteHash(hashStackedNoteSlug(slug)), slug)
    assert.equal(decodeStackedNoteHash(hashSlug(slug)), slug)
  }

  assert.equal(decodeStackedNoteHash('bm90ZXM'), 'notes')
})

test('stacked note payload parsing accepts only complete known states', () => {
  assert.deepEqual(
    readStackedNotePayload({
      slug: 'thoughts/kant',
      title: 'Kant',
      content: '<article>Kant</article>',
      state: 'ready',
    }),
    { slug: 'thoughts/kant', title: 'Kant', content: '<article>Kant</article>', state: 'ready' },
  )
  assert.equal(
    readStackedNotePayload({ slug: 'thoughts/kant', title: 'Kant', content: '', state: 'ready' }),
    null,
  )
  assert.equal(
    readStackedNotePayload({
      slug: 'thoughts/kant',
      title: 'Kant',
      content: '<article>Kant</article>',
      state: 'unknown',
    }),
    null,
  )
})

test('stack URLs use the path as root and query only for additional notes', () => {
  const root = new URL('https://notes.aarnphm.xyz/personal-canon?lang=fr#books')
  assert.deepEqual(stackedNoteSlugs(root), ['personal-canon'])
  assert.equal(stackedNotesUrl(root, ['personal-canon']).search, root.search)
  const next = stackedNotesUrl(root, ['personal-canon', 'thoughts/antinomy-öⁿ'])
  assert.equal(next.pathname, `/${hashStackedNoteSlug('personal-canon')}`)
  assert.deepEqual(next.searchParams.getAll('stackedNotes'), [
    hashStackedNoteSlug('thoughts/antinomy-öⁿ'),
  ])
  assert.equal(next.searchParams.get('lang'), 'fr')
  assert.equal(next.hash, '#books')
  assert.deepEqual(stackedNoteSlugs(next), ['personal-canon', 'thoughts/antinomy-öⁿ'])
})

test('stack URLs accept legacy roots and ignore malformed or duplicate entries', () => {
  const url = new URL('https://notes.aarnphm.xyz/personal-canon')
  for (const hash of [
    hashStackedNoteSlug('personal-canon'),
    '!',
    hashStackedNoteSlug('notes'),
    hashStackedNoteSlug('notes'),
  ]) {
    url.searchParams.append('stackedNotes', hash)
  }
  assert.deepEqual(stackedNoteSlugs(url), ['personal-canon', 'notes'])
  assert.deepEqual(
    stackedNotesUrl(url, stackedNoteSlugs(url)).searchParams.getAll('stackedNotes'),
    [hashStackedNoteSlug('notes')],
  )
  assert.deepEqual(stackedNoteSlugs(new URL('https://notes.aarnphm.xyz/')), ['notes'])
  assert.deepEqual(stackedNoteSlugs(new URL('https://aarnphm.xyz/')), ['index'])
})

test('notes host accepts readable and encoded paths and writes encoded roots', () => {
  for (const slug of ['personal-canon', 'thoughts/kant', 'thoughts/antinomy-öⁿ']) {
    const readable = new URL(`https://notes.aarnphm.xyz/${slug}?lang=fr#section`)
    const encoded = new URL(
      `https://notes.aarnphm.xyz/${hashStackedNoteSlug(slug)}?lang=fr#section`,
    )
    assert.deepEqual(stackedNoteSlugs(readable), [slug])
    assert.deepEqual(stackedNoteSlugs(encoded), [slug])
    assert.equal(stackedNotesUrl(readable, [slug]).href, encoded.href)
    assert.deepEqual(stackedNoteSlugs(stackedNotesUrl(encoded, [slug, 'notes'])), [slug, 'notes'])
  }
  const apex = new URL('https://aarnphm.xyz/personal-canon')
  assert.equal(stackedNotesUrl(apex, ['personal-canon']).href, apex.href)
})

test('stacked note payload cache enforces LRU bound and clears cleanly', t => {
  const prevWindow = (globalThis as { window?: unknown }).window
  const mockWindow: Record<string, unknown> = {}
  ;(globalThis as { window?: unknown }).window = mockWindow
  t.after(() => {
    ;(globalThis as { window?: unknown }).window = prevWindow
  })

  clearStackedNotePayloadCache()

  for (let i = 0; i < MAX_STACKED_NOTE_PAYLOAD_CACHE_ENTRIES; i++) {
    cacheStackedNotePayload({
      slug: `thoughts/note-${i}`,
      title: `Note ${i}`,
      content: `<article>Note ${i}</article>`,
      state: 'ready',
    })
  }

  // Touch note-0 so it becomes most recently used; note-1 should be evicted next
  assert.equal(getCachedStackedNotePayload('thoughts/note-0')?.title, 'Note 0')

  cacheStackedNotePayload({
    slug: 'thoughts/note-overflow',
    title: 'Overflow',
    content: '<article>Overflow</article>',
    state: 'ready',
  })

  assert.equal(getCachedStackedNotePayload('thoughts/note-0')?.title, 'Note 0')
  assert.equal(getCachedStackedNotePayload('thoughts/note-1'), null)
  assert.equal(getCachedStackedNotePayload('thoughts/note-overflow')?.title, 'Overflow')

  clearStackedNotePayloadCache()
  assert.equal(getCachedStackedNotePayload('thoughts/note-0'), null)
  assert.equal(getCachedStackedNotePayload('thoughts/note-overflow'), null)
})
