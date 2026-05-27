import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildStackedNoteHtml,
  dedupeSlugs,
  failedNoteData,
  hashSlug,
  normalizeStackedNoteSlug,
  pendingNoteData,
  shouldIncludeServerBody,
} from '../../worker/stacked'
import {
  decodeStackedNoteHash,
  hashStackedNoteSlug,
  stackedNoteMetadataHtml,
  withStackedNoteMetadata,
} from './stacked-notes'

test('dedupeSlugs preserves first-seen order', () => {
  assert.deepEqual(dedupeSlugs(['notes', 'thoughts/kant', 'notes', 'base']), [
    'notes',
    'thoughts/kant',
    'base',
  ])
})

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

test('pendingNoteData does not need global index metadata', () => {
  const note = pendingNoteData('thoughts/kant')

  assert.equal(note.slug, 'thoughts/kant')
  assert.equal(note.title, 'thoughts/kant')
  assert.equal(note.metadata, '')
  assert.equal(note.state, 'pending')
})

test('stacked refresh seeds the first note body before pending successors', () => {
  assert.equal(shouldIncludeServerBody(0), true)
  assert.equal(shouldIncludeServerBody(1), false)
  assert.equal(shouldIncludeServerBody(3), false)
})

test('stacked note metadata footer prioritizes modified date', () => {
  const footer = stackedNoteMetadataHtml([
    '<li class="published-time"><h2>publié à</h2><div class="container">30 oct. 2024</div></li>',
    '<li class="reading-time"><h2>durée</h2><div class="container">1 min</div></li>',
    '<li class="modified-time"><h2>modifié à</h2><div class="container">26 mai 2026</div></li>',
  ])

  assert.equal(footer.startsWith('<footer class="stacked-note-footer"'), true)
  assert.ok(footer.indexOf('modified-time') < footer.indexOf('published-time'))
  assert.ok(footer.indexOf('published-time') < footer.indexOf('reading-time'))
})

test('stacked note metadata renders before page footer', () => {
  const metadata = '<footer class="stacked-note-footer">modifié à 26 mai 2026</footer>'
  const html = withStackedNoteMetadata(
    '<article>body</article><section class="page-footer popover-hint">backlinks</section>',
    metadata,
  )

  assert.ok(html.indexOf('stacked-note-footer') < html.indexOf('page-footer'))
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
