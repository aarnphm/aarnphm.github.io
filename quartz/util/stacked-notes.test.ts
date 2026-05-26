import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildStackedNoteHtml,
  dedupeSlugs,
  failedNoteData,
  hashSlug,
  normalizeStackedNoteSlug,
  pendingNoteData,
} from '../../worker/stacked'
import { decodeStackedNoteHash, hashStackedNoteSlug } from './stacked-notes'

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
})

test('pendingNoteData does not need global index metadata', () => {
  const note = pendingNoteData('thoughts/kant')

  assert.equal(note.slug, 'thoughts/kant')
  assert.equal(note.title, 'thoughts/kant')
  assert.equal(note.metadata, '')
  assert.equal(note.state, 'pending')
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
