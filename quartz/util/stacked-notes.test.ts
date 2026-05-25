import assert from 'node:assert/strict'
import test from 'node:test'
import { buildStackedNoteHtml, dedupeSlugs, failedNoteData, hashSlug } from '../../worker/stacked'

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
