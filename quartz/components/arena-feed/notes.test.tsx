import type { Element } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toText } from 'hast-util-to-text'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import { visit } from 'unist-util-visit'
import type { ArenaNote } from '../../util/arena-reader'
import { draftFromNote, editDraft } from './model'
import { NotesPanel, type DraftStatus } from './notes'

const note: ArenaNote = {
  id: 'note-1',
  articleId: 'article-1',
  sourceUrl: 'https://example.com/article',
  body: 'A thought',
  snapshotId: null,
  quote: null,
  occurrence: null,
  createdAt: 1,
  updatedAt: 1,
  revision: 1,
  readyRevision: null,
  exportedRevision: null,
  exportReceipt: null,
  deletedAt: null,
}

function editorElements(status: DraftStatus): Element[] {
  const draft = editDraft(draftFromNote('owner', note), note.body)
  const tree = fromHtml(
    renderToString(
      <NotesPanel
        drafts={[draft]}
        entries={[]}
        article={null}
        editing={note.id}
        status={() => status}
        onAdd={() => {}}
        onEdit={() => {}}
        onChange={() => {}}
        onReady={() => {}}
        onRetry={() => {}}
        onDelete={() => {}}
        onResolve={() => {}}
        onOpenArticle={() => {}}
        onOpenSnapshot={() => {}}
        snapshotId={null}
      />,
    ),
    { fragment: true },
  )
  const elements: Element[] = []
  visit(tree, 'element', element => {
    elements.push(element)
  })
  return elements
}

test('pending note sync keeps both editor actions available without a retry control', () => {
  const elements = editorElements('pending')
  const buttons = elements.filter(element => element.tagName === 'button')
  const ready = buttons.find(button => toText(button) === 'ready to backfill')
  const remove = buttons.find(button => toText(button) === 'delete')
  const saveStatus = elements.find(element => element.properties.role === 'status')
  assert.ok(ready)
  assert.ok(remove)
  assert.ok(saveStatus)
  assert.equal(ready.properties.disabled, undefined)
  assert.equal(remove.properties.disabled, undefined)
  assert.equal(
    buttons.some(button => toText(button) === 'sync'),
    false,
  )
  assert.equal(toText(saveStatus), 'Pending sync')
})

test('a failed sync offers a retry without blocking editor actions', () => {
  const elements = editorElements('sync-failed')
  const buttons = elements.filter(element => element.tagName === 'button')
  const saveStatus = elements.find(element => element.properties.role === 'status')
  assert.ok(saveStatus)
  assert.ok(buttons.some(button => toText(button) === 'sync'))
  assert.equal(toText(saveStatus), 'Sync failed')
})
