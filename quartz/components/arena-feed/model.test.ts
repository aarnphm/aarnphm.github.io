import assert from 'node:assert/strict'
import test from 'node:test'
import type { ArenaFeedEntry } from '../../util/arena-feed'
import type { ArenaNote } from '../../util/arena-reader'
import {
  acknowledgeDraft,
  boundedQuote,
  draftFromNote,
  editDraft,
  eligibleEntries,
  mergeRemoteDraft,
  mergeReadLinks,
  nextEntry,
  noteStage,
} from './model'

function entry(id: string, later: boolean): ArenaFeedEntry {
  return {
    articleId: id,
    sourceUrl: `https://example.com/${id}`,
    title: `Article ${id}`,
    kind: 'html',
    later,
    tags: ['systems'],
    savedAt: null,
    occurrences: [
      {
        channelSlug: 'reading',
        channelName: 'Reading',
        blockId: id,
        parentBlockId: null,
        notesHtml: null,
      },
    ],
  }
}

const note: ArenaNote = {
  id: 'note-1',
  articleId: 'a',
  sourceUrl: 'https://example.com/a',
  body: 'A thought',
  snapshotId: 'snapshot-1',
  quote: { exact: 'a passage', prefix: 'before ', suffix: ' after' },
  occurrence: null,
  createdAt: 1,
  updatedAt: 1,
  revision: 2,
  readyRevision: null,
  exportedRevision: null,
  exportReceipt: null,
  deletedAt: null,
}

test('opening, skipping, and writing notes do not create read state', () => {
  const entries = [entry('later-a', true), entry('later-b', true), entry('rest', false)]
  assert.equal(nextEntry(entries, ['later-a'], 'later-a')?.articleId, 'later-b')
  assert.equal(nextEntry(entries, ['later-a', 'later-b'], 'later-b')?.articleId, 'rest')
  assert.deepEqual(eligibleEntries(entries, [], 'unread'), entries)
  assert.equal(eligibleEntries(entries, [], 'read').length, 0)
})

test('unread filtering preserves server order and mark unread restores a link', () => {
  const entries = [entry('a', true), entry('b', true), entry('c', false)]
  const readLinks = [{ articleId: 'a', readAt: 10, updatedAt: 10, revision: 1 }]
  assert.deepEqual(
    eligibleEntries(entries, readLinks, 'unread').map(item => item.articleId),
    ['b', 'c'],
  )
  assert.deepEqual(
    eligibleEntries(entries, readLinks, 'read').map(item => item.articleId),
    ['a'],
  )
  assert.deepEqual(eligibleEntries(entries, [{ ...readLinks[0], readAt: null }], 'unread'), entries)
  assert.deepEqual(
    eligibleEntries(entries, readLinks, 'all', 'reading systems b').map(item => item.articleId),
    ['b'],
  )
})

test('a completed pass waits for an explicit shuffle', () => {
  const entries = [entry('a', true), entry('b', false)]
  assert.equal(nextEntry(entries, ['a', 'b'], 'b'), null)
  assert.equal(nextEntry(entries, [], null)?.articleId, 'a')
})

test('searching curius includes shared and Curius-only links and preserves explicit read state', () => {
  const curius = [{ userId: 3584, linkId: 1 }]
  const links = [
    entry('arena', false),
    { ...entry('shared', true), curius },
    { ...entry('curius-only', false), occurrences: [], curius },
  ]
  assert.deepEqual(
    eligibleEntries(links, [], 'unread', 'curius').map(item => item.articleId),
    ['shared', 'curius-only'],
  )
  const readLinks = [{ articleId: 'shared', readAt: 10, updatedAt: 10, revision: 1 }]
  assert.deepEqual(
    eligibleEntries(links, readLinks, 'unread', 'curius').map(item => item.articleId),
    ['curius-only'],
  )
  assert.deepEqual(
    eligibleEntries(links, readLinks, 'read', 'curius').map(item => item.articleId),
    ['shared'],
  )
})

test('the Curius filter uses source metadata and includes shared, read, and unread links once', () => {
  const curius = [{ userId: 3584, linkId: 1 }]
  const shared = { ...entry('shared', true), curius: [...curius, { userId: 3584, linkId: 2 }] }
  const curiusOnly = { ...entry('curius-only', false), occurrences: [], curius }
  const links = [
    entry('arena', false),
    entry('curius-in-title-and-url', true),
    { ...entry('empty-source', false), curius: [] },
    shared,
    curiusOnly,
  ]
  const readLinks = [{ articleId: 'shared', readAt: 10, updatedAt: 10, revision: 1 }]
  assert.deepEqual(eligibleEntries(links, readLinks, 'curius'), [shared, curiusOnly])
  assert.deepEqual(eligibleEntries(links, readLinks, 'curius', 'reading systems shared'), [shared])
  assert.deepEqual(eligibleEntries(links, readLinks, 'curius', 'CURiUS-ONLY'), [curiusOnly])
  assert.deepEqual(eligibleEntries(links, readLinks, 'curius', 'arena'), [])
})

test('edits invalidate ready status without changing article or snapshot anchors', () => {
  const draft = draftFromNote('owner-a', { ...note, readyRevision: note.revision })
  const edited = editDraft(draft, 'A newer thought')
  assert.equal(edited.ready, false)
  assert.equal(edited.dirty, true)
  assert.equal(edited.note.articleId, note.articleId)
  assert.equal(edited.note.snapshotId, note.snapshotId)
  assert.deepEqual(edited.note.quote, note.quote)
  assert.equal(edited.subject, 'owner-a')
})

test('an autosave acknowledgement retains edits typed while it was in flight', () => {
  const sent = editDraft(draftFromNote('owner-a', note), 'First edit')
  const current = editDraft(sent, 'Second edit while saving')
  const acknowledged = acknowledgeDraft(current, sent, { ...sent.note, revision: 3 })
  assert.equal(acknowledged.note.body, 'Second edit while saving')
  assert.equal(acknowledged.note.revision, 3)
  assert.equal(acknowledged.dirty, true)
  assert.equal(acknowledged.localVersion, current.localVersion)
  const saved = acknowledgeDraft(acknowledged, acknowledged, { ...acknowledged.note, revision: 4 })
  assert.equal(saved.dirty, false)
  assert.equal(saved.note.body, 'Second edit while saving')
})

test('remote edits and deletions retain the local text for conflict resolution', () => {
  const local = editDraft(draftFromNote('owner-a', note), 'Unsynced local text')
  const remote = { ...note, body: 'Other device text', revision: 3 }
  const conflict = mergeRemoteDraft(local, remote)
  assert.equal(conflict.note.body, 'Unsynced local text')
  assert.equal(conflict.conflict?.body, 'Other device text')
  const deleted = mergeRemoteDraft(local, { ...remote, deletedAt: 10 })
  assert.equal(deleted.note.body, 'Unsynced local text')
  assert.equal(deleted.conflict?.deletedAt, 10)
  assert.equal(
    mergeRemoteDraft(draftFromNote('owner-a', note), remote).note.body,
    'Other device text',
  )
})

test('backfill stages belong to the exact saved revision', () => {
  assert.equal(noteStage(note), 'draft')
  assert.equal(noteStage({ ...note, readyRevision: 2 }), 'ready')
  assert.equal(noteStage({ ...note, readyRevision: 2, exportedRevision: 2 }), 'backfilled')
  assert.equal(noteStage({ ...note, revision: 3, readyRevision: 2, exportedRevision: 2 }), 'draft')
})

test('a delayed refresh cannot overwrite a newer acknowledged revision', () => {
  const current = draftFromNote('owner-a', { ...note, body: 'Already saved', revision: 4 })
  assert.equal(mergeRemoteDraft(current, note), current)
})

test('long selected quotes stay within the API limit and retain adjacent context', () => {
  const quote = boundedQuote(
    `${'a'.repeat(4096)}trailing passage`,
    'prefix'.repeat(50),
    'after original selection',
  )
  assert.equal(quote?.exact.length, 4096)
  assert.equal(quote?.prefix.length, 100)
  assert.equal(quote?.suffix, 'trailing passage')
  assert.equal(boundedQuote('  ', 'prefix', 'suffix'), null)
})

test('read marks acknowledged during a refresh survive its older response', () => {
  const latest = { articleId: 'a', readAt: 10, updatedAt: 10, revision: 2 }
  const older = { ...latest, readAt: null, revision: 1 }
  assert.deepEqual(mergeReadLinks([latest], [older]), [latest])
  assert.deepEqual(mergeReadLinks([latest], []), [latest])
  const unread = { ...latest, readAt: null, revision: 3 }
  assert.deepEqual(mergeReadLinks([latest], [unread]), [unread])
})
