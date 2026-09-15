import type { PlatformProxy } from 'wrangler'
import assert from 'node:assert/strict'
import { createHash, randomUUID } from 'node:crypto'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, test } from 'node:test'
import { getPlatformProxy } from 'wrangler'
import type { ArenaNote } from '../quartz/util/arena-reader'
import type { ArenaNoteInput } from './arena-reader-state'
import {
  acknowledgeArenaNoteExport,
  ArenaReaderConflictError,
  ArenaReaderNoteNotFoundError,
  deleteArenaNote,
  exportReadyArenaNotes,
  listArenaNotes,
  listReadLinks,
  saveArenaNote,
  setReadLink,
} from './arena-reader-state'

interface TestEnv {
  ARENA_READER: D1Database
  UNMIGRATED_READER: D1Database
}

let proxy: PlatformProxy<TestEnv> | undefined
let directory: string | undefined

function database(): D1Database {
  assert.ok(proxy)
  return proxy.env.ARENA_READER
}

before(async () => {
  directory = await mkdtemp(path.join(tmpdir(), 'arena-reader-state-'))
  const configPath = path.join(directory, 'wrangler.json')
  await writeFile(
    configPath,
    JSON.stringify({
      name: 'arena-reader-state-test',
      compatibility_date: '2025-01-21',
      d1_databases: [
        {
          binding: 'ARENA_READER',
          database_name: 'arena-reader-state-test',
          database_id: '00000000-0000-0000-0000-000000000001',
        },
        {
          binding: 'UNMIGRATED_READER',
          database_name: 'arena-reader-unmigrated-test',
          database_id: '00000000-0000-0000-0000-000000000002',
        },
      ],
    }),
  )
  proxy = await getPlatformProxy<TestEnv>({
    configPath,
    persist: false,
    remoteBindings: false,
    envFiles: [],
  })
  const migration = await readFile(
    new URL('../migrations/arena-reader/0000_arena_reader.sql', import.meta.url),
    'utf8',
  )
  for (const statement of migration.split('--> statement-breakpoint')) {
    await database().prepare(statement).run()
  }
})

after(async () => {
  await proxy?.dispose()
  if (directory) await rm(directory, { recursive: true, force: true })
})

function input(overrides: Partial<ArenaNoteInput> = {}): ArenaNoteInput {
  return {
    articleId: 'article-a',
    sourceUrl: 'https://example.com/article',
    body: 'A note written on the phone.',
    snapshotId: null,
    quote: null,
    occurrence: null,
    revision: 0,
    ready: false,
    ...overrides,
  }
}

function edit(note: ArenaNote, overrides: Partial<ArenaNoteInput> = {}): ArenaNoteInput {
  return input({ ...note, ready: false, ...overrides })
}

test('read marks are explicit, owner-scoped, reversible, and preserve unread revisions', async () => {
  const subject = randomUUID()
  assert.deepEqual(await listReadLinks(database(), subject), [])
  const marked = await setReadLink(database(), subject, 'article-a', { read: true, revision: 0 })
  assert.equal(marked.articleId, 'article-a')
  assert.equal(marked.revision, 1)
  assert.equal(typeof marked.readAt, 'number')
  assert.deepEqual(await listReadLinks(database(), randomUUID()), [])
  assert.deepEqual(await listReadLinks(database(), subject), [marked])

  const unread = await setReadLink(database(), subject, 'article-a', {
    read: false,
    revision: marked.revision,
  })
  assert.equal(unread.readAt, null)
  assert.equal(unread.revision, 2)
  assert.deepEqual(await listReadLinks(database(), subject), [unread])
  assert.ok(!('subject' in unread))
})

test('read retries are idempotent and old read requests cannot reverse an unread mark', async () => {
  const subject = randomUUID()
  const write = { read: true, revision: 0 }
  const marked = await setReadLink(database(), subject, 'article-a', write)
  assert.deepEqual(await setReadLink(database(), subject, 'article-a', write), marked)
  const unread = await setReadLink(database(), subject, 'article-a', { read: false, revision: 1 })
  assert.deepEqual(
    await setReadLink(database(), subject, 'article-a', { read: false, revision: 1 }),
    unread,
  )
  await assert.rejects(setReadLink(database(), subject, 'article-a', write), error => {
    assert.ok(error instanceof ArenaReaderConflictError)
    assert.deepEqual(error.current, unread)
    return true
  })
  const reread = await setReadLink(database(), subject, 'article-a', { read: true, revision: 2 })
  await assert.rejects(
    setReadLink(database(), subject, 'article-a', write),
    ArenaReaderConflictError,
  )
  assert.deepEqual(await listReadLinks(database(), subject), [reread])
})

test('an explicit unread record blocks a stale first-read request', async () => {
  const subject = randomUUID()
  const unread = await setReadLink(database(), subject, 'article-a', { read: false, revision: 0 })
  assert.equal(unread.readAt, null)
  assert.equal(unread.revision, 1)
  await assert.rejects(
    setReadLink(database(), subject, 'article-a', { read: true, revision: 0 }),
    ArenaReaderConflictError,
  )
  await assert.rejects(
    setReadLink(database(), subject, 'unknown-record', { read: true, revision: 4 }),
    error => {
      assert.ok(error instanceof ArenaReaderConflictError)
      assert.equal(error.current, null)
      return true
    },
  )
})

test('note create and update retries retain one row, source anchors, and the read state', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const create = input({
    snapshotId: 'snapshot-one',
    quote: { exact: 'a quoted passage', prefix: 'before ', suffix: ' after' },
    occurrence: { channelSlug: 'readings', blockId: 'block-54' },
  })
  const created = await saveArenaNote(database(), subject, id, create)
  assert.equal(created.id, id)
  assert.equal(created.revision, 1)
  assert.equal(created.readyRevision, null)
  assert.equal(created.exportedRevision, null)
  assert.deepEqual(created.quote, create.quote)
  assert.deepEqual(created.occurrence, create.occurrence)
  assert.equal(created.snapshotId, 'snapshot-one')
  assert.deepEqual(await saveArenaNote(database(), subject, id, create), created)
  const update = edit(created, { body: 'A longer note with a second sentence.' })
  const updated = await saveArenaNote(database(), subject, id, update)
  assert.equal(updated.revision, 2)
  assert.equal(updated.createdAt, created.createdAt)
  assert.deepEqual(await saveArenaNote(database(), subject, id, update), updated)
  assert.deepEqual(await listArenaNotes(database(), subject, 'article-a'), [updated])
  assert.deepEqual(await listArenaNotes(database(), subject, 'another-article'), [])
  assert.deepEqual(await listReadLinks(database(), subject), [])
  assert.ok(!('subject' in updated))
})

test('note identities and content are isolated by owner', async () => {
  const subject = randomUUID()
  const otherSubject = randomUUID()
  const id = randomUUID()
  const created = await saveArenaNote(database(), subject, id, input())
  assert.deepEqual(await listArenaNotes(database(), otherSubject), [])
  await assert.rejects(
    saveArenaNote(database(), otherSubject, id, edit(created, { body: 'Overwrite attempt' })),
    error => {
      assert.ok(error instanceof ArenaReaderConflictError)
      assert.equal(error.current, null)
      return true
    },
  )
  await assert.rejects(
    deleteArenaNote(database(), otherSubject, id, created.revision),
    ArenaReaderNoteNotFoundError,
  )
  const separate = await saveArenaNote(database(), otherSubject, id, input({ body: 'Other owner' }))
  assert.deepEqual(await listArenaNotes(database(), subject), [created])
  assert.deepEqual(await listArenaNotes(database(), otherSubject), [separate])
})

test('the conditional note write rejects conflicting creations and simultaneous edits', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const created = await saveArenaNote(database(), subject, id, input())
  await assert.rejects(
    saveArenaNote(database(), subject, id, input({ body: 'Different creation' })),
    ArenaReaderConflictError,
  )
  const outcomes = await Promise.allSettled([
    saveArenaNote(database(), subject, id, edit(created, { body: 'Written on the phone' })),
    saveArenaNote(database(), subject, id, edit(created, { body: 'Written on the laptop' })),
  ])
  const successes = outcomes.filter(result => result.status === 'fulfilled')
  const failures = outcomes.filter(result => result.status === 'rejected')
  assert.equal(successes.length, 1)
  assert.equal(failures.length, 1)
  const current = successes[0].value
  assert.equal(current.revision, 2)
  const error: unknown = failures[0].reason
  assert.ok(error instanceof ArenaReaderConflictError)
  assert.deepEqual(error.current, current)
  assert.deepEqual(await listArenaNotes(database(), subject), [current])
})

test('an existing note cannot be moved to a different source or article through an edit', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const created = await saveArenaNote(database(), subject, id, input())
  for (const change of [
    { articleId: 'article-b' },
    { sourceUrl: 'https://example.com/different' },
  ]) {
    await assert.rejects(
      saveArenaNote(database(), subject, id, edit(created, change)),
      ArenaReaderConflictError,
    )
  }
  assert.deepEqual(await listArenaNotes(database(), subject), [created])
})

test('deletion keeps a tombstone and rejects stale edits or resurrection retries', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const created = await saveArenaNote(database(), subject, id, input({ ready: true }))
  const deleted = await deleteArenaNote(database(), subject, id, created.revision)
  assert.equal(deleted.revision, 2)
  assert.equal(deleted.readyRevision, null)
  assert.equal(typeof deleted.deletedAt, 'number')
  assert.equal(deleted.body, created.body)
  assert.deepEqual(await deleteArenaNote(database(), subject, id, created.revision), deleted)
  assert.deepEqual(await listArenaNotes(database(), subject), [deleted])
  assert.deepEqual((await exportReadyArenaNotes(database(), subject)).notes, [])
  await assert.rejects(
    saveArenaNote(database(), subject, id, edit(created, { body: 'An old offline edit' })),
    ArenaReaderConflictError,
  )
  await assert.rejects(saveArenaNote(database(), subject, id, input()), ArenaReaderConflictError)
  await assert.rejects(
    saveArenaNote(database(), subject, id, edit(deleted)),
    ArenaReaderConflictError,
  )
})

test('deleting with a stale revision preserves the newer note', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const created = await saveArenaNote(database(), subject, id, input())
  const updated = await saveArenaNote(database(), subject, id, edit(created, { body: 'New text' }))
  await assert.rejects(
    deleteArenaNote(database(), subject, id, created.revision),
    ArenaReaderConflictError,
  )
  assert.deepEqual(await listArenaNotes(database(), subject), [updated])
})

test('Ready selects a revision and later edits require selecting the edited revision again', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const created = await saveArenaNote(database(), subject, id, input())
  assert.deepEqual((await exportReadyArenaNotes(database(), subject)).notes, [])
  const ready = await saveArenaNote(database(), subject, id, edit(created, { ready: true }))
  const bundle = await exportReadyArenaNotes(database(), subject)
  assert.equal(bundle.schemaVersion, 1)
  assert.equal(ready.readyRevision, ready.revision)
  assert.equal(bundle.notes.length, 1)
  assert.equal(bundle.notes[0].revision, ready.revision)
  assert.equal(bundle.notes[0].bodyHash, createHash('sha256').update(ready.body).digest('hex'))
  assert.deepEqual((await exportReadyArenaNotes(database(), randomUUID())).notes, [])

  const edited = await saveArenaNote(database(), subject, id, edit(ready, { body: 'Edited again' }))
  assert.equal(edited.readyRevision, null)
  assert.deepEqual((await exportReadyArenaNotes(database(), subject)).notes, [])
})

test('export acknowledgements are idempotent and an older receipt leaves a newer Ready revision', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const firstReady = await saveArenaNote(database(), subject, id, input({ ready: true }))
  await exportReadyArenaNotes(database(), subject)
  const edited = await saveArenaNote(database(), subject, id, edit(firstReady, { body: 'Updated' }))
  const secondReady = await saveArenaNote(database(), subject, id, edit(edited, { ready: true }))
  const firstReceipt = { revision: firstReady.revision, receipt: 'backfill-one:hash:target' }
  const acknowledged = await acknowledgeArenaNoteExport(database(), subject, id, firstReceipt)
  assert.equal(acknowledged.revision, secondReady.revision)
  assert.equal(acknowledged.body, secondReady.body)
  assert.equal(acknowledged.readyRevision, secondReady.revision)
  assert.equal(acknowledged.exportedRevision, firstReady.revision)
  assert.deepEqual(
    await acknowledgeArenaNoteExport(database(), subject, id, firstReceipt),
    acknowledged,
  )
  assert.equal(
    (await exportReadyArenaNotes(database(), subject)).notes[0].revision,
    secondReady.revision,
  )

  const secondReceipt = { revision: secondReady.revision, receipt: 'backfill-two:hash:target' }
  const finished = await acknowledgeArenaNoteExport(database(), subject, id, secondReceipt)
  assert.equal(finished.readyRevision, null)
  assert.equal(finished.exportedRevision, secondReady.revision)
  assert.deepEqual((await exportReadyArenaNotes(database(), subject)).notes, [])
  await assert.rejects(
    acknowledgeArenaNoteExport(database(), subject, id, firstReceipt),
    ArenaReaderConflictError,
  )
  await assert.rejects(
    acknowledgeArenaNoteExport(database(), subject, id, {
      ...secondReceipt,
      receipt: 'conflicting-receipt',
    }),
    ArenaReaderConflictError,
  )
})

test('export acknowledgement rejects another owner, an unready current revision, and future revisions', async () => {
  const subject = randomUUID()
  const id = randomUUID()
  const created = await saveArenaNote(database(), subject, id, input())
  const acknowledgement = { revision: created.revision, receipt: 'receipt' }
  await assert.rejects(
    acknowledgeArenaNoteExport(database(), randomUUID(), id, acknowledgement),
    ArenaReaderNoteNotFoundError,
  )
  await assert.rejects(
    acknowledgeArenaNoteExport(database(), subject, id, acknowledgement),
    ArenaReaderConflictError,
  )
  await assert.rejects(
    acknowledgeArenaNoteExport(database(), subject, id, { revision: 50, receipt: 'future' }),
    ArenaReaderConflictError,
  )
})

test('database failures propagate instead of becoming empty history or acknowledged writes', async () => {
  assert.ok(proxy)
  const db = proxy.env.UNMIGRATED_READER
  await assert.rejects(listReadLinks(db, randomUUID()))
  await assert.rejects(listArenaNotes(db, randomUUID()))
  await assert.rejects(setReadLink(db, randomUUID(), 'article-a', { read: true, revision: 0 }))
  await assert.rejects(saveArenaNote(db, randomUUID(), randomUUID(), input()))
})
