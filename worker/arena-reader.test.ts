import type { TestHarness } from 'wrangler'
import assert from 'node:assert/strict'
import { createHash, randomUUID } from 'node:crypto'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, beforeEach, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createTestHarness } from 'wrangler'
import type { ArenaFeedEntry, ArenaFeedManifest } from '../quartz/util/arena-feed'
import type { ArenaReaderArtifact } from '../quartz/util/arena-reader'
import { createArenaReaderSessionCookie } from './arena-reader-auth'
import {
  ARENA_READER_PROFILE,
  arenaReaderSnapshotKey,
  arenaReaderStateKey,
} from './arena-reader-cache'
import { isRecord } from './type-guards'

const origin = 'https://aarnphm.xyz'
const ownerId = 12345
const subject = `github:${ownerId}`
const secret = 'arena-reader-integration-test-secret'

function entry(name: string, later: boolean, blockId: string): ArenaFeedEntry {
  const sourceUrl = `https://example.com/${name}`
  return {
    articleId: `article-v1-${createHash('sha256').update(`arena-article-v1\0${sourceUrl}`).digest('hex')}`,
    sourceUrl,
    title: name,
    kind: 'html',
    later,
    tags: ['fixture'],
    savedAt: null,
    occurrences: [
      {
        channelSlug: 'reading',
        channelName: 'Reading',
        blockId,
        parentBlockId: null,
        notesHtml: '<p>Saved context.</p>',
      },
    ],
  }
}

const first = entry('first-article', true, 'block-1')
const second = entry('second-article', true, 'block-2')
const ordinary: ArenaFeedEntry = { ...entry('ordinary-article', false, 'block-3'), kind: 'pdf' }
const entries = [ordinary, first, second]
const manifest: ArenaFeedManifest = {
  schemaVersion: 1,
  revision: `feed-v1-${createHash('sha256').update(JSON.stringify(entries)).digest('hex')}`,
  entries,
}
const snapshot: ArenaReaderArtifact = {
  schemaVersion: 1,
  articleId: first.articleId,
  snapshotId: randomUUID(),
  title: first.title,
  sourceUrl: first.sourceUrl,
  finalUrl: first.sourceUrl,
  capturedAt: Date.now(),
  profileVersion: ARENA_READER_PROFILE,
  fingerprint: createHash('sha256').update('A fixture quote.').digest('hex'),
  resources: [],
  kind: 'html',
  readerHtml: '<p>A fixture quote.</p>',
  documentHtml: '<main><p>A fixture quote.</p></main>',
  quality: 'complete',
  diagnostics: [],
}

let server: TestHarness | undefined
let directory: string | undefined
let cookie: string

function harness(): TestHarness {
  assert.ok(server)
  return server
}

before(async () => {
  directory = await mkdtemp(path.join(tmpdir(), 'arena-reader-router-'))
  const assetDirectory = path.join(directory, 'assets')
  await mkdir(path.join(assetDirectory, 'static'), { recursive: true })
  await mkdir(path.join(assetDirectory, 'arena'), { recursive: true })
  await writeFile(path.join(assetDirectory, 'static', 'arena-feed.json'), JSON.stringify(manifest))
  await writeFile(
    path.join(assetDirectory, 'arena', 'feed.html'),
    '<!doctype html><html><head><title>Arena reader fixture</title></head><body><main>Arena reader shell</main></body></html>',
  )
  const migration = await readFile(
    new URL('../migrations/arena-reader/0000_arena_reader.sql', import.meta.url),
    'utf8',
  )
  const routerPath = fileURLToPath(new URL('./arena-reader.ts', import.meta.url))
  const main = path.join(directory, 'entry.ts')
  await writeFile(
    main,
    `import { handleArenaReaderRequest } from ${JSON.stringify(routerPath)}
const migration = ${JSON.stringify(migration)}
export default {
  async fetch(request, env) {
    const pathname = new URL(request.url).pathname
    if (pathname === '/__test/setup') {
      for (const statement of migration.split('--> statement-breakpoint')) {
        await env.ARENA_READER.prepare(statement).run()
      }
      return Response.json({ ready: true, secretMatches: env.SESSION_SECRET === ${JSON.stringify(secret)} })
    }
    if (pathname === '/__test/reset') {
      await env.ARENA_READER.batch([
        env.ARENA_READER.prepare('DELETE FROM arena_read_links'),
        env.ARENA_READER.prepare('DELETE FROM arena_notes'),
      ])
      const objects = await env.ARENA_CONTENT.list()
      if (objects.objects.length) await env.ARENA_CONTENT.delete(objects.objects.map(object => object.key))
      await env.ARENA_CONTENT.put(${JSON.stringify(arenaReaderSnapshotKey(first.articleId, snapshot.snapshotId))}, ${JSON.stringify(JSON.stringify(snapshot))})
      await env.ARENA_CONTENT.put(${JSON.stringify(arenaReaderStateKey(first.articleId))}, ${JSON.stringify(JSON.stringify({ schemaVersion: 1, generation: 1, snapshotId: snapshot.snapshotId, lease: null, failure: null }))})
      return Response.json({ reset: true })
    }
    if (pathname === '/__test/counts') {
      const read = await env.ARENA_READER.prepare('SELECT count(*) AS count FROM arena_read_links').first()
      const notes = await env.ARENA_READER.prepare('SELECT count(*) AS count FROM arena_notes').first()
      const cached = await env.ARENA_CONTENT.list()
      return Response.json({ read: read.count, notes: notes.count, cached: cached.objects.length })
    }
    return await handleArenaReaderRequest(request, env) ?? new Response('Outside reader', { status: 404 })
  }
}`,
  )
  server = createTestHarness({
    root: directory,
    workers: [
      {
        config: {
          name: 'arena-reader-router-test',
          main,
          compatibility_date: '2025-01-21',
          compatibility_flags: ['nodejs_compat', 'global_fetch_strictly_public'],
          vars: { SESSION_SECRET: secret, ARENA_OWNER_LOGIN: 'aarnphm' },
          rules: [
            { type: 'Text', globs: ['**/Readability.js', '**/purify.js'], fallthrough: true },
          ],
          assets: { directory: assetDirectory, binding: 'ASSETS', run_worker_first: true },
          d1_databases: [
            {
              binding: 'ARENA_READER',
              database_name: 'arena-reader-router-test',
              database_id: '00000000-0000-0000-0000-000000000003',
            },
          ],
          r2_buckets: [{ binding: 'ARENA_CONTENT', bucket_name: 'arena-reader-router-test' }],
        },
      },
    ],
  })
  await server.listen()
  const setup = await server.fetch(`${origin}/__test/setup`)
  assert.equal(setup.status, 200)
  const setupBody = await responseBody(setup)
  assert.equal(setupBody.secretMatches, true, JSON.stringify(setupBody))
  const session = await createArenaReaderSessionCookie(
    new Request(`${origin}/comments/github/callback`),
    { SESSION_SECRET: secret },
    { id: ownerId, login: 'aarnphm' },
  )
  assert.ok(session)
  cookie = session.split(';')[0]
})

beforeEach(async () => {
  const response = await harness().fetch(`${origin}/__test/reset`)
  assert.equal(response.status, 200)
  await response.text()
})

after(async () => {
  await server?.close()
  if (directory) await rm(directory, { recursive: true, force: true })
})

function fetchReader(url: string, init: RequestInit = {}, authenticated = true) {
  const headers = new Headers(init.headers)
  if (authenticated && !headers.has('Cookie')) headers.set('Cookie', cookie)
  // The development proxy rewrites HTTPS to HTTP; direct dispatch preserves session origins.
  return harness()
    .getWorker()
    .fetch(`${origin}${url}`, { redirect: 'manual', ...init, headers })
}

function mutate(
  url: string,
  body: unknown,
  overrides: HeadersInit = {},
  method = 'PUT',
  authenticated = true,
) {
  const headers = new Headers({
    'Content-Type': 'application/json',
    Origin: origin,
    'X-Arena-Subject': subject,
  })
  new Headers(overrides).forEach((value, key) => headers.set(key, value))
  return fetchReader(url, { method, body: JSON.stringify(body), headers }, authenticated)
}

function record(value: unknown): Record<string, unknown> {
  assert.ok(isRecord(value))
  return value
}

async function responseBody(response: { json(): Promise<unknown> }) {
  return record(await response.json())
}

async function counts() {
  return responseBody(await harness().fetch(`${origin}/__test/counts`))
}

function noteInput(overrides: Record<string, unknown> = {}) {
  return { articleId: first.articleId, body: 'A reader note.', revision: 0, ...overrides }
}

test('authenticated feed loads the real catalogue asset and places Later entries first', async () => {
  const response = await fetchReader('/api/arena/feed?seed=fixture')
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('Cache-Control'), 'private, no-store')
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), null)
  assert.equal(response.headers.get('X-Content-Type-Options'), 'nosniff')
  const body = await responseBody(response)
  assert.equal(body.subject, subject)
  assert.equal(body.revision, manifest.revision)
  assert.deepEqual(body.readLinks, [])
  assert.ok(Array.isArray(body.entries))
  const queue = body.entries.map(record)
  assert.deepEqual(
    queue.map(item => item.later),
    [true, true, false],
  )
  assert.equal(queue[2].articleId, ordinary.articleId)
  const repeated = await responseBody(await fetchReader('/api/arena/feed?seed=fixture'))
  assert.deepEqual(repeated.entries, body.entries)
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 2 })
})

test('the reader shell and its aliases require a verified owner session', async () => {
  for (const suffix of ['', '/', '.html', '/index.html']) {
    const response = await fetchReader(`/arena/feed${suffix}?article=${first.articleId}`, {}, false)
    assert.equal(response.status, 302)
    assert.equal(response.headers.get('Cache-Control'), 'private, no-store')
    const destination = new URL(response.headers.get('Location') ?? '')
    assert.equal(destination.origin, origin)
    assert.equal(destination.pathname, '/comments/github/login')
    await response.text()
  }
  const authenticated = await fetchReader('/arena/feed')
  assert.equal(authenticated.status, 200)
  assert.match(await authenticated.text(), /Arena reader shell/)
})

test('unsigned identity claims and unauthenticated mutations cannot write D1 or render', async () => {
  const before = await counts()
  const routes = [
    ['/api/arena/feed?login=aarnphm', 'GET'],
    [`/api/arena/articles/${first.articleId}/read`, 'PUT'],
    [`/api/arena/notes/${randomUUID()}`, 'PUT'],
    [`/api/arena/articles/${first.articleId}/render`, 'POST'],
  ]
  for (const [url, method] of routes) {
    const response = await fetchReader(
      url,
      {
        method,
        headers: {
          'X-Github-Login': 'aarnphm',
          'X-Arena-Subject': subject,
          Origin: origin,
          'Content-Type': 'application/json',
        },
        ...(method === 'GET' ? {} : { body: '{}' }),
      },
      false,
    )
    assert.equal(response.status, 401)
    assert.equal((await responseBody(response)).error, 'unauthorized')
  }
  assert.deepEqual(await counts(), before)
})

test('mutations reject another origin or an outdated owner without changing stored state', async () => {
  const url = `/api/arena/articles/${first.articleId}/read`
  for (const headers of [
    { Origin: 'https://elsewhere.example' },
    { Origin: '' },
    { 'X-Arena-Subject': 'github:54321' },
    { 'X-Arena-Subject': '' },
    { 'Sec-Fetch-Site': 'cross-site' },
  ]) {
    const response = await mutate(url, { read: true, revision: 0 }, headers)
    assert.equal(response.status, 403)
    assert.equal((await responseBody(response)).error, 'invalid-origin')
  }
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 2 })
})

test('invalid JSON, unknown fields, missing bodies, and oversized requests return precise errors', async () => {
  const url = `/api/arena/articles/${first.articleId}/read`
  const headers = { Origin: origin, 'X-Arena-Subject': subject, 'Content-Type': 'application/json' }
  for (const [body, status, error] of [
    ['{', 400, 'invalid-json'],
    ['null', 400, 'invalid-input'],
    [JSON.stringify({ read: true, revision: 0, subject: 'github:1' }), 400, 'invalid-input'],
    [' '.repeat(128 * 1024 + 1), 413, 'body-too-large'],
  ]) {
    assert.equal(typeof body, 'string')
    const response = await fetchReader(url, { method: 'PUT', headers, body: String(body) })
    assert.equal(response.status, status)
    assert.equal((await responseBody(response)).error, error)
  }
  const absent = await fetchReader(url, { method: 'PUT', headers })
  assert.equal(absent.status, 400)
  await absent.text()
  const wrongType = await mutate(url, { read: true, revision: 0 }, { 'Content-Type': 'text/plain' })
  assert.equal(wrongType.status, 415)
  await wrongType.text()
  const overlongNote = await mutate(
    `/api/arena/notes/${randomUUID()}`,
    noteInput({ body: 'x'.repeat(20_001) }),
  )
  assert.equal(overlongNote.status, 400)
  await overlongNote.text()
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 2 })
})

test('unknown article IDs and client-supplied source URLs cannot enter the reader', async () => {
  const unknown = `article-v1-${'f'.repeat(64)}`
  for (const operation of ['read', 'render']) {
    const response = await mutate(
      `/api/arena/articles/${unknown}/${operation}`,
      operation === 'read' ? { read: true, revision: 0 } : {},
      {},
      operation === 'read' ? 'PUT' : 'POST',
    )
    assert.equal(response.status, 404)
    assert.equal((await responseBody(response)).error, 'unknown-article')
  }
  const unknownNote = await mutate(
    `/api/arena/notes/${randomUUID()}`,
    noteInput({ articleId: unknown }),
  )
  assert.equal(unknownNote.status, 404)
  await unknownNote.text()
  const suppliedUrl = await mutate(
    `/api/arena/notes/${randomUUID()}`,
    noteInput({ sourceUrl: 'http://127.0.0.1/private' }),
  )
  assert.equal(suppliedUrl.status, 400)
  await suppliedUrl.text()
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 2 })
})

test('read and unread HTTP writes use persisted revisions and reject old retries', async () => {
  const url = `/api/arena/articles/${first.articleId}/read`
  const markedResponse = await mutate(url, { read: true, revision: 0 })
  assert.equal(markedResponse.status, 200)
  const marked = record((await responseBody(markedResponse)).readLink)
  assert.equal(marked.articleId, first.articleId)
  assert.equal(marked.revision, 1)
  assert.equal(typeof marked.readAt, 'number')
  const retry = await responseBody(await mutate(url, { read: true, revision: 0 }))
  assert.deepEqual(retry.readLink, marked)
  const unreadResponse = await mutate(url, { read: false, revision: 1 })
  assert.equal(unreadResponse.status, 200)
  const unread = record((await responseBody(unreadResponse)).readLink)
  assert.equal(unread.readAt, null)
  assert.equal(unread.revision, 2)
  const stale = await mutate(url, { read: true, revision: 0 })
  assert.equal(stale.status, 409)
  assert.deepEqual((await responseBody(stale)).current, unread)
  assert.deepEqual((await responseBody(await fetchReader('/api/arena/feed'))).readLinks, [unread])
})

test('quoted notes validate the saved snapshot and current catalogue occurrence', async () => {
  const quote = { exact: 'A fixture quote.', prefix: '', suffix: '' }
  const occurrence = { channelSlug: 'reading', blockId: 'block-1' }
  for (const [input, code] of [
    [noteInput({ quote }), 'missing-snapshot'],
    [noteInput({ snapshotId: randomUUID() }), 'invalid-snapshot'],
    [
      noteInput({ articleId: second.articleId, snapshotId: snapshot.snapshotId }),
      'invalid-snapshot',
    ],
    [noteInput({ occurrence: { ...occurrence, blockId: 'block-2' } }), 'invalid-occurrence'],
    [
      noteInput({ occurrence: { ...occurrence, channelSlug: 'another-channel' } }),
      'invalid-occurrence',
    ],
  ]) {
    const response = await mutate(`/api/arena/notes/${randomUUID()}`, input)
    assert.equal(response.status, 400)
    assert.equal((await responseBody(response)).error, code)
  }
  assert.equal((await counts()).notes, 0)
  const id = randomUUID()
  const input = noteInput({ quote, occurrence, snapshotId: snapshot.snapshotId })
  const response = await mutate(`/api/arena/notes/${id}`, input)
  assert.equal(response.status, 200)
  const saved = record((await responseBody(response)).note)
  assert.equal(saved.sourceUrl, first.sourceUrl)
  assert.deepEqual(saved.quote, quote)
  assert.deepEqual(saved.occurrence, occurrence)
  assert.equal(saved.snapshotId, snapshot.snapshotId)
  assert.deepEqual(
    (await responseBody(await fetchReader(`/api/arena/articles/${first.articleId}/notes`))).notes,
    [saved],
  )
  assert.deepEqual(
    (await responseBody(await fetchReader(`/api/arena/articles/${second.articleId}/notes`))).notes,
    [],
  )
  assert.equal((await counts()).read, 0)
})

test('note text retains Markdown indentation and trailing whitespace through the API', async () => {
  const body = '    const answer = 42\n\nA line with a hard break.  \n'
  const response = await mutate(`/api/arena/notes/${randomUUID()}`, noteInput({ body }))
  assert.equal(response.status, 200)
  assert.equal(record((await responseBody(response)).note).body, body)
  const blank = await mutate(`/api/arena/notes/${randomUUID()}`, noteInput({ body: ' \n\t ' }))
  assert.equal(blank.status, 400)
  await blank.text()
})

test('notes export a selected revision and an old receipt preserves newer ready text', async () => {
  const id = randomUUID()
  const url = `/api/arena/notes/${id}`
  const created = await mutate(url, noteInput({ ready: true }))
  assert.equal(created.status, 200)
  await created.text()
  const exportResponse = await fetchReader('/api/arena/notes?view=ready')
  assert.equal(exportResponse.status, 200)
  const bundle = await responseBody(exportResponse)
  assert.equal(bundle.schemaVersion, 1)
  assert.ok(Array.isArray(bundle.notes))
  assert.equal(bundle.notes.length, 1)
  const exported = record(bundle.notes[0])
  assert.equal(exported.id, id)
  assert.equal(exported.revision, 1)
  assert.equal(exported.bodyHash, createHash('sha256').update('A reader note.').digest('hex'))

  const updated = await mutate(url, noteInput({ revision: 1, body: 'A newer note.', ready: true }))
  assert.equal(updated.status, 200)
  await updated.text()
  const receipt = await mutate(
    `${url}/export-receipt`,
    { revision: 1, receipt: 'verified-file-write' },
    {},
    'POST',
  )
  assert.equal(receipt.status, 200)
  const acknowledged = record((await responseBody(receipt)).note)
  assert.equal(acknowledged.revision, 2)
  assert.equal(acknowledged.readyRevision, 2)
  assert.equal(acknowledged.exportedRevision, 1)
  assert.equal(acknowledged.body, 'A newer note.')
  const pending = await responseBody(await fetchReader('/api/arena/notes?view=ready'))
  assert.ok(Array.isArray(pending.notes))
  assert.equal(record(pending.notes[0]).revision, 2)
})

test('note deletion returns a tombstone and stale saves return the current tombstone', async () => {
  const id = randomUUID()
  const url = `/api/arena/notes/${id}`
  const created = await mutate(url, noteInput())
  assert.equal(created.status, 200)
  await created.text()
  const deletion = await mutate(url, { revision: 1 }, {}, 'DELETE')
  assert.equal(deletion.status, 200)
  const tombstone = record((await responseBody(deletion)).note)
  assert.equal(tombstone.revision, 2)
  assert.equal(typeof tombstone.deletedAt, 'number')
  const retried = await mutate(url, { revision: 1 }, {}, 'DELETE')
  assert.equal(retried.status, 200)
  assert.deepEqual((await responseBody(retried)).note, tombstone)
  const stale = await mutate(url, noteInput({ revision: 1, body: 'Offline draft' }))
  assert.equal(stale.status, 409)
  assert.deepEqual((await responseBody(stale)).current, tombstone)
  assert.deepEqual((await responseBody(await fetchReader('/api/arena/notes?view=inbox'))).notes, [
    tombstone,
  ])
})

test('a first PDF open creates a snapshot and only the second open reports a cache hit', async () => {
  const route = `/api/arena/articles/${ordinary.articleId}/render`
  const firstOpen = await responseBody(await mutate(route, { refresh: false }, {}, 'POST'))
  const secondOpen = await responseBody(await mutate(route, { refresh: false }, {}, 'POST'))
  assert.equal(firstOpen.status, 'ready')
  assert.equal(firstOpen.cached, false)
  assert.equal(record(firstOpen.artifact).kind, 'pdf')
  assert.equal(secondOpen.cached, true)
  assert.equal(record(firstOpen.artifact).snapshotId, record(secondOpen.artifact).snapshotId)
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 4 })
})

test('cached article opening and status use R2 without a Browser binding or a read mark', async () => {
  const render = await mutate(`/api/arena/articles/${first.articleId}/render`, {}, {}, 'POST')
  assert.equal(render.status, 200)
  const body = await responseBody(render)
  assert.equal(body.status, 'ready')
  assert.equal(body.cached, true)
  assert.equal(record(body.artifact).snapshotId, snapshot.snapshotId)
  const saved = await fetchReader(
    `/api/arena/articles/${first.articleId}/snapshots/${snapshot.snapshotId}`,
  )
  assert.equal(saved.status, 200)
  assert.equal(record((await responseBody(saved)).artifact).snapshotId, snapshot.snapshotId)
  const status = await fetchReader(`/api/arena/articles/${first.articleId}/render-status`)
  assert.equal(status.status, 200)
  assert.equal((await responseBody(status)).status, 'ready')
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 2 })
})

test('incorrect methods, invalid inbox views, and oversized shuffle seeds remain private errors', async () => {
  for (const [url, method, status] of [
    ['/api/arena/feed', 'POST', 405],
    ['/api/arena/feed?seed=' + 'a'.repeat(129), 'GET', 400],
    ['/api/arena/notes?view=unknown', 'GET', 400],
    [`/api/arena/articles/${first.articleId}/render`, 'GET', 405],
    ['/api/arena/unknown', 'GET', 404],
  ]) {
    const response = await fetchReader(String(url), { method: String(method) })
    assert.equal(response.status, status)
    assert.equal(response.headers.get('Cache-Control'), 'private, no-store')
    await response.text()
  }
  const outside = await fetchReader('/not-an-arena-route')
  assert.equal(outside.status, 404)
  assert.equal(await outside.text(), 'Outside reader')
})
