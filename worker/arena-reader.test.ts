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
import { mergeCuriusFeed, CURIUS_FEED_URL, type CuriusSavedLink } from '../quartz/util/curius-feed'
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

function entry(
  name: string,
  later: boolean,
  blockId: string,
  sourceUrl = `https://example.com/${name}`,
): ArenaFeedEntry {
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
const arxiv = entry('arxiv-abstract', false, 'block-4', 'https://arxiv.org/abs/2206.00759v3')
const arxivHtml: ArenaFeedEntry = {
  ...entry('arxiv-html', false, 'block-5', 'https://arxiv.org/html/2206.00759v3'),
  kind: 'pdf',
}
const entries = [ordinary, first, second, arxiv, arxivHtml]
const video: ArenaFeedEntry = { ...entry('video', true, 'block-6'), kind: 'video' }
const youtube = entry('youtube-channel', true, 'block-7', 'https://youtube.com/@creator')
const directVideo = entry('video-file', true, 'block-8', 'https://example.com/film.mp4')
const malformedYoutube = entry(
  'youtube-typo',
  true,
  'block-10',
  'https://www.youtube.com87/watch?v=qX6NztnPU-4',
)
const videoChannel: ArenaFeedEntry = {
  ...entry('recorded-talk', true, 'block-9'),
  occurrences: [{ ...first.occurrences[0], blockId: 'block-9', channelSlug: 'video' }],
}
const catalogueEntries = [...entries, video, youtube, directVideo, videoChannel, malformedYoutube]
const manifest: ArenaFeedManifest = {
  schemaVersion: 1,
  revision: `feed-v1-${createHash('sha256').update(JSON.stringify(catalogueEntries)).digest('hex')}`,
  entries: catalogueEntries,
}
const curiusLinks: CuriusSavedLink[] = [
  {
    id: 236997,
    link: `${first.sourceUrl}?curius=3584`,
    title: 'Duplicate title',
    toRead: null,
    createdDate: '2026-09-15T00:09:31.811Z',
  },
  { id: 236998, link: 'https://example.com/curius-paper.pdf', title: 'Curius paper', toRead: true },
  {
    id: 236999,
    link: 'https://example.com/curius-paper.pdf?utm_source=curius',
    title: 'Same paper',
  },
  { id: 237000, link: 'https://www.youtube.com/watch?v=NB1PNDN24cE', title: 'A video' },
]
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
  quality: 'complete',
  diagnostics: [],
}
const arxivSnapshot: ArenaReaderArtifact = {
  ...snapshot,
  articleId: arxiv.articleId,
  snapshotId: randomUUID(),
  title: 'Interpretability Guarantees with Merlin-Arthur Classifiers',
  sourceUrl: arxiv.sourceUrl,
  finalUrl: arxiv.sourceUrl,
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
    path.join(assetDirectory, 'static', 'curius.json'),
    JSON.stringify({ links: curiusLinks }),
  )
  await writeFile(
    path.join(assetDirectory, 'static', 'curius-invalid.json'),
    JSON.stringify({ error: 'Provider unavailable' }),
  )
  await writeFile(
    path.join(assetDirectory, 'arena', 'feed.html'),
    '<!doctype html><html><head><title>Arena reader fixture</title></head><body><main>Arena reader shell</main></body></html>',
  )
  const migration = await readFile(
    new URL('../migrations/arena-reader/0000_arena_reader.sql', import.meta.url),
    'utf8',
  )
  const routerPath = fileURLToPath(new URL('./arena-reader.ts', import.meta.url))
  const cataloguePath = fileURLToPath(new URL('./arena-reader-catalogue.ts', import.meta.url))
  const main = path.join(directory, 'entry.ts')
  await writeFile(
    main,
    `import { handleArenaReaderRequest } from ${JSON.stringify(routerPath)}
import { loadCuriusFeedLinks, CURIUS_FEED_CACHE_KEY } from ${JSON.stringify(cataloguePath)}
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
      await env.ARENA_CONTENT.put(CURIUS_FEED_CACHE_KEY, JSON.stringify({ schemaVersion: 1, fetchedAt: Date.now(), retryAt: 0, links: [] }))
      await env.ARENA_CONTENT.put(${JSON.stringify(arenaReaderSnapshotKey(first.articleId, snapshot.snapshotId))}, ${JSON.stringify(JSON.stringify(snapshot))})
      await env.ARENA_CONTENT.put(${JSON.stringify(arenaReaderStateKey(first.articleId))}, ${JSON.stringify(JSON.stringify({ schemaVersion: 1, generation: 1, snapshotId: snapshot.snapshotId, lease: null, failure: null }))})
      return Response.json({ reset: true })
    }
    if (pathname === '/__test/curius-cache') {
      const mode = new URL(request.url).searchParams.get('mode')
      if (mode === 'delete') await env.ARENA_CONTENT.delete(CURIUS_FEED_CACHE_KEY)
      else if (mode === 'expire') {
        const stored = await env.ARENA_CONTENT.get(CURIUS_FEED_CACHE_KEY)
        await env.ARENA_CONTENT.put(CURIUS_FEED_CACHE_KEY, JSON.stringify({ ...await stored.json(), fetchedAt: 0, retryAt: 0 }))
      } else await env.ARENA_CONTENT.put(CURIUS_FEED_CACHE_KEY, JSON.stringify({ schemaVersion: 1, fetchedAt: Date.now(), retryAt: 0, links: ${JSON.stringify(curiusLinks)} }))
      return Response.json({ ready: true })
    }
    if (pathname === '/__test/curius-refresh') {
      const url = new URL(request.url)
      const source = url.searchParams.get('source')
      const asset = source === 'invalid' ? 'curius-invalid' : source === 'missing' ? 'missing-curius' : 'curius'
      let requests = 0
      let upstreamUrl = null
      try {
        const links = await loadCuriusFeedLinks(env.ARENA_CONTENT, url.searchParams.get('refresh') !== 'false', (input, init) => {
          requests++
          upstreamUrl = String(input)
          return env.ASSETS.fetch(new Request('https://aarnphm.xyz/static/' + asset + '.json', init))
        })
        const stored = await env.ARENA_CONTENT.get(CURIUS_FEED_CACHE_KEY)
        return Response.json({ links, requests, upstreamUrl, cache: await stored.json() })
      } catch (error) {
        return Response.json({ message: error.message, cause: error.cause?.message, requests, upstreamUrl }, { status: 503 })
      }
    }
    if (pathname === '/__test/cache-arxiv') {
      await env.ARENA_CONTENT.put(${JSON.stringify(arenaReaderSnapshotKey(arxiv.articleId, arxivSnapshot.snapshotId))}, ${JSON.stringify(JSON.stringify(arxivSnapshot))})
      await env.ARENA_CONTENT.put(${JSON.stringify(arenaReaderStateKey(arxiv.articleId))}, JSON.stringify({
        schemaVersion: 1,
        generation: 1,
        snapshotId: ${JSON.stringify(arxivSnapshot.snapshotId)},
        lease: null,
        failure: { reason: 'timeout', message: 'The abstract refresh timed out.', retryAt: Date.now() + 600_000 },
      }))
      return Response.json({ cached: true })
    }
    if (pathname === '/__test/counts') {
      const read = await env.ARENA_READER.prepare('SELECT count(*) AS count FROM arena_read_links').first()
      const notes = await env.ARENA_READER.prepare('SELECT count(*) AS count FROM arena_notes').first()
      const cached = await env.ARENA_CONTENT.list()
      return Response.json({ read: read.count, notes: notes.count, cached: cached.objects.filter(object => object.key !== CURIUS_FEED_CACHE_KEY).length })
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
          rules: [{ type: 'Text', globs: ['defuddle/full', '**/purify.js'], fallthrough: true }],
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

test('authenticated feed excludes video sources and places Later reading entries first', async () => {
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
    queue.map(item => item.articleId).sort(),
    entries.map(item => item.articleId).sort(),
  )
  assert.deepEqual(
    queue.map(item => item.later),
    [true, true, false, false, false],
  )
  assert.ok(queue.some(item => item.articleId === ordinary.articleId))
  const repeated = await responseBody(await fetchReader('/api/arena/feed?seed=fixture'))
  assert.deepEqual(repeated.entries, body.entries)
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 2 })
})

test('Curius shares the feed, existing read marks, note storage and PDF capture after deduplication', async () => {
  await mutate(`/api/arena/articles/${first.articleId}/read`, { read: true, revision: 0 })
  await (await harness().fetch(`${origin}/__test/curius-cache`)).text()
  const merged = await mergeCuriusFeed(manifest, curiusLinks, origin)
  const body = await responseBody(await fetchReader('/api/arena/feed?seed=curius'))
  assert.equal(body.revision, merged.revision)
  assert.ok(Array.isArray(body.entries))
  const queue = body.entries.map(record)
  assert.equal(queue.length, entries.length + 1)
  const duplicate = queue.filter(item => item.articleId === first.articleId)
  assert.equal(duplicate.length, 1)
  assert.equal(duplicate[0].title, first.title)
  assert.deepEqual(duplicate[0].occurrences, first.occurrences)
  assert.deepEqual(duplicate[0].curius, [{ userId: 3584, linkId: 236997 }])
  assert.ok(Array.isArray(body.readLinks))
  assert.equal(record(body.readLinks[0]).articleId, first.articleId)
  const paper = queue.find(item => item.title === 'Curius paper')
  assert.ok(paper)
  assert.equal(paper.later, true)
  assert.deepEqual(paper.occurrences, [])
  assert.deepEqual(paper.curius, [
    { userId: 3584, linkId: 236998 },
    { userId: 3584, linkId: 236999 },
  ])

  const rendered = await responseBody(
    await mutate(`/api/arena/articles/${paper.articleId}/render`, {}, {}, 'POST'),
  )
  assert.equal(rendered.status, 'ready')
  assert.equal(record(rendered.artifact).kind, 'pdf')
  const reopened = await responseBody(
    await mutate(`/api/arena/articles/${paper.articleId}/render`, {}, {}, 'POST'),
  )
  assert.equal(reopened.cached, true)
  const noteId = randomUUID()
  const saved = await mutate(`/api/arena/notes/${noteId}`, {
    articleId: paper.articleId,
    body: 'A note on a Curius link.',
    revision: 0,
    ready: true,
  })
  assert.equal(saved.status, 200)
  const note = record((await responseBody(saved)).note)
  assert.equal(note.occurrence, null)
  assert.equal(note.sourceUrl, 'https://example.com/curius-paper.pdf')
  const marked = await mutate(`/api/arena/articles/${paper.articleId}/read`, {
    read: true,
    revision: 0,
  })
  assert.equal(marked.status, 200)
  const fresh = await responseBody(await fetchReader('/api/arena/feed'))
  assert.ok(Array.isArray(fresh.readLinks))
  assert.equal(fresh.readLinks.length, 2)
  const restored = await responseBody(
    await fetchReader(`/api/arena/articles/${paper.articleId}/notes`),
  )
  assert.deepEqual(restored.notes, [note])
})

test('the complete Curius index persists in R2 and a fresh cache avoids another provider request', async () => {
  await (await harness().fetch(`${origin}/__test/curius-cache?mode=delete`)).text()
  const response = await harness().fetch(`${origin}/__test/curius-refresh`)
  const firstLoad = await responseBody(response)
  assert.equal(response.status, 200, JSON.stringify(firstLoad))
  assert.equal(firstLoad.upstreamUrl, CURIUS_FEED_URL)
  assert.equal(firstLoad.requests, 1)
  assert.deepEqual(firstLoad.links, curiusLinks)
  assert.deepEqual(record(firstLoad.cache).links, curiusLinks)
  const fresh = await responseBody(
    await harness().fetch(`${origin}/__test/curius-refresh?source=missing`),
  )
  assert.equal(fresh.requests, 0)
  assert.deepEqual(fresh.links, curiusLinks)
})

test('Curius failures retain the saved catalogue with a retry cooldown, while initial failures stay explicit', async () => {
  for (const source of ['missing', 'invalid']) {
    await (await harness().fetch(`${origin}/__test/curius-cache`)).text()
    await (await harness().fetch(`${origin}/__test/curius-cache?mode=expire`)).text()
    const reading = await responseBody(
      await harness().fetch(`${origin}/__test/curius-refresh?refresh=false&source=${source}`),
    )
    assert.equal(
      reading.requests,
      0,
      'reading uses the same stored index even after its refresh interval',
    )
    const stale = await responseBody(
      await harness().fetch(`${origin}/__test/curius-refresh?source=${source}`),
    )
    assert.equal(stale.requests, 1)
    assert.deepEqual(stale.links, curiusLinks)
    assert.ok(Number(record(stale.cache).retryAt) > Date.now())
    const repeated = await responseBody(
      await harness().fetch(`${origin}/__test/curius-refresh?source=${source}`),
    )
    assert.equal(repeated.requests, 0)
    await (await harness().fetch(`${origin}/__test/curius-cache?mode=delete`)).text()
    const initial = await harness().fetch(`${origin}/__test/curius-refresh?source=${source}`)
    assert.equal(initial.status, 503)
    assert.match(String((await responseBody(initial)).message), /Curius links could not be loaded/)
  }
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

test('arXiv abstract and HTML links open versioned PDFs without a Browser binding', async () => {
  for (const paper of [arxiv, arxivHtml]) {
    const route = `/api/arena/articles/${paper.articleId}/render`
    const response = await mutate(route, {}, {}, 'POST')
    assert.equal(response.status, 200)
    const opened = await responseBody(response)
    assert.equal(opened.status, 'ready')
    assert.equal(opened.cached, false)
    const artifact = record(opened.artifact)
    assert.equal(artifact.kind, 'pdf')
    assert.equal(artifact.articleId, paper.articleId)
    assert.equal(artifact.sourceUrl, paper.sourceUrl)
    assert.equal(artifact.finalUrl, 'https://arxiv.org/pdf/2206.00759v3')
    assert.ok(Array.isArray(artifact.resources))
    assert.equal(artifact.resources.length, 1)
    const resource = record(artifact.resources[0])
    assert.equal(resource.kind, 'pdf')
    assert.equal(resource.contentType, 'application/pdf')
    assert.equal(resource.id, artifact.resourceId)
    assert.ok(typeof resource.url === 'string')
    const resourceUrl = new URL(resource.url, origin)
    assert.equal(resourceUrl.origin, origin)
    assert.equal(
      resourceUrl.pathname,
      `/api/arena/articles/${paper.articleId}/snapshots/${artifact.snapshotId}/resources/${resource.id}`,
    )
    assert.ok(resourceUrl.searchParams.has('token'))
    const reopened = await responseBody(await mutate(route, {}, {}, 'POST'))
    assert.equal(reopened.cached, true)
    assert.equal(record(reopened.artifact).snapshotId, artifact.snapshotId)
  }
  assert.deepEqual(await counts(), { read: 0, notes: 0, cached: 6 })
})

test('a cached arXiv abstract switches to PDF while preserving its quoted notes and read mark', async () => {
  const seed = await harness().fetch(`${origin}/__test/cache-arxiv`)
  assert.equal(seed.status, 200)
  await seed.text()
  const quote = { exact: 'A fixture quote.', prefix: '', suffix: '' }
  const note = await mutate(
    `/api/arena/notes/${randomUUID()}`,
    noteInput({ articleId: arxiv.articleId, snapshotId: arxivSnapshot.snapshotId, quote }),
  )
  assert.equal(note.status, 200)
  const savedNote = record((await responseBody(note)).note)
  const read = await mutate(`/api/arena/articles/${arxiv.articleId}/read`, {
    read: true,
    revision: 0,
  })
  assert.equal(read.status, 200)
  const readLink = record((await responseBody(read)).readLink)

  const route = `/api/arena/articles/${arxiv.articleId}/render`
  const response = await mutate(route, { refresh: false }, {}, 'POST')
  assert.equal(response.status, 200)
  const opened = await responseBody(response)
  assert.equal(opened.status, 'ready')
  assert.equal(opened.cached, false)
  const artifact = record(opened.artifact)
  assert.equal(artifact.kind, 'pdf')
  assert.equal(artifact.title, arxivSnapshot.title)
  assert.equal(artifact.sourceUrl, arxiv.sourceUrl)
  assert.equal(artifact.finalUrl, 'https://arxiv.org/pdf/2206.00759v3')
  assert.notEqual(artifact.snapshotId, arxivSnapshot.snapshotId)
  const reopened = await responseBody(await mutate(route, {}, {}, 'POST'))
  assert.equal(reopened.cached, true)
  assert.equal(record(reopened.artifact).snapshotId, artifact.snapshotId)
  const status = await responseBody(
    await fetchReader(`/api/arena/articles/${arxiv.articleId}/render-status`),
  )
  assert.equal(record(status.artifact).snapshotId, artifact.snapshotId)

  const historical = await fetchReader(
    `/api/arena/articles/${arxiv.articleId}/snapshots/${arxivSnapshot.snapshotId}`,
  )
  assert.equal(historical.status, 200)
  const original = record((await responseBody(historical)).artifact)
  assert.equal(original.kind, 'html')
  assert.equal(original.readerHtml, '<p>A fixture quote.</p>')
  assert.deepEqual(
    (await responseBody(await fetchReader(`/api/arena/articles/${arxiv.articleId}/notes`))).notes,
    [savedNote],
  )
  assert.deepEqual((await responseBody(await fetchReader('/api/arena/feed'))).readLinks, [readLink])
  assert.deepEqual(await counts(), { read: 1, notes: 1, cached: 5 })
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
