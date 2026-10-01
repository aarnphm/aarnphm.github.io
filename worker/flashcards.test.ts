import type { TestHarness } from 'wrangler'
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, beforeEach, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createTestHarness } from 'wrangler'
import { createArenaReaderSessionCookie } from './arena-reader-auth'
import { isRecord } from './type-guards'

const origin = 'https://aarnphm.xyz'
const secret = 'flashcards-integration-test-secret'
const deck = 'fr/les-nombres/flashcards'
const owner = { id: 12345, login: 'aarnphm' }

let server: TestHarness | undefined
let directory: string | undefined
let cookie: string

function harness(): TestHarness {
  assert.ok(server)
  return server
}

async function sessionCookie(
  callbackOrigin = origin,
  signingSecret = secret,
  now = Date.now(),
): Promise<string> {
  const session = await createArenaReaderSessionCookie(
    new Request(`${callbackOrigin}/comments/github/callback`),
    { SESSION_SECRET: signingSecret, ARENA_OWNER_LOGIN: ' AarnPhm ' },
    owner,
    now,
  )
  assert.ok(session)
  return session.split(';')[0]
}

before(async () => {
  directory = await mkdtemp(path.join(tmpdir(), 'flashcards-router-'))
  const migration = await readFile(
    new URL('../migrations/flashcards/0000_sticky_meltdown.sql', import.meta.url),
    'utf8',
  )
  const handlerPath = fileURLToPath(new URL('./flashcards.ts', import.meta.url))
  const main = path.join(directory, 'entry.ts')
  // Rows for two logins mirror production keys; the router only ever sees the owner session.
  await writeFile(
    main,
    `import { handleFlashcardsReview, handleFlashcardsState } from ${JSON.stringify(handlerPath)}
const migration = ${JSON.stringify(migration)}
const seed = (login, cardId) => ({ login, cardId, due: Date.now() + 86_400_000 })
export default {
  async fetch(request, env) {
    const pathname = new URL(request.url).pathname
    if (pathname === '/__test/setup') {
      for (const statement of migration.split('--> statement-breakpoint')) {
        await env.FLASHCARDS.prepare(statement).run()
      }
      return Response.json({ ready: true })
    }
    if (pathname === '/__test/reset') {
      const insert = 'INSERT INTO flashcard_reviews VALUES (?, ?, ?, 3.1, 5, ?, 2, 1, 0, 0, ?)'
      await env.FLASHCARDS.batch([
        env.FLASHCARDS.prepare('DELETE FROM flashcard_reviews'),
        ...[seed('aarnphm', 'owner-card'), seed('mallory', 'mallory-card')].map(row =>
          env.FLASHCARDS.prepare(insert).bind(row.login, row.cardId, ${JSON.stringify(deck)}, row.due, Date.now()),
        ),
      ])
      return Response.json({ reset: true })
    }
    if (pathname === '/__test/rows') {
      const { results } = await env.FLASHCARDS.prepare(
        'SELECT login, card_id AS cardId, reps FROM flashcard_reviews ORDER BY login, card_id',
      ).all()
      return Response.json(results)
    }
    if (pathname === '/api/flashcards/state') return handleFlashcardsState(request, env)
    if (pathname === '/api/flashcards/review') return handleFlashcardsReview(request, env)
    return new Response('not found', { status: 404 })
  }
}`,
  )
  server = createTestHarness({
    root: directory,
    workers: [
      {
        config: {
          name: 'flashcards-router-test',
          main,
          compatibility_date: '2025-01-21',
          compatibility_flags: ['nodejs_compat', 'global_fetch_strictly_public'],
          vars: { SESSION_SECRET: secret, ARENA_OWNER_LOGIN: ' AarnPhm ' },
          d1_databases: [
            {
              binding: 'FLASHCARDS',
              database_name: 'flashcards-router-test',
              database_id: '00000000-0000-0000-0000-000000000004',
            },
          ],
        },
      },
    ],
  })
  await server.listen()
  const setup = await server.fetch(`${origin}/__test/setup`)
  assert.equal(setup.status, 200)
  await setup.text()
  cookie = await sessionCookie()
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

// Direct dispatch keeps the https origin the session is bound to; the dev proxy rewrites it to http.
function call(url: string, headers: Record<string, string> = {}, body?: unknown) {
  return harness()
    .getWorker()
    .fetch(
      `${origin}${url}`,
      body === undefined
        ? { headers }
        : {
            method: 'POST',
            body: JSON.stringify(body),
            headers: { 'Content-Type': 'application/json', ...headers },
          },
    )
}

function review(body: unknown, headers: Record<string, string>) {
  return call('/api/flashcards/review', headers, body)
}

async function json(response: { json(): Promise<unknown> }): Promise<Record<string, unknown>> {
  const value = await response.json()
  assert.ok(isRecord(value))
  return value
}

async function rows() {
  return (await harness().fetch(`${origin}/__test/rows`)).json()
}

const seeded = [
  { login: 'aarnphm', cardId: 'owner-card', reps: 1 },
  { login: 'mallory', cardId: 'mallory-card', reps: 1 },
]

test('state ignores a requested login without an owner session', async () => {
  const sessions: Record<string, string>[] = [
    {},
    { Cookie: await sessionCookie(origin, 'a-different-signing-key') },
    { Cookie: await sessionCookie(origin, secret, Date.now() - 31 * 24 * 60 * 60 * 1000) },
    { Cookie: await sessionCookie('https://notes.aarnphm.xyz') },
  ]
  for (const headers of sessions) {
    for (const login of ['aarnphm', 'mallory']) {
      const response = await call(
        `/api/flashcards/state?deck=${encodeURIComponent(deck)}&login=${login}`,
        headers,
      )
      assert.equal(response.status, 200)
      assert.deepEqual(await json(response), { login: null, states: [] })
      assert.equal(response.headers.get('Cache-Control'), 'private, no-store')
    }
  }
})

test('owner session reads only the owner rows, whatever login the query names', async () => {
  const response = await call(
    `/api/flashcards/state?deck=${encodeURIComponent(deck)}&login=mallory`,
    { Cookie: cookie },
  )
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('Cache-Control'), 'private, no-store')
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), null)
  const body = await json(response)
  assert.equal(body.login, 'aarnphm')
  assert.ok(Array.isArray(body.states))
  assert.deepEqual(
    body.states.map(state => (isRecord(state) ? state.cardId : null)),
    ['owner-card'],
  )
})

test('reviews without an owner session or from another origin write nothing', async () => {
  const input = { cardId: 'owner-card', deckSlug: deck, grade: 3, login: 'aarnphm' }
  const anonymous = await review(input, { Origin: origin })
  assert.equal(anonymous.status, 401)
  await anonymous.text()
  const rejected: Record<string, string>[] = [
    { Cookie: cookie },
    { Cookie: cookie, Origin: 'https://evil.example' },
    { Cookie: cookie, Origin: 'https://notes.aarnphm.xyz' },
    { Cookie: cookie, Origin: origin, 'Sec-Fetch-Site': 'cross-site' },
  ]
  for (const headers of rejected) {
    const response = await review(input, headers)
    assert.equal(response.status, 403, JSON.stringify(headers))
    await response.text()
  }
  assert.deepEqual(await rows(), seeded)
})

test('owner reviews land under the owner login, whatever login the body names', async () => {
  const headers = { Cookie: cookie, Origin: origin, 'Sec-Fetch-Site': 'same-origin' }
  for (const cardId of ['owner-card', 'new-card']) {
    const response = await review({ cardId, deckSlug: deck, grade: 3, login: 'mallory' }, headers)
    assert.equal(response.status, 200)
    assert.equal(response.headers.get('Cache-Control'), 'private, no-store')
    const body = await json(response)
    assert.ok(isRecord(body.state))
    assert.equal(body.state.cardId, cardId)
    assert.ok(typeof body.state.due === 'number' && body.state.due > Date.now())
  }
  assert.deepEqual(await rows(), [
    { login: 'aarnphm', cardId: 'new-card', reps: 1 },
    { login: 'aarnphm', cardId: 'owner-card', reps: 2 },
    { login: 'mallory', cardId: 'mallory-card', reps: 1 },
  ])
})
