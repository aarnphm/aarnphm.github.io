import type { TestHarness, Unstable_RawConfig } from 'wrangler'
import assert from 'node:assert/strict'
import { randomBytes } from 'node:crypto'
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, test } from 'node:test'
import { createElement } from 'preact'
import renderToString from 'preact-render-to-string'
import { createTestHarness } from 'wrangler'
import { TrainingCalendar } from '../quartz/components/triathlon/calendar/TrainingCalendar'
import { hashTrainingPassword, sealTrainingCalendar } from '../quartz/util/training-calendar-crypto'
import { isRecord } from './type-guards'

// Failure modes: public HTML/asset leaks, alternate hosts and encoded paths, wrong passwords,
// CSRF, cookie forgery, week-long persistence, expired/revoked sessions, legacy storage,
// credential rotation, brute force, and
// missing keys or an old plaintext asset. Exercise the real Worker, assets, crypto, and DO.
const origin = 'https://t.aarnphm.xyz'
const password = 'test-only calendar passphrase 2026'
const secretPlan = 'PRIVATE-PRESCRIPTION-71d0bdb6'
const calendar = { version: 1, source: 'trainingpeaks', workouts: [{ title: secretPlan }] }
let server: TestHarness
let evidence: string
let config: Unstable_RawConfig
let counter = 0

before(async () => {
  evidence = await mkdtemp(path.join(tmpdir(), 'garden-training-access-'))
  const assets = path.join(evidence, 'assets')
  await mkdir(path.join(assets, 'static'), { recursive: true })
  const dataKey = randomBytes(32).toString('hex')
  const ciphertext = sealTrainingCalendar(calendar, dataKey)
  assert.ok(!ciphertext.includes(secretPlan))
  assert.equal(sealTrainingCalendar(calendar, undefined), 'null')
  await writeFile(path.join(assets, 'static/training-calendar.json'), ciphertext)
  const html = renderToString(
    createElement(TrainingCalendar, { id: 'test', embedded: false, panel: false }),
  )
  assert.ok(!html.includes('data-training-payload'))
  assert.ok(html.includes('type="password"'))
  assert.ok(
    html.includes('method="post"'),
    'a native form submission must never put the password in a URL',
  )
  await writeFile(path.join(assets, 'triathlon.html'), `<!doctype html>${html}`)
  await writeFile(
    path.join(evidence, 'README.md'),
    'Command: `pnpm test worker/training-calendar.test.ts`\n\nSynthetic calendar and encrypted assets only. The full Worker runs in workerd with real static assets and a SQLite Durable Object. Its temporary configuration root excludes local credential files. Requests save status, headers (session token values redacted), and response bodies. No provider data or production credentials are used.\n\nLifetime fixtures: a fresh login must set Max-Age=604800 and return a deadline seven days ahead; repeated reads must preserve that deadline. The final-day fixture sets the stored absolute expiry to 24 hours ahead and verifies session resumption and private data access. The legacy-storage fixture adds the former non-null idle-expiry column with an expired value, reloads the Worker, and verifies that the existing session and a new login both work. Expiry, logout, credential rotation, and rate limits remain covered.\n',
  )
  config = {
    name: 'training-calendar-test',
    main: path.resolve('worker/index.ts'),
    compatibility_date: '2025-01-21',
    compatibility_flags: ['nodejs_compat', 'global_fetch_strictly_public'],
    rules: [
      { type: 'Text', globs: ['**/*.txt', 'defuddle/full', '**/purify.js'], fallthrough: true },
    ],
    vars: {
      PUBLIC_BASE_URL: 'https://aarnphm.xyz',
      TRAINING_CALENDAR_PASSWORD_HASH: hashTrainingPassword(password),
      TRAINING_CALENDAR_DATA_KEY: dataKey,
    },
    assets: { directory: assets, binding: 'ASSETS', run_worker_first: true },
    kv_namespaces: [{ binding: 'OAUTH_KV', id: 'test-oauth' }],
    durable_objects: {
      bindings: [{ name: 'TRAINING_CALENDAR_ACCESS', class_name: 'TrainingCalendarAccess' }],
    },
    migrations: [{ tag: 'training-calendar-001', new_sqlite_classes: ['TrainingCalendarAccess'] }],
  }
  server = createTestHarness({ root: evidence, workers: [{ config }] })
  await server.listen()
})

after(async () => {
  await server?.close()
  console.log(`Training calendar E2E evidence: ${evidence}`)
})

async function request(route: string, init?: RequestInit, host = origin): Promise<Response> {
  const response = await server.getWorker('training-calendar-test').fetch(`${host}${route}`, init)
  const headers = Object.fromEntries(response.headers)
  if (headers['set-cookie'])
    headers['set-cookie'] = headers['set-cookie'].replace(/^([^=]+)=[^;]*/, '$1=[redacted]')
  await writeFile(
    path.join(evidence, `${++counter}.json`),
    JSON.stringify(
      {
        url: `${host}${route}`,
        method: init?.method ?? 'GET',
        status: response.status,
        headers,
        body: await response.clone().text(),
      },
      null,
      2,
    ),
  )
  return response
}

function mutation(passwordValue = password, ip = '192.0.2.1'): RequestInit {
  return {
    method: 'POST',
    headers: {
      Origin: origin,
      'Content-Type': 'application/json',
      'Sec-Fetch-Site': 'same-origin',
      'CF-Connecting-IP': ip,
    },
    body: JSON.stringify({ password: passwordValue }),
  }
}

async function login(ip = '192.0.2.1'): Promise<string> {
  const response = await request('/api/training-calendar/session', mutation(password, ip))
  assert.equal(response.status, 200)
  const cookie = response.headers.get('Set-Cookie')
  assert.ok(cookie)
  for (const attribute of [
    '__Host-TRAINING_CALENDAR=',
    'HttpOnly',
    'Secure',
    'SameSite=Strict',
    'Path=/',
  ])
    assert.ok(cookie.includes(attribute), attribute)
  assert.ok(!cookie.includes('Domain='))
  return cookie.split(';')[0]
}

test('anonymous requests and alternate asset paths never expose the calendar', async () => {
  for (const host of [origin, 'https://aarnphm.xyz', 'https://portfolio.example.workers.dev']) {
    for (const route of [
      '/api/training-calendar',
      '/static/training-calendar.json',
      '/static/%74raining-calendar.json',
      '/static//training-calendar.json',
      '/static/training-calendar.json?download=1',
    ]) {
      const response = await request(route, undefined, host)
      assert.ok([401, 404].includes(response.status), `${host}${route}: ${response.status}`)
      assert.ok(!(await response.text()).includes(secretPlan))
      assert.match(response.headers.get('Cache-Control') ?? '', /no-store/)
    }
  }
  const head = await request('/api/training-calendar', { method: 'HEAD' })
  assert.equal(head.status, 401)
  const html = await request('/')
  assert.ok(!(await html.text()).includes(secretPlan))
})

test('password exchange, private reads, and logout use revocable opaque sessions', async () => {
  const wrong = await request('/api/training-calendar/session', mutation('incorrect', '192.0.2.2'))
  assert.equal(wrong.status, 401)
  assert.equal(wrong.headers.get('Set-Cookie'), null)
  const cookie = await login()
  const response = await request('/api/training-calendar', { headers: { Cookie: cookie } })
  assert.equal(response.status, 200)
  assert.deepEqual(await response.json(), calendar)
  assert.match(response.headers.get('Cache-Control') ?? '', /private.*no-store/)
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), null)
  const forged = await request('/api/training-calendar', { headers: { Cookie: `${cookie}x` } })
  assert.equal(forged.status, 401)
  const duplicate = await request('/api/training-calendar', {
    headers: { Cookie: `${cookie}; ${cookie}` },
  })
  assert.equal(duplicate.status, 401)
  const otherHost = await request(
    '/api/training-calendar',
    { headers: { Cookie: cookie } },
    'https://aarnphm.xyz',
  )
  assert.equal(otherHost.status, 401)
  const logout = await request('/api/training-calendar/session', {
    method: 'DELETE',
    headers: { Origin: origin, Cookie: cookie },
  })
  assert.equal(logout.status, 200)
  assert.match(logout.headers.get('Set-Cookie') ?? '', /Max-Age=0/)
  assert.equal(
    (await request('/api/training-calendar', { headers: { Cookie: cookie } })).status,
    401,
  )
})

test('cross-origin login/logout, invalid media, and oversized bodies are rejected', async () => {
  for (const foreign of ['https://evil.example', 'https://notes.aarnphm.xyz', 'null']) {
    const init = mutation()
    const headers = new Headers(init.headers)
    headers.set('Origin', foreign)
    assert.equal(
      (await request('/api/training-calendar/session', { ...init, headers })).status,
      403,
    )
    assert.equal(
      (await request('/api/training-calendar/session', { method: 'DELETE', headers })).status,
      403,
    )
  }
  assert.equal((await request('/api/training-calendar/session', { method: 'POST' })).status, 403)
  assert.equal(
    (
      await request('/api/training-calendar/session', {
        method: 'POST',
        headers: { Origin: origin, 'Content-Type': 'text/plain' },
        body: '{}',
      })
    ).status,
    415,
  )
  assert.equal(
    (await request('/api/training-calendar/session', mutation('x'.repeat(5000)))).status,
    413,
  )
})

test('sessions last seven days and reads preserve their original deadline', async () => {
  const started = Date.now()
  const response = await request('/api/training-calendar/session', mutation(password, '192.0.2.6'))
  assert.equal(response.status, 200)
  const cookieHeader = response.headers.get('Set-Cookie')
  assert.ok(cookieHeader)
  assert.match(cookieHeader, /Max-Age=604800(?:;|$)/)
  const session: unknown = await response.json()
  assert.ok(isRecord(session))
  assert.equal(session.authenticated, true)
  assert.ok(typeof session.expiresAt === 'number')
  const week = 7 * 24 * 60 * 60 * 1000
  assert.ok(session.expiresAt >= started + week)
  assert.ok(session.expiresAt <= Date.now() + week)
  const cookie = cookieHeader.split(';')[0]
  const status = await request('/api/training-calendar/session', { headers: { Cookie: cookie } })
  assert.equal(status.status, 200)
  assert.deepEqual(await status.json(), session)

  // Model the final day without waiting a week; the Worker must retain its stored deadline.
  const sql = await server
    .getWorker('training-calendar-test')
    .getDurableObjectStorage('TRAINING_CALENDAR_ACCESS', { name: 'calendar' })
  const deadline = Date.now() + 24 * 60 * 60 * 1000
  await sql.exec('UPDATE sessions SET expires_at = ?', deadline)
  const resumed = await request('/api/training-calendar/session', { headers: { Cookie: cookie } })
  assert.equal(resumed.status, 200)
  assert.deepEqual(await resumed.json(), { authenticated: true, expiresAt: deadline })
  const data = await request('/api/training-calendar', { headers: { Cookie: cookie } })
  assert.equal(data.status, 200)
  assert.deepEqual(await data.json(), calendar)
})

test('expiry and credential rotation invalidate previously valid sessions', async () => {
  const cookie = await login('192.0.2.3')
  const sql = await server
    .getWorker('training-calendar-test')
    .getDurableObjectStorage('TRAINING_CALENDAR_ACCESS', { name: 'calendar' })
  await sql.exec('UPDATE sessions SET expires_at = 0')
  assert.equal(
    (await request('/api/training-calendar', { headers: { Cookie: cookie } })).status,
    401,
  )
  const rotated = await login('192.0.2.4')
  await sql.exec("UPDATE sessions SET credential = 'old-credential'")
  assert.equal(
    (await request('/api/training-calendar', { headers: { Cookie: rotated } })).status,
    401,
  )
})

test('login throttling survives repeated requests and blocks even a correct password', async () => {
  for (let i = 0; i < 5; i++)
    assert.equal(
      (await request('/api/training-calendar/session', mutation('incorrect', '192.0.2.9'))).status,
      401,
    )
  await server
    .getWorker('training-calendar-test')
    .evictDurableObject('TRAINING_CALENDAR_ACCESS', { name: 'calendar' })
  const limited = await request('/api/training-calendar/session', mutation(password, '192.0.2.9'))
  assert.equal(limited.status, 429)
  assert.ok(Number(limited.headers.get('Retry-After')) > 0)
})

test('parallel attempts share a global limit across client IPs and hosts', async () => {
  const sql = await server
    .getWorker('training-calendar-test')
    .getDurableObjectStorage('TRAINING_CALENDAR_ACCESS', { name: 'calendar' })
  await sql.exec('DELETE FROM attempts')
  const attempts = await Promise.all(
    Array.from({ length: 31 }, async (_, index) => {
      const host = index % 2 ? origin : 'https://aarnphm.xyz'
      const init = mutation('incorrect', `198.51.100.${index + 1}`)
      const headers = new Headers(init.headers)
      headers.set('Origin', host)
      return (await request('/api/training-calendar/session', { ...init, headers }, host)).status
    }),
  )
  assert.equal(attempts.filter(status => status === 401).length, 30)
  assert.equal(attempts.filter(status => status === 429).length, 1)
  await sql.exec('DELETE FROM attempts')
})

test('legacy idle-expiry storage preserves valid sessions and accepts new logins', async () => {
  const cookie = await login('192.0.2.13')
  const sql = await server
    .getWorker('training-calendar-test')
    .getDurableObjectStorage('TRAINING_CALENDAR_ACCESS', { name: 'calendar' })
  await sql.exec('ALTER TABLE sessions ADD COLUMN idle_expires_at INTEGER NOT NULL DEFAULT 0')
  await server.update({ root: evidence, workers: [{ config }] })
  assert.equal(
    (await request('/api/training-calendar', { headers: { Cookie: cookie } })).status,
    200,
  )
  const nextCookie = await login('192.0.2.14')
  assert.equal(
    (await request('/api/training-calendar', { headers: { Cookie: nextCookie } })).status,
    200,
  )
})

test('missing secrets and legacy plaintext assets fail closed even for a valid session', async () => {
  const cookie = await login('192.0.2.12')
  await writeFile(
    path.join(evidence, 'assets/static/training-calendar.json'),
    JSON.stringify(calendar),
  )
  await server.update({ root: evidence, workers: [{ config }] })
  assert.equal(
    (await request('/api/training-calendar', { headers: { Cookie: cookie } })).status,
    503,
  )
  const raw = await request('/static/training-calendar.json')
  assert.equal(raw.status, 404)
  assert.ok(!(await raw.text()).includes(secretPlan))
  for (const vars of [
    { TRAINING_CALENDAR_PASSWORD_HASH: config.vars?.TRAINING_CALENDAR_PASSWORD_HASH },
    { TRAINING_CALENDAR_DATA_KEY: config.vars?.TRAINING_CALENDAR_DATA_KEY },
  ]) {
    await server.update({ root: evidence, workers: [{ config: { ...config, vars } }] })
    assert.equal(
      (await request('/api/training-calendar', { headers: { Cookie: cookie } })).status,
      503,
    )
    assert.equal((await request('/api/training-calendar/session', mutation())).status, 503)
  }
})
