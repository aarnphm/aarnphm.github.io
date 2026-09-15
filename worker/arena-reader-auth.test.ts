import assert from 'node:assert/strict'
import test from 'node:test'
import {
  ARENA_READER_SESSION_COOKIE,
  createArenaReaderSessionCookie,
  createArenaResourceCapability,
  getArenaReaderIdentity,
  getArenaReaderLoginUrl,
  isArenaReaderMutationAllowed,
  verifyArenaResourceCapability,
} from './arena-reader-auth'

const origin = 'https://aarnphm.xyz'
const now = Date.UTC(2026, 8, 15, 12)
const env = { SESSION_SECRET: 'arena-reader-unit-test-signing-key' }
const owner = { id: 12345, login: 'aarnphm' }
const callback = new Request(`${origin}/comments/github/callback?code=verified&state=bound`)
const scope = { articleId: 'a1-article', snapshotId: 'snapshot-1', resourceId: 'image-1' }

function authenticatedRequest(cookie: string, url = `${origin}/api/arena/feed`): Request {
  return new Request(url, { headers: { Cookie: cookie.split(';')[0] } })
}

test('verified GitHub owner gets a host cookie and immutable account subject', async () => {
  const cookie = await createArenaReaderSessionCookie(callback, env, owner, now)
  assert.ok(cookie)
  const attributes = cookie.split(';').map(value => value.trim())
  assert.equal(attributes[0].startsWith(`${ARENA_READER_SESSION_COOKIE}=`), true)
  assert.deepEqual(attributes.slice(1), [
    'HttpOnly',
    'Secure',
    'Path=/',
    'SameSite=Lax',
    'Max-Age=2592000',
  ])
  assert.deepEqual(await getArenaReaderIdentity(authenticatedRequest(cookie), env, now), {
    subject: 'github:12345',
  })
  assert.deepEqual(
    await getArenaReaderIdentity(authenticatedRequest(cookie), env, now + 29 * 86400000),
    { subject: 'github:12345' },
  )
})

test('owner membership is checked on issuance and every session read', async () => {
  assert.equal(
    await createArenaReaderSessionCookie(callback, env, { id: 12345, login: 'someone-else' }, now),
    null,
  )
  for (const id of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.equal(await createArenaReaderSessionCookie(callback, env, { ...owner, id }, now), null)
  }
  const cookie = await createArenaReaderSessionCookie(callback, env, owner, now)
  assert.ok(cookie)
  assert.equal(
    await getArenaReaderIdentity(
      authenticatedRequest(cookie),
      { ...env, ARENA_OWNER_LOGIN: 'another-owner' },
      now,
    ),
    null,
  )
  const customOwner = { ...env, ARENA_OWNER_LOGIN: ' AarnPhm ' }
  assert.deepEqual(await getArenaReaderIdentity(authenticatedRequest(cookie), customOwner, now), {
    subject: 'github:12345',
  })
})

test('missing secret, unsigned login and client-supplied identity fail closed', async () => {
  for (const SESSION_SECRET of [undefined, '', '   ']) {
    assert.equal(
      await createArenaReaderSessionCookie(callback, { SESSION_SECRET }, owner, now),
      null,
    )
    assert.equal(
      await getArenaReaderIdentity(
        new Request(`${origin}/api/arena/feed?login=aarnphm&dev=true`, {
          headers: { 'X-Arena-Reader-Dev': 'true', 'X-Github-Login': 'aarnphm' },
        }),
        { SESSION_SECRET },
        now,
      ),
      null,
    )
  }
  assert.equal(
    await getArenaReaderIdentity(
      new Request(`${origin}/api/arena/feed`, {
        headers: { Cookie: `${ARENA_READER_SESSION_COOKIE}=aarnphm` },
      }),
      env,
      now,
    ),
    null,
  )
})

test('sessions expire, reject premature use, and cannot cross origins', async () => {
  const cookie = await createArenaReaderSessionCookie(callback, env, owner, now)
  assert.ok(cookie)
  assert.equal(
    await getArenaReaderIdentity(authenticatedRequest(cookie), env, now + 30 * 86400000),
    null,
  )
  assert.equal(await getArenaReaderIdentity(authenticatedRequest(cookie), env, now - 1000), null)
  for (const alternateOrigin of [
    'https://elsewhere.example',
    'https://aarnphm.xyz:444',
    'http://aarnphm.xyz',
  ]) {
    assert.equal(
      await getArenaReaderIdentity(
        authenticatedRequest(cookie, `${alternateOrigin}/api/arena/feed`),
        env,
        now,
      ),
      null,
    )
  }
  assert.equal(
    await createArenaReaderSessionCookie(
      new Request('http://aarnphm.xyz/callback'),
      env,
      owner,
      now,
    ),
    null,
  )
})

test('tampered, wrong-key, malformed and duplicate cookies never authenticate', async () => {
  const cookie = await createArenaReaderSessionCookie(callback, env, owner, now)
  assert.ok(cookie)
  const cookiePair = cookie.split(';')[0]
  const token = cookiePair.slice(ARENA_READER_SESSION_COOKIE.length + 1)
  const [payload, signature] = token.split('.')
  const tampered = Buffer.from(JSON.stringify({ id: 12345, login: 'aarnphm' })).toString(
    'base64url',
  )
  for (const invalid of [
    `${tampered}.${signature}`,
    `${payload}.${signature[0] === 'A' ? 'B' : 'A'}${signature.slice(1)}`,
    `${token}.extra`,
    'unparseable.%',
    '',
    'a'.repeat(4097),
  ]) {
    assert.equal(
      await getArenaReaderIdentity(
        authenticatedRequest(`${ARENA_READER_SESSION_COOKIE}=${invalid}`),
        env,
        now,
      ),
      null,
    )
  }
  assert.equal(
    await getArenaReaderIdentity(
      authenticatedRequest(cookie),
      { SESSION_SECRET: 'a-different-signing-key' },
      now,
    ),
    null,
  )
  assert.equal(
    await getArenaReaderIdentity(
      new Request(`${origin}/api/arena/feed`, {
        headers: { Cookie: `${cookiePair}; ${cookiePair}` },
      }),
      env,
      now,
    ),
    null,
  )
})

test('local identity requires the explicit environment flag and a literal loopback host', async () => {
  for (const localOrigin of [
    'http://localhost:8787',
    'http://127.0.0.1:8787',
    'http://[::1]:8787',
  ]) {
    const request = new Request(`${localOrigin}/api/arena/feed`)
    assert.deepEqual(await getArenaReaderIdentity(request, { ARENA_READER_DEV: 'true' }), {
      subject: 'dev:aarnphm',
    })
    for (const flag of [undefined, 'false', '1']) {
      assert.equal(await getArenaReaderIdentity(request, { ARENA_READER_DEV: flag }), null)
    }
  }
  for (const host of [
    'aarnphm.xyz',
    'localhost.evil.example',
    '127.0.0.1.evil.example',
    '10.0.0.1',
  ]) {
    assert.equal(
      await getArenaReaderIdentity(
        new Request(`https://${host}/api/arena/feed?dev=true`, {
          headers: { Host: 'localhost', 'X-Forwarded-Host': 'localhost' },
        }),
        { ARENA_READER_DEV: 'true' },
      ),
      null,
    )
  }
})

test('login redirects keep reader selection and drop arbitrary destinations', () => {
  const reader = new Request(`${origin}/arena/feed?article=article-1&view=notes`)
  const login = new URL(getArenaReaderLoginUrl(reader))
  assert.equal(login.origin, origin)
  assert.equal(login.pathname, '/comments/github/login')
  assert.equal(login.searchParams.get('returnTo'), '/arena/feed?article=article-1&view=notes')
  for (const path of ['/api/arena/feed', '/arena/feed/other', '/arena/feed.html', '/other']) {
    const url = new URL(
      getArenaReaderLoginUrl(new Request(`${origin}${path}?returnTo=https://evil.example`)),
    )
    assert.equal(url.searchParams.get('returnTo'), '/arena/feed')
  }
})

test('mutations require an exact same-origin browser request', () => {
  const mutation = (headers: HeadersInit) =>
    new Request(`${origin}/api/arena/articles/article-1/read`, { method: 'PUT', headers })
  assert.equal(isArenaReaderMutationAllowed(mutation({ Origin: origin })), true)
  assert.equal(
    isArenaReaderMutationAllowed(mutation({ Origin: origin, 'Sec-Fetch-Site': 'same-origin' })),
    true,
  )
  for (const Origin of [
    undefined,
    'null',
    'https://evil.example',
    'http://aarnphm.xyz',
    `${origin}:444`,
  ]) {
    const headers = new Headers({ 'Sec-Fetch-Site': 'same-origin' })
    if (Origin !== undefined) headers.set('Origin', Origin)
    assert.equal(isArenaReaderMutationAllowed(mutation(headers)), false)
  }
  for (const fetchSite of ['cross-site', 'same-site', 'none']) {
    assert.equal(
      isArenaReaderMutationAllowed(mutation({ Origin: origin, 'Sec-Fetch-Site': fetchSite })),
      false,
    )
  }
})

test('resource capabilities bind the exact article, snapshot and resource', async () => {
  const token = await createArenaResourceCapability(env, scope, 3600, now)
  assert.equal(await verifyArenaResourceCapability(env, token, scope, now), true)
  assert.equal(await verifyArenaResourceCapability(env, token, scope, now + 3599000), true)
  assert.equal(await verifyArenaResourceCapability(env, token, scope, now + 3600000), false)
  assert.equal(await verifyArenaResourceCapability(env, token, scope, now - 1000), false)
  for (const other of [
    { ...scope, articleId: 'other-article' },
    { ...scope, snapshotId: 'other-snapshot' },
    { ...scope, resourceId: 'other-resource' },
  ]) {
    assert.equal(await verifyArenaResourceCapability(env, token, other, now), false)
  }
  assert.equal(
    await verifyArenaResourceCapability({ SESSION_SECRET: 'different-key' }, token, scope, now),
    false,
  )
  assert.equal(await verifyArenaResourceCapability({}, token, scope, now), false)
})

test('resource signatures cannot be reused as sessions or the reverse', async () => {
  const resourceToken = await createArenaResourceCapability(env, scope, 3600, now)
  assert.equal(
    await getArenaReaderIdentity(
      authenticatedRequest(`${ARENA_READER_SESSION_COOKIE}=${resourceToken}`),
      env,
      now,
    ),
    null,
  )
  const cookie = await createArenaReaderSessionCookie(callback, env, owner, now)
  assert.ok(cookie)
  const sessionToken = cookie.split(';')[0].slice(ARENA_READER_SESSION_COOKIE.length + 1)
  assert.equal(await verifyArenaResourceCapability(env, sessionToken, scope, now), false)
})

test('resource signing rejects traversal scopes, missing keys and unbounded lifetimes', async () => {
  for (const resourceId of ['', '../secret', 'image?url=https://evil.example', 'x'.repeat(161)]) {
    await assert.rejects(createArenaResourceCapability(env, { ...scope, resourceId }, 3600, now))
    assert.equal(
      await verifyArenaResourceCapability(env, 'invalid.token', { ...scope, resourceId }, now),
      false,
    )
  }
  await assert.rejects(createArenaResourceCapability({}, scope, 3600, now))
  for (const lifetime of [0, -1, 3601, 1.5, Infinity]) {
    await assert.rejects(createArenaResourceCapability(env, scope, lifetime, now))
  }
  const token = await createArenaResourceCapability(env, scope, 1, now)
  assert.equal(await verifyArenaResourceCapability(env, token, scope, now), true)
  assert.equal(await verifyArenaResourceCapability(env, token, scope, now + 1000), false)
})
