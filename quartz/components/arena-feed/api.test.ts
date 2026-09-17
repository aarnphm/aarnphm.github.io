import assert from 'node:assert/strict'
import test from 'node:test'
import { parseFeed, readApiResponse, ReaderApiError } from './api'

test('feed responses retain Curius provenance and reject links without a valid source', () => {
  const entry = {
    articleId: 'article-v1-fixture',
    sourceUrl: 'https://example.com/read',
    title: 'Saved on Curius',
    kind: 'html',
    later: false,
    tags: [],
    savedAt: null,
    occurrences: [],
    curius: [{ userId: 3584, linkId: 236997 }],
  }
  const feed = { subject: 'owner', revision: 'feed-fixture', entries: [entry], readLinks: [] }
  assert.deepEqual(parseFeed(feed), feed)
  for (const curius of [undefined, [], [{ userId: 3584, linkId: '236997' }]])
    assert.throws(
      () => parseFeed({ ...feed, entries: [{ ...entry, curius }] }),
      /invalid catalogue/,
    )
})

test('render failures retain typed unavailable results across HTTP error statuses', async () => {
  const unavailable = {
    status: 'unavailable',
    reason: 'blocked',
    message: 'Source asked to retry later.',
    sourceUrl: 'https://example.com',
    retryAfter: 120,
  }
  for (const status of [429, 502, 503]) {
    const response = Response.json(unavailable, { status })
    assert.deepEqual(await readApiResponse(response, true), unavailable)
  }
})

test('render response handling preserves authentication failures', async () => {
  for (const status of [401, 403]) {
    const response = Response.json(
      {
        status: 'unavailable',
        error: 'unauthorized',
        message: 'Sign in first.',
        loginUrl: '/api/auth/login',
      },
      { status },
    )
    await assert.rejects(
      readApiResponse(response, true),
      error =>
        error instanceof ReaderApiError &&
        error.status === status &&
        error.loginUrl === '/api/auth/login',
    )
  }
})

test('failed state requests remain errors instead of empty reading histories', async () => {
  await assert.rejects(
    readApiResponse(
      Response.json({ error: 'database', message: 'Storage unavailable.' }, { status: 503 }),
    ),
    error => error instanceof ReaderApiError && error.status === 503,
  )
  await assert.rejects(
    readApiResponse(
      new Response('<html>static page</html>', { headers: { 'Content-Type': 'text/html' } }),
    ),
    /API is unavailable/,
  )
})
