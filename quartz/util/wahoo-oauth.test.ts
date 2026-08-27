import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_WAHOO_REDIRECT_URI,
  parseWahooAuthorizationCallback,
  parsePastedWahooAuthorizationGrant,
  requireWahooAuthorizationGrant,
  resolveWahooRedirectUri,
} from './wahoo-oauth'

test('parses Wahoo authorization grants and provider errors', () => {
  assert.deepEqual(
    parseWahooAuthorizationCallback(new URLSearchParams({ code: 'grant-code', state: 'nonce' })),
    { kind: 'grant', code: 'grant-code', state: 'nonce' },
  )
  assert.deepEqual(
    parseWahooAuthorizationCallback(
      new URLSearchParams({
        error: 'access_denied',
        error_description: 'Authorization declined',
        state: 'nonce',
      }),
    ),
    {
      kind: 'error',
      error: 'access_denied',
      errorDescription: 'Authorization declined',
      state: 'nonce',
    },
  )
})

test('rejects ambiguous or incomplete Wahoo callbacks', () => {
  for (const searchParams of [
    new URLSearchParams({ code: 'grant-code' }),
    new URLSearchParams({ state: 'nonce' }),
    new URLSearchParams({ code: 'grant-code', error: 'access_denied', state: 'nonce' }),
    new URLSearchParams({ code: 'grant-code', error_description: 'declined', state: 'nonce' }),
    new URLSearchParams('code=one&code=two&state=nonce'),
  ]) {
    assert.throws(() => parseWahooAuthorizationCallback(searchParams))
  }
})

test('requires matching state for callbacks handled by the active authorization process', () => {
  assert.deepEqual(
    requireWahooAuthorizationGrant(
      new URLSearchParams({ code: 'grant-code', state: 'nonce' }),
      'nonce',
    ),
    { code: 'grant-code', state: 'nonce' },
  )
  assert.throws(
    () =>
      requireWahooAuthorizationGrant(
        new URLSearchParams({ code: 'grant-code', state: 'other' }),
        'nonce',
      ),
    /state mismatch/,
  )
  assert.throws(
    () =>
      requireWahooAuthorizationGrant(
        new URLSearchParams({ error: 'access_denied', state: 'nonce' }),
        'nonce',
      ),
    /authorization failed: access_denied/,
  )
})

test('parses a manually pasted callback only for the registered redirect target', () => {
  assert.deepEqual(
    parsePastedWahooAuthorizationGrant(
      `${DEFAULT_WAHOO_REDIRECT_URI}?code=grant-code&state=previous-process-state`,
      DEFAULT_WAHOO_REDIRECT_URI,
      null,
    ),
    { code: 'grant-code', state: 'previous-process-state' },
  )
  assert.throws(
    () =>
      parsePastedWahooAuthorizationGrant(
        'https://attacker.example/oauth/wahoo/callback?code=grant-code&state=nonce',
        DEFAULT_WAHOO_REDIRECT_URI,
        null,
      ),
    /must use https:\/\/aarnphm\.xyz\/oauth\/wahoo\/callback/,
  )
  assert.throws(
    () =>
      parsePastedWahooAuthorizationGrant(
        `${DEFAULT_WAHOO_REDIRECT_URI}?code=grant-code&state=other`,
        DEFAULT_WAHOO_REDIRECT_URI,
        'nonce',
      ),
    /state mismatch/,
  )
})

test('uses the public HTTPS callback and rejects insecure overrides', () => {
  assert.equal(resolveWahooRedirectUri(undefined), DEFAULT_WAHOO_REDIRECT_URI)
  assert.equal(
    resolveWahooRedirectUri('https://example.com/wahoo/callback'),
    'https://example.com/wahoo/callback',
  )
  assert.throws(() => resolveWahooRedirectUri('http://127.0.0.1:8722/'), /must use HTTPS/)
  assert.throws(
    () => resolveWahooRedirectUri('https://user@example.com/callback'),
    /cannot include credentials/,
  )
  assert.throws(
    () => resolveWahooRedirectUri('https://example.com/callback?destination=local'),
    /cannot include credentials, a query, or a fragment/,
  )
})
