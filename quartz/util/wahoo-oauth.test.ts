import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_WAHOO_REDIRECT_URI,
  parseWahooAuthorizationCallback,
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
})
