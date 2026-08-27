import assert from 'node:assert/strict'
import test from 'node:test'
import { handleWahooOAuthCallback } from './wahoo-oauth'

const CALLBACK_URL = 'https://aarnphm.xyz/oauth/wahoo/callback'

test('ignores unrelated Worker routes', () => {
  assert.equal(handleWahooOAuthCallback(new Request('https://aarnphm.xyz/triathlon')), null)
})

test('forwards a valid Wahoo grant to the fixed local listener', () => {
  const response = handleWahooOAuthCallback(
    new Request(`${CALLBACK_URL}?code=grant-code&state=nonce&ignored=value`),
  )
  assert.ok(response)
  assert.equal(response.status, 302)
  assert.equal(
    response.headers.get('Location'),
    'http://127.0.0.1:8722/?code=grant-code&state=nonce',
  )
  assert.equal(response.headers.get('Cache-Control'), 'no-store')
  assert.equal(response.headers.get('Referrer-Policy'), 'no-referrer')
})

test('forwards Wahoo errors without reflecting unsupported fields', () => {
  const response = handleWahooOAuthCallback(
    new Request(
      `${CALLBACK_URL}?error=access_denied&error_description=Authorization%20declined&state=nonce&redirect_uri=https://attacker.example`,
    ),
  )
  assert.ok(response)
  assert.equal(response.status, 302)
  assert.equal(
    response.headers.get('Location'),
    'http://127.0.0.1:8722/?error=access_denied&error_description=Authorization+declined&state=nonce',
  )
})

test('rejects invalid callbacks and unsupported methods without caching them', () => {
  const invalidResponse = handleWahooOAuthCallback(new Request(`${CALLBACK_URL}?code=grant-code`))
  assert.ok(invalidResponse)
  assert.equal(invalidResponse.status, 400)
  assert.equal(invalidResponse.headers.get('Location'), null)
  assert.equal(invalidResponse.headers.get('Cache-Control'), 'no-store')

  const methodResponse = handleWahooOAuthCallback(new Request(CALLBACK_URL, { method: 'POST' }))
  assert.ok(methodResponse)
  assert.equal(methodResponse.status, 405)
  assert.equal(methodResponse.headers.get('Allow'), 'GET')
  assert.equal(methodResponse.headers.get('Cache-Control'), 'no-store')
})
