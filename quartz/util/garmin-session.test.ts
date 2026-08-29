import assert from 'node:assert/strict'
import test from 'node:test'
import {
  assertGarminResponseAuthorized,
  DEFAULT_GARMIN_CONNECT_BASE,
  garminConnectRequestHeaders,
  parseGarminConnectSession,
} from './garmin-session'

test('parses a bearer-only Garmin session', () => {
  const session = parseGarminConnectSession({
    headers: {
      Accept: 'application/json',
      Authorization: 'Bearer test-token',
      'X-Garmin-Client-Platform': 'Android',
    },
  })
  assert.equal(DEFAULT_GARMIN_CONNECT_BASE, 'https://connectapi.garmin.com')
  assert.equal(session.headers.authorization, 'Bearer test-token')
  assert.equal(session.headers.cookie, undefined)
  const headers = new Headers(garminConnectRequestHeaders(session, 'application/json'))
  assert.equal(headers.get('authorization'), 'Bearer test-token')
  assert.equal(headers.get('content-type'), 'application/json')
})

test('reports rejected bearer sessions with the reauthorization command', () => {
  assert.doesNotThrow(() => assertGarminResponseAuthorized(new Response(null, { status: 200 })))
  assert.throws(
    () => assertGarminResponseAuthorized(new Response(null, { status: 401 })),
    /run pnpm garmin:auth/,
  )
  assert.throws(
    () => assertGarminResponseAuthorized(new Response(null, { status: 403 })),
    /run pnpm garmin:auth/,
  )
})

test('rejects malformed and browser-backed Garmin sessions', () => {
  assert.throws(() => parseGarminConnectSession(null), /invalid session/)
  assert.throws(
    () => parseGarminConnectSession({ headers: { Accept: 'application/json' } }),
    /bearer token/,
  )
  assert.throws(
    () =>
      parseGarminConnectSession({
        headers: { Authorization: 'Bearer test-token', Cookie: 'SESSION=browser' },
      }),
    /forbidden cookie/,
  )
  assert.throws(
    () =>
      parseGarminConnectSession({
        headers: { Authorization: 'Bearer test-token', 'Connect-Csrf-Token': 'browser-csrf' },
      }),
    /forbidden connect-csrf-token/,
  )
})
