import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import test from 'node:test'
import {
  assertGarminResponseAuthorized,
  DEFAULT_GARMIN_CONNECT_BASE,
  fetchGarminBytes,
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

test('downloads Garmin bytes with binary content negotiation and bearer authorization', async t => {
  let accept: string | undefined
  let authorization: string | undefined
  const server = createServer((request, response) => {
    accept = request.headers.accept
    authorization = request.headers.authorization
    response.end(Buffer.from([1, 2, 3]))
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject)
      resolve()
    })
  })
  t.after(
    () =>
      new Promise<void>((resolve, reject) => {
        server.close(error => (error ? reject(error) : resolve()))
      }),
  )
  const address = server.address()
  if (address == null || typeof address === 'string') throw new Error('test server has no TCP port')

  const bytes = await fetchGarminBytes(
    { headers: { Accept: 'application/json', Authorization: 'Bearer test-token' } },
    `http://127.0.0.1:${address.port}`,
    '/activity.fit',
  )

  assert.deepEqual(bytes, Uint8Array.from([1, 2, 3]))
  assert.equal(accept, '*/*')
  assert.equal(authorization, 'Bearer test-token')
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
