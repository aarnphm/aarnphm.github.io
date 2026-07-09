import assert from 'node:assert/strict'
import { once } from 'node:events'
import http from 'node:http'
import test from 'node:test'
import { fetchCanonical } from './fetch-canonical'

test('fetchCanonical follows canonical links with a fresh GET', async t => {
  const server = http.createServer((request, response) => {
    if (request.url === '/alias') {
      response.setHeader('content-type', 'text/html; charset=utf-8')
      response.end('<link rel="canonical" href="/target">')
      return
    }

    response.setHeader('content-type', 'application/json')
    response.end(
      JSON.stringify({ method: request.method, requestId: request.headers['x-request-id'] }),
    )
  })

  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  t.after(() => new Promise<void>(resolve => server.close(() => resolve())))

  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('HTTP test server has no port')

  const response = await fetchCanonical(new URL(`http://127.0.0.1:${address.port}/alias`), {
    method: 'POST',
    headers: { 'x-request-id': 'preserved' },
    body: 'request body',
  })

  assert.equal(response.url, `http://127.0.0.1:${address.port}/target`)
  assert.deepEqual(await response.json(), { method: 'GET' })
})

test('fetchCanonical preserves abort ownership while following', async t => {
  let resolveTargetStarted: (() => void) | undefined
  const targetStarted = new Promise<void>(resolve => {
    resolveTargetStarted = resolve
  })
  const server = http.createServer((request, response) => {
    if (request.url === '/alias') {
      response.setHeader('content-type', 'text/html; charset=utf-8')
      response.end('<link rel="canonical" href="/target">')
      return
    }
    resolveTargetStarted?.()
  })

  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  t.after(() => new Promise<void>(resolve => server.close(() => resolve())))

  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('HTTP test server has no port')

  const controller = new AbortController()
  const request = fetchCanonical(new URL(`http://127.0.0.1:${address.port}/alias`), {
    signal: controller.signal,
  })
  await targetStarted
  controller.abort()
  await assert.rejects(
    request,
    error => error instanceof DOMException && error.name === 'AbortError',
  )
})

test('fetchCanonical leaves non-HTML responses untouched', async t => {
  const server = http.createServer((_request, response) => {
    response.setHeader('content-type', 'text/plain')
    response.end('<link rel="canonical" href="/target">')
  })

  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  t.after(() => new Promise<void>(resolve => server.close(() => resolve())))

  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('HTTP test server has no port')

  const response = await fetchCanonical(new URL(`http://127.0.0.1:${address.port}/plain`))
  assert.equal(await response.text(), '<link rel="canonical" href="/target">')
})
