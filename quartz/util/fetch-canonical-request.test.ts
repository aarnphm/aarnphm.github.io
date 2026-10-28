import assert from 'node:assert/strict'
import { once } from 'node:events'
import http from 'node:http'
import test from 'node:test'
import { fetchCanonical } from './fetch-canonical'

// Real HTTP coverage for cancellation races that depend on fetch rejection timing:
// an immediate replacement, an older request settling after its replacement, and
// one consumer leaving while another still needs the same response.
test('an aborted request cannot capture its immediate replacement', { timeout: 5000 }, async t => {
  let requests = 0
  const firstStarted = Promise.withResolvers<void>()
  const replacementStarted = Promise.withResolvers<http.ServerResponse>()
  const server = http.createServer((_request, response) => {
    requests += 1
    if (requests === 1) firstStarted.resolve()
    else replacementStarted.resolve(response)
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  t.after(() => {
    server.closeAllConnections()
    server.close()
  })
  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('HTTP test server has no port')
  const url = new URL(`http://127.0.0.1:${address.port}/day`)
  const controller = new AbortController()
  const first = fetchCanonical(url, { signal: controller.signal })
  const aborted = assert.rejects(first, { name: 'AbortError' })
  await firstStarted.promise

  controller.abort()
  const replacement = fetchCanonical(url)
  // Consume a possible rejection before waiting for the replacement to reach the server.
  const replacementResponse = await Promise.race([
    replacementStarted.promise,
    replacement.then(() => {
      throw new Error('Replacement returned before the server responded')
    }),
  ])
  await aborted
  const joined = fetchCanonical(url)
  replacementResponse.setHeader('content-type', 'text/html')
  replacementResponse.end('<h1>day</h1>')
  const [a, b] = await Promise.all([replacement, joined])
  assert.equal(await a.text(), '<h1>day</h1>')
  assert.equal(await b.text(), '<h1>day</h1>')
  assert.equal(requests, 2, 'the replacement remains shared after the old request settles')
})

test(
  'cancelling one reader preserves another reader of the response',
  { timeout: 5000 },
  async t => {
    let requests = 0
    const started = Promise.withResolvers<http.ServerResponse>()
    const server = http.createServer((_request, response) => {
      requests += 1
      started.resolve(response)
    })
    server.listen(0, '127.0.0.1')
    await once(server, 'listening')
    t.after(() => {
      server.closeAllConnections()
      server.close()
    })
    const address = server.address()
    if (!address || typeof address === 'string') throw new Error('HTTP test server has no port')
    const url = new URL(`http://127.0.0.1:${address.port}/day`)
    const controller = new AbortController()
    const preview = fetchCanonical(url, { signal: controller.signal })
    const aborted = assert.rejects(preview, { name: 'AbortError' })
    const navigation = fetchCanonical(url)
    const response = await started.promise
    controller.abort()
    response.setHeader('content-type', 'text/html')
    response.end('<h1>day</h1>')
    await aborted
    assert.equal(await (await navigation).text(), '<h1>day</h1>')
    assert.equal(requests, 1)
  },
)
