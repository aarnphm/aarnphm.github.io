import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import test from 'node:test'
import { createNavigationResource } from './data'

test('navigation resources share requests and resolve aborts without stale values', async () => {
  const server = createServer((_request, response) => {
    setTimeout(() => {
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end('{"value":42}')
    }, 100)
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const controller = new AbortController()
  const resource = createNavigationResource(controller.signal, async response => {
    const value: { value: number } = await response.json()
    return value
  })
  const path = `http://127.0.0.1:${address.port}/artifact`
  const first = resource.load(path)
  const second = resource.load(path)
  assert.equal(first, second)
  controller.abort()
  assert.deepEqual(await first, { status: 'aborted' })
  assert.deepEqual(resource.peek(path), { status: 'aborted' })
  await new Promise<void>((resolve, reject) =>
    server.close(error => (error ? reject(error) : resolve())),
  )
})
