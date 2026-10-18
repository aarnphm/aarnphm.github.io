import assert from 'node:assert/strict'
import { once } from 'node:events'
import { mkdir, writeFile } from 'node:fs/promises'
import http from 'node:http'
import path from 'node:path'
import test from 'node:test'
import { listenForOAuthCallback } from './oauth-callback'

const artifacts = path.resolve('docs/verification/code-scanning-remediation/artifacts/oauth')

for (const provider of ['Strava', 'Oura']) {
  test(`${provider} callback preserves grants and contains errors on loopback`, async t => {
    for (const fixture of [
      { name: 'success', query: 'code=fixture-code&scope=activity%3Aread_all', status: 200 },
      {
        name: 'provider-error',
        query: new URLSearchParams({ error: '<img src=x onerror=alert(1)>' }).toString(),
        status: 400,
      },
      { name: 'missing-code', query: '', status: 400 },
    ]) {
      await t.test(fixture.name, async () => {
        const { server, grant } = listenForOAuthCallback({
          provider,
          redirectUri: 'http://localhost:0/callback',
        })
        const outcome = grant.then(
          value => ({ value, error: null }),
          (error: unknown) => ({ value: null, error }),
        )
        t.after(() => server.close())
        await once(server, 'listening')
        const address = server.address()
        assert.ok(address && typeof address !== 'string')
        assert.equal(address.address, '127.0.0.1')
        const base = `http://127.0.0.1:${address.port}`

        const malformedStatus = await new Promise<number>(resolve => {
          http.get({ hostname: '127.0.0.1', port: address.port, path: 'http://[' }, response => {
            response.resume()
            response.on('end', () => resolve(response.statusCode ?? 0))
          })
        })
        assert.equal(malformedStatus, 400)
        assert.equal(server.listening, true)

        for (const { route, method, expected } of [
          { route: '/favicon.ico', method: 'GET', expected: 404 },
          { route: '/callback', method: 'POST', expected: 405 },
        ]) {
          const response = await fetch(base + route, { method })
          assert.equal(response.status, expected)
          await response.text()
          assert.equal(server.listening, true)
        }

        const response = await fetch(`${base}/callback?${fixture.query}`)
        const body = await response.text()
        assert.equal(response.status, fixture.status)
        assert.equal(response.headers.get('content-type'), 'text/plain; charset=utf-8')
        assert.equal(response.headers.get('x-content-type-options'), 'nosniff')
        assert.equal(response.headers.get('cache-control'), 'no-store')
        assert.equal(body.includes('<img'), false)
        const result = await outcome
        if (fixture.name === 'success') {
          assert.deepEqual(result.value, { code: 'fixture-code', scope: 'activity:read_all' })
        } else {
          assert.ok(result.error instanceof Error)
          assert.equal(
            result.error.message,
            fixture.name === 'provider-error'
              ? '<img src=x onerror=alert(1)>'
              : 'no authorization code in redirect',
          )
        }
        assert.equal(server.listening, false)
        await mkdir(artifacts, { recursive: true })
        await writeFile(
          path.join(artifacts, `${provider.toLowerCase()}-${fixture.name}.json`),
          JSON.stringify(
            {
              command: 'pnpm test quartz/scripts/oauth-callback.e2e.test.ts',
              provider,
              fixture,
              bindAddress: address.address,
              status: response.status,
              headers: Object.fromEntries(response.headers),
              body,
              grant: result.value,
              error: result.error instanceof Error ? result.error.message : null,
              listeningAfterCallback: server.listening,
            },
            null,
            2,
          ),
        )
      })
    }
  })
}

test('callback rejects non-loopback redirects before listening', () => {
  for (const redirectUri of [
    'http://localhost.attacker.example:8722',
    'http://example.com:8722',
    'https://localhost:8722',
  ]) {
    assert.throws(() => listenForOAuthCallback({ provider: 'Oura', redirectUri }), /loopback/)
  }
})

test('callback binds IPv6 loopback and releases an occupied port safely', async t => {
  const { server, grant } = listenForOAuthCallback({
    provider: 'Oura',
    redirectUri: 'http://[::1]:0/',
  })
  t.after(() => server.close())
  await once(server, 'listening')
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  assert.equal(address.address, '::1')
  const conflict = listenForOAuthCallback({
    provider: 'Oura',
    redirectUri: `http://[::1]:${address.port}/`,
  })
  await assert.rejects(conflict.grant, { code: 'EADDRINUSE' })
  const response = await fetch(`http://[::1]:${address.port}/?code=ipv6`)
  await response.text()
  assert.deepEqual(await grant, { code: 'ipv6', scope: '' })
  assert.equal(server.listening, false)
})

test('callback listener rejects an unavailable bind port', async t => {
  const occupied = http.createServer()
  occupied.listen(0, '127.0.0.1')
  t.after(() => occupied.close())
  await once(occupied, 'listening')
  const address = occupied.address()
  assert.ok(address && typeof address !== 'string')
  const callback = listenForOAuthCallback({
    provider: 'Strava',
    redirectUri: `http://localhost:${address.port}/`,
  })
  await assert.rejects(callback.grant, { code: 'EADDRINUSE' })
})
