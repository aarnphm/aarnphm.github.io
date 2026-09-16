import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { getPlatformProxy } from 'wrangler'
import {
  type ArenaReaderResource,
  arenaReaderSourceHeaders,
  fetchArenaReaderSource,
  isArenaPublicAddress,
  isPublicArenaHostname,
  saveArenaReaderResource,
  serveArenaReaderResource,
  validateArenaReaderTarget,
} from './arena-reader-resources'

test('source requests identify the reader without forwarding caller credentials or identity', () => {
  const headers = arenaReaderSourceHeaders({
    Accept: 'text/html',
    'Accept-Language': 'fr',
    Range: 'bytes=0-1023',
    'User-Agent': 'UntrustedClient',
    Authorization: 'Bearer private',
    Cookie: 'session=private',
    Referer: 'https://private.example/notes',
  })
  assert.equal(headers.get('User-Agent'), 'GardenArenaReader/1.0 (https://aarnphm.xyz/arena)')
  assert.equal(arenaReaderSourceHeaders().get('User-Agent'), headers.get('User-Agent'))
  assert.equal(headers.get('Accept-Encoding'), 'identity')
  assert.equal(headers.get('Accept'), 'text/html')
  assert.equal(headers.get('Accept-Language'), 'fr')
  assert.equal(headers.get('Range'), 'bytes=0-1023')
  for (const name of ['Authorization', 'Cookie', 'Referer']) assert.equal(headers.get(name), null)
  assert.equal(arenaReaderSourceHeaders({ Range: 'bytes=0-1,4-5' }).get('Range'), null)
  assert.equal(arenaReaderSourceHeaders({ Accept: 'x'.repeat(1025) }).get('Accept'), null)
})

test('accepts anonymous public HTTP URLs and preserves source identity', () => {
  for (const url of [
    'https://example.com/article?q=one#section',
    'http://example.com:80/article',
    'https://example.com:443/article',
    'https://xn--bcher-kva.example/article',
    'https://1.1.1.1/article',
    'https://[2606:4700:4700::1111]/article',
  ]) {
    assert.equal(validateArenaReaderTarget(url)?.href, new URL(url).href, url)
  }
})

test('blocks credentials, unexpected ports, local hosts and alternate IP spellings', () => {
  for (const url of [
    'file:///etc/passwd',
    'data:text/html,hello',
    'ftp://example.com/article',
    'https://reader:password@example.com/article',
    'https://reader@example.com/article',
    'https://example.com:8443/article',
    'https://example.com:80/article',
    'http://example.com:443/article',
    'http://localhost/article',
    'http://localhost./article',
    'http://publisher.localhost/article',
    'http://publisher.local/article',
    'http://metadata.google.internal/article',
    'http://publisher.home.arpa/article',
    'http://publisher.test/article',
    'http://printer/article',
    'http://0177.0.0.1/article',
    'http://0x7f000001/article',
    'http://2130706433/article',
    'http://127.1/article',
    'http://0x0a000001/article',
    'http://[::ffff:127.0.0.1]/article',
    'http://[::ffff:7f00:1]/article',
    'http://[0:0:0:0:0:ffff:0a00:0001]/article',
    'http://[64:ff9b::a00:1]/article',
    'http://[2002:7f00:1::]/article',
  ]) {
    assert.equal(validateArenaReaderTarget(url), null, url)
  }
})

test('rejects private, special-use and reserved addresses across both IP families', () => {
  for (const address of [
    '0.0.0.0',
    '0.8.0.1',
    '10.2.3.4',
    '100.64.0.1',
    '100.127.255.254',
    '127.0.0.1',
    '169.254.169.254',
    '172.16.0.1',
    '172.31.255.255',
    '192.0.0.9',
    '192.0.2.1',
    '192.31.196.1',
    '192.52.193.1',
    '192.88.99.2',
    '192.168.0.1',
    '192.175.48.1',
    '198.18.0.1',
    '198.19.255.255',
    '198.51.100.1',
    '203.0.113.1',
    '224.0.0.1',
    '239.1.1.1',
    '240.0.0.1',
    '255.255.255.255',
    '::',
    '::1',
    '::ffff:127.0.0.1',
    '::ffff:0808:0808',
    '::8.8.8.8',
    '64:ff9b::808:808',
    '100::1',
    '100:0:0:1::1',
    '2001::1',
    '2001:10::1',
    '2001:db8::1',
    '2002:7f00:1::',
    '2620:4f:8000::1',
    '3ffe::1',
    '3fff::1',
    '5f00::1',
    'fc00::1',
    'fdff::1',
    'fe80::1',
    'febf::1',
    'fec0::1',
    'ff02::1',
    'example.com',
    'garbage',
    '256.1.1.1',
    '1.1.1',
    '[:::]',
    '[1.1.1.1',
    '1.1.1.1]',
  ])
    assert.equal(isArenaPublicAddress(address), false, address)
  for (const address of [
    '1.1.1.1',
    '8.8.8.8',
    '100.63.255.255',
    '100.128.0.1',
    '172.15.255.255',
    '172.32.0.1',
    '192.0.1.1',
    '192.169.0.1',
    '198.17.255.255',
    '198.20.0.1',
    '2001:4860:4860::8888',
    '2606:4700:4700::1111',
    '[2606:4700:4700::1111]',
  ])
    assert.equal(isArenaPublicAddress(address), true, address)
})

test('rejects blocked fetch targets before a DNS or source request is necessary', async () => {
  assert.equal(await isPublicArenaHostname('metadata.google.internal'), false)
  assert.equal(await isPublicArenaHostname('127.0.0.1'), false)
  assert.equal(await isPublicArenaHostname('1.1.1.1'), true)
  await assert.rejects(
    fetchArenaReaderSource('http://[::ffff:127.0.0.1]/secret'),
    /public addresses/,
  )
  await assert.rejects(
    fetchArenaReaderSource('https://example.com/', { signal: AbortSignal.abort() }),
    { name: 'AbortError' },
  )
})

test('persists bounded resources in real local R2 and serves private ranges', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'arena-resource-r2-'))
  await writeFile(
    path.join(root, 'wrangler.json'),
    JSON.stringify({
      name: 'arena-resource-r2-test',
      compatibility_date: '2025-01-21',
      r2_buckets: [{ binding: 'ARENA_CONTENT', bucket_name: 'test-arena-resources' }],
    }),
  )
  const platform = await getPlatformProxy<{ ARENA_CONTENT: R2Bucket }>({
    configPath: path.join(root, 'wrangler.json'),
    persist: false,
    remoteBindings: false,
  })
  try {
    const bucket = platform.env.ARENA_CONTENT
    const resource: ArenaReaderResource = {
      articleId: 'article-1',
      snapshotId: 'snapshot-1',
      resourceId: 'paper',
      sourceUrl: 'https://example.com/paper.pdf',
      purpose: 'pdf',
    }
    const key = 'arena-reader/v1/article-1/resources/snapshot-1/paper'
    const bytes = '%PDF-1.7\n0123456789'
    await saveArenaReaderResource(
      bucket,
      resource,
      new Response(bytes, {
        headers: { 'Content-Type': 'application/pdf', 'Set-Cookie': 'source=private' },
      }),
    )
    const metadata = await bucket.head(key)
    assert.ok(metadata)
    const serve = (headers: HeadersInit = {}, method = 'GET') =>
      serveArenaReaderResource(
        new Request(
          'https://garden.example/api/arena/articles/article-1/resources/snapshot-1/paper',
          { method, headers },
        ),
        bucket,
        resource,
      )
    await t.test('saved content uses private headers and never rechecks the source', async () => {
      const response = await serve()
      assert.equal(response.status, 200)
      assert.equal(response.headers.get('Content-Type'), 'application/pdf')
      assert.equal(response.headers.get('Content-Length'), String(bytes.length))
      assert.equal(response.headers.get('Cache-Control'), 'private, no-store')
      assert.equal(response.headers.get('X-Content-Type-Options'), 'nosniff')
      assert.equal(response.headers.get('Cross-Origin-Resource-Policy'), 'same-origin')
      assert.equal(response.headers.get('Access-Control-Allow-Origin'), null)
      assert.equal(response.headers.get('Set-Cookie'), null)
      assert.equal(await response.text(), bytes)
      const head = await serve({ Range: 'bytes=0-1' }, 'HEAD')
      assert.equal(head.status, 200)
      assert.equal(head.headers.get('Content-Length'), String(bytes.length))
      assert.equal(await head.text(), '')
    })
    for (const [range, expected, contentRange] of [
      ['bytes=0-4', '%PDF-', 'bytes 0-4/19'],
      ['bytes=14-', '56789', 'bytes 14-18/19'],
      ['bytes=-3', '789', 'bytes 16-18/19'],
      ['bytes=17-99', '89', 'bytes 17-18/19'],
      ['bytes=-99', bytes, 'bytes 0-18/19'],
    ])
      await t.test(range, async () => {
        const response = await serve({ Range: range })
        assert.equal(response.status, 206)
        assert.equal(response.headers.get('Content-Range'), contentRange)
        assert.equal(response.headers.get('Content-Length'), String(expected.length))
        assert.equal(await response.text(), expected)
      })
    await t.test('range errors, validators and If-Range preserve HTTP semantics', async () => {
      for (const range of ['bytes=19-', 'bytes=-0']) {
        const response = await serve({ Range: range })
        assert.equal(response.status, 416)
        assert.equal(response.headers.get('Content-Range'), 'bytes */19')
      }
      for (const range of ['bytes=0-1,4-5', 'items=0-1', 'bytes=9-3']) {
        const response = await serve({ Range: range })
        assert.equal(response.status, 200)
        assert.equal(await response.text(), bytes)
      }
      assert.equal((await serve({ 'If-None-Match': `W/${metadata.httpEtag}` })).status, 304)
      assert.equal((await serve({ 'If-None-Match': '*' })).status, 304)
      assert.equal(
        (await serve({ 'If-Modified-Since': metadata.uploaded.toUTCString() })).status,
        304,
      )
      const partial = await serve({ Range: 'bytes=0-4', 'If-Range': metadata.httpEtag })
      assert.equal(partial.status, 206)
      assert.equal(await partial.text(), '%PDF-')
      const full = await serve({ Range: 'bytes=0-4', 'If-Range': '"stale"' })
      assert.equal(full.status, 200)
      assert.equal(await full.text(), bytes)
    })
    await t.test(
      'uncached HEAD and unsupported methods cannot initiate source fetching',
      async () => {
        const missing = { ...resource, resourceId: 'missing', sourceUrl: 'http://127.0.0.1/secret' }
        const head = await serveArenaReaderResource(
          new Request('https://garden.example/resource', { method: 'HEAD' }),
          bucket,
          missing,
        )
        assert.equal(head.status, 404)
        assert.equal(head.headers.get('Cache-Control'), 'private, no-store')
        assert.equal((await serve({}, 'POST')).status, 405)
        const blocked = await serveArenaReaderResource(
          new Request('https://garden.example/resource'),
          bucket,
          missing,
        )
        assert.equal(blocked.status, 400)
        assert.equal((await blocked.json()).error, 'blocked-source')
      },
    )
    await t.test('MIME failures and partial upstream files are never persisted', async () => {
      for (const [mime, body, purpose, status] of [
        ['image/svg+xml', '<svg/>', 'image', 200],
        ['text/html', '<html/>', 'image', 200],
        ['application/octet-stream', bytes, 'pdf', 200],
        ['application/pdf', '<html/>', 'pdf', 200],
        ['application/pdf', bytes, 'pdf', 206],
      ] satisfies Array<[string, string, 'image' | 'pdf', number]>) {
        const rejected = { ...resource, resourceId: 'rejected', purpose }
        await assert.rejects(
          saveArenaReaderResource(
            bucket,
            rejected,
            new Response(body, { status, headers: { 'Content-Type': mime } }),
          ),
          /supported|PDF/,
        )
        assert.equal(
          await bucket.head('arena-reader/v1/article-1/resources/snapshot-1/rejected'),
          null,
        )
      }
    })
    await t.test(
      'oversize declared and chunked bodies cancel without a partial cache entry',
      async () => {
        let cancelled = false
        const oversized = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new Uint8Array(8 * 1024 * 1024))
          },
          pull(controller) {
            controller.enqueue(new Uint8Array(1))
          },
          cancel() {
            cancelled = true
          },
        })
        const image = {
          ...resource,
          resourceId: 'oversize',
          purpose: 'image',
        } satisfies ArenaReaderResource
        await assert.rejects(
          saveArenaReaderResource(
            bucket,
            image,
            new Response(oversized, { headers: { 'Content-Type': 'image/png' } }),
          ),
          /larger file/,
        )
        assert.equal(cancelled, true)
        assert.equal(
          await bucket.head('arena-reader/v1/article-1/resources/snapshot-1/oversize'),
          null,
        )
        await assert.rejects(
          saveArenaReaderResource(
            bucket,
            resource,
            new Response('%PDF-', {
              headers: {
                'Content-Type': 'application/pdf',
                'Content-Length': String(33 * 1024 * 1024),
              },
            }),
          ),
          /larger file/,
        )
        assert.equal(await (await bucket.get(key))?.text(), bytes)
      },
    )
    await t.test('raster images persist separately from their article snapshot', async () => {
      const png = Uint8Array.from(
        Buffer.from(
          'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==',
          'base64',
        ),
      )
      const image = {
        ...resource,
        resourceId: 'figure-1',
        purpose: 'image',
      } satisfies ArenaReaderResource
      await saveArenaReaderResource(
        bucket,
        image,
        new Response(png, { headers: { 'Content-Type': 'image/png; charset=binary' } }),
      )
      const response = await serveArenaReaderResource(
        new Request('https://garden.example/image'),
        bucket,
        image,
      )
      assert.equal(response.headers.get('Content-Type'), 'image/png')
      assert.deepEqual(new Uint8Array(await response.arrayBuffer()), png)
    })
    await t.test('the first saved copy remains immutable within its snapshot', async () => {
      await saveArenaReaderResource(
        bucket,
        resource,
        new Response('%PDF-1.7\nupdated', { headers: { 'Content-Type': 'application/pdf' } }),
      )
      assert.equal(await (await bucket.get(key))?.text(), bytes)
    })
    await t.test('resource keys cannot escape their article snapshot prefix', async () => {
      const response = await serveArenaReaderResource(
        new Request('https://garden.example/resource'),
        bucket,
        { ...resource, resourceId: '../paper' },
      )
      assert.equal(response.status, 400)
      assert.equal((await response.json()).error, 'invalid-resource')
    })
  } finally {
    await platform.dispose()
    await rm(root, { recursive: true, force: true })
  }
})
