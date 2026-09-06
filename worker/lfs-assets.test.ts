import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { getPlatformProxy } from 'wrangler'
import {
  getObjectFromBucket,
  getObjectInfo,
  lfsAssetContentType,
  lfsPointerRequest,
} from './lfs-assets'

test('reads a chunked LFS pointer independently of playback range and validators', async () => {
  const oid = 'a'.repeat(64)
  const pointer = `version https://git-lfs.github.com/spec/v1\noid sha256:${oid}\nsize 2048\n`
  const body = new ReadableStream({
    start(controller) {
      for (const character of pointer) controller.enqueue(new TextEncoder().encode(character))
      controller.close()
    },
  })
  assert.deepEqual(await getObjectInfo(new Response(body)), {
    hash_algo: 'sha256',
    oid,
    size: 2048,
  })
  assert.equal(await getObjectInfo(new Response('audio bytes')), null)
  assert.equal(await getObjectInfo(new Response('x'.repeat(2048))), null)
  const request = lfsPointerRequest(
    new Request('https://example.com/triathlon/memos/day.m4a', {
      method: 'HEAD',
      headers: { Range: 'bytes=500-', 'If-None-Match': 'pointer-etag' },
    }),
  )
  assert.equal(request.method, 'GET')
  assert.equal(request.headers.get('Range'), null)
  assert.equal(request.headers.get('If-None-Match'), null)
  assert.equal(lfsAssetContentType('/triathlon/memos/day.m4a'), 'audio/mp4')
  assert.equal(lfsAssetContentType('/paper.pdf'), 'application/pdf')
  assert.equal(lfsAssetContentType('/triathlon/memos/day.peaks.json'), undefined)
  assert.equal(lfsAssetContentType('/triathlon/memos/day.qta'), undefined)
  assert.equal(lfsAssetContentType('/triathlon/memos/day.waveform'), undefined)
  assert.equal(lfsAssetContentType('/other/day.m4a'), undefined)
})

test('streams real local R2 objects with HTTP range and conditional semantics', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'memo-r2-test-'))
  await writeFile(
    path.join(root, 'wrangler.json'),
    JSON.stringify({
      name: 'memo-r2-test',
      compatibility_date: '2025-01-21',
      r2_buckets: [{ binding: 'LFS_BUCKET', bucket_name: 'test-memos' }],
    }),
  )
  const platform = await getPlatformProxy<{ LFS_BUCKET: R2Bucket }>({
    configPath: path.join(root, 'wrangler.json'),
    persist: false,
    remoteBindings: false,
  })
  try {
    const bucket = platform.env.LFS_BUCKET
    const object = await bucket.put('memo', '0123456789', {
      httpMetadata: { contentType: 'application/octet-stream' },
    })
    assert.ok(object)
    const serve = (headers: HeadersInit = {}, method = 'GET') =>
      getObjectFromBucket(
        bucket,
        'memo',
        new Request('https://example.com/triathlon/memos/day.m4a', { headers, method }),
        'audio/mp4',
      )
    await t.test('full content and HEAD have the audio MIME and object length', async () => {
      const response = await serve()
      assert.equal(response.status, 200)
      assert.equal(response.headers.get('Content-Type'), 'audio/mp4')
      assert.equal(response.headers.get('Content-Length'), '10')
      assert.equal(response.headers.get('Accept-Ranges'), 'bytes')
      assert.equal(await response.text(), '0123456789')
      const head = await serve({ Range: 'bytes=2-3' }, 'HEAD')
      assert.equal(head.status, 200)
      assert.equal(head.headers.get('Content-Length'), '10')
      assert.equal(await head.text(), '')
    })
    for (const [range, expected, contentRange] of [
      ['bytes=0-1', '01', 'bytes 0-1/10'],
      ['bytes=5-', '56789', 'bytes 5-9/10'],
      ['bytes=-3', '789', 'bytes 7-9/10'],
      ['bytes=8-99', '89', 'bytes 8-9/10'],
      ['bytes=-99', '0123456789', 'bytes 0-9/10'],
    ]) {
      await t.test(range, async () => {
        const response = await serve({ Range: range })
        assert.equal(response.status, 206)
        assert.equal(response.headers.get('Content-Range'), contentRange)
        assert.equal(response.headers.get('Content-Length'), String(expected.length))
        assert.equal(await response.text(), expected)
      })
    }
    await t.test('rejects unsatisfiable ranges and ignores unsupported multi-ranges', async () => {
      for (const range of ['bytes=10-', 'bytes=-0']) {
        const response = await serve({ Range: range })
        assert.equal(response.status, 416)
        assert.equal(response.headers.get('Content-Range'), 'bytes */10')
        assert.equal(await response.text(), '')
      }
      for (const range of ['bytes=0-1,4-5', 'bytes=5-2', 'items=0-1']) {
        const response = await serve({ Range: range })
        assert.equal(response.status, 200)
        assert.equal(await response.text(), '0123456789')
      }
    })
    await t.test('revalidates the object and honors If-Range', async () => {
      assert.equal((await serve({ 'If-None-Match': `W/${object.httpEtag}` })).status, 304)
      assert.equal((await serve({ 'If-None-Match': '*' })).status, 304)
      assert.equal(
        (await serve({ 'If-Modified-Since': object.uploaded.toUTCString() })).status,
        304,
      )
      const partial = await serve({ Range: 'bytes=0-1', 'If-Range': object.httpEtag })
      assert.equal(partial.status, 206)
      assert.equal(await partial.text(), '01')
      const full = await serve({ Range: 'bytes=0-1', 'If-Range': '"stale"' })
      assert.equal(full.status, 200)
      assert.equal(await full.text(), '0123456789')
    })
    await t.test('missing objects return 404', async () => {
      const response = await getObjectFromBucket(
        bucket,
        'missing',
        new Request('https://example.com/missing'),
        'audio/mp4',
      )
      assert.equal(response.status, 404)
    })
  } finally {
    await platform.dispose()
    await rm(root, { recursive: true, force: true })
  }
})
