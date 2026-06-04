import assert from 'node:assert/strict'
import test from 'node:test'
import {
  cacheHeadersForStaticAsset,
  isolationHeadersForStaticAsset,
  requestWithoutCache,
  requestWithoutStaticAssetCache,
  shouldBypassStaticAssetCache,
} from '../../worker/static-assets'

test('marks content-hashed static assets immutable', () => {
  for (const pathname of [
    '/index-47ac7b09.css',
    '/prescript-47ac7b09.js',
    '/static/resource-after-47ac7b09.js',
    '/static/scripts/notebook-runtime.client-47ac7b09.js',
    '/static/scripts/notebook-pyright/typeshed/manifest-47ac7b09.json',
    '/static/scripts/notebook-pyright/typeshed/chunks/0-47ac7b09.json',
    '/static/scripts/emoji/twitter/1f44d-47ac7b09.json',
    '/static/scripts/chunks/chunk-NV3MZTQ2.js',
  ]) {
    assert.deepEqual(cacheHeadersForStaticAsset(pathname, 200), {
      'Cache-Control': 'public, max-age=31536000, immutable',
    })
  }
})

test('keeps the asset manifest volatile', () => {
  assert.deepEqual(cacheHeadersForStaticAsset('/static/scripts/asset-manifest.json', 200), {
    'Cache-Control': 'no-store, no-cache, must-revalidate',
  })
})

test('keeps generated watch css and js volatile', () => {
  for (const pathname of [
    '/index.css',
    '/static/component.css',
    '/static/resource-style.css',
    '/static/resource-style-5.css',
    '/prescript.js',
    '/postscript.js',
    '/static/resource-after.js',
    '/static/resource-after-7.js',
    '/static/scripts/notebook-runtime.worker.js',
  ]) {
    assert.deepEqual(cacheHeadersForStaticAsset(pathname, 200), {
      'Cache-Control': 'no-store, no-cache, must-revalidate',
    })
  }
})

test('does not cache ordinary assets or misses as immutable', () => {
  assert.deepEqual(cacheHeadersForStaticAsset('/static/icon.png', 200), {})
  assert.deepEqual(cacheHeadersForStaticAsset('/static/resource-after-47ac7b09.js', 404), {})
})

test('bypasses validators for volatile generated assets', () => {
  assert.equal(shouldBypassStaticAssetCache('/index.css'), true)
  assert.equal(shouldBypassStaticAssetCache('/static/scripts/asset-manifest.json'), true)
  assert.equal(shouldBypassStaticAssetCache('/static/scripts/script-47ac7b09.js'), false)
  assert.equal(shouldBypassStaticAssetCache('/static/icon.png'), false)

  const request = new Request('https://example.com/index.css', {
    headers: { 'If-None-Match': 'etag', 'If-Modified-Since': 'yesterday', 'X-Test': 'keep' },
  })
  const bypassed = requestWithoutStaticAssetCache(request, '/index.css')

  assert.equal(bypassed.headers.get('If-None-Match'), null)
  assert.equal(bypassed.headers.get('If-Modified-Since'), null)
  assert.equal(bypassed.headers.get('X-Test'), 'keep')
  assert.equal(requestWithoutStaticAssetCache(request, '/static/icon.png'), request)
})

test('strips validators from arbitrary local reload requests', () => {
  const request = new Request('https://example.com/thoughts/radix-attention', {
    headers: { 'If-None-Match': 'etag', 'If-Modified-Since': 'yesterday' },
  })
  const bypassed = requestWithoutCache(request)

  assert.equal(bypassed.headers.get('If-None-Match'), null)
  assert.equal(bypassed.headers.get('If-Modified-Since'), null)
})

test('marks first-party scripts loadable under notebook isolation', () => {
  assert.deepEqual(isolationHeadersForStaticAsset('/static/scripts/script-47ac7b09.js', 200), {
    'Cross-Origin-Resource-Policy': 'same-origin',
  })
  assert.deepEqual(
    isolationHeadersForStaticAsset('/static/scripts/notebook-runtime.worker.js', 200),
    {
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  )
  assert.deepEqual(
    isolationHeadersForStaticAsset('/static/scripts/notebook-runtime.javascript.worker.js', 200),
    {
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  )
  assert.deepEqual(isolationHeadersForStaticAsset('/static/scripts/semantic.worker.js', 200), {
    'Cross-Origin-Resource-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  })
})

test('does not add isolation headers to misses or non-script assets', () => {
  assert.deepEqual(
    isolationHeadersForStaticAsset('/static/scripts/notebook-runtime.worker.js', 404),
    {},
  )
  assert.deepEqual(isolationHeadersForStaticAsset('/static/contentIndex.json', 200), {})
})
