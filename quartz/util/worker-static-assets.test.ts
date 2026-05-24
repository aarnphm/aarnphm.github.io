import assert from 'node:assert/strict'
import test from 'node:test'
import {
  cacheHeadersForStaticAsset,
  isolationHeadersForStaticAsset,
} from '../../worker/static-assets'

test('marks content-hashed static assets immutable', () => {
  for (const pathname of [
    '/index-47ac7b09.css',
    '/prescript-47ac7b09.js',
    '/static/resource-after-47ac7b09.js',
    '/static/scripts/notebook-runtime.client-47ac7b09.js',
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

test('does not cache unhashed assets or misses as immutable', () => {
  assert.deepEqual(cacheHeadersForStaticAsset('/postscript.js', 200), {})
  assert.deepEqual(cacheHeadersForStaticAsset('/static/resource-after-47ac7b09.js', 404), {})
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
