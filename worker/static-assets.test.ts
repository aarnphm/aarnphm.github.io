import assert from 'node:assert/strict'
import test from 'node:test'
import {
  cacheHeadersForStaticAsset,
  isolationHeadersForStaticAsset,
  shouldBypassStaticAssetCache,
} from './static-assets'

test('marks fingerprinted mjs, wasm, bin, gz, zst, and tar assets immutable', () => {
  for (const pathname of [
    '/static/scripts/notebook-runtimes/clang/clang-47ac7b09.mjs',
    '/static/scripts/notebook-runtimes/go/yaegi-47ac7b09.wasm',
    '/static/scripts/notebook-runtimes/haskell/rootfs.tar.zst.chunk-0000-47ac7b09.bin',
    '/static/scripts/notebook-runtimes/rust/libstd.rlib-47ac7b09.gz',
    '/static/scripts/notebook-runtimes/haskell/rootfs.tar-47ac7b09.zst',
    '/static/scripts/notebook-runtimes/archive-47ac7b09.tar',
  ]) {
    assert.deepEqual(cacheHeadersForStaticAsset(pathname, 200), {
      'Cache-Control': 'public, max-age=31536000, immutable',
    })
    assert.equal(shouldBypassStaticAssetCache(pathname), false)
  }
})

test('does not treat unhashed binary runtime assets as volatile no-store scripts', () => {
  for (const pathname of [
    '/static/scripts/notebook-runtimes/go/yaegi.wasm',
    '/static/scripts/notebook-runtimes/haskell/rootfs.tar.zst.chunk-0000.bin',
    '/static/scripts/notebook-runtimes/rust/libstd.rlib.gz',
    '/static/scripts/notebook-runtimes/rust/libproc.so.gz',
    '/static/scripts/notebook-runtimes/haskell/rootfs.tar.zst',
  ]) {
    assert.deepEqual(cacheHeadersForStaticAsset(pathname, 200), {})
    assert.equal(shouldBypassStaticAssetCache(pathname), false)
  }
})

test('adds COEP require-corp to hyphenated and subdirectory worker scripts', () => {
  for (const pathname of [
    '/static/scripts/notebook-pyright-worker.js',
    '/static/scripts/notebook-pyright-worker-47ac7b09.js',
    '/static/scripts/notebook-runtimes/clang/worker.js',
    '/static/scripts/notebook-runtimes/clang/worker-47ac7b09.js',
    '/static/scripts/notebook-runtime.worker.js',
    '/static/scripts/notebook-runtime.javascript.worker.js',
  ]) {
    assert.deepEqual(isolationHeadersForStaticAsset(pathname, 200), {
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    })
  }

  assert.deepEqual(
    isolationHeadersForStaticAsset('/static/scripts/notebook-runtimes/clang/clang.mjs', 200),
    { 'Cross-Origin-Resource-Policy': 'same-origin' },
  )
})
