import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  NativeRuntimePackKernel,
  emptyNativeRuntimePackManifest,
  readNativeRuntimePackManifest,
} from './runtime-pack'

describe('native runtime pack manifest', () => {
  test('reads self-hosted runtime pack entries', () => {
    const manifest = readNativeRuntimePackManifest({
      version: 1,
      runtimes: {
        go: { worker: 'go/worker.js', assets: ['go/main.wasm', 'go/wasm_exec.js'] },
        unknown: { worker: 'nope.js' },
      },
    })
    assert.deepStrictEqual(manifest, {
      version: 1,
      runtimes: { go: { worker: 'go/worker.js', assets: ['go/main.wasm', 'go/wasm_exec.js'] } },
    })
  })

  test('rejects malformed manifests', () => {
    assert.strictEqual(readNativeRuntimePackManifest({ version: 2, runtimes: {} }), undefined)
    assert.strictEqual(readNativeRuntimePackManifest({ version: 1, runtimes: [] }), undefined)
    assert.deepStrictEqual(
      readNativeRuntimePackManifest({ version: 1, runtimes: { go: { assets: ['go.wasm'] } } }),
      emptyNativeRuntimePackManifest,
    )
  })

  test('reports missing runtime pack entries from the same-origin manifest', async () => {
    const originalFetch = globalThis.fetch
    globalThis.fetch = async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(emptyNativeRuntimePackManifest), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    try {
      const kernel = new NativeRuntimePackKernel('go', {
        runtimeId: 'r',
        sourcePath: 's',
        workerUrl: 'https://example.test/static/scripts/notebook-runtimes/manifest.json',
      })
      await assert.rejects(
        kernel.init({ signal: new AbortController().signal }),
        /does not list go/,
      )
    } finally {
      globalThis.fetch = originalFetch
    }
  })
})
