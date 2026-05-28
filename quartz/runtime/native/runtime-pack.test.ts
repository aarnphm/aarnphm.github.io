import assert from 'node:assert'
import test, { describe } from 'node:test'
import type { RuntimeEvent } from '../notebook/kernel'
import { codeCellId } from '../../util/notebook/identity'
import {
  NativeRuntimePackKernel,
  emptyNativeRuntimePackManifest,
  isNativeRuntimePackAvailableEntry,
  readNativeRuntimePackManifest,
} from './runtime-pack'

describe('native runtime pack manifest', () => {
  test('reads self-hosted runtime pack entries', () => {
    const manifest = readNativeRuntimePackManifest({
      version: 1,
      runtimes: {
        c: { worker: 'clang/worker.js', assets: ['clang/bundle.js', 'clang/llvm.core.wasm'] },
        cpp: { worker: 'clang/worker.js', assets: ['clang/bundle.js', 'clang/llvm.core.wasm'] },
        go: { worker: 'go/worker.js', assets: ['go/main.wasm', 'go/wasm_exec.js'] },
        haskell: { available: false, reason: 'Haskell runtime pack is not shipped yet' },
        ocaml: { worker: 'ocaml/worker.js', assets: ['ocaml/toplevel.js'] },
        wasm: { worker: 'wasm/worker.js', assets: ['wasm/wabt.mjs'] },
        unknown: { worker: 'nope.js' },
      },
    })
    assert.ok(manifest)
    assert.deepStrictEqual(manifest, {
      version: 1,
      runtimes: {
        c: { worker: 'clang/worker.js', assets: ['clang/bundle.js', 'clang/llvm.core.wasm'] },
        cpp: { worker: 'clang/worker.js', assets: ['clang/bundle.js', 'clang/llvm.core.wasm'] },
        go: { worker: 'go/worker.js', assets: ['go/main.wasm', 'go/wasm_exec.js'] },
        haskell: { available: false, reason: 'Haskell runtime pack is not shipped yet' },
        ocaml: { worker: 'ocaml/worker.js', assets: ['ocaml/toplevel.js'] },
        wasm: { worker: 'wasm/worker.js', assets: ['wasm/wabt.mjs'] },
      },
    })
    const goEntry = manifest.runtimes.go
    const haskellEntry = manifest.runtimes.haskell
    assert.ok(goEntry)
    assert.ok(haskellEntry)
    assert.strictEqual(isNativeRuntimePackAvailableEntry(goEntry), true)
    assert.strictEqual(isNativeRuntimePackAvailableEntry(haskellEntry), false)
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

  test('reports unavailable runtime pack entries as execution failures', async () => {
    const reason = 'Rust runtime pack is not shipped yet'
    const originalFetch = globalThis.fetch
    globalThis.fetch = async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(
        JSON.stringify({ version: 1, runtimes: { rust: { available: false, reason } } }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      )
    try {
      const kernel = new NativeRuntimePackKernel('rust', {
        runtimeId: 'r',
        sourcePath: 's',
        workerUrl: 'https://example.test/static/scripts/notebook-runtimes/manifest.json',
      })
      const cellId = codeCellId(1)
      const events: RuntimeEvent[] = []
      for await (const event of kernel.execute(cellId, 'fn main() {}')) events.push(event)
      assert.deepStrictEqual(events, [
        { type: 'started', cellId },
        {
          type: 'output',
          cellId,
          output: {
            type: 'error',
            ename: 'UnsupportedRuntimeFeature',
            evalue: reason,
            traceback: reason,
          },
        },
        { type: 'done', cellId, executionCount: null, failed: true },
      ])
    } finally {
      globalThis.fetch = originalFetch
    }
  })
})
