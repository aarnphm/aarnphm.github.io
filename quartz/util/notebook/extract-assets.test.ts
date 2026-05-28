import assert from 'node:assert'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test, { describe } from 'node:test'
import type { BuildCtx } from '../ctx'
import { resetWriteCache } from '../../plugins/emitters/helpers'
import {
  extractInlineNotebookAssets,
  notebookAssetDirectory,
  notebookAssetMinBytes,
} from './extract-assets'

function makeCtx(): { ctx: BuildCtx; output: string } {
  const output = mkdtempSync(join(tmpdir(), 'notebook-assets-test-'))
  const ctx = {
    buildId: 'test-build',
    argv: { directory: output, output, verbose: false, serve: false, watch: false },
    cfg: { configuration: { ignorePatterns: [] } },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  } as unknown as BuildCtx
  return { ctx, output }
}

function bigPngDataUri(): string {
  const bytes = new Uint8Array(notebookAssetMinBytes + 256)
  for (let i = 0; i < bytes.length; i += 1) bytes[i] = i % 256
  const base64 = Buffer.from(bytes).toString('base64')
  return `data:image/png;base64,${base64}`
}

function smallPngDataUri(): string {
  const bytes = new Uint8Array(64).fill(1)
  const base64 = Buffer.from(bytes).toString('base64')
  return `data:image/png;base64,${base64}`
}

describe('extractInlineNotebookAssets', () => {
  test('replaces large data URIs with file references and writes them', async () => {
    const { ctx, output } = makeCtx()
    try {
      const uri = bigPngDataUri()
      const chunk = `<img src="${uri}">`
      const result = await extractInlineNotebookAssets([chunk], ctx)
      assert.strictEqual(result.extracted.length, 1)
      const extracted = result.extracted[0]
      assert.strictEqual(extracted.mime, 'image/png')
      assert.match(extracted.url, /^\/static\/notebook-assets\/[a-f0-9]+\.png$/)
      assert.ok(result.chunks[0].includes(extracted.url))
      const onDisk = join(output, notebookAssetDirectory, `${extracted.hash}.png`)
      assert.ok(existsSync(onDisk))
      const written = readFileSync(onDisk)
      assert.strictEqual(written.byteLength, extracted.byteLength)
    } finally {
      rmSync(output, { recursive: true, force: true })
    }
  })

  test('keeps small images inline', async () => {
    const { ctx, output } = makeCtx()
    try {
      const uri = smallPngDataUri()
      const chunk = `<img src="${uri}">`
      const result = await extractInlineNotebookAssets([chunk], ctx)
      assert.strictEqual(result.extracted.length, 0)
      assert.strictEqual(result.chunks[0], chunk)
    } finally {
      rmSync(output, { recursive: true, force: true })
    }
  })

  test('deduplicates writes for identical payloads across chunks', async () => {
    const { ctx, output } = makeCtx()
    try {
      const uri = bigPngDataUri()
      const chunks = [
        `<img src="${uri}">`,
        `<img src="${uri}">`,
        `paragraph with <img src="${uri}">`,
      ]
      const result = await extractInlineNotebookAssets(chunks, ctx)
      assert.strictEqual(result.extracted.length, 3)
      const hashes = new Set(result.extracted.map(asset => asset.hash))
      assert.strictEqual(hashes.size, 1)
    } finally {
      rmSync(output, { recursive: true, force: true })
    }
  })

  test('deduplicates writes only within the current build', async () => {
    const { ctx, output } = makeCtx()
    try {
      const uri = bigPngDataUri()
      const chunk = `<img src="${uri}">`
      const result = await extractInlineNotebookAssets([chunk], ctx)
      const extracted = result.extracted[0]
      const onDisk = join(output, notebookAssetDirectory, `${extracted.hash}.png`)
      assert.ok(existsSync(onDisk))

      rmSync(join(output, notebookAssetDirectory), { recursive: true, force: true })
      await extractInlineNotebookAssets([chunk], ctx)
      assert.strictEqual(existsSync(onDisk), false)

      ctx.buildId = 'next-build'
      resetWriteCache()
      await extractInlineNotebookAssets([chunk], ctx)
      assert.ok(existsSync(onDisk))
    } finally {
      rmSync(output, { recursive: true, force: true })
    }
  })
})
