import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../../util/ctx'
import { resolveAsset } from '../../../util/asset-manifest'
import {
  chunkedFileAssetChunkPath,
  chunkedFileAssetChunkReference,
  chunkedFileAssetManifestPath,
  type ChunkedFileAssetDescriptor,
  writeChunkedFileAsset,
} from './chunked-file-assets'

function testCtx(output: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: 'content',
      verbose: false,
      output,
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {} as BuildCtx['cfg'],
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

test('derives chunked file asset paths from a directory descriptor', () => {
  const descriptor: ChunkedFileAssetDescriptor = {
    baseDir: 'static/scripts/notebook-pyright/typeshed',
    manifestName: 'manifest.json',
    chunkDir: 'chunks',
    maxBytes: 1024,
  }

  assert.equal(
    chunkedFileAssetManifestPath(descriptor),
    'static/scripts/notebook-pyright/typeshed/manifest.json',
  )
  assert.equal(
    chunkedFileAssetChunkPath(descriptor, 7),
    'static/scripts/notebook-pyright/typeshed/chunks/7.json',
  )
  assert.equal(
    chunkedFileAssetChunkReference(
      descriptor,
      'static/scripts/notebook-pyright/typeshed/chunks/7-12345678.json',
    ),
    'chunks/7-12345678.json',
  )
})

test('registers content-hashed chunk paths as manifest-relative entries', async () => {
  const output = await mkdtemp(path.join(os.tmpdir(), 'garden-chunked-file-assets-'))
  try {
    const ctx = testCtx(output)
    const descriptor: ChunkedFileAssetDescriptor = {
      baseDir: 'static/scripts/notebook-pyright/site-packages',
      manifestName: 'manifest.json',
      chunkDir: 'chunks',
      maxBytes: 1024,
    }

    await writeChunkedFileAsset(ctx, descriptor, [
      JSON.stringify({ files: { '/site-packages/numpy/__init__.pyi': 'from typing import Any' } }),
    ])

    const logicalChunk = chunkedFileAssetChunkPath(descriptor, 0)
    const emittedChunk = resolveAsset(ctx, logicalChunk)
    assert.match(
      emittedChunk,
      /^static\/scripts\/notebook-pyright\/site-packages\/chunks\/0-[0-9a-f]{8}\.json$/,
    )
    assert.match(
      chunkedFileAssetChunkReference(descriptor, emittedChunk),
      /^chunks\/0-[0-9a-f]{8}\.json$/,
    )
    assert.match(
      resolveAsset(ctx, chunkedFileAssetManifestPath(descriptor)),
      /^static\/scripts\/notebook-pyright\/site-packages\/manifest-[0-9a-f]{8}\.json$/,
    )
  } finally {
    await rm(output, { recursive: true, force: true })
  }
})
