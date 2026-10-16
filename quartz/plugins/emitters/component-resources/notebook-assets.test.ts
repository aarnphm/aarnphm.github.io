import assert from 'node:assert/strict'
import { Buffer } from 'node:buffer'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../../util/ctx'
import { assetManifestRecord } from '../../../util/asset-manifest'
import { getPreservedOutputFiles, resetOutputAssetClaims } from '../../../util/output-assets'
import {
  nativeRuntimePackChunkBytes,
  nativeRuntimePackFileChunks,
  readNativeRuntimePackGitLfsPointer,
  resetNotebookRuntimeAssetCache,
  writeNotebookPyrightAssets,
} from './notebook-assets'

test('chunks native runtime pack files under the Cloudflare asset budget', () => {
  const content = Buffer.alloc(nativeRuntimePackChunkBytes * 2 + 11, 7)
  const chunks = nativeRuntimePackFileChunks(content)

  assert.equal(chunks.length, 3)
  assert(chunks.every(chunk => chunk.byteLength <= nativeRuntimePackChunkBytes))
  assert.deepEqual(Buffer.concat([...chunks]), content)
})

test('reads Git LFS pointers for native runtime pack files', () => {
  const oid = '35f68f56fd'.padEnd(64, '0')
  const pointer = Buffer.from(
    `version https://git-lfs.github.com/spec/v1\noid sha256:${oid}\nsize 4919\n`,
  )

  assert.deepEqual(readNativeRuntimePackGitLfsPointer(pointer, 'haskell/rootfs.tar.zst'), {
    oid,
    size: 4919,
  })
  assert.equal(
    readNativeRuntimePackGitLfsPointer(Buffer.from('plain wasm bytes'), 'go/yaegi.wasm'),
    undefined,
  )
  assert.throws(() =>
    readNativeRuntimePackGitLfsPointer(
      Buffer.from('version https://git-lfs.github.com/spec/v1\noid sha256:nope\n'),
      'haskell/rootfs.tar.zst',
    ),
  )
})

test('caches writeNotebookPyrightAssets on globalThis and restores assetManifest and claims', async () => {
  const output = await mkdtemp(path.join(tmpdir(), 'quartz-notebook-pyright-cache-'))
  try {
    resetNotebookRuntimeAssetCache()
    resetOutputAssetClaims()
    const makeCtx = (buildId: string): BuildCtx =>
      ({
        buildId,
        argv: {
          directory: 'content',
          output,
          verbose: false,
          serve: false,
          watch: true,
          port: 8080,
          wsPort: 3001,
          force: false,
        },
        cfg: {
          configuration: { ignorePatterns: [] },
          plugins: { transformers: [], filters: [], emitters: [] },
        },
        allSlugs: [],
        allFiles: [],
        incremental: false,
      }) as unknown as BuildCtx

    const ctx1 = makeCtx('build-1')
    const firstFiles = await writeNotebookPyrightAssets(ctx1)
    const firstManifest = assetManifestRecord(ctx1)
    const firstStat = await stat(firstFiles[0])

    resetOutputAssetClaims()
    const ctx2 = makeCtx('build-2')
    const secondFiles = await writeNotebookPyrightAssets(ctx2)
    const secondManifest = assetManifestRecord(ctx2)
    const secondStat = await stat(secondFiles[0])

    assert.deepEqual(secondFiles, firstFiles)
    assert.deepEqual(secondManifest, firstManifest)
    assert.equal(secondStat.mtimeMs, firstStat.mtimeMs)
    const preserved = getPreservedOutputFiles()
    for (const file of secondFiles) {
      assert.equal(preserved.has(path.resolve(file) as typeof file), true)
    }
  } finally {
    resetNotebookRuntimeAssetCache()
    resetOutputAssetClaims()
    await rm(output, { recursive: true, force: true })
  }
})
