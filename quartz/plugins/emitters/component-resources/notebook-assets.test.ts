import assert from 'node:assert/strict'
import { Buffer } from 'node:buffer'
import test from 'node:test'
import {
  nativeRuntimePackChunkBytes,
  nativeRuntimePackFileChunks,
  readNativeRuntimePackGitLfsPointer,
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
