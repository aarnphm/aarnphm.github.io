import assert from 'node:assert/strict'
import test from 'node:test'
import {
  chunkNotebookPyrightTextAsset,
  chunkNotebookPyrightTypeshedFiles,
  notebookPyrightAssetManifestChunks,
  notebookPyrightTypeshedFiles,
} from './notebook-pyright-assets'

const encoder = new TextEncoder()

function jsonBytes(value: unknown): number {
  return encoder.encode(JSON.stringify(value)).byteLength
}

function textBytes(value: string): number {
  return encoder.encode(value).byteLength
}

test('chunks notebook pyright typeshed files under the byte budget', () => {
  const files = {
    '/typeshed/stdlib/b.pyi': 'b'.repeat(24),
    '/typeshed/stdlib/a.pyi': 'a'.repeat(24),
    '/typeshed/stdlib/c.pyi': 'c'.repeat(24),
  }
  const maxBytes = jsonBytes({
    files: {
      '/typeshed/stdlib/a.pyi': files['/typeshed/stdlib/a.pyi'],
      '/typeshed/stdlib/b.pyi': files['/typeshed/stdlib/b.pyi'],
    },
  })

  const chunks = chunkNotebookPyrightTypeshedFiles(files, maxBytes)

  assert.equal(chunks.length, 2)
  assert.deepEqual(Object.keys(chunks[0].files), [
    '/typeshed/stdlib/a.pyi',
    '/typeshed/stdlib/b.pyi',
  ])
  assert.deepEqual(Object.keys(chunks[1].files), ['/typeshed/stdlib/c.pyi'])
  assert(chunks.every(chunk => jsonBytes({ files: chunk.files }) <= maxBytes))
})

test('chunks notebook pyright text assets on utf-8 boundaries', () => {
  const chunks = chunkNotebookPyrightTextAsset('abc\u{1f9ea}def', 4)

  assert.deepEqual(chunks, ['abc', '\u{1f9ea}', 'def'])
  assert.equal(chunks.join(''), 'abc\u{1f9ea}def')
  assert(chunks.every(chunk => textBytes(chunk) <= 4))
})

test('allows one oversized typeshed file to occupy its own chunk', () => {
  const chunks = chunkNotebookPyrightTypeshedFiles(
    { '/typeshed/stdlib/huge.pyi': 'x'.repeat(200), '/typeshed/stdlib/small.pyi': 'y' },
    96,
  )

  assert.equal(chunks.length, 2)
  assert.deepEqual(Object.keys(chunks[0].files), ['/typeshed/stdlib/huge.pyi'])
  assert.deepEqual(Object.keys(chunks[1].files), ['/typeshed/stdlib/small.pyi'])
})

test('reads notebook pyright typeshed manifests and chunks', () => {
  assert.deepEqual(
    notebookPyrightAssetManifestChunks({ chunks: ['a.json', 'b.json'] }, 'test asset'),
    ['a.json', 'b.json'],
  )
  assert.deepEqual(
    notebookPyrightTypeshedFiles({ files: { '/typeshed/stdlib/os.pyi': 'class X: ...' } }),
    { '/typeshed/stdlib/os.pyi': 'class X: ...' },
  )
  assert.throws(() => notebookPyrightAssetManifestChunks({ chunks: [] }, 'test asset'))
  assert.throws(() => notebookPyrightTypeshedFiles({ files: { '/bad.pyi': { nested: true } } }))
})
