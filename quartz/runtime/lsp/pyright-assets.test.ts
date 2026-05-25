import assert from 'node:assert/strict'
import test from 'node:test'
import {
  chunkNotebookPyrightTypeshedFiles,
  notebookPyrightAssetManifestChunks,
  notebookPyrightAssetManifestEntry,
  notebookPyrightPackageStubFiles,
  notebookPyrightPyodidePackageImports,
  notebookPyrightTypeshedFiles,
} from './pyright-assets'

const encoder = new TextEncoder()

function jsonBytes(value: unknown): number {
  return encoder.encode(JSON.stringify(value)).byteLength
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
    notebookPyrightAssetManifestChunks(
      { chunks: ['chunks/0.json', 'chunks/1.json'] },
      'test asset',
    ),
    ['chunks/0.json', 'chunks/1.json'],
  )
  assert.equal(
    notebookPyrightAssetManifestEntry({ entry: 'notebook-pyright-worker.js' }, 'test worker'),
    'notebook-pyright-worker.js',
  )
  assert.deepEqual(
    notebookPyrightTypeshedFiles({ files: { '/typeshed/stdlib/os.pyi': 'class X: ...' } }),
    { '/typeshed/stdlib/os.pyi': 'class X: ...' },
  )
  assert.throws(() => notebookPyrightAssetManifestChunks({ chunks: [] }, 'test asset'))
  assert.throws(() => notebookPyrightAssetManifestEntry({ entry: '' }, 'test worker'))
  assert.throws(() => notebookPyrightTypeshedFiles({ files: { '/bad.pyi': { nested: true } } }))
})

test('reads pyodide package imports for notebook completion stubs', () => {
  assert.deepEqual(
    notebookPyrightPyodidePackageImports({
      packages: {
        numpy: { imports: ['numpy'] },
        matplotlib: { imports: ['pylab', 'mpl_toolkits', 'matplotlib'] },
        invalid: { imports: ['bad-name', 'also bad'] },
        duplicate: { imports: ['numpy'] },
        empty: { imports: [] },
      },
    }),
    ['matplotlib', 'mpl_toolkits', 'numpy', 'pylab'],
  )
  assert.throws(() => notebookPyrightPyodidePackageImports({ packages: [] }))
})

test('creates package stubs under notebook site-packages', () => {
  const packageFiles = notebookPyrightPackageStubFiles(
    '/runtime/site-packages',
    ['numpy', 'bad-name', 'PIL', 'numpy'],
    'from typing import Any\n',
  )
  assert.deepEqual(Object.keys(packageFiles), [
    '/runtime/site-packages/numpy/__init__.pyi',
    '/runtime/site-packages/PIL/__init__.pyi',
  ])
  assert.equal(
    packageFiles['/runtime/site-packages/numpy/__init__.pyi'],
    'from typing import Any\n',
  )
})
