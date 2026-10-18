import type { EditorView } from '@codemirror/view'
import { LSPClient } from '@codemirror/lsp-client'
import { Text } from '@codemirror/state'
import DOMPurify from 'dompurify'
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  NotebookPythonWorkspace,
  hasPythonHoverTokenAt,
  sanitizeNotebookHtml,
  type NotebookLspCell,
} from './pyright'
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

test('NotebookPythonWorkspace preserves lastSource on closeFile and avoids Text allocations on unchanged refresh', () => {
  let cells: NotebookLspCell[] = [
    { id: 'cell-1', source: 'x = 1\ny = x + 2', language: 'python', executionIndex: 1 },
    { id: 'cell-2', source: 'print(y)', language: 'python', executionIndex: 2 },
  ]
  const client = new LSPClient({ rootUri: 'file:///quartz-notebook/rt/' })
  const workspace = new NotebookPythonWorkspace(
    client,
    'file:///quartz-notebook/rt/',
    'notes/demo.md',
    () => cells,
  )

  const initialFilesRef = workspace.files
  const cell1File = workspace.files[0]!
  const initialCell1Doc = cell1File.doc
  assert.equal(cell1File.lastSource, 'x = 1\ny = x + 2')

  const fakeView = { state: { doc: initialCell1Doc } } as unknown as EditorView
  workspace.openFile(cell1File.uri, 'python', fakeView)
  assert.equal(cell1File.view, fakeView)
  assert.equal(cell1File.doc, initialCell1Doc)
  assert.equal(workspace.files, initialFilesRef)

  workspace.closeFile(cell1File.uri, fakeView)
  assert.equal(cell1File.view, null)
  assert.equal(cell1File.lastSource, 'x = 1\ny = x + 2')
  assert.equal(cell1File.doc, initialCell1Doc)

  // Calling getFile with a missing URI triggers refreshFiles(); unchanged cells must keep their Text & files array ref
  assert.equal(workspace.getFile('file:///quartz-notebook/rt/notes%2Fdemo.md/missing.py'), null)
  assert.equal(workspace.files, initialFilesRef)
  assert.equal(workspace.files[0]!.doc, initialCell1Doc)

  // Editing cell while open updates lastSource on closeFile so subsequent refreshFiles() stays a no-op
  const editedSource = 'x = 10\ny = x + 20'
  const editedDoc = Text.of(editedSource.split('\n'))
  const editedView = { state: { doc: editedDoc } } as unknown as EditorView
  workspace.openFile(cell1File.uri, 'python', editedView)
  cells = [
    { id: 'cell-1', source: editedSource, language: 'python', executionIndex: 3 },
    { id: 'cell-2', source: 'print(y)', language: 'python', executionIndex: 2 },
  ]
  workspace.closeFile(cell1File.uri, editedView)
  assert.equal(cell1File.lastSource, editedSource)
  assert.equal(cell1File.doc, editedDoc)

  assert.equal(workspace.getFile('file:///quartz-notebook/rt/notes%2Fdemo.md/missing.py'), null)
  assert.equal(workspace.files, initialFilesRef)
  assert.equal(workspace.files[0]!.doc, editedDoc)
  assert.equal(workspace.files[0]!.executionIndex, 3)
})

test('hasPythonHoverTokenAt skips whitespace, punctuation, and comments while matching Python identifiers', () => {
  const source = [
    'import numpy as np  # comment with identifier_in_comment',
    'result = np.array([1, 2, 3]) + 42',
    'msg = "# not a comment" + value_after_hash_string',
    '   ',
  ].join('\n')
  const doc = Text.of(source.split('\n'))

  const line1 = doc.line(1)
  assert.equal(hasPythonHoverTokenAt(doc, line1.from + line1.text.indexOf('numpy')), true)
  assert.equal(hasPythonHoverTokenAt(doc, line1.from + line1.text.indexOf('np')), true)
  assert.equal(
    hasPythonHoverTokenAt(doc, line1.from + line1.text.indexOf('identifier_in_comment')),
    false,
  )
  assert.equal(hasPythonHoverTokenAt(doc, line1.from + line1.text.indexOf('#')), false)

  const line2 = doc.line(2)
  assert.equal(hasPythonHoverTokenAt(doc, line2.from + line2.text.indexOf('result')), true)
  assert.equal(hasPythonHoverTokenAt(doc, line2.from + line2.text.indexOf('array')), true)
  assert.equal(hasPythonHoverTokenAt(doc, line2.from + line2.text.indexOf(' = ') + 1), false)
  assert.equal(hasPythonHoverTokenAt(doc, line2.from + line2.text.indexOf('([') + 1), false)

  const line3 = doc.line(3)
  assert.equal(
    hasPythonHoverTokenAt(doc, line3.from + line3.text.indexOf('value_after_hash_string')),
    true,
  )

  const line4 = doc.line(4)
  assert.equal(hasPythonHoverTokenAt(doc, line4.from + 1), false)
})

test('sanitizeNotebookHtml caches sanitized HTML and bounds cache size to 256 entries', () => {
  const descriptor = Object.getOwnPropertyDescriptor(DOMPurify, 'sanitize')
  let calls = 0
  Object.defineProperty(DOMPurify, 'sanitize', {
    configurable: true,
    writable: true,
    value: (html: string) => {
      calls += 1
      return `sanitized:${html}`
    },
  })

  try {
    const first = sanitizeNotebookHtml('<p>alpha</p>')
    const second = sanitizeNotebookHtml('<p>alpha</p>')
    assert.equal(first, 'sanitized:<p>alpha</p>')
    assert.equal(second, 'sanitized:<p>alpha</p>')
    assert.equal(calls, 1)

    for (let i = 0; i < 256; i++) {
      sanitizeNotebookHtml(`<p>item-${i}</p>`)
    }
    const callsBeforeEvictedLookup = calls
    sanitizeNotebookHtml('<p>item-255</p>')
    assert.equal(calls, callsBeforeEvictedLookup)

    sanitizeNotebookHtml('<p>alpha</p>')
    assert.equal(calls, callsBeforeEvictedLookup + 1)
  } finally {
    if (descriptor) {
      Object.defineProperty(DOMPurify, 'sanitize', descriptor)
    } else {
      Reflect.deleteProperty(DOMPurify, 'sanitize')
    }
  }
})
