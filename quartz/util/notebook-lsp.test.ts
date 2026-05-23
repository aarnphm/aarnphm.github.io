import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  notebookDocumentPath,
  notebookDocumentUri,
  notebookWorkspaceRootUri,
} from './notebook-lsp-uri'

describe('notebook lsp uri', () => {
  test('produces a stable synthetic module uri for runtime id', () => {
    const uri = notebookDocumentUri('abc123')
    assert.strictEqual(uri, 'file:///notebook/abc123/notebook.py')
  })

  test('document path and uri preserve runtime id', () => {
    assert.strictEqual(notebookDocumentPath('run-42-abc'), '/notebook/run-42-abc/notebook.py')
    assert.strictEqual(notebookDocumentUri('run-42-abc'), 'file:///notebook/run-42-abc/notebook.py')
  })

  test('workspace root ends with trailing slash and matches document uri prefix', () => {
    const root = notebookWorkspaceRootUri('rt-9')
    const document = notebookDocumentUri('rt-9')
    assert.ok(root.endsWith('/'))
    assert.ok(document.startsWith(root))
  })

  test('document uri and path target one synthetic notebook module', () => {
    assert.strictEqual(notebookDocumentPath('rt-9'), '/notebook/rt-9/notebook.py')
    assert.strictEqual(notebookDocumentUri('rt-9'), 'file:///notebook/rt-9/notebook.py')
    assert.ok(notebookDocumentUri('rt-9').startsWith(notebookWorkspaceRootUri('rt-9')))
  })
})
