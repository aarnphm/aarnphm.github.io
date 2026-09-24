import assert from 'node:assert'
import test, { describe, before } from 'node:test'
import { javascriptBackend } from '../javascript/backend'
import {
  cBackend,
  cppBackend,
  goBackend,
  haskellBackend,
  mojoBackend,
  ocamlBackend,
  rustBackend,
  wasmBackend,
} from '../native/backend'
import { pythonBackend } from '../python/backend'
import { backendFor, backendForShellMagic, registerBackend, unregisterBackend } from './backend'
import {
  nextNotebookCellId,
  notebookRuntimeKernelLanguages,
  notebookRunAndAdvanceKey,
  notebookRunKey,
  notebookRuntimePreloadLanguages,
} from './client'

before(async () => {
  await import('./registry')
})

describe('Notebook runtime keyboard commands', () => {
  test('classifies Cmd+Enter as run and advance', () => {
    const cmdEnter = { key: 'Enter', metaKey: true, ctrlKey: false, shiftKey: false, altKey: false }
    assert.strictEqual(notebookRunAndAdvanceKey(cmdEnter), true)
    assert.strictEqual(notebookRunKey(cmdEnter), true)
  })

  test('keeps other modified Enter gestures on the run-only path', () => {
    const ctrlEnter = {
      key: 'Enter',
      metaKey: false,
      ctrlKey: true,
      shiftKey: false,
      altKey: false,
    }
    const shiftEnter = {
      key: 'Enter',
      metaKey: false,
      ctrlKey: false,
      shiftKey: true,
      altKey: false,
    }
    assert.strictEqual(notebookRunAndAdvanceKey(ctrlEnter), false)
    assert.strictEqual(notebookRunKey(ctrlEnter), true)
    assert.strictEqual(notebookRunAndAdvanceKey(shiftEnter), false)
    assert.strictEqual(notebookRunKey(shiftEnter), true)
  })

  test('resolves the next runtime cell without wrapping', () => {
    const cells = [{ id: 'cell-1' }, { id: 'cell-2' }, { id: 'cell-3' }]
    assert.strictEqual(nextNotebookCellId(cells, 'cell-1'), 'cell-2')
    assert.strictEqual(nextNotebookCellId(cells, 'cell-3'), undefined)
    assert.strictEqual(nextNotebookCellId(cells, 'missing'), undefined)
  })

  test('deduplicates executable runtime languages and skips lazy preload entries', () => {
    const payload = {
      language: 'python',
      cells: [
        { language: 'python' },
        { language: 'javascript' },
        { language: 'rust' },
        { language: 'js' },
        { language: 'haskell' },
        { language: 'wat' },
        { language: 'cpp' },
        { language: 'c++' },
      ],
    }

    assert.deepStrictEqual(notebookRuntimeKernelLanguages(payload), [
      'python',
      'javascript',
      'rust',
      'haskell',
      'wasm',
      'cpp',
    ])
    assert.deepStrictEqual(notebookRuntimePreloadLanguages(payload), [
      'python',
      'javascript',
      'rust',
      'haskell',
      'wasm',
    ])
  })
})

describe('LanguageBackend registry', () => {
  test('resolves a backend by shell magic', () => {
    assert.strictEqual(backendForShellMagic('python-shell'), pythonBackend)
    assert.strictEqual(backendForShellMagic('py-shell'), pythonBackend)
    assert.strictEqual(backendForShellMagic('javascript'), javascriptBackend)
    assert.strictEqual(backendForShellMagic('js'), javascriptBackend)
    assert.strictEqual(backendForShellMagic('javascript-shell'), javascriptBackend)
    assert.strictEqual(backendForShellMagic('rust-shell'), rustBackend)
    assert.strictEqual(backendForShellMagic('rust'), undefined)
    assert.strictEqual(backendForShellMagic('c-shell'), cBackend)
    assert.strictEqual(backendForShellMagic('c'), undefined)
    assert.strictEqual(backendForShellMagic('cpp-shell'), cppBackend)
    assert.strictEqual(backendForShellMagic('c++-shell'), cppBackend)
    assert.strictEqual(backendForShellMagic('cpp'), undefined)
    assert.strictEqual(backendForShellMagic('mojo-shell'), mojoBackend)
    assert.strictEqual(backendForShellMagic('haskell-shell'), haskellBackend)
    assert.strictEqual(backendForShellMagic('haskell'), undefined)
    assert.strictEqual(backendForShellMagic('ocaml-shell'), ocamlBackend)
    assert.strictEqual(backendForShellMagic('ocaml'), undefined)
    assert.strictEqual(backendForShellMagic('go-shell'), goBackend)
    assert.strictEqual(backendForShellMagic('go'), undefined)
    assert.strictEqual(backendForShellMagic('wasm-shell'), wasmBackend)
    assert.strictEqual(backendForShellMagic('wat-shell'), wasmBackend)
    assert.strictEqual(backendForShellMagic('wasm'), undefined)
  })

  test('canExecute on python backend rejects threading and accepts plain code', () => {
    const accepted = pythonBackend.canExecute('x = 1\nprint(x)')
    assert.strictEqual(accepted.ok, true)
    const rejected = pythonBackend.canExecute('from threading import Thread\nt = Thread()')
    assert.strictEqual(rejected.ok, false)
    if (!rejected.ok) assert.match(rejected.reason, /threading/i)
  })

  test('canExecute on javascript backend accepts javascript magics', () => {
    assert.strictEqual(javascriptBackend.canExecute('console.log("hi")').ok, true)
    assert.strictEqual(javascriptBackend.canExecute('%%javascript\nconsole.log("hi")').ok, true)
    const rejected = javascriptBackend.canExecute('%%bash\necho hi')
    assert.strictEqual(rejected.ok, false)
    if (!rejected.ok) assert.match(rejected.reason, /%%bash/)
  })

  test('python backend owns notebook module import resolution', () => {
    assert.deepStrictEqual(
      pythonBackend.moduleResolver?.importNames('import foo\nfrom bar import baz'),
      ['foo', 'bar'],
    )
    assert.deepStrictEqual(
      pythonBackend.moduleResolver?.importNames(
        'from IPython.utils.frame import extract_module_locals\nimport js\nimport pyodide\nimport foo',
      ),
      ['foo'],
    )
    assert.match(
      pythonBackend.moduleResolver?.moduleSource(
        JSON.stringify({ cells: [{ cell_type: 'code', source: 'x = 1' }] }),
        'foo.ipynb',
      ) ?? '',
      /^x = 1$/,
    )
  })

  test('unregister removes by name and clears shell magics', () => {
    unregisterBackend('python')
    assert.strictEqual(backendFor('python'), undefined)
    assert.strictEqual(backendForShellMagic('python-shell'), undefined)
    registerBackend(pythonBackend)
    assert.strictEqual(backendFor('python'), pythonBackend)
    assert.strictEqual(backendForShellMagic('python-shell'), pythonBackend)
  })
})
