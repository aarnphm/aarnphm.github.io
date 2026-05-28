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
import {
  backendFor,
  backendForShellMagic,
  listBackends,
  registerBackend,
  unregisterBackend,
} from './backend'
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
  test('resolves a backend by every alias and file extension', () => {
    assert.strictEqual(backendFor('python'), pythonBackend)
    assert.strictEqual(backendFor('Python'), pythonBackend)
    assert.strictEqual(backendFor('py'), pythonBackend)
    assert.strictEqual(backendFor('ipython'), pythonBackend)
    assert.strictEqual(backendFor('.ipynb'), pythonBackend)
    assert.strictEqual(backendFor('.py'), pythonBackend)
    assert.strictEqual(backendFor('javascript'), javascriptBackend)
    assert.strictEqual(backendFor('js'), javascriptBackend)
    assert.strictEqual(backendFor('ijavascript'), javascriptBackend)
    assert.strictEqual(backendFor('.mjs'), javascriptBackend)
    assert.strictEqual(backendFor('rust'), rustBackend)
    assert.strictEqual(backendFor('.rs'), rustBackend)
    assert.strictEqual(backendFor('c'), cBackend)
    assert.strictEqual(backendFor('.c'), cBackend)
    assert.strictEqual(backendFor('cpp'), cppBackend)
    assert.strictEqual(backendFor('c++'), cppBackend)
    assert.strictEqual(backendFor('.cpp'), cppBackend)
    assert.strictEqual(backendFor('mojo'), mojoBackend)
    assert.strictEqual(backendFor('.mojo'), mojoBackend)
    assert.strictEqual(backendFor('haskell'), haskellBackend)
    assert.strictEqual(backendFor('runghc'), haskellBackend)
    assert.strictEqual(backendFor('.hs'), haskellBackend)
    assert.strictEqual(backendFor('ocaml'), ocamlBackend)
    assert.strictEqual(backendFor('.ml'), ocamlBackend)
    assert.strictEqual(backendFor('go'), goBackend)
    assert.strictEqual(backendFor('golang'), goBackend)
    assert.strictEqual(backendFor('.go'), goBackend)
    assert.strictEqual(backendFor('wasm'), wasmBackend)
    assert.strictEqual(backendFor('wat'), wasmBackend)
    assert.strictEqual(backendFor('.wat'), wasmBackend)
  })

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

  test('returns undefined for unregistered languages', () => {
    assert.strictEqual(backendFor('ruby'), undefined)
    assert.strictEqual(backendFor('php'), undefined)
  })

  test('listBackends deduplicates by identity', () => {
    const backends = listBackends()
    assert.strictEqual(backends.filter(b => b.name === 'python').length, 1)
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

  test('python kernelFactory returns a Python kernel', async () => {
    const kernel = await pythonBackend.kernelFactory({
      runtimeId: 'r',
      sourcePath: 's',
      workerUrl: '/static/scripts/notebook-runtime.worker.js',
    })
    assert.strictEqual(kernel.language, 'python')
    assert.strictEqual(typeof kernel.execute, 'function')
    assert.strictEqual(typeof kernel.interrupt, 'function')
  })

  test('javascript kernelFactory returns a JavaScript kernel', async () => {
    const kernel = await javascriptBackend.kernelFactory({
      runtimeId: 'r',
      sourcePath: 's',
      workerUrl: '/static/scripts/notebook-runtime.javascript.worker.js',
    })
    assert.strictEqual(kernel.language, 'javascript')
    assert.strictEqual(typeof kernel.execute, 'function')
    assert.strictEqual(typeof kernel.interrupt, 'function')
  })

  test('native browser backends boot through self-hosted runtime packs', async () => {
    for (const backend of [
      rustBackend,
      cBackend,
      cppBackend,
      mojoBackend,
      haskellBackend,
      ocamlBackend,
      goBackend,
      wasmBackend,
    ]) {
      const accepted = backend.canExecute('main = print "hi"')
      assert.strictEqual(accepted.ok, true)
      const kernel = await backend.kernelFactory({
        runtimeId: 'r',
        sourcePath: 's',
        workerUrl: '/static/scripts/notebook-runtimes/manifest.json',
      })
      assert.strictEqual(kernel.language, backend.name)
    }
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
