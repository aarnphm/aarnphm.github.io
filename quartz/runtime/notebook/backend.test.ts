import assert from 'node:assert'
import test, { describe, before } from 'node:test'
import { javascriptBackend } from '../javascript/backend'
import { haskellBackend, mojoBackend, ocamlBackend, rustBackend } from '../native/backend'
import { pythonBackend } from '../python/backend'
import {
  backendFor,
  backendForShellMagic,
  listBackends,
  registerBackend,
  unregisterBackend,
} from './backend'

before(async () => {
  await import('./registry')
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
    assert.strictEqual(backendFor('mojo'), mojoBackend)
    assert.strictEqual(backendFor('.mojo'), mojoBackend)
    assert.strictEqual(backendFor('haskell'), haskellBackend)
    assert.strictEqual(backendFor('runghc'), haskellBackend)
    assert.strictEqual(backendFor('.hs'), haskellBackend)
    assert.strictEqual(backendFor('ocaml'), ocamlBackend)
    assert.strictEqual(backendFor('.ml'), ocamlBackend)
  })

  test('resolves a backend by shell magic', () => {
    assert.strictEqual(backendForShellMagic('python-shell'), pythonBackend)
    assert.strictEqual(backendForShellMagic('py-shell'), pythonBackend)
    assert.strictEqual(backendForShellMagic('javascript'), javascriptBackend)
    assert.strictEqual(backendForShellMagic('js'), javascriptBackend)
    assert.strictEqual(backendForShellMagic('javascript-shell'), javascriptBackend)
    assert.strictEqual(backendForShellMagic('rust-shell'), rustBackend)
    assert.strictEqual(backendForShellMagic('rust'), undefined)
    assert.strictEqual(backendForShellMagic('mojo-shell'), mojoBackend)
    assert.strictEqual(backendForShellMagic('haskell-shell'), haskellBackend)
    assert.strictEqual(backendForShellMagic('ocaml-shell'), ocamlBackend)
    assert.strictEqual(backendForShellMagic('ocaml'), undefined)
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

  test('rust backend executes through the playground sandbox', () => {
    assert.strictEqual(rustBackend.canExecute('fn main() { println!("hi"); }').ok, true)
    const rejected = rustBackend.canExecute('')
    assert.strictEqual(rejected.ok, false)
    if (!rejected.ok) assert.match(rejected.reason, /source code/)
  })

  test('native browser backends report unavailable compiler runtimes', async () => {
    for (const backend of [mojoBackend, haskellBackend, ocamlBackend]) {
      const accepted = backend.canExecute('main = print "hi"')
      assert.strictEqual(accepted.ok, false)
      if (!accepted.ok) assert.match(accepted.reason, /native compiler runtime/)
      const kernel = await backend.kernelFactory({ runtimeId: 'r', sourcePath: 's' })
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
