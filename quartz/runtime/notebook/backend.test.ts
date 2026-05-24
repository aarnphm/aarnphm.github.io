import assert from 'node:assert'
import test, { describe, before } from 'node:test'
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
  })

  test('resolves a backend by shell magic', () => {
    assert.strictEqual(backendForShellMagic('python-shell'), pythonBackend)
    assert.strictEqual(backendForShellMagic('py-shell'), pythonBackend)
    assert.strictEqual(backendForShellMagic('haskell-shell'), undefined)
  })

  test('returns undefined for unregistered languages', () => {
    assert.strictEqual(backendFor('rust'), undefined)
    assert.strictEqual(backendFor('ocaml'), undefined)
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

  test('unregister removes by name and clears shell magics', () => {
    unregisterBackend('python')
    assert.strictEqual(backendFor('python'), undefined)
    assert.strictEqual(backendForShellMagic('python-shell'), undefined)
    registerBackend(pythonBackend)
    assert.strictEqual(backendFor('python'), pythonBackend)
    assert.strictEqual(backendForShellMagic('python-shell'), pythonBackend)
  })
})
