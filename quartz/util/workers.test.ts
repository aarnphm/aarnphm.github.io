import assert from 'node:assert'
import test, { describe } from 'node:test'
import { isWorkerEntryPath, workerEntryPattern } from './workers'

describe('worker entry matching', () => {
  test('only includes TypeScript worker entrypoints', () => {
    assert.strictEqual(workerEntryPattern, 'quartz/**/*.worker.ts')
    assert.strictEqual(isWorkerEntryPath('quartz/workers/semantic.worker.ts'), true)
    assert.strictEqual(isWorkerEntryPath('quartz/runtime/python/pyodide-worker.js'), false)
  })
})
