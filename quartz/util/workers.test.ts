import assert from 'node:assert'
import test, { describe } from 'node:test'
import { isWorkerEntryPath, workerEntryPattern } from './workers'

describe('worker entry matching', () => {
  test('only includes TypeScript worker entrypoints', () => {
    assert.strictEqual(workerEntryPattern, 'quartz/**/*.worker.ts')
    assert.strictEqual(isWorkerEntryPath('quartz/workers/semantic.worker.ts'), true)
    assert.strictEqual(
      isWorkerEntryPath('quartz/components/scripts/notebook-runtime.frame.html'),
      false,
    )
    assert.strictEqual(
      isWorkerEntryPath('quartz/components/scripts/notebook-runtime.worker.html'),
      false,
    )
  })
})
