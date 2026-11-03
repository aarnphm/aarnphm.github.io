import assert from 'node:assert/strict'
import test from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import { createSourceRebuildQueue } from './source-rebuild-queue.js'

async function until(predicate) {
  const openedAt = performance.now()
  while (!predicate()) {
    if (performance.now() - openedAt > 2000) throw new Error('timed out waiting for the queue')
    await delay(5)
  }
}

test('edits that arrive during a rebuild flush as one later rebuild', async () => {
  const batches = []
  let finish = () => {}
  const queue = createSourceRebuildQueue({
    quietMs: 40,
    maxWaitMs: 150,
    build: takePaths => {
      batches.push(takePaths())
      return new Promise(resolve => {
        finish = () => resolve(true)
      })
    },
  })

  const done = queue.start()
  await until(() => batches.length === 1)
  assert.deepEqual(batches[0], [])

  for (const fp of ['a.scss', 'b.scss', 'c.scss', 'd.scss', 'e.scss', 'f.scss', 'g.scss']) {
    queue.add(fp)
    await delay(5)
  }
  finish()
  await delay(15)
  assert.equal(batches.length, 1)

  queue.add('h.scss')
  await until(() => batches.length === 2)
  assert.deepEqual(batches[1], [
    'a.scss',
    'b.scss',
    'c.scss',
    'd.scss',
    'e.scss',
    'f.scss',
    'g.scss',
    'h.scss',
  ])
  finish()
  await done
  assert.equal(batches.length, 2)
})

test('a quiet burst is one rebuild, and a stale result does not ask the browser to load', async () => {
  const batches = []
  const loads = []
  let finish = () => {}
  const queue = createSourceRebuildQueue({
    quietMs: 30,
    maxWaitMs: 80,
    build: async takePaths => {
      const paths = takePaths()
      batches.push(paths)
      await new Promise(resolve => {
        finish = resolve
      })
      if (paths.length > 0 && queue.pending === 0) loads.push(paths)
      return true
    },
  })

  const done = queue.start()
  await until(() => batches.length === 1)
  finish()
  queue.add('one.ts')
  await delay(10)
  queue.add('two.ts')
  await until(() => batches.length === 2)
  assert.deepEqual(batches[1], ['one.ts', 'two.ts'])
  queue.add('three.ts')
  finish()
  await until(() => batches.length === 3)
  assert.deepEqual(loads, [])
  assert.deepEqual(batches[2], ['three.ts'])
  finish()
  await done
  assert.deepEqual(loads, [['three.ts']])
})
