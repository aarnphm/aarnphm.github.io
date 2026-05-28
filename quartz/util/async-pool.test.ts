import assert from 'node:assert/strict'
import test from 'node:test'
import { mapConcurrent } from './async-pool'

test('mapConcurrent preserves input order while running bounded work', async () => {
  let active = 0
  let maxActive = 0
  const values = [30, 20, 10, 0]

  const results = await mapConcurrent(values, 2, async (value, index) => {
    active += 1
    maxActive = Math.max(maxActive, active)
    await new Promise(resolve => setTimeout(resolve, value))
    active -= 1
    return `${index}:${value}`
  })

  assert.deepEqual(results, ['0:30', '1:20', '2:10', '3:0'])
  assert.equal(maxActive, 2)
})

test('mapConcurrent clamps concurrency to at least one worker', async () => {
  const results = await mapConcurrent([1, 2], 0, async value => value * 2)

  assert.deepEqual(results, [2, 4])
})
