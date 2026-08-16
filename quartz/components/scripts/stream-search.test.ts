import assert from 'node:assert/strict'
import test from 'node:test'
import { mapWithConcurrency } from './stream-search.inline'

test('stream fragment loading preserves result order and bounds concurrent requests', async () => {
  let active = 0
  let maximumActive = 0
  const values = [18, 2, 12, 1, 6]

  const results = await mapWithConcurrency(values, 2, async value => {
    active += 1
    maximumActive = Math.max(maximumActive, active)
    await new Promise(resolve => setTimeout(resolve, value))
    active -= 1
    return `loaded-${value}`
  })

  assert.deepEqual(
    results,
    values.map(value => `loaded-${value}`),
  )
  assert.equal(maximumActive, 2)
})
