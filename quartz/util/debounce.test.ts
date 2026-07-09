import assert from 'node:assert/strict'
import test from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import { debounce } from './debounce'

test('debounce keeps the latest call', async () => {
  const values: string[] = []
  const append = debounce((value: string) => values.push(value), 5)

  append('first')
  append('second')
  await delay(15)

  assert.deepEqual(values, ['second'])
})

test('debounce cancellation clears pending work', async () => {
  let calls = 0
  const increment = debounce(() => calls++, 5)

  increment()
  increment.cancel()
  await delay(15)

  assert.equal(calls, 0)
})
