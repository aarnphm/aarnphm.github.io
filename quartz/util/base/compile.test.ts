import assert from 'node:assert/strict'
import test from 'node:test'
import { compileBaseConfig } from './compile'

test('compileBaseConfig requires a views array', () => {
  assert.throws(
    () => compileBaseConfig('filters: status == "done"', 'notes.base'),
    /Invalid base configuration in notes\.base: 'views' must be an array/,
  )
  assert.throws(
    () => compileBaseConfig('views: {}'),
    /Invalid base configuration: 'views' must be an array/,
  )
})

test('compileBaseConfig accepts an empty views array', () => {
  const result = compileBaseConfig('views: []', 'notes.base')

  assert.deepEqual(result.config.views, [])
})
