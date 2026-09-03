import assert from 'node:assert/strict'
import test from 'node:test'
import {
  environmentCursorIndex,
  environmentViewFromKey,
  type EnvironmentView,
} from './environment-tabs'

test('environment tabs wrap with arrows and respect Home and End', () => {
  const views: readonly EnvironmentView[] = ['cumulative', 'uv-index', 'temperature', 'cloud-cover']
  assert.equal(environmentViewFromKey('cumulative', 'ArrowLeft', views), 'cloud-cover')
  assert.equal(environmentViewFromKey('cloud-cover', 'ArrowRight', views), 'cumulative')
  assert.equal(environmentViewFromKey('temperature', 'Home', views), 'cumulative')
  assert.equal(environmentViewFromKey('uv-index', 'End', views), 'cloud-cover')
  assert.equal(environmentViewFromKey('uv-index', 'Enter', views), null)
})

test('environment tabs navigate only among rendered views', () => {
  const views: readonly EnvironmentView[] = ['uv-index', 'temperature']
  assert.equal(environmentViewFromKey('uv-index', 'ArrowLeft', views), 'temperature')
  assert.equal(environmentViewFromKey('temperature', 'ArrowRight', views), 'uv-index')
  assert.equal(environmentViewFromKey('cumulative', 'ArrowRight', views), null)
})

test('environment cursor preserves zero and clamps serialized indices', () => {
  assert.equal(environmentCursorIndex('0', 320), 0)
  assert.equal(environmentCursorIndex('1', 320), 1)
  assert.equal(environmentCursorIndex('999', 320), 319)
  assert.equal(environmentCursorIndex('-1', 320), 0)
  assert.equal(environmentCursorIndex(undefined, 320), 319)
  assert.equal(environmentCursorIndex('invalid', 320), 319)
})
