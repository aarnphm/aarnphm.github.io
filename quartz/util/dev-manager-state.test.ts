import assert from 'node:assert/strict'
import test from 'node:test'
import {
  applyQuartzDevEvent,
  createQuartzManagerState,
  type QuartzManagerState,
} from './dev-manager-state'

const delayMs = 250

function readyState(epoch: string): QuartzManagerState {
  return { currentEpoch: epoch, quartz: 'ready', wrangler: 'ready', publicAvailable: true }
}

test('initial build lifecycle starts wrangler once', () => {
  const state = createQuartzManagerState()

  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'build:start', epoch: 'a', reason: 'initial' }, delayMs),
    [],
  )
  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'public:remove:start', epoch: 'a' }, delayMs),
    [{ type: 'stop-wrangler', reason: 'stopping wrangler while public is regenerated' }],
  )
  assert.deepEqual(
    applyQuartzDevEvent(
      state,
      { type: 'build:ready', epoch: 'a', files: 10, elapsedMs: 45 },
      delayMs,
    ),
    [{ type: 'schedule-wrangler-start', delayMs }],
  )
  assert.equal(state.quartz, 'ready')
  assert.equal(state.publicAvailable, true)
})

test('source hard rebuild stops wrangler when output is removed then schedules restart', () => {
  const state = readyState('a')

  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'build:start', epoch: 'b', reason: 'source' }, delayMs),
    [],
  )
  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'public:remove:start', epoch: 'b' }, delayMs),
    [{ type: 'stop-wrangler', reason: 'stopping wrangler while public is regenerated' }],
  )
  assert.deepEqual(
    applyQuartzDevEvent(
      state,
      { type: 'build:ready', epoch: 'b', files: 10, elapsedMs: 45 },
      delayMs,
    ),
    [{ type: 'schedule-wrangler-start', delayMs }],
  )
})

test('content rebuild keeps wrangler running until updated output is ready', () => {
  const state = readyState('a')

  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'build:start', epoch: 'b', reason: 'content' }, delayMs),
    [],
  )
  assert.equal(state.quartz, 'building')
  assert.equal(state.publicAvailable, true)
  assert.equal(state.wrangler, 'ready')
  assert.deepEqual(
    applyQuartzDevEvent(
      state,
      { type: 'build:ready', epoch: 'b', files: 3, elapsedMs: 20 },
      delayMs,
    ),
    [{ type: 'schedule-wrangler-start', delayMs }],
  )
  assert.equal(state.wrangler, 'ready')
})

test('style rebuild keeps the current output available', () => {
  const state = readyState('a')
  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'build:start', epoch: 'b', reason: 'source' }, delayMs),
    [],
  )
  assert.equal(state.publicAvailable, true)
  assert.equal(state.wrangler, 'ready')
})

test('failed content rebuild leaves the existing server available', () => {
  const state = readyState('a')
  applyQuartzDevEvent(state, { type: 'build:start', epoch: 'b', reason: 'content' }, delayMs)
  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'build:error', epoch: 'b', message: 'boom' }, delayMs),
    [],
  )
  assert.equal(state.quartz, 'failed')
  assert.equal(state.publicAvailable, true)
  assert.equal(state.wrangler, 'ready')
})

test('stale ready event does not schedule wrangler restart', () => {
  const state = readyState('current')

  assert.deepEqual(
    applyQuartzDevEvent(
      state,
      { type: 'build:ready', epoch: 'stale', files: 10, elapsedMs: 45 },
      delayMs,
    ),
    [],
  )
})

test('build error leaves wrangler stopped when public was removed', () => {
  const state = createQuartzManagerState()
  applyQuartzDevEvent(state, { type: 'build:start', epoch: 'a', reason: 'source' }, delayMs)
  applyQuartzDevEvent(state, { type: 'public:remove:start', epoch: 'a' }, delayMs)

  assert.deepEqual(
    applyQuartzDevEvent(state, { type: 'build:error', epoch: 'a', message: 'boom' }, delayMs),
    [{ type: 'stop-wrangler', reason: 'stopping wrangler after failed rebuild' }],
  )
  assert.equal(state.quartz, 'failed')
  assert.equal(state.publicAvailable, false)
})
