import assert from 'node:assert/strict'
import test from 'node:test'
import {
  QUARTZ_DEV_EVENT_PREFIX,
  parseQuartzDevEvent,
  splitDevEventLines,
  type QuartzDevEvent,
} from './dev-events'

test('parseQuartzDevEvent accepts valid events', () => {
  const event: QuartzDevEvent = { type: 'build:start', epoch: 'epoch-1', reason: 'initial' }

  assert.deepEqual(parseQuartzDevEvent(`${QUARTZ_DEV_EVENT_PREFIX}${JSON.stringify(event)}`), event)
})

test('parseQuartzDevEvent rejects malformed json', () => {
  assert.equal(parseQuartzDevEvent(`${QUARTZ_DEV_EVENT_PREFIX}{bad`), undefined)
})

test('parseQuartzDevEvent ignores unrelated log text', () => {
  assert.equal(parseQuartzDevEvent('Done processing 3 files in 24ms'), undefined)
})

test('splitDevEventLines preserves partial chunk boundaries', () => {
  const first = splitDevEventLines('', `${QUARTZ_DEV_EVENT_PREFIX}{"type":"build:start"`)
  assert.deepEqual(first.lines, [])

  const second = splitDevEventLines(first.rest, ',"epoch":"epoch-1","reason":"content"}\nplain log')
  assert.deepEqual(second.lines, [
    `${QUARTZ_DEV_EVENT_PREFIX}{"type":"build:start","epoch":"epoch-1","reason":"content"}`,
  ])
  assert.equal(second.rest, 'plain log')
})
