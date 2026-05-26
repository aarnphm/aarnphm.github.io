import assert from 'node:assert/strict'
import test from 'node:test'
import type { ArenaChannel } from '../plugins/transformers/arena'
import { collectArenaEmitState, planArenaPartialEmit } from './arena-page-partial'

function channel(slug: string, title: string, json = false): ArenaChannel {
  return {
    id: slug,
    name: title,
    slug,
    metadata: json ? { json: true } : undefined,
    blocks: [
      {
        id: `${slug}-block`,
        title,
        content: title,
        url: `https://example.com/${slug}`,
        metadata: { date: '05/25/2026' },
      },
    ],
  }
}

test('arena partial planner selects only changed channels', () => {
  const alpha = channel('alpha', 'Alpha', true)
  const beta = channel('beta', 'Beta')
  const previous = collectArenaEmitState([alpha, beta])

  const changedAlpha = channel('alpha', 'Alpha changed', true)
  const plan = planArenaPartialEmit(previous, [changedAlpha, beta])

  assert.deepEqual(
    plan.changedChannels.map(channel => channel.slug),
    ['alpha'],
  )
  assert.deepEqual(plan.deletedChannels, [])
  assert.equal(plan.hasChanges, true)
})

test('arena partial planner reports deleted channel output state', () => {
  const alpha = channel('alpha', 'Alpha', true)
  const beta = channel('beta', 'Beta')
  const previous = collectArenaEmitState([alpha, beta])

  const plan = planArenaPartialEmit(previous, [beta])

  assert.deepEqual(plan.changedChannels, [])
  assert.deepEqual(
    plan.deletedChannels.map(([slug, state]) => [slug, state.jsonEnabled]),
    [['alpha', true]],
  )
  assert.equal(plan.hasChanges, true)
})

test('arena partial planner ignores unchanged channel sets', () => {
  const alpha = channel('alpha', 'Alpha', true)
  const beta = channel('beta', 'Beta')
  const previous = collectArenaEmitState([alpha, beta])

  const plan = planArenaPartialEmit(previous, [alpha, beta])

  assert.deepEqual(plan.changedChannels, [])
  assert.deepEqual(plan.deletedChannels, [])
  assert.equal(plan.hasChanges, false)
})
