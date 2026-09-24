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

test('arena partial planner ignores unchanged channel sets', () => {
  const alpha = channel('alpha', 'Alpha', true)
  const beta = channel('beta', 'Beta')
  const previous = collectArenaEmitState([alpha, beta])

  const plan = planArenaPartialEmit(previous, [alpha, beta])

  assert.deepEqual(plan.changedChannels, [])
  assert.deepEqual(plan.deletedChannels, [])
  assert.equal(plan.hasChanges, false)
})

test('arena partial planner emits an initial empty feed and removes the final channel', () => {
  const initial = planArenaPartialEmit(undefined, [])
  assert.equal(initial.hasChanges, true)
  assert.deepEqual(initial.changedChannels, [])

  const removed = planArenaPartialEmit(collectArenaEmitState([channel('alpha', 'Alpha')]), [])
  assert.equal(removed.hasChanges, true)
  assert.deepEqual(
    removed.deletedChannels.map(([slug]) => slug),
    ['alpha'],
  )
  assert.equal(planArenaPartialEmit(initial.nextState, []).hasChanges, false)
})

test('arena partial planner detects nested note and priority changes', () => {
  const original = channel('alpha', 'Alpha')
  original.blocks[0].subItems = [{ id: 'note', content: 'Original note' }]
  const state = collectArenaEmitState([original])
  const updated = structuredClone(original)
  updated.blocks[0].subItems = [{ id: 'note', content: 'Updated note', metadata: { later: true } }]
  const plan = planArenaPartialEmit(state, [updated])
  assert.equal(plan.hasChanges, true)
  assert.deepEqual(
    plan.changedChannels.map(channel => channel.slug),
    ['alpha'],
  )
})

test('arena partial planner refreshes shared projections when channels are reordered', () => {
  const alpha = channel('alpha', 'Alpha')
  const beta = channel('beta', 'Beta')
  const plan = planArenaPartialEmit(collectArenaEmitState([alpha, beta]), [beta, alpha])
  assert.equal(plan.hasChanges, true)
  assert.deepEqual(plan.changedChannels, [])
  assert.deepEqual(plan.deletedChannels, [])
})
