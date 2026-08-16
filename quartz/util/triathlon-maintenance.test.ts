import assert from 'node:assert/strict'
import test from 'node:test'
import { parseTriathlonMaintenance } from './triathlon-maintenance'

const maintenance = {
  chain: {
    '1': { distance: '621 mile', lubricant: 'Muc-Off Dry Lube', since: '2026-05-16', waxed: false },
    '3': { distance: null, lubricant: 'UFO Wax Drip-On', since: '2026-08-10', waxed: true },
  },
  tires: {
    front: {
      tires: [
        [
          { type: 'Vittoria Corsa N.EXT' },
          { distance: '751.81 mile' },
          { start: '2026-05-16' },
          { end: '2026-07-16' },
          { reason: 'training to race tires' },
        ],
        [
          { type: 'Pirelli P Zero Race SL-R' },
          { distance: null },
          { start: '2026-08-12' },
          { end: null },
          { reason: null },
        ],
      ],
      tube: [
        [
          { type: 'Pirelli P Zero TPU' },
          { distance: null },
          { start: '2026-08-12' },
          { end: null },
          { reason: null },
        ],
      ],
    },
  },
}

test('normalizes chain and wheel maintenance records from frontmatter', () => {
  const parsed = parseTriathlonMaintenance(maintenance)
  assert.deepEqual(parsed?.chains, [
    { id: '3', distance: null, lubricant: 'UFO Wax Drip-On', since: '2026-08-10', waxed: true },
    {
      id: '1',
      distance: '621 mile',
      lubricant: 'Muc-Off Dry Lube',
      since: '2026-05-16',
      waxed: false,
    },
  ])
  assert.deepEqual(
    parsed?.wheels.map(entry => [entry.position, entry.part, entry.type, entry.end]),
    [
      ['front', 'tire', 'Pirelli P Zero Race SL-R', null],
      ['front', 'tube', 'Pirelli P Zero TPU', null],
      ['front', 'tire', 'Vittoria Corsa N.EXT', '2026-07-16'],
    ],
  )
})

test('drops malformed records and rejects empty maintenance data', () => {
  assert.equal(parseTriathlonMaintenance(null), null)
  assert.equal(parseTriathlonMaintenance({ chain: { '1': { since: '2026-05-16' } } }), null)
  assert.equal(
    parseTriathlonMaintenance({
      tires: { front: { tires: [[{ type: 'Pirelli' }, { start: '2026-08-12' }]] } },
    }),
    null,
  )
})
