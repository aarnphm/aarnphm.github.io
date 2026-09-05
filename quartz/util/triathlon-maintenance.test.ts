import assert from 'node:assert/strict'
import test from 'node:test'
import {
  formatTriathlonMaintenanceDistance,
  parseTriathlonMaintenance,
} from './triathlon-maintenance'

const maintenance = {
  OSPW: [
    [
      { type: 'Ultegra R8100 Pulley Wheel' },
      { distance: null },
      {
        range: [
          { start: '2026-05-16', end: '2026-08-10', reason: 'upgraded to CeramicSpeed OSPW' },
        ],
      },
    ],
    [
      { type: 'CeramicSpeed OSPW RS 5 Spoke' },
      { distance: null },
      { range: [{ start: '2026-08-10', end: null }] },
    ],
  ],
  'bottom bracket': [
    [
      { type: 'FSA T47 BBright' },
      { distance: 1721.5 },
      { range: [{ start: '2026-05-16', end: '2026-08-20' }] },
      { reason: 'upgraded to CeramicSpeed' },
    ],
  ],
  chain: {
    '1': { distance: 621, lubricant: 'Muc-Off Dry Lube', since: '2026-05-16', waxed: false },
    '3': { distance: null, lubricant: 'UFO Wax Drip-On', since: '2026-08-10', waxed: true },
  },
  service: {
    soloist: [{ date: '2026-08-20', distance: 1721.5, place: 'Racer Sportif' }],
    speedmax: null,
  },
  tires: {
    front: {
      tires: [
        [
          { type: 'Vittoria Corsa N.EXT' },
          { distance: 751.81 },
          { start: '2026-05-16' },
          { end: '2026-07-16' },
          { reason: 'training to race tires' },
        ],
        [
          { type: 'Pirelli P Zero Race SL-R' },
          { distance: null },
          {
            range: [
              { start: '2026-07-16', end: '2026-08-10' },
              { start: '2026-08-18', end: null },
            ],
          },
          { reason: 'punctures, repaired' },
          { repaired: true },
        ],
      ],
      tube: [
        [
          { type: 'Pirelli P Zero TPU' },
          { distance: null },
          {
            range: [
              { start: '2026-08-12', end: '2026-09-03', reason: 'slow puncture', repaired: true },
              { start: '2026-09-04' },
              { end: null },
            ],
          },
        ],
      ],
    },
  },
}

test('normalizes service, component, chain, and wheel maintenance records from frontmatter', () => {
  const parsed = parseTriathlonMaintenance(maintenance)
  assert.deepEqual(parsed?.services, [
    { bike: 'soloist', date: '2026-08-20', distanceMiles: 1721.5, place: 'Racer Sportif' },
  ])
  assert.deepEqual(parsed?.components, [
    {
      component: 'OSPW',
      type: 'CeramicSpeed OSPW RS 5 Spoke',
      distanceMiles: null,
      ranges: [{ start: '2026-08-10', end: null }],
      reason: null,
    },
    {
      component: 'OSPW',
      type: 'Ultegra R8100 Pulley Wheel',
      distanceMiles: null,
      ranges: [{ start: '2026-05-16', end: '2026-08-10' }],
      reason: 'upgraded to CeramicSpeed OSPW',
    },
    {
      component: 'bottom bracket',
      type: 'FSA T47 BBright',
      distanceMiles: 1721.5,
      ranges: [{ start: '2026-05-16', end: '2026-08-20' }],
      reason: 'upgraded to CeramicSpeed',
    },
  ])
  assert.deepEqual(parsed?.chains, [
    {
      id: '3',
      distanceMiles: null,
      lubricant: 'UFO Wax Drip-On',
      since: '2026-08-10',
      waxed: true,
    },
    {
      id: '1',
      distanceMiles: 621,
      lubricant: 'Muc-Off Dry Lube',
      since: '2026-05-16',
      waxed: false,
    },
  ])
  assert.deepEqual(
    parsed?.wheels.map(entry => [
      entry.position,
      entry.part,
      entry.type,
      entry.ranges,
      entry.repaired,
    ]),
    [
      [
        'front',
        'tube',
        'Pirelli P Zero TPU',
        [
          { start: '2026-08-12', end: '2026-09-03' },
          { start: '2026-09-04', end: null },
        ],
        true,
      ],
      [
        'front',
        'tire',
        'Pirelli P Zero Race SL-R',
        [
          { start: '2026-07-16', end: '2026-08-10' },
          { start: '2026-08-18', end: null },
        ],
        true,
      ],
      ['front', 'tire', 'Vittoria Corsa N.EXT', [{ start: '2026-05-16', end: '2026-07-16' }], null],
    ],
  )
  assert.equal(parsed?.components[1]?.reason, 'upgraded to CeramicSpeed OSPW')
  assert.equal(parsed?.wheels[0]?.reason, 'slow puncture')
})

test('formats mile source values in the selected distance system', () => {
  assert.equal(formatTriathlonMaintenanceDistance(1721.5, 'imperial'), '1,721.5 mi')
  assert.equal(formatTriathlonMaintenanceDistance(1721.5, 'metric'), '2,770.49 km')
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
  assert.equal(
    parseTriathlonMaintenance({
      tires: {
        front: {
          tires: [[{ type: 'Pirelli' }, { distance: null }, { range: [] }, { reason: null }]],
        },
      },
    }),
    null,
  )
  assert.equal(
    parseTriathlonMaintenance({
      tires: {
        front: {
          tube: [
            [
              { type: 'Pirelli' },
              { distance: null },
              { range: [{ start: '2026-08-12', reason: 42 }, { end: null }] },
            ],
          ],
        },
      },
    }),
    null,
  )
  assert.equal(
    parseTriathlonMaintenance({
      tires: {
        front: {
          tires: [
            [
              { type: 'Pirelli' },
              { distance: null },
              { range: [{ start: '2026-08-12', end: null }] },
              { reason: null },
              { repaired: 'yes' },
            ],
          ],
        },
      },
    }),
    null,
  )
  assert.equal(
    parseTriathlonMaintenance({ service: { soloist: [{ date: '2026-08-20', distance: null }] } }),
    null,
  )
  assert.equal(
    parseTriathlonMaintenance({
      OSPW: [[{ type: 'CeramicSpeed OSPW' }, { distance: null }, { range: [] }]],
    }),
    null,
  )
  assert.equal(
    parseTriathlonMaintenance({
      chain: {
        '1': {
          distance: '621 mile',
          lubricant: 'Muc-Off Dry Lube',
          since: '2026-05-16',
          waxed: false,
        },
      },
    }),
    null,
  )
})
