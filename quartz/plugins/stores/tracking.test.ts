import assert from 'node:assert/strict'
import test from 'node:test'
import { parseTrackingBlock } from './tracking'

test('parses manual fueling against a Strava activity ID', () => {
  assert.deepEqual(
    parseTrackingBlock(
      null,
      ['date: 2026-07-19', 'activity: 19382727312', 'fueling: 140'].join('\n'),
    ),
    {
      day: {
        date: '2026-07-19',
        weightLbs: null,
        weightKg: null,
        windKph: null,
        windDir: null,
        race: false,
        event: null,
      },
      fueling: { date: '2026-07-19', activityId: 19382727312, caloriesConsumed: 140 },
    },
  )
})

test('requires both a positive Strava activity ID and fueling value', () => {
  assert.equal(parseTrackingBlock(null, 'date: 2026-07-19\nfueling: 140')?.fueling, null)
  assert.equal(
    parseTrackingBlock(null, 'date: 2026-07-19\nactivity: 19382727312\nfueling: 0')?.fueling,
    null,
  )
})
