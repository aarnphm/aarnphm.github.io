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
      trainingExclusion: null,
    },
  )
})

test('accepts explicit zero fueling and rejects missing IDs or negative values', () => {
  assert.equal(parseTrackingBlock(null, 'date: 2026-07-19\nfueling: 140')?.fueling, null)
  assert.deepEqual(
    parseTrackingBlock(null, 'date: 2026-07-19\nactivity: 19382727312\nfueling: 0')?.fueling,
    { date: '2026-07-19', activityId: 19382727312, caloriesConsumed: 0 },
  )
  assert.equal(
    parseTrackingBlock(null, 'date: 2026-07-19\nactivity: 19382727312\nfueling: -1')?.fueling,
    null,
  )
})

test('parses skipTraining against a Strava activity ID', () => {
  assert.deepEqual(
    parseTrackingBlock(
      null,
      ['date: 2026-07-26', 'activity: 19476629599', 'skipTraining: true'].join('\n'),
    )?.trainingExclusion,
    { date: '2026-07-26', activityId: 19476629599 },
  )
  assert.equal(
    parseTrackingBlock(null, 'date: 2026-07-26\nskipTraining: true')?.trainingExclusion,
    null,
  )
  assert.equal(
    parseTrackingBlock(null, 'date: 2026-07-26\nactivity: 19476629599\nskipTraining: false')
      ?.trainingExclusion,
    null,
  )
})
