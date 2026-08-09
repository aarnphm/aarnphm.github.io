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
      strength: null,
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

test('parses repeated strength exercises with aggregate and individual sets', () => {
  const parsed = parseTrackingBlock(
    null,
    [
      'date: 2026-08-06',
      'activity: 19633452010',
      'strengthVolume: 1800.1 lb',
      'strengthSets: 15',
      'strengthReps: 90',
      'exercise: Press Up Position Walk Out | 2 sets / 20 reps',
      'exercise: KB Straight Leg Deadlift | 10 reps @ 50 lb | 10 reps @ 50 lb',
      'exercise: Plank | 1 set / 30s',
    ].join('\n'),
  )

  assert.deepEqual(parsed?.strength, {
    date: '2026-08-06',
    activityId: 19633452010,
    volumeKg: 816.512,
    totalSets: 15,
    totalReps: 90,
    exercises: [
      {
        name: 'Press Up Position Walk Out',
        setCount: 2,
        repetitions: 20,
        durationS: null,
        sets: [],
      },
      {
        name: 'KB Straight Leg Deadlift',
        setCount: 2,
        repetitions: 20,
        durationS: null,
        sets: [
          { repetitions: 10, durationS: null, weightKg: 22.68 },
          { repetitions: 10, durationS: null, weightKg: 22.68 },
        ],
      },
      { name: 'Plank', setCount: 1, repetitions: null, durationS: 30, sets: [] },
    ],
  })
})

test('requires a Strava activity ID before emitting strength metadata', () => {
  assert.equal(
    parseTrackingBlock(
      null,
      ['date: 2026-08-06', 'strengthVolume: 1800.1 lb', 'exercise: Plank | 30s'].join('\n'),
    )?.strength,
    null,
  )
})
