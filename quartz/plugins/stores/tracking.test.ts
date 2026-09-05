import assert from 'node:assert/strict'
import test from 'node:test'
import { parseTrackingBlock } from './tracking'

test('parses a virtual Garmin attachment without a tracking date', () => {
  assert.deepEqual(
    parseTrackingBlock(null, 'activity: 20037941355\ngarmin: 24239315396\nvirtual: true'),
    {
      day: null,
      activity: { activityId: 20037941355, garminActivityId: 24239315396, virtual: true },
      fueling: null,
      strength: null,
      sauna: null,
      trainingExclusion: null,
    },
  )
  assert.deepEqual(
    parseTrackingBlock(null, 'activity: 20037941355\ngarmin: 24239315396')?.activity,
    { activityId: 20037941355, garminActivityId: 24239315396, virtual: false },
  )
  assert.equal(
    parseTrackingBlock(null, 'activity: 20037941355\nvirtual: false')?.activity?.virtual,
    false,
  )
  assert.equal(
    parseTrackingBlock(
      null,
      'date: 2026-09-04\nactivity: 20037941355\ngarmin: 24239315396\nvirtual: true',
    )?.day?.date,
    '2026-09-04',
  )
})

test('rejects malformed virtual activity links without inventing a day', () => {
  for (const activity of ['0', '-1', '20037941355.5', '1e3', '9007199254740992', ''])
    assert.equal(
      parseTrackingBlock(null, `activity: ${activity}\ngarmin: 24239315396\nvirtual: true`),
      null,
    )
  for (const garmin of ['0', '-1', '24239315396.5', '1e3', '9007199254740992'])
    assert.equal(
      parseTrackingBlock(null, `activity: 20037941355\ngarmin: ${garmin}\nvirtual: true`),
      null,
    )
  assert.equal(parseTrackingBlock(null, 'activity: 20037941355\nvirtual: maybe'), null)
  assert.equal(parseTrackingBlock(null, 'activity: 20037941355'), null)
})

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
      activity: null,
      strength: null,
      sauna: null,
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

test('parses a manual sauna session without treating its activity kind as a Strava ID', () => {
  const parsed = parseTrackingBlock(
    null,
    [
      'title: Untangle',
      'date: 2026-08-23',
      'time: 18:30',
      'duration: 75 mins',
      'activity: sauna',
      'temperature: 196F',
      'humidity: 11%',
      'cooldown: cold plunge',
      'htl: 7.7',
    ].join('\n'),
  )

  assert.deepEqual(parsed?.sauna, {
    id: 8_202_608_231_830,
    stravaActivityId: null,
    garminActivityId: null,
    title: 'Untangle',
    date: '2026-08-23',
    time: '18:30',
    durationS: 4_500,
    temperatureC: 91.111,
    humidityPct: 11,
    cooldown: 'cold plunge',
    heatTrainingLoad: 7.7,
  })
  assert.equal(parsed?.fueling, null)
  assert.equal(parsed?.strength, null)
  assert.equal(parsed?.trainingExclusion, null)
})

test('allows an omitted heat load and rejects incomplete sauna metadata', () => {
  assert.deepEqual(
    parseTrackingBlock(
      null,
      [
        'date: 2026-08-20',
        'time: 15:30',
        'duration: 65 min',
        'activity: sauna',
        'temperature: 185F',
        'humidity: 11%',
        'cooldown: natural',
      ].join('\n'),
    )?.sauna,
    {
      id: 8_202_608_201_530,
      stravaActivityId: null,
      garminActivityId: null,
      title: null,
      date: '2026-08-20',
      time: '15:30',
      durationS: 3_900,
      temperatureC: 85,
      humidityPct: 11,
      cooldown: 'natural',
      heatTrainingLoad: null,
    },
  )
  assert.equal(
    parseTrackingBlock(
      null,
      [
        'date: 2026-08-20',
        'time: 15:30',
        'duration: 65 min',
        'activity: sauna',
        'temperature: 185F',
        'humidity: 101%',
        'cooldown: natural',
      ].join('\n'),
    )?.sauna,
    null,
  )
})

test('parses sauna provider attachments and rejects invalid activity IDs', () => {
  const body = [
    'date: 2026-09-02',
    'time: 17:30',
    'duration: 75 mins',
    'activity: sauna',
    'temperature: 160F',
    'humidity: 11%',
    'cooldown: cold plunge',
    'htl: 7.7',
  ]
  const parsed = parseTrackingBlock(
    null,
    [...body, 'strava: 20012367069', 'garmin: 24229638323'].join('\n'),
  )?.sauna
  assert.equal(parsed?.stravaActivityId, 20_012_367_069)
  assert.equal(parsed?.garminActivityId, 24_229_638_323)
  assert.equal(parseTrackingBlock(null, [...body, 'strava: 20012367069.5'].join('\n'))?.sauna, null)
  assert.equal(parseTrackingBlock(null, [...body, 'garmin: 24229638323.5'].join('\n'))?.sauna, null)
})
