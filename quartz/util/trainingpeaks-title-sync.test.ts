import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity, StravaRawCache } from '../plugins/stores/strava'
import {
  selectTrainingPeaksTitleSources,
  trainingPeaksTitleCandidates,
  trainingPeaksSaunaDescription,
  trainingPeaksStartMatches,
  trainingPeaksSupportedTitle,
  type TrainingPeaksTitleSource,
  type TrainingPeaksWorkoutSummary,
} from './trainingpeaks-title-sync'

function activity(overrides: Partial<RawStravaActivity> = {}): RawStravaActivity {
  return {
    id: 20261500024,
    name: 'Guided Down: Untangled',
    sportType: 'Workout',
    distance: 0,
    startDate: '2026-09-20T22:26:21Z',
    startDateLocal: '2026-09-20T18:26:21Z',
    movingTime: 4195,
    elapsedTime: 4195,
    totalElevationGain: 0,
    averageSpeed: 0,
    ...overrides,
  }
}

function cache(
  activities: RawStravaActivity[],
  description = '@ Othership. HTL 7.6\r\n\r\nCold plunge',
): StravaRawCache {
  return {
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: 0 },
    lastSync: 0,
    lastActivityStart: 0,
    activities: Object.fromEntries(activities.map(value => [value.id, value])),
    activityDetails: Object.fromEntries(
      activities.map(value => [
        value.id,
        {
          description,
          calories: null,
          laps: [],
          segmentEfforts: [],
          splitsMetric: [],
          splitsStandard: [],
        },
      ]),
    ),
  }
}

function source(overrides: Partial<RawStravaActivity> = {}): TrainingPeaksTitleSource {
  return selectTrainingPeaksTitleSources(cache([activity(overrides)]), [])[0]
}

function workout(
  overrides: Partial<TrainingPeaksWorkoutSummary> = {},
): TrainingPeaksWorkoutSummary {
  return {
    id: '3962796453',
    date: '2026-09-20',
    startTime: '18:26:21',
    durationS: 4195,
    title: 'Cardio',
    workoutTypeId: 100,
    distance: 0,
    completed: true,
    ...overrides,
  }
}

test('selects sauna from Strava metadata without a manual tracking block and normalizes line endings', () => {
  const selected = source()
  assert.equal(selected.date, '2026-09-20')
  assert.equal(selected.startTime, '18:26:21')
  assert.equal(selected.title, 'Guided Down: Untangled')
  assert.equal(
    selected.description,
    'Sauna / passive heat session.\n\n@ Othership. HTL 7.6\n\nCold plunge\n\nStrava: https://www.strava.com/activities/20261500024',
  )
})

test('selects every activity while limiting sauna descriptions to heat sessions', () => {
  assert.equal(
    selectTrainingPeaksTitleSources(cache([activity({ sportType: 'Ride', distance: 5000 })]), [])[0]
      .description,
    undefined,
  )
  for (const description of ['cool-down: stretching', 'Walk to Othership', 'slightly more effort'])
    assert.equal(
      selectTrainingPeaksTitleSources(cache([activity()], description), [])[0].description,
      undefined,
    )
  assert.equal(selectTrainingPeaksTitleSources(cache([activity({ elapsedTime: 0 })]), []).length, 0)
})

test('recognizes explicit passive heat and avoids repeating an existing sauna prefix', () => {
  assert.equal(
    selectTrainingPeaksTitleSources(
      cache(
        [activity({ name: 'Passive Heat Training', sportType: 'Yoga' })],
        'watch died after 20 min',
      ),
      [],
    ).length,
    1,
  )
  assert.equal(
    selectTrainingPeaksTitleSources(cache([activity({ name: 'Sauna - Free Flow' })]), [])[0].title,
    'Sauna - Free Flow',
  )
})

test('candidate matching requires local date, actual time, compatible sport and completed data', () => {
  assert.deepEqual(trainingPeaksTitleCandidates(source(), [workout({ durationS: 4190 })]), [
    workout({ durationS: 4190 }),
  ])
  for (const changed of [
    { date: '2026-09-21' },
    { startTime: '18:29:00' },
    { durationS: 6000 },
    { distance: 1000 },
    { workoutTypeId: 2 },
    { completed: false },
  ])
    assert.deepEqual(trainingPeaksTitleCandidates(source(), [workout(changed)]), [])
  assert.equal(
    trainingPeaksTitleCandidates(source(), [workout(), workout({ id: 'duplicate' })]).length,
    2,
  )
})

test('checks actual local recording times within two minutes and rejects invalid clocks', () => {
  for (const clock of ['18:24:21', '18:26:21', '18:28:21'])
    assert.ok(trainingPeaksStartMatches(source(), clock))
  for (const clock of ['06:26:21', '18:28:22', '', '24:26:21', '18:60:21', '18:26:60'])
    assert.equal(trainingPeaksStartMatches(source(), clock), false)
  assert.ok(trainingPeaksStartMatches({ ...source(), startTime: '00:00:21' }, '00:01:00'))
  assert.deepEqual(
    trainingPeaksTitleCandidates(source(), [
      workout(),
      workout({ id: 'earlier', startTime: '10:26:21' }),
    ]),
    [workout()],
  )
})

test('description sync preserves notes and updates its own section without accumulating duplicates', () => {
  const selected = source()
  assert.ok(selected.description)
  const original = { stravaId: selected.stravaId, description: selected.description }
  const initial = trainingPeaksSaunaDescription(original, 'James: keep this easy.')
  assert.equal(initial, `James: keep this easy.\n\n${original.description}`)
  assert.equal(trainingPeaksSaunaDescription(original, initial), initial)
  const revised = { ...original, description: original.description.replace('7.6', '7.7') }
  assert.equal(
    trainingPeaksSaunaDescription(revised, `${initial}\n\nFelt relaxed.`),
    `James: keep this easy.\n\n${revised.description}\n\nFelt relaxed.`,
  )
  assert.equal(
    trainingPeaksSaunaDescription(original, '@ Othership. HTL 7.6\r\n\r\nCold plunge'),
    original.description,
  )
})

test('selects exact Strava titles for every sport and keeps ordinary descriptions untouched', () => {
  for (const sportType of [
    'Run',
    'TrailRun',
    'Ride',
    'VirtualRide',
    'EBikeRide',
    'Swim',
    'Walk',
    'Hike',
    'WeightTraining',
    'PhysicalTherapy',
    'Yoga',
    'Pilates',
    'RockClimbing',
  ]) {
    const selected = selectTrainingPeaksTitleSources(
      cache([activity({ sportType, name: '  Session 🏊 & intervals  ' })], ''),
      [],
    )[0]
    assert.equal(selected.title, '  Session 🏊 & intervals  ')
    assert.equal(selected.sportType, sportType)
    assert.equal(selected.description, undefined)
  }
  assert.deepEqual(selectTrainingPeaksTitleSources(cache([activity({ name: ' ' })]), []), [])
})

test('matches moving and elapsed durations without changing either provider metric', () => {
  const ride = source({ sportType: 'Ride', distance: 24000, movingTime: 3500, elapsedTime: 4000 })
  for (const durationS of [3490, 3750, 4000])
    assert.equal(
      trainingPeaksTitleCandidates(ride, [
        workout({ workoutTypeId: 2, distance: 24010, durationS }),
      ]).length,
      1,
    )
  assert.deepEqual(
    trainingPeaksTitleCandidates(ride, [
      workout({ workoutTypeId: 3, distance: 24010, durationS: 3500 }),
    ]),
    [],
  )
  assert.deepEqual(
    trainingPeaksTitleCandidates(ride, [
      workout({ workoutTypeId: 2, distance: 10000, durationS: 3500 }),
    ]),
    [],
  )
})

test('matches swim active time using the same recording start and distance', () => {
  const swim = source({ sportType: 'Swim', distance: 550, movingTime: 1447, elapsedTime: 1447 })
  const recorded = workout({ workoutTypeId: 1, distance: 550, durationS: 719 })
  assert.equal(trainingPeaksTitleCandidates(swim, [recorded]).length, 1)
  assert.deepEqual(trainingPeaksTitleCandidates(swim, [{ ...recorded, startTime: '18:27:21' }]), [])
})

test('handles strength and walks recorded as Other without allowing incompatible measured sports', () => {
  for (const [sportType, workoutTypeId] of [
    ['WeightTraining', 9],
    ['WeightTraining', 100],
    ['Walk', 13],
    ['Walk', 100],
    ['Run', 100],
  ]) {
    if (typeof sportType !== 'string' || typeof workoutTypeId !== 'number')
      throw new Error('invalid fixture')
    assert.equal(
      trainingPeaksTitleCandidates(source({ sportType }), [workout({ workoutTypeId })]).length,
      1,
    )
  }
})

test('removes unsupported emoji graphemes without leaving joiners or empty-looking titles', () => {
  assert.equal(trainingPeaksSupportedTitle('🧗 but Z2 🙂‍↕️'), 'but Z2')
  assert.equal(trainingPeaksSupportedTitle('🧜‍♂️🧜‍♀️ LCB'), 'LCB')
  assert.equal(trainingPeaksSupportedTitle('🐿️🦅'), '')
  assert.equal(trainingPeaksSupportedTitle('Échauffement + 4×100m ♡'), 'Échauffement + 4×100m ♡')
})

test('matches structured strength including a partial recording with the same finish time', () => {
  const stretch = source({
    sportType: 'PhysicalTherapy',
    startDateLocal: '2026-09-20T18:12:06Z',
    elapsedTime: 724,
    movingTime: 724,
  })
  const recorded = workout({
    id: 'strength:42',
    workoutTypeId: 29,
    startTime: '18:15:20',
    endTime: '18:24:12',
    durationS: 532,
  })
  assert.deepEqual(trainingPeaksTitleCandidates(stretch, [recorded]), [recorded])
  assert.deepEqual(
    trainingPeaksTitleCandidates(stretch, [{ ...recorded, endTime: '18:25:00' }]),
    [],
  )
  assert.deepEqual(trainingPeaksTitleCandidates({ ...stretch, sportType: 'Run' }, [recorded]), [])
})
