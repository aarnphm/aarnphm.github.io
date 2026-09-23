import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity, StravaRawCache } from '../plugins/stores/strava'
import {
  selectTrainingPeaksSaunaSources,
  trainingPeaksSaunaCandidates,
  trainingPeaksSaunaDescription,
  trainingPeaksStartMatches,
  type TrainingPeaksSaunaSource,
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

function source(): TrainingPeaksSaunaSource {
  return selectTrainingPeaksSaunaSources(cache([activity()]), [])[0]
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
    sport: 'Other',
    distance: 0,
    planned: false,
    ...overrides,
  }
}

test('selects sauna from Strava metadata without a manual tracking block and normalizes line endings', () => {
  const selected = source()
  assert.equal(selected.date, '2026-09-20')
  assert.equal(selected.startTime, '18:26:21')
  assert.equal(selected.title, 'Sauna - Guided Down: Untangled')
  assert.equal(
    selected.description,
    'Sauna / passive heat session.\n\n@ Othership. HTL 7.6\n\nCold plunge\n\nStrava: https://www.strava.com/activities/20261500024',
  )
})

test('does not select commutes to Othership, ordinary workouts, or incidental cool-down notes', () => {
  assert.equal(
    selectTrainingPeaksSaunaSources(cache([activity({ sportType: 'Ride', distance: 5000 })]), [])
      .length,
    0,
  )
  for (const description of ['cool-down: stretching', 'Walk to Othership', 'slightly more effort'])
    assert.equal(selectTrainingPeaksSaunaSources(cache([activity()], description), []).length, 0)
  assert.equal(selectTrainingPeaksSaunaSources(cache([activity({ elapsedTime: 0 })]), []).length, 0)
})

test('recognizes explicit passive heat and avoids repeating an existing sauna prefix', () => {
  assert.equal(
    selectTrainingPeaksSaunaSources(
      cache(
        [activity({ name: 'Passive Heat Training', sportType: 'Yoga' })],
        'watch died after 20 min',
      ),
      [],
    ).length,
    1,
  )
  assert.equal(
    selectTrainingPeaksSaunaSources(cache([activity({ name: 'Sauna - Free Flow' })]), [])[0].title,
    'Sauna - Free Flow',
  )
})

test('candidate matching requires the local date, stationary Other sport, completed data, and close duration', () => {
  assert.deepEqual(trainingPeaksSaunaCandidates(source(), [workout({ durationS: 4190 })]), [
    workout({ durationS: 4190 }),
  ])
  for (const changed of [
    { date: '2026-09-21' },
    { startTime: '18:29:00' },
    { durationS: 4500 },
    { distance: 1 },
    { sport: 'Bike' },
    { planned: true },
  ])
    assert.deepEqual(trainingPeaksSaunaCandidates(source(), [workout(changed)]), [])
  assert.equal(
    trainingPeaksSaunaCandidates(source(), [workout(), workout({ id: 'duplicate' })]).length,
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
    trainingPeaksSaunaCandidates(source(), [
      workout(),
      workout({ id: 'earlier', startTime: '10:26:21' }),
    ]),
    [workout()],
  )
})

test('description sync preserves notes and updates its own section without accumulating duplicates', () => {
  const original = source()
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
