import assert from 'node:assert/strict'
import test from 'node:test'
import {
  linkTrainingPeaksCalendarActivities,
  type TrainingPeaksLinkActivity,
} from './trainingpeaks-activity-links'
import {
  parseTrainingPeaksCalendar,
  type TrainingPeaksCalendar,
  type TrainingPeaksCalendarActivityLink,
  type TrainingPeaksCalendarPeak,
  type TrainingPeaksCalendarWorkout,
} from './trainingpeaks-calendar'

// Real-cache verification cannot manufacture duplicate candidates, conflicting
// source IDs, stale links, or mixed provider evidence. These boundary cases must
// leave gaps rather than send a workout to an unrelated activity page.
function workout(
  overrides: Partial<TrainingPeaksCalendarWorkout> = {},
): TrainingPeaksCalendarWorkout {
  return {
    id: '42',
    source: 'trainingpeaks',
    date: '2026-10-05',
    title: 'Evening run',
    sport: 'run',
    workoutTypeId: 3,
    description: '',
    planned: { durationSeconds: 3600, distanceMeters: 10000, tss: 60 },
    actual: { durationSeconds: 2700, distanceMeters: 8000, tss: 47.5 },
    status: 'completed',
    startTime: '20:00:42',
    startTimePlanned: null,
    order: null,
    ...overrides,
  }
}

function activity(overrides: Partial<TrainingPeaksLinkActivity> = {}): TrainingPeaksLinkActivity {
  return {
    id: 123,
    name: 'The existing activity title',
    sport: 'run',
    startDate: '2026-10-06T00:00:42Z',
    startDateLocal: '2026-10-05T20:00:42Z',
    distance: 8000,
    movingTime: 2700,
    elapsedTime: 3000,
    ...overrides,
  }
}

function calendar(workouts: TrainingPeaksCalendarWorkout[]): TrainingPeaksCalendar {
  return {
    version: 1,
    source: 'trainingpeaks',
    fetchedAt: '2026-10-06T12:00:00.000Z',
    coverage: [{ since: '2026-10-01', until: '2026-10-31', fetchedAt: '2026-10-06T12:00:00.000Z' }],
    workouts,
  }
}

test('links by the recorded local date and keeps native TrainingPeaks measurements untouched', () => {
  const input = calendar([workout()])
  const snapshot = structuredClone(input)
  const linked = linkTrainingPeaksCalendarActivities(input, [activity()])
  assert.deepEqual(linked.workouts[0].activity, {
    id: 123,
    date: '2026-10-05',
    title: 'The existing activity title',
    match: 'recording',
  })
  assert.deepEqual(linked.workouts[0].actual, snapshot.workouts[0].actual)
  assert.deepEqual(linked.workouts[0].planned, snapshot.workouts[0].planned)
  assert.deepEqual(input, snapshot)
})

test('linked recording peaks survive the public calendar parser with source provenance', () => {
  const peaks: TrainingPeaksCalendarPeak[] = [
    { seconds: 5, heartRateBpm: 170, heartRateSource: 'garmin', powerWatts: 500 },
    { seconds: 60, heartRateBpm: null, heartRateSource: null, powerWatts: 0 },
  ]
  const linked = linkTrainingPeaksCalendarActivities(calendar([workout()]), [activity({ peaks })])
  const parsed = parseTrainingPeaksCalendar(JSON.parse(JSON.stringify(linked)))
  assert.deepEqual(parsed?.workouts[0].activity?.peaks, peaks)
  assert.deepEqual(parsed?.workouts[0].actual, workout().actual)
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([workout()]), [
      activity({ id: 124, peaks }),
      activity(),
    ]).workouts[0].activity,
    undefined,
  )
  for (const invalid of [
    [{ ...peaks[0], seconds: 59 }],
    [{ ...peaks[0], heartRateBpm: 0 }],
    [{ ...peaks[0], heartRateSource: null }],
    [{ ...peaks[0], powerWatts: -1 }],
    [peaks[0], peaks[0]],
  ]) {
    const corrupted = structuredClone(linked)
    assert.ok(corrupted.workouts[0].activity)
    Object.assign(corrupted.workouts[0].activity, { peaks: invalid })
    assert.equal(parseTrainingPeaksCalendar(corrupted), null)
  }
})

test('rejects title-only matches, conflicting recordings, and activities outside the supplied emitted set', () => {
  for (const candidate of [
    activity({ startDateLocal: '2026-10-04T20:00:42Z' }),
    activity({ sport: 'bike' }),
    activity({ startDateLocal: '2026-10-05T08:00:42Z' }),
    activity({ movingTime: 7200, elapsedTime: 7200 }),
    activity({ distance: 20000 }),
    activity({ id: -1 }),
  ])
    assert.equal(
      linkTrainingPeaksCalendarActivities(calendar([workout()]), [candidate]).workouts[0].activity,
      undefined,
    )
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([workout()]), []).workouts[0].activity,
    undefined,
  )
})

test('requires unique one-to-one recording evidence across all workouts and activities', () => {
  const sameRecording = calendar([workout(), workout({ id: '43', title: 'A duplicate import' })])
  assert.ok(
    linkTrainingPeaksCalendarActivities(sameRecording, [activity()]).workouts.every(
      item => !item.activity,
    ),
  )
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([workout()]), [activity(), activity({ id: 124 })])
      .workouts[0].activity,
    undefined,
  )
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([workout()]), [activity(), activity()]).workouts[0]
      .activity,
    undefined,
  )
  const commutes = calendar([
    workout({ sport: 'bike', workoutTypeId: 2, startTime: '08:00:00' }),
    workout({ id: '43', sport: 'bike', workoutTypeId: 2, startTime: '17:00:00' }),
  ])
  const recordings = [
    activity({ id: 123, sport: 'bike', startDateLocal: '2026-10-05T08:00:00Z' }),
    activity({ id: 124, sport: 'bike', startDateLocal: '2026-10-05T17:00:00Z' }),
  ]
  assert.deepEqual(
    linkTrainingPeaksCalendarActivities(commutes, recordings).workouts.map(
      item => item.activity?.id,
    ),
    [123, 124],
  )
  assert.deepEqual(
    linkTrainingPeaksCalendarActivities(commutes, recordings.toReversed()).workouts.map(
      item => item.activity?.id,
    ),
    [123, 124],
  )
})

test('uses explicit Strava IDs with date, sport, and recording conflict checks', () => {
  const referenced = workout({ description: 'Strava: https://www.strava.com/activities/123' })
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([referenced]), [activity()]).workouts[0].activity
      ?.match,
    'source-id',
  )
  for (const description of [
    'https://www.strava.com/activities/999',
    'https://www.strava.com/activities/123 and https://strava.com/activities/124',
  ])
    assert.equal(
      linkTrainingPeaksCalendarActivities(calendar([workout({ description })]), [activity()])
        .workouts[0].activity,
      undefined,
    )
  for (const candidate of [
    activity({ sport: 'bike' }),
    activity({ distance: 20000 }),
    activity({ startDateLocal: '2026-10-05T08:00:00Z' }),
  ])
    assert.equal(
      linkTrainingPeaksCalendarActivities(calendar([referenced]), [candidate]).workouts[0].activity,
      undefined,
    )
  assert.equal(
    linkTrainingPeaksCalendarActivities(
      calendar([workout({ description: 'https://strava.com.example.org/activities/123' })]),
      [activity()],
    ).workouts[0].activity?.match,
    'recording',
  )
})

test('allows device timer differences within one verified recording and never mixes provider metrics', () => {
  const strength = workout({
    sport: 'strength',
    workoutTypeId: 9,
    actual: { durationSeconds: 1445, distanceMeters: 0, tss: 12 },
  })
  const recorded = activity({
    sport: 'strength',
    distance: 0,
    movingTime: 1654,
    elapsedTime: 1654,
    garmin: {
      startDate: '2026-10-06T00:00:42Z',
      distanceM: null,
      movingTimeS: 1040,
      elapsedTimeS: 1654,
    },
  })
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([strength]), [recorded]).workouts[0].activity?.id,
    123,
  )
  const conflicting = activity({
    distance: 5000,
    garmin: {
      startDate: '2026-10-06T00:00:42Z',
      distanceM: 8000,
      movingTimeS: 6000,
      elapsedTimeS: 7200,
    },
  })
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([workout()]), [conflicting]).workouts[0].activity,
    undefined,
  )
  const companion = activity({
    startDate: '2026-10-06T00:01:42Z',
    startDateLocal: '2026-10-05T20:01:42Z',
    distance: 5000,
    garmin: {
      startDate: '2026-10-06T00:00:42Z',
      distanceM: 8000,
      movingTimeS: 2600,
      elapsedTimeS: 3000,
    },
  })
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([workout()]), [companion]).workouts[0].activity
      ?.id,
    123,
  )
})

test('matches ancillary sessions without treating zero distance as independent recording evidence', () => {
  const sauna = workout({
    sport: 'other',
    workoutTypeId: 100,
    actual: { durationSeconds: 2700, distanceMeters: 0, tss: 20 },
  })
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([sauna]), [
      activity({ sport: 'sauna', distance: 0 }),
    ]).workouts[0].activity?.id,
    123,
  )
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([sauna]), [
      activity({ sport: 'run', distance: 0 }),
    ]).workouts[0].activity,
    undefined,
  )
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([{ ...sauna, startTime: null }]), [
      activity({ sport: 'sauna', distance: 0 }),
    ]).workouts[0].activity,
    undefined,
  )
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([workout({ startTime: null })]), [activity()])
      .workouts[0].activity?.id,
    123,
  )
})

test('clears stale links and never attaches planned workouts', () => {
  const previous: TrainingPeaksCalendarActivityLink = {
    id: 123,
    date: '2026-10-05',
    title: 'Old activity',
    match: 'recording',
  }
  const linked = workout({ activity: previous })
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([linked]), []).workouts[0].activity,
    undefined,
  )
  const planned = workout({
    status: 'planned',
    actual: { durationSeconds: null, distanceMeters: null, tss: null },
    description: 'https://www.strava.com/activities/123',
  })
  assert.equal(
    linkTrainingPeaksCalendarActivities(calendar([planned]), [activity()]).workouts[0].activity,
    undefined,
  )
})

test('validates enriched browser payload links while preserving the cache model', () => {
  const link: TrainingPeaksCalendarActivityLink = {
    id: 123,
    date: '2026-10-05',
    title: 'Existing activity',
    match: 'recording',
  }
  const input = calendar([workout({ activity: link })])
  assert.deepEqual(parseTrainingPeaksCalendar(input), input)
  for (const invalid of [
    { ...link, id: '123' },
    { ...link, id: -1 },
    { ...link, date: '2026-10-06' },
    { ...link, match: 'date-only' },
    { ...link, title: null },
  ])
    assert.equal(
      parseTrainingPeaksCalendar({
        ...input,
        workouts: [{ ...input.workouts[0], activity: invalid }],
      }),
      null,
    )
})
