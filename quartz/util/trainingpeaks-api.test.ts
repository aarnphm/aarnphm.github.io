import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseTrainingPeaksWorkout,
  TrainingPeaksApi,
  trainingPeaksMetadataPayload,
  verifyTrainingPeaksMetadata,
  type TrainingPeaksWorkout,
  parseTrainingPeaksStrengthWorkout,
  trainingPeaksStrengthTitlePayload,
  verifyTrainingPeaksStrengthTitle,
} from './trainingpeaks-api'
import {
  trainingPeaksWorkoutSummary,
  trainingPeaksTitleProtection,
  trainingPeaksStrengthTitleProtection,
} from './trainingpeaks-title-sync'

// Field shapes follow a completed Other workout returned by the TrainingPeaks web API.
function workout(overrides: Partial<TrainingPeaksWorkout> = {}): TrainingPeaksWorkout {
  return {
    workoutId: 42,
    athleteId: 1,
    title: 'Cardio',
    description: 'Existing coach note.',
    workoutTypeValueId: 100,
    workoutDay: '2026-09-20T00:00:00',
    startTime: '2026-09-20T18:26:21',
    startTimePlanned: null,
    totalTime: 1.1653252840042114,
    distance: 0,
    isLocked: null,
    totalTimePlanned: null,
    distancePlanned: null,
    tssPlanned: null,
    ifPlanned: null,
    caloriesPlanned: null,
    velocityPlanned: null,
    energyPlanned: null,
    elevationGainPlanned: null,
    completed: null,
    tssActual: 23,
    heartRateAverage: 105,
    structure: null,
    workoutComments: [{ comment: 'Keep this note.', personId: 2 }],
    lastModifiedDate: '2026-09-22T15:01:58',
    sharedWorkoutInformationExpireKey: 'old-expiry-key',
    ...overrides,
  }
}

test('validates provider identity and rejects incomplete workout schemas', () => {
  const current = workout()
  assert.equal(parseTrainingPeaksWorkout(current, 1), current)
  assert.throws(() => parseTrainingPeaksWorkout(current, 2), /another athlete/)
  for (const invalid of [
    null,
    [],
    { ...current, workoutId: '42' },
    { ...current, totalTime: '1.16' },
    { ...current, totalTime: NaN },
    { ...current, totalTimePlanned: undefined },
    { ...current, description: undefined },
  ])
    assert.throws(() => parseTrainingPeaksWorkout(invalid, 1), /unexpected workout schema/)
})

test('converts API hours and local timestamps without relying on the nullable completed flag', () => {
  assert.deepEqual(trainingPeaksWorkoutSummary(workout()), {
    id: '42',
    date: '2026-09-20',
    startTime: '18:26:21',
    durationS: 4195,
    title: 'Cardio',
    workoutTypeId: 100,
    distance: 0,
    completed: true,
  })
  for (const change of [{ totalTime: null }, { totalTime: 0 }])
    assert.equal(trainingPeaksWorkoutSummary(workout(change)).completed, false)
  for (const change of [
    { startTimePlanned: '2026-09-20T18:30:00' },
    { totalTimePlanned: 1 },
    { tssPlanned: 0 },
    { distancePlanned: 1000 },
  ])
    assert.equal(trainingPeaksWorkoutSummary(workout(change)).completed, true)
  assert.equal(trainingPeaksWorkoutSummary(workout({ workoutTypeValueId: 2 })).workoutTypeId, 2)
  assert.equal(trainingPeaksWorkoutSummary(workout({ startTime: null })).startTime, '')
})

test('protects completed plans and authored instructions from Strava title changes', () => {
  const imported = workout({ description: null, workoutComments: [] })
  assert.equal(trainingPeaksTitleProtection(imported), null)
  for (const change of [
    { totalTimePlanned: 1 },
    { distancePlanned: 0 },
    { tssPlanned: 0 },
    { ifPlanned: 0.7 },
    { caloriesPlanned: 250 },
    { velocityPlanned: 3 },
    { energyPlanned: 300 },
    { elevationGainPlanned: 0 },
    { startTimePlanned: '2026-09-20T18:30:00' },
    { structure: '{"structure":[]}' },
  ])
    assert.equal(trainingPeaksTitleProtection({ ...imported, ...change }), 'Planned workout')
  assert.equal(
    trainingPeaksTitleProtection({ ...imported, description: 'Half mile warmup, then 4 x 800m.' }),
    'Workout has authored instructions',
  )
  const managed =
    'Sauna / passive heat session.\n\nHTL 7.7\n\nStrava: https://www.strava.com/activities/123'
  assert.equal(trainingPeaksTitleProtection({ ...imported, description: managed }), null)
  assert.equal(
    trainingPeaksTitleProtection({
      ...imported,
      description: managed + '\n\nJames: keep this short.',
    }),
    'Workout has authored instructions',
  )
})

test('metadata payload preserves metrics, comments, structure, and unknown provider fields', () => {
  const current = workout({ structure: '{"structure":[]}', futureField: { value: [1, 2] } })
  const snapshot = structuredClone(current)
  const payload = trainingPeaksMetadataPayload(current, 'Sauna - Free Flow', 'Updated description.')
  assert.deepEqual(payload, {
    ...snapshot,
    title: 'Sauna - Free Flow',
    description: 'Updated description.',
  })
  assert.deepEqual(current, snapshot)
  const withoutDescription = workout({ description: null })
  assert.deepEqual(trainingPeaksMetadataPayload(withoutDescription, 'New title'), {
    ...withoutDescription,
    title: 'New title',
  })
  assert.throws(() => trainingPeaksMetadataPayload(current, ' ', ''), /must not be empty/)
  assert.throws(
    () => trainingPeaksMetadataPayload(workout({ isLocked: true }), 'Sauna', ''),
    /locked/,
  )
})

test('readback accepts server save markers and fails if metadata or workout data changed', () => {
  const expected = workout()
  verifyTrainingPeaksMetadata(expected, {
    ...expected,
    lastModifiedDate: '2026-09-22T16:01:58',
    sharedWorkoutInformationExpireKey: 'new-expiry-key',
  })
  for (const change of [{ title: 'Different' }, { description: 'Different' }])
    assert.throws(
      () => verifyTrainingPeaksMetadata(expected, workout(change)),
      /metadata readback failed/,
    )
  for (const change of [
    { athleteId: 2 },
    { workoutId: 43 },
    { totalTime: 2 },
    { tssActual: 25 },
    { workoutComments: [] },
    { distancePlanned: 0 },
  ])
    assert.throws(
      () => verifyTrainingPeaksMetadata(expected, workout(change)),
      /changed additional fields/,
    )
})

test('requires scoped authentication and rejects cookie header injection', () => {
  assert.throws(() => new TrainingPeaksApi({}), /Set TRAININGPEAKS_AUTH_COOKIE/)
  assert.doesNotThrow(() => new TrainingPeaksApi({ accessToken: 'Bearer existing-token' }))
  assert.doesNotThrow(() => new TrainingPeaksApi({ authCookie: 'existing-cookie' }))
  for (const accessToken of ['token\r\nInjected: value', 'token with spaces', 'token\u0000'])
    assert.throws(() => new TrainingPeaksApi({ accessToken }), /single bearer token/)
  for (const authCookie of ['cookie; extra=value', 'cookie\r\nInjected: value'])
    assert.throws(
      () => new TrainingPeaksApi({ authCookie }),
      /only the Production_tpAuth cookie value/,
    )
})

test('strength titles preserve exercises, files, plans and completed metrics', () => {
  const current = parseTrainingPeaksStrengthWorkout(
    {
      id: '42',
      calendarId: 1,
      title: 'Yoga',
      instructions: null,
      prescribedDate: '2026-09-20',
      startDateTime: '2026-09-20T18:15:20',
      completedDateTime: '2026-09-20T18:24:12',
      executedDurationInSeconds: 532,
      workoutType: 'StructuredStrength',
      workoutSubTypeId: 22,
      isLocked: false,
      prescribedDurationInSeconds: null,
      completedTss: 5,
      blocks: [{ id: 'block', sets: [1, 2] }],
      files: [{ fileName: 'recording.fit' }],
    },
    1,
  )
  const payload = trainingPeaksStrengthTitlePayload(current, 'Lower body stretch')
  assert.deepEqual(payload, { ...current, title: 'Lower body stretch' })
  verifyTrainingPeaksStrengthTitle(payload, { ...payload, lastUpdatedAt: '2026-10-06T20:00:00' })
  for (const change of [{ title: 'Yoga' }, { blocks: [] }, { completedTss: 0 }, { calendarId: 2 }])
    assert.throws(
      () => verifyTrainingPeaksStrengthTitle(payload, { ...payload, ...change }),
      /readback/,
    )
  assert.throws(() => parseTrainingPeaksStrengthWorkout(current, 2), /another athlete/)
  assert.throws(() => parseTrainingPeaksStrengthWorkout({ ...current, id: null }, 1), /schema/)
  assert.throws(
    () => trainingPeaksStrengthTitlePayload({ ...current, isLocked: true }, 'Stretch'),
    /locked/,
  )
  assert.throws(() => trainingPeaksStrengthTitlePayload(current, ''), /empty/)
})

test('protects strength prescriptions without treating an imported activity date as a plan', () => {
  const imported = parseTrainingPeaksStrengthWorkout(
    {
      id: '42',
      calendarId: 1,
      title: 'Yoga',
      instructions: null,
      prescribedDate: '2026-09-20',
      prescribedStartTime: null,
      startDateTime: '2026-09-20T18:15:20',
      completedDateTime: '2026-09-20T18:24:12',
      executedDurationInSeconds: 532,
      workoutType: 'StructuredStrength',
      workoutSubTypeId: 22,
      isLocked: false,
      hasPrescribedData: false,
      complianceState: 'Unplanned',
      prescribedDurationInSeconds: null,
      prescribedTss: null,
      prescribedIntensityFactor: null,
      blocks: [],
      sequenceSummary: [],
      completedTss: 5,
    },
    1,
  )
  assert.equal(trainingPeaksStrengthTitleProtection(imported), null)
  for (const change of [
    { hasPrescribedData: true },
    { prescribedDurationInSeconds: 1200 },
    { prescribedTss: 0 },
    { prescribedIntensityFactor: 0.5 },
    { prescribedStartTime: '18:00:00' },
    { blocks: [{ id: 'exercise' }] },
    { sequenceSummary: [{ id: 'exercise' }] },
    { complianceState: 'Complete' },
  ])
    assert.equal(
      trainingPeaksStrengthTitleProtection({ ...imported, ...change }),
      'Planned workout',
    )
  assert.equal(
    trainingPeaksStrengthTitleProtection({ ...imported, instructions: '3 x 10 split squats' }),
    'Workout has authored instructions',
  )
})
