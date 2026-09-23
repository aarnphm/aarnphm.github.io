import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseTrainingPeaksWorkout,
  TrainingPeaksApi,
  trainingPeaksMetadataPayload,
  verifyTrainingPeaksMetadata,
  type TrainingPeaksWorkout,
} from './trainingpeaks-api'
import { trainingPeaksWorkoutSummary } from './trainingpeaks-title-sync'

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
    sport: 'Other',
    distance: 0,
    planned: false,
  })
  for (const change of [
    { totalTime: null },
    { totalTime: 0 },
    { startTimePlanned: '2026-09-20T18:30:00' },
    { totalTimePlanned: 1 },
    { tssPlanned: 0 },
    { distancePlanned: 1000 },
  ])
    assert.ok(trainingPeaksWorkoutSummary(workout(change)).planned)
  assert.equal(trainingPeaksWorkoutSummary(workout({ workoutTypeValueId: 2 })).sport, '')
  assert.equal(trainingPeaksWorkoutSummary(workout({ startTime: null })).startTime, '')
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
