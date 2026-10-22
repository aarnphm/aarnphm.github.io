import assert from 'node:assert/strict'
import test from 'node:test'
import type { TrainingPeaksCalendar, TrainingPeaksCalendarWorkout } from './trainingpeaks-calendar'
import { applyManualStrength, buildPayload, type StravaRawCache } from '../plugins/stores/strava'
import { enrichTrainingPeaksStrength } from './strength-trainingpeaks'

const cache: StravaRawCache = {
  version: 8,
  athleteId: 1,
  auth: { refreshToken: '', obtainedAt: 0 },
  lastSync: 1,
  lastActivityStart: 1,
  activities: {
    20682792884: {
      id: 20682792884,
      name: '(E) Power & Control',
      sportType: 'WeightTraining',
      distance: 0,
      movingTime: 1360,
      elapsedTime: 1360,
      totalElevationGain: 0,
      averageSpeed: 0,
      startDate: '2026-10-22T01:38:50Z',
      startDateLocal: '2026-10-21T21:38:50Z',
    },
  },
}

const workout = (
  overrides: Partial<TrainingPeaksCalendarWorkout> = {},
): TrainingPeaksCalendarWorkout => ({
  id: '4010499933',
  source: 'trainingpeaks',
  date: '2026-10-21',
  title: '(E) Power & Control',
  sport: 'strength',
  workoutTypeId: 9,
  description: '',
  planned: { durationSeconds: 1200, distanceMeters: null, tss: null },
  actual: { durationSeconds: 1360, distanceMeters: 0, tss: null },
  status: 'completed',
  startTime: '21:38:50',
  startTimePlanned: null,
  order: null,
  ...overrides,
})
const calendar = (workouts: TrainingPeaksCalendarWorkout[]): TrainingPeaksCalendar => ({
  version: 1,
  source: 'trainingpeaks',
  fetchedAt: null,
  coverage: [],
  workouts,
})
const description = `Warm-up (2 min)
30s Leg swings
30s Inchworms
30s Bodyweight squats
30s Arm circles
Main Circuit — 2 rounds, 30s work / 15s rest
Squat jump
Plank shoulder taps
Split squat
Mountain climbers
Step-back lunge
Cool-down (2 min) — stretch flow`

test('a matched TrainingPeaks description overrides Strava routine defaults', () => {
  const payload = buildPayload(cache, null, null)
  enrichTrainingPeaksStrength(payload, cache, calendar([workout({ description })]))
  const activity = payload.details['20682792884']
  assert.equal(activity.strength?.source, 'trainingpeaks-description')
  assert.equal(activity.strength?.prescription?.rounds, 2)
  assert.equal(activity.strength?.prescription?.trainingPeaksWorkoutId, '4010499933')
  assert.equal(activity.elapsedTimeS, 1360)
})

test('a unique same-day named plan supplies exercises for an existing recording', () => {
  const payload = buildPayload(cache, null, null)
  enrichTrainingPeaksStrength(
    payload,
    cache,
    calendar([
      workout({
        status: 'planned',
        actual: { durationSeconds: null, distanceMeters: null, tss: null },
        description,
      }),
    ]),
  )
  assert.equal(payload.details['20682792884'].strength?.source, 'trainingpeaks-description')
  const empty = buildPayload(null, null, null)
  enrichTrainingPeaksStrength(empty, null, calendar([workout({ status: 'planned', description })]))
  assert.equal(Object.keys(empty.details).length, 0)
})

test('competing recordings or planned prescriptions cannot claim the same session', () => {
  const payload = buildPayload(cache, null, null)
  const before = structuredClone(payload.details['20682792884'].strength)
  enrichTrainingPeaksStrength(
    payload,
    cache,
    calendar([
      workout({ status: 'planned', description }),
      workout({ id: '999', status: 'planned', description }),
    ]),
  )
  assert.deepEqual(payload.details['20682792884'].strength, before)
  const first = cache.activities['20682792884']
  const duplicated = {
    ...cache,
    activities: { ...cache.activities, 20682792885: { ...first, id: 20682792885 } },
  }
  const two = buildPayload(duplicated, null, null)
  enrichTrainingPeaksStrength(two, duplicated, calendar([workout({ description })]))
  assert.ok(
    Object.values(two.details).every(activity => activity.strength?.source === 'routine-title'),
  )
})

test('manual totals remain authoritative across TrainingPeaks reprojection', () => {
  const payload = buildPayload(cache, null, null)
  const correction = {
    activityId: 20682792884,
    date: '2026-10-21',
    exercises: [],
    volumeKg: 10,
    totalSets: 1,
    totalReps: 5,
  }
  applyManualStrength(payload, [correction])
  enrichTrainingPeaksStrength(payload, cache, calendar([workout({ description })]))
  const strength = payload.details['20682792884'].strength
  assert.equal(strength?.source, 'trainingpeaks-description')
  assert.equal(strength?.totalsSource, 'manual')
  assert.equal(strength?.volumeKg, 10)
  assert.equal(strength?.totalSets, 1)
  assert.equal(strength?.totalReps, 5)
  assert.equal(strength?.exercises.length, 10)
  assert.equal(strength?.prescription?.rounds, 2)
})
