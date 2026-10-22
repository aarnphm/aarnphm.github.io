import assert from 'node:assert/strict'
import test from 'node:test'
import {
  applyManualStrength,
  buildPayload,
  type RawStravaActivity,
  type StravaRawCache,
} from '../plugins/stores/strava'
import { parseTrackingBlock } from '../plugins/stores/tracking'

const description = `Focus: controlled explosive movement.

Warm-up (2 min)
30s Leg swings (front-back, each leg)
30s Inchworms
30s Bodyweight squats (slow)
30s Arm circles + torso twists

Main Circuit — 4 rounds, 40s work / 20s rest (16 min)
Squat jump, soft landing — land quiet, absorb through hips/knees, no rush
Plank shoulder taps — hips still, brace core throughout
Split squat with rear foot elevated (or flat) (alternate legs each round) — controlled tempo
Mountain climbers (moderate pace, not max speed) — hips low, core braced
Step-back lunge to balance hold (alternate legs) — 1s hold at top on standing leg

Cool-down (2 min) — figure-4 glute stretch, downward dog, seated hamstring stretch`

function cacheFor(name: string, body: string | null, sportType = 'WeightTraining'): StravaRawCache {
  const activity: RawStravaActivity = {
    id: 20682792884,
    name,
    sportType,
    distance: 0,
    movingTime: 1360,
    elapsedTime: 1360,
    totalElevationGain: 0,
    startDate: '2026-10-22T01:38:50Z',
    startDateLocal: '2026-10-21T21:38:50Z',
    averageSpeed: 0,
    calories: 221,
  }
  return {
    version: 8,
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: 0 },
    lastSync: Date.parse('2026-10-22T02:00:00Z'),
    lastActivityStart: 0,
    activities: { [activity.id]: activity },
    activityDetails: {
      [activity.id]: {
        description: body,
        calories: 221,
        laps: [],
        segmentEfforts: [],
        splitsMetric: [],
        splitsStandard: [],
      },
    },
  }
}

const project = (name: string, body: string | null, sportType = 'WeightTraining') =>
  buildPayload(cacheFor(name, body, sportType), null, null).details['20682792884']

test('projects the described prescription without changing recorded measurements', () => {
  const activity = project('(E) Power & Control', description)
  assert.equal(activity.date, '2026-10-21')
  assert.equal(activity.movingTimeS, 1360)
  assert.equal(activity.calories, 221)
  assert.equal(activity.strength?.source, 'strava-description')
  assert.equal(activity.strength?.exercises.length, 10)
  const main = activity.strength?.exercises.slice(4, 9)
  assert.ok(main?.every(exercise => exercise.setCount === 4 && exercise.durationS === 160))
  assert.equal(main[0].name, 'Squat Jump')
  assert.match(main[0].instructions ?? '', /soft landing/)
  assert.deepEqual(activity.strength?.prescription, {
    rounds: 4,
    workS: 40,
    restS: 20,
    durationS: 1440,
  })
  assert.equal(activity.strength?.totalSets, null)
  assert.equal(activity.strength?.totalReps, null)
  assert.equal(activity.strength?.volumeKg, null)
})

test('reads changed rounds and effort from the description instead of title defaults', () => {
  const activity = project(
    '(E) Power & Control',
    description.replace('4 rounds, 40s work / 20s rest', '2 rounds, 30s work / 15s rest'),
  )
  assert.equal(activity.strength?.exercises[4].setCount, 2)
  assert.equal(activity.strength?.exercises[4].durationS, 60)
  assert.equal(activity.strength?.prescription?.durationS, 690)
})

test('recognizes the five saved titles with prefixes and punctuation normalization', () => {
  for (const name of [
    'Foundational Stability',
    '(D) Posterior Chain & Hips',
    '(E) power & control',
    'Core (anti-rotation)',
    'Fatigue endurance',
  ]) {
    const activity = project(name, null)
    assert.equal(activity.strength?.source, 'routine-title', name)
    assert.equal(activity.strength?.exercises.length, 10, name)
  }
  assert.equal(project('Fatigue endurance', null).strength?.exercises[4].setCount, 3)
  assert.equal(project('Fatigue endurance', null).strength?.prescription?.durationS, 1140)
})

test('accepts a structured description with an unrelated title and rejects incomplete or ambiguous circuits', () => {
  assert.equal(project('Evening strength', description).strength?.source, 'strava-description')
  for (const body of [
    description.replace('Warm-up (2 min)', 'Preparation'),
    description.replace('40s work', 'work to fatigue'),
    description + '\n' + description,
  ])
    assert.equal(project('(E) Power & Control', body).strength, null)
  for (const name of ['Strength', 'Power & Control recovery walk', 'Posterior Chain', ''])
    assert.equal(project(name, null).strength, null)
  assert.equal(project('(E) Power & Control', description, 'Run').strength, null)
})

test('an explicit manual exercise correction retains precedence', () => {
  const payload = buildPayload(cacheFor('(E) Power & Control', description), null, null)
  const correction = parseTrackingBlock(
    null,
    'date: 2026-10-21\nactivity: 20682792884\nexercise: Squat Jump | 30s | 30s',
  )?.strength
  assert.ok(correction)
  applyManualStrength(payload, [correction])
  assert.equal(payload.details['20682792884'].strength?.source, 'manual')
  assert.equal(payload.details['20682792884'].strength?.exercises.length, 1)
  assert.equal(payload.details['20682792884'].strength?.prescription, undefined)
})

test('manual totals retain the inferred exercises and their independent provenance', () => {
  const payload = buildPayload(cacheFor('(E) Power & Control', description), null, null)
  const inferred = structuredClone(payload.details['20682792884'].strength)
  assert.ok(inferred)
  const totals = parseTrackingBlock(
    null,
    'date: 2026-10-21\nactivity: 20682792884\nstrengthVolume: 0 kg\nstrengthSets: 6\nstrengthReps: 59',
  )?.strength
  assert.ok(totals)
  applyManualStrength(payload, [totals])
  assert.deepEqual(payload.details['20682792884'].strength, {
    ...inferred,
    volumeKg: 0,
    totalSets: 6,
    totalReps: 59,
    totalsSource: 'manual',
  })
})

test('TrainingPeaks form cues retain the concise names used in earlier logs', () => {
  for (const [instruction, expected] of [
    [
      'hip flexor stretch, calf stretch, standing quad stretch',
      'Hip Flexor, Calf, and Standing Quad Stretch Flow',
    ],
    [
      'standing calf stretch, seated forward fold, pigeon stretch each side',
      'Standing Calf, Seated Forward Fold, and Pigeon Stretch Flow',
    ],
    [
      "child's pose, thread-the-needle stretch each side",
      "Child's Pose and Thread-the-Needle Stretch",
    ],
    [
      'full-body stretch flow: hip flexors, hamstrings, calves, lower back',
      'Hip Flexor, Hamstring, Calf, and Lower Back Stretch Flow',
    ],
  ]) {
    const activity = project(
      'Strength',
      description.replace(/Cool-down[^]*$/, `Cool-down (2 min) — ${instruction}`),
    )
    assert.equal(activity.strength?.exercises.at(-1)?.name, expected)
    assert.equal(activity.strength?.exercises.at(-1)?.instructions, instruction)
  }
  const activity = project(
    'Strength',
    description.replace(
      'Squat jump, soft landing — land quiet, absorb through hips/knees, no rush',
      'Bodyweight squat with 2s pause at bottom',
    ),
  )
  assert.equal(activity.strength?.exercises[4].name, 'Bodyweight Squat with 2s Pause')
})

test('editing one inferred activity cannot change the next activity prescription', () => {
  const first = project('(E) Power & Control', null)
  assert.ok(first.strength)
  first.strength.exercises[4].name = 'changed'
  first.strength.exercises[4].sets[0].durationS = 1
  const next = project('(E) Power & Control', null)
  assert.match(next.strength?.exercises[4].name ?? '', /Squat/i)
  assert.equal(next.strength?.exercises[4].sets[0].durationS, 40)
})
