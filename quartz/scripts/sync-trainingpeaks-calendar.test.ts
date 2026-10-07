import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import type { TrainingPeaksWorkout } from '../util/trainingpeaks-api'
import {
  emptyTrainingPeaksCalendar,
  mergeTrainingPeaksCalendar,
  parseTrainingPeaksCalendar,
} from '../util/trainingpeaks-calendar'
import {
  readTrainingPeaksCalendarCache,
  writeTrainingPeaksCalendarCache,
} from '../util/trainingpeaks-calendar-cache'
import {
  normalizeTrainingPeaksCalendarWorkout,
  parseTrainingPeaksCalendarArgs,
  trainingPeaksCalendarMonths,
} from './sync-trainingpeaks-calendar'

// A successful live pull cannot exercise malformed metrics, HTML, invalid dates,
// moved/deleted IDs, overlapping coverage, or corrupt prior caches. These checks
// cover the normalization and cache boundaries where those failures lose data.
function workout(overrides: Partial<TrainingPeaksWorkout> = {}): TrainingPeaksWorkout {
  return {
    workoutId: 42,
    athleteId: 1,
    title: 'Evening run',
    description: '10min easy\n4 × 600m (<2:10)',
    workoutTypeValueId: 3,
    workoutDay: '2026-10-04T00:00:00',
    startTime: '2026-10-04T23:45:00',
    startTimePlanned: '2026-10-04T18:30:00',
    totalTime: 0.75,
    distance: 8000,
    isLocked: null,
    totalTimePlanned: 1,
    distancePlanned: 10000,
    tssPlanned: 60,
    tssActual: 47.5,
    ifPlanned: null,
    caloriesPlanned: null,
    velocityPlanned: null,
    energyPlanned: null,
    elevationGainPlanned: null,
    completed: null,
    orderOnDay: 2,
    ...overrides,
  }
}

test('normalizes native metrics and local dates without treating nullable completion as evidence', () => {
  const normalized = normalizeTrainingPeaksCalendarWorkout(workout())
  assert.deepEqual(normalized, {
    id: '42',
    source: 'trainingpeaks',
    date: '2026-10-04',
    title: 'Evening run',
    sport: 'run',
    workoutTypeId: 3,
    description: '10min easy\n4 × 600m (<2:10)',
    planned: { durationSeconds: 3600, distanceMeters: 10000, tss: 60 },
    actual: { durationSeconds: 2700, distanceMeters: 8000, tss: 47.5 },
    status: 'completed',
    startTime: '23:45:00',
    startTimePlanned: '18:30:00',
    order: 2,
  })
  const planned = normalizeTrainingPeaksCalendarWorkout(
    workout({ totalTime: null, distance: 0, tssActual: 0, completed: true }),
  )
  assert.equal(planned.status, 'planned')
  assert.deepEqual(planned.actual, { durationSeconds: null, distanceMeters: 0, tss: 0 })
  assert.equal(
    normalizeTrainingPeaksCalendarWorkout(
      workout({ totalTime: null, distance: null, tssActual: 5, workoutDay: '2027-01-04T00:00:00' }),
    ).status,
    'completed',
  )
  assert.equal(
    normalizeTrainingPeaksCalendarWorkout(workout({ workoutTypeValueId: 999 })).sport,
    'other',
  )
})

test('converts provider HTML into text and refuses invalid native values', () => {
  const normalized = normalizeTrainingPeaksCalendarWorkout(
    workout({
      description:
        '<p>Swim &amp; drills</p><script>alert(1)</script><p>4 × 100m<br>Easy &lt;2:00</p>',
    }),
  )
  assert.equal(normalized.description, 'Swim & drills\n\n4 × 100m\nEasy <2:00')
  for (const change of [
    { workoutDay: '2026-02-30T00:00:00' },
    { totalTime: -1 },
    { distancePlanned: Infinity },
    { tssActual: '47.5' },
    { startTimePlanned: '2026-10-04T25:00:00' },
  ])
    assert.throws(() => normalizeTrainingPeaksCalendarWorkout(workout(change)), /TrainingPeaks/)
})

test('defaults to the local calendar year with a ninety-day horizon and validates bounded month pulls', () => {
  assert.deepEqual(parseTrainingPeaksCalendarArgs([], '2026-10-06'), {
    since: '2026-01-01',
    until: '2027-01-04',
  })
  assert.deepEqual(parseTrainingPeaksCalendarArgs([], '2026-02-01'), {
    since: '2026-01-01',
    until: '2026-12-31',
  })
  assert.deepEqual(trainingPeaksCalendarMonths('2026-01-30', '2026-03-02'), [
    { since: '2026-01-30', until: '2026-01-31' },
    { since: '2026-02-01', until: '2026-02-28' },
    { since: '2026-03-01', until: '2026-03-02' },
  ])
  for (const args of [
    ['--since', '2026-02-30'],
    ['--until'],
    ['--since', '2026-03-01', '--until', '2026-02-01'],
    ['--write'],
  ])
    assert.throws(() => parseTrainingPeaksCalendarArgs(args, '2026-10-06'))
})

test('replaces only a fully refreshed range, keeps history and removes stale moved IDs', () => {
  const oldFetch = '2026-10-01T12:00:00.000Z'
  const nextFetch = '2026-10-06T12:00:00.000Z'
  const existing = mergeTrainingPeaksCalendar(
    emptyTrainingPeaksCalendar(),
    [
      normalizeTrainingPeaksCalendarWorkout(
        workout({ workoutId: 1, workoutDay: '2026-09-29T00:00:00' }),
      ),
      normalizeTrainingPeaksCalendarWorkout(
        workout({ workoutId: 2, workoutDay: '2026-10-04T00:00:00' }),
      ),
      normalizeTrainingPeaksCalendarWorkout(
        workout({ workoutId: 3, workoutDay: '2026-10-05T00:00:00' }),
      ),
      normalizeTrainingPeaksCalendarWorkout(
        workout({ workoutId: 4, workoutDay: '2026-10-10T00:00:00' }),
      ),
    ],
    { since: '2026-09-01', until: '2026-10-31', fetchedAt: oldFetch },
  )
  const refreshed = mergeTrainingPeaksCalendar(
    existing,
    [
      normalizeTrainingPeaksCalendarWorkout(
        workout({ workoutId: 1, workoutDay: '2026-10-02T00:00:00' }),
      ),
      normalizeTrainingPeaksCalendarWorkout(workout({ workoutId: 3, title: 'Updated' })),
    ],
    { since: '2026-10-01', until: '2026-10-06', fetchedAt: nextFetch },
  )
  assert.deepEqual(
    refreshed.workouts.map(item => [item.id, item.date]),
    [
      ['1', '2026-10-02'],
      ['3', '2026-10-04'],
      ['4', '2026-10-10'],
    ],
  )
  assert.deepEqual(refreshed.coverage, [
    { since: '2026-09-01', until: '2026-09-30', fetchedAt: oldFetch },
    { since: '2026-10-01', until: '2026-10-06', fetchedAt: nextFetch },
    { since: '2026-10-07', until: '2026-10-31', fetchedAt: oldFetch },
  ])
  assert.equal(existing.workouts.length, 4)
  assert.throws(
    () =>
      mergeTrainingPeaksCalendar(existing, [refreshed.workouts[2]], {
        since: '2026-10-01',
        until: '2026-10-06',
        fetchedAt: nextFetch,
      }),
    /outside/,
  )
  assert.throws(
    () =>
      mergeTrainingPeaksCalendar(existing, [refreshed.workouts[0], refreshed.workouts[0]], {
        since: '2026-10-01',
        until: '2026-10-06',
        fetchedAt: nextFetch,
      }),
    /duplicate/,
  )
})

test('rejects malformed caches and keeps local athlete identity outside the public model', async () => {
  const calendar = mergeTrainingPeaksCalendar(
    emptyTrainingPeaksCalendar(),
    [normalizeTrainingPeaksCalendarWorkout(workout())],
    { since: '2026-10-01', until: '2026-10-31', fetchedAt: '2026-10-06T12:00:00.000Z' },
  )
  assert.deepEqual(parseTrainingPeaksCalendar(calendar), calendar)
  assert.deepEqual(parseTrainingPeaksCalendar({ ...calendar, athleteId: 1 }), calendar)
  for (const invalid of [
    null,
    { ...calendar, version: 2 },
    { ...calendar, coverage: [] },
    { ...calendar, workouts: [calendar.workouts[0], calendar.workouts[0]] },
    { ...calendar, workouts: [{ ...calendar.workouts[0], actual: { durationSeconds: -1 } }] },
  ])
    assert.equal(parseTrainingPeaksCalendar(invalid), null)
  const directory = await mkdtemp(join(tmpdir(), 'trainingpeaks-calendar-'))
  const path = join(directory, 'calendar.json')
  try {
    assert.equal(await readTrainingPeaksCalendarCache(path), null)
    await writeTrainingPeaksCalendarCache({ athleteId: 1, calendar }, path)
    assert.deepEqual(await readTrainingPeaksCalendarCache(path), { athleteId: 1, calendar })
    assert.equal((await stat(path)).mode & 0o777, 0o600)
    assert.equal(JSON.parse(await readFile(path, 'utf8')).athleteId, 1)
    await writeFile(path, '{"truncated":')
    await assert.rejects(readTrainingPeaksCalendarCache(path), /TrainingPeaks calendar cache/)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
