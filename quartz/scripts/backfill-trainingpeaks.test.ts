import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import type { TrainingPeaksBackfillPlan } from '../plugins/stores/activity-bridge'
import type { RawStravaActivity, StravaStreams } from '../plugins/stores/strava'
import {
  parseTrainingPeaksBackfillArgs,
  selectTrainingPeaksBackfillPlans,
  stravaTrainingPeaksFile,
  trainingPeaksBackfillFilename,
  trainingPeaksBackfillUsage,
  TRAININGPEAKS_BACKFILL_DIR,
  writeTrainingPeaksFile,
} from './backfill-trainingpeaks'

function plan(
  sourceProvider: 'garmin' | 'strava' | 'wahoo',
  id: string,
  localDate: string,
): TrainingPeaksBackfillPlan {
  if (sourceProvider === 'strava') {
    return {
      sourceProvider,
      title: `Ride ${id}`,
      localDate,
      sport: 'bike',
      source: {
        id,
        name: `Ride ${id}`,
        sportType: 'Ride',
        startDate: `${localDate}T12:00:00.000Z`,
        startDateLocal: `${localDate}T08:00:00`,
        distanceM: 40_000,
        movingTimeS: 3_600,
        elapsedTimeS: 3_700,
      },
    }
  }
  if (sourceProvider === 'wahoo') {
    return {
      sourceProvider,
      title: `Ride ${id}`,
      localDate,
      sport: 'bike',
      source: {
        id: `wahoo:${id}`,
        name: `Ride ${id}`,
        workoutId: Number(id),
        sport: 'bike',
        startDate: `${localDate}T12:00:00.000Z`,
        startDateLocal: `${localDate}T08:00:00`,
        distanceM: 40_000,
        movingTimeS: 3_600,
        elapsedTimeS: 3_700,
        fitUrl: `https://cdn.wahooligan.com/${id}.fit`,
        fitSha256: 'a'.repeat(64),
      },
    }
  }
  return {
    sourceProvider,
    title: `Ride ${id}`,
    localDate,
    sport: 'bike',
    source: {
      id: `connect:${id}`,
      name: `Ride ${id}`,
      sport: 'bike',
      startDate: `${localDate}T12:00:00.000Z`,
      startDateLocal: `${localDate}T08:00:00`,
      distanceM: 40_000,
      movingTimeS: 3_600,
      elapsedTimeS: 3_700,
    },
  }
}

function stravaActivity(): RawStravaActivity {
  return {
    id: 123,
    name: 'Threshold & tempo <ride>',
    sportType: 'Ride',
    distance: 1_000,
    movingTime: 60,
    elapsedTime: 65,
    totalElevationGain: 10,
    startDate: '2026-08-31T12:00:00.000Z',
    startDateLocal: '2026-08-31T08:00:00',
    averageSpeed: 16,
    calories: 100,
  }
}

function stravaStreams(): StravaStreams {
  return {
    time: [0, 60],
    latlng: [
      [43.64, -79.38],
      [43.65, -79.39],
    ],
    altitude: [100, 110],
    distance: [0, 1_000],
    heartrate: [140, 160],
    cadence: [85, 90],
    watts: [200, 250],
  }
}

test('parses source-specific bounded TrainingPeaks backfill arguments', () => {
  assert.deepEqual(parseTrainingPeaksBackfillArgs(['--source', 'garmin']), {
    source: 'garmin',
    write: false,
    openCalendar: false,
    since: null,
    until: null,
    limit: null,
    ids: [],
    outputDir: join(TRAININGPEAKS_BACKFILL_DIR, 'garmin'),
    delayMs: 1000,
  })
  assert.deepEqual(
    parseTrainingPeaksBackfillArgs([
      '--source',
      'strava',
      '--write',
      '--since',
      '2026-08-01',
      '--until',
      '2026-08-31',
      '--limit',
      '4',
      '--id',
      'connect:2',
      '--id',
      '2',
      '--output',
      '/tmp/trainingpeaks',
      '--delay-ms',
      '3',
    ]),
    {
      source: 'strava',
      write: true,
      openCalendar: false,
      since: '2026-08-01',
      until: '2026-08-31',
      limit: 4,
      ids: ['2'],
      outputDir: '/tmp/trainingpeaks',
      delayMs: 3,
    },
  )
  assert.throws(() => parseTrainingPeaksBackfillArgs([]), /--source/)
  assert.throws(
    () =>
      parseTrainingPeaksBackfillArgs([
        '--source',
        'strava',
        '--since',
        '2026-09-01',
        '--until',
        '2026-08-01',
      ]),
    /on or before/,
  )
  assert.throws(
    () => parseTrainingPeaksBackfillArgs(['--source', 'strava', '--id', 'garmin:2']),
    /positive activity id/,
  )
  assert.throws(
    () => parseTrainingPeaksBackfillArgs(['--source', 'strava', '--output', ' ']),
    /nonempty/,
  )
  assert.throws(() => parseTrainingPeaksBackfillArgs(['--source', 'strava', '--upload']), /unknown/)
  assert.throws(
    () => parseTrainingPeaksBackfillArgs(['--source', 'wahoo', '--open-calendar']),
    /requires --write/,
  )
  assert.deepEqual(
    parseTrainingPeaksBackfillArgs(['--source', 'wahoo', '--write', '--open-calendar']),
    {
      source: 'wahoo',
      write: true,
      openCalendar: true,
      since: null,
      until: null,
      limit: null,
      ids: [],
      outputDir: join(TRAININGPEAKS_BACKFILL_DIR, 'wahoo'),
      delayMs: 1000,
    },
  )
  assert.match(trainingPeaksBackfillUsage(), /drag-and-drop upload/)
})

test('selects plans by source, local day, activity id, and limit', () => {
  const args = parseTrainingPeaksBackfillArgs([
    '--source',
    'strava',
    '--since',
    '2026-08-02',
    '--until',
    '2026-08-03',
    '--id',
    '2',
    '--id',
    '3',
    '--limit',
    '1',
  ])
  assert.deepEqual(
    selectTrainingPeaksBackfillPlans(
      [
        plan('garmin', '2', '2026-08-02'),
        plan('strava', '1', '2026-08-01'),
        plan('strava', '2', '2026-08-02'),
        plan('strava', '3', '2026-08-03'),
        plan('wahoo', '2', '2026-08-02'),
      ],
      args,
    ).map(candidate => candidate.source.id),
    ['2'],
  )
})

test('creates stable source-provenance filenames', () => {
  assert.equal(
    trainingPeaksBackfillFilename(plan('strava', '123', '2026-08-31')),
    '2026-08-31-strava-123.tcx',
  )
  assert.equal(
    trainingPeaksBackfillFilename(plan('garmin', '124', '2026-08-31'), 'fit'),
    '2026-08-31-garmin-124.fit',
  )
  assert.equal(
    trainingPeaksBackfillFilename(plan('wahoo', '125', '2026-08-31'), 'fit'),
    '2026-08-31-wahoo-125.fit',
  )
})

test('generates a source-labelled Strava TCX with ride telemetry', () => {
  const file = stravaTrainingPeaksFile(stravaActivity(), stravaStreams(), 'bike')
  const tcx = new TextDecoder().decode(file.bytes)

  assert.equal(file.kind, 'tcx')
  assert.equal(file.provenance, 'strava-generated-tcx')
  assert.match(file.sha256, /^[a-f0-9]{64}$/)
  assert.match(tcx, /<Activity Sport="Biking">/)
  assert.match(tcx, /<Cadence>90<\/Cadence>/)
  assert.match(tcx, /<ns3:Watts>250<\/ns3:Watts>/)
  assert.match(tcx, /Strava 123: Threshold &amp; tempo &lt;ride&gt;/)
  assert.equal(tcx.match(/<Trackpoint>/g)?.length, 2)
})

test('preserves standard and extension cadence in generated run TCX', () => {
  const file = stravaTrainingPeaksFile(stravaActivity(), stravaStreams(), 'run')
  const tcx = new TextDecoder().decode(file.bytes)

  assert.match(tcx, /<Activity Sport="Running">/)
  assert.match(tcx, /<Cadence>90<\/Cadence>/)
  assert.match(tcx, /<ns3:RunCadence>90<\/ns3:RunCadence>/)
})

test('refuses to generate Strava TCX without a timed stream', () => {
  assert.throws(
    () => stravaTrainingPeaksFile(stravaActivity(), { ...stravaStreams(), time: [0] }, 'bike'),
    /no timed stream/,
  )
})

test('writes private workout files idempotently and refuses different bytes', async t => {
  const root = await fs.mkdtemp(join(os.tmpdir(), 'trainingpeaks-backfill-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const path = join(root, 'nested', 'activity.fit')
  const bytes = Uint8Array.from([1, 2, 3, 4])

  assert.equal(await writeTrainingPeaksFile(path, bytes), 'created')
  assert.equal(await writeTrainingPeaksFile(path, bytes), 'existing')
  assert.deepEqual(await fs.readFile(path), Buffer.from(bytes))
  assert.equal((await fs.stat(path)).mode & 0o777, 0o600)
  await assert.rejects(
    writeTrainingPeaksFile(path, Uint8Array.from([4, 3, 2, 1])),
    /refusing to replace/,
  )
})
