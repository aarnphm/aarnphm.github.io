import assert from 'node:assert/strict'
import test from 'node:test'
import type { AppleSwim } from '../plugins/stores/apple'
import type { RawStravaActivity } from '../plugins/stores/strava'
import {
  applePoolSwimProjection,
  encodeGarminSwimBackfill,
  garminBackfillFilename,
  garminSwimFitInput,
  shouldDeferPoolSwimForApple,
  type TimedStravaStreams,
} from './backfill-garmin-runs'

function stravaActivity(): RawStravaActivity {
  return {
    id: 19370081352,
    name: 'Tapering swim',
    sportType: 'Swim',
    distance: 400,
    movingTime: 543,
    elapsedTime: 978,
    totalElevationGain: 0,
    startDate: '2026-07-19T01:02:50Z',
    startDateLocal: '2026-07-18T21:02:50Z',
    averageSpeed: 0.737,
    calories: 104,
  }
}

function timedStreams(): TimedStravaStreams {
  return {
    time: [0, 30, 60],
    latlng: [],
    altitude: [],
    distance: [0, 25, 50],
    heartrate: [110, 120, 130],
    cadence: [],
    watts: [],
    temp: [],
  }
}

function appleSwim(values: Partial<AppleSwim> = {}): AppleSwim {
  return {
    id: 'apple-swim',
    date: '2026-07-18',
    start: '2026-07-19T01:02:50Z',
    end: '2026-07-19T01:03:50Z',
    totalM: 50,
    laps: 2,
    activeTimeS: 45,
    strokeCount: 9,
    strokeTimeS: 20,
    strokes: { freestyle: 25, kickboard: 25 },
    intervals: [
      {
        start: '2026-07-19T01:03:00Z',
        end: '2026-07-19T01:03:20Z',
        distanceM: 25,
        startElapsedS: 10,
        endElapsedS: 30,
        durationS: 20,
        strokeCount: 9,
        strokeTimeS: 20,
        stroke: 'freestyle',
      },
      {
        start: '2026-07-19T01:03:30Z',
        end: '2026-07-19T01:03:50Z',
        distanceM: 25,
        startElapsedS: 40,
        endElapsedS: 60,
        durationS: 20,
        strokeCount: null,
        strokeTimeS: null,
        stroke: 'kickboard',
      },
    ],
    ...values,
  }
}

test('projects future swim backfills into pool or open-water FIT inputs', () => {
  const pool = garminSwimFitInput(stravaActivity(), timedStreams())
  assert.equal(pool.kind, 'pool')
  assert.equal(pool.poolLengthMeters, 25)
  assert.equal(pool.elapsedTimeSeconds, 978)
  assert.equal(pool.timerTimeSeconds, 543)
  assert.equal(pool.samples.length, 3)
  assert.equal(pool.samples[1]?.heartRateBpm, 120)

  const gpsStreams = timedStreams()
  gpsStreams.latlng = [
    [43.1, -79.1],
    [43.2, -79.2],
    [43.3, -79.3],
  ]
  const openWater = garminSwimFitInput(stravaActivity(), gpsStreams)
  assert.equal(openWater.kind, 'openWater')
  assert.equal(openWater.samples[2]?.latitudeDegrees, 43.3)
})

test('encodes every future pool swim as a validated FIT activity', () => {
  const encoded = encodeGarminSwimBackfill(stravaActivity(), timedStreams())
  assert.equal(encoded.validation.valid, true)
  assert.equal(encoded.validation.integrity, true)
  assert.equal(encoded.validation.counts.fileIds, 1)
  assert.equal(encoded.validation.counts.lengths, 16)
  assert.equal(encoded.validation.counts.laps, 1)
  assert.equal(encoded.validation.counts.sessions, 1)
  assert.equal(encoded.validation.counts.activities, 1)
})

test('uses complete Apple intervals as the pool distance and stroke authority', () => {
  const activity = stravaActivity()
  const swim = appleSwim()
  const streams = timedStreams()
  streams.time = [0, 20, 35, 50, 60]
  streams.distance = [0, 10, 20, 40, 50]
  streams.heartrate = [110, 115, 120, 125, 130]
  const projection = applePoolSwimProjection(activity, swim)
  const input = garminSwimFitInput(activity, streams, swim)

  assert.ok(projection)
  assert.equal(projection.distanceMeters, 50)
  assert.equal(projection.poolLengthMeters, 25)
  assert.equal(projection.lengths.length, 2)
  assert.equal(input.kind, 'pool')
  assert.equal(input.distanceMeters, 50)
  assert.equal(input.timerTimeSeconds, 40)
  assert.deepEqual(
    input.samples.map(sample => sample.distanceMeters),
    [0, 12.5, 25, 37.5, 50],
  )
  if (input.kind !== 'pool') assert.fail('expected pool swim input')
  assert.deepEqual(
    input.lengths?.map(length => ({ strokes: length.totalStrokes, stroke: length.swimStroke })),
    [
      { strokes: 9, stroke: 'freestyle' },
      { strokes: undefined, stroke: 'drill' },
    ],
  )

  const encoded = encodeGarminSwimBackfill(activity, streams, swim)
  assert.equal(encoded.validation.valid, true)
  assert.equal(encoded.validation.counts.lengths, 2)
})

test('defers a recent pool upload until complete Apple lengths arrive', () => {
  const activity = stravaActivity()
  const noApple = garminSwimFitInput(activity, timedStreams())
  const completeApple = garminSwimFitInput(activity, timedStreams(), appleSwim())
  const now = Date.parse(activity.startDate) + activity.elapsedTime * 1000 + 24 * 60 * 60 * 1000

  assert.equal(shouldDeferPoolSwimForApple(activity, noApple, now), true)
  assert.equal(shouldDeferPoolSwimForApple(activity, completeApple, now), false)
  assert.equal(applePoolSwimProjection(activity, appleSwim({ totalM: 75 })), null)
  assert.equal(applePoolSwimProjection(activity, appleSwim({ strokeCount: 40 })), null)
  assert.equal(
    applePoolSwimProjection(
      activity,
      appleSwim({
        intervals: appleSwim().intervals?.map((interval, index) =>
          index === 0 ? { ...interval, strokeCount: null, strokeTimeS: null } : interval,
        ),
      }),
    ),
    null,
  )
})

test('uses FIT filenames for swims while preserving TCX for runs', () => {
  const activity = stravaActivity()
  assert.equal(garminBackfillFilename(activity, 'swim'), '2026-07-19-19370081352.fit')
  assert.equal(garminBackfillFilename(activity, 'run'), '2026-07-19-19370081352.tcx')
})
