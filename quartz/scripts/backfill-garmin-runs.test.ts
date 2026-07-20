import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity } from '../plugins/stores/strava'
import {
  encodeGarminSwimBackfill,
  garminBackfillFilename,
  garminSwimFitInput,
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

test('uses FIT filenames for swims while preserving TCX for runs', () => {
  const activity = stravaActivity()
  assert.equal(garminBackfillFilename(activity, 'swim'), '2026-07-19-19370081352.fit')
  assert.equal(garminBackfillFilename(activity, 'run'), '2026-07-19-19370081352.tcx')
})
