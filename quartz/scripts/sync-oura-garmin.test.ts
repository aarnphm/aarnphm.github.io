import assert from 'node:assert/strict'
import test from 'node:test'
import { isoOffsetMinutes, seriesSamples, sleepStages } from './sync-oura-garmin'

const BEDTIME = '2026-07-24T03:26:00.000-04:00'
const SLEEP_START = new Date(BEDTIME)

test('isoOffsetMinutes reads the offset Oura stamps on bedtimes', () => {
  assert.equal(isoOffsetMinutes(BEDTIME), -240)
  assert.equal(isoOffsetMinutes('2026-01-12T22:15:00.000+05:30'), 330)
  assert.equal(isoOffsetMinutes('2026-01-12T22:15:00.000Z'), 0)
  assert.throws(() => isoOffsetMinutes('2026-01-12T22:15:00.000'), /no UTC offset/)
})

test('sleepStages walks the 5-minute hypnogram in Oura phase order', () => {
  const stages = sleepStages('4221 3', SLEEP_START)
  assert.deepEqual(
    stages.map(stage => stage.level),
    ['awake', 'light', 'light', 'deep', 'rem'],
  )
  assert.deepEqual(
    stages.map(stage => (stage.startTime.valueOf() - SLEEP_START.valueOf()) / 60_000),
    [0, 5, 10, 15, 25],
  )
})

test('seriesSamples drops nulls and anything outside the sleep window', () => {
  const sleepEnd = new Date(SLEEP_START.valueOf() + 15 * 60_000)
  const samples = seriesSamples(
    { startTs: BEDTIME, intervalS: 300, items: [null, 56, 54, 53, 88] },
    SLEEP_START,
    sleepEnd,
  )
  assert.deepEqual(
    samples.map(sample => sample.heartRateBpm),
    [56, 54, 53],
  )
  assert.equal(samples[0].time.valueOf(), SLEEP_START.valueOf() + 5 * 60_000)
})

test('seriesSamples rejects an unparseable series start', () => {
  assert.throws(
    () =>
      seriesSamples(
        { startTs: 'not-a-time', intervalS: 300, items: [50] },
        SLEEP_START,
        SLEEP_START,
      ),
    /not a timestamp/,
  )
})
