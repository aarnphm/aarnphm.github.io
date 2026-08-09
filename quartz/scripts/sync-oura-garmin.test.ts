import assert from 'node:assert/strict'
import test from 'node:test'
import {
  garminManualSleepPayload,
  manualSleepMatches,
  ouraSleepDurations,
  seriesSamples,
  sleepStages,
} from './sync-oura-garmin'

const BEDTIME = '2026-07-24T03:26:00.000-04:00'
const SLEEP_START = new Date(BEDTIME)

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

test('ouraSleepDurations uses Oura exact totals instead of five-minute stage estimates', () => {
  assert.deepEqual(
    ouraSleepDurations({
      timeInBedS: 24_546,
      totalSleepS: 21_510,
      deepS: 5_340,
      lightS: 11_460,
      remS: 4_710,
      awakeS: 3_036,
    }),
    {
      timeInBedSeconds: 24_546,
      totalSleepSeconds: 21_510,
      deepSeconds: 5_340,
      lightSeconds: 11_460,
      remSeconds: 4_710,
      awakeSeconds: 3_036,
    },
  )
})

test('ouraSleepDurations rejects incomplete duration summaries', () => {
  assert.equal(
    ouraSleepDurations({
      timeInBedS: 24_546,
      totalSleepS: 21_510,
      deepS: 5_340,
      lightS: 11_460,
      remS: null,
      awakeS: 3_036,
    }),
    null,
  )
})

test('garminManualSleepPayload follows the Garmin web manual-sleep contract', () => {
  const sleepEnd = new Date(SLEEP_START.valueOf() + 7 * 60 * 60_000)
  assert.deepEqual(garminManualSleepPayload('2026-07-24', SLEEP_START, sleepEnd, 123), {
    calendarDate: '2026-07-24',
    sleepStartTimestampGMT: SLEEP_START.valueOf(),
    sleepEndTimestampGMT: sleepEnd.valueOf(),
    sleepTimeSeconds: 7 * 60 * 60,
    napTimeSeconds: 0,
    sleepWindowConfirmed: true,
    sleepWindowConfirmationType: 'manually_confirmed',
    userProfilePK: 123,
  })
})

test('manualSleepMatches requires the stored interval and manual source', () => {
  const sleepEnd = new Date(SLEEP_START.valueOf() + 7 * 60 * 60_000)
  const payload = garminManualSleepPayload('2026-07-24', SLEEP_START, sleepEnd, 123)
  assert.equal(manualSleepMatches(payload, payload), true)
  assert.equal(
    manualSleepMatches({ ...payload, sleepWindowConfirmationType: 'MANUALLY_CONFIRMED' }, payload),
    true,
  )
  assert.equal(manualSleepMatches({ ...payload, sleepTimeSeconds: 1 }, payload), false)
  assert.equal(manualSleepMatches({ ...payload, sleepWindowConfirmed: false }, payload), false)
  assert.equal(manualSleepMatches(null, payload), false)
})

test('garminManualSleepPayload rejects invalid intervals and profile ids', () => {
  assert.throws(
    () => garminManualSleepPayload('2026-07-24', SLEEP_START, SLEEP_START, 123),
    /valid increasing interval/,
  )
  assert.throws(
    () =>
      garminManualSleepPayload(
        '2026-07-24',
        SLEEP_START,
        new Date(SLEEP_START.valueOf() + 60_000),
        0,
      ),
    /positive integer/,
  )
})
