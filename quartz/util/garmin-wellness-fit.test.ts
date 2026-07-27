import { Decoder, Stream, Utils, type FitMessages } from '@garmin/fitsdk'
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  collapseSleepStages,
  encodeGarminWellnessFit,
  validateGarminWellnessFit,
  type GarminSleepLevel,
  type GarminWellnessFitInput,
} from './garmin-wellness-fit'

const SLEEP_START = new Date('2026-07-24T03:30:00.000Z')
const FIVE_MINUTES = 5 * 60_000

function decode(bytes: Uint8Array): FitMessages {
  const result = new Decoder(Stream.fromByteArray(bytes)).read()
  assert.deepEqual(result.errors, [])
  return result.messages
}

function stagesFrom(levels: readonly GarminSleepLevel[]) {
  return levels.map((level, index) => ({
    startTime: new Date(SLEEP_START.valueOf() + index * FIVE_MINUTES),
    level,
  }))
}

function input(overrides: Partial<GarminWellnessFitInput> = {}): GarminWellnessFitInput {
  const stages = stagesFrom(['awake', 'light', 'light', 'deep', 'rem', 'light'])
  return {
    serialNumber: 3_141_592_653,
    productName: 'Oura Sleep Bridge',
    sleepStart: SLEEP_START,
    sleepEnd: new Date(SLEEP_START.valueOf() + 6 * FIVE_MINUTES),
    localOffsetMinutes: -240,
    stages,
    heartRate: [
      { time: SLEEP_START, heartRateBpm: 56 },
      { time: new Date(SLEEP_START.valueOf() + FIVE_MINUTES), heartRateBpm: 54.4 },
    ],
    restingHeartRate: 47,
    ...overrides,
  }
}

test('collapseSleepStages drops repeated levels and keeps the first timestamp', () => {
  const collapsed = collapseSleepStages(stagesFrom(['light', 'light', 'deep', 'deep', 'light']))
  assert.deepEqual(
    collapsed.map(stage => stage.level),
    ['light', 'deep', 'light'],
  )
  assert.equal(collapsed[1].startTime.valueOf(), SLEEP_START.valueOf() + 2 * FIVE_MINUTES)
})

test('encodeGarminWellnessFit writes a monitoring file Garmin can decode', () => {
  const encoded = encodeGarminWellnessFit(input())
  assert.equal(encoded.validation.valid, true)
  assert.deepEqual(encoded.validation.errors, [])
  const messages = decode(encoded.bytes)
  assert.equal(messages.fileIdMesgs?.length, 1)
  assert.equal(messages.fileIdMesgs?.[0].type, 'monitoringB')
  assert.equal(messages.fileIdMesgs?.[0].serialNumber, 3_141_592_653)
  assert.equal(messages.monitoringInfoMesgs?.length, 1)
})

test('sleep levels collapse to transitions and close with a wake marker', () => {
  const encoded = encodeGarminWellnessFit(input())
  const messages = decode(encoded.bytes)
  assert.deepEqual(
    messages.sleepLevelMesgs?.map(mesg => mesg.sleepLevel),
    ['awake', 'light', 'deep', 'rem', 'light', 'awake'],
  )
  const last = messages.sleepLevelMesgs?.[messages.sleepLevelMesgs.length - 1]
  assert.equal(last?.timestamp?.valueOf(), SLEEP_START.valueOf() + 6 * FIVE_MINUTES)
})

test('a sleep window ending on its final stage gets no extra wake marker', () => {
  const encoded = encodeGarminWellnessFit(
    input({ sleepEnd: new Date(SLEEP_START.valueOf() + 5 * FIVE_MINUTES) }),
  )
  const messages = decode(encoded.bytes)
  assert.deepEqual(
    messages.sleepLevelMesgs?.map(mesg => mesg.sleepLevel),
    ['awake', 'light', 'deep', 'rem', 'light'],
  )
})

test('heart rate rides monitoring messages with a local timestamp and rounded bpm', () => {
  const encoded = encodeGarminWellnessFit(input())
  const messages = decode(encoded.bytes)
  assert.deepEqual(
    messages.monitoringMesgs?.map(mesg => mesg.heartRate),
    [56, 54],
  )
  const first = messages.monitoringMesgs?.[0]
  assert.equal(
    first?.localTimestamp,
    Utils.convertDateToDateTime(new Date(SLEEP_START.valueOf() - 240 * 60_000)),
  )
  assert.equal(messages.monitoringHrDataMesgs?.length, 1)
  assert.equal(messages.monitoringHrDataMesgs?.[0].restingHeartRate, 47)
  assert.equal(messages.monitoringHrDataMesgs?.[0].currentDayRestingHeartRate, 47)
})

test('heart rate is optional', () => {
  const encoded = encodeGarminWellnessFit(
    input({ heartRate: undefined, restingHeartRate: undefined }),
  )
  assert.equal(encoded.validation.valid, true)
  assert.equal(encoded.validation.counts.monitorings, 0)
  assert.equal(encoded.validation.counts.monitoringHrData, 0)
})

test('validateGarminWellnessFit rejects bytes that are not a monitoring file', () => {
  const validation = validateGarminWellnessFit(new Uint8Array([1, 2, 3, 4]))
  assert.equal(validation.valid, false)
  assert.equal(validation.isFit, false)
  assert.deepEqual(validation.counts.sleepLevels, 0)
})

test('input guards reject malformed nights', () => {
  assert.throws(() => encodeGarminWellnessFit(input({ stages: [] })), /stages must not be empty/)
  assert.throws(
    () => encodeGarminWellnessFit(input({ sleepEnd: SLEEP_START })),
    /sleepEnd must be after sleepStart/,
  )
  assert.throws(
    () =>
      encodeGarminWellnessFit(
        input({
          stages: [
            { startTime: SLEEP_START, level: 'light' },
            { startTime: SLEEP_START, level: 'deep' },
          ],
        }),
      ),
    /strictly increasing/,
  )
  assert.throws(
    () =>
      encodeGarminWellnessFit(
        input({
          stages: [{ startTime: new Date(SLEEP_START.valueOf() - FIVE_MINUTES), level: 'light' }],
        }),
      ),
    /inside the sleep window/,
  )
  assert.throws(
    () => encodeGarminWellnessFit(input({ heartRate: [{ time: SLEEP_START, heartRateBpm: 4 }] })),
    /out of range/,
  )
  assert.throws(
    () => encodeGarminWellnessFit(input({ localOffsetMinutes: 24 * 60 })),
    /whole-minute UTC offset/,
  )
})
