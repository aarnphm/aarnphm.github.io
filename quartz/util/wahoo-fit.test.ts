import {
  Encoder,
  Profile,
  type DeviceInfoMesg,
  type FileIdMesg,
  type RecordMesg,
  type SessionMesg,
} from '@garmin/fitsdk'
import assert from 'node:assert/strict'
import test from 'node:test'
import { decodeWahooFit, wahooFitSha256 } from './wahoo-fit'

const START = new Date('2026-08-27T12:00:00.000Z')
const SEMICIRCLES_PER_DEGREE = 2 ** 31 / 180

function activityFit(): Uint8Array {
  const encoder = new Encoder()
  const file: FileIdMesg = {
    type: 'activity',
    manufacturer: 'wahooFitness',
    product: 1,
    productName: 'ELEMNT BOLT',
    serialNumber: 42,
    timeCreated: START,
  }
  encoder.onMesg(Profile.MesgNum.FILE_ID, file)
  const device: DeviceInfoMesg = {
    timestamp: START,
    deviceIndex: 'creator',
    manufacturer: 'wahooFitness',
    product: 1,
    productName: 'ELEMNT BOLT',
  }
  encoder.onMesg(Profile.MesgNum.DEVICE_INFO, device)
  const records: RecordMesg[] = [
    {
      timestamp: START,
      positionLat: 43.65 * SEMICIRCLES_PER_DEGREE,
      positionLong: -79.38 * SEMICIRCLES_PER_DEGREE,
      altitude: 100,
      distance: 0,
      power: 200,
      heartRate: 130,
      cadence: 88,
      speed: 8,
      leftRightBalance: 52 | 0x80,
    },
    {
      timestamp: new Date(START.getTime() + 60_000),
      positionLat: 43.66 * SEMICIRCLES_PER_DEGREE,
      positionLong: -79.37 * SEMICIRCLES_PER_DEGREE,
      altitude: 110,
      distance: 500,
      power: 240,
      heartRate: 145,
      cadence: 92,
      speed: 9,
      leftRightBalance: 53 | 0x80,
    },
  ]
  for (const record of records) encoder.onMesg(Profile.MesgNum.RECORD, record)
  const session: SessionMesg = {
    timestamp: new Date(START.getTime() + 60_000),
    startTime: START,
    sport: 'cycling',
    totalElapsedTime: 60,
    totalTimerTime: 58,
    totalMovingTime: 57,
    totalDistance: 500,
    totalCalories: 42,
    avgHeartRate: 138,
    maxHeartRate: 145,
    avgPower: 220,
    normalizedPower: 230,
    maxPower: 240,
    avgCadence: 90,
    totalAscent: 10,
    totalDescent: 2,
    totalWork: 13_200,
    trainingStressScore: 5,
    intensityFactor: 0.7,
  }
  encoder.onMesg(Profile.MesgNum.SESSION, session)
  return encoder.close()
}

test('decodes Wahoo FIT summary, device, aligned streams, and balance', () => {
  const bytes = activityFit()
  const fit = decodeWahooFit(bytes)

  assert.equal(fit.startDate, START.toISOString())
  assert.equal(fit.sport, 'cycling')
  assert.equal(fit.sourceDevice, 'ELEMNT BOLT')
  assert.equal(fit.distanceM, 500)
  assert.equal(fit.movingTimeS, 57)
  assert.equal(fit.metrics.normalizedPower, 230)
  assert.deepEqual(fit.streams.time, [0, 60])
  assert.deepEqual(fit.streams.rightBalance, [52, 53])
  assert.ok(Math.abs((fit.streams.latlng[0]?.[0] ?? 0) - 43.65) < 0.0001)
  assert.match(wahooFitSha256(bytes), /^[a-f0-9]{64}$/)
})

test('rejects non-FIT bytes', () => {
  assert.throws(() => decodeWahooFit(Uint8Array.from([1, 2, 3])), /not FIT/)
})
