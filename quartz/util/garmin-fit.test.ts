import {
  Decoder,
  Encoder,
  Profile,
  Stream,
  type EventMesg,
  type FileIdMesg,
  type FitMessages,
  type LapMesg,
  type RecordMesg,
  type SessionMesg,
} from '@garmin/fitsdk'
import assert from 'node:assert/strict'
import test from 'node:test'
import { deflateRawSync } from 'node:zlib'
import {
  decodeGarminRideFit,
  encodeGarminSwimFit,
  garminFitBytesFromArchive,
  garminRideFitFromArchive,
  validateGarminFit,
  type GarminOpenWaterSwimInput,
  type GarminPoolSwimInput,
} from './garmin-fit'

const START = new Date('2026-07-19T12:00:00.000Z')

function gearChangeData(
  frontGearNum: number,
  frontTeeth: number,
  rearGearNum: number,
  rearTeeth: number,
): number {
  return (rearGearNum | (rearTeeth << 8) | (frontGearNum << 16) | (frontTeeth << 24)) >>> 0
}

function gearFit(): Uint8Array {
  const encoder = new Encoder()
  const fileId: FileIdMesg = {
    type: 'activity',
    manufacturer: 'development',
    product: 1,
    serialNumber: 1,
    timeCreated: START,
  }
  encoder.onMesg(Profile.MesgNum.FILE_ID, fileId)
  const events = [
    { timestamp: START, frontGearNum: 2, frontTeeth: 52, rearGearNum: 3, rearTeeth: 27 },
    {
      timestamp: new Date(START.getTime() + 5_000),
      frontGearNum: 2,
      frontTeeth: 52,
      rearGearNum: 3,
      rearTeeth: 27,
    },
    {
      timestamp: new Date(START.getTime() + 10_000),
      frontGearNum: 1,
      frontTeeth: 36,
      rearGearNum: 5,
      rearTeeth: 21,
    },
  ]
  for (const event of events) {
    const message: EventMesg = {
      timestamp: event.timestamp,
      event: 'rearGearChange',
      eventType: 'marker',
      data: gearChangeData(
        event.frontGearNum,
        event.frontTeeth,
        event.rearGearNum,
        event.rearTeeth,
      ),
    }
    encoder.onMesg(Profile.MesgNum.EVENT, message)
  }
  return encoder.close()
}

function cyclingDynamicsFit(): Uint8Array {
  const encoder = new Encoder()
  const fileId: FileIdMesg = {
    type: 'activity',
    manufacturer: 'development',
    product: 1,
    serialNumber: 1,
    timeCreated: START,
  }
  encoder.onMesg(Profile.MesgNum.FILE_ID, fileId)
  const records: RecordMesg[] = [
    {
      timestamp: START,
      distance: 0,
      power: 200,
      leftPedalSmoothness: 21,
      rightPedalSmoothness: 23,
      leftTorqueEffectiveness: 70,
      rightTorqueEffectiveness: 72,
      leftPowerPhase: [350, 190],
      rightPowerPhase: [348, 198],
    },
    {
      timestamp: new Date(START.getTime() + 10_000),
      distance: 100,
      power: 220,
      leftPedalSmoothness: 22,
      rightPedalSmoothness: 24,
      leftTorqueEffectiveness: 74,
      rightTorqueEffectiveness: 76,
      leftPowerPhase: [352, 192],
      rightPowerPhase: [350, 200],
    },
    {
      timestamp: new Date(START.getTime() + 20_000),
      distance: 200,
      power: 240,
      leftPedalSmoothness: 0,
      rightPedalSmoothness: 25,
      leftTorqueEffectiveness: 0,
      rightTorqueEffectiveness: 78,
      leftPowerPhase: [0, 194],
      rightPowerPhase: [352, 202],
    },
  ]
  for (const record of records) encoder.onMesg(Profile.MesgNum.RECORD, record)
  const positions: EventMesg[] = [
    {
      timestamp: START,
      event: 'riderPositionChange',
      eventType: 'marker',
      data: 0,
      riderPosition: 'seated',
    },
    {
      timestamp: new Date(START.getTime() + 5_000),
      event: 'riderPositionChange',
      eventType: 'marker',
      data: 1,
      riderPosition: 'standing',
    },
    {
      timestamp: new Date(START.getTime() + 15_000),
      event: 'riderPositionChange',
      eventType: 'marker',
      data: 0,
      riderPosition: 'seated',
    },
  ]
  for (const position of positions) encoder.onMesg(Profile.MesgNum.EVENT, position)
  const lap: LapMesg = {
    timestamp: new Date(START.getTime() + 20_000),
    startTime: START,
    totalElapsedTime: 20,
    totalTimerTime: 20,
    timeStanding: 10,
  }
  encoder.onMesg(Profile.MesgNum.LAP, lap)
  const session: SessionMesg = {
    timestamp: new Date(START.getTime() + 20_000),
    startTime: START,
    totalElapsedTime: 20,
    totalTimerTime: 20,
    sport: 'cycling',
    totalTrainingEffect: 3,
    totalAnaerobicTrainingEffect: 1.3,
  }
  encoder.onMesg(Profile.MesgNum.SESSION, session)
  return encoder.close()
}

function zipFit(fit: Uint8Array): Uint8Array {
  const name = Buffer.from('activity.fit')
  const compressed = deflateRawSync(fit)
  const local = Buffer.alloc(30 + name.length)
  local.writeUInt32LE(0x04034b50, 0)
  local.writeUInt16LE(20, 4)
  local.writeUInt16LE(0, 6)
  local.writeUInt16LE(8, 8)
  local.writeUInt32LE(0, 14)
  local.writeUInt32LE(compressed.length, 18)
  local.writeUInt32LE(fit.length, 22)
  local.writeUInt16LE(name.length, 26)
  name.copy(local, 30)
  const central = Buffer.alloc(46 + name.length)
  central.writeUInt32LE(0x02014b50, 0)
  central.writeUInt16LE(20, 4)
  central.writeUInt16LE(20, 6)
  central.writeUInt16LE(0, 8)
  central.writeUInt16LE(8, 10)
  central.writeUInt32LE(0, 16)
  central.writeUInt32LE(compressed.length, 20)
  central.writeUInt32LE(fit.length, 24)
  central.writeUInt16LE(name.length, 28)
  central.writeUInt32LE(0, 42)
  name.copy(central, 46)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(1, 8)
  end.writeUInt16LE(1, 10)
  end.writeUInt32LE(central.length, 12)
  end.writeUInt32LE(local.length + compressed.length, 16)
  return Uint8Array.from(Buffer.concat([local, compressed, central, end]))
}

function decode(bytes: Uint8Array): FitMessages {
  const result = new Decoder(Stream.fromByteArray(bytes)).read()
  assert.deepEqual(result.errors, [])
  return result.messages
}

function poolInput(): GarminPoolSwimInput {
  return {
    kind: 'pool',
    sourceId: 12345,
    name: 'Pool swim',
    startTime: START,
    distanceMeters: 100,
    elapsedTimeSeconds: 120,
    timerTimeSeconds: 120,
    poolLengthMeters: 25,
    calories: 140,
    totalStrokes: 60,
    swimStroke: 'freestyle',
    samples: [
      {
        elapsedSeconds: 0,
        distanceMeters: 0,
        heartRateBpm: 110,
        cadenceRpm: 28,
        temperatureCelsius: 27,
      },
      {
        elapsedSeconds: 30,
        distanceMeters: 25,
        heartRateBpm: 120,
        cadenceRpm: 30,
        temperatureCelsius: 27,
      },
      {
        elapsedSeconds: 90,
        distanceMeters: 75,
        heartRateBpm: 132,
        cadenceRpm: 32,
        temperatureCelsius: 27,
      },
      {
        elapsedSeconds: 120,
        distanceMeters: 100,
        heartRateBpm: 128,
        cadenceRpm: 31,
        temperatureCelsius: 27,
      },
    ],
  }
}

function openWaterInput(): GarminOpenWaterSwimInput {
  return {
    kind: 'openWater',
    sourceId: 'open-water-1',
    name: 'Lake swim',
    startTime: START,
    distanceMeters: 200,
    elapsedTimeSeconds: 120,
    timerTimeSeconds: 120,
    totalStrokes: 96,
    samples: [
      {
        elapsedSeconds: 0,
        distanceMeters: 0,
        latitudeDegrees: 43.6532,
        longitudeDegrees: -79.3832,
        altitudeMeters: 75,
        heartRateBpm: 112,
      },
      {
        elapsedSeconds: 60,
        distanceMeters: 100,
        latitudeDegrees: 43.6538,
        longitudeDegrees: -79.3824,
        altitudeMeters: 75.5,
        heartRateBpm: 126,
      },
      {
        elapsedSeconds: 120,
        distanceMeters: 200,
        latitudeDegrees: 43.6544,
        longitudeDegrees: -79.3816,
        altitudeMeters: 76,
        heartRateBpm: 130,
      },
    ],
  }
}

test('decodes electronic shifting state from Garmin FIT events', () => {
  assert.deepEqual(decodeGarminRideFit(gearFit()).gearShifts, [
    {
      timestamp: '2026-07-19T12:00:00.000Z',
      frontGearNum: 2,
      frontTeeth: 52,
      rearGearNum: 3,
      rearTeeth: 27,
    },
    {
      timestamp: '2026-07-19T12:00:10.000Z',
      frontGearNum: 1,
      frontTeeth: 36,
      rearGearNum: 5,
      rearTeeth: 21,
    },
  ])
})

test('extracts and decodes a deflated FIT member from the Garmin archive', () => {
  const fit = gearFit()
  const archive = zipFit(fit)

  assert.deepEqual(garminFitBytesFromArchive(archive), fit)
  assert.deepEqual(
    garminRideFitFromArchive(archive).gearShifts,
    decodeGarminRideFit(fit).gearShifts,
  )
})

test('decodes cycling dynamics and rider position from Garmin FIT records', () => {
  const ride = decodeGarminRideFit(cyclingDynamicsFit())
  const dynamics = ride.cyclingDynamics

  assert.deepEqual(dynamics.time, [0, 10, 20])
  assert.deepEqual(dynamics.distance, [0, 100, 200])
  assert.deepEqual(dynamics.leftPedalSmoothness, [21, 22, null])
  assert.deepEqual(dynamics.rightPedalSmoothness, [23, 24, 25])
  assert.deepEqual(dynamics.leftTorqueEffectiveness, [70, 74, null])
  assert.deepEqual(dynamics.rightTorqueEffectiveness, [72, 76, 78])
  assert.deepEqual(dynamics.leftPowerPhaseStart, [350.2, 351.6, 0])
  assert.deepEqual(dynamics.leftPowerPhaseEnd, [189.8, 192.7, 194.1])
  assert.deepEqual(dynamics.rightPowerPhaseStart, [347.3, 350.2, 351.6])
  assert.deepEqual(dynamics.rightPowerPhaseEnd, [198.3, 199.7, 202.5])
  assert.deepEqual(dynamics.positionChanges, [
    { elapsedS: 0, distanceM: 0, position: 'seated' },
    { elapsedS: 5, distanceM: 50, position: 'standing' },
    { elapsedS: 15, distanceM: 150, position: 'seated' },
  ])
  assert.equal(dynamics.seatedTimeS, 10)
  assert.equal(dynamics.standingTimeS, 10)
  assert.deepEqual(ride.trainingEffect, { aerobic: 3, anaerobic: 1.3 })
})

test('rejects a Garmin archive without a FIT member', () => {
  assert.throws(() => garminFitBytesFromArchive(new Uint8Array([1, 2, 3, 4])), /ZIP end record/)
})

test('encodes a 25 metre pool swim with interpolated length messages', () => {
  const encoded = encodeGarminSwimFit(poolInput())
  const messages = decode(encoded.bytes)

  assert.equal(encoded.validation.valid, true)
  assert.equal(encoded.validation.isFit, true)
  assert.equal(encoded.validation.integrity, true)
  assert.equal(encoded.validation.counts.fileIds, 1)
  assert.equal(encoded.validation.counts.deviceInfos, 1)
  assert.equal(encoded.validation.counts.events, 2)
  assert.equal(encoded.validation.counts.lengths, 4)
  assert.equal(encoded.validation.counts.laps, 1)
  assert.equal(encoded.validation.counts.sessions, 1)
  assert.equal(encoded.validation.counts.activities, 1)

  const lengths = messages.lengthMesgs ?? []
  assert.equal(lengths.length, 4)
  assert.deepEqual(
    lengths.map(length => {
      assert.ok(length.timestamp instanceof Date)
      return (length.timestamp.getTime() - START.getTime()) / 1000
    }),
    [30, 60, 90, 120],
  )
  assert.deepEqual(
    lengths.map(length => length.lengthType),
    ['active', 'active', 'active', 'active'],
  )
  assert.equal(
    lengths.reduce((sum, length) => sum + (length.totalStrokes ?? 0), 0),
    60,
  )

  const lap = messages.lapMesgs?.[0]
  const session = messages.sessionMesgs?.[0]
  assert.equal(lap?.sport, 'swimming')
  assert.equal(lap?.subSport, 'lapSwimming')
  assert.equal(lap?.totalDistance, 100)
  assert.equal(lap?.totalStrokes, 60)
  assert.equal(lap?.firstLengthIndex, 0)
  assert.equal(lap?.numLengths, 4)
  assert.equal(lap?.numActiveLengths, 4)
  assert.equal(session?.sport, 'swimming')
  assert.equal(session?.subSport, 'lapSwimming')
  assert.equal(session?.totalDistance, 100)
  assert.equal(session?.totalStrokes, 60)
  assert.equal(session?.poolLength, 25)
  assert.equal(session?.poolLengthUnit, 'metric')
  assert.equal(session?.numLengths, 4)
  assert.equal(session?.numActiveLengths, 4)
  assert.equal(session?.firstLapIndex, 0)
  assert.equal(session?.numLaps, 1)
})

test('keeps pool length elapsed time while distributing active timer time', () => {
  const input = poolInput()
  input.timerTimeSeconds = 60
  const messages = decode(encodeGarminSwimFit(input).bytes)
  const lengths = messages.lengthMesgs ?? []

  assert.equal(
    lengths.reduce((sum, length) => sum + (length.totalElapsedTime ?? 0), 0),
    120,
  )
  assert.equal(
    lengths.reduce((sum, length) => sum + (length.totalTimerTime ?? 0), 0),
    60,
  )
  assert.equal(messages.sessionMesgs?.[0]?.totalTimerTime, 60)
})

test('preserves explicit pool length strokes, cadence, style, and timing', () => {
  const input = poolInput()
  input.distanceMeters = 75
  input.elapsedTimeSeconds = 100
  input.timerTimeSeconds = 60
  input.lengths = [
    {
      startElapsedSeconds: 10,
      endElapsedSeconds: 30,
      distanceMeters: 25,
      totalStrokes: 9,
      strokeTimeSeconds: 20,
      swimStroke: 'freestyle',
    },
    {
      startElapsedSeconds: 40,
      endElapsedSeconds: 60,
      distanceMeters: 25,
      totalStrokes: 10,
      strokeTimeSeconds: 20,
      swimStroke: 'breaststroke',
    },
    { startElapsedSeconds: 70, endElapsedSeconds: 100, distanceMeters: 25, swimStroke: 'drill' },
  ]
  input.samples = [
    { elapsedSeconds: 0, distanceMeters: 0, heartRateBpm: 110 },
    { elapsedSeconds: 30, distanceMeters: 25, heartRateBpm: 120 },
    { elapsedSeconds: 60, distanceMeters: 50, heartRateBpm: 130 },
    { elapsedSeconds: 100, distanceMeters: 75, heartRateBpm: 125 },
  ]

  const encoded = encodeGarminSwimFit(input)
  const messages = decode(encoded.bytes)
  const lengths = messages.lengthMesgs ?? []

  assert.equal(encoded.validation.valid, true)
  assert.deepEqual(
    lengths.map(length => length.totalStrokes),
    [9, 10, 0],
  )
  assert.deepEqual(
    lengths.map(length => length.avgSwimmingCadence),
    [27, 30, undefined],
  )
  assert.deepEqual(
    lengths.map(length => length.swimStroke),
    ['freestyle', 'breaststroke', 'drill'],
  )
  assert.deepEqual(
    lengths.map(length => length.totalElapsedTime),
    [20, 20, 30],
  )
  assert.equal(messages.lapMesgs?.[0]?.totalStrokes, 19)
  assert.equal(messages.lapMesgs?.[0]?.avgCadence, 29)
  assert.equal(messages.sessionMesgs?.[0]?.totalDistance, 75)
  assert.equal(messages.sessionMesgs?.[0]?.numLengths, 3)
  assert.equal(messages.sessionMesgs?.[0]?.numActiveLengths, 3)
  assert.equal(messages.sessionMesgs?.[0]?.swimStroke, 'mixed')
  assert.equal(messages.sessionMesgs?.[0]?.avgCadence, 29)
  assert.equal(messages.sessionMesgs?.[0]?.maxCadence, 30)
  assert.equal(messages.sessionMesgs?.[0]?.maxSpeed, 1.25)
})

test('rejects explicit lengths whose distance disagrees with the pool activity', () => {
  const input = poolInput()
  input.lengths = [
    {
      startElapsedSeconds: 0,
      endElapsedSeconds: 30,
      distanceMeters: 25,
      totalStrokes: 10,
      strokeTimeSeconds: 30,
    },
  ]

  assert.throws(() => encodeGarminSwimFit(input), /length distance does not match distanceMeters/)
})

test('encodes open-water GPS coordinates without pool lengths', () => {
  const encoded = encodeGarminSwimFit(openWaterInput())
  const messages = decode(encoded.bytes)
  const records = messages.recordMesgs ?? []
  const session = messages.sessionMesgs?.[0]

  assert.equal(encoded.validation.valid, true)
  assert.equal(encoded.validation.counts.lengths, 0)
  assert.equal(messages.lengthMesgs, undefined)
  assert.equal(records.length, 3)
  assert.ok(records.every(record => record.positionLat != null && record.positionLong != null))
  assert.equal(records[0].positionLat, Math.round((43.6532 * 2 ** 31) / 180))
  assert.equal(records[0].positionLong, Math.round((-79.3832 * 2 ** 31) / 180))
  assert.equal(session?.sport, 'swimming')
  assert.equal(session?.subSport, 'openWater')
  assert.equal(session?.totalDistance, 200)
  assert.equal(session?.totalStrokes, 96)
  assert.equal(session?.numLaps, 1)
  assert.equal(messages.activityMesgs?.length, 1)
})

test('writes a finite computed speed into every record', () => {
  const poolRecords = decode(encodeGarminSwimFit(poolInput()).bytes).recordMesgs ?? []
  const openWaterRecords = decode(encodeGarminSwimFit(openWaterInput()).bytes).recordMesgs ?? []

  assert.ok(poolRecords.length >= 5)
  assert.ok(openWaterRecords.length >= 3)
  for (const record of [...poolRecords, ...openWaterRecords]) {
    assert.equal(typeof record.speed, 'number')
    assert.equal(Number.isFinite(record.speed), true)
    assert.equal(typeof record.enhancedSpeed, 'number')
    assert.equal(Number.isFinite(record.enhancedSpeed), true)
  }
})

test('detects CRC corruption in an otherwise recognizable FIT stream', () => {
  const bytes = encodeGarminSwimFit(poolInput()).bytes.slice()
  bytes[bytes.length - 1] ^= 0xff
  const validation = validateGarminFit(bytes)

  assert.equal(validation.isFit, true)
  assert.equal(validation.integrity, false)
  assert.equal(validation.valid, false)
  assert.ok(validation.errors.includes('FIT integrity check failed'))
})

test('rejects invalid byte streams without throwing', () => {
  const validation = validateGarminFit(new Uint8Array([1, 2, 3, 4]))

  assert.equal(validation.isFit, false)
  assert.equal(validation.integrity, false)
  assert.equal(validation.valid, false)
  assert.deepEqual(validation.counts, {
    fileIds: 0,
    deviceInfos: 0,
    events: 0,
    records: 0,
    lengths: 0,
    laps: 0,
    sessions: 0,
    activities: 0,
  })
  assert.ok(validation.errors.length > 0)
})
