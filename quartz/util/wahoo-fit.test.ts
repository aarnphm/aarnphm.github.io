import {
  Encoder,
  Profile,
  type DeviceInfoMesg,
  type DeveloperDataIdMesg,
  type EventMesg,
  type FieldDescriptionMesg,
  type FileIdMesg,
  type RecordMesg,
  type SegmentLapMesg,
  type SessionMesg,
} from '@garmin/fitsdk'
import assert from 'node:assert/strict'
import test from 'node:test'
import { decodeWahooFit, wahooFitSha256 } from './wahoo-fit'

const START = new Date('2026-08-27T12:00:00.000Z')
const SEMICIRCLES_PER_DEGREE = 2 ** 31 / 180

function activityFit(): Uint8Array {
  const encoder = new Encoder()
  const developer: DeveloperDataIdMesg = {
    developerDataIndex: 0,
    applicationId: Array.from({ length: 16 }, () => 1),
  }
  const fields: [number, FieldDescriptionMesg][] = [
    [
      1,
      {
        developerDataIndex: 0,
        fieldDefinitionNumber: 1,
        fitBaseTypeId: 132,
        fieldName: 'tyme_breath_rate',
        units: 'brpm',
      },
    ],
    [
      2,
      {
        developerDataIndex: 0,
        fieldDefinitionNumber: 2,
        fitBaseTypeId: 132,
        fieldName: 'tyme_minute_volume',
        units: 'vol/min',
      },
    ],
    [
      3,
      {
        developerDataIndex: 0,
        fieldDefinitionNumber: 3,
        fitBaseTypeId: 132,
        fieldName: 'tyme_tidal_volume',
        units: 'vol/br',
      },
    ],
    [
      4,
      {
        developerDataIndex: 0,
        fieldDefinitionNumber: 4,
        fitBaseTypeId: 132,
        fieldName: 'fluid_loss_ml',
        units: 'mL',
      },
    ],
    [
      5,
      {
        developerDataIndex: 0,
        fieldDefinitionNumber: 5,
        fitBaseTypeId: 132,
        fieldName: 'sodium_loss_mg',
        units: 'mg',
      },
    ],
    [
      6,
      {
        developerDataIndex: 0,
        fieldDefinitionNumber: 6,
        fitBaseTypeId: 132,
        fieldName: 'skin_temperature',
        units: 'deg C',
        scale: 100,
      },
    ],
    [
      7,
      {
        developerDataIndex: 0,
        fieldDefinitionNumber: 7,
        fitBaseTypeId: 2,
        fieldName: 'heat_strain_index',
        scale: 10,
      },
    ],
  ]
  encoder.onMesg(Profile.MesgNum.DEVELOPER_DATA_ID, developer)
  for (const [key, field] of fields) {
    encoder.addDeveloperField(key, developer, field)
    encoder.onMesg(Profile.MesgNum.FIELD_DESCRIPTION, field)
  }
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
      power: 0,
      heartRate: 130,
      cadence: 0,
      speed: 8,
      leftRightBalance: 52 | 0x80,
      leftPedalSmoothness: 21.5,
      rightPedalSmoothness: 22.5,
      leftTorqueEffectiveness: 75,
      rightTorqueEffectiveness: 77,
      leftPowerPhase: [350, 190],
      rightPowerPhase: [348, 198],
      coreTemperature: 37.16,
      saturatedHemoglobinPercent: 62,
      totalHemoglobinConc: 12.1,
      developerFields: { 1: 24, 2: 40, 3: 1200, 4: 0, 5: 0, 6: 3340, 7: 0 },
    },
    {
      timestamp: new Date(START.getTime() + 60_000),
      positionLat: 43.66 * SEMICIRCLES_PER_DEGREE,
      positionLong: -79.37 * SEMICIRCLES_PER_DEGREE,
      altitude: 110,
      distance: 500,
      power: 0,
      heartRate: 145,
      cadence: 0,
      speed: 9,
      leftRightBalance: 53 | 0x80,
      leftPedalSmoothness: 22,
      rightPedalSmoothness: 23,
      leftTorqueEffectiveness: 76,
      rightTorqueEffectiveness: 78,
      leftPowerPhase: [352, 192],
      rightPowerPhase: [350, 200],
      coreTemperature: 37.19,
      saturatedHemoglobinPercent: 58,
      totalHemoglobinConc: 12.3,
      developerFields: { 1: 26, 2: 60, 3: 1400, 4: 900, 5: 740, 6: 3350, 7: 30 },
    },
  ]
  for (const record of records) encoder.onMesg(Profile.MesgNum.RECORD, record)
  const segments: SegmentLapMesg[] = [
    {
      startTime: START,
      timestamp: new Date(START.getTime() + 60_000),
      uuid: 'WAHOO_ON_ROUTE_CLIMB-snake-road',
      name: 'Snake Road',
      totalTimerTime: 58,
      totalDistance: 480,
      totalAscent: 12,
      avgGrade: 2.5,
      avgSpeed: 8.3,
      avgHeartRate: 140,
      avgPower: 222,
      avgCadence: 91,
    },
    {
      startTime: START,
      timestamp: new Date(START.getTime() + 30_000),
      uuid: 'WAHOO_OFF_ROUTE_CLIMB-1',
      name: '1',
      totalTimerTime: 30,
      totalDistance: 200,
    },
    {
      startTime: START,
      timestamp: new Date(START.getTime() + 60_000),
      uuid: 'WAHOO_OFF_ROUTE_CLIMB-1',
      name: '1',
    },
    {
      startTime: START,
      timestamp: new Date(START.getTime() + 60_000),
      uuid: 'STRAVA_SEGMENT-1',
      name: 'Unrelated',
      totalTimerTime: 60,
      totalDistance: 500,
    },
    {
      startTime: START,
      timestamp: START,
      uuid: 'WAHOO_ON_ROUTE_CLIMB-zero-duration',
      name: 'Zero duration',
      totalTimerTime: 0,
      totalDistance: 100,
    },
    {
      startTime: new Date(START.getTime() + 60_000),
      timestamp: START,
      uuid: 'WAHOO_OFF_ROUTE_CLIMB-reversed',
      name: 'Reversed',
      totalTimerTime: 60,
      totalDistance: 100,
    },
  ]
  for (const segment of segments) encoder.onMesg(Profile.MesgNum.SEGMENT_LAP, segment)
  const gear: EventMesg = {
    timestamp: new Date(START.getTime() + 30_000),
    event: 'rearGearChange',
    eventType: 'marker',
    frontGearNum: 2,
    frontGear: 54,
    rearGearNum: 5,
    rearGear: 21,
  }
  encoder.onMesg(Profile.MesgNum.EVENT, gear)
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
  assert.deepEqual(fit.streams.respiration, [24, 26])
  assert.deepEqual(fit.streams.muscleOxygenPercent, [62, 58])
  assert.deepEqual(fit.streams.totalHemoglobinConcentration, [12.1, 12.3])
  assert.deepEqual(fit.streams.coreTemperatureC, [37.16, 37.19])
  assert.deepEqual(fit.streams.skinTemperatureC, [33.4, 33.5])
  assert.deepEqual(fit.streams.heatStrainIndex, [0, 3])
  assert.deepEqual(fit.streams.minuteVentilation, [40, 60])
  assert.deepEqual(fit.streams.tidalVolume, [1200, 1400])
  assert.deepEqual(fit.sweatLoss, { fluidMl: 900, sodiumMg: 740 })
  assert.deepEqual(fit.cyclingDynamics.leftPedalSmoothness, [21.5, 22])
  assert.deepEqual(fit.cyclingDynamics.rightTorqueEffectiveness, [77, 78])
  assert.deepEqual(fit.cyclingDynamics.leftPowerPhaseStart, [350.2, 351.6])
  assert.deepEqual(fit.gearShifts, [
    {
      timestamp: new Date(START.getTime() + 30_000).toISOString(),
      frontGearNum: 2,
      frontTeeth: 54,
      rearGearNum: 5,
      rearTeeth: 21,
    },
  ])
  assert.ok(Math.abs((fit.streams.latlng[0]?.[0] ?? 0) - 43.65) < 0.0001)
  assert.match(wahooFitSha256(bytes), /^[a-f0-9]{64}$/)
})

test('rejects non-FIT bytes', () => {
  assert.throws(() => decodeWahooFit(Uint8Array.from([1, 2, 3])), /not FIT/)
})

test('recognizes Wahoo Summit Segment and Freeride prefixes only', () => {
  const segments = decodeWahooFit(activityFit()).summitSegments

  assert.deepEqual(
    segments.map(segment => [segment.feature, segment.uuid]),
    [
      ['summit-freeride', 'WAHOO_OFF_ROUTE_CLIMB-1'],
      ['summit-segment', 'WAHOO_ON_ROUTE_CLIMB-snake-road'],
    ],
  )
  assert.ok(segments.every(segment => segment.name !== 'Unrelated'))
})

test('keeps the latest and longest duplicate Wahoo Summit result', () => {
  const segment = decodeWahooFit(activityFit()).summitSegments.find(
    value => value.feature === 'summit-freeride',
  )

  assert.ok(segment)
  assert.equal(segment.endDate, new Date(START.getTime() + 60_000).toISOString())
  assert.equal(segment.durationS, 60)
  assert.equal(segment.distanceM, 500)
})

test('prefers Summit lap fields and derives absent metrics from enclosed records', () => {
  const segments = decodeWahooFit(activityFit()).summitSegments
  const routed = segments.find(segment => segment.feature === 'summit-segment')
  const freeride = segments.find(segment => segment.feature === 'summit-freeride')

  assert.ok(routed)
  assert.deepEqual(
    {
      distanceM: routed.distanceM,
      durationS: routed.durationS,
      elevationGainM: routed.elevationGainM,
      avgGradePct: routed.avgGradePct,
      avgSpeedMps: routed.avgSpeedMps,
      avgHeartRate: routed.avgHeartRate,
      avgPower: routed.avgPower,
      avgCadence: routed.avgCadence,
    },
    {
      distanceM: 480,
      durationS: 58,
      elevationGainM: 12,
      avgGradePct: 2.5,
      avgSpeedMps: 8.3,
      avgHeartRate: 140,
      avgPower: 222,
      avgCadence: 91,
    },
  )
  assert.ok(freeride)
  assert.equal(freeride.elevationGainM, 10)
  assert.equal(freeride.avgGradePct, 2)
  assert.ok(Math.abs((freeride.avgSpeedMps ?? 0) - 500 / 60) < 0.0001)
  assert.equal(freeride.avgHeartRate, 137.5)
  assert.equal(freeride.avgPower, 0)
  assert.equal(freeride.avgCadence, 0)
})
