import assert from 'node:assert/strict'
import test from 'node:test'
import type {
  AppleCache,
  AppleSwim,
  AppleSwimInterval,
  AppleWorkout,
} from '../plugins/stores/apple'
import type { CoreBodyTemperatureCache } from '../plugins/stores/core-body-temperature'
import type { OuraCache } from '../plugins/stores/oura'
import { emptyGarminFueling, emptyGarminMetrics, type GarminCache } from '../plugins/stores/garmin'
import {
  emptyPayload,
  type ActivityThermalSource,
  type GarminVerification,
  type StravaActivityDetail,
  type StravaPayload,
} from '../plugins/stores/strava'
import {
  applyManualActivityTracking,
  enrichActivityDevices,
  enrichCalculatedExerciseLoads,
  enrichCalculatedIntensityFactors,
  enrichCalculatedTrainingEffects,
  enrichCoreBodyTemperature,
  enrichRouteLessHeartRate,
  enrichRunPaceZones,
  enrichRunDynamics,
  enrichSwimMetrics,
  swimActivityIntervals,
} from './strava-payload'
import { swimLengthAverages } from './swim-metrics'

const detail = (values: Partial<StravaActivityDetail> = {}): StravaActivityDetail => ({
  id: 1,
  sport: 'swim',
  name: 'Pool swim',
  date: '2026-07-09',
  start: '2026-07-09T20:13:31Z',
  distanceKm: 1,
  movingTimeS: 1_590,
  elapsedTimeS: 1_590,
  maxSpeedKph: null,
  elevationM: 0,
  avgHr: 140,
  maxHr: 160,
  avgWatts: null,
  npWatts: null,
  maxWatts: null,
  kilojoules: null,
  deviceWatts: false,
  avgCadence: null,
  sufferScore: null,
  calories: null,
  deviceTemperatureC: null,
  ambientTemperatureC: null,
  windKph: null,
  windDir: null,
  windDirDeg: null,
  windGustKph: null,
  averageRelativeHumidityPct: null,
  relativeHumidityProvenance: null,
  location: null,
  fueling: null,
  strength: null,
  sauna: null,
  garmin: null,
  computer: null,
  device: null,
  staminaTrace: null,
  performanceConditionTrace: null,
  calculatedIntensityFactor: null,
  calculatedExerciseLoad: null,
  anaerobicPowerEstimate: null,
  calculatedTrainingEffect: null,
  gearShifts: [],
  cyclingDynamics: null,
  runWalk: null,
  route: [],
  heartRateTrace: [],
  mapRoute: [],
  analysisRanges: [],
  runSplitsMetric: [],
  runSplitsStandard: [],
  runPaceZones: null,
  minAlt: 0,
  maxAlt: 0,
  descentM: 0,
  hrZones: null,
  powerZones: null,
  powerHist: null,
  powerWithoutZeros: null,
  powerCurve: null,
  activityCriticalPower: null,
  bestEfforts: null,
  strokes: null,
  strokeCount: null,
  strokeRateSpm: null,
  swimPaceSPer100m: null,
  swimPaceSource: null,
  swimDurationS: null,
  swimIntervals: [],
  swimLocation: null,
  waterTemperatureC: null,
  analyses: {
    native: { myWindsock: null, pelotan: null },
    derived: { environment: null, uvScore: null, apparentWind: null },
  },
  ...values,
})

test('copies the shared analytics pace distribution into run activity details', () => {
  const payload = emptyPayload()
  payload.details = { '1': detail({ id: 1, sport: 'run' }), '2': detail({ id: 2, sport: 'bike' }) }
  enrichRunPaceZones(payload, {
    activities: [
      { id: 1, paceZoneSeconds: [120, 240, 360, 480, 600, 720] },
      { id: 2, paceZoneSeconds: null },
    ],
    paceZoneBoundsSPerKm: [387.114, 333.676, 299.501, 280.238, 263.461],
    tenKmRaceTimeS: 3_000,
  })
  assert.deepEqual(payload.details['1'].runPaceZones, {
    zoneSeconds: [120, 240, 360, 480, 600, 720],
    boundsSPerKm: [387.114, 333.676, 299.501, 280.238, 263.461],
    tenKmRaceTimeS: 3_000,
  })
  assert.equal(payload.details['2'].runPaceZones, null)
})

const appleSwim = (values: Partial<AppleSwim> = {}): AppleSwim => ({
  id: 'apple-swim',
  date: '2026-07-09',
  start: '2026-07-09T20:13:30Z',
  end: '2026-07-09T20:40:10Z',
  activeTimeS: 1_600,
  totalM: 1_000,
  laps: 40,
  strokes: { freestyle: 600, breaststroke: 400 },
  strokeCount: 700,
  strokeTimeS: 1_500,
  intervals: [],
  location: 'pool',
  waterTemperatureC: 27.8,
  ...values,
})

const garminVerification = (activityId: string): GarminVerification => ({
  activityId,
  name: 'Pool swim',
  sourceDevice: 'Forerunner 970',
  startDate: '2026-07-09T20:13:31Z',
  startDiffS: 0,
  distanceM: 50,
  distanceDeltaM: 0,
  distanceDeltaPct: 0,
  movingTimeS: 51,
  movingTimeDeltaS: 0,
  elapsedTimeS: 66,
  elapsedTimeDeltaS: 0,
  totalCalories: null,
  caloriesDelta: null,
  avgHeartRate: null,
  avgHeartRateDelta: null,
  avgPower: null,
  avgPowerDelta: null,
  avgCadence: 24.7,
  normalizedPower: null,
  maxPower: null,
  totalWorkKJ: null,
  totalWorkDeltaKJ: null,
  trainingStressScore: null,
  intensityFactor: null,
  trainingEffectActivityId: activityId,
  aerobicTrainingEffect: 2.9,
  anaerobicTrainingEffect: 1.7,
  exerciseLoad: 95,
  trainingEffectLabel: 'AEROBIC_BASE',
  aerobicTrainingEffectMessage: 'MAINTAINING_AEROBIC_BASE_7',
  anaerobicTrainingEffectMessage: 'MINOR_ANAEROBIC_BENEFIT_15',
})

const appleRun = (values: Partial<AppleWorkout> = {}): AppleWorkout => ({
  id: 'apple-run',
  activity: 'running',
  start: '2026-07-17T00:30:45Z',
  end: '2026-07-17T01:10:45Z',
  durationS: 2_400,
  distanceM: 5_643.5,
  source: 'Runna',
  device: 'Apple Watch',
  heartRate: [],
  strideLengthM: [],
  groundContactTimeMs: [],
  verticalOscillationCm: [],
  ...values,
})

const payloadWith = (...details: StravaActivityDetail[]): StravaPayload => {
  const payload = emptyPayload(1)
  for (const item of details) payload.details[String(item.id)] = item
  return payload
}

test('resolves run, walk, and swim devices only from exact model evidence', () => {
  const run = detail({
    id: 1,
    sport: 'run',
    start: '2026-07-17T00:30:45Z',
    distanceKm: 5.6435,
    movingTimeS: 2_400,
    elapsedTimeS: 2_400,
    garmin: { ...garminVerification('garmin-run'), sourceDevice: null },
  })
  const walk = detail({
    id: 2,
    sport: 'walk',
    start: '2026-07-18T12:00:00Z',
    distanceKm: 2,
    movingTimeS: 1_200,
    elapsedTimeS: 1_200,
    garmin: { ...garminVerification('garmin-walk'), sourceDevice: null },
  })
  const swim = detail({
    id: 3,
    sport: 'swim',
    start: '2026-07-19T12:00:00Z',
    distanceKm: 1,
    movingTimeS: 1_800,
    elapsedTimeS: 1_800,
  })
  const bike = detail({ id: 4, sport: 'bike', device: null })
  const garminRun = detail({
    id: 5,
    sport: 'run',
    start: '2026-07-20T12:00:00Z',
    distanceKm: 8,
    movingTimeS: 2_800,
    elapsedTimeS: 2_800,
    garmin: { ...garminVerification('garmin-only-run'), sourceDevice: null },
  })
  const exactGarminRun = detail({
    id: 6,
    sport: 'run',
    start: '2026-07-21T12:00:00Z',
    distanceKm: 8,
    movingTimeS: 2_800,
    elapsedTimeS: 2_800,
    garmin: garminVerification('exact-garmin-run'),
  })
  const workout = (
    id: string,
    activity: string,
    start: string,
    durationS: number,
    distanceM: number,
    values: Partial<AppleWorkout>,
  ): AppleWorkout => ({
    id,
    activity,
    start,
    end: new Date(Date.parse(start) + durationS * 1_000).toISOString(),
    durationS,
    distanceM,
    heartRate: [],
    ...values,
  })
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    workouts: {
      run: workout('run', 'running', run.start, run.movingTimeS, run.distanceKm * 1_000, {
        source: 'Runna',
        device: 'Apple Watch',
      }),
      walk: workout('walk', 'walking', walk.start, walk.movingTimeS, walk.distanceKm * 1_000, {
        source: 'appl-watch-ultra-3',
        device: 'Apple Watch',
      }),
      swim: workout('swim', 'swimming', swim.start, swim.movingTimeS, swim.distanceKm * 1_000, {
        device: 'Apple Watch Ultra 3 49mm',
      }),
    },
  }

  enrichActivityDevices(payloadWith(run, walk, swim, bike, garminRun, exactGarminRun), apple)

  assert.equal(run.device, null)
  assert.equal(walk.device, 'apple-watch-ultra-3')
  assert.equal(swim.device, 'apple-watch-ultra-3')
  assert.equal(bike.device, null)
  assert.equal(garminRun.device, null)
  assert.equal(exactGarminRun.device, 'garmin-forerunner-970')
})

test('enriches SSR payloads with pace-derived intensity factors and exercise loads', () => {
  const swim = detail()
  const payload = payloadWith(swim)

  enrichCalculatedIntensityFactors(payload, [{ id: swim.id, paceIntensityFactor: 1.011 }], 249, 173)
  enrichCalculatedExerciseLoads(payload)
  enrichCalculatedTrainingEffects(payload)

  assert.deepEqual(swim.calculatedIntensityFactor, { value: 1.011, source: 'pace' })
  assert.deepEqual(swim.calculatedExerciseLoad, { value: 45.1, source: 'pace' })
  assert.deepEqual(swim.calculatedTrainingEffect, {
    aerobic: 3.1,
    anaerobic: 0,
    evidence: {
      aerobic: { source: 'exercise-load', load: 45.1 },
      anaerobic: { source: 'heart-rate', seconds: 0 },
    },
  })
})

test('aligns native Apple running dynamics to the matching run route', () => {
  const start = '2026-07-17T00:30:45Z'
  const run = detail({
    sport: 'run',
    name: 'Runna run',
    start,
    distanceKm: 5.6435,
    route: [0, 5, 30].map((elapsedS, index) => ({
      x: index / 2,
      y: index / 2,
      d: index,
      alt: 100,
      w: 0,
      hr: 150,
      cad: 80,
      stamina: null,
      potentialStamina: null,
      resp: null,
      tempC: null,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
      lat: 43,
      lng: -79,
      elapsedS,
      speedKph: 10,
    })),
  })
  const sparseDuplicate = appleRun({
    id: 'strava-copy',
    source: 'Strava',
    strideLengthM: [{ time: start, value: 1.5 }],
  })
  const native = appleRun({
    strideLengthM: [
      { time: start, value: 1.18 },
      { time: '2026-07-17T00:30:50Z', value: 1.21 },
    ],
    groundContactTimeMs: [
      { time: start, value: 241 },
      { time: '2026-07-17T00:30:50Z', value: 238 },
    ],
    verticalOscillationCm: [
      { time: start, value: 9.8 },
      { time: '2026-07-17T00:30:50Z', value: 9.6 },
    ],
  })
  const payload = payloadWith(run)
  const apple: AppleCache = {
    version: 9,
    lastSync: 1,
    days: {},
    workouts: { [sparseDuplicate.id]: sparseDuplicate, [native.id]: native },
  }

  enrichRunDynamics(payload, apple)

  assert.deepEqual(
    payload.details['1'].route.map(point => ({
      strideLengthM: point.strideLengthM,
      groundContactTimeMs: point.groundContactTimeMs,
      verticalOscillationCm: point.verticalOscillationCm,
    })),
    [
      { strideLengthM: 1.18, groundContactTimeMs: 241, verticalOscillationCm: 9.8 },
      { strideLengthM: 1.21, groundContactTimeMs: 238, verticalOscillationCm: 9.6 },
      { strideLengthM: null, groundContactTimeMs: null, verticalOscillationCm: null },
    ],
  )
})

test('enriches a route-less treatment with sparse Apple heart rate samples', () => {
  const start = '2026-07-27T20:15:24Z'
  const treatment = detail({
    sport: 'treatment',
    name: 'Post race full body physio',
    start,
    movingTimeS: 1_291,
    avgHr: null,
    maxHr: null,
    route: [],
    heartRateTrace: [],
  })
  const workout: AppleWorkout = {
    id: 'apple-physio',
    activity: 'other',
    start,
    end: '2026-07-27T20:36:55Z',
    durationS: 1_291,
    averageHeartRateBpm: 63,
    source: 'Strava',
    heartRate: [
      { time: '2026-07-27T20:18:55.096Z', bpm: 60 },
      { time: '2026-07-27T20:22:37Z', bpm: 62 },
      { time: '2026-07-27T20:22:55Z', bpm: 67 },
      { time: '2026-07-27T20:23:01Z', bpm: 64 },
    ],
  }
  const payload = payloadWith(treatment)

  enrichRouteLessHeartRate(payload, {
    version: 9,
    lastSync: 1,
    days: {},
    workouts: { [workout.id]: workout },
  })

  assert.equal(treatment.avgHr, 63)
  assert.equal(treatment.maxHr, 67)
  assert.deepEqual(
    treatment.heartRateTrace.map(point => ({
      elapsedS: point.elapsedS,
      heartRate: point.heartRate,
    })),
    [
      { elapsedS: 0, heartRate: null },
      { elapsedS: 211.1, heartRate: 60 },
      { elapsedS: 433, heartRate: 62 },
      { elapsedS: 451, heartRate: 67 },
      { elapsedS: 457, heartRate: 64 },
      { elapsedS: 1_291, heartRate: null },
    ],
  )
})

test('keeps manual sauna heart rate Oura-only', () => {
  const start = '2026-08-23T22:30:00Z'
  const sauna = detail({
    sport: 'sauna',
    name: 'sauna',
    start,
    movingTimeS: 4_500,
    avgHr: null,
    maxHr: null,
    route: [],
    heartRateTrace: [],
    sauna: {
      time: '18:30',
      temperatureC: 91.111,
      humidityPct: 11,
      cooldown: 'cold plunge',
      heatTrainingLoad: 7.7,
      heartRateSource: null,
      source: 'manual',
    },
  })
  const workout: AppleWorkout = {
    id: 'apple-sauna',
    activity: 'other',
    start,
    end: '2026-08-23T23:45:00Z',
    durationS: 4_500,
    averageHeartRateBpm: 120,
    source: 'Apple Watch',
    heartRate: [
      { time: '2026-08-23T22:35:00Z', bpm: 110 },
      { time: '2026-08-23T23:30:00Z', bpm: 130 },
    ],
  }

  enrichRouteLessHeartRate(payloadWith(sauna), {
    version: 9,
    lastSync: 1,
    days: {},
    workouts: { [workout.id]: workout },
  })

  assert.equal(sauna.avgHr, null)
  assert.equal(sauna.maxHr, null)
  assert.deepEqual(sauna.heartRateTrace, [])
})

test('applies attached manual sauna metadata through the shared payload path', () => {
  const activity = detail({
    id: 20012367069,
    sport: 'treatment',
    name: 'Guided Down: Self-Care Sweat',
    date: '2026-09-02',
    start: '2026-09-02T21:30:00Z',
    movingTimeS: 4_000,
    elapsedTimeS: 4_000,
    avgHr: 100,
    maxHr: 150,
  })
  const payload = payloadWith(activity)
  payload.days = [
    {
      date: activity.date,
      durationS: activity.movingTimeS,
      dominant: 'treatment',
      items: [
        { id: activity.id, sport: 'treatment', distanceKm: 0, durationS: activity.movingTimeS },
      ],
    },
  ]
  const oura: OuraCache = { lastSync: 1, days: {} }
  const garminActivityId = 24_229_638_323
  const metrics = emptyGarminMetrics()
  metrics.aerobicTrainingEffect = 0.4
  metrics.anaerobicTrainingEffect = 0
  metrics.exerciseLoad = 7
  metrics.trainingEffectLabel = 'RECOVERY'
  metrics.aerobicTrainingEffectMessage = 'RECOVERY_5'
  metrics.anaerobicTrainingEffectMessage = 'NO_ANAEROBIC_BENEFIT_0'
  const garmin: GarminCache = {
    lastSync: 1,
    activities: {
      [`connect:${garminActivityId}`]: {
        id: `connect:${garminActivityId}`,
        name: 'Cardio',
        sport: null,
        startDate: activity.start,
        startDateLocal: '2026-09-02T17:30:00',
        distanceM: null,
        movingTimeS: null,
        elapsedTimeS: activity.elapsedTimeS,
        sourceDevice: 'Forerunner 970',
        sourceFile: null,
        metrics,
        fueling: emptyGarminFueling('Forerunner 970'),
      },
    },
  }

  applyManualActivityTracking(
    payload,
    {
      activities: [],
      fueling: [],
      strength: [],
      sauna: [
        {
          id: 8_202_609_021_730,
          stravaActivityId: activity.id,
          garminActivityId,
          title: 'Guided Down, Self-Care Sweat',
          date: activity.date,
          time: '17:30',
          durationS: 4_500,
          temperatureC: 71.111,
          humidityPct: 11,
          cooldown: 'cold plunge',
          heatTrainingLoad: 7.7,
        },
      ],
    },
    oura,
    null,
    garmin,
  )

  assert.equal(activity.sport, 'sauna')
  assert.equal(activity.name, 'Guided Down, Self-Care Sweat')
  assert.deepEqual(activity.sauna, {
    time: '17:30',
    temperatureC: 71.111,
    humidityPct: 11,
    cooldown: 'cold plunge',
    heatTrainingLoad: 7.7,
    heartRateSource: null,
    source: 'manual',
  })
  assert.equal(activity.garmin?.trainingEffectActivityId, `connect:${garminActivityId}`)
  assert.equal(activity.garmin?.aerobicTrainingEffect, 0.4)
  assert.equal(activity.garmin?.anaerobicTrainingEffect, 0)
  assert.equal(activity.garmin?.exerciseLoad, 7)
  assert.equal(activity.garmin?.trainingEffectLabel, 'RECOVERY')
  assert.equal(payload.days[0]?.items[0]?.sport, 'sauna')
  assert.equal(payload.days[0]?.dominant, 'sauna')
})

test('run and walk device thermal values win per channel before bounded CORE app fallback', () => {
  const start = '2026-07-29T19:00:00.000Z'
  const core: CoreBodyTemperatureCache = {
    version: 1,
    lastSync: 1,
    samples: [
      {
        time: start,
        coreTemperatureC: 37.5,
        skinTemperatureC: 32,
        heatStrainIndex: 2,
        quality: 4,
        heartRate: 150,
      },
      {
        time: '2026-07-29T19:02:00.000Z',
        coreTemperatureC: 37.7,
        skinTemperatureC: 33,
        heatStrainIndex: 3,
        quality: 4,
        heartRate: 155,
      },
    ],
  }

  const fitSource = (value: number | null): ActivityThermalSource | null =>
    value == null ? null : 'core-fit'
  const sports: ('run' | 'walk')[] = ['run', 'walk']
  for (const sport of sports) {
    const activity = detail({
      sport,
      start,
      route: [0, 60, 120, 300].map((elapsedS, index) => {
        const native = [
          { heatStrainIndex: 0, coreTemperatureC: null, skinTemperatureC: 31 },
          { heatStrainIndex: null, coreTemperatureC: 37.1, skinTemperatureC: null },
          { heatStrainIndex: 1, coreTemperatureC: 37.2, skinTemperatureC: 31.5 },
          { heatStrainIndex: null, coreTemperatureC: null, skinTemperatureC: null },
        ][index]
        return {
          x: index / 3,
          y: index / 3,
          d: index,
          alt: 100,
          w: 0,
          hr: 150,
          cad: 80,
          stamina: null,
          potentialStamina: null,
          resp: null,
          tempC: 25,
          heatStrainIndex: native.heatStrainIndex,
          heatStrainSource: fitSource(native.heatStrainIndex),
          coreTemperatureC: native.coreTemperatureC,
          coreTemperatureSource: fitSource(native.coreTemperatureC),
          skinTemperatureC: native.skinTemperatureC,
          skinTemperatureSource: fitSource(native.skinTemperatureC),
          lat: 43,
          lng: -79,
          elapsedS,
          speedKph: 10,
        }
      }),
    })
    enrichCoreBodyTemperature(payloadWith(activity), core)

    assert.deepEqual(
      activity.route.map(point => ({
        heat: [point.heatStrainIndex, point.heatStrainSource],
        core: [point.coreTemperatureC, point.coreTemperatureSource],
        skin: [point.skinTemperatureC, point.skinTemperatureSource],
      })),
      [
        { heat: [0, 'core-fit'], core: [37.5, 'core-app'], skin: [31, 'core-fit'] },
        { heat: [2.5, 'core-app'], core: [37.1, 'core-fit'], skin: [32.5, 'core-app'] },
        { heat: [1, 'core-fit'], core: [37.2, 'core-fit'], skin: [31.5, 'core-fit'] },
        { heat: [null, null], core: [null, null], skin: [null, null] },
      ],
      sport,
    )
  }
})

test('enriches route-less yoga timelines with nearby CORE app samples', () => {
  const start = '2026-07-29T19:00:00.000Z'
  const yoga = detail({
    sport: 'yoga',
    start,
    movingTimeS: 300,
    route: [],
    heartRateTrace: [0, 60, 120, 300].map(elapsedS => ({
      distanceKm: 0,
      elapsedS,
      heartRate: 90,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    })),
  })
  const payload = payloadWith(yoga)
  const core: CoreBodyTemperatureCache = {
    version: 1,
    lastSync: 1,
    samples: [
      {
        time: start,
        coreTemperatureC: 37.5,
        skinTemperatureC: 32,
        heatStrainIndex: 2,
        quality: 4,
        heartRate: 90,
      },
      {
        time: '2026-07-29T19:02:00.000Z',
        coreTemperatureC: 37.7,
        skinTemperatureC: 33,
        heatStrainIndex: 3,
        quality: 4,
        heartRate: 95,
      },
    ],
  }

  enrichCoreBodyTemperature(payload, core)

  assert.deepEqual(
    yoga.heartRateTrace.map(point => ({
      coreTemperatureC: point.coreTemperatureC,
      skinTemperatureC: point.skinTemperatureC,
      heatStrainIndex: point.heatStrainIndex,
      source: point.coreTemperatureSource,
    })),
    [
      { coreTemperatureC: 37.5, skinTemperatureC: 32, heatStrainIndex: 2, source: 'core-app' },
      { coreTemperatureC: 37.6, skinTemperatureC: 32.5, heatStrainIndex: 2.5, source: 'core-app' },
      { coreTemperatureC: 37.7, skinTemperatureC: 33, heatStrainIndex: 3, source: 'core-app' },
      { coreTemperatureC: null, skinTemperatureC: null, heatStrainIndex: null, source: null },
    ],
  )
})

test('enriches swim detail and trend with Apple count, rate, and active-time pace', () => {
  const payload = payloadWith(detail())
  const swim = appleSwim({
    intervals: [
      {
        start: '2026-07-09T20:13:30Z',
        end: '2026-07-09T20:13:55Z',
        distanceM: 25,
        strokeCount: 10,
        strokeTimeS: 25,
        stroke: 'freestyle',
      },
      {
        start: '2026-07-09T20:14:10Z',
        end: '2026-07-09T20:14:36Z',
        distanceM: 25,
        strokeCount: 11,
        strokeTimeS: 13,
        stroke: 'freestyle',
      },
    ],
  })
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [swim.id ?? swim.date]: swim },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple, null)

  assert.deepEqual(payload.details['1'].strokes, swim.strokes)
  assert.equal(payload.details['1'].strokeCount, 700)
  assert.equal(payload.details['1'].strokeRateSpm, 28)
  assert.equal(payload.details['1'].swimPaceSPer100m, 160)
  assert.equal(payload.details['1'].swimDurationS, 1_600)
  assert.equal(payload.details['1'].swimLocation, 'pool')
  assert.equal(payload.details['1'].waterTemperatureC, 27.8)
  assert.deepEqual(payload.details['1'].swimIntervals, [
    {
      startElapsedS: 0,
      endElapsedS: 25,
      distanceM: 25,
      durationS: 25,
      cumulativeDistanceM: 25,
      paceSPer100m: 100,
      strokeCount: 10,
      strokeTimeS: 25,
      strokeRateSpm: 24,
      stroke: 'freestyle',
    },
    {
      startElapsedS: 40,
      endElapsedS: 66,
      distanceM: 25,
      durationS: 26,
      cumulativeDistanceM: 50,
      paceSPer100m: 104,
      strokeCount: 11,
      strokeTimeS: 13,
      strokeRateSpm: 50.8,
      stroke: 'freestyle',
    },
  ])
  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-09',
      start: '2026-07-09T20:13:31Z',
      paceSPer100m: 160,
      paceSource: 'active',
      strokeRateSpm: 28,
    },
  ])
})

test('uses the device cadence as swim stroke rate when Apple stroke timing is unavailable', () => {
  const payload = payloadWith(detail({ avgCadence: 26 }))

  enrichSwimMetrics(payload, null, null)

  assert.equal(payload.details['1'].strokeRateSpm, 26)
  assert.equal(payload.swimTrend[0]?.strokeRateSpm, 26)
})

test('uses Garmin FIT pool lengths for native swim pace, strokes, cadence, and SWOLF inputs', () => {
  const activityId = 'connect:24227073871'
  const payload = payloadWith(
    detail({
      distanceKm: 0.05,
      movingTimeS: 51,
      elapsedTimeS: 66,
      garmin: garminVerification(activityId),
      heartRateTrace: [0, 25, 32, 40, 53, 66].map(elapsedS => ({
        distanceKm: 0,
        elapsedS,
        heartRate: 120,
        heatStrainIndex: null,
        heatStrainSource: null,
        coreTemperatureC: null,
        coreTemperatureSource: null,
        skinTemperatureC: null,
        skinTemperatureSource: null,
      })),
    }),
  )
  const conflictingApple = appleSwim({
    totalM: 50,
    activeTimeS: 60,
    strokeCount: 30,
    strokeTimeS: 60,
    strokes: { breaststroke: 30 },
    intervals: [],
  })
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [conflictingApple.id ?? conflictingApple.date]: conflictingApple },
    workouts: {},
  }
  const garmin: GarminCache = {
    version: 12,
    lastSync: 1,
    activities: {},
    swims: {
      [activityId]: {
        location: 'pool',
        elapsedTimeS: 66,
        activeTimeS: 51,
        distanceM: 50,
        strokeCount: 21,
        strokeRateSpm: 24.7,
        poolLengthM: 25,
        laps: [
          {
            startElapsedS: 0,
            endElapsedS: 25,
            distanceM: 25,
            durationS: 25,
            strokeCount: 10,
            strokeTimeS: 25,
            strokeRateSpm: 24,
            stroke: 'freestyle',
            averageHeartRate: 125,
            elevationGainM: 3,
          },
          {
            startElapsedS: 40,
            endElapsedS: 66,
            distanceM: 25,
            durationS: 26,
            strokeCount: 11,
            strokeTimeS: 26,
            strokeRateSpm: 25.4,
            stroke: 'freestyle',
            averageHeartRate: 135,
            elevationGainM: 4,
          },
        ],
        lengths: [
          {
            startElapsedS: 0,
            endElapsedS: 25,
            distanceM: 25,
            durationS: 25,
            strokeCount: 10,
            strokeTimeS: 25,
            strokeRateSpm: 24,
            stroke: 'freestyle',
          },
          {
            startElapsedS: 40,
            endElapsedS: 66,
            distanceM: 25,
            durationS: 26,
            strokeCount: 11,
            strokeTimeS: 26,
            strokeRateSpm: 25.4,
            stroke: 'freestyle',
          },
        ],
      },
    },
  }

  enrichSwimMetrics(payload, apple, garmin)

  const enriched = payload.details['1']
  assert.deepEqual(enriched.strokes, { freestyle: 50 })
  assert.equal(enriched.strokeCount, 21)
  assert.equal(enriched.strokeRateSpm, 24.7)
  assert.equal(enriched.swimPaceSPer100m, 102)
  assert.equal(enriched.swimPaceSource, 'stroke')
  assert.equal(enriched.swimDurationS, 66)
  assert.equal(enriched.swimLocation, 'pool')
  assert.equal(enriched.waterTemperatureC, 27.8)
  assert.deepEqual(swimLengthAverages(enriched.swimIntervals), {
    strokesPerLength: 10.5,
    swolf: 36,
  })
  assert.deepEqual(
    enriched.swimIntervals.map(interval => interval.cumulativeDistanceM),
    [25, 50],
  )
  assert.deepEqual(
    enriched.heartRateTrace.map(point => point.distanceKm),
    [0, 0.025, 0.025, 0.025, 0.0375, 0.05],
  )
  assert.deepEqual(
    enriched.analysisRanges.map(range => [
      range.id,
      range.startElapsedS,
      range.endElapsedS,
      range.startDistanceKm,
      range.endDistanceKm,
      range.elevationGainM,
      range.averageHeartRate,
      range.averageCadence,
    ]),
    [
      ['garmin-swim-lap:1', 0, 25, 0, 0.025, null, 125, 24],
      ['garmin-swim-lap:2', 40, 66, 0.025, 0.05, null, 135, 25.4],
    ],
  )
})

test('prefers a complete measured-length pace over workout duration for pool swims', () => {
  const payload = payloadWith(detail({ distanceKm: 0.1, movingTimeS: 160, swimLocation: 'pool' }))
  const swim = appleSwim({
    totalM: 100,
    activeTimeS: 160,
    laps: 4,
    intervals: [
      {
        start: '2026-07-09T20:13:30Z',
        end: '2026-07-09T20:13:55Z',
        distanceM: 25,
        strokeCount: 10,
        strokeTimeS: 25,
        stroke: 'freestyle',
      },
      {
        start: '2026-07-09T20:14:10Z',
        end: '2026-07-09T20:14:40Z',
        distanceM: 25,
        strokeCount: 11,
        strokeTimeS: 30,
        stroke: 'freestyle',
      },
      {
        start: '2026-07-09T20:15:00Z',
        end: '2026-07-09T20:15:35Z',
        distanceM: 25,
        strokeCount: 12,
        strokeTimeS: 35,
        stroke: 'freestyle',
      },
      {
        start: '2026-07-09T20:16:00Z',
        end: '2026-07-09T20:16:40Z',
        distanceM: 25,
        strokeCount: 13,
        strokeTimeS: 40,
        stroke: 'freestyle',
      },
    ],
  })
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [swim.id ?? swim.date]: swim },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple, null)

  assert.equal(payload.details['1'].swimPaceSPer100m, 130)
  assert.equal(payload.swimTrend[0]?.paceSPer100m, 130)
})

test('uses corrected distance metrics with measured open-water environment', () => {
  const start = '2026-07-26T12:43:52Z'
  const payload = payloadWith(
    detail({
      name: 'Toronto Triathlon Festival',
      date: '2026-07-26',
      start,
      distanceKm: 1.5,
      movingTimeS: 2_460,
    }),
  )
  const measured = appleSwim({
    id: 'apple-watch',
    date: '2026-07-26',
    start,
    end: '2026-07-26T13:24:52Z',
    totalM: 438.9,
    activeTimeS: 2_460,
    strokeCount: 1_314,
    strokeTimeS: 2_465,
    strokes: {},
    location: 'openWater',
    waterTemperatureC: 14.4,
  })
  const corrected = appleSwim({
    id: 'strava-copy',
    date: '2026-07-26',
    start: '2026-07-26T12:43:54Z',
    end: '2026-07-26T13:24:54Z',
    totalM: 1_500,
    activeTimeS: 2_460,
    strokeCount: null,
    strokeTimeS: null,
    strokes: {},
    location: null,
    waterTemperatureC: null,
  })
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { measured, corrected },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple, null)

  assert.equal(payload.details['1'].swimPaceSPer100m, 164)
  assert.equal(payload.details['1'].swimLocation, 'openWater')
  assert.equal(payload.details['1'].waterTemperatureC, 14.4)
  assert.equal(payload.details['1'].strokeCount, 1_314)
  assert.equal(payload.details['1'].strokeRateSpm, 32)
})

test('keeps kickboard lengths in the distance series without inventing stroke rate', () => {
  assert.deepEqual(
    swimActivityIntervals(
      appleSwim({
        start: '2026-07-09T20:00:00Z',
        end: '2026-07-09T20:02:00Z',
        intervals: [
          {
            start: '2026-07-09T20:00:10Z',
            end: '2026-07-09T20:00:40Z',
            distanceM: 25,
            strokeCount: null,
            strokeTimeS: null,
            stroke: 'kickboard',
          },
          {
            start: '2026-07-09T20:01:00Z',
            end: '2026-07-09T20:01:25Z',
            distanceM: 25,
            strokeCount: 10,
            strokeTimeS: 20,
            stroke: 'freestyle',
          },
        ],
      }),
    ),
    {
      durationS: 120,
      intervals: [
        {
          startElapsedS: 10,
          endElapsedS: 40,
          distanceM: 25,
          durationS: 30,
          cumulativeDistanceM: 25,
          paceSPer100m: 120,
          strokeCount: null,
          strokeTimeS: null,
          strokeRateSpm: null,
          stroke: 'kickboard',
        },
        {
          startElapsedS: 60,
          endElapsedS: 85,
          distanceM: 25,
          durationS: 25,
          cumulativeDistanceM: 50,
          paceSPer100m: 100,
          strokeCount: 10,
          strokeTimeS: 20,
          strokeRateSpm: 30,
          stroke: 'freestyle',
        },
      ],
    },
  )
})

test('uses exported subsecond offsets and duration for pace and elapsed endpoint', () => {
  assert.deepEqual(
    swimActivityIntervals(
      appleSwim({
        start: '2026-07-09T20:00:00Z',
        end: '2026-07-09T20:01:00Z',
        intervals: [
          {
            start: '2026-07-09T20:00:10Z',
            end: '2026-07-09T20:00:37Z',
            distanceM: 22.86,
            startElapsedS: 10.4,
            endElapsedS: 37.1,
            durationS: 26.66,
            strokeCount: 16,
            strokeTimeS: 20,
            stroke: 'freestyle',
          },
        ],
      }),
    ),
    {
      durationS: 60,
      intervals: [
        {
          startElapsedS: 10.4,
          endElapsedS: 37.1,
          distanceM: 22.9,
          durationS: 26.7,
          cumulativeDistanceM: 22.9,
          paceSPer100m: 116.6,
          strokeCount: 16,
          strokeTimeS: 20,
          strokeRateSpm: 48,
          stroke: 'freestyle',
        },
      ],
    },
  )
})

test('keeps two same-date swim activities as separate trend observations', () => {
  const morningActivity = detail({
    id: 1,
    start: '2026-07-09T10:01:00Z',
    distanceKm: 0.5,
    movingTimeS: 620,
  })
  const eveningActivity = detail({
    id: 2,
    start: '2026-07-09T18:02:00Z',
    distanceKm: 1,
    movingTimeS: 1_520,
  })
  const morning = appleSwim({
    id: 'morning',
    start: '2026-07-09T10:00:00Z',
    end: '2026-07-09T10:10:00Z',
    totalM: 500,
    activeTimeS: 600,
    strokeCount: 300,
    strokeTimeS: 600,
    intervals: [
      {
        start: '2026-07-09T10:00:00Z',
        end: '2026-07-09T10:00:25Z',
        distanceM: 25,
        strokeCount: 10,
        strokeTimeS: 25,
        stroke: 'freestyle',
      },
    ],
  })
  const evening = appleSwim({
    id: 'evening',
    start: '2026-07-09T18:00:00Z',
    end: '2026-07-09T18:25:00Z',
    totalM: 1_000,
    activeTimeS: 1_500,
    strokeCount: 560,
    strokeTimeS: 1_200,
    intervals: [
      {
        start: '2026-07-09T18:00:00Z',
        end: '2026-07-09T18:00:30Z',
        distanceM: 50,
        strokeCount: 16,
        strokeTimeS: 30,
        stroke: 'breaststroke',
      },
    ],
  })
  const payload = payloadWith(eveningActivity, morningActivity)
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { evening, morning },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple, null)

  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-09',
      start: '2026-07-09T10:01:00Z',
      paceSPer100m: 120,
      paceSource: 'active',
      strokeRateSpm: 30,
    },
    {
      id: 2,
      date: '2026-07-09',
      start: '2026-07-09T18:02:00Z',
      paceSPer100m: 150,
      paceSource: 'active',
      strokeRateSpm: 28,
    },
  ])
  assert.deepEqual(payload.details['1'].swimIntervals, [
    {
      startElapsedS: 0,
      endElapsedS: 25,
      distanceM: 25,
      durationS: 25,
      cumulativeDistanceM: 25,
      paceSPer100m: 100,
      strokeCount: 10,
      strokeTimeS: 25,
      strokeRateSpm: 24,
      stroke: 'freestyle',
    },
  ])
  assert.deepEqual(payload.details['2'].swimIntervals, [
    {
      startElapsedS: 0,
      endElapsedS: 30,
      distanceM: 50,
      durationS: 30,
      cumulativeDistanceM: 50,
      paceSPer100m: 60,
      strokeCount: 16,
      strokeTimeS: 30,
      strokeRateSpm: 32,
      stroke: 'breaststroke',
    },
  ])
})

test('keeps valid Strava pace history and drops an implausible GPS swim', () => {
  const payload = payloadWith(
    detail({ id: 1, date: '2026-07-08', start: '2026-07-08T20:00:00Z' }),
    detail({
      id: 2,
      date: '2026-07-09',
      distanceKm: 2.3,
      movingTimeS: 306,
      start: '2026-07-09T20:00:00Z',
    }),
  )

  enrichSwimMetrics(payload, null, null)

  assert.equal(payload.details['1'].swimPaceSPer100m, 159)
  assert.equal(payload.details['2'].swimPaceSPer100m, null)
  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-08',
      start: '2026-07-08T20:00:00Z',
      paceSPer100m: 159,
      paceSource: 'moving',
      strokeRateSpm: null,
    },
  ])
})

test('keeps a valid stroke-rate observation when pace is unavailable', () => {
  const payload = payloadWith(detail({ movingTimeS: 20 }))
  const swim = appleSwim({ activeTimeS: 20 })
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [swim.id ?? swim.date]: swim },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple, null)

  assert.equal(payload.details['1'].swimPaceSPer100m, null)
  assert.equal(payload.details['1'].strokeRateSpm, 28)
  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-09',
      start: '2026-07-09T20:13:31Z',
      paceSPer100m: null,
      paceSource: null,
      strokeRateSpm: 28,
    },
  ])
})

const lapInterval = (index: number, durationS: number): AppleSwimInterval => ({
  start: new Date(Date.parse('2026-07-09T20:13:30Z') + index * 40_000).toISOString(),
  end: new Date(
    Date.parse('2026-07-09T20:13:30Z') + index * 40_000 + durationS * 1_000,
  ).toISOString(),
  distanceM: 25,
  startElapsedS: index * 40,
  endElapsedS: index * 40 + durationS,
  durationS,
  strokeCount: 12,
  strokeTimeS: durationS,
  stroke: 'freestyle',
})

test('paces a swim by stroke time when lap distance overshoots the reported total', () => {
  const payload = payloadWith(detail({ distanceKm: 0.19, movingTimeS: 320 }))
  const swim = appleSwim({
    totalM: 190,
    laps: 8,
    activeTimeS: 320,
    intervals: Array.from({ length: 8 }, (_, index) => lapInterval(index, 30)),
  })

  enrichSwimMetrics(
    payload,
    { version: 4, lastSync: 1, days: {}, swims: { [swim.id ?? swim.date]: swim }, workouts: {} },
    null,
  )

  assert.equal(payload.details['1'].swimPaceSPer100m, 120)
  assert.equal(payload.details['1'].swimPaceSource, 'stroke')
})

test('falls back to active time when intervals cover a fraction of the swim', () => {
  const payload = payloadWith(detail({ movingTimeS: 1_600 }))
  const swim = appleSwim({
    totalM: 1_000,
    activeTimeS: 1_600,
    intervals: Array.from({ length: 4 }, (_, index) => lapInterval(index, 30)),
  })

  enrichSwimMetrics(
    payload,
    { version: 4, lastSync: 1, days: {}, swims: { [swim.id ?? swim.date]: swim }, workouts: {} },
    null,
  )

  assert.equal(payload.details['1'].swimPaceSPer100m, 160)
  assert.equal(payload.details['1'].swimPaceSource, 'active')
})
