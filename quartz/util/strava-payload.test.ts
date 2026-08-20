import assert from 'node:assert/strict'
import test from 'node:test'
import type {
  AppleCache,
  AppleSwim,
  AppleSwimInterval,
  AppleWorkout,
} from '../plugins/stores/apple'
import type { CoreBodyTemperatureCache } from '../plugins/stores/core-body-temperature'
import {
  emptyPayload,
  type StravaActivityDetail,
  type StravaPayload,
} from '../plugins/stores/strava'
import {
  enrichCalculatedExerciseLoads,
  enrichCalculatedIntensityFactors,
  enrichCalculatedTrainingEffects,
  enrichCoreBodyTemperature,
  enrichRouteLessHeartRate,
  enrichRunDynamics,
  enrichSwimMetrics,
  swimActivityIntervals,
} from './strava-payload'

const detail = (values: Partial<StravaActivityDetail> = {}): StravaActivityDetail => ({
  id: 1,
  sport: 'swim',
  name: 'Pool swim',
  date: '2026-07-09',
  start: '2026-07-09T20:13:31Z',
  distanceKm: 1,
  movingTimeS: 1_590,
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
  avgTemp: null,
  windKph: null,
  windDir: null,
  windDirDeg: null,
  windGustKph: null,
  location: null,
  fueling: null,
  strength: null,
  garmin: null,
  calculatedIntensityFactor: null,
  calculatedExerciseLoad: null,
  calculatedTrainingEffect: null,
  gearShifts: [],
  cyclingDynamics: null,
  route: [],
  heartRateTrace: [],
  mapRoute: [],
  analysisRanges: [],
  runSplitsMetric: [],
  runSplitsStandard: [],
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
  ...values,
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

test('enriches SSR payloads with pace-derived intensity factors and exercise loads', () => {
  const swim = detail()
  const payload = payloadWith(swim)

  enrichCalculatedIntensityFactors(payload, [{ id: swim.id, paceIntensityFactor: 1.011 }], 249, 173)
  enrichCalculatedExerciseLoads(payload)
  enrichCalculatedTrainingEffects(payload)

  assert.deepEqual(swim.calculatedIntensityFactor, { value: 1.011, source: 'pace' })
  assert.deepEqual(swim.calculatedExerciseLoad, { value: 45.1, source: 'pace' })
  assert.deepEqual(swim.calculatedTrainingEffect, { aerobic: 3.1, anaerobic: 0 })
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
      coreTemperatureC: null,
      skinTemperatureC: null,
      coreTemperatureSource: null,
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

test('CORE app samples override FIT thermal values only near onboard timestamps', () => {
  const start = '2026-07-29T19:00:00.000Z'
  const run = detail({
    sport: 'run',
    start,
    route: [0, 60, 120, 300].map((elapsedS, index) => ({
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
      heatStrainIndex: 1,
      coreTemperatureC: 37,
      skinTemperatureC: 31,
      coreTemperatureSource: 'core-fit',
      lat: 43,
      lng: -79,
      elapsedS,
      speedKph: 10,
    })),
  })
  const payload = payloadWith(run)
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

  enrichCoreBodyTemperature(payload, core)

  assert.deepEqual(
    run.route.map(point => ({
      coreTemperatureC: point.coreTemperatureC,
      skinTemperatureC: point.skinTemperatureC,
      heatStrainIndex: point.heatStrainIndex,
      source: point.coreTemperatureSource,
    })),
    [
      { coreTemperatureC: 37.5, skinTemperatureC: 32, heatStrainIndex: 2, source: 'core-app' },
      { coreTemperatureC: 37.6, skinTemperatureC: 32.5, heatStrainIndex: 2.5, source: 'core-app' },
      { coreTemperatureC: 37.7, skinTemperatureC: 33, heatStrainIndex: 3, source: 'core-app' },
      { coreTemperatureC: 37, skinTemperatureC: 31, heatStrainIndex: 1, source: 'core-fit' },
    ],
  )
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
      coreTemperatureC: null,
      skinTemperatureC: null,
      coreTemperatureSource: null,
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

  enrichSwimMetrics(payload, apple)

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

  enrichSwimMetrics(payload, null)

  assert.equal(payload.details['1'].strokeRateSpm, 26)
  assert.equal(payload.swimTrend[0]?.strokeRateSpm, 26)
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

  enrichSwimMetrics(payload, apple)

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

  enrichSwimMetrics(payload, apple)

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

  enrichSwimMetrics(payload, apple)

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

  enrichSwimMetrics(payload, null)

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

  enrichSwimMetrics(payload, apple)

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

  enrichSwimMetrics(payload, {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [swim.id ?? swim.date]: swim },
    workouts: {},
  })

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

  enrichSwimMetrics(payload, {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [swim.id ?? swim.date]: swim },
    workouts: {},
  })

  assert.equal(payload.details['1'].swimPaceSPer100m, 160)
  assert.equal(payload.details['1'].swimPaceSource, 'active')
})
