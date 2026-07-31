import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity } from './strava'
import {
  isUsableCoreTemperatureSample,
  matchCoreBodyTemperatureActivity,
  mergeCoreBodyTemperatureSamples,
  parseCoreBodyTemperatureApiSamples,
  parseCoreBodyTemperatureCache,
  parseCoreBodyTemperatureCsv,
  type CoreBodyTemperatureCache,
  type CoreBodyTemperatureSample,
} from './core-body-temperature'

const sample = (
  time: string,
  values: Partial<CoreBodyTemperatureSample> = {},
): CoreBodyTemperatureSample => ({
  time,
  coreTemperatureC: 37,
  skinTemperatureC: 32,
  heatStrainIndex: 1,
  quality: 4,
  heartRate: 140,
  ...values,
})

const activity: RawStravaActivity = {
  id: 1,
  name: 'Run',
  sportType: 'Run',
  distance: 10_000,
  movingTime: 3_600,
  elapsedTime: 3_660,
  totalElevationGain: 30,
  startDate: '2026-07-29T19:00:00.000Z',
  startDateLocal: '2026-07-29T15:00:00.000-04:00',
  averageSpeed: 2.78,
}

test('parseCoreBodyTemperatureCsv reads CORE Cloud columns and normalizes quality flags', () => {
  const samples = parseCoreBodyTemperatureCsv(
    [
      'Timestamp UTC,Core Body Temperature [°C],Skin Temperature [°C],Heat Strain Index,TempQuality,Heart Rate [bpm]',
      '2026-07-29 19:00:00,37.01,31.2,1.25,20,141',
      '2026-07-29 19:01:00,37.05,31.4,1.31,1,142',
    ].join('\n'),
  )

  assert.deepEqual(samples, [
    sample('2026-07-29T19:00:00.000Z', {
      coreTemperatureC: 37.01,
      skinTemperatureC: 31.2,
      heatStrainIndex: 1.25,
      quality: 4,
      heartRate: 141,
    }),
    sample('2026-07-29T19:01:00.000Z', {
      coreTemperatureC: 37.05,
      skinTemperatureC: 31.4,
      heatStrainIndex: 1.31,
      quality: 1,
      heartRate: 142,
    }),
  ])
  assert.equal(isUsableCoreTemperatureSample(samples[0]), true)
  assert.equal(isUsableCoreTemperatureSample(samples[1]), false)
})

test('parseCoreBodyTemperatureApiSamples reads the first-party CORE API schema', () => {
  const samples = parseCoreBodyTemperatureApiSamples([
    {
      timeUtc: '2026-07-29T19:00:00Z',
      timeLocal: '2026-07-29T15:00:00-04:00',
      coreTemp: 37.21,
      skinTemp: 32.4,
      heatStrainIndex: 2.3,
      heatZone: 1,
      quality: 20,
      heartrate: 148,
    },
    { timeUtc: 'invalid', coreTemp: 37 },
  ])

  assert.deepEqual(samples, [
    sample('2026-07-29T19:00:00.000Z', {
      coreTemperatureC: 37.21,
      skinTemperatureC: 32.4,
      heatStrainIndex: 2.3,
      quality: 4,
      heartRate: 148,
    }),
  ])
})

test('parseCoreBodyTemperatureCsv accepts semicolon-delimited split date and time exports', () => {
  const samples = parseCoreBodyTemperatureCsv(
    ['Date;Time;CoreTemp;SkinTemp;HSI;Quality', '29.07.2026;19:30:00;37,4;32,8;2,1;3'].join('\n'),
  )

  assert.equal(samples.length, 1)
  assert.equal(samples[0].coreTemperatureC, 37.4)
  assert.equal(samples[0].skinTemperatureC, 32.8)
  assert.equal(samples[0].heatStrainIndex, 2.1)
})

test('mergeCoreBodyTemperatureSamples fills sparse duplicate timestamps', () => {
  const merged = mergeCoreBodyTemperatureSamples(
    [sample('2026-07-29T19:00:00.000Z', { skinTemperatureC: null, heatStrainIndex: null })],
    [
      sample('2026-07-29T19:00:00.000Z', {
        coreTemperatureC: null,
        skinTemperatureC: 33,
        heatStrainIndex: 2,
      }),
    ],
  )

  assert.deepEqual(
    merged[0],
    sample('2026-07-29T19:00:00.000Z', { skinTemperatureC: 33, heatStrainIndex: 2 }),
  )
})

test('matchCoreBodyTemperatureActivity clips onboard samples to the Strava activity', () => {
  const cache: CoreBodyTemperatureCache = {
    version: 1,
    lastSync: 1,
    samples: [
      sample('2026-07-29T18:59:00.000Z'),
      sample('2026-07-29T19:00:00.000Z'),
      sample('2026-07-29T19:30:00.000Z'),
      sample('2026-07-29T20:01:00.000Z'),
      sample('2026-07-29T20:02:00.000Z'),
    ],
  }

  assert.deepEqual(
    matchCoreBodyTemperatureActivity(activity, cache).map(value => value.elapsedS),
    [0, 1_800, 3_660],
  )
})

test('parseCoreBodyTemperatureCache rejects malformed records without losing valid samples', () => {
  const cache = parseCoreBodyTemperatureCache({
    version: 1,
    lastSync: 100,
    samples: [sample('2026-07-29T19:00:00.000Z'), { time: 'invalid' }],
  })

  assert.equal(cache?.lastSync, 100)
  assert.equal(cache?.samples.length, 1)
})
