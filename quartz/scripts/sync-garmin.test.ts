import assert from 'node:assert/strict'
import test from 'node:test'
import {
  emptyGarminFueling,
  emptyGarminMetrics,
  type GarminActivity,
  type GarminVo2Day,
  type GarminWeightSample,
} from '../plugins/stores/garmin'
import {
  garminRefreshStart,
  mergeGarminFitTrainingEffect,
  mergeGarminVo2Range,
  mergeGarminWeightRange,
  reconcileGarminActivities,
  resolveGarminFetch,
  resolveGarminWeightDay,
} from './sync-garmin'

function activity(id = 'connect:123', startDateLocal = '2026-08-14T13:27:27.0'): GarminActivity {
  return {
    id,
    name: 'Imported ride',
    sport: 'bike',
    startDate: '2026-08-14T17:27:27.000Z',
    startDateLocal,
    distanceM: 45_395,
    movingTimeS: 5_947,
    elapsedTimeS: 8_785,
    sourceDevice: null,
    sourceFile: null,
    metrics: emptyGarminMetrics(),
    fueling: emptyGarminFueling(),
  }
}

test('recovers missing Garmin Connect training effect from the stored FIT', () => {
  const recovered = mergeGarminFitTrainingEffect(activity(), { aerobic: 3, anaerobic: 1.3 })

  assert.equal(recovered.metrics.aerobicTrainingEffect, 3)
  assert.equal(recovered.metrics.anaerobicTrainingEffect, 1.3)
})

test('keeps Garmin Connect training effect when the FIT also contains values', () => {
  const source = activity()
  source.metrics.aerobicTrainingEffect = 4.2
  source.metrics.anaerobicTrainingEffect = 0.8

  assert.equal(mergeGarminFitTrainingEffect(source, { aerobic: 3, anaerobic: 1.3 }), source)
})

test('preserves Garmin cache data only when a fetch fails', () => {
  const previous = [{ id: 1 }]

  assert.equal(resolveGarminFetch({ ok: false }, previous), previous)
  assert.equal(resolveGarminFetch({ ok: true }, previous), undefined)
  assert.deepEqual(resolveGarminFetch({ ok: true, value: [] }, previous), [])
})

test('preserves same-day Garmin weigh-ins when dayview fails', () => {
  const sample = (ts: number, weightKg: number): GarminWeightSample => ({
    ts,
    date: '2026-07-09',
    weightKg,
    bmi: null,
    bodyFatPct: null,
    bodyWaterPct: null,
    muscleMassKg: null,
    boneMassKg: null,
  })
  const summary = sample(300, 88)
  const previous = [sample(100, 88.84), sample(200, 87.55)]

  assert.deepEqual(resolveGarminWeightDay('2026-07-09', { ok: false }, summary, previous), previous)
  assert.deepEqual(resolveGarminWeightDay('2026-07-09', { ok: true }, summary, previous), [summary])
})

test('Garmin routine refresh overlaps the latest activity and schema refresh keeps history', () => {
  const cache = {
    version: 11,
    lastSync: 1,
    activities: {
      old: activity('old', '2023-09-27T05:45:21.0'),
      latest: activity('latest', '2026-08-31T16:48:05.0'),
    },
  }

  assert.equal(garminRefreshStart(cache, '2026-05-15', 3), '2026-08-28')
  assert.equal(garminRefreshStart({ ...cache, version: 10 }, '2026-05-15', 3), '2023-09-27')
  assert.equal(garminRefreshStart(null, '2026-05-15', 3), '2026-05-15')
})

test('Garmin refresh reconciliation prunes the fetched range and preserves older history', () => {
  const previous = {
    old: activity('old', '2026-08-01T08:00:00.0'),
    deleted: activity('deleted', '2026-08-29T08:00:00.0'),
    future: activity('future', '2026-09-02T08:00:00.0'),
  }

  assert.deepEqual(
    Object.keys(reconcileGarminActivities(previous, '2026-08-28', '2026-09-01', false)),
    ['old', 'future'],
  )
  assert.deepEqual(
    Object.keys(reconcileGarminActivities(previous, '2026-08-28', '2026-09-01', true)),
    ['old', 'deleted', 'future'],
  )
})

test('replaces only the fetched Garmin VO2max date range', () => {
  const previous: Record<string, GarminVo2Day> = {
    '2026-07-26': { date: '2026-07-26', generic: 51, cycling: 54 },
    '2026-07-27': { date: '2026-07-27', generic: 52, cycling: 55 },
    '2026-07-28': { date: '2026-07-28', generic: 53, cycling: 56 },
  }
  const fetched = { '2026-07-27': { date: '2026-07-27', generic: 54, cycling: 57 } }

  assert.deepEqual(mergeGarminVo2Range(previous, fetched, '2026-07-27', '2026-07-27'), {
    '2026-07-26': previous['2026-07-26'],
    '2026-07-27': fetched['2026-07-27'],
    '2026-07-28': previous['2026-07-28'],
  })
})

test('replaces only the fetched Garmin weight date range', () => {
  const sample = (date: string, ts: number, weightKg: number): GarminWeightSample => ({
    ts,
    date,
    weightKg,
    bmi: null,
    bodyFatPct: null,
    bodyWaterPct: null,
    muscleMassKg: null,
    boneMassKg: null,
  })
  const previous = [
    sample('2026-07-26', 100, 88),
    sample('2026-07-27', 200, 87.8),
    sample('2026-07-27', 300, 87.7),
    sample('2026-07-28', 400, 87.6),
  ]
  const fetched = [sample('2026-07-27', 500, 87.5)]

  assert.deepEqual(mergeGarminWeightRange(previous, fetched, '2026-07-27', '2026-07-27'), [
    previous[0],
    previous[3],
    fetched[0],
  ])
})
