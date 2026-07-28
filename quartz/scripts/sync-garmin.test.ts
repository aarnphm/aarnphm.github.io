import assert from 'node:assert/strict'
import test from 'node:test'
import type { GarminVo2Day, GarminWeightSample } from '../plugins/stores/garmin'
import {
  initialGarminSyncRecords,
  mergeGarminVo2Range,
  mergeGarminWeightRange,
  resolveGarminFetch,
  resolveGarminWeightDay,
} from './sync-garmin'

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

test('keeps untouched Garmin records during a capped sync', () => {
  const previous = { one: { value: 1 }, two: { value: 2 } }

  assert.deepEqual(initialGarminSyncRecords(previous, true), previous)
  assert.deepEqual(initialGarminSyncRecords(previous, false), {})
  assert.notEqual(initialGarminSyncRecords(previous, true), previous)
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
