import assert from 'node:assert/strict'
import test from 'node:test'
import type { GarminWeightSample } from '../plugins/stores/garmin'
import { initialGarminSyncRecords, resolveGarminFetch, resolveGarminWeightDay } from './sync-garmin'

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
