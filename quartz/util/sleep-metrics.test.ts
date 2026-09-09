import assert from 'node:assert/strict'
import test from 'node:test'
import type { GarminSleepSummary } from '../plugins/stores/garmin'
import { isSleepMetrics, resolveSleepMetrics } from './sleep-metrics'

const date = '2026-09-08'
const garmin: GarminSleepSummary = {
  source: 'garmin',
  date,
  startTime: '2026-09-08T02:00:00Z',
  endTime: '2026-09-08T10:00:00Z',
  averageBreathsPerMinute: 15,
  lowestBreathsPerMinute: 11,
  highestBreathsPerMinute: 19,
  averageSpO2: 97,
  lowestSpO2: 91,
  bodyBatteryStart: 64,
  bodyBatteryEnd: 100,
  bodyBatteryChange: 36,
  averageStress: 11,
  restlessMoments: 39,
}

test('selects Oura respiration while preserving native Garmin overnight measurements', () => {
  const metrics = resolveSleepMetrics({ avgBreath: 14.25 }, garmin)
  assert.equal(metrics?.averageBreathsPerMinute, 14.25)
  assert.equal(metrics?.respirationSource, 'oura')
  assert.deepEqual(metrics?.garmin, garmin)
  assert.equal(isSleepMetrics(JSON.parse(JSON.stringify(metrics)), date), true)
})

test('uses Garmin respiration only when Oura has no valid measurement', () => {
  for (const avgBreath of [null, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const metrics = resolveSleepMetrics({ avgBreath }, garmin)
    assert.equal(metrics?.averageBreathsPerMinute, 15)
    assert.equal(metrics?.respirationSource, 'garmin')
  }
  assert.equal(resolveSleepMetrics(undefined, garmin)?.respirationSource, 'garmin')
  assert.deepEqual(resolveSleepMetrics({ avgBreath: 14.25 }, null), {
    averageBreathsPerMinute: 14.25,
    respirationSource: 'oura',
    garmin: null,
  })
  assert.equal(resolveSleepMetrics(null, null), null)
})

test('retains zero Garmin stress and battery gain without inventing respiration', () => {
  const metrics = resolveSleepMetrics(null, {
    ...garmin,
    averageBreathsPerMinute: null,
    averageStress: 0,
    bodyBatteryChange: 0,
  })
  assert.equal(metrics?.averageBreathsPerMinute, null)
  assert.equal(metrics?.respirationSource, null)
  assert.equal(metrics?.garmin?.averageStress, 0)
  assert.equal(metrics?.garmin?.bodyBatteryChange, 0)
  assert.equal(isSleepMetrics(metrics, date), true)
})

test('validates date, provenance and finite values across the serialized boundary', () => {
  const metrics = resolveSleepMetrics(null, garmin)
  assert.equal(isSleepMetrics(metrics, '2026-09-07'), false)
  assert.equal(isSleepMetrics({ ...metrics, respirationSource: 'apple' }, date), false)
  assert.equal(isSleepMetrics({ ...metrics, averageBreathsPerMinute: 14 }, date), false)
  assert.equal(isSleepMetrics({ ...metrics, garmin: null }, date), false)
  assert.equal(isSleepMetrics({ ...metrics, garmin: { ...garmin, averageSpO2: NaN } }, date), false)
  assert.equal(isSleepMetrics({ ...metrics, averageBreathsPerMinute: null }, date), false)
  assert.equal(isSleepMetrics(undefined, date), false)
  assert.equal(isSleepMetrics(null, date), true)
})

test('preserves respiration timestamps and gaps through the serialized sleep boundary', () => {
  const timestamp = Date.parse('2026-09-08T05:30:56Z')
  const respiration = [
    { timestamp, breathsPerMinute: 19 },
    { timestamp: timestamp + 64_000, breathsPerMinute: null },
    { timestamp: timestamp + 184_000, breathsPerMinute: 15 },
  ]
  const metrics = resolveSleepMetrics(
    { avgBreath: 16.1 },
    { ...garmin, utcOffsetMinutes: -240, respiration },
  )
  assert.equal(metrics?.averageBreathsPerMinute, 16.1)
  assert.equal(isSleepMetrics(JSON.parse(JSON.stringify(metrics)), date), true)
  assert.deepEqual(metrics?.garmin?.respiration, respiration)
  for (const invalid of [
    [{ timestamp, breathsPerMinute: 0 }],
    [{ timestamp: Number.NaN, breathsPerMinute: 15 }],
    [{ timestamp, breathsPerMinute: '15' }],
    [...respiration, respiration[0]],
    respiration.toReversed(),
  ])
    assert.equal(
      isSleepMetrics({ ...metrics, garmin: { ...garmin, respiration: invalid } }, date),
      false,
    )
  assert.equal(
    isSleepMetrics({ ...metrics, garmin: { ...garmin, utcOffsetMinutes: 0.5 } }, date),
    false,
  )
})
