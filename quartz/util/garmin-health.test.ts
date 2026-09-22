import assert from 'node:assert/strict'
import test from 'node:test'
import { latestGarminReadiness, morningGarminReadiness } from '../plugins/stores/garmin-health'
import { raw, date, fetchedAt, health } from './fixtures/garmin-health'
import {
  garminBodyBattery,
  garminTrainingReadiness,
  garminTrainingStatus,
  garminEnduranceScore,
  garminHillScore,
  garminHealthFetchResult,
  garminHealthFetchError,
  isGarminHealthDay,
} from './garmin-health'
import { isRecord } from './type-guards'
assert.ok(isRecord(raw))

test('Garmin health parses actual daily reports with timestamps, native units and device-aligned load', () => {
  const battery = health.bodyBattery.value
  assert.equal(battery?.samples.length, 6)
  assert.equal(battery?.samples.at(-1)?.value, 92)
  assert.equal(battery?.utcOffsetMinutes, -240)
  assert.equal(battery?.charged, 57)
  assert.equal(battery?.drained, 8)
  assert.equal(latestGarminReadiness(health)?.score, 78)
  assert.equal(latestGarminReadiness(health)?.recoveryTimeMinutes, 372)
  assert.equal(latestGarminReadiness(health)?.timestamp, Date.parse('2026-09-22T00:28:04Z'))
  assert.equal(morningGarminReadiness(health)?.score, 72)
  assert.equal(health.trainingStatus.value?.acuteLoad, 839)
  assert.equal(health.trainingStatus.value?.chronicLoad, 789)
  assert.equal(health.trainingStatus.value?.loadRatio, 1)
  assert.equal(health.trainingStatus.value?.loadFocus?.categories[0].targetMin, 674)
  assert.equal(health.enduranceScore.value?.score, 7918)
  assert.equal(
    health.enduranceScore.value?.thresholds.find(row => row.label === 'expert')?.lower,
    7300,
  )
  assert.equal(health.hillScore.value?.score, 37)
  assert.equal(isGarminHealthDay(JSON.parse(JSON.stringify(health)), date), true)
})

test('Garmin recovery preserves numeric minutes when a feedback phrase contradicts the snapshot', () => {
  const postExercise = health.trainingReadiness.value?.find(
    row => row.inputContext === 'AFTER_POST_EXERCISE_RESET',
  )
  assert.equal(postExercise?.recoveryTimeChange, 'REACHED_ZERO')
  assert.equal(postExercise?.recoveryTimeMinutes, 932)
})

test('Garmin Body Battery honors descriptor columns, zero values and missing sample gaps', () => {
  const parsed = garminBodyBattery(
    [
      {
        date,
        charged: 0,
        drained: 0,
        bodyBatteryValueDescriptorDTOList: [
          { bodyBatteryValueDescriptorIndex: 1, bodyBatteryValueDescriptorKey: 'timestamp' },
          { bodyBatteryValueDescriptorIndex: 0, bodyBatteryValueDescriptorKey: 'bodyBatteryLevel' },
        ],
        bodyBatteryValuesArray: [
          [0, 1000000000000],
          [-1, 1000000001000],
          [75, 1000000002000],
        ],
      },
    ],
    date,
  )
  assert.deepEqual(
    parsed?.samples.map(row => row.value),
    [0, null, 75],
  )
  assert.equal(parsed?.charged, 0)
  assert.equal(garminBodyBattery(raw.bodyBattery, '2026-09-20'), null)
  assert.throws(() =>
    garminBodyBattery(
      { date, bodyBatteryValueDescriptorDTOList: [{ bodyBatteryValueDescriptorKey: 'unknown' }] },
      date,
    ),
  )
})

test('Garmin status retains the observation date and matches load focus to the selected device', () => {
  const parsed = garminTrainingStatus(
    {
      mostRecentTrainingStatus: {
        latestTrainingStatusData: {
          secondary: { calendarDate: date, trainingStatus: 1 },
          primary: { calendarDate: '2026-09-20', primaryTrainingDevice: true, trainingStatus: 4 },
        },
      },
      mostRecentTrainingLoadBalance: {
        metricsTrainingLoadBalanceDTOMap: {
          secondary: { calendarDate: date, monthlyLoadAerobicLow: 999 },
          primary: { calendarDate: '2026-09-20', monthlyLoadAerobicLow: 123 },
        },
      },
    },
    date,
  )
  assert.equal(parsed?.date, '2026-09-20')
  assert.equal(parsed?.status, 4)
  assert.equal(parsed?.loadFocus?.categories[0].load, 123)
  assert.equal(garminEnduranceScore(raw.endurance, '2026-09-20'), null)
  assert.equal(garminHillScore(raw.hill, '2026-09-20'), null)
})

test('Garmin health refresh distinguishes unavailable data from failed requests and preserves prior evidence', () => {
  const failed = garminHealthFetchError(health.bodyBattery, fetchedAt + 1000)
  assert.equal(failed.status, 'error')
  assert.equal(failed.fetchedAt, fetchedAt)
  assert.deepEqual(failed.value, health.bodyBattery.value)
  assert.equal(garminHealthFetchError(undefined, fetchedAt).value, null)
  assert.deepEqual(garminHealthFetchResult(null, fetchedAt), {
    status: 'unavailable',
    fetchedAt,
    attemptedAt: fetchedAt,
    value: null,
  })
  assert.equal(garminTrainingReadiness([], date), null)
  assert.equal(garminTrainingStatus({}, date), null)
  assert.throws(() => garminHillScore('invalid', date))
  assert.equal(isGarminHealthDay({ ...health, date: '2026-09-20' }, date), false)
  assert.equal(
    isGarminHealthDay(
      {
        ...health,
        trainingReadiness: { ...health.trainingReadiness, value: [{ date, score: '78' }] },
      },
      date,
    ),
    false,
  )
})
