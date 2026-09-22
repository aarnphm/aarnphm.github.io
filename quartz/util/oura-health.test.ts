import assert from 'node:assert/strict'
import test from 'node:test'
import type { OuraDayDetail, OuraDaily } from '../plugins/stores/oura'
import { applyOuraSleepRows } from '../scripts/sync-oura'
import {
  applyOuraHealthRow,
  emptyOuraHealth,
  isOuraHealthDay,
  ouraRestorationBaseline,
} from './oura-health'

const date = '2026-09-21'

test('Oura collections retain native units, measured zero, nulls and valid scores', () => {
  const day = emptyOuraHealth(date)
  applyOuraHealthRow(day, 'daily_stress', {
    stress_high: 0,
    recovery_high: 5400,
    day_summary: 'restored',
  })
  applyOuraHealthRow(day, 'daily_resilience', {
    level: 'solid',
    contributors: { sleep_recovery: 61, daytime_recovery: 0, stress: 101 },
  })
  applyOuraHealthRow(day, 'daily_spo2', {
    spo2_percentage: { average: 96.5 },
    breathing_disturbance_index: 0,
  })
  applyOuraHealthRow(day, 'daily_activity', {
    steps: 0,
    active_calories: 0,
    target_calories: 0,
    resting_time: 21600,
    non_wear_time: 0,
    contributors: { stay_active: 90 },
  })
  applyOuraHealthRow(day, 'daily_readiness', { temperature_trend_deviation: -0.12 })
  applyOuraHealthRow(day, 'daily_cardiovascular_age', {
    vascular_age: 24,
    pulse_wave_velocity: 5.98,
  })
  applyOuraHealthRow(day, 'vO2_max', { vo2_max: 55 })
  applyOuraHealthRow(day, 'sleep_time', {
    optimal_bedtime: { start_offset: -1800, end_offset: 2700, day_tz: -14400 },
    recommendation: 'follow_optimal_bedtime',
  })
  assert.deepEqual(day.stress, { stressS: 0, restoredS: 5400, summary: 'restored' })
  assert.deepEqual(day.resilience, {
    level: 'solid',
    sleepRecovery: 61,
    daytimeRecovery: 0,
    stress: null,
  })
  assert.equal(day.activity?.targetCalories, null)
  assert.equal(day.activity?.steps, 0)
  assert.equal(day.spo2?.breathingDisturbanceIndex, 0)
  assert.equal(day.temperatureTrendC, -0.12)
  assert.equal(day.sleepTime?.startOffsetS, -1800)
  assert.equal(isOuraHealthDay(JSON.parse(JSON.stringify(day)), date), true)
  assert.equal(isOuraHealthDay(day, '2026-09-20'), false)
  assert.equal(isOuraHealthDay({ ...day, failedCollections: [1] }, date), false)
  assert.equal(
    isOuraHealthDay({ ...day, activity: { ...day.activity, steps: '100' } }, date),
    false,
  )
  applyOuraHealthRow(day, 'daily_stress', {})
  assert.equal(day.stress?.restoredS, null)
  assert.equal(day.vo2Max, 55)
})

test('restoration baseline uses only available preceding 14 days, includes zero, and requires three days', () => {
  const details: Record<string, OuraDayDetail> = {}
  const days: Record<string, OuraDaily> = {}
  const rows = ['2026-09-01', '2026-09-18', '2026-09-19', '2026-09-20', date, '2026-09-22'].map(
    day => ({
      id: day,
      type: 'long_sleep',
      day,
      bedtime_start: `${day}T01:00:00Z`,
      total_sleep_duration: 24000,
    }),
  )
  applyOuraSleepRows(rows, days, details, '2026-09-01', '2026-09-22')
  for (const [day, detail] of Object.entries(details))
    detail.health = {
      ...emptyOuraHealth(day),
      stress: { stressS: null, restoredS: day === '2026-09-18' ? 0 : 3600, summary: null },
    }
  assert.deepEqual(ouraRestorationBaseline(date, details), { seconds: 2400, days: 3 })
  delete details['2026-09-18']
  assert.equal(ouraRestorationBaseline(date, details), null)
})
