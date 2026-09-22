import assert from 'node:assert/strict'
import test from 'node:test'
import type { OuraDaily, OuraDayDetail } from '../plugins/stores/oura'
import { emptyOuraDaily } from '../plugins/stores/oura'
import { emptyOuraHealth } from '../util/oura-health'
import { applyOuraSleepRows, ouraRefreshRange } from './sync-oura'

const NOW = Date.parse('2026-09-01T16:00:00Z')

test('Oura routine refresh uses the shared inclusive calendar window', () => {
  assert.deepEqual(ouraRefreshRange(false, 3, NOW), {
    start: '2026-08-29',
    end: '2026-09-01',
    endExclusive: '2026-09-02',
    heartRateStart: '2026-08-29',
  })
})

test('Oura schema refresh keeps the full daily backfill and bounded heart rate', () => {
  assert.deepEqual(ouraRefreshRange(true, 3, NOW), {
    start: '2025-09-01',
    end: '2026-09-01',
    endExclusive: '2026-09-02',
    heartRateStart: '2026-08-29',
  })
})

const sleepRow = (id: string, type: string, start: string, seconds: number) => ({
  id,
  type,
  day: '2026-09-21',
  bedtime_start: start,
  bedtime_end: start.slice(0, 11) + '15:00:00-04:00',
  total_sleep_duration: seconds,
  time_in_bed: seconds + 300,
  sleep_phase_5_min: '421234',
  average_hrv: 52,
  lowest_heart_rate: 48,
  heart_rate: { timestamp: start, interval: 300, items: [55, null, 53] },
  hrv: { timestamp: start, interval: 300, items: [52, null, 54] },
})

test('retains multiple confirmed naps and late-nap score dates without changing night metrics', () => {
  const date = '2026-09-21'
  const days: Record<string, OuraDaily> = {
    [date]: { ...emptyOuraDaily(date), sleepScore: 83, readiness: 81 },
  }
  const details: Record<string, OuraDayDetail> = {}
  const night = {
    ...sleepRow('night', 'long_sleep', `${date}T01:00:00-04:00`, 24000),
    average_hrv: 68,
  }
  const nap = { ...sleepRow('nap', 'sleep', `${date}T14:00:00-04:00`, 1800), average_heart_rate: 0 }
  const late = {
    ...sleepRow('late', 'late_nap', `${date}T19:00:00-04:00`, 1200),
    bedtime_end: `${date}T19:25:00-04:00`,
    day: '2026-09-22',
    sleep_score_delta: 0,
    readiness_score_delta: 2,
  }
  applyOuraSleepRows(
    [
      late,
      night,
      nap,
      nap,
      sleepRow('rest', 'rest', `${date}T12:00:00-04:00`, 2000),
      sleepRow('deleted', 'deleted', `${date}T12:00:00-04:00`, 2000),
    ],
    days,
    details,
    date,
    date,
  )
  assert.equal(days[date].sleepDurationS, 24000)
  assert.equal(days[date].hrv, 68)
  assert.equal(days[date].sleepScore, 83)
  assert.equal(days[date].readiness, 81)
  assert.deepEqual(
    details[date].naps?.map(n => n.id),
    ['nap', 'late'],
  )
  assert.equal(details[date].naps?.[1].reportedDay, '2026-09-22')
  assert.equal(details[date].naps?.[1].sleepScoreDelta, 0)
  assert.equal(details[date].naps?.[1].readinessScoreDelta, 2)
  assert.equal(details[date].naps?.[0].sleepScore, null)
  assert.equal(details[date].naps?.[0].avgHr, null)
  assert.deepEqual(details[date].naps?.[0].hr?.items, [55, null, 53])
})

test('refresh replaces naps, retains out-of-window history, and supports a nap-only day', () => {
  const days: Record<string, OuraDaily> = {}
  const details: Record<string, OuraDayDetail> = {}
  applyOuraSleepRows(
    [
      sleepRow('a', 'sleep', '2026-09-20T14:00:00-04:00', 1800),
      sleepRow('b', 'sleep', '2026-09-21T14:00:00-04:00', 1800),
    ],
    days,
    details,
    '2026-09-20',
    '2026-09-21',
  )
  assert.equal(details['2026-09-21'].totalSleepS, null)
  assert.equal(days['2026-09-21'].sleepDurationS, null)
  applyOuraSleepRows([], days, details, '2026-09-21', '2026-09-21')
  assert.equal(details['2026-09-20'].naps?.length, 1)
  assert.deepEqual(details['2026-09-21'].naps, [])
})

test('rejects invalid naps and retains the longest main sleep independently', () => {
  const date = '2026-09-21'
  const days: Record<string, OuraDaily> = {}
  const details: Record<string, OuraDayDetail> = {}
  const valid = sleepRow('ok', 'sleep', `${date}T14:00:00-04:00`, 1800)
  applyOuraSleepRows(
    [
      { ...valid, id: 'zero', total_sleep_duration: 0 },
      { ...valid, id: 'invalid', bedtime_end: 'invalid' },
      { ...valid, id: 'reverse', bedtime_end: `${date}T13:00:00-04:00` },
      { ...valid, id: 'unknown', type: 'future_type' },
      sleepRow('longer', 'long_sleep', `${date}T01:00:00-04:00`, 24000),
      sleepRow('shorter', 'long_sleep', `${date}T09:00:00-04:00`, 12000),
    ],
    days,
    details,
    date,
    date,
  )
  assert.deepEqual(details[date].naps, [])
  assert.equal(days[date].sleepDurationS, 24000)
})

test('refresh retains health and records fine sleep stages, movement and recording metadata', () => {
  const date = '2026-09-21'
  const details: Record<string, OuraDayDetail> = {}
  const days: Record<string, OuraDaily> = {}
  const row = {
    ...sleepRow('night', 'long_sleep', `${date}T01:00:00-04:00`, 24000),
    sleep_phase_30_sec: '44332211',
    movement_30_sec: '11112341',
    low_battery_alert: false,
    sleep_algorithm_version: 'v2',
  }
  applyOuraSleepRows([row], days, details, date, date)
  details[date].health = {
    ...emptyOuraHealth(date),
    stress: { stressS: 0, restoredS: 3600, summary: 'restored' },
  }
  applyOuraSleepRows([row], days, details, date, date)
  assert.equal(details[date].phase30Sec, '44332211')
  assert.equal(details[date].movement30Sec, '11112341')
  assert.equal(details[date].lowBatteryAlert, false)
  assert.equal(details[date].sleepAlgorithmVersion, 'v2')
  assert.equal(details[date].health?.stress?.restoredS, 3600)
})
