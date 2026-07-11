import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity } from '../plugins/stores/strava'
import { reconcileStravaActivities, stravaFetchAfter } from './strava-sync-window'

const activity = (id: number, startDate: string, calories?: number): RawStravaActivity => ({
  id,
  name: `activity ${id}`,
  sportType: 'Run',
  distance: 5000,
  movingTime: 1500,
  elapsedTime: 1600,
  totalElevationGain: 10,
  startDate,
  startDateLocal: startDate,
  averageSpeed: 3.33,
  calories,
})

test('stravaFetchAfter refreshes a bounded recent overlap', () => {
  const last = Math.floor(Date.parse('2026-07-07T14:01:53Z') / 1000)

  assert.equal(stravaFetchAfter(last, false, 7), last - 7 * 86_400)
})

test('stravaFetchAfter preserves full refresh and empty-cache behavior', () => {
  assert.equal(stravaFetchAfter(123, true, 7), 0)
  assert.equal(stravaFetchAfter(0, false, 7), 0)
  assert.equal(stravaFetchAfter(undefined, false, 7), 0)
  assert.equal(stravaFetchAfter(123, false, 0), 123)
})

test('reconcileStravaActivities prunes deleted overlap rows and preserves older history', () => {
  const after = Math.floor(Date.parse('2026-06-01T00:00:00Z') / 1000)
  const old = activity(1, '2026-05-01T12:00:00Z')
  const deleted = activity(2, '2026-06-02T12:00:00Z')
  const refreshed = activity(3, '2026-06-03T12:00:00Z')
  const result = reconcileStravaActivities(
    { '1': old, '2': deleted, '3': activity(3, refreshed.startDate, 640) },
    [refreshed],
    after,
  )

  assert.deepEqual(Object.keys(result.activities).sort(), ['1', '3'])
  assert.equal(result.activities['3'].calories, 640)
  assert.deepEqual(result.removedIds, ['2'])
})

test('reconcileStravaActivities mirrors a full refresh and retains distinct fetched ids', () => {
  const first = activity(4, '2026-06-04T12:00:00Z')
  const second = { ...first, id: 5 }
  const result = reconcileStravaActivities(
    { '1': activity(1, '2026-05-01T12:00:00Z') },
    [first, second],
    0,
  )

  assert.deepEqual(Object.keys(result.activities).sort(), ['4', '5'])
  assert.deepEqual(result.removedIds, ['1'])
})
