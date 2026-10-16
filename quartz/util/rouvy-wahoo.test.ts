import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity } from '../plugins/stores/strava'
import { emptyWahooMetrics, type WahooActivity } from '../plugins/stores/wahoo'
import { planRouvyWahooPairs, isWahooStravaDuplicate } from './rouvy-wahoo'

const rouvy: RawStravaActivity = {
  id: 20611099208,
  name: 'ROUVY - Overgear',
  deviceName: 'Rouvy',
  sportType: 'VirtualRide',
  startDate: '2026-10-16T15:42:03Z',
  startDateLocal: '2026-10-16T11:42:03Z',
  elapsedTime: 4609,
  movingTime: 4507,
  distance: 33792.7,
  totalElevationGain: 402,
  averageSpeed: 7.498,
}
const wahoo: WahooActivity = {
  id: 'wahoo:506980247',
  workoutId: 506980247,
  workoutTypeId: 0,
  name: 'Virtual Cycling',
  sport: 'bike',
  startDate: '2026-10-16T15:42:02Z',
  startDateLocal: '2026-10-16T11:42:02Z',
  elapsedTimeS: 4610,
  movingTimeS: 4610,
  distanceM: 37920.3,
  sourceDevice: 'ELEMNT BOLT',
  sourceFile: {
    url: 'https://example.invalid/original.fit',
    sha256: 'a'.repeat(64),
    byteLength: 1,
    profileVersion: '1',
  },
  metrics: emptyWahooMetrics(),
  sweatLoss: { fluidMl: null, sodiumMg: null },
  summary: {
    id: 1,
    name: null,
    timeZone: null,
    manual: false,
    edited: false,
    fitnessAppId: 1,
    durationPausedS: null,
    createdAt: null,
    updatedAt: null,
  },
}
const duplicate: RawStravaActivity = {
  ...rouvy,
  id: 20611106109,
  deviceName: 'Wahoo ELEMNT BOLT 3',
  name: 'Indoor Cycling',
  sportType: 'Ride',
  startDate: wahoo.startDate,
  elapsedTime: 4610,
  distance: 37920.3,
}

test('pairs virtual route and native recording despite their different distances', () => {
  const pairs = planRouvyWahooPairs([rouvy, duplicate], [wahoo], '2026-10-16')
  assert.equal(pairs.length, 1)
  assert.equal(pairs[0].canonical.id, rouvy.id)
  assert.equal(pairs[0].duplicate?.id, duplicate.id)
})
test('requires unique matches in both directions', () => {
  assert.deepEqual(planRouvyWahooPairs([rouvy, { ...rouvy, id: 2 }], [wahoo], '2026-10-16'), [])
  assert.deepEqual(
    planRouvyWahooPairs([rouvy], [wahoo, { ...wahoo, id: 'wahoo:2', workoutId: 2 }], '2026-10-16'),
    [],
  )
})
test('rejects short coverage, adjacent recordings and unknown application identity', () => {
  for (const source of [
    { ...wahoo, elapsedTimeS: 600 },
    { ...wahoo, startDate: '2026-10-16T17:00:00Z' },
  ])
    assert.deepEqual(planRouvyWahooPairs([rouvy], [source], '2026-10-16'), [])
  assert.deepEqual(
    planRouvyWahooPairs([{ ...rouvy, deviceName: undefined }], [wahoo], '2026-10-16'),
    [],
  )
  assert.deepEqual(planRouvyWahooPairs([rouvy], [wahoo], '2026-10-17'), [])
})
test('duplicate suppression requires Wahoo device identity and matching recording geometry', () => {
  assert.equal(isWahooStravaDuplicate(duplicate, wahoo), true)
  assert.equal(isWahooStravaDuplicate({ ...duplicate, deviceName: 'Garmin' }, wahoo), false)
  assert.equal(isWahooStravaDuplicate({ ...duplicate, distance: 1000 }, wahoo), false)
  assert.equal(isWahooStravaDuplicate({ ...duplicate, elapsedTime: 600 }, wahoo), false)
})
test('ambiguous Strava duplicates are never selected for cleanup', () => {
  const pairs = planRouvyWahooPairs(
    [rouvy, duplicate, { ...duplicate, id: 3 }],
    [wahoo],
    '2026-10-16',
  )
  assert.equal(pairs.length, 1)
  assert.equal(pairs[0].duplicate, null)
})
test('pair remains importable after Strava duplicate is removed', () => {
  const pairs = planRouvyWahooPairs([rouvy], [wahoo], '2026-10-16')
  assert.equal(pairs.length, 1)
  assert.equal(pairs[0].duplicate, null)
})
