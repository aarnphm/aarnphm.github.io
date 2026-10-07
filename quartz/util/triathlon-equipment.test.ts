import assert from 'node:assert/strict'
import test from 'node:test'
import type { RawStravaActivity, StravaRawCache } from '../plugins/stores/strava'
import { buildTriathlonEquipment, formatEquipmentDistance } from './triathlon-equipment'

const activity = (id: number, gearId: string | null, date: string): RawStravaActivity => ({
  id,
  gearId,
  name: 'Recorded activity',
  sportType: 'Ride',
  distance: 500,
  movingTime: 100,
  elapsedTime: 100,
  totalElevationGain: 0,
  startDate: date,
  startDateLocal: date,
  averageSpeed: 5,
})

test('uses native lifetime distance and joins activity history by stable gear ID', () => {
  const cache: Pick<StravaRawCache, 'activities' | 'gear'> = {
    gear: {
      b1: {
        id: 'b1',
        name: 'Renamed Soloist',
        brandName: null,
        modelName: null,
        description: 'Ultegra Di2, 2025',
        distanceM: 100_000,
      },
      b2: { id: 'b2', name: 'Speedmax', brandName: null, modelName: null, distanceM: 0 },
      g1: { id: 'g1', name: 'Shoes', brandName: null, modelName: null, distanceM: null },
    },
    activities: {
      1: activity(1, 'b1', '2026-09-15T00:15:00Z'),
      2: activity(2, 'b1', '2026-05-26T10:00:00Z'),
      3: activity(3, null, '2026-04-01T10:00:00Z'),
      4: { ...activity(4, 'g1', '2026-09-11T10:00:00Z'), sportType: 'Run' },
    },
  }
  assert.deepEqual(buildTriathlonEquipment(cache), {
    b1: {
      id: 'b1',
      name: 'Renamed Soloist',
      description: 'Ultegra Di2, 2025',
      lifetimeDistanceM: 100_000,
      activityCount: 2,
      firstRecorded: '2026-05-26',
      lastRecorded: '2026-09-15',
      source: 'strava',
    },
    b2: {
      id: 'b2',
      name: 'Speedmax',
      description: null,
      lifetimeDistanceM: 0,
      activityCount: 0,
      firstRecorded: null,
      lastRecorded: null,
      source: 'strava',
    },
    g1: {
      id: 'g1',
      name: 'Shoes',
      description: null,
      lifetimeDistanceM: null,
      activityCount: 1,
      firstRecorded: '2026-09-11',
      lastRecorded: '2026-09-11',
      source: 'strava',
    },
  })
  cache.activities['1'].gearId = 'b2'
  assert.equal(buildTriathlonEquipment(cache).b1.activityCount, 1)
  assert.equal(buildTriathlonEquipment(cache).b2.activityCount, 1)
})

test('missing gear does not fabricate lifetime totals or zero mileage', () => {
  assert.deepEqual(buildTriathlonEquipment(null), {})
  assert.deepEqual(
    buildTriathlonEquipment({ activities: { 1: activity(1, 'b1', '2026-09-15') } }),
    {},
  )
  for (const distanceM of [-1, Infinity, NaN]) {
    const equipment = buildTriathlonEquipment({
      activities: {},
      gear: { b1: { id: 'b1', name: null, brandName: null, modelName: null, distanceM } },
    })
    assert.equal(equipment.b1.lifetimeDistanceM, null)
  }
})

test('lifetime mileage formats from the same metre value in either distance system', () => {
  assert.equal(formatEquipmentDistance(3_723_142, 'imperial'), '2,313.45 mi')
  assert.equal(formatEquipmentDistance(3_723_142, 'metric'), '3,723.14 km')
  assert.equal(formatEquipmentDistance(79_577, 'imperial'), '49.45 mi')
  assert.equal(formatEquipmentDistance(0, 'metric'), '0 km')
  assert.equal(formatEquipmentDistance(0, 'imperial'), '0 mi')
})
