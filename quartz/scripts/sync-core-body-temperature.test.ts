import assert from 'node:assert/strict'
import test from 'node:test'
import {
  coreCloudDataUrl,
  coreCloudRefreshStart,
  coreCloudWindows,
  coreBodyTemperatureImportCandidates,
} from './sync-core-body-temperature'

test('coreBodyTemperatureImportCandidates prefers an explicit export', () => {
  assert.deepEqual(coreBodyTemperatureImportCandidates('~/Downloads/core.csv', '/Users/test'), [
    '/Users/test/Downloads/core.csv',
  ])
})

test('coreCloudWindows covers the requested range without overlapping API windows', () => {
  assert.deepEqual(coreCloudWindows('2026-07-01T00:00:00.000Z', '2026-07-03T00:00:00.000Z', 2), [
    { start: '2026-07-01T00:00:00.000Z', end: '2026-07-02T23:59:59.999Z' },
    { start: '2026-07-03T00:00:00.000Z', end: '2026-07-03T00:00:00.000Z' },
  ])
})

test('coreCloudDataUrl encodes the first-party device data contract', () => {
  assert.equal(
    coreCloudDataUrl('sensor/1', {
      start: '2026-07-29T00:00:00.000Z',
      end: '2026-07-30T00:00:00.000Z',
    }),
    'https://core-api.corebodytemp.com/api/v1/devices/device-data/sensor%2F1/?startGte=2026-07-29T00%3A00%3A00.000Z&startLte=2026-07-30T00%3A00%3A00.000Z',
  )
})

test('coreCloudRefreshStart overlaps the latest cached sample by SYNC_REFRESH_DAYS', () => {
  const previous = {
    version: 1,
    lastSync: 1,
    samples: [
      {
        time: '2026-10-02T12:00:00.000Z',
        coreTemperatureC: 37.4,
        skinTemperatureC: 33.1,
        heatStrainIndex: null,
        quality: 3,
        heartRate: null,
      },
    ],
  }
  assert.equal(coreCloudRefreshStart(previous, '2026-05-15', 2), '2026-09-30T12:00:00.000Z')
  assert.equal(coreCloudRefreshStart(null, '2026-05-15', 2), '2026-05-15T00:00:00.000Z')
})
