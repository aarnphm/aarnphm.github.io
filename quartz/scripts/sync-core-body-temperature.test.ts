import assert from 'node:assert/strict'
import test from 'node:test'
import {
  coreCloudDataUrl,
  coreCloudWindows,
  coreBodyTemperatureICloudPath,
  coreBodyTemperatureImportCandidates,
  expandCoreBodyTemperaturePath,
} from './sync-core-body-temperature'

test('coreBodyTemperatureImportCandidates prefers an explicit export', () => {
  assert.deepEqual(coreBodyTemperatureImportCandidates('~/Downloads/core.csv', '/Users/test'), [
    '/Users/test/Downloads/core.csv',
  ])
})

test('coreBodyTemperatureImportCandidates checks iCloud and the cache import boundary', () => {
  assert.deepEqual(coreBodyTemperatureImportCandidates(undefined, '/Users/test'), [
    '/Users/test/Library/Mobile Documents/com~apple~CloudDocs/CORE/core-body-temperature.csv',
    'quartz/.quartz-cache/core-body-temperature-import.csv',
  ])
})

test('expandCoreBodyTemperaturePath resolves the documented iCloud destination', () => {
  assert.equal(
    expandCoreBodyTemperaturePath('iCloud Drive/CORE/core-body-temperature.csv', '/Users/test'),
    coreBodyTemperatureICloudPath('/Users/test'),
  )
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
