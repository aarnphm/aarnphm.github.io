import assert from 'node:assert/strict'
import test from 'node:test'
import { clusterMapDensity } from './base-map-density'

const group = (lon: number, lat = 43.65) =>
  [0, 0.0001, 0.0002].map(offset => ({ lon: lon + offset, lat }))

test('density groups nearby places and excludes isolated places', () => {
  const points = [...group(-79.4), ...group(-79.3), { lon: -78, lat: 43.65 }]
  const result = clusterMapDensity(points, 13)
  assert.equal(result.features.length, 2)
  assert.deepEqual(
    result.features.map(item => item.properties.count),
    [3, 3],
  )
  assert.deepEqual(clusterMapDensity([...points].reverse(), 13), result)
  assert.deepEqual(clusterMapDensity([], 13).features, [])
  assert.deepEqual(clusterMapDensity(points.slice(-1), 13).features, [])
})

test('coincident places terminate and dateline neighbors remain together', () => {
  assert.equal(
    clusterMapDensity(
      Array.from({ length: 30 }, () => ({ lon: 10, lat: 90 })),
      13,
    ).features[0]?.properties.count,
    30,
  )
  const result = clusterMapDensity(
    [
      { lon: 179.9999, lat: 0 },
      { lon: -179.9999, lat: 0 },
      { lon: 180, lat: 0 },
    ],
    13,
  )
  assert.equal(result.features.length, 1)
  assert.ok(Math.abs(result.features[0].geometry.coordinates[0]) > 179)
  assert.ok(result.features[0].geometry.coordinates.every(Number.isFinite))
})

test('zoom separates neighborhoods and filters remove their density', () => {
  const points = [
    { lon: -79.4, lat: 43.65 },
    { lon: -79.398, lat: 43.65 },
    { lon: -79.396, lat: 43.65 },
  ]
  assert.equal(clusterMapDensity(points, 11).features.length, 1)
  assert.equal(clusterMapDensity(points, 17).features.length, 0)
  assert.equal(clusterMapDensity(points.slice(0, 2), 11).features.length, 0)
})
