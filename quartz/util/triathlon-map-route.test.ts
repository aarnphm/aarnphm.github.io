import assert from 'node:assert/strict'
import test from 'node:test'
import {
  clipMapRoute,
  mapRoutePointAtDistance,
  rawMapRouteSegments,
  simplifyMapRoute,
  type MapRoutePoint,
} from './triathlon-map-route'

test('geometry simplification keeps road-scale turns', () => {
  const route: MapRoutePoint[] = [
    { lat: 43, lng: -79, d: 0 },
    { lat: 43, lng: -78.9995, d: 0.04 },
    { lat: 43, lng: -78.999, d: 0.08 },
    { lat: 43.0005, lng: -78.999, d: 0.14 },
    { lat: 43.001, lng: -78.999, d: 0.2 },
    { lat: 43.001, lng: -78.9995, d: 0.24 },
    { lat: 43.001, lng: -79, d: 0.28 },
  ]

  assert.deepEqual(simplifyMapRoute(route, 2), [route[0], route[2], route[4], route[6]])
})

test('raw route projection keeps plausible pauses connected', () => {
  const latlng: [number, number][] = [
    [43, -79],
    [43, -78.9999],
    [43, -78.9998],
    [43.001, -78.9998],
    [43.001, -78.9997],
    [43.001, -78.9996],
  ]
  const segments = rawMapRouteSegments(
    latlng,
    [0, 10, 20, 120, 130, 140],
    [0, 1, 2, 90, 91, 92],
    0,
    5,
  )

  assert.equal(segments.length, 1)
  assert.deepEqual([segments[0][0].d, segments[0].at(-1)?.d], [0, 0.14])
})

test('raw route projection separates impossible GPS teleports', () => {
  const segments = rawMapRouteSegments(
    [
      [43, -79],
      [43.0001, -79],
      [44, -78],
      [44.0001, -78],
    ],
    [0, 10, 20, 30],
    [0, 1, 2, 3],
    0,
    3,
  )

  assert.equal(segments.length, 2)
  assert.deepEqual(
    segments.map(segment => [segment[0].d, segment.at(-1)?.d]),
    [
      [0, 0.01],
      [0.02, 0.03],
    ],
  )
})

test('distance clipping and hover interpolation use the display geometry', () => {
  const route: MapRoutePoint[][] = [
    [
      { lat: 43, lng: -79, d: 0 },
      { lat: 43, lng: -78.99, d: 1 },
      { lat: 43.01, lng: -78.99, d: 2 },
    ],
  ]
  const clipped = clipMapRoute(route, 0.25, 1.5)

  assert.equal(clipped.length, 1)
  assert.deepEqual(
    clipped[0].map(point => point.d),
    [0.25, 1, 1.5],
  )
  assert.deepEqual(mapRoutePointAtDistance(route, 1.5), { lat: 43.005, lng: -78.99, d: 1.5 })
})
