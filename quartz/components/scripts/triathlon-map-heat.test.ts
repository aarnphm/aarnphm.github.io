import assert from 'node:assert/strict'
import test from 'node:test'
import type { MapRoutePoint } from '../../util/triathlon-map-route'
import {
  streetMapFeatureCollection,
  type StreetMapActivity,
  type StreetMetricValues,
} from './triathlon-map-heat'

const route = (coordinates: [number, number][]): MapRoutePoint[] =>
  coordinates.map(([lng, lat], index) => ({ lng, lat, d: index / 10 }))

const unavailableMetrics = (): StreetMetricValues => ({ w: -1, hr: -1, cad: -1, spd: -1 })

const activity = (
  id: number,
  segments: MapRoutePoint[][],
  metrics: StreetMetricValues = unavailableMetrics(),
): StreetMapActivity => ({ id, segments, metrics })

const road = (coordinates: [number, number][], roadClass = 'street'): Record<string, unknown> => ({
  type: 'Feature',
  properties: { class: roadClass },
  geometry: { type: 'LineString', coordinates },
})

test('heat geometry follows the street centerline instead of the offset GPS trace', () => {
  const heat = streetMapFeatureCollection(
    [
      road([
        [-79, 43],
        [-78.999, 43],
      ]),
    ],
    [
      activity(1, [
        route([
          [-79, 43.00009],
          [-78.999, 43.00009],
        ]),
      ]),
    ],
    1,
  )

  assert.ok(heat.features.length > 0)
  assert.ok(
    heat.features.every(feature =>
      feature.geometry.coordinates.every(([, latitude]) => latitude === 43),
    ),
  )
})

test('disconnected GPS segments do not highlight the street between them', () => {
  const heat = streetMapFeatureCollection(
    [
      road([
        [-79, 43],
        [-78.998, 43],
      ]),
    ],
    [
      activity(1, [
        route([
          [-79, 43],
          [-78.9996, 43],
        ]),
        route([
          [-78.9984, 43],
          [-78.998, 43],
        ]),
      ]),
    ],
    1,
  )

  assert.ok(heat.features.length > 0)
  assert.ok(
    heat.features.every(feature => {
      const midpoint = (feature.geometry.coordinates[0][0] + feature.geometry.coordinates[1][0]) / 2
      return midpoint < -78.9994 || midpoint > -78.9986
    }),
  )
})

test('cross-street GPS edges do not color a directionally incompatible street', () => {
  const heat = streetMapFeatureCollection(
    [
      road([
        [-79, 43],
        [-78.999, 43],
      ]),
    ],
    [
      activity(1, [
        route([
          [-78.9995, 42.9995],
          [-78.9995, 43.0005],
        ]),
      ]),
    ],
    1,
  )

  assert.deepEqual(heat.features, [])
})

test('heat density counts activities once per street piece', () => {
  const sharedRoad = road([
    [-79, 43],
    [-78.9995, 43],
  ])
  const sharedRoute = route([
    [-79, 43.00005],
    [-78.9995, 43.00005],
  ])
  const heat = streetMapFeatureCollection(
    [sharedRoad],
    [activity(1, [sharedRoute]), activity(2, [sharedRoute, sharedRoute])],
    2,
  )

  assert.ok(heat.features.length > 0)
  assert.ok(
    heat.features.every(
      feature => feature.properties.visits === 2 && feature.properties.heat === 7,
    ),
  )
})

test('rail, ferry, aerialway, and golf geometries never become heat streets', () => {
  const trace = activity(1, [
    route([
      [-79, 43],
      [-78.999, 43],
    ]),
  ])
  const heat = streetMapFeatureCollection(
    [
      road(
        [
          [-79, 43],
          [-78.999, 43],
        ],
        'major_rail',
      ),
      road(
        [
          [-79, 43],
          [-78.999, 43],
        ],
        'ferry',
      ),
      road(
        [
          [-79, 43],
          [-78.999, 43],
        ],
        'aerialway',
      ),
      road(
        [
          [-79, 43],
          [-78.999, 43],
        ],
        'golf',
      ),
    ],
    [trace],
    1,
  )

  assert.deepEqual(heat.features, [])
})

test('street metrics average the available values of visiting activities', () => {
  const sharedRoad = road([
    [-79, 43],
    [-78.9995, 43],
  ])
  const sharedRoute = route([
    [-79, 43.00005],
    [-78.9995, 43.00005],
  ])
  const streets = streetMapFeatureCollection(
    [sharedRoad],
    [
      activity(1, [sharedRoute], { w: 0.2, hr: 0.3, cad: -1, spd: 0.4 }),
      activity(2, [sharedRoute], { w: 0.8, hr: -1, cad: -1, spd: 0.6 }),
    ],
    2,
  )

  assert.ok(streets.features.length > 0)
  assert.ok(
    streets.features.every(
      feature =>
        feature.properties.w === 0.5 &&
        feature.properties.hr === 0.3 &&
        feature.properties.cad === -1 &&
        feature.properties.spd === 0.5,
    ),
  )
})
