import assert from 'node:assert/strict'
import test from 'node:test'
import { buildPayload, type StravaMapPoint } from '../../../plugins/stores/strava'
import { mapActivity as mapStravaActivity } from '../../../scripts/sync-strava'
import { DEFAULT_TRIATHLON_PRESENTATION } from '../../../util/triathlon-presentation'
import {
  buildOverview,
  fcBounds,
  heatCasingWidthExpr,
  heatWidthExpr,
  initialMapModel,
  lineFeatures,
  pctRange,
  readOverviewMode,
  readRouteSport,
  routeFC,
  streetMetricCasingWidthExpr,
  streetMetricWidthExpr,
  updateMap,
} from './model'

const mapActivity = (id: number, sportType: string, mapRoute: StravaMapPoint[][]) => {
  const start = '2026-09-04T12:00:00Z'
  const payload = buildPayload(
    {
      athleteId: 1,
      auth: { refreshToken: '', obtainedAt: 0 },
      lastSync: Date.parse(start),
      lastActivityStart: Date.parse(start) / 1000,
      activities: {
        [id]: mapStravaActivity({
          id,
          name: sportType,
          type: sportType === 'VirtualRun' ? 'Run' : 'Ride',
          sport_type: sportType,
          distance: 1_000,
          moving_time: 300,
          elapsed_time: 300,
          total_elevation_gain: 0,
          start_date: start,
          start_date_local: start,
          average_speed: 1_000 / 300,
          average_watts: 150,
          device_watts: true,
        }),
      },
    },
    null,
    null,
  )
  return { ...payload.details[id], mapRoute }
}

test('Strava virtual ride tags exclude courses from overview bounds, visits, and metric ranges', () => {
  const outdoor = mapActivity(1, 'Ride', [
    [
      { lat: 43.7, lng: -79.4, d: 0 },
      { lat: 43.701, lng: -79.399, d: 1 },
    ],
  ])
  const virtual = mapActivity(2, 'VirtualRide', [
    [
      { lat: 45.062314, lng: 6.036319, d: 0 },
      { lat: 45.063, lng: 6.037, d: 1 },
    ],
  ])
  const virtualLocal = { ...outdoor, id: 3, virtual: true, avgWatts: 900 }
  const overview = (details: Record<string, typeof outdoor>) =>
    buildOverview(DEFAULT_TRIATHLON_PRESENTATION, { details, health: {} }, new Set(['bike']))

  assert.equal(virtual.virtual, true)
  assert.deepEqual(overview({ 1: outdoor, 2: virtual, 3: virtualLocal }), overview({ 1: outdoor }))
  assert.deepEqual(fcBounds(overview({ 1: outdoor, 2: virtual }).traces), [
    [-79.4, 43.7],
    [-79.399, 43.701],
  ])
  assert.deepEqual(fcBounds(routeFC(virtual)), [
    [6.036319, 45.062314],
    [6.037, 45.063],
  ])
  assert.deepEqual(fcBounds(overview({ 2: { ...virtual, virtual: false } }).traces), [
    [6.036319, 45.062314],
    [6.037, 45.063],
  ])
})

test('Strava virtual run tags retain detail routes without adding outdoor overview bounds', () => {
  const virtual = mapActivity(1, 'VirtualRun', [
    [
      { lat: 45.062314, lng: 6.036319, d: 0 },
      { lat: 45.063, lng: 6.037, d: 1 },
    ],
  ])
  const overview = buildOverview(
    DEFAULT_TRIATHLON_PRESENTATION,
    { details: { 1: virtual }, health: {} },
    new Set(['run']),
  )

  assert.equal(fcBounds(overview.traces), null)
  assert.deepEqual(overview.streetActivities, [])
  assert.equal(routeFC(virtual).features.length, 1)
})

const containsZoomExpression = (value: unknown): boolean =>
  Array.isArray(value) &&
  ((value.length === 1 && value[0] === 'zoom') || value.some(containsZoomExpression))

const assertTopLevelZoomExpression = (expression: unknown[]): void => {
  assert.equal(expression[0], 'interpolate')
  assert.deepEqual(expression[2], ['zoom'])
  assert.equal(containsZoomExpression(expression.slice(3)), false)
}

test('map geometry derives line features and geographic bounds', () => {
  const features = lineFeatures([
    { lat: 43, lng: -79, d: 0 },
    { lat: 43.1, lng: -78.8, d: 10 },
  ])
  assert.equal(features.length, 1)
  assert.deepEqual(fcBounds({ type: 'FeatureCollection', features }), [
    [-79, 43],
    [-78.8, 43.1],
  ])
  assert.deepEqual(pctRange([1, 2, 3, 100]), [1, 100])
})

test('map parsers reject values outside the closed route domains', () => {
  assert.equal(readOverviewMode('hr'), 'hr')
  assert.equal(readOverviewMode('elevation'), null)
  assert.equal(readRouteSport('bike'), 'bike')
  assert.equal(readRouteSport('run'), 'run')
  assert.equal(readRouteSport('swim'), 'swim')
  assert.equal(readRouteSport('walk'), 'walk')
  assert.equal(readRouteSport('strength'), null)
})

test('map route widths keep zoom at the top level for Mapbox', () => {
  for (const expression of [
    heatWidthExpr,
    heatCasingWidthExpr,
    streetMetricWidthExpr,
    streetMetricCasingWidthExpr,
  ])
    assertTopLevelZoomExpression(expression)
})

test('map reducer rejects stale loads and resets route state', () => {
  const first = updateMap(initialMapModel(), { type: 'load' })
  const second = updateMap(first.model, { type: 'load' })
  const stale = updateMap(second.model, { type: 'loaded', request: 1 })
  assert.equal(stale.model.status, 'loading')
  assert.deepEqual(stale.effects, [])

  const ready = updateMap(stale.model, { type: 'loaded', request: 2 })
  const selected = updateMap(ready.model, { type: 'select-route', id: '42', metric: 3 })
  const ranged = updateMap(selected.model, {
    type: 'select-range',
    range: {
      kind: 'climb',
      id: 'climb',
      label: 'Climb',
      startElapsedS: 0,
      endElapsedS: 600,
      startDistanceKm: 2,
      endDistanceKm: 5,
      durationS: 600,
      distanceKm: 3,
      elevationGainM: 100,
      averageSpeedKph: 18,
      averageHeartRate: 150,
      averageWatts: 250,
      averageCadence: 85,
      averageRespirationRate: null,
      averageTemperatureC: null,
    },
  })
  const cleared = updateMap(ranged.model, { type: 'clear-route' })
  assert.equal(cleared.model.selectedRouteId, null)
  assert.equal(cleared.model.analysisRange, null)
  assert.deepEqual(cleared.effects, [{ type: 'draw-overview', options: { fit: false } }])

  const reset = updateMap(ranged.model, { type: 'reset' })
  assert.equal(reset.model.selectedRouteId, null)
  assert.equal(reset.model.analysisRange, null)
  assert.deepEqual(reset.effects, [{ type: 'draw-overview' }])
})
