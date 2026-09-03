import assert from 'node:assert/strict'
import test from 'node:test'
import {
  fcBounds,
  heatCasingWidthExpr,
  heatWidthExpr,
  initialMapModel,
  lineFeatures,
  pctRange,
  readOverviewMode,
  readRouteSport,
  streetMetricCasingWidthExpr,
  streetMetricWidthExpr,
  updateMap,
} from './model'

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
