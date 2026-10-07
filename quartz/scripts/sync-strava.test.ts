import assert from 'node:assert/strict'
import test from 'node:test'
import { mapActivity, parseGear, parseRunSplits } from './sync-strava'

test('retains activity gear assignments and explicit removal for every sport', () => {
  for (const sport_type of ['Ride', 'VirtualRide', 'Run', 'Walk', 'Hike', 'Workout']) {
    assert.equal(mapActivity({ id: 1, sport_type, gear_id: ' b123 ' }).gearId, 'b123')
    assert.equal(mapActivity({ id: 1, sport_type, gear_id: null }).gearId, null)
  }
  for (const gear_id of [undefined, 123, false])
    assert.equal(Object.hasOwn(mapActivity({ id: 1, gear_id }), 'gearId'), false)
})

test('parses named bikes and shoes, zero mileage, and detailed model fallback', () => {
  assert.deepEqual(parseGear({ id: ' b123 ', name: ' Speedmax ', distance: 0 }), {
    id: 'b123',
    name: 'Speedmax',
    brandName: null,
    modelName: null,
    distanceM: 0,
  })
  assert.equal(
    parseGear({ id: 's123', name: 'Running shoes', distance: 1234 })?.name,
    'Running shoes',
  )
  assert.equal(
    parseGear({ id: 'b123', brand_name: 'Canyon', model_name: 'Speedmax' })?.name,
    'Canyon Speedmax',
  )
  assert.equal(parseGear({ id: 'b123', distance: -1 })?.distanceM, null)
  assert.equal(parseGear({ id: 'b123' })?.name, null)
  for (const value of [null, [], {}, { id: 123 }, { id: '  ' }])
    assert.equal(parseGear(value), null)
})

test('retains detailed gear descriptions and explicit clearing without inventing summary text', () => {
  assert.equal(
    parseGear({ id: 'b123', description: '  Crankset: 54/40T\nWheels: 85 mm  ' })?.description,
    'Crankset: 54/40T\nWheels: 85 mm',
  )
  for (const description of [null, '', ' \t '])
    assert.equal(parseGear({ id: 'b123', description })?.description, null)
  for (const value of [{ id: 'b123' }, { id: 'b123', description: 123 }])
    assert.equal(Object.hasOwn(parseGear(value) ?? {}, 'description'), false)
})

test('preserves explicit Strava trainer flags without inventing missing values', () => {
  assert.equal(mapActivity({ id: 1, trainer: true }).trainer, true)
  assert.equal(mapActivity({ id: 2, trainer: false }).trainer, false)
  for (const raw of [{ id: 3 }, { id: 4, trainer: null }, { id: 5, trainer: 'false' }])
    assert.equal(Object.hasOwn(mapActivity(raw), 'trainer'), false)
})

test('retains Strava maximum speed, including zero, and omits unavailable values', () => {
  assert.equal(mapActivity({ id: 1, max_speed: 12.3 }).maxSpeed, 12.3)
  assert.equal(mapActivity({ id: 1, max_speed: 0 }).maxSpeed, 0)
  assert.equal(mapActivity({ id: 1 }).maxSpeed, undefined)
})

test('trims the Strava activity summary device name', () => {
  const activity = mapActivity({ id: 1, device_name: '  Apple Watch Ultra 3  ' })

  assert.equal(activity.deviceName, 'Apple Watch Ultra 3')
})

test('omits blank or missing Strava activity summary device names', () => {
  for (const raw of [{ id: 1 }, { id: 2, device_name: ' \t ' }]) {
    assert.equal(Object.hasOwn(mapActivity(raw), 'deviceName'), false)
  }
})

test('normalizes Strava run splits and derives missing average speed', () => {
  assert.deepEqual(
    parseRunSplits([
      {
        split: 1,
        distance: 1_000,
        elapsed_time: 305,
        moving_time: 300,
        average_speed: 10 / 3,
        elevation_difference: 4.2,
        pace_zone: 2,
      },
      { split: 2, distance: 800, elapsed_time: 250, moving_time: 240, elevation_difference: -3 },
      { split: 3, distance: 0, elapsed_time: 10, moving_time: 10, average_speed: 1 },
    ]),
    [
      {
        split: 1,
        distance: 1_000,
        elapsedTime: 305,
        movingTime: 300,
        averageSpeed: 10 / 3,
        elevationDifference: 4.2,
        paceZone: 2,
      },
      {
        split: 2,
        distance: 800,
        elapsedTime: 250,
        movingTime: 240,
        averageSpeed: 10 / 3,
        elevationDifference: -3,
        paceZone: null,
      },
    ],
  )
})

test('rejects malformed Strava run split containers', () => {
  assert.deepEqual(parseRunSplits(null), [])
  assert.deepEqual(parseRunSplits({ split: 1 }), [])
})
