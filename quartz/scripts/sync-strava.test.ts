import assert from 'node:assert/strict'
import test from 'node:test'
import { mapActivity, parseRunSplits } from './sync-strava'

test('maps the exact device name from a Strava activity summary', () => {
  const activity = mapActivity({ id: 1, device_name: 'Garmin Forerunner 970' })

  assert.equal(activity.deviceName, 'Garmin Forerunner 970')
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
