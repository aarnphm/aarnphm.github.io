import assert from 'node:assert/strict'
import test from 'node:test'
import { parseRunSplits } from './sync-strava'

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
