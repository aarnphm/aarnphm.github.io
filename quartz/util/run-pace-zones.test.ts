import assert from 'node:assert/strict'
import test from 'node:test'
import { runPaceZoneRange, runPaceZoneReference } from './run-pace-zones'

test('derives the configured six-zone reference from a 10 km race time', () => {
  const reference = runPaceZoneReference('00:50:00')
  assert.equal(reference.tenKmRaceTimeS, 3_000)
  assert.deepEqual(
    reference.paceZoneBoundsSPerKm.map(value => Math.round(value)),
    [387, 334, 300, 280, 263],
  )
  assert.deepEqual(runPaceZoneRange(reference.paceZoneBoundsSPerKm, 0), {
    fastestSPerKm: reference.paceZoneBoundsSPerKm[0],
    slowestSPerKm: null,
  })
  assert.deepEqual(runPaceZoneRange(reference.paceZoneBoundsSPerKm, 5), {
    fastestSPerKm: null,
    slowestSPerKm: reference.paceZoneBoundsSPerKm[4],
  })
})

test('scales every pace-zone boundary with the configured race-time ratio', () => {
  const fifty = runPaceZoneReference('50:00')
  const fortyFive = runPaceZoneReference('45:00')
  assert.equal(fortyFive.tenKmRaceTimeS, 2_700)
  assert.deepEqual(
    fortyFive.paceZoneBoundsSPerKm.map((value, index) =>
      Number((value / fifty.paceZoneBoundsSPerKm[index]).toFixed(3)),
    ),
    [0.9, 0.9, 0.9, 0.9, 0.9],
  )
  assert.deepEqual(runPaceZoneReference('invalid'), {
    paceZoneBoundsSPerKm: [],
    tenKmRaceTimeS: null,
  })
})
