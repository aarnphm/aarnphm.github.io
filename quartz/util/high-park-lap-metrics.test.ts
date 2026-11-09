import assert from 'node:assert/strict'
import test from 'node:test'
import { LAP_BINS, lapExtras, rideAltitude } from './high-park-lap-metrics'

// A 2 km lap from km 10 to 12, sampled every 50 m: a 10 m hill in the first half.
const route = Array.from({ length: 41 }, (_, i) => {
  const d = 10 + i * 0.05
  return {
    d,
    alt: 100 + 10 * Math.sin(Math.min(1, i / 20) * Math.PI),
    w: i < 20 ? 300 : 0,
    hr: 150,
    cad: i < 20 ? 90 : 0,
    speedKph: i < 20 ? 25 : 45,
    rightPowerPct: 48,
    resp: 30,
  }
})

test('a lap reads its own stretch of the route, dynamics, torque and shifts by distance', () => {
  const lap = lapExtras(
    {
      route,
      cyclingDynamics: {
        distanceKm: [9.9, 10.5, 11.5, 12.5],
        leftTorqueEffectiveness: [10, 80, 70, 10],
        rightTorqueEffectiveness: [10, 70, null, 10],
        leftPedalSmoothness: [5, 25, 21, 5],
        rightPedalSmoothness: [5, 20, 22, 5],
      },
      cyclingTorque: {
        points: [
          { distanceKm: 10.2, durationS: 10, torqueNm: 30 },
          { distanceKm: 11.2, durationS: 30, torqueNm: 10 },
          { distanceKm: 13, durationS: 10, torqueNm: 99 },
        ],
      },
      gearShifts: [
        { distanceKm: 9 },
        { distanceKm: 10.1 },
        { distanceKm: 11.9 },
        { distanceKm: 12 },
      ],
    },
    10,
    12,
  )
  assert.equal(lap.teL, 75)
  assert.equal(lap.teR, 70)
  assert.equal(lap.psL, 23)
  assert.equal(lap.nm, 15)
  assert.equal(lap.shifts, 2)
  assert.equal(lap.balL, 52)
  assert.equal(lap.gain, 10)
  assert.equal(lap.resp, 30)
  assert.equal(lap.core, null)
  assert.equal(lap.profile?.w.length, LAP_BINS)
  assert.equal(lap.profile?.w[0], 300)
  assert.equal(lap.profile?.w.at(-1), 0)
  assert.equal(lap.profile?.spd.at(-1), 45)
})

test('a lap without distance bounds has no extras, and the ride altitude starts at zero', () => {
  assert.equal(lapExtras({ route }, null, null).profile, null)
  const altitude = rideAltitude([lapExtras({ route }, 10, 12).altitude, null])
  assert.equal(altitude?.length, LAP_BINS)
  assert.equal(Math.min(...altitude!), 0)
})
