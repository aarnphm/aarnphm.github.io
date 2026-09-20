import assert from 'node:assert/strict'
import test from 'node:test'
import type { GardenEnvironmentSample } from '../../../util/activity-environment'
import type { CyclingPowerPoint } from '../../../util/cycling-power'
import {
  DEFAULT_TRIATHLON_PRESENTATION,
  type TriathlonPresentation,
} from '../../../util/triathlon-presentation'
import { cyclingPowerReadout, cyclingPowerWindAtElapsed } from './cycling-power'

const weather = (elapsedS: number, headwindKph: number | null): GardenEnvironmentSample => ({
  elapsedS,
  distanceKm: elapsedS / 100,
  headwindKph,
  crosswindKph: 0,
  apparentAirSpeedKph: 30,
  yawDeg: 0,
  uvIndex: null,
  cumulativeSed: null,
  cumulativeMovingTelemetrySed: null,
  ambientTemperatureC: null,
  cloudCoverPct: null,
})

const point: CyclingPowerPoint = {
  elapsedS: 300,
  distanceKm: 2,
  elevationM: 100,
  power30sWatts: 0,
  power5mWatts: null,
  cumulativePowerWatts: 150,
}

test('power readout retains coasting zeroes and distinguishes an unavailable averaging window', () => {
  const presentation: TriathlonPresentation = {
    ...DEFAULT_TRIATHLON_PRESENTATION,
    distance: 'metric',
  }
  const zero = cyclingPowerReadout(presentation, point, '30', [weather(300, 0)])
  assert.match(zero, /30 s average 0 W/)
  assert.match(zero, /ride average 150 W/)
  assert.match(zero, /headwind \+0\.0 km\/h/)
  assert.match(zero, /100 m/)
  const missing = cyclingPowerReadout(presentation, point, '300', [])
  assert.match(missing, /5 min average —/)
  assert.match(missing, /wind unavailable/)
})

test('wind context respects signed values and interpolates only within covered intervals', () => {
  const samples = [weather(0, -10), weather(60, 10), weather(120, null), weather(180, 5)]
  assert.equal(cyclingPowerWindAtElapsed(samples, 0, 'headwindKph'), -10)
  assert.equal(cyclingPowerWindAtElapsed(samples, 30, 'headwindKph'), 0)
  assert.equal(cyclingPowerWindAtElapsed(samples, 90, 'headwindKph'), null)
  assert.equal(cyclingPowerWindAtElapsed(samples, 150, 'headwindKph'), null)
  assert.equal(cyclingPowerWindAtElapsed(samples, -1, 'headwindKph'), null)
  assert.equal(cyclingPowerWindAtElapsed(samples, 181, 'headwindKph'), null)
})

test('imperial power readouts convert wind and elevation without changing watts', () => {
  const readout = cyclingPowerReadout(DEFAULT_TRIATHLON_PRESENTATION, point, '30', [
    weather(300, -16.09344),
  ])
  assert.match(readout, /headwind −10\.0 mph/)
  assert.match(readout, /328 ft/)
  assert.match(readout, /30 s average 0 W/)
  assert.match(readout, /ride average 150 W/)
})

test('power-only recordings omit unavailable distance instead of reporting a measured zero', () => {
  const readout = cyclingPowerReadout(
    DEFAULT_TRIATHLON_PRESENTATION,
    { ...point, distanceKm: 0 },
    '30',
    [],
    false,
  )
  assert.doesNotMatch(readout, /\bmi\b/)
  assert.match(readout, /30 s average 0 W/)
})
