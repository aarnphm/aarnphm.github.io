import assert from 'node:assert/strict'
import test from 'node:test'
import { visitorPrecipitation } from './visitor-precipitation'

test('current rain families include drizzle, freezing rain, showers and thunderstorms', () => {
  for (const code of [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 97, 99]) {
    const state = visitorPrecipitation(code)
    assert.equal(state.kind, 'rain', `Weather code ${code}`)
    assert(state.strength > 0 && state.strength <= 1)
  }
  assert(visitorPrecipitation(51).strength < visitorPrecipitation(65).strength)
  assert(visitorPrecipitation(61).strength < visitorPrecipitation(63).strength)
})

test('current snowfall families include grains and snow showers', () => {
  for (const code of [71, 73, 75, 77, 85, 86]) {
    const state = visitorPrecipitation(code)
    assert.equal(state.kind, 'snow', `Weather code ${code}`)
    assert(state.strength > 0 && state.strength <= 1)
  }
  assert(visitorPrecipitation(71).strength < visitorPrecipitation(75).strength)
})

test('dry, unknown and invalid readings do not imply falling precipitation', () => {
  for (const code of [0, 1, 2, 3, 45, 48, 50, 60, 70, 90, -1, NaN, Infinity, null, undefined]) {
    const state = visitorPrecipitation(code)
    assert.equal(state.kind, null)
    assert.equal(state.strength, 0)
  }
})
