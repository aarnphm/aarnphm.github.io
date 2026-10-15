import assert from 'node:assert/strict'
import test from 'node:test'
import { visitorPrecipitation } from './visitor-precipitation'

test('drizzle and freezing drizzle stay lighter than rain', () => {
  for (const code of [51, 53, 55, 56, 57]) {
    const state = visitorPrecipitation(code)
    assert.equal(state.fall, 'drizzle', `Weather code ${code}`)
    assert(state.heavy > 0 && state.heavy <= 1)
  }
  assert(visitorPrecipitation(51).heavy < visitorPrecipitation(55).heavy)
})

test('rain families include freezing rain, showers and thunderstorms', () => {
  for (const code of [61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 97, 99]) {
    const state = visitorPrecipitation(code)
    assert.equal(state.fall, 'rain', `Weather code ${code}`)
    assert(state.heavy > 0 && state.heavy <= 1)
    assert.equal(state.storm, code >= 95, `Weather code ${code}`)
  }
  assert(visitorPrecipitation(61).heavy < visitorPrecipitation(63).heavy)
})

test('snowfall families include grains and snow showers', () => {
  for (const code of [71, 73, 75, 77, 85, 86]) {
    const state = visitorPrecipitation(code)
    assert.equal(state.fall, 'snow', `Weather code ${code}`)
    assert(state.heavy > 0 && state.heavy <= 1)
  }
  assert(visitorPrecipitation(71).heavy < visitorPrecipitation(75).heavy)
})

test('fog is dry, and unknown or invalid readings do not imply weather', () => {
  for (const code of [45, 48])
    assert.deepEqual(visitorPrecipitation(code), {
      fall: 'none',
      heavy: 0,
      fog: true,
      storm: false,
    })
  for (const code of [0, 1, 2, 3, 50, 60, 70, 90, -1, NaN, Infinity, null, undefined])
    assert.deepEqual(visitorPrecipitation(code), {
      fall: 'none',
      heavy: 0,
      fog: false,
      storm: false,
    })
})
