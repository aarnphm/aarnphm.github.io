import assert from 'node:assert/strict'
import test from 'node:test'
import { frontStrength, loopAlong, loopUp, sampleWind, windTables } from './wind-field'

const periods = [6, 6.27, 7.5, 9]

test('the air is still for a third of every period', () => {
  for (const period of periods) {
    const { breeze, gust } = windTables(period)
    let still = 0
    for (let index = 0; index < breeze.length; index++) {
      if (breeze[index] === 0) still++
      assert(gust[index] <= breeze[index] + 1e-6 || breeze[index] === 0)
      if (breeze[index] === 0) assert(gust[index] < 0.01)
    }
    assert(still / breeze.length > 0.33, `period ${period}: ${still / breeze.length}`)
  }
})

test('two gusts pass, the second weaker and 1.3 seconds behind', () => {
  for (const period of periods) {
    const { gust } = windTables(period)
    assert(sampleWind(gust, 0.34) > 0.95)
    const second = sampleWind(gust, 0.34 + 1.3 / period)
    assert(second > 0.6 && second < 0.8)
    assert(sampleWind(gust, 0.34 + 0.65 / period) < second)
  }
})

test('a crown leans with the gust, swings back past rest and comes to rest before the next', () => {
  for (const period of periods)
    for (const shape of ['broadleaf', 'conifer'] as const) {
      const lean = windTables(period).lean[shape]
      let peak = 0
      let trough = 0
      let at = 0
      for (let index = 0; index < lean.length; index++) {
        if (lean[index] > peak) {
          peak = lean[index]
          at = index / lean.length
        }
        trough = Math.min(trough, lean[index])
      }
      assert(Math.abs(peak - 1) < 1e-6)
      assert(at > 0.34 && at < 0.34 + 2.4 / period, `peak at ${at}`)
      assert(trough < -0.08, `${shape} ${period}: ${trough}`)
      assert(Math.abs(lean[0]) < 0.4)
    }
  const { lean } = windTables(6.27)
  const trough = (table: Float32Array) => Math.min(...table)
  assert(trough(lean.conifer) < trough(lean.broadleaf))
})

test('a front is strongest along its own band', () => {
  let low = Infinity
  let high = -Infinity
  const bands = new Set<number>()
  for (let front = -20; front < 200; front++) {
    let best = 0
    let where = 0
    for (let across = 0; across <= 1; across += 0.05) {
      const strength = frontStrength(front, across)
      low = Math.min(low, strength)
      high = Math.max(high, strength)
      if (strength > best) {
        best = strength
        where = across
      }
    }
    bands.add(Math.round(where * 20))
  }
  assert(low >= 0.65 && high <= 1.25)
  assert(bands.size > 12)
})

test('a clump leaves rest only when the air moves', () => {
  for (let angle = 0; angle < Math.PI * 2; angle += 0.1) {
    assert.equal(loopAlong(angle, 0, 0), 0)
    assert.equal(loopUp(angle, 0, 0), 0)
    assert(loopAlong(angle, 1, 1) >= 0 && loopAlong(angle, 1, 1) <= 2.2 + 1e-9)
    assert(loopUp(angle, 1, 1) >= 0 && loopUp(angle, 1, 1) <= 1.25 + 1e-9)
  }
  assert.equal(loopAlong(0, 1, 1), 0)
  assert(loopUp(Math.PI / 2, 1, 1) > 1)
})
