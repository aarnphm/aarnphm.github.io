import assert from 'node:assert/strict'
import test from 'node:test'
import { daySleepReadout, decodeDaySleepValues } from './day-sleep'

test('daily sleep series decoder preserves gaps and rejects malformed values', () => {
  assert.deepEqual(decodeDaySleepValues('54,,56.5,57'), [54, null, 56.5, 57])
  assert.deepEqual(decodeDaySleepValues('54,'), [54, null])
  assert.equal(decodeDaySleepValues('54,nope,57'), null)
  assert.equal(decodeDaySleepValues('54'), null)
})

test('daily sleep readout clamps the sample and formats its wall clock', () => {
  assert.equal(daySleepReadout(88, 300, [54, null, 56], 0, 'bpm'), '01:28 · 54 bpm')
  assert.equal(daySleepReadout(88, 300, [54, null, 56], 1, 'bpm'), '01:33 · — bpm')
  assert.equal(daySleepReadout(88, 300, [54, null, 56], 9, 'ms'), '01:38 · 56 ms')
})
