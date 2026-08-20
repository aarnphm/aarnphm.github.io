import assert from 'node:assert/strict'
import test from 'node:test'
import { daySleepStageLabel } from '../../../util/triathlon-card'
import { daySleepReadout, daySleepUnitLabel, decodeDaySleepValues } from './day-sleep'

test('daily sleep series decoder preserves gaps and rejects malformed values', () => {
  assert.deepEqual(decodeDaySleepValues('54,,56.5,57'), [54, null, 56.5, 57])
  assert.deepEqual(decodeDaySleepValues('54,'), [54, null])
  assert.equal(decodeDaySleepValues('54,nope,57'), null)
  assert.equal(decodeDaySleepValues('54'), null)
})

test('daily sleep readout clamps the sample and formats its wall clock', () => {
  const bpm = daySleepUnitLabel('bpm')
  assert.equal(daySleepReadout(88, 300, [54, null, 56], 0, bpm), '01:28 · 54 bpm')
  assert.equal(daySleepReadout(88, 300, [54, null, 56], 1, bpm), '01:33 · — bpm')
  assert.equal(
    daySleepReadout(88, 300, [54, null, 56], 9, daySleepUnitLabel('ms')),
    '01:38 · 56 ms',
  )
})

test('daily sleep stage readout names the lane under the cursor', () => {
  const stage = (value: number | null): string => daySleepStageLabel('en', value)
  assert.equal(daySleepReadout(88, 300, [0, 3, null], 0, stage), '01:28 · awake')
  assert.equal(daySleepReadout(88, 300, [0, 3, null], 1, stage), '01:33 · deep')
  assert.equal(daySleepReadout(88, 300, [0, 3, null], 2, stage), '01:38 · —')
})
