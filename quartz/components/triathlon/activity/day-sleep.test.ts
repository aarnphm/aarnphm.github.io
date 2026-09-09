import assert from 'node:assert/strict'
import test from 'node:test'
import { daySleepStageLabel } from '../../../util/triathlon-card'
import {
  daySleepReadout,
  daySleepTimeGeometry,
  daySleepUnitLabel,
  decodeDaySleepTimes,
  decodeDaySleepValues,
} from './day-sleep'

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

test('respiration scrubbing follows actual sample times across a clipped first interval and gaps', () => {
  const times = [0, 64, 184, 604]
  assert.deepEqual(decodeDaySleepTimes('0,64,184,604', 4), times)
  assert.equal(decodeDaySleepTimes('0,64,,604', 4), null)
  assert.equal(decodeDaySleepTimes('0,64,64', 3), null)
  assert.equal(decodeDaySleepTimes('0,-64,184', 3), null)
  assert.equal(decodeDaySleepTimes('0,64', 3), null)
  const geometry = daySleepTimeGeometry(times, 100)
  assert.equal(geometry.x(1), (64 / 604) * 100)
  assert.equal(geometry.indexAt(184 / 604), 2)
  assert.equal(geometry.indexAt(0.95), 3)
  assert.equal(geometry.indexAt(-1), 0)
  assert.equal(geometry.indexAt(2), 3)
  const label = daySleepUnitLabel('brpm')
  assert.equal(
    daySleepReadout(90 + 56 / 60, 0, [19, 15, null, 12], 1, label, times),
    '01:32 · 15 brpm',
  )
  assert.equal(
    daySleepReadout(90 + 56 / 60, 0, [19, 15, null, 12], 2, label, times),
    '01:34 · — brpm',
  )
})
